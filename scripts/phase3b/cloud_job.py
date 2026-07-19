#!/usr/bin/env python3
"""Offline, approval-gated supervisor for the AtlasLens Phase 3B corpus job."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_REFUSED = 3
EXIT_CANCELLED = 130
EXIT_DEADLINE = 124
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
REDACTED_PATH = "<workspace-path>"


class CloudJobError(ValueError):
    """Stable, non-sensitive cloud-package validation error."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise CloudJobError(code)


def _object(value: object, code: str) -> dict[str, Any]:
    _require(isinstance(value, dict), code)
    return cast(dict[str, Any], value)


def _text(value: object, code: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), code)
    return cast(str, value)


def _number(value: object, code: str) -> float:
    _require(isinstance(value, int | float) and not isinstance(value, bool), code)
    result = float(cast(int | float, value))
    _require(math.isfinite(result), code)
    return result


def _integer(value: object, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return cast(int, value)


def _bool(value: object, code: str) -> bool:
    _require(isinstance(value, bool), code)
    return cast(bool, value)


def load_config(path: Path) -> dict[str, Any]:
    """Load one explicit configuration file without consulting environment files."""
    _require(path.suffix.lower() == ".json" and not path.is_symlink(), "config_path_refused")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CloudJobError("config_unreadable") from error
    return _object(raw, "config_root_not_object")


def _posix_workspace_path(value: object, code: str) -> PurePosixPath:
    path = PurePosixPath(_text(value, code))
    _require(path.is_absolute() and str(path).startswith("/workspace/"), code)
    _require(".." not in path.parts, code)
    return path


def validate_config(config: Mapping[str, Any]) -> dict[str, object]:
    """Validate immutable runtime, path, deadline, and budget invariants."""
    _require(config.get("schema_version") == "atlaslens-phase3b-cloud-job-v1", "schema")
    _require(config.get("phase") == "3B1", "phase")
    execution_enabled = _bool(config.get("execution_enabled"), "execution_enabled")
    _require(config.get("approval_required") is True, "approval_required")
    expected_approval_state = (
        "USER_APPROVED" if execution_enabled else "DO_NOT_RUN_UNTIL_USER_APPROVAL"
    )
    _require(config.get("approval_state") == expected_approval_state, "approval_state")

    container = _object(config.get("container"), "container")
    image = _text(container.get("base_image"), "container_base_image")
    _require("@sha256:" in image, "container_image_not_digest_pinned")
    digest = image.rsplit("@sha256:", 1)[1]
    _require(bool(SHA256_RE.fullmatch(digest)), "container_digest")
    _require(container.get("python_version") == "3.12", "container_python")
    _require(container.get("dependency_lock") == "services/api/uv.lock", "dependency_lock")
    _require(
        bool(SHA256_RE.fullmatch(_text(container.get("dependency_lock_sha256"), "lock_hash"))),
        "lock_hash",
    )

    paths = _object(config.get("paths"), "paths")
    parsed = {
        name: _posix_workspace_path(paths.get(name), f"path_{name}")
        for name in (
            "input_root",
            "corpus_root",
            "manifest",
            "work_root",
            "output_root",
            "checkpoint_root",
        )
    }
    input_root = parsed["input_root"]
    _require(parsed["corpus_root"].is_relative_to(input_root), "corpus_outside_input")
    _require(
        parsed["manifest"].is_relative_to(parsed["corpus_root"]),
        "manifest_outside_corpus",
    )
    mutable_roots = [parsed["work_root"], parsed["output_root"], parsed["checkpoint_root"]]
    _require(
        all(not item.is_relative_to(input_root) for item in mutable_roots),
        "mutable_under_input",
    )
    _require(len(set(mutable_roots)) == len(mutable_roots), "mutable_paths_overlap")
    _require(all(item.name == "<run-id>" for item in mutable_roots), "mutable_not_run_scoped")
    _require(_bool(paths.get("input_read_only"), "input_read_only"), "input_not_read_only")

    runtime = _object(config.get("runtime"), "runtime")
    hard_seconds = _integer(runtime.get("hard_deadline_seconds"), "hard_deadline")
    checkpoint_seconds = _integer(runtime.get("checkpoint_interval_seconds"), "checkpoint_interval")
    grace_seconds = _integer(runtime.get("termination_grace_seconds"), "termination_grace")
    _require(0 < checkpoint_seconds < hard_seconds <= 24 * 60 * 60, "runtime_bounds")
    _require(1 <= grace_seconds <= 300, "termination_grace")
    _require(runtime.get("resume_mode") == "checkpoint", "resume_mode")

    budget = _object(config.get("budget"), "budget")
    hourly = _number(budget.get("gpu_hourly_rate_usd"), "budget_hourly")
    expected_hours = _number(budget.get("expected_gpu_hours"), "budget_expected_hours")
    maximum_hours = _number(budget.get("maximum_gpu_hours"), "budget_maximum_hours")
    soft = _number(budget.get("soft_limit_usd"), "budget_soft")
    hard = _number(budget.get("hard_limit_usd"), "budget_hard")
    expected_first_month = _number(
        budget.get("expected_first_month_usd"), "budget_expected_first_month"
    )
    maximum_first_month = _number(
        budget.get("maximum_first_month_usd"), "budget_maximum_first_month"
    )
    _require(hourly > 0 and 0 < expected_hours <= maximum_hours, "budget_hours")
    _require(0 < expected_first_month <= maximum_first_month <= soft < hard, "budget_limits")
    _require(maximum_hours * 3600 <= hard_seconds, "deadline_below_maximum_hours")
    _require(budget.get("currency") == "USD", "budget_currency")

    controls = _object(config.get("controls"), "controls")
    for key in (
        "network_calls_in_validation",
        "network_calls_in_dry_run",
        "runpod_api_calls",
        "embedded_credentials",
        "public_ports",
    ):
        _require(controls.get(key) is False, f"control_{key}")
    _require(controls.get("cleanup_default") == "dry-run", "cleanup_default")

    pipeline = _object(config.get("pipeline"), "pipeline")
    _require(pipeline.get("executable") == "atlaslens-corpus", "pipeline_executable")
    config_root = PurePosixPath("/opt/atlaslens/config")
    for key in ("config", "source_policy", "manifest_schema", "leakage_policy"):
        pipeline_path = PurePosixPath(_text(pipeline.get(key), f"pipeline_{key}"))
        _require(
            pipeline_path.is_absolute()
            and ".." not in pipeline_path.parts
            and pipeline_path.is_relative_to(config_root),
            f"pipeline_{key}",
        )
    _require(
        pipeline.get("production_descriptor_provider_required") is True
        and pipeline.get("test_provider_forbidden") is True
        and pipeline.get("implicit_network_fallback") is False,
        "pipeline_provider_controls",
    )

    artifacts = _object(config.get("artifacts"), "artifacts")
    _require(artifacts.get("hash_algorithm") == "sha256", "artifact_hash")
    _require(
        artifacts.get("symlinks_allowed") is False
        and artifacts.get("raw_images_in_output") is False,
        "artifact_safety",
    )

    return {
        "status": "passed",
        "schema_version": config["schema_version"],
        "execution_enabled": execution_enabled,
        "dependency_lock_sha256": container["dependency_lock_sha256"],
        "network_calls": 0,
        "runpod_api_calls": 0,
    }


def _resolve_template(value: str, run_id: str) -> str:
    return value.replace("<run-id>", run_id)


def build_pipeline_command(
    config: Mapping[str, Any], run_id: str, *, dry_run: bool = False
) -> list[str]:
    """Build an explicit, path-complete command for the repository-native CLI."""
    _require(bool(RUN_ID_RE.fullmatch(run_id)), "run_id_invalid")
    paths = _object(config.get("paths"), "paths")
    pipeline = _object(config.get("pipeline"), "pipeline")
    source_policy = _text(pipeline.get("source_policy"), "source_policy")
    manifest_schema = _text(pipeline.get("manifest_schema"), "manifest_schema")
    pipeline_config = _text(pipeline.get("config"), "pipeline_config")
    command = [
        _text(pipeline.get("executable"), "pipeline_executable"),
        "run-pipeline",
        "--manifest",
        _resolve_template(_text(paths.get("manifest"), "manifest"), run_id),
        "--corpus-root",
        _resolve_template(_text(paths.get("corpus_root"), "corpus_root"), run_id),
        "--source-policy",
        source_policy,
        "--manifest-schema",
        manifest_schema,
        "--work-dir",
        _resolve_template(_text(paths.get("work_root"), "work_root"), run_id),
        "--output-dir",
        _resolve_template(_text(paths.get("output_root"), "output_root"), run_id),
        "--checkpoint",
        _resolve_template(_text(paths.get("checkpoint_root"), "checkpoint_root"), run_id),
        "--config",
        pipeline_config,
        "--resume",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one receipt atomically on the destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    """Write UTF-8 text atomically without exposing a partial receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_artifact_files(root: Path, excluded: set[Path]) -> list[Path]:
    if not root.exists():
        return []
    _require(root.is_dir() and not root.is_symlink(), "artifact_root_unsafe")
    files: list[Path] = []
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "artifact_symlink_refused")
        if path.is_file() and path.resolve() not in excluded:
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def write_artifact_inventory(output_root: Path, checkpoint_root: Path) -> dict[str, object]:
    """Write deterministic JSON inventory and GNU-compatible SHA-256 lines."""
    inventory_path = output_root / "artifact-inventory.json"
    checksums_path = output_root / "SHA256SUMS"
    excluded = {inventory_path.resolve(), checksums_path.resolve()}
    records: list[dict[str, object]] = []
    checksum_lines: list[str] = []
    for label, root in (("output", output_root), ("checkpoint", checkpoint_root)):
        for path in _safe_artifact_files(root, excluded):
            relative = path.relative_to(root).as_posix()
            sha256 = _sha256(path)
            records.append(
                {
                    "root": label,
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256,
                }
            )
            checksum_lines.append(f"{sha256}  {label}/{relative}")
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        checksums_path,
        "\n".join(checksum_lines) + ("\n" if checksum_lines else ""),
    )
    payload: dict[str, object] = {
        "schema_version": "atlaslens-phase3b-artifact-inventory-v1",
        "hash_algorithm": "sha256",
        "artifact_count": len(records),
        "artifacts": records,
    }
    atomic_write_json(inventory_path, payload)
    return payload


@dataclass
class SupervisorResult:
    """Sanitized result from one supervised child execution."""

    exit_code: int
    reason: str
    checkpoint_hooks: int
    elapsed_seconds: float


def _offline_environment() -> dict[str, str]:
    """Pass only runtime essentials and force common ML clients offline."""
    allowed = {
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "PYTHONPATH",
        "SystemRoot",
        "TEMP",
        "TMP",
        "WINDIR",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "UV_OFFLINE": "1",
        }
    )
    return environment


def supervise_command(
    command: Sequence[str],
    *,
    checkpoint_root: Path,
    deadline_seconds: float,
    checkpoint_interval_seconds: float,
    termination_grace_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
) -> SupervisorResult:
    """Run a child with deadline, periodic checkpoint hooks, and SIGTERM forwarding."""
    _require(bool(command), "empty_command")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    stop_reason: str | None = None
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_reason
        stop_reason = signal.Signals(signum).name

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.signal(signum, request_stop)
        except (OSError, ValueError):
            continue

    child = subprocess.Popen(  # noqa: S603, S607
        list(command), env=_offline_environment()
    )
    started = monotonic()
    next_checkpoint = started + checkpoint_interval_seconds
    hooks = 0
    reason = "completed"
    receipt_exit_code: int | None = None
    elapsed_seconds = 0.0
    try:
        while child.poll() is None:
            now = monotonic()
            if stop_reason is not None:
                reason = "cancelled"
                hooks += 1
                atomic_write_json(
                    checkpoint_root / "supervisor-checkpoint-request.json",
                    {
                        "schema_version": "atlaslens-phase3b-checkpoint-hook-v1",
                        "sequence": hooks,
                        "reason": "signal",
                    },
                )
                child.terminate()
                break
            if now - started >= deadline_seconds:
                reason = "deadline_exceeded"
                hooks += 1
                atomic_write_json(
                    checkpoint_root / "supervisor-checkpoint-request.json",
                    {
                        "schema_version": "atlaslens-phase3b-checkpoint-hook-v1",
                        "sequence": hooks,
                        "reason": "deadline",
                    },
                )
                child.terminate()
                break
            if now >= next_checkpoint:
                hooks += 1
                atomic_write_json(
                    checkpoint_root / "supervisor-checkpoint-request.json",
                    {
                        "schema_version": "atlaslens-phase3b-checkpoint-hook-v1",
                        "sequence": hooks,
                        "reason": "periodic",
                    },
                )
                next_checkpoint = now + checkpoint_interval_seconds
            time.sleep(min(0.1, checkpoint_interval_seconds))

        if child.poll() is None:
            try:
                child.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        return_code = child.returncode if child.returncode is not None else 1
        if reason == "deadline_exceeded":
            return_code = EXIT_DEADLINE
        elif reason == "cancelled":
            return_code = EXIT_CANCELLED
        elif return_code != 0:
            reason = "child_failed"
        receipt_exit_code = return_code
        elapsed_seconds = max(0.0, monotonic() - started)
        return SupervisorResult(return_code, reason, hooks, elapsed_seconds)
    finally:
        elapsed_seconds = max(elapsed_seconds, monotonic() - started)
        for registered_signal, previous in previous_handlers.items():
            signal.signal(registered_signal, previous)
        atomic_write_json(
            checkpoint_root / "supervisor-final-receipt.json",
            {
                "schema_version": "atlaslens-phase3b-supervisor-receipt-v1",
                "reason": reason,
                "checkpoint_hooks": hooks,
                "exit_code": receipt_exit_code,
                "elapsed_seconds": round(elapsed_seconds, 6),
            },
        )


def _local_path(template: object, run_id: str) -> Path:
    return Path(_resolve_template(_text(template, "runtime_path"), run_id))


def validate_execution_pipeline_settings(config: Mapping[str, Any]) -> None:
    """Require every production-only provider, index, and holdout setting."""
    pipeline = _object(config.get("pipeline"), "pipeline")
    provider = _text(
        pipeline.get("production_descriptor_provider"),
        "production_descriptor_provider_required",
    )
    _require(
        re.search(
            r"(?:^|[-_.])(test|synthetic|deterministic)(?:$|[-_.])",
            provider.lower(),
        )
        is None,
        "test_provider_forbidden",
    )
    batch_size = _integer(pipeline.get("descriptor_batch_size"), "descriptor_batch_size")
    _require(
        0 < batch_size <= 4096,
        "descriptor_batch_size",
    )
    _require(pipeline.get("index_backend") in {"exact", "faiss"}, "index_backend")
    _text(pipeline.get("index_version"), "index_version")
    shard_size = _integer(pipeline.get("shard_size"), "shard_size")
    _require(0 < shard_size <= 1_000_000, "shard_size")
    holdout_path = _posix_workspace_path(pipeline.get("holdout_path"), "holdout_path")
    _require(str(holdout_path).startswith("/workspace/input/"), "holdout_outside_input")
    holdout_hash = _text(pipeline.get("expected_holdout_hash"), "expected_holdout_hash")
    _require(bool(SHA256_RE.fullmatch(holdout_hash)), "expected_holdout_hash")
    abstention_distance = _number(pipeline.get("abstain_if_distance_gt"), "abstain_if_distance_gt")
    _require(0 < abstention_distance <= 2, "abstain_if_distance_gt")


def validate_runtime_layout(config: Mapping[str, Any], run_id: str) -> None:
    """Refuse execution unless immutable inputs and separate mutable roots are real."""
    paths = _object(config.get("paths"), "paths")
    input_root = _local_path(paths.get("input_root"), run_id)
    corpus_root = _local_path(paths.get("corpus_root"), run_id)
    manifest = _local_path(paths.get("manifest"), run_id)
    _require(input_root.is_dir() and not input_root.is_symlink(), "input_root_missing")
    _require(not os.access(input_root, os.W_OK), "input_root_writable")
    try:
        resolved_input = input_root.resolve(strict=True)
        resolved_corpus = corpus_root.resolve(strict=True)
        resolved_manifest = manifest.resolve(strict=True)
    except OSError as error:
        raise CloudJobError("input_artifact_missing") from error
    _require(
        resolved_corpus.is_dir()
        and resolved_corpus.is_relative_to(resolved_input)
        and not corpus_root.is_symlink(),
        "corpus_root_unsafe",
    )
    _require(
        resolved_manifest.is_file()
        and resolved_manifest.is_relative_to(resolved_corpus)
        and not manifest.is_symlink(),
        "manifest_unsafe",
    )
    for name in ("work_root", "output_root", "checkpoint_root"):
        target = _local_path(paths.get(name), run_id)
        _require(not target.is_symlink(), "mutable_root_symlink")
        target.mkdir(parents=True, exist_ok=True)
        _require(target.is_dir(), "mutable_root_invalid")


def write_job_receipts(
    output_root: Path,
    config: Mapping[str, Any],
    *,
    run_id: str,
    approval_receipt: str,
    result: SupervisorResult,
) -> None:
    """Record bounded runtime evidence without claiming access to provider billing."""
    serialized_config = json.dumps(
        config, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    config_hash = hashlib.sha256(serialized_config).hexdigest()
    approval_hash = hashlib.sha256(approval_receipt.encode("utf-8")).hexdigest()
    budget = _object(config.get("budget"), "budget")
    estimated_compute_cost = (
        result.elapsed_seconds / 3600 * _number(budget.get("gpu_hourly_rate_usd"), "budget_hourly")
    )
    atomic_write_json(
        output_root / "supervisor-receipt.json",
        {
            "schema_version": "atlaslens-phase3b-supervisor-output-v1",
            "run_id": run_id,
            "status": "completed" if result.exit_code == 0 else "failed",
            "exit_code": result.exit_code,
            "reason": result.reason,
            "checkpoint_hooks": result.checkpoint_hooks,
            "elapsed_seconds": round(result.elapsed_seconds, 6),
            "config_sha256": config_hash,
            "approval_receipt_sha256": approval_hash,
        },
    )
    atomic_write_json(
        output_root / "cost-receipt.json",
        {
            "schema_version": "atlaslens-phase3b-cost-receipt-v1",
            "currency": budget["currency"],
            "elapsed_compute_estimate_usd": round(estimated_compute_cost, 6),
            "soft_limit_usd": budget["soft_limit_usd"],
            "hard_limit_usd": budget["hard_limit_usd"],
            "provider_billed_cost_usd": None,
            "provider_billing_observed": False,
            "estimate_scope": "elapsed_compute_only_excludes_storage_tax_and_external_costs",
        },
    )


def run_job(
    config: Mapping[str, Any],
    *,
    run_id: str,
    execute: bool,
    approval_receipt: str | None,
) -> dict[str, object]:
    """Plan by default; execute only when both config and operator approval allow it."""
    validate_config(config)
    command = build_pipeline_command(config, run_id, dry_run=not execute)
    if not execute:
        return {
            "status": "dry-run",
            "run_id": run_id,
            "command": command,
            "network_calls": 0,
            "runpod_api_calls": 0,
        }
    _require(config.get("execution_enabled") is True, "execution_not_enabled")
    approval = _text(approval_receipt, "approval_receipt_required")
    validate_execution_pipeline_settings(config)
    validate_runtime_layout(config, run_id)
    paths = _object(config.get("paths"), "paths")
    runtime = _object(config.get("runtime"), "runtime")
    checkpoint_root = _local_path(paths.get("checkpoint_root"), run_id)
    output_root = _local_path(paths.get("output_root"), run_id)
    result = supervise_command(
        command,
        checkpoint_root=checkpoint_root,
        deadline_seconds=float(_integer(runtime.get("hard_deadline_seconds"), "deadline")),
        checkpoint_interval_seconds=float(
            _integer(runtime.get("checkpoint_interval_seconds"), "checkpoint_interval")
        ),
        termination_grace_seconds=float(
            _integer(runtime.get("termination_grace_seconds"), "termination_grace")
        ),
    )
    write_job_receipts(
        output_root,
        config,
        run_id=run_id,
        approval_receipt=approval,
        result=result,
    )
    inventory = write_artifact_inventory(output_root, checkpoint_root)
    return {
        "status": "completed" if result.exit_code == 0 else "failed",
        "run_id": run_id,
        "exit_code": result.exit_code,
        "reason": result.reason,
        "checkpoint_hooks": result.checkpoint_hooks,
        "artifact_count": inventory["artifact_count"],
    }


def cleanup_run(
    config: Mapping[str, Any],
    *,
    run_id: str,
    execute: bool,
    confirm_run_id: str | None,
) -> dict[str, object]:
    """List cleanup targets by default; delete only an exact confirmed run ID."""
    validate_config(config)
    _require(bool(RUN_ID_RE.fullmatch(run_id)), "run_id_invalid")
    paths = _object(config.get("paths"), "paths")
    targets = [
        _local_path(paths.get(name), run_id)
        for name in ("work_root", "output_root", "checkpoint_root")
    ]
    existing = [target for target in targets if target.exists()]
    for target in existing:
        _require(not target.is_symlink(), "cleanup_symlink_refused")
        _require(target.name == run_id, "cleanup_target_not_run_scoped")
    if not execute:
        return {
            "status": "dry-run",
            "run_id": run_id,
            "targets": [REDACTED_PATH for _ in existing],
            "target_count": len(existing),
        }
    _require(confirm_run_id == run_id, "cleanup_confirmation_mismatch")
    for target in existing:
        shutil.rmtree(target)
    return {"status": "deleted", "run_id": run_id, "target_count": len(existing)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate offline job controls.")
    validate.add_argument("--config", type=Path, required=True)

    run = subparsers.add_parser("run", help="Plan safely or execute an approved job.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    run.add_argument("--approval-receipt")

    inventory = subparsers.add_parser("inventory", help="Hash explicit artifact roots.")
    inventory.add_argument("--output-root", type=Path, required=True)
    inventory.add_argument("--checkpoint-root", type=Path, required=True)

    cleanup = subparsers.add_parser("cleanup", help="Preview cleanup unless explicitly confirmed.")
    cleanup.add_argument("--config", type=Path, required=True)
    cleanup.add_argument("--run-id", required=True)
    cleanup.add_argument("--execute", action="store_true")
    cleanup.add_argument("--confirm-run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inventory":
            result = write_artifact_inventory(args.output_root, args.checkpoint_root)
        else:
            config = load_config(args.config)
            if args.command == "validate-config":
                result = validate_config(config)
            elif args.command == "run":
                result = run_job(
                    config,
                    run_id=cast(str, args.run_id),
                    execute=bool(args.execute),
                    approval_receipt=cast(str | None, args.approval_receipt),
                )
            else:
                result = cleanup_run(
                    config,
                    run_id=cast(str, args.run_id),
                    execute=bool(args.execute),
                    confirm_run_id=cast(str | None, args.confirm_run_id),
                )
    except CloudJobError as error:
        print(json.dumps({"status": "refused", "code": str(error)}, sort_keys=True))
        return EXIT_REFUSED
    except (OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "code": error.__class__.__name__}, sort_keys=True))
        return EXIT_INVALID
    print(json.dumps(result, sort_keys=True))
    if isinstance(result, dict) and isinstance(result.get("exit_code"), int):
        return cast(int, result["exit_code"])
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
