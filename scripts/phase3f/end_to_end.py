"""Secret-free local controls for the one-command Phase 3F operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

REQUIRED_BRANCH = "feature/phase3f-multiregion-pilot"
FROZEN_START_HEAD = "17e1a14f2ccc96418a076d451a03fcdf981988bb"
CURRENT_SCHEMA = "atlaslens-phase3f-local-first-current-v1"
HISTORICAL_BUDGET_CAP_USD = Decimal("10")
CLOSED_POD_DISK_ALLOWANCE_USD = Decimal("0.10")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


class EndToEndError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        artifact: str | None = None,
        field: str | None = None,
        expected: str | int | bool | None = None,
        actual: str | int | bool | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.artifact = artifact
        self.field = field
        self.expected = expected
        self.actual = actual

    def public_document(self) -> dict[str, object]:
        document: dict[str, object] = {"error_code": self.code}
        for key in ("artifact", "field", "expected", "actual"):
            value = getattr(self, key)
            if value is not None:
                document[key] = value
        return document


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise EndToEndError(code)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=D:/geoSearch", *arguments],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    _require(completed.returncode == 0, "GIT_PREFLIGHT_FAILED")
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _read_json(path: Path, *, max_bytes: int) -> dict[str, object]:
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= max_bytes,
        "RUNTIME_STATE_INVALID",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EndToEndError("RUNTIME_STATE_INVALID") from exc
    _require(isinstance(value, dict), "RUNTIME_STATE_INVALID")
    return cast(dict[str, object], value)


def _sha256_path(path: Path, *, artifact: str, max_bytes: int) -> str:
    if not path.is_file() or path.is_symlink():
        raise EndToEndError(
            "TRAINING_READINESS_REPORT_MISSING",
            artifact=artifact,
        )
    if path.stat().st_size > max_bytes:
        raise EndToEndError(
            "TRAINING_READINESS_REPORT_INVALID",
            artifact=artifact,
            field="size_bytes",
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integrity_json(
    path: Path,
    *,
    artifact: str,
    missing_code: str,
    invalid_code: str,
    max_bytes: int,
) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise EndToEndError(missing_code, artifact=artifact)
    if path.stat().st_size > max_bytes:
        raise EndToEndError(invalid_code, artifact=artifact, field="size_bytes")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EndToEndError(invalid_code, artifact=artifact, field="json") from exc
    if not isinstance(value, dict):
        raise EndToEndError(invalid_code, artifact=artifact, field="document")
    return cast(dict[str, object], value)


def _sealed_error(exc: BaseException) -> EndToEndError:
    raw_code = getattr(exc, "code", None)
    code = (
        raw_code
        if isinstance(raw_code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", raw_code)
        else "SEALED_BUNDLE_INVALID"
    )
    artifact_by_code = {
        "SEALED_BUNDLE_MISSING": "sealed-acquisition",
        "SEALED_INVENTORY_INVALID": "checksum-inventory.json",
        "SEALED_INVENTORY_MISMATCH": "checksum-inventory.json",
        "SEALED_MANIFEST_INVALID": "manifest.json",
        "SEALED_MANIFEST_MISMATCH": "manifest.json",
        "SEALED_SELECTION_INVALID": "selection-lock.json",
        "SEALED_ASSETS_INVALID": "sealed-assets.json",
        "SEALED_MEDIA_MISMATCH": "assets",
        "SEALED_RECEIPT_INVALID": "acquisition-receipt.json",
        "SEALED_PROVENANCE_INVALID": "provenance-aggregate.json",
    }
    return EndToEndError(
        code,
        artifact=artifact_by_code.get(code, "sealed-acquisition"),
        field=(
            "sha256"
            if code in {"SEALED_INVENTORY_MISMATCH", "SEALED_MEDIA_MISMATCH"}
            else None
        ),
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _runtime_snapshot(runtime_root: Path) -> dict[str, object]:
    current_path = runtime_root / "current.json"
    if not current_path.exists():
        return {
            "current_run_present": False,
            "run_id": None,
            "metadata_row_count": 0,
            "metadata_sha256": None,
            "checkpoint_present": False,
        }
    current = _read_json(current_path, max_bytes=4096)
    run_id = current.get("run_id")
    _require(
        current.get("schema") == CURRENT_SCHEMA
        and isinstance(run_id, str)
        and bool(_RUN_ID.fullmatch(run_id)),
        "CURRENT_RUN_INVALID",
    )
    safe_run_id = cast(str, run_id)
    checkpoint = runtime_root / safe_run_id / "acquisition-work" / "metadata-pages.json"
    sealed = runtime_root / safe_run_id / "sealed-acquisition"
    if checkpoint.exists():
        document = _read_json(checkpoint, max_bytes=64 * 1024 * 1024)
        raw_rows = document.get("rows_by_city")
        _require(isinstance(raw_rows, dict), "METADATA_CHECKPOINT_INVALID")
        rows = cast(dict[str, object], raw_rows)
        _require(all(isinstance(value, list) for value in rows.values()), "METADATA_CHECKPOINT_INVALID")
        payload = checkpoint.read_bytes()
        return {
            "current_run_present": True,
            "run_id": safe_run_id,
            "metadata_row_count": sum(len(cast(list[object], value)) for value in rows.values()),
            "metadata_sha256": hashlib.sha256(payload).hexdigest(),
            "checkpoint_present": True,
            "sealed": False,
        }
    _require(sealed.is_dir() and not sealed.is_symlink(), "CURRENT_RUN_CHECKPOINT_MISSING")
    manifest = _read_json(sealed / "manifest.json", max_bytes=1024 * 1024)
    return {
        "current_run_present": True,
        "run_id": safe_run_id,
        "metadata_row_count": None,
        "metadata_sha256": None,
        "checkpoint_present": False,
        "sealed": manifest.get("sealed") is True,
        "asset_count": manifest.get("asset_count"),
    }


def preflight(repository: Path, runtime_root: Path) -> dict[str, object]:
    repository = repository.resolve()
    _require(repository.is_dir() and not repository.is_symlink(), "REPOSITORY_INVALID")
    branch = _git(repository, "branch", "--show-current")
    head = _git(repository, "rev-parse", "HEAD")
    _require(branch == REQUIRED_BRANCH, "GIT_BRANCH_MISMATCH")
    _git(repository, "merge-base", "--is-ancestor", FROZEN_START_HEAD, head)
    status = tuple(
        row
        for row in _git(repository, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if row
    )
    _require(status in {(), ("?? asda.html",)}, "GIT_WORKTREE_NOT_CLEAN")
    c_free = shutil.disk_usage("C:\\").free
    d_free = shutil.disk_usage("D:\\").free
    _require(c_free >= 35 * 1024**3, "C_DISK_GATE_FAILED")
    _require(d_free >= 8 * 1024**3, "D_DISK_GATE_FAILED")
    snapshot = _runtime_snapshot(runtime_root.resolve()) if runtime_root.exists() else {
        "current_run_present": False,
        "run_id": None,
        "metadata_row_count": 0,
        "metadata_sha256": None,
        "checkpoint_present": False,
    }
    return {
        "action": "preflight",
        "branch": branch,
        "head": head,
        "frozen_start_is_ancestor": True,
        "worktree_clean": True,
        "ignored_untracked": ["asda.html"] if status else [],
        "c_free_bytes": c_free,
        "d_free_bytes": d_free,
        "runtime": snapshot,
        "runpod_api_calls": 0,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def readiness(repository: Path, runtime_root: Path) -> dict[str, object]:
    source = repository.resolve() / "services" / "api" / "src"
    _require(source.is_dir() and not source.is_symlink(), "PHASE3F_SOURCE_ROOT_INVALID")
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.training import (  # noqa: PLC0415
        require_training_ready,
    )

    snapshot = _runtime_snapshot(runtime_root.resolve())
    run_id = snapshot["run_id"]
    _require(isinstance(run_id, str), "CURRENT_RUN_INVALID")
    safe_run_id = cast(str, run_id)
    run_root = runtime_root.resolve() / safe_run_id
    report_path = run_root / "training-readiness.json"
    sealed = require_training_ready(
        run_root / "sealed-acquisition",
        report_path=report_path,
    )
    return {
        "action": "readiness",
        "run_id": safe_run_id,
        "asset_count": len(sealed.split.assets),
        "ready_for_training": True,
        "report_path_name": report_path.name,
        "runpod_api_calls": 0,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _cloud_training_phase(cloud_runtime: Path, run_id: str) -> str:
    if not cloud_runtime.exists():
        return "cloud_inventory"
    if not cloud_runtime.is_dir() or cloud_runtime.is_symlink():
        raise EndToEndError(
            "CLOUD_RUNTIME_STATE_INVALID",
            artifact="cloud-runtime",
        )
    operator_path = cloud_runtime / "_operator" / "phase3f-current.json"
    if operator_path.exists():
        operator = _integrity_json(
            operator_path,
            artifact="phase3f-current.json",
            missing_code="CLOUD_OPERATOR_STATE_MISSING",
            invalid_code="CLOUD_OPERATOR_STATE_INVALID",
            max_bytes=1024 * 1024,
        )
        operator_run_id = operator.get("run_id")
        operator_stage = operator.get("stage")
        if (
            not isinstance(operator_run_id, str)
            or not _RUN_ID.fullmatch(operator_run_id)
            or operator_stage not in {"preflight", "running", "failed", "terminated"}
        ):
            raise EndToEndError(
                "CLOUD_OPERATOR_STATE_INVALID",
                artifact="phase3f-current.json",
                field="run_id_or_stage",
            )
        if operator_stage in {"preflight", "running"}:
            if operator_run_id != run_id:
                raise EndToEndError(
                    "ACTIVE_CLOUD_RUN_CONFLICT",
                    artifact="phase3f-current.json",
                    field="run_id",
                )
            supervisor_pid = operator.get("supervisor_pid")
            if (
                isinstance(supervisor_pid, bool)
                or not isinstance(supervisor_pid, int)
                or supervisor_pid <= 0
            ):
                raise EndToEndError(
                    "CLOUD_OPERATOR_STATE_INVALID",
                    artifact="phase3f-current.json",
                    field="supervisor_pid",
                )
            if not _process_is_running(supervisor_pid):
                raise EndToEndError(
                    "STALE_OPERATOR_RECEIPT_REQUIRES_TERMINATE",
                    artifact="phase3f-current.json",
                    field="supervisor_pid",
                )
            return "training_running"

    receipts_root = cloud_runtime / "_receipts"
    if not receipts_root.exists():
        return "cloud_inventory"
    if not receipts_root.is_dir() or receipts_root.is_symlink():
        raise EndToEndError(
            "CLOUD_TRAINING_RECEIPT_INVALID",
            artifact="_receipts",
        )
    try:
        receipt_paths = tuple(sorted(receipts_root.glob("*.json")))
    except OSError as exc:
        raise EndToEndError(
            "CLOUD_TRAINING_RECEIPT_INVALID",
            artifact="_receipts",
        ) from exc
    for receipt_path in receipt_paths:
        receipt = _integrity_json(
            receipt_path,
            artifact="training-receipt.json",
            missing_code="CLOUD_TRAINING_RECEIPT_INVALID",
            invalid_code="CLOUD_TRAINING_RECEIPT_INVALID",
            max_bytes=1024 * 1024,
        )
        if receipt.get("run_id") != run_id:
            continue
        if (
            receipt.get("schema") != "atlaslens-phase3f-local-supervisor-receipt-v1"
            or not isinstance(receipt.get("outcome"), str)
            or receipt.get("pod_termination_verified") is not True
            or receipt.get("secret_values_included") is not False
        ):
            raise EndToEndError(
                "CLOUD_TRAINING_RECEIPT_INVALID",
                artifact="training-receipt.json",
                field="completion_evidence",
            )
        return "training_completed"
    return "cloud_inventory"


def resume_plan(
    repository: Path,
    runtime_root: Path,
    cloud_runtime: Path,
) -> dict[str, object]:
    source = repository.resolve() / "services" / "api" / "src"
    _require(source.is_dir() and not source.is_symlink(), "PHASE3F_SOURCE_ROOT_INVALID")
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.local_first import (  # noqa: PLC0415
        verify_sealed_acquisition,
    )
    from atlaslens_api.phase3f.training import (  # noqa: PLC0415
        training_readiness_document,
    )

    resolved_runtime = runtime_root.resolve()
    current = _integrity_json(
        resolved_runtime / "current.json",
        artifact="current.json",
        missing_code="PHASE3F_RESUME_STATE_MISSING",
        invalid_code="CURRENT_RUN_INVALID",
        max_bytes=4096,
    )
    run_id = current.get("run_id")
    if (
        current.get("schema") != CURRENT_SCHEMA
        or not isinstance(run_id, str)
        or not _RUN_ID.fullmatch(run_id)
    ):
        raise EndToEndError(
            "CURRENT_RUN_INVALID",
            artifact="current.json",
            field="schema_or_run_id",
        )
    run_root = resolved_runtime / run_id
    state = _integrity_json(
        run_root / "state.json",
        artifact="state.json",
        missing_code="LOCAL_STATE_MISSING",
        invalid_code="LOCAL_STATE_INVALID",
        max_bytes=1024 * 1024,
    )
    stage = state.get("stage")
    if (
        state.get("schema") != "atlaslens-phase3f-local-first-state-v1"
        or state.get("run_id") != run_id
        or not isinstance(stage, str)
    ):
        raise EndToEndError(
            "LOCAL_STATE_INVALID",
            artifact="state.json",
            field="schema_run_id_or_stage",
        )

    sealed_root = run_root / "sealed-acquisition"
    if not sealed_root.exists():
        work_root = run_root / "acquisition-work"
        metadata_checkpoint = work_root / "metadata-pages.json"
        if stage == "ACQUISITION_SEALED" or not work_root.is_dir():
            raise EndToEndError(
                "SEALED_BUNDLE_MISSING",
                artifact="sealed-acquisition",
                field="directory",
                expected=True,
                actual=False,
            )
        if work_root.is_symlink() or not metadata_checkpoint.is_file():
            raise EndToEndError(
                "ACQUISITION_WORK_MISSING",
                artifact="acquisition-work",
                field="metadata-pages.json",
            )
        _runtime_snapshot(resolved_runtime)
        return {
            "schema": "atlaslens-phase3f-resume-plan-v1",
            "action": "resume-plan",
            "run_id": run_id,
            "state_stage": stage,
            "next_phase": "local_acquisition",
            "acquisition_required": True,
            "mapillary_required": True,
            "dataset_write_count": 0,
            "cloud_readiness_transition_count": 0,
            "runpod_api_calls": 0,
            "cloud_mutations": 0,
            "secrets_included": False,
        }
    if not sealed_root.is_dir() or sealed_root.is_symlink():
        raise EndToEndError(
            "SEALED_BUNDLE_INVALID",
            artifact="sealed-acquisition",
            field="directory",
        )

    try:
        sealed = verify_sealed_acquisition(sealed_root)
    except Exception as exc:
        raise _sealed_error(exc) from exc
    if sealed.run_id != run_id:
        raise EndToEndError(
            "SEALED_RUN_ID_MISMATCH",
            artifact="manifest.json",
            field="run_id",
        )
    asset_count = len(sealed.split.assets)
    if asset_count != 830:
        raise EndToEndError(
            "PHASE3F_DATASET_ASSET_COUNT_MISMATCH",
            artifact="sealed-assets.json",
            field="asset_count",
            expected=830,
            actual=asset_count,
        )
    if state.get("asset_count") != asset_count:
        raise EndToEndError(
            "LOCAL_STATE_ASSET_COUNT_MISMATCH",
            artifact="state.json",
            field="asset_count",
            expected=asset_count,
            actual=cast(int | None, state.get("asset_count")),
        )
    if state.get("model_loaded") is not False:
        raise EndToEndError(
            "LOCAL_STATE_MODEL_LOADED_INVALID",
            artifact="state.json",
            field="model_loaded",
            expected=False,
            actual=cast(bool | None, state.get("model_loaded")),
        )

    report_path = run_root / "training-readiness.json"
    report = _integrity_json(
        report_path,
        artifact="training-readiness.json",
        missing_code="TRAINING_READINESS_REPORT_MISSING",
        invalid_code="TRAINING_READINESS_REPORT_INVALID",
        max_bytes=1024 * 1024,
    )
    actual_report_sha256 = _sha256_path(
        report_path,
        artifact="training-readiness.json",
        max_bytes=1024 * 1024,
    )
    try:
        expected_report = training_readiness_document(sealed)
    except Exception as exc:
        raw_code = getattr(exc, "code", None)
        code = raw_code if isinstance(raw_code, str) else "TRAINING_READINESS_RECOMPUTE_FAILED"
        raise EndToEndError(
            code,
            artifact="sealed-acquisition",
            field="training_readiness",
        ) from exc
    expected_report_bytes = _canonical_bytes(expected_report)
    expected_report_sha256 = hashlib.sha256(expected_report_bytes).hexdigest()
    if report_path.read_bytes() != expected_report_bytes:
        raise EndToEndError(
            "TRAINING_READINESS_HASH_MISMATCH",
            artifact="training-readiness.json",
            field="sha256",
            expected=expected_report_sha256,
            actual=actual_report_sha256,
        )
    if (
        report.get("ready") is not True
        or report.get("outcome") != "READY_FOR_TRAINING"
        or report.get("asset_count") != 830
        or report.get("gpu_started") is not False
        or report.get("cloud_mutations") != 0
        or report.get("secrets_included") is not False
        or not isinstance(report.get("checks"), dict)
        or not all(cast(dict[str, object], report["checks"]).values())
    ):
        raise EndToEndError(
            "TRAINING_READINESS_NOT_READY",
            artifact="training-readiness.json",
            field="ready_or_safety_fields",
        )

    next_phase = _cloud_training_phase(cloud_runtime.resolve(), run_id)
    sealed_assets_path = sealed_root / "sealed-assets.json"
    state_readiness_sha256 = state.get("readiness_report_sha256")
    return {
        "schema": "atlaslens-phase3f-resume-plan-v1",
        "action": "resume-plan",
        "run_id": run_id,
        "state_stage": stage,
        "next_phase": next_phase,
        "acquisition_required": False,
        "mapillary_required": False,
        "dataset_write_count": 0,
        "cloud_readiness_transition_count": 1 if next_phase == "cloud_inventory" else 0,
        "asset_count": asset_count,
        "readiness_sha256": actual_report_sha256,
        "sealed_assets_sha256": hashlib.sha256(sealed_assets_path.read_bytes()).hexdigest(),
        "readiness_recomputed_exact": True,
        "state_readiness_hash_matches": state_readiness_sha256 == actual_report_sha256,
        "model_loaded": False,
        "gpu_started": False,
        "runpod_api_calls": 0,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def _budget_decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise EndToEndError("CLOUD_PLAN_BUDGET_RECEIPT_INVALID")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise EndToEndError("CLOUD_PLAN_BUDGET_RECEIPT_INVALID") from exc
    _require(parsed.is_finite() and parsed >= 0, "CLOUD_PLAN_BUDGET_RECEIPT_INVALID")
    return parsed


def _receipt_source_snapshot(cloud_runtime: Path) -> tuple[int, str, tuple[str, ...]]:
    paths: list[Path] = []
    supervisor_root = cloud_runtime / "_receipts"
    if supervisor_root.exists():
        _require(
            supervisor_root.is_dir() and not supervisor_root.is_symlink(),
            "CLOUD_PLAN_RECEIPT_STATE_INVALID",
        )
        paths.extend(sorted(supervisor_root.glob("*.json")))
    operator_root = cloud_runtime / "_operator"
    current = operator_root / "phase3f-current.json"
    if current.exists():
        paths.append(current)
    archive = operator_root / "archive"
    if archive.exists():
        _require(
            archive.is_dir() and not archive.is_symlink(),
            "CLOUD_PLAN_RECEIPT_STATE_INVALID",
        )
        paths.extend(sorted(archive.glob("*.json")))
    _require(len(paths) <= 10_000, "CLOUD_PLAN_RECEIPT_STATE_INVALID")
    hashes: list[str] = []
    for path in paths:
        _require(
            path.is_file()
            and not path.is_symlink()
            and 0 < path.stat().st_size <= 1024 * 1024,
            "CLOUD_PLAN_RECEIPT_STATE_INVALID",
        )
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    ordered = tuple(sorted(hashes))
    digest = hashlib.sha256(
        json.dumps(ordered, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return len(ordered), digest, ordered


def _source_snapshot_matches(
    hashes: tuple[str, ...],
    *,
    expected_count: object,
    expected_sha256: object,
) -> bool:
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or expected_count < 0
    ):
        return False
    if len(hashes) == expected_count:
        return hashlib.sha256(
            json.dumps(hashes, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest() == expected_sha256
    if len(hashes) != expected_count + 1 or len(hashes) > 256:
        return False
    for index in range(len(hashes)):
        prior = hashes[:index] + hashes[index + 1 :]
        digest = hashlib.sha256(
            json.dumps(prior, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        if digest == expected_sha256:
            return True
    return False


def _latest_budget_snapshot(cloud_runtime: Path, run_id: str) -> dict[str, object]:
    root = cloud_runtime / "_budget" / "reconciliations"
    _require(root.is_dir() and not root.is_symlink(), "CLOUD_PLAN_BUDGET_RECEIPT_MISSING")
    candidates: list[tuple[datetime, Path, dict[str, object]]] = []
    for path in sorted(root.glob("*.json")):
        row = _integrity_json(
            path,
            artifact="budget-reconciliation.json",
            missing_code="CLOUD_PLAN_BUDGET_RECEIPT_MISSING",
            invalid_code="CLOUD_PLAN_BUDGET_RECEIPT_INVALID",
            max_bytes=1024 * 1024,
        )
        timestamp_value = row.get("reconciled_at")
        _require(isinstance(timestamp_value, str), "CLOUD_PLAN_BUDGET_RECEIPT_INVALID")
        timestamp = cast(str, timestamp_value)
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise EndToEndError("CLOUD_PLAN_BUDGET_RECEIPT_INVALID") from exc
        _require(
            parsed.tzinfo is not None and parsed.utcoffset() is not None,
            "CLOUD_PLAN_BUDGET_RECEIPT_INVALID",
        )
        candidates.append((parsed.astimezone(UTC), path, row))
    _require(bool(candidates), "CLOUD_PLAN_BUDGET_RECEIPT_MISSING")
    _timestamp, path, row = max(candidates, key=lambda item: (item[0], item[1].name))
    receipt_id = row.get("receipt_id")
    receipt_sha256 = row.get("receipt_sha256")
    _require(
        row.get("schema") == "atlaslens-phase3f-budget-reconciliation-v1"
        and isinstance(receipt_id, str)
        and bool(_RUN_ID.fullmatch(receipt_id))
        and path.stem == receipt_id
        and isinstance(receipt_sha256, str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", receipt_sha256)),
        "CLOUD_PLAN_BUDGET_RECEIPT_INVALID",
    )
    receipt_base = {
        key: value
        for key, value in row.items()
        if key not in {"receipt_id", "receipt_sha256"}
    }
    computed = hashlib.sha256(
        json.dumps(receipt_base, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    expected_run_hash = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    source_count, _source_sha256, source_hashes = _receipt_source_snapshot(cloud_runtime)
    snapshot_source_count = row.get("source_receipt_count")
    snapshot_source_sha256 = row.get("source_receipts_sha256")
    inventory = row.get("inventory")
    _require(
        computed == receipt_sha256
        and receipt_id == computed[:32]
        and row.get("run_id_sha256") == expected_run_hash
        and inventory
        == {"pods": 0, "endpoints": 0, "network_volumes": 0, "templates": 0}
        and row.get("cloud_mutations") == 0
        and row.get("secret_values_included") is False
        and row.get("provider_response_body_included") is False
        and row.get("unrecognized_active_billing") is False
        and _source_snapshot_matches(
            source_hashes,
            expected_count=snapshot_source_count,
            expected_sha256=snapshot_source_sha256,
        ),
        "CLOUD_PLAN_BUDGET_RECEIPT_INVALID",
    )
    actual = _budget_decimal(row.get("actual_billed_usd"))
    active = _budget_decimal(row.get("active_exposure_usd"))
    unbilled = _budget_decimal(row.get("conservative_unbilled_estimate_usd"))
    proposed = _budget_decimal(row.get("proposed_run_max_usd"))
    remaining = _budget_decimal(row.get("remaining_authorized_usd"))
    from atlaslens_api.phase3f.operator import (  # noqa: PLC0415
        operator_budget_linkage,
        read_operator_receipt,
    )

    pending_by_key: dict[str, Decimal] = {}
    pending_compute_by_key: dict[str, Decimal] = {}
    for receipt_path in (
        *sorted((cloud_runtime / "_operator" / "archive").glob("*.json")),
        cloud_runtime / "_operator" / "phase3f-current.json",
    ):
        if not receipt_path.exists():
            continue
        receipt = read_operator_receipt(receipt_path)
        if not (
            receipt.cleanup_verified
            and receipt.stage in {"failed", "terminated"}
            and receipt.pod_id is not None
            and receipt.finished_at is not None
        ):
            continue
        try:
            finished = datetime.fromisoformat(receipt.finished_at).astimezone(UTC)
            started = datetime.fromisoformat(
                receipt.pod_bound_at or receipt.started_at
            ).astimezone(UTC)
        except ValueError as exc:
            raise EndToEndError("CLOUD_PLAN_BILLING_LAG_RECEIPT_INVALID") from exc
        if finished <= _timestamp:
            continue
        duration = Decimal(str((finished - started).total_seconds()))
        _require(
            Decimal("0") <= duration <= Decimal(receipt.max_wall_minutes * 60),
            "CLOUD_PLAN_BILLING_LAG_RECEIPT_INVALID",
        )
        observed_prices = tuple(
            price
            for price in (
                receipt.selected_uninterruptable_price,
                receipt.create_cost_per_hr,
            )
            if price is not None
        )
        hourly = max(observed_prices) if observed_prices else receipt.max_gpu_hourly_usd
        compute = hourly * duration / Decimal(3600)
        estimate = min(
            receipt.max_spend_usd,
            compute + CLOSED_POD_DISK_ALLOWANCE_USD,
        )
        key = operator_budget_linkage(receipt)
        pending_by_key[key] = max(pending_by_key.get(key, Decimal("0")), estimate)
        pending_compute_by_key[key] = max(
            pending_compute_by_key.get(key, Decimal("0")), compute
        )
    lag_estimate = sum(pending_by_key.values(), Decimal("0"))
    lag_compute = sum(pending_compute_by_key.values(), Decimal("0"))
    corrected_unbilled = unbilled + lag_estimate
    corrected_historical = actual + corrected_unbilled + active
    corrected_remaining = max(
        Decimal("0"), HISTORICAL_BUDGET_CAP_USD - corrected_historical
    )
    projected_total = corrected_historical + proposed
    _require(
        active == 0
        and proposed <= corrected_remaining
        and projected_total <= HISTORICAL_BUDGET_CAP_USD,
        "CLOUD_PLAN_BUDGET_BLOCKED",
    )
    return {
        "receipt_id": receipt_id,
        "actual_billed_usd": str(actual),
        "active_exposure_usd": str(active),
        "conservative_unbilled_estimate_usd": str(corrected_unbilled),
        "snapshot_conservative_unbilled_estimate_usd": str(unbilled),
        "billing_lag_attempt_count": len(pending_by_key),
        "billing_lag_compute_estimate_usd": str(lag_compute),
        "billing_lag_disk_allowance_usd": str(
            CLOSED_POD_DISK_ALLOWANCE_USD * len(pending_by_key)
        ),
        "billing_lag_local_estimate_usd": str(lag_estimate),
        "proposed_run_max_usd": str(proposed),
        "snapshot_remaining_authorized_usd": str(remaining),
        "remaining_authorized_usd": str(corrected_remaining),
        "projected_total_usd": str(projected_total),
        "source_receipt_count": snapshot_source_count,
        "current_source_receipt_count": source_count,
        "post_snapshot_receipt_count": source_count - cast(int, snapshot_source_count),
        "duplicate_billing_record_count": row.get("duplicate_billing_record_count"),
    }


def cloud_plan(
    repository: Path,
    runtime_root: Path,
    cloud_runtime: Path,
) -> dict[str, object]:
    local = resume_plan(repository, runtime_root, cloud_runtime)
    _require(
        local.get("acquisition_required") is False
        and local.get("mapillary_required") is False,
        "CLOUD_PLAN_DATASET_NOT_READY",
    )
    run_id = cast(str, local["run_id"])
    source = repository.resolve() / "services" / "api" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.operator import (  # noqa: PLC0415
        OperatorReceipt,
        Phase3FOperatorError,
        operator_archive_index_matches,
        prepare_operator_attempt,
        read_operator_receipt,
        reconcile_operator_archive_index,
    )

    real_operator_root = cloud_runtime / "_operator"
    current_path = real_operator_root / "phase3f-current.json"
    active_local_receipts = 0
    unclean_local_receipts = 0
    if current_path.exists():
        current = read_operator_receipt(current_path)
        active_local_receipts = int(current.stage in {"preflight", "running"})
        unclean_local_receipts = int(
            not current.cleanup_verified
            and current.stage in {"failed", "terminated"}
        )
    initial_index = reconcile_operator_archive_index(real_operator_root, write=False)
    stored_index_matches = operator_archive_index_matches(
        real_operator_root,
        initial_index,
    )
    active_local_receipts += cast(int, initial_index["active_receipt_count"])
    unclean_local_receipts += cast(int, initial_index["unclean_receipt_count"])
    local_blockers: list[str] = (
        ["TRAINING_ALREADY_COMPLETED"]
        if local.get("next_phase") == "training_completed"
        else []
    )
    if not stored_index_matches:
        local_blockers.append("OPERATOR_ARCHIVE_INDEX_MISMATCH")
    if cast(int, initial_index["partial_file_count"]) > 0:
        local_blockers.append("OPERATOR_ARCHIVE_PARTIAL_REQUIRES_RECONCILIATION")
    simulated_index = initial_index
    attempt_id_ready = False
    lock_released = False
    with tempfile.TemporaryDirectory(prefix="atlaslens-phase3f-cloud-plan-") as temporary:
        simulated_root = Path(temporary) / "_operator"
        if real_operator_root.exists():
            _require(
                real_operator_root.is_dir() and not real_operator_root.is_symlink(),
                "CLOUD_PLAN_RECEIPT_STATE_INVALID",
            )
            shutil.copytree(real_operator_root, simulated_root, symlinks=True)
        else:
            simulated_root.mkdir(parents=True)
        simulated_current = simulated_root / "phase3f-current.json"
        seed = hashlib.sha256(
            (
                "atlaslens-phase3f-cloud-plan-attempt-v1:"
                + run_id
                + ":"
                + cast(str, initial_index["chain_head_sha256"])
                + ":"
                + cast(str, local["readiness_sha256"])
            ).encode("ascii")
        ).hexdigest()
        simulated_receipt = OperatorReceipt(
            run_id=run_id,
            attempt_id=seed[:32],
            run_marker=f"atlaslens-phase3f-{run_id}",
            pod_id=None,
            supervisor_pid=os.getpid(),
            stage="preflight",
            started_at="1970-01-01T00:00:00+00:00",
            finished_at=None,
            max_spend_usd=Decimal("3"),
            soft_stop_usd=Decimal("2.95"),
            hard_stop_usd=Decimal("2.99"),
            max_gpu_hourly_usd=Decimal("0.50"),
            max_wall_minutes=345,
            cleanup_verified=False,
        )
        try:
            prepare_operator_attempt(
                simulated_current,
                simulated_receipt,
            )
            confirmed = read_operator_receipt(simulated_current)
            attempt_id_ready = confirmed.attempt_id == simulated_receipt.attempt_id
            simulated_index = reconcile_operator_archive_index(
                simulated_root,
                write=False,
            )
            lock_released = not (simulated_root / ".archive.lock").exists()
        except Phase3FOperatorError as exc:
            local_blockers.append(exc.code)

    local_blockers.extend(
        code
        for condition, code in (
            (active_local_receipts > 0, "ACTIVE_LOCAL_OPERATOR_RECEIPT"),
            (unclean_local_receipts > 0, "UNCLEAN_LOCAL_OPERATOR_RECEIPT"),
            (not attempt_id_ready, "ATTEMPT_ID_NOT_READY"),
            (not lock_released, "OPERATOR_LOCK_NOT_RELEASED"),
        )
        if condition and code not in local_blockers
    )
    archive_conflicts = cast(int, simulated_index["archive_conflicts"])
    if archive_conflicts and "OPERATOR_ARCHIVE_CONFLICT" not in local_blockers:
        local_blockers.append("OPERATOR_ARCHIVE_CONFLICT")
    budget = None if local_blockers else _latest_budget_snapshot(cloud_runtime, run_id)
    ready = not local_blockers
    return {
        "schema": "atlaslens-phase3f-cloud-plan-v1",
        "action": "cloud-plan",
        "run_id": run_id,
        "ready_for_live_inventory": ready,
        "ready_for_create_after_live_gates": ready,
        "live_inventory_required": True,
        "local_blockers": local_blockers,
        "archive_conflicts": archive_conflicts,
        "active_local_receipts": active_local_receipts,
        "unclean_local_receipts": unclean_local_receipts,
        "dataset_asset_count": local["asset_count"],
        "readiness_sha256": local["readiness_sha256"],
        "sealed_assets_sha256": local["sealed_assets_sha256"],
        "archive_receipt_count": simulated_index["archive_receipt_count"],
        "legacy_attempt_count": simulated_index["legacy_attempt_count"],
        "duplicate_group_count": simulated_index["duplicate_group_count"],
        "duplicate_receipt_count": simulated_index["duplicate_receipt_count"],
        "duplicate_budget_linkage_count": simulated_index[
            "duplicate_budget_linkage_count"
        ],
        "recoverable_partial_file_count": simulated_index["partial_file_count"],
        "filename_parse_failure_count": 0,
        "filename_content_identity_mismatch_count": 0,
        "index_hash_chain_mismatch_count": int(not stored_index_matches),
        "attempt_id_ready": attempt_id_ready,
        "operator_lock_released": lock_released,
        "budget_snapshot": budget,
        "dataset_write_count": 0,
        "secret_prompt_count": 0,
        "runpod_api_calls": 0,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def remote_environment_plan(repository: Path) -> dict[str, object]:
    """Validate the paid-runtime dependency contract without network or mutation."""
    resolved = repository.resolve()
    source = resolved / "services" / "api" / "src"
    _require(source.is_dir() and not source.is_symlink(), "PHASE3F_SOURCE_ROOT_INVALID")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.remote_environment import (  # noqa: PLC0415
        REMOTE_ENVIRONMENT_PLAN_SCHEMA,
        RemoteEnvironmentError,
        evaluate_remote_environment,
        load_remote_environment_contract,
        sha256_file,
        verify_framework_wheelhouse,
    )

    contract_path = resolved / "config" / "phase3f-remote-environment.json"
    lock_path = resolved / "config" / "phase3f-training-requirements.lock"
    contract_sha256 = sha256_file(contract_path)
    lock_sha256 = sha256_file(lock_path)
    contract = load_remote_environment_contract(
        contract_path,
        lock_path,
        expected_contract_sha256=contract_sha256,
        expected_lock_sha256=lock_sha256,
    )
    local = evaluate_remote_environment(
        contract,
        expected_interpreter_class="project_venv",
        vendor_root=resolved / ".local" / "vendor" / "megaloc",
        stage="project",
    )
    docker = shutil.which("docker")
    linux_status = "not_verified_cache_absent"
    if docker is not None:
        try:
            inspected = subprocess.run(
                [docker, "image", "inspect", contract.image],
                cwd=resolved,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=30,
            )
            if inspected.returncode == 0:
                linux_status = "cached_image_present_live_contract_not_executed"
        except (OSError, subprocess.TimeoutExpired):
            linux_status = "not_verified_cache_unavailable"
    blockers = (
        []
        if local.passed
        else [local.failure_code or "REMOTE_DEPENDENCY_IMPORT_FAILED"]
    )
    wheelhouse = None
    try:
        wheelhouse = verify_framework_wheelhouse(
            contract,
            resolved.joinpath(*contract.framework_wheelhouse_local_path.split("/")),
        )
    except RemoteEnvironmentError as exc:
        blockers.append(exc.code)
    return {
        "schema": REMOTE_ENVIRONMENT_PLAN_SCHEMA,
        "action": "remote-environment-plan",
        "image": contract.image,
        "expected_image_digest": contract.image.split("@", 1)[1],
        "image_identity_policy": "requested_and_observed_exact_digest_when_reported",
        "python_requirement": "==3.12.*",
        "torch_requirement": contract.torch_requirement,
        "torchvision_requirement": contract.torchvision_requirement,
        "supported_torchvision_pairs": [
            {"torch_minor": torch, "torchvision_minor": torchvision}
            for torch, torchvision in contract.supported_torchvision_pairs
        ],
        "compatibility_source": "https://github.com/pytorch/vision#installation",
        "historical_observed_torch": "2.9.1+cu128",
        "historical_observed_torchvision": None,
        "cuda_requirement": contract.cuda_requirement,
        "environment_contract_sha256": contract.contract_sha256,
        "dependency_lock_sha256": contract.requirements_lock_sha256,
        "framework_wheelhouse_lock_sha256": (
            contract.framework_wheelhouse_lock_sha256
        ),
        "framework_wheelhouse_present": wheelhouse is not None,
        "framework_wheelhouse_artifact_count": (
            len(wheelhouse.artifacts) if wheelhouse is not None else 0
        ),
        "framework_wheelhouse_inventory_sha256": (
            wheelhouse.inventory_sha256 if wheelhouse is not None else None
        ),
        "framework_transfer_ready": wheelhouse is not None,
        "torchvision_companion_filename": contract.torchvision_companion_filename,
        "torchvision_companion_version": contract.torchvision_companion_version,
        "torchvision_companion_sha256": contract.torchvision_companion_sha256,
        "torchvision_companion_source": contract.torchvision_companion_source_url,
        "torchvision_companion_present": wheelhouse is not None,
        "package_index_resolution_on_pod": False,
        "module_count": len(contract.checks),
        "full_report_check_count": len(contract.checks) + 11 + 11,
        "local_dependency_check_count": local.report["check_count"],
        "all_local_imports_passed": local.passed,
        "linux_verification_status": linux_status,
        "bootstrap_required": True,
        "compatibility_smoke_required": True,
        "compatibility_smoke_timeout_seconds": contract.compatibility_smoke_timeout_seconds,
        "torch_download_prohibited": True,
        "estimated_bootstrap_timeout_seconds": contract.bootstrap_timeout_seconds,
        "interpreter_class": "project_venv",
        "system_site_packages": True,
        "official_package_index": contract.official_index_url,
        "model_downloads": 0,
        "dataset_downloads": 0,
        "local_blockers": blockers,
        "ready_for_live_bootstrap": not blockers,
        "runpod_api_calls": 0,
        "create_attempts": 0,
        "cloud_mutations": 0,
        "network_calls": 0,
        "dataset_writes": 0,
        "secrets_included": False,
    }


def training_plan(
    repository: Path,
    runtime_root: Path,
    cloud_runtime: Path,
) -> dict[str, object]:
    """Build a zero-network retry plan from integrity-bound local evidence only."""
    cloud = cloud_plan(repository, runtime_root, cloud_runtime)
    run_id = cast(str, cloud["run_id"])
    source = repository.resolve() / "services" / "api" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.training import training_config_sha256  # noqa: PLC0415
    from atlaslens_api.phase3f.pipeline import MODEL_SHA256  # noqa: PLC0415
    from atlaslens_api.phase3f.training_recovery import (  # noqa: PLC0415
        inspect_local_checkpoint,
    )

    remote_environment = remote_environment_plan(repository)
    environment_contract_sha256 = cast(
        str, remote_environment["environment_contract_sha256"]
    )
    config_sha256 = training_config_sha256(
        seed=20260720,
        max_epochs=8,
        environment_contract_sha256=environment_contract_sha256,
    )
    smoke_config_sha256 = training_config_sha256(seed=20260720, max_epochs=8)
    checkpoint = inspect_local_checkpoint(
        cloud_runtime.resolve() / "_training" / run_id / "checkpoints",
        expected_run_id=run_id,
        expected_readiness_sha256=cast(str, cloud["readiness_sha256"]),
        expected_sealed_assets_sha256=cast(str, cloud["sealed_assets_sha256"]),
        expected_training_config_sha256=config_sha256,
    )
    blockers = list(cast(list[str], cloud["local_blockers"]))
    blockers.extend(cast(list[str], remote_environment["local_blockers"]))
    if checkpoint.present and not checkpoint.valid:
        blockers.append(checkpoint.failure_code or "TRAINING_CHECKPOINT_INVALID")
    smoke_path = (
        repository.resolve()
        / ".local"
        / "phase3f"
        / "verification"
        / run_id
        / "retry-smoke.json"
    )
    smoke: dict[str, object] | None = None
    if smoke_path.is_file() and not smoke_path.is_symlink():
        smoke = _integrity_json(
            smoke_path,
            artifact="retry-smoke.json",
            missing_code="LOCAL_CUDA_SMOKE_MISSING",
            invalid_code="LOCAL_CUDA_SMOKE_INVALID",
            max_bytes=1024 * 1024,
        )
        smoke_valid = (
            smoke.get("schema") == "atlaslens-phase3f-local-cuda-smoke-v1"
            and smoke.get("run_id") == run_id
            and smoke.get("readiness_sha256") == cloud["readiness_sha256"]
            and smoke.get("sealed_assets_sha256") == cloud["sealed_assets_sha256"]
            and smoke.get("training_config_sha256") == smoke_config_sha256
            and smoke.get("model_sha256") == MODEL_SHA256
            and smoke.get("local_cuda_smoke_passed") is True
            and smoke.get("mini_epoch_passed") is True
            and smoke.get("checkpoint_roundtrip_passed") is True
            and smoke.get("failure_salvage_passed") is True
            and smoke.get("locked_holdout_access_count") == 0
            and smoke.get("nonfinite_loss_count") == 0
            and smoke.get("model_parameters_changed") is True
            and smoke.get("network_calls") == 0
            and smoke.get("runpod_api_calls") == 0
            and smoke.get("mapillary_api_calls") == 0
            and smoke.get("cloud_mutations") == 0
            and smoke.get("production_training_state_advanced") is False
            and smoke.get("production_configuration_changed") is False
            and smoke.get("temporary_cleanup_verified") is True
            and smoke.get("secrets_included") is False
        )
        if not smoke_valid:
            blockers.append("LOCAL_CUDA_SMOKE_INVALID")
    else:
        blockers.append("LOCAL_CUDA_SMOKE_MISSING")

    latest_failure: dict[str, object] | None = None
    attempts = cloud_runtime.resolve() / "_training" / run_id / "attempts"
    if attempts.is_dir() and not attempts.is_symlink():
        candidates = sorted(attempts.glob("*/recovery-*/recovery/failure.json"))
        if candidates:
            latest_failure = _integrity_json(
                candidates[-1],
                artifact="remote-failure.json",
                missing_code="REMOTE_FAILURE_RECEIPT_MISSING",
                invalid_code="REMOTE_FAILURE_RECEIPT_INVALID",
                max_bytes=1024 * 1024,
            )
    start_epoch = checkpoint.next_epoch if checkpoint.valid else 0
    start_step = checkpoint.optimizer_steps if checkpoint.valid else 0
    estimated_wall_minutes = max(30, int(330 * (8 - min(start_epoch, 7)) / 8))
    budget = cast(dict[str, object] | None, cloud["budget_snapshot"])
    proposed = Decimal("3") if budget is None else Decimal(cast(str, budget["proposed_run_max_usd"]))
    remaining = None if budget is None else Decimal(cast(str, budget["remaining_authorized_usd"]))
    projection = None if remaining is None else str(remaining - proposed)
    failure_code = (
        latest_failure.get("failure_code")
        if isinstance(latest_failure, dict)
        else "REMOTE_TRAINING_UNKNOWN_FAILURE"
    )
    return {
        "schema": "atlaslens-phase3f-training-plan-v1",
        "action": "training-plan",
        "run_id": run_id,
        "dataset_asset_count": cloud["dataset_asset_count"],
        "readiness_sha256": cloud["readiness_sha256"],
        "sealed_assets_sha256": cloud["sealed_assets_sha256"],
        "training_config_sha256": config_sha256,
        "local_smoke_training_config_sha256": smoke_config_sha256,
        "environment_contract_sha256": environment_contract_sha256,
        "dependency_lock_sha256": remote_environment["dependency_lock_sha256"],
        "remote_environment_plan": remote_environment,
        "failure_forensic": {
            "failure_code": failure_code,
            "evidence_sufficient": latest_failure is not None,
            "process_exit_code": (
                latest_failure.get("process_exit_code") if latest_failure else None
            ),
            "process_signal": (
                latest_failure.get("process_signal") if latest_failure else None
            ),
            "last_completed_epoch": (
                latest_failure.get("last_completed_epoch") if latest_failure else None
            ),
            "last_epoch": latest_failure.get("last_epoch") if latest_failure else None,
            "last_step": latest_failure.get("last_step") if latest_failure else None,
            "holdout_open_count": (
                latest_failure.get("holdout_open_count") if latest_failure else None
            ),
        },
        "local_cuda_smoke_passed": (
            smoke.get("local_cuda_smoke_passed") if smoke is not None else False
        ),
        "mini_epoch_passed": smoke.get("mini_epoch_passed") if smoke else False,
        "checkpoint_roundtrip_passed": (
            smoke.get("checkpoint_roundtrip_passed") if smoke else False
        ),
        "failure_salvage_passed": (
            smoke.get("failure_salvage_passed") if smoke else False
        ),
        "locked_holdout_access_count": (
            smoke.get("locked_holdout_access_count") if smoke else None
        ),
        "nonfinite_loss_count": smoke.get("nonfinite_loss_count") if smoke else None,
        "model_parameters_changed": (
            smoke.get("model_parameters_changed") if smoke else False
        ),
        "local_cuda_smoke": smoke,
        "checkpoint": checkpoint.to_public_dict(),
        "retry_mode": (
            "resume" if checkpoint.valid else "blocked" if checkpoint.present else "fresh"
        ),
        "start_epoch": start_epoch,
        "start_step": start_step,
        "estimated_remaining_wall_minutes": estimated_wall_minutes,
        "estimated_maximum_new_cost_usd": str(proposed),
        "budget_projection": {
            "authoritative_snapshot": budget,
            "actual_billed_usd": (
                budget.get("actual_billed_usd") if budget is not None else None
            ),
            "conservative_unbilled_estimate_usd": (
                budget.get("conservative_unbilled_estimate_usd")
                if budget is not None
                else None
            ),
            "proposed_run_max_usd": (
                budget.get("proposed_run_max_usd") if budget is not None else None
            ),
            "projected_total_usd": (
                budget.get("projected_total_usd") if budget is not None else None
            ),
            "projected_remaining_authorized_usd_after_retry": projection,
        },
        "local_blockers": blockers,
        "ready_for_training_retry": not blockers,
        "dataset_write_count": 0,
        "secret_prompt_count": 0,
        "runpod_api_calls": 0,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def reconcile_local_receipts(cloud_runtime: Path) -> dict[str, object]:
    source = Path(__file__).resolve().parents[2] / "services" / "api" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.operator import (  # noqa: PLC0415
        reconcile_local_operator_receipts,
    )

    operator_root = cloud_runtime.resolve() / "_operator"
    result = reconcile_local_operator_receipts(operator_root)
    return {
        "schema": "atlaslens-phase3f-local-receipt-reconciliation-v1",
        "action": "reconcile-local-receipts",
        **result,
        "dataset_write_count": 0,
        "secret_prompt_count": 0,
        "runpod_api_calls": 0,
        "mapillary_api_calls": 0,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def status(repository: Path, local_runtime: Path, cloud_runtime: Path) -> dict[str, object]:
    source = repository.resolve() / "services" / "api" / "src"
    _require(source.is_dir(), "PHASE3F_SOURCE_ROOT_INVALID")
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.local_first import (  # noqa: PLC0415
        status as local_status,
    )

    local = local_status(local_runtime)
    operator_path = cloud_runtime / "_operator" / "phase3f-current.json"
    operator: dict[str, object] | None = None
    if operator_path.exists():
        raw = _read_json(operator_path, max_bytes=1024 * 1024)
        operator = {
            key: raw.get(key)
            for key in (
                "run_id",
                "stage",
                "started_at",
                "finished_at",
                "cleanup_verified",
            )
        }
    return {
        "action": "status",
        "local": local,
        "cloud_operator": operator,
        "live_cloud_inventory_checked": False,
        "secrets_included": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-end-to-end")
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "readiness",
            "resume-plan",
            "cloud-plan",
            "remote-environment-plan",
            "training-plan",
            "reconcile-local-receipts",
            "status",
        ),
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cloud-runtime-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "preflight":
            result = preflight(args.repository_root, args.runtime_root)
        elif args.action == "readiness":
            result = readiness(args.repository_root, args.runtime_root)
        elif args.action == "resume-plan":
            _require(args.cloud_runtime_root is not None, "CLOUD_RUNTIME_ROOT_MISSING")
            result = resume_plan(
                args.repository_root,
                args.runtime_root,
                args.cloud_runtime_root,
            )
        elif args.action == "cloud-plan":
            _require(args.cloud_runtime_root is not None, "CLOUD_RUNTIME_ROOT_MISSING")
            result = cloud_plan(
                args.repository_root,
                args.runtime_root,
                args.cloud_runtime_root,
            )
        elif args.action == "remote-environment-plan":
            result = remote_environment_plan(args.repository_root)
        elif args.action == "training-plan":
            _require(args.cloud_runtime_root is not None, "CLOUD_RUNTIME_ROOT_MISSING")
            result = training_plan(
                args.repository_root,
                args.runtime_root,
                args.cloud_runtime_root,
            )
        elif args.action == "reconcile-local-receipts":
            _require(args.cloud_runtime_root is not None, "CLOUD_RUNTIME_ROOT_MISSING")
            result = reconcile_local_receipts(args.cloud_runtime_root)
        else:
            _require(args.cloud_runtime_root is not None, "CLOUD_RUNTIME_ROOT_MISSING")
            result = status(args.repository_root, args.runtime_root, args.cloud_runtime_root)
    except Exception as exc:  # sanitized operator boundary
        code = getattr(exc, "code", None)
        if isinstance(exc, EndToEndError) and len(exc.public_document()) > 1:
            print(json.dumps(exc.public_document(), separators=(",", ":"), sort_keys=True))
        else:
            print(code if isinstance(code, str) else "PHASE3F_END_TO_END_FAILED")
        return 1
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
