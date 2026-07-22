"""One-shot local RunPod supervisor for the authorized Phase 3F pilot."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import NamedTuple, cast

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "services" / "api" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from atlaslens_api.phase3f.deadline import (  # noqa: E402
    BOOTSTRAP_BUDGET_SECONDS,
    SALVAGE_RESERVE_SECONDS,
    TrainingDeadlineError,
    build_training_deadline_plan,
    compute_child_deadline,
    require_training_deadline_plan,
)
from atlaslens_api.phase3f.operator import (  # noqa: E402
    OperatorReceipt,
    Phase3FOperatorError,
    inspect_operator_inventory,
    operator_budget_linkage,
    prepare_operator_attempt,
    read_operator_receipt,
    require_startable_receipt,
    terminate_receipt_bound_pod,
    write_operator_receipt,
)
from atlaslens_api.phase3f.memory import (  # noqa: E402
    FALLBACK_CLOUD_VRAM_GIB,
    MemoryPlanError,
    require_production_memory_evidence,
)
from atlaslens_api.phase3f.runpod import (  # noqa: E402
    GPUOffer,
    PodConnection,
    PodConnectivityProgressDiagnostic,
    PodGPUAttestationProgressDiagnostic,
    PodRentalAttestationDiagnostic,
    RunPodAPIError,
    RunPodBillingSnapshot,
    RunPodConfig,
    RunPodInventory,
    RunPodV1Client,
    build_create_payload,
)
from atlaslens_api.phase3f.remote_environment import (  # noqa: E402
    DEPENDENCY_FAILURE_CODES,
    REMOTE_DEPENDENCY_REPORT_SCHEMA,
    REMOTE_ENVIRONMENT_RECEIPT_SCHEMA,
    FrameworkWheelhouse,
    RemoteEnvironmentError,
    load_remote_environment_contract,
    sha256_file as environment_sha256_file,
    verify_framework_wheelhouse,
)
from atlaslens_api.phase3f.safety import (  # noqa: E402
    BudgetPolicy,
    Phase3FSafetyError,
    PodRecord,
    PodRequest,
    RunPodLease,
    SinglePodSession,
    TARGET_BUDGET_USD,
)
from atlaslens_api.phase3f.supervisor import (  # noqa: E402
    LICENSE_CRLF_SHA256,
    LICENSE_LF_SHA256,
    MODEL_SHA256,
    MODEL_SIZE,
    Phase3FSupervisorError,
    SOURCE_CRLF_SHA256,
    SOURCE_LF_SHA256,
    prepare_transfer_bundle,
    publish_verified_output,
    require_empty_inventory,
    require_inventory_restored,
    verify_output_archive,
)
from atlaslens_api.phase3f.training import training_config_sha256  # noqa: E402
from atlaslens_api.phase3f.training_recovery import (  # noqa: E402
    SALVAGE_RECEIPT_SCHEMA,
    TrainingRecoveryError,
    atomic_json,
    extract_recovery_archive,
    inspect_local_checkpoint,
    sha256_path,
    store_verified_checkpoint_archive,
)

REQUIRED_BRANCH = "feature/phase3f-multiregion-pilot"
REQUIRED_BASE = "183b7c2f410c41d3449238f156e73fed493e9dd9"
REQUIRED_PHASE3F_COMMIT = "4d6e8f222bb4fc3b3d77463649ff221a7a6ad676"
REQUIRED_ORIGIN = "https://github.com/TahaFirat/atlaslens-private.git"
MEGALOC_SOURCE_REVISION = "1af071c68fc3ab6c6018c5c868391763516e50f7"
MEGALOC_MODEL_REVISION = "7cb9f7970d366fdf059963d04d372e503e8e9df9"
IMAGE = (
    "runpod/pytorch@sha256:"
    "60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f"
)
GPU_PREFERENCES = (
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
)
E2E_RUN_BUDGET_USD = Decimal("3")
E2E_HISTORICAL_BUDGET_USD = Decimal("10")
E2E_CLOSED_POD_DISK_ALLOWANCE_USD = Decimal("0.10")
MAX_HISTORICAL_RECEIPTS = 1_000
CHECKPOINT_SYNC_SECONDS = 5 * 60
REMOTE_ENVIRONMENT_CONTRACT = ROOT / "config" / "phase3f-remote-environment.json"
REMOTE_REQUIREMENTS_LOCK = ROOT / "config" / "phase3f-training-requirements.lock"
REMOTE_FRAMEWORK_WHEELHOUSE = (
    ROOT / ".local" / "phase3f" / "wheelhouse" / "linux-cp312-cu128"
)
_SUPERVISOR_RECEIPT_SCHEMA = "atlaslens-phase3f-local-supervisor-receipt-v1"
_BUDGET_RECONCILIATION_SCHEMA = "atlaslens-phase3f-budget-reconciliation-v1"
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SupervisorExecutionError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class BudgetReconciliation(NamedTuple):
    actual_billed_usd: Decimal
    conservative_unbilled_estimate_usd: Decimal
    active_exposure_usd: Decimal
    closed_unused_reservation_released_usd: Decimal
    proposed_run_max_usd: Decimal
    historical_total_cap_usd: Decimal
    remaining_authorized_usd: Decimal
    projected_total_usd: Decimal
    legacy_closed_reservation_total_usd: Decimal
    local_conservative_historical_usd: Decimal
    provider_balance_delta_usd: Decimal
    source_receipts_sha256: str
    provider_snapshot_sha256: str
    source_receipt_count: int
    duplicate_billing_record_count: int
    unrecognized_active_billing: bool
    billing_lag_attempt_count: int
    billing_lag_local_estimate_usd: Decimal
    provider_increment_since_prior_snapshot_usd: Decimal


class RemoteProcessResult(NamedTuple):
    return_code: int
    elapsed_seconds: float


class SalvageResult(NamedTuple):
    attempted: bool
    succeeded: bool
    artifact_count: int
    checkpoint_present: bool
    checkpoint_sha256: str | None
    failure_code: str


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SupervisorExecutionError(code)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=D:/geoSearch", *args],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    _require(completed.returncode == 0, "GIT_PREFLIGHT_FAILED")
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _preflight_repository() -> tuple[str, tuple[str, ...]]:
    _require(_git("branch", "--show-current") == REQUIRED_BRANCH, "GIT_BRANCH_MISMATCH")
    head = _git("rev-parse", "HEAD")
    _require(_git("merge-base", "main", "HEAD") == REQUIRED_BASE, "GIT_BASE_MISMATCH")
    _git("merge-base", "--is-ancestor", REQUIRED_PHASE3F_COMMIT, "HEAD")
    _require(_git("remote", "get-url", "origin") == REQUIRED_ORIGIN, "GIT_ORIGIN_MISMATCH")
    status = tuple(
        line for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if line
    )
    _require(status in {(), ("?? asda.html",)}, "GIT_WORKTREE_NOT_CLEAN")
    raw = subprocess.run(
        ["git", "-c", "safe.directory=D:/geoSearch", "ls-files", "-z"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    _require(raw.returncode == 0, "GIT_TRACKED_FILES_UNAVAILABLE")
    tracked = tuple(
        value.decode("utf-8", errors="strict")
        for value in raw.stdout.split(b"\x00")
        if value
    )
    return head, tracked


def _disk_gate() -> None:
    c_free = shutil.disk_usage("C:\\").free
    d_free = shutil.disk_usage("D:\\").free
    _require(c_free >= 35 * 1024**3, "C_DISK_GATE_FAILED")
    _require(d_free >= 8 * 1024**3, "D_DISK_GATE_FAILED")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_crlf_artifact(
    path: Path,
    *,
    crlf_size: int,
    crlf_sha256: str,
    lf_size: int,
    lf_sha256: str,
    newline_count: int,
) -> None:
    _require(path.is_file() and not path.is_symlink(), "MEGALOC_VENDOR_MISSING")
    payload = path.read_bytes()
    _require(
        len(payload) == crlf_size and hashlib.sha256(payload).hexdigest() == crlf_sha256,
        "MEGALOC_VENDOR_HASH_MISMATCH",
    )
    _require(
        not payload.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")),
        "MEGALOC_VENDOR_BOM_INVALID",
    )
    _require(
        payload.count(b"\r\n") == newline_count
        and payload.count(b"\n") == newline_count
        and b"\r" not in payload.replace(b"\r\n", b""),
        "MEGALOC_VENDOR_LINE_ENDINGS_INVALID",
    )
    canonical = payload.replace(b"\r\n", b"\n")
    _require(
        len(canonical) == lf_size and hashlib.sha256(canonical).hexdigest() == lf_sha256,
        "CANONICAL_VENDOR_MISMATCH",
    )


def _require_model_receipt(receipt_path: Path) -> None:
    _require(
        receipt_path.is_file()
        and not receipt_path.is_symlink()
        and 0 < receipt_path.stat().st_size <= 64 * 1024,
        "MEGALOC_RECEIPT_MISSING",
    )
    try:
        value = json.loads(receipt_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisorExecutionError("MEGALOC_RECEIPT_INVALID") from exc
    _require(isinstance(value, dict), "MEGALOC_RECEIPT_INVALID")
    receipt = value
    _require(
        receipt.get("schema_version") == "atlaslens-model-receipt-v1"
        and receipt.get("source_revision") == MEGALOC_SOURCE_REVISION
        and receipt.get("model_revision") == MEGALOC_MODEL_REVISION
        and receipt.get("weight_file") == "model.safetensors"
        and receipt.get("weight_size") == MODEL_SIZE
        and receipt.get("weight_sha256") == MODEL_SHA256
        and receipt.get("license_text_sha256") == LICENSE_CRLF_SHA256,
        "MEGALOC_RECEIPT_INVALID",
    )
    review = receipt.get("static_security_review")
    transformation = receipt.get("transformation")
    source_files = receipt.get("source_files")
    _require(
        isinstance(review, dict)
        and review.get("status") == "pass"
        and review.get("lf_ast_parse") == "pass"
        and review.get("crlf_ast_parse") == "pass"
        and review.get("ast_semantics_identical") is True
        and review.get("network_access_findings") == 0
        and review.get("subprocess_or_shell_findings") == 0
        and review.get("dynamic_download_findings") == 0
        and review.get("environment_or_credential_findings") == 0
        and review.get("unexpected_file_write_findings") == 0,
        "MEGALOC_RECEIPT_SECURITY_REVIEW_INVALID",
    )
    _require(
        isinstance(transformation, dict)
        and transformation.get("encoding_conversion") is False
        and transformation.get("bom_added") is False
        and transformation.get("whitespace_or_source_semantics_changed") is False
        and transformation.get("reverse_canonicalization_byte_identical") is True,
        "MEGALOC_RECEIPT_TRANSFORMATION_INVALID",
    )
    _require(isinstance(source_files, dict), "MEGALOC_RECEIPT_INVALID")
    source = source_files.get("megaloc_model.py")
    license_row = source_files.get("LICENSE")
    _require(
        isinstance(source, dict)
        and source.get("upstream_lf_size") == 15_258
        and source.get("upstream_lf_sha256") == SOURCE_LF_SHA256
        and source.get("local_windows_crlf_size") == 15_726
        and source.get("local_windows_crlf_sha256") == SOURCE_CRLF_SHA256
        and source.get("canonical_lf_sha256") == SOURCE_LF_SHA256
        and source.get("lf_newline_count") == 468
        and source.get("crlf_newline_count") == 468,
        "MEGALOC_RECEIPT_INVALID",
    )
    _require(
        isinstance(license_row, dict)
        and license_row.get("upstream_lf_size") == 1_086
        and license_row.get("upstream_lf_sha256") == LICENSE_LF_SHA256
        and license_row.get("local_windows_crlf_size") == 1_107
        and license_row.get("local_windows_crlf_sha256") == LICENSE_CRLF_SHA256
        and license_row.get("canonical_lf_sha256") == LICENSE_LF_SHA256
        and license_row.get("lf_newline_count") == 21
        and license_row.get("crlf_newline_count") == 21,
        "MEGALOC_RECEIPT_INVALID",
    )


def _verify_local_readiness() -> None:
    model_root = ROOT / ".local" / "models" / "phase6c" / "megaloc"
    model = model_root / "model.safetensors"
    _require(model.is_file() and not model.is_symlink(), "MEGALOC_MODEL_MISSING")
    _require(
        model.stat().st_size == MODEL_SIZE and _sha256_path(model) == MODEL_SHA256,
        "MEGALOC_MODEL_HASH_MISMATCH",
    )
    vendor = ROOT / ".local" / "vendor" / "megaloc"
    _require_crlf_artifact(
        vendor / "megaloc_model.py",
        crlf_size=15_726,
        crlf_sha256=SOURCE_CRLF_SHA256,
        lf_size=15_258,
        lf_sha256=SOURCE_LF_SHA256,
        newline_count=468,
    )
    _require_crlf_artifact(
        vendor / "LICENSE",
        crlf_size=1_107,
        crlf_sha256=LICENSE_CRLF_SHA256,
        lf_size=1_086,
        lf_sha256=LICENSE_LF_SHA256,
        newline_count=21,
    )
    _require_model_receipt(model_root / "receipt.json")


def _require_production_memory_plan(sealed_root: Path, *, run_id: str) -> int:
    readiness = sealed_root.parent / "training-readiness.json"
    evidence = (
        ROOT
        / ".local"
        / "phase3f"
        / "verification"
        / run_id
        / "production-memory-smoke.json"
    )
    _require(
        readiness.is_file()
        and not readiness.is_symlink()
        and evidence.is_file()
        and not evidence.is_symlink(),
        "GPU_MEMORY_PLAN_UNSATISFIED",
    )
    try:
        document = json.loads(evidence.read_text(encoding="utf-8"))
        _require(isinstance(document, dict), "GPU_MEMORY_PLAN_UNSATISFIED")
        measured = require_production_memory_evidence(
            cast(dict[str, object], document),
            run_id=run_id,
            readiness_sha256=_sha256_path(readiness),
            sealed_assets_sha256=_sha256_path(sealed_root / "sealed-assets.json"),
            checksum_inventory_sha256=_sha256_path(
                sealed_root / "checksum-inventory.json"
            ),
            model_sha256=MODEL_SHA256,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, MemoryPlanError, ValueError) as exc:
        raise SupervisorExecutionError("GPU_MEMORY_PLAN_UNSATISFIED") from exc
    _require(
        measured.requirement.required_vram_gib <= FALLBACK_CLOUD_VRAM_GIB,
        "GPU_MEMORY_PLAN_UNSATISFIED",
    )
    return measured.requirement.required_vram_gib


def _remote_environment_preflight() -> tuple[str, str, FrameworkWheelhouse]:
    contract_sha256 = environment_sha256_file(REMOTE_ENVIRONMENT_CONTRACT)
    lock_sha256 = environment_sha256_file(REMOTE_REQUIREMENTS_LOCK)
    contract = load_remote_environment_contract(
        REMOTE_ENVIRONMENT_CONTRACT,
        REMOTE_REQUIREMENTS_LOCK,
        expected_contract_sha256=contract_sha256,
        expected_lock_sha256=lock_sha256,
    )
    wheelhouse = verify_framework_wheelhouse(contract, REMOTE_FRAMEWORK_WHEELHOUSE)
    return contract_sha256, lock_sha256, wheelhouse


def _require_safe_runtime_path(path: Path) -> Path:
    resolved = path.resolve()
    repository = ROOT.resolve()
    try:
        relative = resolved.relative_to(repository)
    except ValueError:
        return resolved
    _require(relative.parts[:1] == (".local",), "OPERATOR_RUNTIME_PATH_NOT_PRIVATE")
    return resolved


def _emit(code: str, **fields: object) -> None:
    print(
        json.dumps(
            {"event": code, **fields},
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _run_command(arguments: list[str], *, timeout_seconds: float) -> None:
    try:
        completed = subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=max(1.0, timeout_seconds),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SupervisorExecutionError("TRANSFER_COMMAND_FAILED") from exc
    _require(completed.returncode == 0, "TRANSFER_COMMAND_FAILED")


def _run_sha256_command(arguments: list[str], *, timeout_seconds: float) -> str:
    try:
        completed = subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=max(1.0, timeout_seconds),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SupervisorExecutionError("TRANSFER_COMMAND_FAILED") from exc
    _require(completed.returncode == 0, "TRANSFER_COMMAND_FAILED")
    try:
        digest = completed.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise SupervisorExecutionError("CHECKPOINT_REMOTE_HASH_INVALID") from exc
    _require(bool(_SHA256.fullmatch(digest)), "CHECKPOINT_REMOTE_HASH_INVALID")
    return digest


def _ssh_arguments(key: Path, known_hosts: Path, connection: PodConnection) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key),
        "-p",
        str(connection.public_ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        f"root@{connection.public_ip}",
    ]


def _scp_arguments(key: Path, known_hosts: Path, connection: PodConnection) -> list[str]:
    return [
        "scp",
        "-q",
        "-i",
        str(key),
        "-P",
        str(connection.public_ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
    ]


def _wait_connection(
    client: RunPodV1Client,
    lease: RunPodLease,
    started: float,
    *,
    key: Path,
    known_hosts: Path,
) -> PodConnection:
    lease.assert_within_limits(elapsed_seconds=int(time.monotonic() - started))
    connection = client.await_pod_connectivity(
        lease.pod,
        lease.request,
        ssh_probe=lambda candidate, timeout_seconds: _probe_ssh(
            key,
            known_hosts,
            candidate,
            timeout_seconds=timeout_seconds,
        ),
    )
    lease.assert_within_limits(elapsed_seconds=int(time.monotonic() - started))
    return connection


def _probe_ssh(
    key: Path,
    known_hosts: Path,
    connection: PodConnection,
    *,
    timeout_seconds: float,
) -> bool:
    try:
        completed = subprocess.run(
            [*_ssh_arguments(key, known_hosts, connection), "true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=max(0.1, min(timeout_seconds, 15.0)),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _run_watched(
    arguments: list[str],
    *,
    lease: RunPodLease,
    started: float,
    periodic: Callable[[], object] | None = None,
    periodic_seconds: int = CHECKPOINT_SYNC_SECONDS,
    status_event: str = "PHASE3F_CLOUD_JOB_RUNNING",
) -> RemoteProcessResult:
    next_status = 0
    next_periodic = periodic_seconds
    process_started = time.monotonic()
    try:
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise SupervisorExecutionError("REMOTE_JOB_START_FAILED") from exc
    try:
        while process.poll() is None:
            elapsed = int(time.monotonic() - started)
            lease.assert_within_limits(elapsed_seconds=elapsed)
            if elapsed >= next_status:
                _emit(status_event, elapsed_seconds=elapsed)
                next_status = elapsed + 60
            process_elapsed = time.monotonic() - process_started
            if periodic is not None and process_elapsed >= next_periodic:
                periodic()
                next_periodic += periodic_seconds
            time.sleep(15)
        return RemoteProcessResult(
            return_code=process.returncode if process.returncode is not None else 1,
            elapsed_seconds=time.monotonic() - process_started,
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


def _sync_remote_checkpoint(
    ssh: list[str],
    scp: list[str],
    *,
    public_ip: str,
    remote_transfer: str,
    checkpoint_store: Path,
    partial_path: Path,
    run_id: str,
    readiness_sha256: str,
    sealed_assets_sha256: str,
    config_sha256: str,
    timeout_seconds: float,
) -> bool:
    sync_started = time.monotonic()

    def remaining() -> float:
        return timeout_seconds - (time.monotonic() - sync_started)

    partial_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial_path.unlink(missing_ok=True)
    remote_archive = f"{remote_transfer}/checkpoint-sync.tar"
    try:
        _require(remaining() > 0, "CHECKPOINT_SYNC_TIMEOUT")
        remote_manifest_sha256 = _run_sha256_command(
            [
                *ssh,
                "bash",
                "-lc",
                shlex.quote(
                    "phase3f_generation=$(python -c \"import json; "
                    "print(json.load(open('/workspace/phase3f-work/training-checkpoint/latest.json', "
                    "encoding='utf-8'))['generation'])\") "
                    "&& (sha256sum /workspace/phase3f-work/training-checkpoint/"
                    "\"$phase3f_generation\"/checkpoint-manifest.json; "
                    "if test -f /workspace/phase3f-work/training-checkpoint/holdout-state.json; "
                    "then sha256sum /workspace/phase3f-work/training-checkpoint/"
                    "holdout-state.json; fi) | sha256sum | cut -d' ' -f1"
                ),
            ],
            timeout_seconds=max(1.0, remaining()),
        )
        sync_state_path = checkpoint_store / "sync-state.json"
        if sync_state_path.is_file() and not sync_state_path.is_symlink():
            try:
                sync_state = json.loads(sync_state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                sync_state = None
            if (
                isinstance(sync_state, dict)
                and sync_state.get("remote_manifest_sha256") == remote_manifest_sha256
            ):
                return False
        _run_command(
            [
                *ssh,
                "bash",
                "-lc",
                shlex.quote(
                    "test -f /workspace/phase3f-work/training-checkpoint/latest.json "
                    "&& phase3f_generation=$(python -c \"import json; "
                    "print(json.load(open('/workspace/phase3f-work/training-checkpoint/latest.json', "
                    "encoding='utf-8'))['generation'])\") "
                    "&& tar -C /workspace/phase3f-work/training-checkpoint "
                    f"-cf {remote_archive} latest.json \"$phase3f_generation\" "
                    "&& (if test -f /workspace/phase3f-work/training-checkpoint/"
                    "best-trainable.safetensors; then tar -C /workspace/phase3f-work/"
                    "training-checkpoint -rf "
                    f"{remote_archive} best-trainable.safetensors; fi) "
                    "&& (if test -f /workspace/phase3f-work/training-checkpoint/"
                    "holdout-state.json; then tar -C /workspace/phase3f-work/"
                    "training-checkpoint -rf "
                    f"{remote_archive} holdout-state.json; fi)"
                ),
            ],
            timeout_seconds=max(1.0, remaining()),
        )
        _require(remaining() > 0, "CHECKPOINT_SYNC_TIMEOUT")
        _run_command(
            [*scp, f"root@{public_ip}:{remote_archive}", str(partial_path)],
            timeout_seconds=max(1.0, remaining()),
        )
        _require(remaining() > 0, "CHECKPOINT_SYNC_TIMEOUT")
        status = store_verified_checkpoint_archive(
            partial_path,
            checkpoint_store,
            expected_run_id=run_id,
            expected_readiness_sha256=readiness_sha256,
            expected_sealed_assets_sha256=sealed_assets_sha256,
            expected_training_config_sha256=config_sha256,
        )
        atomic_json(
            sync_state_path,
            {
                "schema": "atlaslens-phase3f-checkpoint-sync-v1",
                "remote_manifest_sha256": remote_manifest_sha256,
                "checkpoint_archive_sha256": status.archive_sha256,
                "optimizer_steps": status.optimizer_steps,
                "secrets_included": False,
            },
        )
        _emit(
            "PHASE3F_CHECKPOINT_SYNCED",
            checkpoint_sha256=status.archive_sha256,
            optimizer_steps=status.optimizer_steps,
            holdout_open_count=status.holdout_open_count,
        )
        return True
    except (OSError, SupervisorExecutionError, TrainingRecoveryError):
        partial_path.unlink(missing_ok=True)
        return False


def _dependency_failure_evidence(files: tuple[Path, ...]) -> tuple[str, str]:
    required = {
        "failure.json",
        "dependency-report.json",
        "environment-receipt.json",
        "bootstrap-stdout.log",
        "bootstrap-stderr.log",
        "checksum-inventory.json",
    }
    by_name = {path.name: path for path in files}
    _require(required.issubset(by_name), "REMOTE_DEPENDENCY_SALVAGE_INCOMPLETE")
    try:
        failure = json.loads(by_name["failure.json"].read_text(encoding="utf-8"))
        report = json.loads(
            by_name["dependency-report.json"].read_text(encoding="utf-8")
        )
        receipt = json.loads(
            by_name["environment-receipt.json"].read_text(encoding="utf-8")
        )
        inventory = json.loads(
            by_name["checksum-inventory.json"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SupervisorExecutionError("REMOTE_DEPENDENCY_SALVAGE_INVALID") from exc
    _require(
        isinstance(failure, dict)
        and isinstance(report, dict)
        and isinstance(receipt, dict)
        and isinstance(inventory, dict),
        "REMOTE_DEPENDENCY_SALVAGE_INVALID",
    )
    exact = report.get("dependency_failure_code")
    module = report.get("failed_module")
    exception_class = report.get("failure_exception_class")
    _require(
        report.get("schema") == REMOTE_DEPENDENCY_REPORT_SCHEMA
        and report.get("outcome") == "failed"
        and exact in DEPENDENCY_FAILURE_CODES
        and isinstance(module, str)
        and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", module))
        and isinstance(exception_class, str)
        and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", exception_class))
        and report.get("secrets_included") is False
        and receipt.get("schema") == REMOTE_ENVIRONMENT_RECEIPT_SCHEMA
        and receipt.get("outcome") == "failed"
        and receipt.get("dependency_failure_code") == exact
        and receipt.get("failed_module") == module
        and receipt.get("secrets_included") is False
        and failure.get("failure_code") == "REMOTE_TRAINING_DEPENDENCY_FAILED"
        and failure.get("dependency_failure_code") == exact
        and failure.get("failed_module") == module
        and failure.get("secrets_included") is False,
        "REMOTE_DEPENDENCY_SALVAGE_INVALID",
    )
    rows = inventory.get("files")
    _require(
        inventory.get("schema")
        == "atlaslens-phase3f-bootstrap-checksum-inventory-v1"
        and isinstance(rows, list)
        and inventory.get("file_count") == len(rows)
        and inventory.get("secrets_included") is False,
        "REMOTE_DEPENDENCY_SALVAGE_INVALID",
    )
    inventory_names = {row.get("name") for row in rows if isinstance(row, dict)}
    _require(
        inventory_names == set(by_name) - {"checksum-inventory.json"},
        "REMOTE_DEPENDENCY_SALVAGE_INVALID",
    )
    for row in rows:
        _require(isinstance(row, dict), "REMOTE_DEPENDENCY_SALVAGE_INVALID")
        name = row.get("name")
        size = row.get("size_bytes")
        sha256 = row.get("sha256")
        _require(
            isinstance(name, str)
            and name in by_name
            and isinstance(size, int)
            and not isinstance(size, bool)
            and by_name[name].stat().st_size == size
            and isinstance(sha256, str)
            and bool(_SHA256.fullmatch(sha256))
            and sha256_path(by_name[name], max_bytes=1024 * 1024) == sha256,
            "REMOTE_DEPENDENCY_SALVAGE_INVALID",
        )
    return cast(str, exact), cast(str, module)


def _salvage_remote_failure(
    ssh: list[str],
    scp: list[str],
    *,
    public_ip: str,
    remote_transfer: str,
    attempt_root: Path,
    checkpoint_store: Path,
    run_id: str,
    readiness_sha256: str,
    sealed_assets_sha256: str,
    config_sha256: str,
    process_result: RemoteProcessResult,
    timeout_seconds: float,
) -> SalvageResult:
    started = time.monotonic()
    attempt_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    recovery_partial = attempt_root / "recovery.tar.partial"
    recovery_archive = attempt_root / "recovery.tar"
    checkpoint_partial = attempt_root / "checkpoint.tar.partial"
    artifact_count = 0
    checkpoint_present = False
    checkpoint_sha256: str | None = None
    failure_code = "REMOTE_TRAINING_UNKNOWN_FAILURE"
    dependency_failure_code: str | None = None
    dependency_failed_module: str | None = None
    salvage_failure_code: str | None = None
    remote_exit_code: int | None = (
        process_result.return_code if process_result.return_code >= 0 else None
    )
    remote_signal: int | None = (
        -process_result.return_code if process_result.return_code < 0 else None
    )
    succeeded = False

    def remaining() -> float:
        return min(timeout_seconds, SALVAGE_RESERVE_SECONDS) - (time.monotonic() - started)

    try:
        _require(remaining() > 0, "REMOTE_TRAINING_SALVAGE_TIMEOUT")
        remote_archive = f"{remote_transfer}/failure-recovery.tar"
        _run_command(
            [
                *ssh,
                "bash",
                "-lc",
                shlex.quote(
                    "test -d /workspace/phase3f-work/recovery "
                    f"&& tar -C /workspace/phase3f-work -cf {remote_archive} recovery"
                ),
            ],
            timeout_seconds=max(1.0, remaining()),
        )
        _run_command(
            [*scp, f"root@{public_ip}:{remote_archive}", str(recovery_partial)],
            timeout_seconds=max(1.0, remaining()),
        )
        recovery_digest = sha256_path(recovery_partial)
        os.replace(recovery_partial, recovery_archive)
        files = extract_recovery_archive(
            recovery_archive,
            attempt_root / f"recovery-{recovery_digest}",
        )
        _require(remaining() > 0, "REMOTE_TRAINING_SALVAGE_TIMEOUT")
        artifact_count = len(files) + 1
        failure_candidates = tuple(
            path for path in files if path.name == "failure.json"
        )
        if len(failure_candidates) == 1:
            failure = json.loads(failure_candidates[0].read_text(encoding="utf-8"))
            candidate = failure.get("failure_code") if isinstance(failure, dict) else None
            if isinstance(candidate, str) and re.fullmatch(r"REMOTE_TRAINING_[A-Z0-9_]+", candidate):
                failure_code = candidate
            exit_candidate = failure.get("process_exit_code")
            signal_candidate = failure.get("process_signal")
            if (
                isinstance(exit_candidate, int)
                and not isinstance(exit_candidate, bool)
                and 0 <= exit_candidate <= 255
            ):
                remote_exit_code = exit_candidate
            if (
                isinstance(signal_candidate, int)
                and not isinstance(signal_candidate, bool)
                and 1 <= signal_candidate <= 64
            ):
                remote_signal = signal_candidate
        if failure_code == "REMOTE_TRAINING_DEPENDENCY_FAILED":
            dependency_failure_code, dependency_failed_module = (
                _dependency_failure_evidence(files)
            )
            failure_code = dependency_failure_code
        checkpoint_synced = _sync_remote_checkpoint(
            ssh,
            scp,
            public_ip=public_ip,
            remote_transfer=remote_transfer,
            checkpoint_store=checkpoint_store,
            partial_path=checkpoint_partial,
            run_id=run_id,
            readiness_sha256=readiness_sha256,
            sealed_assets_sha256=sealed_assets_sha256,
            config_sha256=config_sha256,
            timeout_seconds=max(1.0, remaining()),
        )
        checkpoint = inspect_local_checkpoint(
            checkpoint_store,
            expected_run_id=run_id,
            expected_readiness_sha256=readiness_sha256,
            expected_sealed_assets_sha256=sealed_assets_sha256,
            expected_training_config_sha256=config_sha256,
        )
        checkpoint_present = checkpoint.valid
        if checkpoint.valid:
            checkpoint_sha256 = checkpoint.archive_sha256
            artifact_count += int(checkpoint_synced)
        succeeded = True
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        SupervisorExecutionError,
        TrainingRecoveryError,
        RemoteEnvironmentError,
    ) as exc:
        if isinstance(exc, SupervisorExecutionError) and exc.code == "REMOTE_TRAINING_SALVAGE_TIMEOUT":
            salvage_failure_code = exc.code
        else:
            salvage_failure_code = "REMOTE_TRAINING_SALVAGE_FAILED"
    finally:
        recovery_partial.unlink(missing_ok=True)
        checkpoint_partial.unlink(missing_ok=True)
        atomic_json(
            attempt_root / "salvage-receipt.json",
            {
                "schema": SALVAGE_RECEIPT_SCHEMA,
                "run_id": run_id,
                "attempted": True,
                "succeeded": succeeded,
                "artifact_count": artifact_count,
                "checkpoint_present": checkpoint_present,
                "checkpoint_sha256": checkpoint_sha256,
                "failure_code": failure_code,
                "dependency_failure_code": dependency_failure_code,
                "dependency_failed_module": dependency_failed_module,
                "salvage_failure_code": salvage_failure_code,
                "remote_process_exit_code": remote_exit_code,
                "remote_process_signal": remote_signal,
                "elapsed_seconds": time.monotonic() - started,
                "secret_values_included": False,
            },
        )
    return SalvageResult(
        attempted=True,
        succeeded=succeeded,
        artifact_count=artifact_count,
        checkpoint_present=checkpoint_present,
        checkpoint_sha256=checkpoint_sha256,
        failure_code=failure_code,
    )


def _operation(
    client: RunPodV1Client,
    lease: RunPodLease,
    *,
    bundle_root: Path,
    key: Path,
    known_hosts: Path,
    download_path: Path,
    run_id: str,
    started: float,
    operator_receipt: OperatorReceipt,
    operator_receipt_path: Path,
    training_dataset: Path | None = None,
    checkpoint_store: Path | None = None,
    resume_archive: Path | None = None,
    recovery_attempt_root: Path | None = None,
    readiness_sha256: str | None = None,
    sealed_assets_sha256: str | None = None,
    config_sha256: str | None = None,
    environment_contract_sha256: str,
    dependency_lock_sha256: str,
) -> dict[str, object]:
    bound_receipt = read_operator_receipt(operator_receipt_path)
    _require(
        bound_receipt.run_id == operator_receipt.run_id
        and bound_receipt.run_marker == operator_receipt.run_marker
        and bound_receipt.pod_id == lease.pod.pod_id
        and bound_receipt.stage == "running",
        "OPERATOR_RECEIPT_POD_BINDING_MISSING",
    )
    _emit("PHASE3F_POD_CREATED", run_id=run_id)
    _emit("PHASE3F_WAITING_FOR_SSH", run_id=run_id)
    connection = _wait_connection(
        client,
        lease,
        started,
        key=key,
        known_hosts=known_hosts,
    )
    _emit(
        "PHASE3F_SSH_READY",
        gpu=connection.gpu_display_name,
        hourly_price_usd=str(connection.hourly_price),
    )
    ssh = _ssh_arguments(key, known_hosts, connection)
    scp = _scp_arguments(key, known_hosts, connection)

    def remaining() -> float:
        return lease.policy.max_runtime_seconds - (time.monotonic() - started)

    remote_transfer = "/workspace/phase3f-transfer"
    _emit("PHASE3F_TRANSFER_STARTED", run_id=run_id)
    _require(
        bundle_root.joinpath("wheelhouse").is_dir()
        and bundle_root.joinpath(
            "wheelhouse", "framework-checksum-inventory.json"
        ).is_file(),
        "REMOTE_TORCHVISION_COMPANION_MISSING",
    )
    _run_command(
        [
            *ssh,
            "mkdir -p /workspace/phase3f-transfer/vendor "
            "/workspace/phase3f-transfer/wheelhouse",
        ],
        timeout_seconds=remaining(),
    )
    transfer_items = [
        (bundle_root / "atlaslens-phase3f-source.tar", f"{remote_transfer}/source.tar"),
        (bundle_root / "model.safetensors", f"{remote_transfer}/model.safetensors"),
        (bundle_root / "vendor" / "megaloc_model.py", f"{remote_transfer}/vendor/megaloc_model.py"),
        (bundle_root / "vendor" / "LICENSE", f"{remote_transfer}/vendor/LICENSE"),
    ]
    transfer_items.extend(
        (path, f"{remote_transfer}/wheelhouse/{path.name}")
        for path in sorted(
            (bundle_root / "wheelhouse").iterdir(), key=lambda item: item.name
        )
        if path.is_file() and not path.is_symlink()
    )
    if training_dataset is not None:
        transfer_items.append((training_dataset, f"{remote_transfer}/sealed-acquisition.tar"))
    if resume_archive is not None:
        transfer_items.append((resume_archive, f"{remote_transfer}/resume-checkpoint.tar"))
    for local, remote in transfer_items:
        _run_command(
            [*scp, str(local), f"root@{connection.public_ip}:{remote}"],
            timeout_seconds=remaining(),
        )
        lease.assert_within_limits(elapsed_seconds=int(time.monotonic() - started))
    _emit("PHASE3F_TRANSFER_VERIFIED", run_id=run_id)
    remote_contract = "/workspace/phase3f-repo/config/phase3f-remote-environment.json"
    remote_lock = "/workspace/phase3f-repo/config/phase3f-training-requirements.lock"
    remote_venv = "/workspace/phase3f-venv"
    remote_python = f"{remote_venv}/bin/python"
    source_preparation = [
        "rm -rf /workspace/phase3f-repo /workspace/phase3f-output /workspace/phase3f-work "
        "/workspace/phase3f-dataset /workspace/phase3f-venv",
        "mkdir -p /workspace/phase3f-repo /workspace/phase3f-work/recovery",
        "tar -xf /workspace/phase3f-transfer/source.tar -C /workspace/phase3f-repo",
    ]
    bootstrap_command = (
        "unset MAPILLARY_ACCESS_TOKEN RUNPOD_API_KEY; "
        f"timeout --kill-after=10s --signal=TERM {BOOTSTRAP_BUDGET_SECONDS + 30} "
        "python /workspace/phase3f-repo/scripts/phase3f/remote_bootstrap.py "
        "--repository-root /workspace/phase3f-repo "
        f"--contract {remote_contract} "
        f"--requirements-lock {remote_lock} "
        f"--expected-contract-sha256 {environment_contract_sha256} "
        f"--expected-lock-sha256 {dependency_lock_sha256} "
        "--wheelhouse /workspace/phase3f-transfer/wheelhouse "
        "--wheelhouse-inventory "
        "/workspace/phase3f-transfer/wheelhouse/framework-checksum-inventory.json "
        "--vendor-root /workspace/phase3f-transfer/vendor "
        "--model /workspace/phase3f-transfer/model.safetensors "
        f"--venv-root {remote_venv} "
        "--recovery-root /workspace/phase3f-work/recovery "
        f"--timeout-seconds {BOOTSTRAP_BUDGET_SECONDS}"
    )
    bootstrap_remote_command = " && ".join((*source_preparation, bootstrap_command))
    training_preparation: list[str] = []
    if training_dataset is not None:
        training_preparation.extend(
            (
                "mkdir -p /workspace/phase3f-dataset",
                "tar -xf /workspace/phase3f-transfer/sealed-acquisition.tar -C /workspace/phase3f-dataset",
                "unset MAPILLARY_ACCESS_TOKEN RUNPOD_API_KEY",
            )
        )
        if resume_archive is not None:
            training_preparation.append(
                "mkdir -p /workspace/phase3f-work/training-checkpoint "
                "&& tar -xf /workspace/phase3f-transfer/resume-checkpoint.tar "
                "-C /workspace/phase3f-work/training-checkpoint"
            )
    prepared = " && ".join(training_preparation)

    def build_remote_command(deadline_epoch: int, training_seconds: int) -> str:
        job_command = (
            (
                "ATLASLENS_PHASE3F_CHILD=training_job.py "
                f"{remote_python} "
                "/workspace/phase3f-repo/scripts/phase3f/remote_training.py "
                "--repository-root /workspace/phase3f-repo "
                f"--run-id {run_id} "
                "--sealed-root /workspace/phase3f-dataset/sealed-acquisition "
                "--model /workspace/phase3f-transfer/model.safetensors "
                "--vendor-root /workspace/phase3f-transfer/vendor "
                "--work-root /workspace/phase3f-work "
                "--output-root /workspace/phase3f-output "
                "--recovery-root /workspace/phase3f-work/recovery "
                f"--environment-contract {remote_contract} "
                f"--requirements-lock {remote_lock} "
                f"--expected-contract-sha256 {environment_contract_sha256} "
                f"--expected-lock-sha256 {dependency_lock_sha256} "
                "--environment-receipt "
                "/workspace/phase3f-work/recovery/environment-receipt.json "
                f"--deadline-epoch {deadline_epoch} "
                f"--timeout-seconds {training_seconds}"
            )
            if training_dataset is not None
            else (
                "timeout --signal=TERM "
                f"{training_seconds} {remote_python} "
                "/workspace/phase3f-repo/scripts/phase3f/cloud_job.py "
                "--repository-root /workspace/phase3f-repo "
                f"--run-id {run_id} "
                "--model /workspace/phase3f-transfer/model.safetensors "
                "--vendor-root /workspace/phase3f-transfer/vendor "
                "--work-root /workspace/phase3f-work "
                "--output-root /workspace/phase3f-output "
                f"--deadline-epoch {deadline_epoch}"
            )
        )
        return (
            f"{prepared + ' && ' if prepared else ''}{job_command}; phase3f_rc=$?; "
            "if [ $phase3f_rc -eq 0 ]; then "
            "tar -C /workspace -cf /workspace/phase3f-transfer/output.tar "
            "phase3f-output; fi; exit $phase3f_rc"
        )
    checkpoint_enabled = all(
        value is not None
        for value in (
            checkpoint_store,
            recovery_attempt_root,
            readiness_sha256,
            sealed_assets_sha256,
            config_sha256,
        )
    )

    def salvage_failure(process_result: RemoteProcessResult) -> SalvageResult:
        return (
            _salvage_remote_failure(
                ssh,
                scp,
                public_ip=connection.public_ip,
                remote_transfer=remote_transfer,
                attempt_root=cast(Path, recovery_attempt_root),
                checkpoint_store=cast(Path, checkpoint_store),
                run_id=run_id,
                readiness_sha256=cast(str, readiness_sha256),
                sealed_assets_sha256=cast(str, sealed_assets_sha256),
                config_sha256=cast(str, config_sha256),
                process_result=process_result,
                timeout_seconds=min(SALVAGE_RESERVE_SECONDS, max(1.0, remaining())),
            )
            if checkpoint_enabled
            else SalvageResult(
                True, False, 0, False, None, "REMOTE_TRAINING_UNKNOWN_FAILURE"
            )
        )

    try:
        _emit(
            "PHASE3F_REMOTE_BOOTSTRAP_STARTED",
            run_id=run_id,
            timeout_seconds=BOOTSTRAP_BUDGET_SECONDS,
            cloud_mutations=0,
        )
        bootstrap_result = _run_watched(
            [*ssh, "bash", "-lc", shlex.quote(bootstrap_remote_command)],
            lease=lease,
            started=started,
            status_event="PHASE3F_REMOTE_BOOTSTRAP_RUNNING",
            periodic_seconds=60,
        )
        if bootstrap_result is None:
            bootstrap_result = RemoteProcessResult(0, 0.0)
        if bootstrap_result.return_code != 0:
            raise SupervisorExecutionError(
                salvage_failure(bootstrap_result).failure_code
            )
        _emit(
            "PHASE3F_REMOTE_BASE_ENVIRONMENT_REPORTED",
            run_id=run_id,
            secrets_included=False,
        )
        _emit(
            "PHASE3F_REMOTE_LIGHTWEIGHT_DEPENDENCIES_READY",
            run_id=run_id,
            torch_downloaded=False,
            torchvision_companion_source="transferred_hash_pinned_wheel",
            package_index_resolution_on_pod=False,
            torch_identity_unchanged=True,
        )
        _emit(
            "PHASE3F_REMOTE_TRAINING_SMOKE_PASSED",
            run_id=run_id,
            holdout_open_count=0,
            checkpoint_write_count=0,
        )
        try:
            child_deadline = compute_child_deadline(
                total_runtime_seconds=lease.policy.max_runtime_seconds,
                started_monotonic=started,
                now_monotonic=time.monotonic(),
                now_epoch=time.time(),
            )
        except TrainingDeadlineError as exc:
            raise SupervisorExecutionError(exc.code) from exc
        remote_command = build_remote_command(
            child_deadline.deadline_epoch,
            child_deadline.training_budget_seconds,
        )
        _emit(
            "PHASE3F_TRAINING_DEADLINE_ATTESTED",
            elapsed_monotonic_seconds=child_deadline.elapsed_monotonic_seconds,
            remaining_total_seconds=child_deadline.remaining_total_seconds,
            training_budget_seconds=child_deadline.training_budget_seconds,
            deadline_class="future_epoch_seconds",
        )
        _emit(
            "PHASE3F_REMOTE_DEPENDENCIES_READY",
            run_id=run_id,
            environment_contract_sha256=environment_contract_sha256,
            dependency_lock_sha256=dependency_lock_sha256,
        )
        _emit("PHASE3F_CLOUD_JOB_STARTED", run_id=run_id)
        partial = (
            cast(Path, recovery_attempt_root).parent / ".checkpoint-sync.partial"
            if checkpoint_enabled
            else download_path.with_name(".checkpoint-sync.partial")
        )
        watched_arguments = [*ssh, "bash", "-lc", shlex.quote(remote_command)]
        result = (
            _run_watched(
                watched_arguments,
                lease=lease,
                started=started,
                periodic=lambda: _sync_remote_checkpoint(
                    ssh,
                    scp,
                    public_ip=connection.public_ip,
                    remote_transfer=remote_transfer,
                    checkpoint_store=cast(Path, checkpoint_store),
                    partial_path=partial,
                    run_id=run_id,
                    readiness_sha256=cast(str, readiness_sha256),
                    sealed_assets_sha256=cast(str, sealed_assets_sha256),
                    config_sha256=cast(str, config_sha256),
                    timeout_seconds=min(120.0, max(1.0, remaining())),
                ),
            )
            if checkpoint_enabled
            else _run_watched(watched_arguments, lease=lease, started=started)
        )
        if result is None:  # Backward-compatible test seam for the command runner.
            result = RemoteProcessResult(0, 0.0)
        if result.return_code != 0:
            raise SupervisorExecutionError(salvage_failure(result).failure_code)
        _run_command(
            [
                *scp,
                f"root@{connection.public_ip}:{remote_transfer}/output.tar",
                str(download_path),
            ],
            timeout_seconds=remaining(),
        )
        _emit("PHASE3F_ARTIFACT_DOWNLOADED", run_id=run_id)
    finally:
        cleanup = (
            "rm -rf /workspace/phase3f-work /workspace/phase3f-output /workspace/phase3f-dataset "
            "/workspace/phase3f-repo /workspace/phase3f-transfer /workspace/phase3f-venv"
        )
        try:
            _run_command([*ssh, cleanup], timeout_seconds=min(60.0, max(1.0, remaining())))
        except SupervisorExecutionError:
            pass
    return {
        "pod_id_sha256": hashlib.sha256(lease.pod.pod_id.encode()).hexdigest(),
        "gpu": connection.gpu_display_name,
        "hourly_price_usd": str(connection.hourly_price),
        "training_budget_seconds": child_deadline.training_budget_seconds,
        "deadline_class": "future_epoch_seconds",
    }


def _post_inventory(
    client: RunPodV1Client,
    before: RunPodInventory,
) -> RunPodInventory:
    after = client.inventory()
    for _ in range(20):
        if after == before:
            return after
        time.sleep(3)
        after = client.inventory()
    return after


def _write_receipt(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _budget_decimal(value: object, code: str) -> Decimal:
    _require(not isinstance(value, bool), code)
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise SupervisorExecutionError(code) from exc
    _require(amount.is_finite() and amount >= 0, code)
    return amount


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _receipt_key(
    run_id: str,
    pod_id_sha256: str | None,
    attempt_id: str | None = None,
) -> str:
    if attempt_id is not None:
        _require(bool(_RUN_ID.fullmatch(attempt_id)), "HISTORICAL_RECEIPT_INVALID")
        return f"attempt:{run_id}:{attempt_id}"
    if pod_id_sha256 is not None:
        _require(bool(_SHA256.fullmatch(pod_id_sha256)), "HISTORICAL_RECEIPT_INVALID")
        return f"pod:{pod_id_sha256}"
    return f"run:{run_id}"


def _operator_receipt_key(receipt: OperatorReceipt) -> str:
    return operator_budget_linkage(receipt)


def _receipt_paths(runtime_root: Path) -> tuple[list[Path], list[Path]]:
    supervisor_paths: list[Path] = []
    receipt_root = runtime_root / "_receipts"
    if receipt_root.exists():
        _require(
            receipt_root.is_dir() and not receipt_root.is_symlink(),
            "HISTORICAL_RECEIPTS_INVALID",
        )
        supervisor_paths = sorted(receipt_root.glob("*.json"))
    operator_paths: list[Path] = []
    operator_root = runtime_root / "_operator"
    current = operator_root / "phase3f-current.json"
    if current.exists():
        operator_paths.append(current)
    archive = operator_root / "archive"
    if archive.exists():
        _require(
            archive.is_dir() and not archive.is_symlink(),
            "HISTORICAL_RECEIPTS_INVALID",
        )
        operator_paths.extend(sorted(archive.glob("*.json")))
    _require(
        len(supervisor_paths) + len(operator_paths) <= MAX_HISTORICAL_RECEIPTS,
        "HISTORICAL_RECEIPT_CAP_EXCEEDED",
    )
    for path in (*supervisor_paths, *operator_paths):
        _require(
            path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024 * 1024,
            "HISTORICAL_RECEIPT_INVALID",
        )
    return supervisor_paths, operator_paths


def _receipt_datetime(value: str | None) -> datetime:
    if value is None:
        raise SupervisorExecutionError("HISTORICAL_RECEIPT_TIME_INVALID")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SupervisorExecutionError("HISTORICAL_RECEIPT_TIME_INVALID") from exc
    _require(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        "HISTORICAL_RECEIPT_TIME_INVALID",
    )
    return parsed.astimezone(UTC)


def _closed_operator_estimate(receipt: OperatorReceipt) -> Decimal:
    if receipt.pod_id is None:
        return Decimal("0")
    started = _receipt_datetime(receipt.pod_bound_at or receipt.started_at)
    finished = _receipt_datetime(receipt.finished_at)
    duration_seconds = Decimal(str((finished - started).total_seconds()))
    _require(
        Decimal("0") <= duration_seconds <= Decimal(receipt.max_wall_minutes * 60),
        "HISTORICAL_RECEIPT_TIME_INVALID",
    )
    hourly_candidates: list[Decimal] = []
    if receipt.selected_uninterruptable_price is not None:
        hourly_candidates.append(receipt.selected_uninterruptable_price)
    if receipt.create_cost_per_hr is not None:
        hourly_candidates.append(receipt.create_cost_per_hr)
    hourly = max(hourly_candidates) if hourly_candidates else receipt.max_gpu_hourly_usd
    compute = hourly * duration_seconds / Decimal(3600)
    return min(
        receipt.max_spend_usd,
        compute + E2E_CLOSED_POD_DISK_ALLOWANCE_USD,
    )


def _latest_prior_budget_snapshot(
    runtime_root: Path,
) -> tuple[datetime, Decimal, Decimal] | None:
    root = runtime_root / "_budget" / "reconciliations"
    if not root.exists():
        return None
    _require(root.is_dir() and not root.is_symlink(), "HISTORICAL_BUDGET_RECEIPT_INVALID")
    candidates: list[tuple[datetime, Decimal, Decimal]] = []
    for path in sorted(root.glob("*.json")):
        _require(
            path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024 * 1024,
            "HISTORICAL_BUDGET_RECEIPT_INVALID",
        )
        try:
            row = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SupervisorExecutionError("HISTORICAL_BUDGET_RECEIPT_INVALID") from exc
        _require(
            isinstance(row, dict)
            and row.get("schema") == _BUDGET_RECONCILIATION_SCHEMA,
            "HISTORICAL_BUDGET_RECEIPT_INVALID",
        )
        reconciled_at = row.get("reconciled_at")
        _require(
            isinstance(reconciled_at, str),
            "HISTORICAL_BUDGET_RECEIPT_INVALID",
        )
        timestamp = _receipt_datetime(cast(str, reconciled_at))
        actual = _budget_decimal(
            row.get("actual_billed_usd"), "HISTORICAL_BUDGET_RECEIPT_INVALID"
        )
        unbilled = _budget_decimal(
            row.get("conservative_unbilled_estimate_usd"),
            "HISTORICAL_BUDGET_RECEIPT_INVALID",
        )
        candidates.append((timestamp, actual, unbilled))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def _post_snapshot_closed_estimate(
    runtime_root: Path,
    *,
    reconciled_at: datetime,
) -> tuple[Decimal, int]:
    _supervisor_paths, operator_paths = _receipt_paths(runtime_root)
    estimates: dict[str, Decimal] = {}
    for path in operator_paths:
        receipt = read_operator_receipt(path)
        if not (
            receipt.cleanup_verified
            and receipt.stage in {"failed", "terminated"}
            and receipt.pod_id is not None
            and _receipt_datetime(receipt.finished_at) > reconciled_at
        ):
            continue
        key = _operator_receipt_key(receipt)
        estimates[key] = max(
            estimates.get(key, Decimal("0")),
            _closed_operator_estimate(receipt),
        )
    return sum(estimates.values(), Decimal("0")), len(estimates)


def _historical_budget_evidence(
    runtime_root: Path,
) -> tuple[Decimal, Decimal, Decimal, Decimal, str, int, int]:
    supervisor_paths, operator_paths = _receipt_paths(runtime_root)
    local_estimate_by_key: dict[str, Decimal] = {}
    actual_by_key: dict[str, Decimal] = {}
    active_by_key: dict[str, Decimal] = {}
    closed_reservation_by_key: dict[str, Decimal] = {}
    source_hashes: list[str] = []
    duplicate_billing_records = 0
    seen_budget_keys: set[str] = set()

    for path in operator_paths:
        source_hashes.append(_sha256_path(path))
        receipt = read_operator_receipt(path)
        key = _operator_receipt_key(receipt)
        if key in seen_budget_keys:
            duplicate_billing_records += 1
        seen_budget_keys.add(key)
        closed = receipt.cleanup_verified and receipt.stage in {"failed", "terminated"}
        if closed:
            if receipt.pod_id is not None:
                closed_reservation_by_key[key] = max(
                    closed_reservation_by_key.get(key, Decimal("0")),
                    receipt.max_spend_usd,
                )
                local_estimate_by_key[key] = max(
                    local_estimate_by_key.get(key, Decimal("0")),
                    _closed_operator_estimate(receipt),
                )
        else:
            active_by_key[key] = max(
                active_by_key.get(key, Decimal("0")),
                receipt.max_spend_usd,
            )

    for path in supervisor_paths:
        source_hashes.append(_sha256_path(path))
        try:
            row = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SupervisorExecutionError("HISTORICAL_RECEIPT_INVALID") from exc
        _require(isinstance(row, dict), "HISTORICAL_RECEIPT_INVALID")
        run_id = row.get("run_id")
        pod_id_sha256 = row.get("pod_id_sha256")
        attempt_id = row.get("attempt_id")
        _require(
            row.get("schema") == _SUPERVISOR_RECEIPT_SCHEMA
            and isinstance(run_id, str)
            and bool(_RUN_ID.fullmatch(run_id))
            and (pod_id_sha256 is None or isinstance(pod_id_sha256, str))
            and (attempt_id is None or isinstance(attempt_id, str)),
            "HISTORICAL_RECEIPT_INVALID",
        )
        key = _receipt_key(run_id, pod_id_sha256, attempt_id)
        if key in seen_budget_keys:
            duplicate_billing_records += 1
        seen_budget_keys.add(key)
        conservative = _budget_decimal(
            row.get("conservative_incremental_upper_usd"),
            "HISTORICAL_RECEIPT_INVALID",
        )
        local_estimate_by_key[key] = max(
            local_estimate_by_key.get(key, Decimal("0")),
            conservative,
        )
        if row.get("actual_spend_available") is True:
            actual = _budget_decimal(
                row.get("actual_spend_usd"),
                "HISTORICAL_RECEIPT_INVALID",
            )
            actual_by_key[key] = max(actual_by_key.get(key, Decimal("0")), actual)
        else:
            _require(
                row.get("actual_spend_usd") is None,
                "HISTORICAL_RECEIPT_INVALID",
            )

    return (
        sum(actual_by_key.values(), Decimal("0")),
        sum(local_estimate_by_key.values(), Decimal("0")),
        sum(active_by_key.values(), Decimal("0")),
        sum(closed_reservation_by_key.values(), Decimal("0")),
        _canonical_sha256(sorted(source_hashes)),
        len(source_hashes),
        duplicate_billing_records,
    )


def _budget_reconciliation(
    policy: BudgetPolicy,
    runtime_root: Path,
    billing: RunPodBillingSnapshot,
) -> BudgetReconciliation:
    _require(
        policy.absolute_usd <= E2E_RUN_BUDGET_USD,
        "TRAINING_RUN_BUDGET_EXCEEDS_3_USD",
    )
    (
        local_actual,
        local_conservative,
        active_exposure,
        closed_reservations,
        source_receipts_sha256,
        source_receipt_count,
        duplicate_billing_records,
    ) = _historical_budget_evidence(runtime_root)
    provider_balance_delta = max(
        Decimal("0"),
        E2E_HISTORICAL_BUDGET_USD - billing.client_balance_usd,
    )
    unrecognized_active_billing = billing.current_spend_per_hour_usd != 0
    if unrecognized_active_billing:
        active_exposure += E2E_HISTORICAL_BUDGET_USD
    actual_billed = max(local_actual, provider_balance_delta)
    conservative_unbilled = max(Decimal("0"), local_conservative - actual_billed)
    prior_snapshot = _latest_prior_budget_snapshot(runtime_root)
    billing_lag_estimate = Decimal("0")
    billing_lag_attempt_count = 0
    provider_increment = Decimal("0")
    if prior_snapshot is not None:
        prior_at, prior_actual, prior_unbilled = prior_snapshot
        billing_lag_estimate, billing_lag_attempt_count = _post_snapshot_closed_estimate(
            runtime_root,
            reconciled_at=prior_at,
        )
        provider_increment = max(Decimal("0"), actual_billed - prior_actual)
        lag_unbilled = max(
            Decimal("0"),
            prior_unbilled + billing_lag_estimate - provider_increment,
        )
        conservative_unbilled = max(conservative_unbilled, lag_unbilled)
    historical_and_active = actual_billed + conservative_unbilled + active_exposure
    remaining = max(Decimal("0"), E2E_HISTORICAL_BUDGET_USD - historical_and_active)
    projected = historical_and_active + policy.absolute_usd
    closed_released = max(
        Decimal("0"),
        closed_reservations - max(actual_billed, local_conservative),
    )
    provider_snapshot_sha256 = _canonical_sha256(
        {
            "client_balance_usd": str(billing.client_balance_usd),
            "current_spend_per_hour_usd": str(billing.current_spend_per_hour_usd),
        }
    )
    return BudgetReconciliation(
        actual_billed_usd=actual_billed,
        conservative_unbilled_estimate_usd=conservative_unbilled,
        active_exposure_usd=active_exposure,
        closed_unused_reservation_released_usd=closed_released,
        proposed_run_max_usd=policy.absolute_usd,
        historical_total_cap_usd=E2E_HISTORICAL_BUDGET_USD,
        remaining_authorized_usd=remaining,
        projected_total_usd=projected,
        legacy_closed_reservation_total_usd=closed_reservations,
        local_conservative_historical_usd=local_conservative,
        provider_balance_delta_usd=provider_balance_delta,
        source_receipts_sha256=source_receipts_sha256,
        provider_snapshot_sha256=provider_snapshot_sha256,
        source_receipt_count=source_receipt_count,
        duplicate_billing_record_count=duplicate_billing_records,
        unrecognized_active_billing=unrecognized_active_billing,
        billing_lag_attempt_count=billing_lag_attempt_count,
        billing_lag_local_estimate_usd=billing_lag_estimate,
        provider_increment_since_prior_snapshot_usd=provider_increment,
    )


def _write_budget_reconciliation(
    runtime_root: Path,
    *,
    run_id: str,
    reconciliation: BudgetReconciliation,
    api_request_count: int,
    cloud_mutation_count: int,
) -> Path:
    _require(cloud_mutation_count == 0, "BUDGET_RECONCILIATION_AFTER_MUTATION")
    reconciled_at = datetime.now(UTC).isoformat()
    totals = {
        "actual_billed_usd": str(reconciliation.actual_billed_usd),
        "conservative_unbilled_estimate_usd": str(
            reconciliation.conservative_unbilled_estimate_usd
        ),
        "active_exposure_usd": str(reconciliation.active_exposure_usd),
        "closed_unused_reservation_released_usd": str(
            reconciliation.closed_unused_reservation_released_usd
        ),
        "proposed_run_max_usd": str(reconciliation.proposed_run_max_usd),
        "historical_total_cap_usd": str(reconciliation.historical_total_cap_usd),
        "remaining_authorized_usd": str(reconciliation.remaining_authorized_usd),
        "projected_total_usd": str(reconciliation.projected_total_usd),
        "legacy_closed_reservation_total_usd": str(
            reconciliation.legacy_closed_reservation_total_usd
        ),
        "local_conservative_historical_usd": str(
            reconciliation.local_conservative_historical_usd
        ),
        "billing_lag_attempt_count": reconciliation.billing_lag_attempt_count,
        "billing_lag_local_estimate_usd": str(
            reconciliation.billing_lag_local_estimate_usd
        ),
        "provider_increment_since_prior_snapshot_usd": str(
            reconciliation.provider_increment_since_prior_snapshot_usd
        ),
    }
    receipt_base: dict[str, object] = {
        "schema": _BUDGET_RECONCILIATION_SCHEMA,
        "reconciled_at": reconciled_at,
        "run_id_sha256": hashlib.sha256(run_id.encode("utf-8")).hexdigest(),
        **totals,
        "source_receipts_sha256": reconciliation.source_receipts_sha256,
        "provider_snapshot_sha256": reconciliation.provider_snapshot_sha256,
        "source_receipt_count": reconciliation.source_receipt_count,
        "duplicate_billing_record_count": reconciliation.duplicate_billing_record_count,
        "unrecognized_active_billing": reconciliation.unrecognized_active_billing,
        "inventory": {"pods": 0, "endpoints": 0, "network_volumes": 0, "templates": 0},
        "runpod_api_request_count": api_request_count,
        "cloud_mutations": cloud_mutation_count,
        "provider_response_body_included": False,
        "secret_values_included": False,
    }
    receipt_sha256 = _canonical_sha256(receipt_base)
    receipt_id = receipt_sha256[:32]
    document = {
        **receipt_base,
        "receipt_id": receipt_id,
        "receipt_sha256": receipt_sha256,
    }
    path = runtime_root / "_budget" / "reconciliations" / f"{receipt_id}.json"
    _write_receipt(path, document)
    return path


def _require_e2e_budget(
    policy: BudgetPolicy,
    runtime_root: Path,
    *,
    run_id: str,
    billing: RunPodBillingSnapshot,
    api_request_count: int,
    cloud_mutation_count: int,
) -> tuple[BudgetReconciliation, Path]:
    reconciliation = _budget_reconciliation(policy, runtime_root, billing)
    receipt_path = _write_budget_reconciliation(
        runtime_root,
        run_id=run_id,
        reconciliation=reconciliation,
        api_request_count=api_request_count,
        cloud_mutation_count=cloud_mutation_count,
    )
    _require(
        not reconciliation.unrecognized_active_billing,
        "UNRECOGNIZED_ACTIVE_BILLING_EXPOSURE",
    )
    _require(
        reconciliation.actual_billed_usd <= E2E_HISTORICAL_BUDGET_USD,
        "HISTORICAL_BUDGET_ALREADY_EXCEEDED",
    )
    _require(
        reconciliation.projected_total_usd <= E2E_HISTORICAL_BUDGET_USD,
        "BUDGET_INSUFFICIENT",
    )
    return reconciliation, receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-supervisor")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--live-readiness", action="store_true")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--terminate", action="store_true")
    actions.add_argument("--deadline-plan", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=Path(r"C:\AtlasLensRuntime\phase3f"))
    parser.add_argument("--operator-receipt", type=Path)
    parser.add_argument("--max-spend-usd", type=Decimal, default=Decimal("10"))
    parser.add_argument("--soft-stop-usd", type=Decimal, default=Decimal("7.5"))
    parser.add_argument("--hard-stop-usd", type=Decimal, default=Decimal("9"))
    parser.add_argument("--max-gpu-hourly-usd", type=Decimal, default=Decimal("0.50"))
    parser.add_argument("--max-wall-minutes", type=int, default=345)
    parser.add_argument("--sealed-acquisition", type=Path)
    parser.add_argument("--dataset-run-id")
    return parser


def _policy(args: argparse.Namespace) -> BudgetPolicy:
    try:
        deadline = require_training_deadline_plan(args.max_wall_minutes)
    except TrainingDeadlineError as exc:
        raise SupervisorExecutionError(exc.code) from exc
    return BudgetPolicy(
        target_usd=min(TARGET_BUDGET_USD, args.soft_stop_usd - Decimal("0.01")),
        soft_stop_usd=args.soft_stop_usd,
        terminate_usd=args.hard_stop_usd,
        absolute_usd=args.max_spend_usd,
        max_hourly_cost_usd=args.max_gpu_hourly_usd,
        max_runtime_seconds=deadline.requested_total_seconds,
    )


def _payload_contract_public(
    config: RunPodConfig,
    *,
    gpu_type_id: str,
    cloud_type: str,
) -> dict[str, object]:
    preview_request = PodRequest(
        run_marker="atlaslens-phase3f-contract-preview",
        idempotency_key="atlaslens-phase3f-contract-preview",
        hourly_cost_usd=Decimal("0.50"),
        max_runtime_seconds=30 * 60,
        gpu_type_id=gpu_type_id,
        public_ports=(22,),
    )
    preview_offer = GPUOffer(
        gpu_type_id=gpu_type_id,
        display_name=gpu_type_id,
        memory_gb=16,
        hourly_price=Decimal("0.50"),
        stock_status="Low",
        cloud_type=cloud_type,
        secure_cloud=cloud_type == "SECURE",
        community_cloud=cloud_type == "COMMUNITY",
        available_gpu_counts=None,
    )
    _payload, report = build_create_payload(config, preview_request, preview_offer)
    return report.to_public_dict()


def _operator_receipt_path(args: argparse.Namespace) -> Path:
    candidate = args.operator_receipt
    if candidate is None:
        candidate = args.runtime_root / "_operator" / "phase3f-current.json"
    return _require_safe_runtime_path(candidate)


def _run_control(
    args: argparse.Namespace,
    *,
    token: str,
    receipt_path: Path,
) -> int:
    receipt = read_operator_receipt(receipt_path)
    config = RunPodConfig(
        image_name=IMAGE,
        gpu_type_preferences=GPU_PREFERENCES,
        container_disk_gb=40,
        min_gpu_memory_gb=16,
    )
    with RunPodV1Client(api_token=token, config=config) as client:
        if args.status:
            status = inspect_operator_inventory(receipt, client.inventory())
            action = "status"
        else:
            status = terminate_receipt_bound_pod(client, receipt)
            receipt = receipt.update_lifecycle(
                stage="terminated",
                finished_at=datetime.now(UTC).isoformat(),
                cleanup_verified=status.cleanup_verified,
            )
            write_operator_receipt(receipt_path, receipt)
            action = "terminate"
    print(
        json.dumps(
            status.to_public_dict(receipt, action=action),
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _run_dry_run(
    args: argparse.Namespace,
    *,
    receipt_path: Path,
) -> int:
    policy = _policy(args)
    _disk_gate()
    head, _tracked = _preflight_repository()
    _verify_local_readiness()
    contract_sha256, lock_sha256, wheelhouse = _remote_environment_preflight()
    require_startable_receipt(receipt_path)
    config = RunPodConfig(
        image_name=IMAGE,
        gpu_type_preferences=GPU_PREFERENCES,
        container_disk_gb=40,
        min_gpu_memory_gb=16,
    )
    contract = _payload_contract_public(
        config,
        gpu_type_id=GPU_PREFERENCES[0],
        cloud_type="SECURE",
    )
    print(
        json.dumps(
            {
                "action": "dry-run",
                "branch": REQUIRED_BRANCH,
                "head": head,
                "max_spend_usd": str(policy.absolute_usd),
                "soft_stop_usd": str(policy.soft_stop_usd),
                "hard_stop_usd": str(policy.terminate_usd),
                "max_gpu_hourly_usd": str(policy.max_hourly_cost_usd),
                "max_wall_minutes": policy.max_runtime_seconds // 60,
                "model_sha256": MODEL_SHA256,
                "vendor_canonical_lf_sha256": SOURCE_LF_SHA256,
                "environment_contract_sha256": contract_sha256,
                "dependency_lock_sha256": lock_sha256,
                "framework_wheelhouse_inventory_sha256": (
                    wheelhouse.inventory_sha256
                ),
                "torchvision_companion_sha256": wheelhouse.companion.sha256,
                "package_index_resolution_on_pod": False,
                "runpod_api_calls": 0,
                "cloud_mutations": 0,
                "create_attempts": 0,
                **contract,
                "ready_for_execute": True,
                "secret_values_included": False,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _run_live_readiness(
    args: argparse.Namespace,
    *,
    token: str,
    receipt_path: Path,
) -> int:
    policy = _policy(args)
    _disk_gate()
    _head, _tracked = _preflight_repository()
    _verify_local_readiness()
    require_startable_receipt(receipt_path)
    config = RunPodConfig(
        image_name=IMAGE,
        gpu_type_preferences=GPU_PREFERENCES,
        container_disk_gb=40,
        min_gpu_memory_gb=16,
    )
    with RunPodV1Client(api_token=token, config=config) as client:
        inventory = client.inventory()
        require_empty_inventory(inventory)
        try:
            report = client.check_gpu_availability(
                max_hourly_price=policy.max_hourly_cost_usd,
            )
        except RunPodAPIError as exc:
            print(
                json.dumps(
                    {
                        "action": "live-readiness",
                        "ready_for_execute": False,
                        "blocker_code": exc.code,
                        "graphql_errors": [
                            item.to_public_dict() for item in client.last_graphql_errors
                        ],
                        "runpod_api_calls": client.api_request_count,
                        "cloud_mutations": client.cloud_mutation_count,
                        "pods": len(inventory.pods),
                        "endpoints": len(inventory.endpoint_ids),
                        "network_volumes": len(inventory.network_volume_ids),
                        "templates": len(inventory.template_ids),
                        "secret_values_included": False,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
            return 1
        ready = report.selected_offer is not None
        contract: dict[str, object] = {}
        if report.selected_offer is not None:
            contract = _payload_contract_public(
                config,
                gpu_type_id=report.selected_offer.gpu_type_id,
                cloud_type=report.selected_offer.cloud_type,
            )
        print(
            json.dumps(
                {
                    "action": "live-readiness",
                    **report.to_public_dict(),
                    "ready_for_execute": ready,
                    "blocker_code": None if ready else "NO_ELIGIBLE_GPU_OFFER",
                    "graphql_errors": [],
                    "runpod_api_calls": client.api_request_count,
                    "cloud_mutations": client.cloud_mutation_count,
                    "create_attempts": 0,
                    **contract,
                    "pods": len(inventory.pods),
                    "endpoints": len(inventory.endpoint_ids),
                    "network_volumes": len(inventory.network_volume_ids),
                    "templates": len(inventory.template_ids),
                    "secret_values_included": False,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if ready else 1


def _record_failed_operator_state(
    receipt_path: Path,
    *,
    cleanup_verified: bool,
    no_pod_capacity_race: bool = False,
) -> None:
    try:
        receipt = read_operator_receipt(receipt_path)
        if receipt.stage == "terminated":
            return
        write_operator_receipt(
            receipt_path,
            receipt.update_lifecycle(
                stage=(
                    "terminated"
                    if no_pod_capacity_race and cleanup_verified
                    else "failed"
                ),
                finished_at=datetime.now(UTC).isoformat(),
                cleanup_verified=cleanup_verified,
            ),
        )
    except (OSError, Phase3FOperatorError):
        pass


def _bind_operator_pod(
    receipt_path: Path,
    *,
    expected_run_id: str,
    expected_attempt_id: str,
    pod: PodRecord,
) -> None:
    receipt = read_operator_receipt(receipt_path)
    _require(
        hmac.compare_digest(receipt.run_id, expected_run_id),
        "OPERATOR_RECEIPT_RUN_ID_MISMATCH",
    )
    _require(
        receipt.attempt_id is not None
        and hmac.compare_digest(receipt.attempt_id, expected_attempt_id),
        "OPERATOR_RECEIPT_ATTEMPT_ID_MISMATCH",
    )
    _require(
        pod.run_marker is not None
        and hmac.compare_digest(receipt.run_marker, pod.run_marker),
        "OPERATOR_RECEIPT_POD_MARKER_MISMATCH",
    )
    _require(
        receipt.pod_id is None or hmac.compare_digest(receipt.pod_id, pod.pod_id),
        "OPERATOR_RECEIPT_POD_ID_CONFLICT",
    )
    _require(receipt.stage in {"preflight", "running"}, "OPERATOR_RECEIPT_STAGE_INVALID")
    if receipt.pod_id is None or receipt.stage != "running":
        write_operator_receipt(
            receipt_path,
            receipt.update_lifecycle(
                stage="running",
                pod_id=pod.pod_id,
                pod_bound_at=datetime.now(UTC).isoformat(),
            ),
        )
    confirmed = read_operator_receipt(receipt_path)
    _require(
        confirmed.stage == "running"
        and confirmed.pod_id is not None
        and confirmed.attempt_id is not None
        and pod.run_marker is not None
        and hmac.compare_digest(confirmed.attempt_id, expected_attempt_id)
        and hmac.compare_digest(confirmed.pod_id, pod.pod_id)
        and hmac.compare_digest(confirmed.run_marker, pod.run_marker),
        "OPERATOR_RECEIPT_POD_BINDING_MISSING",
    )


def _record_operator_rental_attestation(
    receipt_path: Path,
    *,
    expected_run_id: str,
    expected_attempt_id: str,
    attestation: PodRentalAttestationDiagnostic,
) -> None:
    receipt = read_operator_receipt(receipt_path)
    _require(receipt.run_id == expected_run_id, "OPERATOR_RECEIPT_RUN_ID_MISMATCH")
    _require(
        receipt.attempt_id is not None
        and hmac.compare_digest(receipt.attempt_id, expected_attempt_id),
        "OPERATOR_RECEIPT_ATTEMPT_ID_MISMATCH",
    )
    _require(receipt.stage == "running", "OPERATOR_RECEIPT_STAGE_INVALID")
    _require(receipt.pod_id is not None, "OPERATOR_RECEIPT_POD_ID_MISSING")
    allocation_attested_at = datetime.now(UTC).isoformat()
    write_operator_receipt(
        receipt_path,
        receipt.record_rental_attestation(
            attestation,
            allocation_attested_at=allocation_attested_at,
        ),
    )
    _emit(
        "PHASE3F_POD_RENTAL_EVIDENCE",
        allocation_attested_at=allocation_attested_at,
        **attestation.to_public_dict(),
    )


def _record_operator_gpu_attestation_progress(
    receipt_path: Path,
    *,
    expected_run_id: str,
    expected_attempt_id: str,
    diagnostic: PodGPUAttestationProgressDiagnostic,
) -> None:
    receipt = read_operator_receipt(receipt_path)
    _require(receipt.run_id == expected_run_id, "OPERATOR_RECEIPT_RUN_ID_MISMATCH")
    _require(
        receipt.attempt_id is not None
        and hmac.compare_digest(receipt.attempt_id, expected_attempt_id),
        "OPERATOR_RECEIPT_ATTEMPT_ID_MISMATCH",
    )
    _require(receipt.stage == "running", "OPERATOR_RECEIPT_STAGE_INVALID")
    _require(receipt.pod_id is not None, "OPERATOR_RECEIPT_POD_ID_MISSING")
    write_operator_receipt(
        receipt_path,
        receipt.record_gpu_attestation_progress(diagnostic),
    )
    _emit("PHASE3F_POD_GPU_ATTESTATION", **diagnostic.to_public_dict())


def _record_operator_connectivity_progress(
    receipt_path: Path,
    *,
    expected_run_id: str,
    expected_attempt_id: str,
    diagnostic: PodConnectivityProgressDiagnostic,
) -> None:
    receipt = read_operator_receipt(receipt_path)
    _require(receipt.run_id == expected_run_id, "OPERATOR_RECEIPT_RUN_ID_MISMATCH")
    _require(
        receipt.attempt_id is not None
        and hmac.compare_digest(receipt.attempt_id, expected_attempt_id),
        "OPERATOR_RECEIPT_ATTEMPT_ID_MISMATCH",
    )
    _require(receipt.stage == "running", "OPERATOR_RECEIPT_STAGE_INVALID")
    _require(receipt.pod_id is not None, "OPERATOR_RECEIPT_POD_ID_MISSING")
    write_operator_receipt(
        receipt_path,
        receipt.record_connectivity_progress(diagnostic),
    )
    _emit("PHASE3F_POD_CONNECTIVITY", **diagnostic.to_public_dict())


def _run_execute(
    args: argparse.Namespace,
    *,
    token: str,
    receipt_path: Path,
) -> int:
    policy = _policy(args)
    run_id = args.dataset_run_id or os.urandom(16).hex()
    _require(bool(_RUN_ID.fullmatch(run_id)), "RUN_ID_INVALID")
    key_root = Path(r"C:\tmp")
    key = key_root / f"atlaslens-phase3f-{run_id}"
    known_hosts = key_root / f"atlaslens-phase3f-{run_id}.known-hosts"
    local_root = ROOT / ".local" / "phase3f" / "runs" / run_id
    transfer_root = local_root / "transfer"
    download_path = local_root / "output.tar"
    extracted_parent = local_root / "extracted"
    started_at = datetime.now(UTC)
    started = time.monotonic()
    attempt_id = os.urandom(16).hex()
    operator_receipt = OperatorReceipt(
        run_id=run_id,
        attempt_id=attempt_id,
        run_marker=f"atlaslens-phase3f-{run_id}",
        pod_id=None,
        supervisor_pid=os.getpid(),
        stage="preflight",
        started_at=started_at.isoformat(),
        finished_at=None,
        max_spend_usd=policy.absolute_usd,
        soft_stop_usd=policy.soft_stop_usd,
        hard_stop_usd=policy.terminate_usd,
        max_gpu_hourly_usd=policy.max_hourly_cost_usd,
        max_wall_minutes=policy.max_runtime_seconds // 60,
        cleanup_verified=False,
    )
    owns_operator_receipt = False
    create_entered = False
    session: SinglePodSession | None = None
    capacity_race_inventory_verified = False
    full_inventory_restored = False
    budget_reconciliation: BudgetReconciliation | None = None
    budget_receipt_id: str | None = None
    checkpoint_store: Path | None = None
    resume_archive: Path | None = None
    recovery_attempt_root: Path | None = None
    readiness_sha256: str | None = None
    sealed_assets_sha256: str | None = None
    config_sha256: str | None = None
    environment_contract_sha256: str | None = None
    dependency_lock_sha256: str | None = None
    framework_wheelhouse: FrameworkWheelhouse | None = None
    minimum_gpu_memory_gib = 16
    try:
        _disk_gate()
        head, tracked = _preflight_repository()
        _verify_local_readiness()
        if args.sealed_acquisition is not None:
            minimum_gpu_memory_gib = _require_production_memory_plan(
                args.sealed_acquisition,
                run_id=run_id,
            )
        environment_contract_sha256, dependency_lock_sha256, framework_wheelhouse = (
            _remote_environment_preflight()
        )
        if args.sealed_acquisition is not None:
            billing_config = RunPodConfig(
                image_name=IMAGE,
                gpu_type_preferences=GPU_PREFERENCES,
                container_disk_gb=40,
                min_gpu_memory_gb=minimum_gpu_memory_gib,
            )
            with RunPodV1Client(api_token=token, config=billing_config) as billing_client:
                reconciliation_inventory = billing_client.inventory()
                require_empty_inventory(reconciliation_inventory)
                billing_snapshot = billing_client.account_billing_snapshot()
                budget_reconciliation, budget_receipt_path = _require_e2e_budget(
                    policy,
                    args.runtime_root,
                    run_id=run_id,
                    billing=billing_snapshot,
                    api_request_count=billing_client.api_request_count,
                    cloud_mutation_count=billing_client.cloud_mutation_count,
                )
                budget_receipt_id = budget_receipt_path.stem
                _emit(
                    "PHASE3F_BUDGET_RECONCILED",
                    actual_billed_usd=str(budget_reconciliation.actual_billed_usd),
                    conservative_unbilled_estimate_usd=str(
                        budget_reconciliation.conservative_unbilled_estimate_usd
                    ),
                    active_exposure_usd=str(budget_reconciliation.active_exposure_usd),
                    proposed_run_max_usd=str(budget_reconciliation.proposed_run_max_usd),
                    historical_total_cap_usd=str(
                        budget_reconciliation.historical_total_cap_usd
                    ),
                    remaining_authorized_usd=str(
                        budget_reconciliation.remaining_authorized_usd
                    ),
                    receipt_id=budget_receipt_id,
                    cloud_mutations=0,
                )
        prepare_operator_attempt(receipt_path, operator_receipt)
        owns_operator_receipt = True
        _emit("PHASE3F_LOCAL_PREFLIGHT_OK", branch=REQUIRED_BRANCH, head=head)
        key_root.mkdir(parents=True, exist_ok=True)
        _run_command(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                f"atlaslens-phase3f-{run_id}",
                "-f",
                str(key),
            ],
            timeout_seconds=30,
        )
        public_key = key.with_suffix(".pub").read_text(encoding="ascii").strip()
        bundle = prepare_transfer_bundle(
            ROOT,
            transfer_root,
            commit_sha=head,
            tracked_paths=tracked,
            model_path=ROOT / ".local" / "models" / "phase6c" / "megaloc" / "model.safetensors",
            vendor_root=ROOT / ".local" / "vendor" / "megaloc",
            sealed_root=args.sealed_acquisition,
            framework_wheelhouse=framework_wheelhouse,
        )
        _require(
            args.sealed_acquisition is None or bundle.dataset_archive is not None,
            "TRAINING_DATASET_ARCHIVE_MISSING",
        )
        if args.sealed_acquisition is not None:
            readiness_path = args.sealed_acquisition.parent / "training-readiness.json"
            _require(
                readiness_path.is_file() and not readiness_path.is_symlink(),
                "TRAINING_READINESS_REPORT_MISSING",
            )
            readiness_sha256 = sha256_path(readiness_path, max_bytes=1024 * 1024)
            sealed_assets_sha256 = sha256_path(
                args.sealed_acquisition / "sealed-assets.json",
                max_bytes=64 * 1024 * 1024,
            )
            config_sha256 = training_config_sha256(
                seed=20260720,
                max_epochs=8,
                environment_contract_sha256=environment_contract_sha256,
            )
            checkpoint_store = args.runtime_root / "_training" / run_id / "checkpoints"
            recovery_attempt_root = (
                args.runtime_root / "_training" / run_id / "attempts" / attempt_id
            )
            checkpoint = inspect_local_checkpoint(
                checkpoint_store,
                expected_run_id=run_id,
                expected_readiness_sha256=readiness_sha256,
                expected_sealed_assets_sha256=sealed_assets_sha256,
                expected_training_config_sha256=config_sha256,
            )
            _require(
                not checkpoint.present or checkpoint.valid,
                checkpoint.failure_code or "TRAINING_CHECKPOINT_INVALID",
            )
            resume_archive = checkpoint.archive_path if checkpoint.valid else None
        config = RunPodConfig(
            image_name=IMAGE,
            gpu_type_preferences=GPU_PREFERENCES,
            container_disk_gb=40,
            min_gpu_memory_gb=minimum_gpu_memory_gib,
            ssh_public_key=public_key,
        )
        with RunPodV1Client(
            api_token=token,
            config=config,
            require_receipt_binding=True,
            bind_created_pod=lambda pod: _bind_operator_pod(
                receipt_path,
                expected_run_id=run_id,
                expected_attempt_id=attempt_id,
                pod=pod,
            ),
            record_rental_attestation=lambda attestation: (
                _record_operator_rental_attestation(
                    receipt_path,
                    expected_run_id=run_id,
                    expected_attempt_id=attempt_id,
                    attestation=attestation,
                )
            ),
            record_gpu_attestation_progress=lambda diagnostic: (
                _record_operator_gpu_attestation_progress(
                    receipt_path,
                    expected_run_id=run_id,
                    expected_attempt_id=attempt_id,
                    diagnostic=diagnostic,
                )
            ),
            record_connectivity_progress=lambda diagnostic: (
                _record_operator_connectivity_progress(
                    receipt_path,
                    expected_run_id=run_id,
                    expected_attempt_id=attempt_id,
                    diagnostic=diagnostic,
                )
            ),
        ) as client:
            before = client.inventory()
            require_empty_inventory(before)
            _emit("PHASE3F_CLOUD_INVENTORY_EMPTY", resources=0)
            try:
                offer = client.select_gpu_offer(
                    max_hourly_price=policy.max_hourly_cost_usd
                )
            except RunPodAPIError as exc:
                if exc.code == "NO_ELIGIBLE_GPU_OFFER":
                    raise SupervisorExecutionError(
                        "GPU_MEMORY_PLAN_UNSATISFIED"
                    ) from exc
                raise
            _emit(
                "PHASE3F_GPU_OFFER_SELECTED",
                gpu=offer.display_name,
                hourly_price_usd=str(offer.hourly_price),
            )
            request = PodRequest(
                run_marker=operator_receipt.run_marker,
                idempotency_key=f"atlaslens-phase3f-create-{attempt_id}",
                hourly_cost_usd=policy.max_hourly_cost_usd,
                max_runtime_seconds=policy.max_runtime_seconds,
                gpu_type_id=offer.gpu_type_id,
                public_ports=(22,),
            )
            session = SinglePodSession(
                client,
                policy=policy,
                cleanup_poll_attempts=20,
                cleanup_poll_seconds=3.0,
            )
            try:
                create_entered = True
                execution = session.execute(
                    request,
                    lambda lease: _operation(
                        client,
                        lease,
                        bundle_root=bundle.root,
                        key=key,
                        known_hosts=known_hosts,
                        download_path=download_path,
                        run_id=run_id,
                        started=started,
                        operator_receipt=operator_receipt,
                        operator_receipt_path=receipt_path,
                        training_dataset=(
                            bundle.dataset_archive.path
                            if bundle.dataset_archive is not None
                            else None
                        ),
                        checkpoint_store=checkpoint_store,
                        resume_archive=resume_archive,
                        recovery_attempt_root=recovery_attempt_root,
                        readiness_sha256=readiness_sha256,
                        sealed_assets_sha256=sealed_assets_sha256,
                        config_sha256=config_sha256,
                        environment_contract_sha256=environment_contract_sha256,
                        dependency_lock_sha256=dependency_lock_sha256,
                    ),
                )
            except (Phase3FSafetyError, RunPodAPIError) as exc:
                diagnostic = client.last_create_response_diagnostic
                if diagnostic is not None:
                    _emit(
                        "PHASE3F_CREATE_RESPONSE_CLASSIFIED",
                        **diagnostic.to_public_dict(),
                    )
                after_failure = _post_inventory(client, before)
                require_inventory_restored(before, after_failure)
                full_inventory_restored = True
                if exc.code == "GPU_CAPACITY_RACE_NO_POD":
                    capacity_race_inventory_verified = True
                    _emit(
                        "PHASE3F_GPU_CAPACITY_RACE_NO_POD",
                        create_attempts=1,
                        resources=0,
                        cleanup_verified=True,
                    )
                raise
            except BaseException:
                after_failure = _post_inventory(client, before)
                require_inventory_restored(before, after_failure)
                full_inventory_restored = True
                raise
            after = _post_inventory(client, before)
            require_inventory_restored(before, after)
            full_inventory_restored = True
            _emit("PHASE3F_CLOUD_CLEANUP_VERIFIED", resources=0)
        current_operator_receipt = read_operator_receipt(receipt_path)
        write_operator_receipt(
            receipt_path,
            current_operator_receipt.update_lifecycle(
                stage="terminated",
                finished_at=datetime.now(UTC).isoformat(),
                cleanup_verified=True,
            ),
        )
        _disk_gate()
        verified = verify_output_archive(download_path, extracted_parent)
        published = publish_verified_output(verified, args.runtime_root)
        elapsed = time.monotonic() - started
        hourly = Decimal(str(execution.value["hourly_price_usd"]))
        conservative_compute = hourly * Decimal(str(elapsed)) / Decimal(3600)
        conservative_total = conservative_compute + Decimal("0.10")
        _require(conservative_total < policy.absolute_usd, "ABSOLUTE_BUDGET_REACHED")
        receipt = {
            "schema": "atlaslens-phase3f-local-supervisor-receipt-v1",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "outcome": verified.outcome,
            "pod_id_sha256": execution.value["pod_id_sha256"],
            "gpu": execution.value["gpu"],
            "hourly_price_usd": str(hourly),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "actual_spend_usd": None,
            "actual_spend_available": False,
            "conservative_compute_usd": str(conservative_compute),
            "conservative_storage_allowance_usd": "0.10",
            "conservative_incremental_upper_usd": str(conservative_total),
            "max_spend_usd": str(policy.absolute_usd),
            "soft_stop_usd": str(policy.soft_stop_usd),
            "hard_stop_usd": str(policy.terminate_usd),
            "max_gpu_hourly_usd": str(policy.max_hourly_cost_usd),
            "max_wall_minutes": policy.max_runtime_seconds // 60,
            "single_pod_create_attempts": execution.audit.create_attempts,
            "pod_termination_verified": execution.audit.termination_verified,
            "post_pods": len(after.pods),
            "post_endpoints": len(after.endpoint_ids),
            "post_network_volumes": len(after.network_volume_ids),
            "post_templates": len(after.template_ids),
            "publication_hash": verified.publication_hash,
            "output_inventory_sha256": verified.inventory_sha256,
            "output_size_bytes": verified.total_size_bytes,
            "published_path_name": published.name,
            "secret_values_included": False,
        }
        if args.sealed_acquisition is not None:
            reconciliation = budget_reconciliation
            if reconciliation is None:
                raise SupervisorExecutionError("BUDGET_RECONCILIATION_MISSING")
            receipt.update(
                {
                    "training_run_budget_usd": str(E2E_RUN_BUDGET_USD),
                    "actual_billed_usd": str(reconciliation.actual_billed_usd),
                    "conservative_unbilled_estimate_usd": str(
                        reconciliation.conservative_unbilled_estimate_usd
                    ),
                    "active_exposure_usd": str(reconciliation.active_exposure_usd),
                    "closed_unused_reservation_released_usd": str(
                        reconciliation.closed_unused_reservation_released_usd
                    ),
                    "remaining_authorized_usd": str(reconciliation.remaining_authorized_usd),
                    "budget_reconciliation_receipt_id": budget_receipt_id,
                    "historical_budget_limit_usd": str(E2E_HISTORICAL_BUDGET_USD),
                }
            )
        _write_receipt(
            args.runtime_root / "_receipts" / f"{published.name}.json",
            receipt,
        )
        _emit(
            "PHASE3F_OUTPUT_VERIFIED",
            outcome=verified.outcome,
            publication_path_name=published.name,
        )
        print(
            json.dumps(
                {
                    "outcome": verified.outcome,
                    "run_id": run_id,
                    "publication_hash": verified.publication_hash,
                    "cleanup_verified": True,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    except (
        Phase3FOperatorError,
        Phase3FSafetyError,
        Phase3FSupervisorError,
        RunPodAPIError,
        SupervisorExecutionError,
        TrainingRecoveryError,
        RemoteEnvironmentError,
    ) as exc:
        if owns_operator_receipt:
            cleanup_verified = bool(
                not create_entered
                or (
                    session is not None
                    and session.last_audit is not None
                    and session.last_audit.termination_verified
                    and full_inventory_restored
                )
            )
            no_pod_capacity_race = bool(
                isinstance(exc, (Phase3FSafetyError, RunPodAPIError))
                and exc.code == "GPU_CAPACITY_RACE_NO_POD"
                and capacity_race_inventory_verified
                and session is not None
                and session.last_audit is not None
                and session.last_audit.termination_attempts == 0
            )
            _record_failed_operator_state(
                receipt_path,
                cleanup_verified=cleanup_verified,
                no_pod_capacity_race=no_pod_capacity_race,
            )
        print(exc.code)
        return 1
    except BaseException:
        if owns_operator_receipt:
            cleanup_verified = bool(
                not create_entered
                or (
                    session is not None
                    and session.last_audit is not None
                    and session.last_audit.termination_verified
                    and full_inventory_restored
                )
            )
            _record_failed_operator_state(
                receipt_path,
                cleanup_verified=cleanup_verified,
            )
        print("PHASE3F_SUPERVISOR_UNEXPECTED_FAILURE")
        return 1
    finally:
        for path in (key, key.with_suffix(".pub"), known_hosts):
            try:
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            except OSError:
                pass
        if local_root.exists():
            shutil.rmtree(local_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.deadline_plan:
        plan = build_training_deadline_plan(args.max_wall_minutes)
        print(json.dumps(plan.to_public_dict(), separators=(",", ":"), sort_keys=True))
        return 0 if plan.ready_for_cloud else 1
    if (args.sealed_acquisition is None) != (args.dataset_run_id is None):
        print("PHASE3F_TRAINING_DATASET_ARGUMENTS_INCOMPLETE")
        return 1
    if args.dataset_run_id is not None and not _RUN_ID.fullmatch(args.dataset_run_id):
        print("RUN_ID_INVALID")
        return 1
    if args.sealed_acquisition is not None:
        os.environ.pop("MAPILLARY_ACCESS_TOKEN", None)
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        print("RUNPOD_API_KEY_NOT_VISIBLE_IN_APP")
        return 1
    try:
        args.runtime_root = _require_safe_runtime_path(args.runtime_root)
        receipt_path = _operator_receipt_path(args)
        if args.status or args.terminate:
            return _run_control(args, token=token, receipt_path=receipt_path)
        if args.live_readiness:
            return _run_live_readiness(
                args,
                token=token,
                receipt_path=receipt_path,
            )
        if not args.execute:
            return _run_dry_run(args, receipt_path=receipt_path)
        return _run_execute(args, token=token, receipt_path=receipt_path)
    except (
        Phase3FOperatorError,
        Phase3FSafetyError,
        Phase3FSupervisorError,
        RunPodAPIError,
        SupervisorExecutionError,
        TrainingRecoveryError,
    ) as exc:
        print(exc.code)
        return 1
    except BaseException:
        print("PHASE3F_SUPERVISOR_UNEXPECTED_FAILURE")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
