"""Typed Phase 3F remote-training failure and checkpoint recovery primitives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Final, cast
from uuid import uuid4

REMOTE_FAILURE_SCHEMA: Final = "atlaslens-phase3f-remote-training-failure-v1"
REMOTE_PROGRESS_SCHEMA: Final = "atlaslens-phase3f-remote-training-progress-v1"
REMOTE_TELEMETRY_SCHEMA: Final = "atlaslens-phase3f-remote-training-telemetry-v1"
CHECKPOINT_MANIFEST_SCHEMA: Final = "atlaslens-phase3f-training-checkpoint-manifest-v2"
CHECKPOINT_POINTER_SCHEMA: Final = "atlaslens-phase3f-training-checkpoint-pointer-v2"
LOCAL_CHECKPOINT_SCHEMA: Final = "atlaslens-phase3f-local-training-checkpoint-v1"
SALVAGE_RECEIPT_SCHEMA: Final = "atlaslens-phase3f-training-salvage-v1"
MAX_TAIL_LINES: Final = 100
MAX_TAIL_LINE_CHARS: Final = 512
MAX_RECOVERY_ARCHIVE_BYTES: Final = 3 * 1024 * 1024 * 1024
MAX_RECOVERY_FILES: Final = 2_048
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_GENERATION = re.compile(r"^generation-[0-9]{8}-[0-9a-f]{16}$")
_URL = re.compile(r"(?i)\b(?:https?|ssh)://\S+")
_IPV4 = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:authorization|api[_-]?key|access[_-]?token|secret|password)"
    r"\s*[:=]\s*\S+"
)
_LONG_VALUE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9+/_=-]{32,}(?![A-Za-z0-9])")
_PRIVATE_PATH = re.compile(r"(?i)(?:[A-Z]:\\Users\\[^\\\s]+|/home/[^/\s]+)")


class TrainingRecoveryError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise TrainingRecoveryError(code)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, value: object) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path, *, max_bytes: int = MAX_RECOVERY_ARCHIVE_BYTES) -> str:
    _require(path.is_file() and not path.is_symlink(), "RECOVERY_ARTIFACT_INVALID")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            _require(size <= max_bytes, "RECOVERY_ARTIFACT_TOO_LARGE")
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_log_line(value: str) -> str:
    line = "".join(character for character in value if character >= " " or character == "\t")
    line = _URL.sub("[REDACTED_URL]", line)
    line = _IPV4.sub("[REDACTED_IP]", line)
    line = _PRIVATE_PATH.sub("[REDACTED_PATH]", line)
    line = _SECRET_ASSIGNMENT.sub("[REDACTED_SECRET]", line)
    line = _LONG_VALUE.sub(
        lambda match: (
            match.group(0)
            if _SAFE_CODE.fullmatch(match.group(0))
            else "[REDACTED_VALUE]"
        ),
        line,
    )
    return line[:MAX_TAIL_LINE_CHARS]


def sanitized_tail(value: str, *, maximum_lines: int = MAX_TAIL_LINES) -> tuple[str, ...]:
    _require(1 <= maximum_lines <= MAX_TAIL_LINES, "REMOTE_LOG_TAIL_LIMIT_INVALID")
    return tuple(
        sanitize_log_line(line)
        for line in value.splitlines()[-maximum_lines:]
        if line.strip()
    )


_TYPED_REMOTE_CODES: Final[tuple[tuple[str, str], ...]] = (
    ("TRAINING_CUDA_OOM", "REMOTE_TRAINING_CUDA_OOM"),
    ("CUDA_OUT_OF_MEMORY", "REMOTE_TRAINING_CUDA_OOM"),
    ("TRAINING_WALL_LIMIT_REACHED", "REMOTE_TRAINING_DEADLINE"),
    ("REMOTE_TRAINING_DEADLINE", "REMOTE_TRAINING_DEADLINE"),
    ("NO_SPACE_LEFT", "REMOTE_TRAINING_DISK_FULL"),
    ("ENOSPC", "REMOTE_TRAINING_DISK_FULL"),
    ("TRAINING_IMAGE_INVALID", "REMOTE_TRAINING_DATALOADER_FAILED"),
    ("DATALOADER", "REMOTE_TRAINING_DATALOADER_FAILED"),
    ("NONFINITE_LOSS", "REMOTE_TRAINING_NONFINITE_LOSS"),
    ("TRAINING_CHECKPOINT", "REMOTE_TRAINING_CHECKPOINT_FAILED"),
    ("DEPENDENCY", "REMOTE_TRAINING_DEPENDENCY_FAILED"),
    ("RUNTIME_MISSING", "REMOTE_TRAINING_DEPENDENCY_FAILED"),
)


@dataclass(frozen=True, slots=True)
class RemoteFailureDiagnostic:
    failure_code: str
    process_exit_code: int | None
    process_signal: int | None
    exception_class: str | None
    message_code: str | None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "failure_code": self.failure_code,
            "process_exit_code": self.process_exit_code,
            "process_signal": self.process_signal,
            "exception_class": self.exception_class,
            "message_code": self.message_code,
            "stdout_tail": list(self.stdout_tail),
            "stderr_tail": list(self.stderr_tail),
        }


def classify_remote_failure(
    return_code: int | None,
    *,
    stdout: str = "",
    stderr: str = "",
    explicit_code: str | None = None,
    exception_class: str | None = None,
) -> RemoteFailureDiagnostic:
    stdout_tail = sanitized_tail(stdout)
    stderr_tail = sanitized_tail(stderr)
    candidates = tuple(
        value.upper()
        for value in (*stdout_tail, *stderr_tail, explicit_code or "")
    )
    combined = "\n".join(candidates)
    failure_code = "REMOTE_TRAINING_UNKNOWN_FAILURE"
    if explicit_code is not None and _SAFE_CODE.fullmatch(explicit_code):
        if explicit_code.startswith("REMOTE_TRAINING_"):
            failure_code = explicit_code
        else:
            for marker, mapped in _TYPED_REMOTE_CODES:
                if marker in explicit_code:
                    failure_code = mapped
                    break
    if failure_code == "REMOTE_TRAINING_UNKNOWN_FAILURE":
        for marker, mapped in _TYPED_REMOTE_CODES:
            if marker in combined:
                failure_code = mapped
                break
    signal_number: int | None = None
    exit_code = return_code
    if return_code is not None and return_code < 0:
        signal_number = -return_code
        exit_code = None
    elif return_code is not None and 128 < return_code <= 192:
        signal_number = return_code - 128
    if signal_number is not None and failure_code == "REMOTE_TRAINING_UNKNOWN_FAILURE":
        failure_code = "REMOTE_TRAINING_PROCESS_SIGNALLED"
    if return_code in {124, 137, 143} and failure_code == "REMOTE_TRAINING_UNKNOWN_FAILURE":
        failure_code = (
            "REMOTE_TRAINING_DEADLINE"
            if return_code == 124
            else "REMOTE_TRAINING_PROCESS_SIGNALLED"
        )
    safe_exception = (
        exception_class
        if isinstance(exception_class, str)
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", exception_class)
        else None
    )
    safe_message = (
        explicit_code
        if isinstance(explicit_code, str) and _SAFE_CODE.fullmatch(explicit_code)
        else None
    )
    return RemoteFailureDiagnostic(
        failure_code=failure_code,
        process_exit_code=exit_code,
        process_signal=signal_number,
        exception_class=safe_exception,
        message_code=safe_message,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


@dataclass(frozen=True, slots=True)
class CheckpointStatus:
    present: bool
    valid: bool
    failure_code: str | None
    archive_path: Path | None
    archive_sha256: str | None
    archive_size_bytes: int | None
    completed_epoch: int
    next_epoch: int
    next_batch_index: int
    optimizer_steps: int
    holdout_open_count: int
    dataset_readiness_sha256: str | None
    sealed_assets_sha256: str | None
    training_config_sha256: str | None
    artifact_count: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "present": self.present,
            "valid": self.valid,
            "failure_code": self.failure_code,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "completed_epoch": self.completed_epoch,
            "next_epoch": self.next_epoch,
            "next_batch_index": self.next_batch_index,
            "optimizer_steps": self.optimizer_steps,
            "holdout_open_count": self.holdout_open_count,
            "dataset_readiness_sha256": self.dataset_readiness_sha256,
            "sealed_assets_sha256": self.sealed_assets_sha256,
            "training_config_sha256": self.training_config_sha256,
            "artifact_count": self.artifact_count,
        }


def _missing_checkpoint() -> CheckpointStatus:
    return CheckpointStatus(
        present=False,
        valid=False,
        failure_code=None,
        archive_path=None,
        archive_sha256=None,
        archive_size_bytes=None,
        completed_epoch=-1,
        next_epoch=0,
        next_batch_index=0,
        optimizer_steps=0,
        holdout_open_count=0,
        dataset_readiness_sha256=None,
        sealed_assets_sha256=None,
        training_config_sha256=None,
        artifact_count=0,
    )


def _safe_tar_members(archive: tarfile.TarFile) -> tuple[tarfile.TarInfo, ...]:
    members = tuple(archive.getmembers())
    _require(len(members) <= MAX_RECOVERY_FILES, "TRAINING_CHECKPOINT_FILE_LIMIT")
    for member in members:
        pure = PurePosixPath(member.name)
        _require(
            not pure.is_absolute()
            and bool(pure.parts)
            and all(part not in {"", ".", ".."} for part in pure.parts)
            and not member.issym()
            and not member.islnk()
            and (member.isfile() or member.isdir())
            and not member.name.endswith((".partial", ".tmp")),
            "TRAINING_CHECKPOINT_ARCHIVE_INVALID",
        )
    return members


def _extract_verified_tar(archive_path: Path, destination: Path) -> None:
    _require(
        archive_path.stat().st_size <= MAX_RECOVERY_ARCHIVE_BYTES,
        "RECOVERY_ARTIFACT_TOO_LARGE",
    )
    total = 0
    try:
        with tarfile.open(archive_path, "r:") as archive:
            members = _safe_tar_members(archive)
            for member in members:
                target = destination.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(mode=0o700, parents=True, exist_ok=True)
                    continue
                total += member.size
                _require(total <= MAX_RECOVERY_ARCHIVE_BYTES, "RECOVERY_ARTIFACT_TOO_LARGE")
                source = archive.extractfile(member)
                if source is None:
                    raise TrainingRecoveryError("TRAINING_CHECKPOINT_ARCHIVE_INVALID")
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
    except (OSError, tarfile.TarError) as exc:
        raise TrainingRecoveryError("TRAINING_CHECKPOINT_ARCHIVE_INVALID") from exc


def extract_recovery_archive(archive_path: Path, destination: Path) -> tuple[Path, ...]:
    """Extract a bounded regular-file-only archive into a new destination."""
    _require(not destination.exists(), "RECOVERY_DESTINATION_EXISTS")
    destination.mkdir(mode=0o700, parents=True)
    try:
        _extract_verified_tar(archive_path, destination)
    except TrainingRecoveryError:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return tuple(sorted(path for path in destination.rglob("*") if path.is_file()))


def _read_json(path: Path, *, max_bytes: int = 1024 * 1024) -> dict[str, object]:
    _require(
        path.is_file()
        and not path.is_symlink()
        and 0 < path.stat().st_size <= max_bytes,
        "TRAINING_CHECKPOINT_INVALID",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingRecoveryError("TRAINING_CHECKPOINT_INVALID") from exc
    _require(isinstance(value, dict), "TRAINING_CHECKPOINT_INVALID")
    return cast(dict[str, object], value)


def validate_checkpoint_tree(
    checkpoint_root: Path,
    *,
    expected_run_id: str,
    expected_readiness_sha256: str,
    expected_sealed_assets_sha256: str,
    expected_training_config_sha256: str,
) -> CheckpointStatus:
    _require(bool(_HEX32.fullmatch(expected_run_id)), "RUN_ID_INVALID")
    pointer = _read_json(checkpoint_root / "latest.json")
    generation = pointer.get("generation")
    _require(
        pointer.get("schema") == CHECKPOINT_POINTER_SCHEMA
        and isinstance(generation, str)
        and bool(_GENERATION.fullmatch(generation)),
        "TRAINING_CHECKPOINT_INVALID",
    )
    generation = cast(str, generation)
    generation_root = checkpoint_root / generation
    _require(
        generation_root.is_dir() and not generation_root.is_symlink(),
        "TRAINING_CHECKPOINT_INVALID",
    )
    manifest_path = generation_root / "checkpoint-manifest.json"
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts")
    _require(
        manifest.get("schema") == CHECKPOINT_MANIFEST_SCHEMA
        and manifest.get("run_id") == expected_run_id
        and manifest.get("dataset_readiness_sha256") == expected_readiness_sha256
        and manifest.get("sealed_assets_sha256") == expected_sealed_assets_sha256
        and manifest.get("training_config_sha256") == expected_training_config_sha256
        and manifest.get("holdout_open_count") == 0
        and isinstance(artifacts, list)
        and bool(artifacts),
        "TRAINING_CHECKPOINT_BINDING_MISMATCH",
    )
    artifacts = cast(list[object], artifacts)
    required = {
        "trainable.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng.pt",
        "training-state.json",
    }
    seen: set[str] = set()
    for item in artifacts:
        _require(isinstance(item, dict), "TRAINING_CHECKPOINT_INVALID")
        row = cast(dict[str, object], item)
        relative = row.get("path")
        size = row.get("size_bytes")
        digest = row.get("sha256")
        _require(
            isinstance(relative, str)
            and relative in required
            and relative not in seen
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0
            and isinstance(digest, str)
            and bool(_SHA256.fullmatch(digest)),
            "TRAINING_CHECKPOINT_INVALID",
        )
        relative = cast(str, relative)
        path = generation_root / relative
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == size
            and sha256_path(path) == digest,
            "TRAINING_CHECKPOINT_HASH_MISMATCH",
        )
        seen.add(relative)
    _require(seen == required, "TRAINING_CHECKPOINT_INVALID")
    state = _read_json(generation_root / "training-state.json")
    completed_epoch = state.get("completed_epoch")
    next_epoch = state.get("next_epoch")
    next_batch_index = state.get("next_batch_index")
    optimizer_steps = state.get("optimizer_steps")
    holdout_open_count = state.get("holdout_open_count")
    integer_fields = (
        completed_epoch,
        next_epoch,
        next_batch_index,
        optimizer_steps,
        holdout_open_count,
    )
    _require(
        state.get("schema") == "atlaslens-phase3f-training-checkpoint-v2"
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in integer_fields
        ),
        "TRAINING_CHECKPOINT_INVALID",
    )
    completed_epoch = cast(int, completed_epoch)
    next_epoch = cast(int, next_epoch)
    next_batch_index = cast(int, next_batch_index)
    optimizer_steps = cast(int, optimizer_steps)
    holdout_open_count = cast(int, holdout_open_count)
    _require(
        completed_epoch >= -1
        and next_epoch >= 0
        and next_batch_index >= 0
        and optimizer_steps >= 0
        and holdout_open_count == 0,
        "TRAINING_CHECKPOINT_INVALID",
    )
    return CheckpointStatus(
        present=True,
        valid=True,
        failure_code=None,
        archive_path=None,
        archive_sha256=None,
        archive_size_bytes=None,
        completed_epoch=completed_epoch,
        next_epoch=next_epoch,
        next_batch_index=next_batch_index,
        optimizer_steps=optimizer_steps,
        holdout_open_count=0,
        dataset_readiness_sha256=expected_readiness_sha256,
        sealed_assets_sha256=expected_sealed_assets_sha256,
        training_config_sha256=expected_training_config_sha256,
        artifact_count=len(artifacts),
    )


def inspect_checkpoint_archive(
    archive_path: Path,
    *,
    expected_run_id: str,
    expected_readiness_sha256: str,
    expected_sealed_assets_sha256: str,
    expected_training_config_sha256: str,
) -> CheckpointStatus:
    digest = sha256_path(archive_path)
    size = archive_path.stat().st_size
    with tempfile.TemporaryDirectory(prefix="atlaslens-phase3f-checkpoint-") as temporary:
        temporary_root = Path(temporary)
        _extract_verified_tar(archive_path, temporary_root)
        candidates = tuple(temporary_root.rglob("latest.json"))
        _require(len(candidates) == 1, "TRAINING_CHECKPOINT_ARCHIVE_INVALID")
        tree = validate_checkpoint_tree(
            candidates[0].parent,
            expected_run_id=expected_run_id,
            expected_readiness_sha256=expected_readiness_sha256,
            expected_sealed_assets_sha256=expected_sealed_assets_sha256,
            expected_training_config_sha256=expected_training_config_sha256,
        )
        holdout_path = candidates[0].parent / "holdout-state.json"
        if holdout_path.exists():
            holdout = _read_json(holdout_path)
            _require(
                holdout.get("schema") == "atlaslens-phase3f-holdout-state-v1"
                and holdout.get("run_id") == expected_run_id
                and holdout.get("holdout_open_count") == 1,
                "TRAINING_HOLDOUT_STATE_INVALID",
            )
            tree = replace(
                tree,
                valid=False,
                failure_code="TRAINING_HOLDOUT_ALREADY_OPENED",
                holdout_open_count=1,
            )
    return replace(
        tree,
        archive_path=archive_path,
        archive_sha256=digest,
        archive_size_bytes=size,
    )


def inspect_local_checkpoint(
    checkpoint_store: Path,
    *,
    expected_run_id: str,
    expected_readiness_sha256: str,
    expected_sealed_assets_sha256: str,
    expected_training_config_sha256: str,
) -> CheckpointStatus:
    pointer_path = checkpoint_store / "latest.json"
    if not pointer_path.exists():
        return _missing_checkpoint()
    try:
        pointer = _read_json(pointer_path)
        archive_name = pointer.get("archive_name")
        digest = pointer.get("archive_sha256")
        size = pointer.get("archive_size_bytes")
        _require(
            pointer.get("schema") == LOCAL_CHECKPOINT_SCHEMA
            and pointer.get("run_id") == expected_run_id
            and isinstance(archive_name, str)
            and bool(re.fullmatch(r"checkpoint-[0-9a-f]{64}\.tar", archive_name))
            and isinstance(digest, str)
            and bool(_SHA256.fullmatch(digest))
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0,
            "TRAINING_CHECKPOINT_INVALID",
        )
        archive_name = cast(str, archive_name)
        archive_path = checkpoint_store / archive_name
        _require(
            archive_path.is_file()
            and not archive_path.is_symlink()
            and archive_path.stat().st_size == size
            and sha256_path(archive_path) == digest,
            "TRAINING_CHECKPOINT_HASH_MISMATCH",
        )
        return inspect_checkpoint_archive(
            archive_path,
            expected_run_id=expected_run_id,
            expected_readiness_sha256=expected_readiness_sha256,
            expected_sealed_assets_sha256=expected_sealed_assets_sha256,
            expected_training_config_sha256=expected_training_config_sha256,
        )
    except TrainingRecoveryError as exc:
        return replace(
            _missing_checkpoint(),
            present=True,
            failure_code=exc.code,
        )


def store_verified_checkpoint_archive(
    partial_archive: Path,
    checkpoint_store: Path,
    *,
    expected_run_id: str,
    expected_readiness_sha256: str,
    expected_sealed_assets_sha256: str,
    expected_training_config_sha256: str,
) -> CheckpointStatus:
    status = inspect_checkpoint_archive(
        partial_archive,
        expected_run_id=expected_run_id,
        expected_readiness_sha256=expected_readiness_sha256,
        expected_sealed_assets_sha256=expected_sealed_assets_sha256,
        expected_training_config_sha256=expected_training_config_sha256,
    )
    digest = cast(str, status.archive_sha256)
    checkpoint_store.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = checkpoint_store / f"checkpoint-{digest}.tar"
    if destination.exists():
        _require(
            destination.is_file()
            and not destination.is_symlink()
            and destination.stat().st_size == partial_archive.stat().st_size
            and sha256_path(destination) == digest,
            "TRAINING_CHECKPOINT_ARCHIVE_CONFLICT",
        )
        partial_archive.unlink()
    else:
        os.replace(partial_archive, destination)
    atomic_json(
        checkpoint_store / "latest.json",
        {
            "schema": LOCAL_CHECKPOINT_SCHEMA,
            "run_id": expected_run_id,
            "archive_name": destination.name,
            "archive_sha256": digest,
            "archive_size_bytes": destination.stat().st_size,
            "completed_epoch": status.completed_epoch,
            "next_epoch": status.next_epoch,
            "next_batch_index": status.next_batch_index,
            "optimizer_steps": status.optimizer_steps,
            "holdout_open_count": status.holdout_open_count,
            "dataset_readiness_sha256": expected_readiness_sha256,
            "sealed_assets_sha256": expected_sealed_assets_sha256,
            "training_config_sha256": expected_training_config_sha256,
            "secret_values_included": False,
        },
    )
    return replace(status, archive_path=destination)


__all__ = [
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CHECKPOINT_POINTER_SCHEMA",
    "CheckpointStatus",
    "LOCAL_CHECKPOINT_SCHEMA",
    "MAX_RECOVERY_ARCHIVE_BYTES",
    "REMOTE_FAILURE_SCHEMA",
    "REMOTE_PROGRESS_SCHEMA",
    "REMOTE_TELEMETRY_SCHEMA",
    "RemoteFailureDiagnostic",
    "SALVAGE_RECEIPT_SCHEMA",
    "TrainingRecoveryError",
    "atomic_json",
    "canonical_bytes",
    "classify_remote_failure",
    "extract_recovery_archive",
    "inspect_checkpoint_archive",
    "inspect_local_checkpoint",
    "sanitize_log_line",
    "sanitized_tail",
    "sha256_path",
    "store_verified_checkpoint_archive",
    "validate_checkpoint_tree",
]
