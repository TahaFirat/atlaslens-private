from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

_SCHEMA_VERSION = "atlaslens-rapidocr-runtime-verification-v1"
_PROVIDER = "rapidocr"


class RapidOCRRuntimeVerification(BaseModel):
    """Non-sensitive proof produced only after a successful real OCR smoke."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^atlaslens-rapidocr-runtime-verification-v1$")
    provider: str = Field(pattern=r"^rapidocr$")
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    provider_version: str = Field(min_length=1, max_length=40)
    model_revision: str = Field(min_length=1, max_length=160)
    runtime_family: str = Field(pattern=r"^onnxruntime-(cpu|gpu)$")
    runtime_version: str = Field(min_length=1, max_length=40)
    device: str = Field(pattern=r"^(cpu|cuda)$")
    artifact_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    worker_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    load_verified: bool
    inference_verified: bool
    verified_at: datetime


class RapidOCRVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verification: RapidOCRRuntimeVerification | None = None
    reason_code: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,120}$")


def rapidocr_verification_path(project_root: Path) -> Path:
    return (
        project_root.resolve()
        / ".local"
        / "runtime"
        / "phase6b"
        / "rapidocr-verification.json"
    )


def write_rapidocr_runtime_verification(
    project_root: Path,
    model_root: Path,
    *,
    source_revision: str,
    model_revision: str,
    model_load_succeeded: bool,
    real_inference_succeeded: bool,
) -> Path:
    """Atomically persist proof after, and only after, a real RapidOCR result.

    The receipt intentionally contains no image hash, OCR text, local path, or timing.
    Callers must invoke this only after the real worker returned a validated result.
    """

    if not model_load_succeeded or not real_inference_succeeded:
        raise ValueError("real RapidOCR load and inference are required")
    artifact = _read_artifact_receipt(model_root)
    provider_version = artifact.get("provider_version")
    runtime_family = artifact.get("runtime_family")
    runtime_version = artifact.get("runtime_version")
    if (
        not isinstance(provider_version, str)
        or not provider_version
        or not isinstance(runtime_family, str)
        or not runtime_family
        or not isinstance(runtime_version, str)
        or not runtime_version
    ):
        raise ValueError("RapidOCR artifact receipt is invalid")
    device = "cuda" if runtime_family == "onnxruntime-gpu" else "cpu"
    receipt_path, provider_source, worker_source = _bound_files(
        project_root, model_root
    )
    verification = RapidOCRRuntimeVerification(
        schema_version=_SCHEMA_VERSION,
        provider=_PROVIDER,
        source_revision=source_revision,
        provider_version=provider_version,
        model_revision=model_revision,
        runtime_family=runtime_family,
        runtime_version=runtime_version,
        device=device,
        artifact_receipt_sha256=_sha256(receipt_path),
        provider_source_sha256=_sha256(provider_source),
        worker_source_sha256=_sha256(worker_source),
        load_verified=True,
        inference_verified=True,
        verified_at=datetime.now(UTC),
    )
    destination = rapidocr_verification_path(project_root)
    _ensure_private_runtime_directory(destination.parent, project_root.resolve())
    if destination.is_symlink():
        raise ValueError("RapidOCR runtime verification path is unsafe")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            verification.model_dump_json(indent=2), encoding="utf-8"
        )
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return destination


def validate_rapidocr_runtime_verification(
    project_root: Path,
    model_root: Path,
    *,
    expected_source_revision: str,
    expected_model_revision: str,
) -> RapidOCRVerificationResult:
    """Validate that runtime proof still matches artifacts and implementation."""

    verification_path = rapidocr_verification_path(project_root)
    if not verification_path.is_file() or verification_path.is_symlink():
        return RapidOCRVerificationResult(reason_code="runtime_verification_missing")
    try:
        verification = RapidOCRRuntimeVerification.model_validate_json(
            verification_path.read_text(encoding="utf-8")
        )
        artifact = _read_artifact_receipt(model_root)
        receipt_path, provider_source, worker_source = _bound_files(
            project_root, model_root
        )
        expected = {
            "source_revision": expected_source_revision,
            "model_revision": expected_model_revision,
            "provider_version": artifact.get("provider_version"),
            "runtime_family": artifact.get("runtime_family"),
            "runtime_version": artifact.get("runtime_version"),
            "artifact_receipt_sha256": _sha256(receipt_path),
            "provider_source_sha256": _sha256(provider_source),
            "worker_source_sha256": _sha256(worker_source),
        }
        if any(getattr(verification, key) != value for key, value in expected.items()):
            return RapidOCRVerificationResult(
                reason_code="runtime_verification_stale"
            )
        expected_device = (
            "cuda" if verification.runtime_family == "onnxruntime-gpu" else "cpu"
        )
        if (
            verification.device != expected_device
            or not verification.load_verified
            or not verification.inference_verified
        ):
            return RapidOCRVerificationResult(
                reason_code="runtime_verification_invalid"
            )
    except (OSError, ValueError, json.JSONDecodeError):
        return RapidOCRVerificationResult(reason_code="runtime_verification_invalid")
    return RapidOCRVerificationResult(verification=verification)


def _read_artifact_receipt(model_root: Path) -> dict[str, object]:
    requested = model_root.expanduser()
    if requested.is_symlink():
        raise ValueError("RapidOCR model root is unsafe")
    root = requested.resolve(strict=True)
    receipt = root / "receipt.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("RapidOCR artifact receipt is missing")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "atlaslens-rapidocr-v1":
        raise ValueError("RapidOCR artifact receipt is invalid")
    return payload


def _bound_files(
    project_root: Path, model_root: Path
) -> tuple[Path, Path, Path]:
    root = project_root.resolve(strict=True)
    receipt = model_root.expanduser().resolve(strict=True) / "receipt.json"
    provider_source = (
        root / "services/api/src/atlaslens_api/providers/rapidocr.py"
    ).resolve(strict=True)
    worker_source = (
        root / "services/api/src/atlaslens_api/providers/rapidocr_worker.py"
    ).resolve(strict=True)
    for source in (receipt, provider_source, worker_source):
        if source.is_symlink() or not source.is_file():
            raise ValueError("RapidOCR verification input is unsafe")
    return receipt, provider_source, worker_source


def _ensure_private_runtime_directory(directory: Path, project_root: Path) -> None:
    local_root = project_root / ".local"
    if local_root.is_symlink():
        raise ValueError("private runtime root is unsafe")
    local_root.mkdir(exist_ok=True)
    current = local_root
    for part in ("runtime", "phase6b"):
        current = current / part
        if current.is_symlink():
            raise ValueError("private runtime directory is unsafe")
        current.mkdir(exist_ok=True)
    if directory.resolve() != current.resolve():
        raise ValueError("private runtime directory is invalid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
