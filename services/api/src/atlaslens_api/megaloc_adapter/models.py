"""Configuration, approval, and readiness models for Phase 3B2 MegaLoc."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from atlaslens_api.corpus_index.artifacts import canonical_json_bytes
from atlaslens_api.megaloc_adapter.errors import MegaLocAdapterError

MEGALOC_DESCRIPTOR_DIMENSION = 8_448
MEGALOC_PREPROCESSING_ID = "rgb-imagenet-max560-multiple14-v1"
MEGALOC_PROVIDER_ID = "megaloc"

type MegaLocDevice = Literal["auto", "cpu", "cuda"]
type ResolvedMegaLocDevice = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class MegaLocApproval:
    """Explicit human/legal/technical approvals; absence always fails closed."""

    artifact_hash_approved: bool
    source_code_approved: bool
    source_license_approved: bool
    weight_license_approved: bool
    descriptor_contract_approved: bool
    preprocessing_approved: bool
    production_use_approved: bool
    approval_record_id: str
    weight_license_record_id: str

    def __post_init__(self) -> None:
        if not self.approval_record_id.strip() or not self.weight_license_record_id.strip():
            raise ValueError("MegaLoc approval record IDs must not be blank")

    @property
    def fully_approved(self) -> bool:
        return all(
            (
                self.artifact_hash_approved,
                self.source_code_approved,
                self.source_license_approved,
                self.weight_license_approved,
                self.descriptor_contract_approved,
                self.preprocessing_approved,
                self.production_use_approved,
            )
        )


@dataclass(frozen=True, slots=True)
class MegaLocAdapterConfig:
    """Resolved, explicit offline configuration with no user-directory defaults."""

    project_root: Path
    source_dir: Path
    source_code_path: Path
    source_code_sha256: str
    source_revision: str
    source_license_path: Path
    source_license_sha256: str
    source_license_spdx: str
    artifact_path: Path
    artifact_sha256: str
    artifact_size: int
    artifact_format: Literal["safetensors"]
    receipt_path: Path
    receipt_sha256: str
    smoke_receipt_path: Path
    model_id: str
    model_revision: str
    weight_license_spdx: str
    descriptor_version: str
    descriptor_dimension: Literal[8448]
    preprocessing_id: str
    maximum_edge: int
    device: MegaLocDevice
    max_batch_size: int
    max_input_bytes: int
    max_image_pixels: int
    deterministic_algorithms: bool
    reduce_batch_on_cuda_oom: bool
    approval: MegaLocApproval

    def __post_init__(self) -> None:
        for field, value in (
            ("source_code_sha256", self.source_code_sha256),
            ("source_license_sha256", self.source_license_sha256),
            ("artifact_sha256", self.artifact_sha256),
            ("receipt_sha256", self.receipt_sha256),
        ):
            _require_sha256(value, field)
        for field, value in (
            ("source_revision", self.source_revision),
            ("source_license_spdx", self.source_license_spdx),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("weight_license_spdx", self.weight_license_spdx),
            ("descriptor_version", self.descriptor_version),
            ("preprocessing_id", self.preprocessing_id),
        ):
            if not value.strip():
                raise ValueError(f"{field} must not be blank")
        if self.descriptor_dimension != MEGALOC_DESCRIPTOR_DIMENSION:
            raise ValueError("MegaLoc descriptor dimension must be 8448")
        if self.preprocessing_id != MEGALOC_PREPROCESSING_ID:
            raise ValueError("unsupported MegaLoc preprocessing profile")
        if self.artifact_size <= 0:
            raise ValueError("MegaLoc artifact size must be positive")
        if not 224 <= self.maximum_edge <= 1_120 or self.maximum_edge % 14:
            raise ValueError("MegaLoc maximum edge must be a multiple of 14")
        if not 1 <= self.max_batch_size <= 64:
            raise ValueError("MegaLoc max batch size must be between 1 and 64")
        if self.max_input_bytes <= 0 or self.max_image_pixels <= 0:
            raise ValueError("MegaLoc input bounds must be positive")


@dataclass(frozen=True, slots=True)
class MegaLocReadiness:
    state: Literal["ready", "not_ready"]
    reason_code: str | None
    artifact_present: bool
    artifact_receipt_verified: bool
    source_verified: bool
    license_records_verified: bool
    approvals_verified: bool
    smoke_verified: bool
    descriptor_version: str | None
    descriptor_dimension: int
    preprocessing_id: str | None
    artifact_sha256: str | None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def to_json(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason_code": self.reason_code,
            "artifact_present": self.artifact_present,
            "artifact_receipt_verified": self.artifact_receipt_verified,
            "source_verified": self.source_verified,
            "license_records_verified": self.license_records_verified,
            "approvals_verified": self.approvals_verified,
            "smoke_verified": self.smoke_verified,
            "descriptor_version": self.descriptor_version,
            "descriptor_dimension": self.descriptor_dimension,
            "preprocessing_id": self.preprocessing_id,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class MegaLocSmokeReceipt:
    artifact_sha256: str
    source_revision: str
    model_revision: str
    descriptor_version: str
    descriptor_dimension: Literal[8448]
    preprocessing_id: str
    device: ResolvedMegaLocDevice
    smoke_input_sha256: str
    descriptor_l2_norm: float

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        _require_sha256(self.smoke_input_sha256, "smoke_input_sha256")
        for field, value in (
            ("source_revision", self.source_revision),
            ("model_revision", self.model_revision),
            ("descriptor_version", self.descriptor_version),
            ("preprocessing_id", self.preprocessing_id),
        ):
            if not value.strip():
                raise ValueError(f"{field} must not be blank")
        if self.descriptor_dimension != MEGALOC_DESCRIPTOR_DIMENSION:
            raise ValueError("MegaLoc smoke dimension must be 8448")
        if self.preprocessing_id != MEGALOC_PREPROCESSING_ID:
            raise ValueError("MegaLoc smoke preprocessing is incompatible")
        if not math.isfinite(self.descriptor_l2_norm) or abs(self.descriptor_l2_norm - 1.0) > 1e-3:
            raise ValueError("MegaLoc smoke descriptor must be L2 normalized")

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-megaloc-production-smoke-v1",
            "provider": MEGALOC_PROVIDER_ID,
            "artifact_sha256": self.artifact_sha256,
            "source_revision": self.source_revision,
            "model_revision": self.model_revision,
            "descriptor_version": self.descriptor_version,
            "descriptor_dimension": self.descriptor_dimension,
            "preprocessing_id": self.preprocessing_id,
            "device": self.device,
            "smoke_input_sha256": self.smoke_input_sha256,
            "descriptor_l2_norm": self.descriptor_l2_norm,
            "smoke_verified": True,
        }

    @classmethod
    def from_json(cls, value: object) -> MegaLocSmokeReceipt:
        required_keys = {
            "schema",
            "provider",
            "artifact_sha256",
            "source_revision",
            "model_revision",
            "descriptor_version",
            "descriptor_dimension",
            "preprocessing_id",
            "device",
            "smoke_input_sha256",
            "descriptor_l2_norm",
            "smoke_verified",
        }
        if not isinstance(value, dict) or set(value) != required_keys:
            raise ValueError("MegaLoc smoke receipt fields are invalid")
        if (
            value.get("schema") != "atlaslens-megaloc-production-smoke-v1"
            or value.get("provider") != MEGALOC_PROVIDER_ID
            or value.get("smoke_verified") is not True
            or value.get("descriptor_dimension") != MEGALOC_DESCRIPTOR_DIMENSION
            or value.get("device") not in {"cpu", "cuda"}
        ):
            raise ValueError("MegaLoc smoke receipt contract is invalid")
        text_fields = (
            "artifact_sha256",
            "source_revision",
            "model_revision",
            "descriptor_version",
            "preprocessing_id",
            "smoke_input_sha256",
        )
        if any(not isinstance(value.get(field), str) for field in text_fields):
            raise ValueError("MegaLoc smoke receipt text is invalid")
        norm = value.get("descriptor_l2_norm")
        if isinstance(norm, bool) or not isinstance(norm, int | float):
            raise ValueError("MegaLoc smoke norm is invalid")
        return cls(
            artifact_sha256=str(value["artifact_sha256"]),
            source_revision=str(value["source_revision"]),
            model_revision=str(value["model_revision"]),
            descriptor_version=str(value["descriptor_version"]),
            descriptor_dimension=8448,
            preprocessing_id=str(value["preprocessing_id"]),
            device=cast(ResolvedMegaLocDevice, value["device"]),
            smoke_input_sha256=str(value["smoke_input_sha256"]),
            descriptor_l2_norm=float(norm),
        )


def load_megaloc_config(path: Path, *, project_root: Path) -> MegaLocAdapterConfig:
    """Load a strict repository-relative configuration without environment access."""

    root = project_root.resolve()
    requested_path = path if path.is_absolute() else root / path
    config_path = _contained_file(
        root, requested_path.resolve(), "MEGALOC_CONFIG_MISSING"
    )
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MegaLocAdapterError("MEGALOC_CONFIG_INVALID") from exc
    if not isinstance(value, dict) or value.get("schema") != "atlaslens-megaloc-phase3b2-v1":
        raise MegaLocAdapterError("MEGALOC_CONFIG_INVALID")
    payload = cast(dict[str, Any], value)
    try:
        paths = _object(payload, "paths")
        source = _object(payload, "source")
        artifact = _object(payload, "artifact")
        licenses = _object(payload, "licenses")
        descriptor = _object(payload, "descriptor")
        execution = _object(payload, "execution")
        approval_value = _object(payload, "approval")
        approval = MegaLocApproval(
            artifact_hash_approved=_boolean(approval_value, "artifact_hash_approved"),
            source_code_approved=_boolean(approval_value, "source_code_approved"),
            source_license_approved=_boolean(approval_value, "source_license_approved"),
            weight_license_approved=_boolean(approval_value, "weight_license_approved"),
            descriptor_contract_approved=_boolean(
                approval_value, "descriptor_contract_approved"
            ),
            preprocessing_approved=_boolean(approval_value, "preprocessing_approved"),
            production_use_approved=_boolean(approval_value, "production_use_approved"),
            approval_record_id=_text(approval_value, "approval_record_id"),
            weight_license_record_id=_text(approval_value, "weight_license_record_id"),
        )
        return MegaLocAdapterConfig(
            project_root=root,
            source_dir=_relative_path(root, paths, "source_dir"),
            source_code_path=_relative_path(root, paths, "source_code_path"),
            source_code_sha256=_text(source, "code_sha256"),
            source_revision=_text(source, "revision"),
            source_license_path=_relative_path(root, paths, "source_license_path"),
            source_license_sha256=_text(licenses, "source_license_sha256"),
            source_license_spdx=_text(licenses, "source_license_spdx"),
            artifact_path=_relative_path(root, paths, "artifact_path"),
            artifact_sha256=_text(artifact, "sha256"),
            artifact_size=_integer(artifact, "size"),
            artifact_format=_safetensors_format(artifact),
            receipt_path=_relative_path(root, paths, "receipt_path"),
            receipt_sha256=_text(artifact, "receipt_sha256"),
            smoke_receipt_path=_relative_path(root, paths, "smoke_receipt_path"),
            model_id=_text(artifact, "model_id"),
            model_revision=_text(artifact, "model_revision"),
            weight_license_spdx=_text(licenses, "weight_license_spdx"),
            descriptor_version=_text(descriptor, "version"),
            descriptor_dimension=_dimension(descriptor),
            preprocessing_id=_text(descriptor, "preprocessing_id"),
            maximum_edge=_integer(descriptor, "maximum_edge"),
            device=_device(execution),
            max_batch_size=_integer(execution, "max_batch_size"),
            max_input_bytes=_integer(execution, "max_input_bytes"),
            max_image_pixels=_integer(execution, "max_image_pixels"),
            deterministic_algorithms=_boolean(execution, "deterministic_algorithms"),
            reduce_batch_on_cuda_oom=_boolean(execution, "reduce_batch_on_cuda_oom"),
            approval=approval,
        )
    except (KeyError, TypeError, ValueError, MegaLocAdapterError) as exc:
        if isinstance(exc, MegaLocAdapterError):
            raise
        raise MegaLocAdapterError("MEGALOC_CONFIG_INVALID") from exc


def write_megaloc_smoke_receipt(path: Path, receipt: MegaLocSmokeReceipt) -> None:
    """Atomically persist a caller-authorized smoke receipt at an explicit path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(receipt.to_json()))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _contained_file(root: Path, path: Path, missing_code: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MegaLocAdapterError("MEGALOC_PATH_INVALID") from exc
    if not path.is_file() or path.is_symlink():
        raise MegaLocAdapterError(missing_code)
    return path


def _relative_path(root: Path, value: dict[str, Any], key: str) -> Path:
    raw = _text(value, key)
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise MegaLocAdapterError("MEGALOC_PATH_INVALID")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MegaLocAdapterError("MEGALOC_PATH_INVALID") from exc
    return resolved


def _object(value: dict[str, Any], key: str) -> dict[str, Any]:
    child = value[key]
    if not isinstance(child, dict):
        raise TypeError(key)
    return cast(dict[str, Any], child)


def _text(value: dict[str, Any], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip():
        raise TypeError(key)
    return item


def _integer(value: dict[str, Any], key: str) -> int:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(key)
    return cast(int, item)


def _boolean(value: dict[str, Any], key: str) -> bool:
    item = value[key]
    if not isinstance(item, bool):
        raise TypeError(key)
    return item


def _dimension(value: dict[str, Any]) -> Literal[8448]:
    if _integer(value, "dimension") != MEGALOC_DESCRIPTOR_DIMENSION:
        raise ValueError("dimension")
    return 8448


def _safetensors_format(value: dict[str, Any]) -> Literal["safetensors"]:
    if _text(value, "format") != "safetensors":
        raise ValueError("format")
    return "safetensors"


def _device(value: dict[str, Any]) -> MegaLocDevice:
    device = _text(value, "device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device")
    return cast(MegaLocDevice, device)


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
