"""Typed, model-independent corpus descriptor and provenance models."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from atlaslens_api.corpus_index.errors import DescriptorValidationError

type FloatVector = NDArray[np.float32]
type FloatMatrix = NDArray[np.float32]
type RuntimeKind = Literal["production", "test_only"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required_text(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be blank")


def _sha256(value: str, field: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class DescriptorSpec:
    """Stable identity and shape of one descriptor family."""

    provider_id: str
    version: str
    dimension: int
    metric: Literal["cosine"] = "cosine"
    artifact_sha256: str | None = None
    preprocessing_version: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.provider_id, "provider_id")
        _required_text(self.version, "version")
        if self.dimension <= 0 or self.dimension > 1_000_000:
            raise ValueError("dimension must be between 1 and 1,000,000")
        if self.artifact_sha256 is not None:
            _sha256(self.artifact_sha256, "artifact_sha256")
        if self.preprocessing_version is not None:
            _required_text(self.preprocessing_version, "preprocessing_version")

    @property
    def identity(self) -> str:
        identity = f"{self.provider_id}:{self.version}:{self.dimension}:{self.metric}"
        if self.artifact_sha256 is not None:
            identity += f":artifact={self.artifact_sha256}"
        if self.preprocessing_version is not None:
            identity += f":preprocessing={self.preprocessing_version}"
        return identity

    def to_json(self) -> dict[str, str | int]:
        value: dict[str, str | int] = {
            "provider_id": self.provider_id,
            "version": self.version,
            "dimension": self.dimension,
            "metric": self.metric,
        }
        if self.artifact_sha256 is not None:
            value["artifact_sha256"] = self.artifact_sha256
        if self.preprocessing_version is not None:
            value["preprocessing_version"] = self.preprocessing_version
        return value

    @classmethod
    def from_json(cls, value: object) -> DescriptorSpec:
        if not isinstance(value, dict):
            raise ValueError("descriptor spec must be an object")
        provider_id = value.get("provider_id")
        version = value.get("version")
        dimension = value.get("dimension")
        metric = value.get("metric")
        artifact_sha256 = value.get("artifact_sha256")
        preprocessing_version = value.get("preprocessing_version")
        if (
            not isinstance(provider_id, str)
            or not isinstance(version, str)
            or not isinstance(dimension, int)
            or metric != "cosine"
            or (artifact_sha256 is not None and not isinstance(artifact_sha256, str))
            or (
                preprocessing_version is not None
                and not isinstance(preprocessing_version, str)
            )
        ):
            raise ValueError("descriptor spec is invalid")
        return cls(
            provider_id=provider_id,
            version=version,
            dimension=dimension,
            artifact_sha256=artifact_sha256,
            preprocessing_version=preprocessing_version,
        )


@dataclass(frozen=True, slots=True)
class ProviderApproval:
    """Explicit rights review bound to one production adapter artifact."""

    provider_id: str
    version: str
    dimension: int
    artifact_sha256: str
    license_record_id: str
    rights_approved: bool

    def __post_init__(self) -> None:
        _required_text(self.provider_id, "provider_id")
        _required_text(self.version, "version")
        _required_text(self.license_record_id, "license_record_id")
        _sha256(self.artifact_sha256, "artifact_sha256")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")


@dataclass(frozen=True, slots=True)
class AssetProvenance:
    """Minimum index mapping needed for rights, retrieval, and revocation."""

    asset_id: str
    source_id: str
    rights_decision: str
    license_record_id: str
    provenance_summary: str
    content_sha256: str
    split: str
    rights_validated: bool
    revoked: bool = False
    province: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    contributor_id: str | None = None
    capture_run_id: str | None = None
    sequence_id: str | None = None
    sampling_cell: str | None = None

    def __post_init__(self) -> None:
        for field, value in (
            ("asset_id", self.asset_id),
            ("source_id", self.source_id),
            ("rights_decision", self.rights_decision),
            ("license_record_id", self.license_record_id),
            ("provenance_summary", self.provenance_summary),
            ("split", self.split),
        ):
            _required_text(value, field)
        _sha256(self.content_sha256, "content_sha256")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        if self.latitude is not None and (
            not math.isfinite(self.latitude) or not -90.0 <= self.latitude <= 90.0
        ):
            raise ValueError("latitude must be a finite WGS84 value")
        if self.longitude is not None and (
            not math.isfinite(self.longitude) or not -180.0 <= self.longitude <= 180.0
        ):
            raise ValueError("longitude must be a finite WGS84 value")

    def to_json(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "source_id": self.source_id,
            "rights_decision": self.rights_decision,
            "license_record_id": self.license_record_id,
            "provenance_summary": self.provenance_summary,
            "content_sha256": self.content_sha256,
            "split": self.split,
            "rights_validated": self.rights_validated,
            "revoked": self.revoked,
            "province": self.province,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "contributor_id": self.contributor_id,
            "capture_run_id": self.capture_run_id,
            "sequence_id": self.sequence_id,
            "sampling_cell": self.sampling_cell,
        }

    @classmethod
    def from_json(cls, value: object) -> AssetProvenance:
        if not isinstance(value, dict):
            raise ValueError("asset provenance must be an object")
        required_strings = (
            "asset_id",
            "source_id",
            "rights_decision",
            "license_record_id",
            "provenance_summary",
            "content_sha256",
            "split",
        )
        if any(not isinstance(value.get(field), str) for field in required_strings):
            raise ValueError("asset provenance has invalid required text")
        if not isinstance(value.get("rights_validated"), bool) or not isinstance(
            value.get("revoked", False), bool
        ):
            raise ValueError("asset provenance has invalid rights state")
        optional_strings = (
            "province",
            "contributor_id",
            "capture_run_id",
            "sequence_id",
            "sampling_cell",
        )
        if any(
            value.get(field) is not None and not isinstance(value.get(field), str)
            for field in optional_strings
        ):
            raise ValueError("asset provenance has invalid optional text")
        latitude = value.get("latitude")
        longitude = value.get("longitude")
        if latitude is not None and not isinstance(latitude, int | float):
            raise ValueError("asset provenance latitude is invalid")
        if longitude is not None and not isinstance(longitude, int | float):
            raise ValueError("asset provenance longitude is invalid")
        return cls(
            asset_id=str(value["asset_id"]),
            source_id=str(value["source_id"]),
            rights_decision=str(value["rights_decision"]),
            license_record_id=str(value["license_record_id"]),
            provenance_summary=str(value["provenance_summary"]),
            content_sha256=str(value["content_sha256"]),
            split=str(value["split"]),
            rights_validated=bool(value["rights_validated"]),
            revoked=bool(value.get("revoked", False)),
            province=value.get("province"),
            latitude=float(latitude) if latitude is not None else None,
            longitude=float(longitude) if longitude is not None else None,
            contributor_id=value.get("contributor_id"),
            capture_run_id=value.get("capture_run_id"),
            sequence_id=value.get("sequence_id"),
            sampling_cell=value.get("sampling_cell"),
        )


@dataclass(frozen=True, slots=True)
class DescriptorAsset:
    locator: Path
    provenance: AssetProvenance


@dataclass(frozen=True, slots=True)
class DescriptorRecord:
    provenance: AssetProvenance
    vector: FloatVector

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", normalize_vector(self.vector))


@dataclass(frozen=True, slots=True)
class SearchHit:
    rank: int
    cosine_distance: float
    reference_asset_id: str
    source_id: str
    provenance_summary: str
    province: str | None
    latitude: float | None
    longitude: float | None


def normalize_vector(value: object, *, dimension: int | None = None) -> FloatVector:
    """Return a contiguous, immutable unit vector or reject it fail closed."""

    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise DescriptorValidationError("descriptor must be numeric") from exc
    if vector.ndim != 1:
        raise DescriptorValidationError("descriptor must be one-dimensional")
    if dimension is not None and vector.shape != (dimension,):
        raise DescriptorValidationError("descriptor dimension mismatch")
    if not np.isfinite(vector).all():
        raise DescriptorValidationError("descriptor contains a non-finite value")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise DescriptorValidationError("descriptor norm must be positive and finite")
    normalized = np.ascontiguousarray(vector / norm, dtype=np.float32)
    normalized.setflags(write=False)
    return normalized


def normalize_matrix(value: object, *, rows: int, dimension: int) -> FloatMatrix:
    """Validate and L2-normalize a provider batch without accepting partial rows."""

    try:
        matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise DescriptorValidationError("descriptor batch must be numeric") from exc
    if matrix.shape != (rows, dimension):
        raise DescriptorValidationError("descriptor batch shape mismatch")
    if not np.isfinite(matrix).all():
        raise DescriptorValidationError("descriptor batch contains a non-finite value")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise DescriptorValidationError("every descriptor norm must be positive and finite")
    return np.ascontiguousarray(matrix / norms[:, np.newaxis], dtype=np.float32)
