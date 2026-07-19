from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class EmbeddingSpec(RetrievalModel):
    provider: str = Field(min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=80)
    dimension: int = Field(gt=0, le=1_000_000)
    metric: Literal["cosine"] = "cosine"


@dataclass(frozen=True, slots=True)
class Embedding:
    spec: EmbeddingSpec
    vector: NDArray[np.float32]

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.spec.dimension:
            raise ValueError("embedding dimension does not match its provider specification")
        if not np.isfinite(vector).all():
            raise ValueError("embedding values must be finite")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("embedding norm must be positive and finite")
        normalized = np.ascontiguousarray(vector / norm, dtype=np.float32)
        normalized.setflags(write=False)
        object.__setattr__(self, "vector", normalized)


class ImageMetadata(RetrievalModel):
    image_id: UUID
    index_id: int = Field(ge=0, le=9_223_372_036_854_775_807)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    country: str | None = Field(default=None, max_length=120)
    region: str | None = Field(default=None, max_length=160)
    city: str | None = Field(default=None, max_length=160)
    source: str = Field(min_length=1, max_length=500)
    license: str = Field(min_length=1, max_length=500)
    capture_type: str = Field(min_length=1, max_length=40)
    hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_provider: str = Field(min_length=1, max_length=80)
    embedding_version: str = Field(min_length=1, max_length=80)
    asset_key: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    )
    source_record_id: str | None = Field(default=None, max_length=160)
    source_url: str | None = Field(default=None, max_length=500)
    license_url: str | None = Field(default=None, max_length=500)
    attribution: str = Field(default="attribution unavailable", min_length=1, max_length=500)
    display_allowed: bool = False
    capture_family_id: str | None = Field(default=None, max_length=160)
    coordinate_kind: Literal[
        "operator_provided", "camera_raw", "map_matched", "object", "manual", "unknown"
    ] = "operator_provided"
    coordinate_uncertainty_m: float | None = Field(default=None, gt=0)
    captured_at: str | None = Field(default=None, max_length=80)
    heading_degrees: float | None = Field(default=None, ge=0, lt=360)
    perceptual_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    terms_version: str | None = Field(default=None, max_length=80)
    continent: str | None = Field(default=None, max_length=40)
    geographic_cell: str | None = Field(default=None, max_length=100)

    @field_validator("latitude", "longitude")
    @classmethod
    def finite_coordinate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("coordinate must be finite")
        return value

    @field_validator("country", "region", "city", mode="before")
    @classmethod
    def blank_optional_text(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class RetrievalHit(RetrievalModel):
    distance: float = Field(ge=0, le=2)
    provider: EmbeddingSpec
    metadata: ImageMetadata


class ProviderDiagnostics(RetrievalModel):
    status: Literal["ready", "empty", "unavailable", "invalid"]
    index_size: int = Field(ge=0)
    embedding_provider: str
    embedding_version: str
    dimension: int = Field(ge=0)
    build_time_ms: int = Field(ge=0)
    storage_size_bytes: int = Field(ge=0)
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class IndexMatch:
    index_id: int
    distance: float


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    image_path: Path
    latitude: float
    longitude: float
    country: str | None
    region: str | None
    city: str | None
    license: str
    source: str
    content_hash: str
    asset_key: str | None = None
    source_record_id: str | None = None
    source_url: str | None = None
    license_url: str | None = None
    attribution: str = "attribution unavailable"
    display_allowed: bool = False
    capture_type: str = "user_provided"
    capture_family_id: str | None = None
    coordinate_kind: Literal[
        "operator_provided", "camera_raw", "map_matched", "object", "manual", "unknown"
    ] = "operator_provided"
    coordinate_uncertainty_m: float | None = None
    captured_at: str | None = None
    heading_degrees: float | None = None
    perceptual_hash: str | None = None
    terms_version: str | None = None
    continent: str | None = None
    geographic_cell: str | None = None


class ImportSummary(RetrievalModel):
    validated: int = Field(ge=0)
    added: int = Field(ge=0)
    duplicates: int = Field(ge=0)
    index_size: int = Field(ge=0)
