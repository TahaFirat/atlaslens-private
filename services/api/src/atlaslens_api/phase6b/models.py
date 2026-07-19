from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type GeographicProviderState = Literal["completed", "skipped", "disabled", "failed", "timeout"]
type GeographicScoreSemantics = Literal["similarity", "direct_regression", "sample_density"]
type GeographicSourceFamily = Literal[
    "mp16_family",
    "osv5m_family",
    "yfcc_family",
    "inat_family",
    "textual_evidence_family",
    "cloud_reasoning_family",
]
type BoundedDiagnosticValue = str | int | float | bool | None

_SAFE_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,119}$")


class Phase6BModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def normalize_longitude(value: float) -> float:
    """Normalize a finite longitude to the canonical [-180, 180) interval."""

    if not math.isfinite(value):
        raise ValueError("longitude must be finite")
    return ((value + 180.0) % 360.0) - 180.0


class GeographicCandidate(Phase6BModel):
    candidate_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9._-]+$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float
    raw_score: float | None = None
    provider_rank: int = Field(ge=1, le=100)
    sample_support: int = Field(default=1, ge=1, le=4096)
    metadata: dict[str, BoundedDiagnosticValue] = Field(default_factory=dict, max_length=16)

    @field_validator("latitude", "raw_score")
    @classmethod
    def finite_values(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("candidate values must be finite")
        return value

    @field_validator("longitude", mode="before")
    @classmethod
    def canonical_longitude(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("longitude must be numeric")
        return normalize_longitude(float(value))

    @field_validator("metadata")
    @classmethod
    def bounded_metadata(
        cls, value: dict[str, BoundedDiagnosticValue]
    ) -> dict[str, BoundedDiagnosticValue]:
        for key, item in value.items():
            if not _SAFE_IDENTIFIER.fullmatch(key):
                raise ValueError("candidate metadata keys must be safe identifiers")
            if isinstance(item, str) and len(item) > 240:
                raise ValueError("candidate metadata strings are bounded")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("candidate metadata numbers must be finite")
        return value


class GeographicProviderResult(Phase6BModel):
    """AtlasLens-owned normalized boundary for every geographic model.

    Raw provider scores are retained only with their declared semantics. They are
    never promoted to confidence and Phase 6B fusion does not compare them across
    providers.
    """

    provider: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9._-]+$")
    model_id: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    source_family: GeographicSourceFamily
    status: GeographicProviderState
    device: Literal["cuda", "cpu"]
    duration_ms: int = Field(ge=0)
    score_semantics: GeographicScoreSemantics
    candidates: tuple[GeographicCandidate, ...] = Field(default=(), max_length=100)
    warnings: tuple[str, ...] = Field(default=(), max_length=20)
    reason_code: str | None = Field(
        default=None, min_length=1, max_length=120, pattern=r"^[a-z0-9][a-z0-9._-]+$"
    )
    diagnostics: dict[str, BoundedDiagnosticValue] = Field(default_factory=dict, max_length=20)

    @field_validator("warnings")
    @classmethod
    def bounded_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(len(item) > 160 or not _SAFE_IDENTIFIER.fullmatch(item) for item in value):
            raise ValueError("provider warnings must be bounded safe identifiers")
        return value

    @field_validator("diagnostics")
    @classmethod
    def bounded_diagnostics(
        cls, value: dict[str, BoundedDiagnosticValue]
    ) -> dict[str, BoundedDiagnosticValue]:
        return GeographicCandidate.bounded_metadata(value)

    @model_validator(mode="after")
    def coherent_outcome(self) -> GeographicProviderResult:
        if self.status == "completed":
            if not self.candidates or self.reason_code is not None:
                raise ValueError("completed providers require candidates and no failure reason")
            ranks = [item.provider_rank for item in self.candidates]
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError("provider ranks must be contiguous")
            if len({item.candidate_id for item in self.candidates}) != len(self.candidates):
                raise ValueError("candidate ids must be unique")
        else:
            if self.candidates:
                raise ValueError("non-completed providers cannot expose candidates")
            if self.reason_code is None:
                raise ValueError("non-completed providers require a safe reason code")
        return self


class GeographicProviderCapability(Phase6BModel):
    provider: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9._-]+$")
    available: bool
    state: Literal["ready", "disabled", "not_installed", "unavailable", "failed"]
    reason_code: str | None = Field(
        default=None, min_length=1, max_length=120, pattern=r"^[a-z0-9][a-z0-9._-]+$"
    )
    model_id: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    source_revision: str = Field(min_length=1, max_length=160)
    device: Literal["cuda", "cpu"]
    offline: Literal[True] = True
    dataset_required_for_inference: Literal[False] = False

    @model_validator(mode="after")
    def coherent_capability(self) -> GeographicProviderCapability:
        if self.available != (self.state == "ready"):
            raise ValueError("only a ready capability may be available")
        if self.available and self.reason_code is not None:
            raise ValueError("ready capability cannot have an unavailable reason")
        if not self.available and self.reason_code is None:
            raise ValueError("unavailable capability requires a reason")
        return self
