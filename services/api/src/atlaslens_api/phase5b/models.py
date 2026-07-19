from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.fusion import CandidateBatch
from atlaslens_api.schemas import (
    MapConstraintSummary,
    ModelPredictionDiagnostics,
    Phase5BAssessment,
    Phase5BDiagnostics,
    PlaceEvidenceSummary,
    Provenance,
    RetrievalMatchSummary,
)

SourceKind = Literal[
    "global_model",
    "ocr_place",
    "visual_retrieval",
    "landmark_research",
    "visual_clue",
    "gazetteer",
]


class _Phase5BModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class EvidenceContradiction(_Phase5BModel):
    reason_code: str = Field(
        min_length=1, max_length=120, pattern=r"^[a-z0-9_.-]+$"
    )
    strength: float = Field(ge=0, le=1)

    @field_validator("strength")
    @classmethod
    def finite_strength(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("contradiction strength must be finite")
        return value


class EvidenceHypothesis(_Phase5BModel):
    id: str = Field(min_length=1, max_length=128)
    provider_id: str = Field(min_length=1, max_length=80)
    source_id: str = Field(min_length=1, max_length=160)
    source_kind: SourceKind
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    raw_score: float = Field(ge=0, le=1)
    score_semantics: str = Field(min_length=1, max_length=120)
    uncertainty_radius_km: float = Field(gt=0, le=20_050)
    original_rank: int = Field(ge=1, le=1_000_000)
    capture_family_id: str | None = Field(default=None, min_length=1, max_length=160)
    content_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=24)
    provenance: tuple[Provenance, ...] = Field(min_length=1, max_length=16)
    place_matches: tuple[PlaceEvidenceSummary, ...] = Field(default=(), max_length=12)
    retrieval_matches: tuple[RetrievalMatchSummary, ...] = Field(default=(), max_length=16)
    map_observations: tuple[MapConstraintSummary, ...] = Field(default=(), max_length=32)
    supports: tuple[str, ...] = Field(default=(), max_length=24)
    contradictions: tuple[EvidenceContradiction, ...] = Field(default=(), max_length=24)
    model_prediction: ModelPredictionDiagnostics | None = None

    @field_validator("latitude", "longitude", "raw_score", "uncertainty_radius_km")
    @classmethod
    def finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("phase5b hypothesis numbers must be finite")
        return value

    @field_validator("supports")
    @classmethod
    def unique_supports(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("support reason codes must be unique")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("evidence ids must be unique")
        return value


class Phase5BCluster(_Phase5BModel):
    id: str = Field(min_length=1, max_length=128)
    rank: int = Field(ge=1)
    original_best_rank: int = Field(ge=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_km: float = Field(gt=0)
    uncertainty_basis: Literal["phase5b.evidence_dispersion_and_source_floor"] = (
        "phase5b.evidence_dispersion_and_source_floor"
    )
    member_ids: tuple[str, ...] = Field(min_length=1)
    suppressed_member_count: int = Field(ge=0)
    assessment: Phase5BAssessment


class Phase5BEngineResult(_Phase5BModel):
    clusters: tuple[Phase5BCluster, ...]
    abstained: bool
    abstention_reason: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def valid_abstention(self) -> Phase5BEngineResult:
        if self.abstained != (self.abstention_reason is not None):
            raise ValueError("phase5b abstention flag and reason must agree")
        return self

    @property
    def viable_clusters(self) -> tuple[Phase5BCluster, ...]:
        return tuple(
            cluster
            for cluster in self.clusters
            if cluster.assessment.classification != "contradicted"
        )


@dataclass(frozen=True, slots=True)
class Phase5BIntegrationResult:
    batch: CandidateBatch
    diagnostics: Phase5BDiagnostics
    engine_result: Phase5BEngineResult
