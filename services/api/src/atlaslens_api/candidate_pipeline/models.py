from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlaslens_api.reranking.models import ScoreBreakdown


class PipelineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SignalClassification(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


class GeometryClassification(StrEnum):
    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


class FinalClassification(StrEnum):
    RETRIEVAL_ONLY = "retrieval_only"
    MAP_SUPPORTED = "map_supported"
    GEOMETRY_SUPPORTED = "geometry_supported"
    MULTI_SOURCE_SUPPORTED = "multi_source_supported"
    CONTRADICTED = "contradicted"
    ABSTAINED = "abstained"


class MapConstraintSignal(PipelineModel):
    classification: SignalClassification
    strength: float = Field(ge=0, le=1)
    provider_id: str = Field(min_length=1, max_length=80)
    provider_version: str = Field(min_length=1, max_length=80)
    observation_code: str = Field(min_length=1, max_length=120)
    provenance: tuple[str, ...] = Field(min_length=1)


class GeometrySignal(PipelineModel):
    reference_hit_id: UUID
    classification: GeometryClassification
    strength: float = Field(ge=0, le=1)
    provider_id: str = Field(min_length=1, max_length=80)
    provider_version: str = Field(min_length=1, max_length=80)
    reason_code: str = Field(min_length=1, max_length=120)
    provenance: tuple[str, ...] = Field(min_length=1)


class Contradiction(PipelineModel):
    source: Literal["map", "geometry"]
    reason_code: str = Field(min_length=1, max_length=120)
    strength: float = Field(gt=0, le=1)


class ReferenceAttribution(PipelineModel):
    hit_id: UUID
    source: str = Field(min_length=1, max_length=500)
    license: str = Field(min_length=1, max_length=500)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class FinalCandidateAssessment(PipelineModel):
    hypothesis_id: str
    classification: FinalClassification
    relative_rank_score: float = Field(ge=0, le=1)
    score_semantics: Literal["uncalibrated_relative_rank"] = "uncalibrated_relative_rank"
    reranker_version: Literal["phase4-v1"] = "phase4-v1"
    score_breakdown: tuple[ScoreBreakdown, ...]
    source_diversity: int = Field(gt=0)
    contributing_hit_ids: tuple[UUID, ...] = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_km: float = Field(gt=0)
    map_observations: tuple[MapConstraintSignal, ...]
    geometry_results: tuple[GeometrySignal, ...]
    contradictions: tuple[Contradiction, ...]
    reference_attributions: tuple[ReferenceAttribution, ...]
    provenance: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_verification_semantics(self) -> FinalCandidateAssessment:
        if self.classification == FinalClassification.CONTRADICTED and not self.contradictions:
            raise ValueError("contradicted assessments require contradiction evidence")
        return self


class PipelineProgress(PipelineModel):
    stage: Literal["reranking", "map_constraints", "geometry", "assessment", "completed"]
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    hypothesis_id: str | None = None


class PartialFailure(PipelineModel):
    code: str = Field(min_length=1, max_length=120)
    hypothesis_id: str | None = None
    reference_hit_id: UUID | None = None


class PipelineOutcome(PipelineModel):
    assessments: tuple[FinalCandidateAssessment, ...]
    abstention_reason: str | None = None
    partial_failures: tuple[PartialFailure, ...]
    processed_candidates: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_abstention(self) -> PipelineOutcome:
        accepted = [
            item
            for item in self.assessments
            if item.classification
            not in {FinalClassification.CONTRADICTED, FinalClassification.ABSTAINED}
        ]
        if not accepted and self.abstention_reason is None:
            raise ValueError("outcome without a viable assessment must abstain")
        if accepted and self.abstention_reason is not None:
            raise ValueError("viable assessments cannot also abstain")
        return self
