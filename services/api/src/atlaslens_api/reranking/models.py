from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RerankModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class HypothesisClassification(StrEnum):
    MULTI_SOURCE = "multi_source"
    SINGLE_SOURCE = "single_source"
    SINGLETON = "singleton"


class RerankClassification(StrEnum):
    COMPETITIVE = "competitive"
    WEAK = "weak"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_SUPPORT = "insufficient_support"


class HypothesisMember(RerankModel):
    hit_id: UUID
    index_id: int = Field(ge=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    distance: float = Field(ge=0, le=2)
    source: str = Field(min_length=1, max_length=500)
    license: str = Field(min_length=1, max_length=500)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_provider: str = Field(min_length=1, max_length=80)
    embedding_version: str = Field(min_length=1, max_length=80)
    role: Literal["supporting", "duplicate_hash", "source_limited", "outlier"]

    @field_validator("latitude", "longitude", "distance")
    @classmethod
    def finite_numbers(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("member coordinates and distance must be finite")
        return value


class CandidateHypothesis(RerankModel):
    id: str = Field(min_length=1, max_length=100)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_km: float = Field(gt=0)
    uncertainty_basis: Literal["phase4.robust_geodesic_dispersion"]
    classification: HypothesisClassification
    hit_ids: tuple[UUID, ...] = Field(min_length=1)
    supporting_hit_ids: tuple[UUID, ...] = Field(min_length=1)
    suppressed_hit_ids: tuple[UUID, ...] = ()
    outlier_hit_ids: tuple[UUID, ...] = ()
    members: tuple[HypothesisMember, ...] = Field(min_length=1)
    source_count: int = Field(gt=0)
    hash_count: int = Field(gt=0)
    contributing_source_count: int = Field(gt=0)
    contributing_hash_count: int = Field(gt=0)

    @field_validator("latitude", "longitude", "uncertainty_radius_km")
    @classmethod
    def finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("hypothesis values must be finite")
        return value

    @model_validator(mode="after")
    def validate_membership(self) -> CandidateHypothesis:
        all_ids = set(self.hit_ids)
        if len(all_ids) != len(self.hit_ids):
            raise ValueError("hit_ids must be unique")
        member_ids = {member.hit_id for member in self.members}
        if member_ids != all_ids or len(member_ids) != len(self.members):
            raise ValueError("members must contain every hit exactly once")
        supporting = set(self.supporting_hit_ids)
        suppressed = set(self.suppressed_hit_ids)
        outliers = set(self.outlier_hit_ids)
        if supporting & suppressed or supporting & outliers or suppressed & outliers:
            raise ValueError("hit classifications must be disjoint")
        if supporting | suppressed | outliers != all_ids:
            raise ValueError("every hit must have a classification")
        return self


class ScoreBreakdown(RerankModel):
    feature: str = Field(min_length=1, max_length=80)
    raw: float = Field(ge=0, le=1)
    weight: float = Field(ge=0)
    contribution: float = Field(ge=0)
    reason: str = Field(min_length=1, max_length=200)


class RerankResult(RerankModel):
    hypothesis_id: str
    rank: int = Field(ge=1)
    relative_rank_score: float = Field(ge=0, le=1)
    score_semantics: Literal["uncalibrated_relative_rank"] = "uncalibrated_relative_rank"
    reranker_version: Literal["phase4-v1"] = "phase4-v1"
    classification: RerankClassification
    score_breakdown: tuple[ScoreBreakdown, ...]
    contradiction_strength: float = Field(ge=0, le=1)
    contradiction_penalty: float = Field(ge=0, le=1)
    source_diversity: int = Field(gt=0)
    hash_diversity: int = Field(gt=0)
    contributing_hit_ids: tuple[UUID, ...] = Field(min_length=1)


class RerankOutcome(RerankModel):
    results: tuple[RerankResult, ...]
    abstention_reason: str | None = None
    dropped_hypothesis_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_outcome(self) -> RerankOutcome:
        if not self.results and self.abstention_reason is None:
            raise ValueError("empty rerank result must abstain")
        if self.results and self.abstention_reason is not None:
            raise ValueError("ranked results cannot also abstain")
        return self
