from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FEATURE_NAMES = frozenset(
    {"retrieval_similarity", "support_count", "source_diversity", "hash_diversity", "compactness"}
)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClusteringConfig(ConfigModel):
    radius_km: float = Field(gt=0, le=2_000)
    uncertainty_floor_km: float = Field(gt=0, le=1_000)
    dispersion_multiplier: float = Field(gt=0, le=10)
    max_hits_per_source: int = Field(gt=0, le=100)
    max_hypotheses: int = Field(gt=0, le=100)


class ScoringConfig(ConfigModel):
    weights: dict[str, float]
    support_normalizer: int = Field(gt=0, le=1_000)
    contradiction_penalty: float = Field(ge=0, le=1)
    contradiction_threshold: float = Field(ge=0, le=1)
    minimum_relative_score: float = Field(ge=0, le=1)
    minimum_supporting_hits: int = Field(gt=0, le=100)
    max_candidates: int = Field(gt=0, le=100)

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or set(value) - FEATURE_NAMES:
            raise ValueError("weights must contain only supported feature names")
        if any(not math.isfinite(weight) or weight < 0 for weight in value.values()):
            raise ValueError("weights must be finite and non-negative")
        if sum(value.values()) <= 0:
            raise ValueError("at least one feature weight must be positive")
        return value


class PipelineConfig(ConfigModel):
    timeout_seconds: float = Field(gt=0, le=300)
    max_candidates: int = Field(gt=0, le=100)
    max_references_per_candidate: int = Field(gt=0, le=100)


class RerankConfig(ConfigModel):
    version: Literal["phase4-v1"]
    clustering: ClusteringConfig
    scoring: ScoringConfig
    pipeline: PipelineConfig

    @model_validator(mode="after")
    def validate_candidate_bounds(self) -> RerankConfig:
        if self.pipeline.max_candidates > self.scoring.max_candidates:
            raise ValueError("pipeline candidate bound cannot exceed reranker output bound")
        return self

    @classmethod
    def from_path(cls, path: Path) -> RerankConfig:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("reranking config is unreadable or invalid JSON") from exc
        return cls.model_validate(raw)
