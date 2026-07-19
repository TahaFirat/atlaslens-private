from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FEATURE_NAMES = frozenset(
    {
        "model_prior",
        "place_support",
        "retrieval_similarity",
        "retrieval_compactness",
        "map_support",
        "provider_diversity",
        "source_diversity",
    }
)


def default_phase5b_config_path() -> Path:
    return Path(__file__).resolve().parents[5] / "config" / "reranking" / "phase5b-v1.json"


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClusteringConfig(_ConfigModel):
    radius_km: float = Field(gt=0, le=2_000)
    max_input_hypotheses: int = Field(gt=0, le=1_000)
    max_clusters: int = Field(gt=0, le=100)


class SuppressionConfig(_ConfigModel):
    max_members_per_provider: int = Field(gt=0, le=100)
    max_members_per_source: int = Field(gt=0, le=100)
    max_members_per_capture_family: int = Field(gt=0, le=100)
    max_members_per_content_hash: int = Field(gt=0, le=100)


class ScoringConfig(_ConfigModel):
    weights: dict[str, float]
    provider_diversity_normalizer: int = Field(gt=0, le=100)
    source_diversity_normalizer: int = Field(gt=0, le=100)
    contradiction_penalty: float = Field(ge=0, le=1)
    contradiction_threshold: float = Field(ge=0, le=1)
    minimum_relative_score: float = Field(ge=0, le=1)
    max_candidates: int = Field(gt=0, le=100)

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != FEATURE_NAMES:
            raise ValueError("phase5b weights must contain every supported feature exactly once")
        if any(not math.isfinite(weight) or weight < 0 for weight in value.values()):
            raise ValueError("phase5b weights must be finite and non-negative")
        if not math.isclose(sum(value.values()), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("phase5b positive feature weights must sum to one")
        return value


class UncertaintyConfig(_ConfigModel):
    model_only_floor_km: float = Field(ge=750, le=5_000)
    place_supported_floor_km: float = Field(gt=0, le=1_000)
    retrieval_supported_floor_km: float = Field(gt=0, le=1_000)
    multi_source_floor_km: float = Field(gt=0, le=1_000)
    dispersion_multiplier: float = Field(gt=0, le=10)


class Phase5BRerankConfig(_ConfigModel):
    version: Literal["phase5b-v1"]
    clustering: ClusteringConfig
    suppression: SuppressionConfig
    scoring: ScoringConfig
    uncertainty: UncertaintyConfig

    @model_validator(mode="after")
    def validate_bounds(self) -> Phase5BRerankConfig:
        if self.scoring.max_candidates > self.clustering.max_clusters:
            raise ValueError("candidate output cannot exceed the cluster bound")
        if self.uncertainty.multi_source_floor_km > min(
            self.uncertainty.place_supported_floor_km,
            self.uncertainty.retrieval_supported_floor_km,
        ):
            raise ValueError("multi-source uncertainty floor must not exceed single-source floors")
        return self

    @classmethod
    def from_path(cls, path: Path) -> Phase5BRerankConfig:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("phase5b reranking config is unreadable or invalid JSON") from exc
        return cls.model_validate(raw)
