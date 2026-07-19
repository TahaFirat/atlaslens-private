from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class EvaluationRecord(EvaluationModel):
    image_asset_key: str = Field(min_length=1, max_length=128)
    local_reference: str = Field(min_length=1, max_length=500)
    true_latitude: float = Field(ge=-90, le=90)
    true_longitude: float = Field(ge=-180, le=180)
    country_code: str = Field(pattern=r"^[A-Z]{2}$")
    region: str | None = Field(default=None, max_length=160)
    city_or_area: str | None = Field(default=None, max_length=160)
    continent: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=500)
    source_record_id: str = Field(min_length=1, max_length=200)
    license: str = Field(min_length=1, max_length=500)
    attribution: str = Field(min_length=1, max_length=500)
    split: Literal["calibration", "validation", "test"]
    scene_category: str = Field(min_length=1, max_length=80)
    geographic_cell: str = Field(min_length=1, max_length=120)
    capture_family_id: str = Field(min_length=1, max_length=160)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    perceptual_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")

    @field_validator("true_latitude", "true_longitude")
    @classmethod
    def finite_coordinate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("coordinate must be finite")
        return value

    @field_validator("region", "city_or_area", mode="before")
    @classmethod
    def blank_optional(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


@dataclass(frozen=True, slots=True)
class ValidatedEvaluationAsset:
    record: EvaluationRecord
    path: Path

    def __repr__(self) -> str:
        return (
            f"ValidatedEvaluationAsset(asset_key={self.record.image_asset_key!r}, "
            "path=<redacted>)"
        )


class ManifestValidationReport(EvaluationModel):
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_count: int = Field(gt=0)
    split_counts: dict[str, int]
    continent_counts: dict[str, int]
    country_counts: dict[str, int]
    scene_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidatedManifest:
    assets: tuple[ValidatedEvaluationAsset, ...]
    report: ManifestValidationReport


class CandidatePrediction(EvaluationModel):
    rank: int = Field(ge=1, le=100)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    raw_score: float | None = None
    score_type: str = Field(min_length=1, max_length=80)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    region: str | None = Field(default=None, max_length=160)
    city_or_area: str | None = Field(default=None, max_length=160)
    uncertainty_radius_km: float | None = Field(default=None, gt=0)

    @field_validator("latitude", "longitude", "raw_score", "uncertainty_radius_km")
    @classmethod
    def finite_value(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("prediction values must be finite")
        return value


class ProviderPrediction(EvaluationModel):
    candidates: tuple[CandidatePrediction, ...] = ()
    abstained: bool = False
    failure_code: str | None = Field(default=None, max_length=120)
    latency_ms: float = Field(ge=0)
    device: Literal["cpu", "cuda", "other"]

    @model_validator(mode="after")
    def validate_outcome(self) -> ProviderPrediction:
        states = (
            int(bool(self.candidates))
            + int(self.abstained)
            + int(self.failure_code is not None)
        )
        if states != 1:
            raise ValueError("prediction must contain candidates, abstain, or fail")
        ranks = [candidate.rank for candidate in self.candidates]
        if ranks and ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("candidate ranks must be contiguous")
        return self


class EvaluationProvider(Protocol):
    provider_id: str
    model_revision: str

    def predict(self, image_path: Path) -> ProviderPrediction: ...


@runtime_checkable
class CloseableEvaluationProvider(Protocol):
    def close(self) -> None: ...


class RatioMetric(EvaluationModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_ratio(self) -> RatioMetric:
        if self.numerator > self.denominator:
            raise ValueError("metric numerator exceeds denominator")
        expected = self.numerator / self.denominator if self.denominator else None
        if self.value != expected:
            raise ValueError("metric value does not match its counts")
        return self


class ErrorDistribution(EvaluationModel):
    denominator: int = Field(ge=0)
    mean_km: float | None = Field(default=None, ge=0)
    median_km: float | None = Field(default=None, ge=0)
    p75_km: float | None = Field(default=None, ge=0)
    p90_km: float | None = Field(default=None, ge=0)
    p95_km: float | None = Field(default=None, ge=0)


class LatencyDistribution(EvaluationModel):
    denominator: int = Field(ge=0)
    mean_ms: float | None = Field(default=None, ge=0)
    median_ms: float | None = Field(default=None, ge=0)
    p95_ms: float | None = Field(default=None, ge=0)


class GroupMetrics(EvaluationModel):
    image_count: int = Field(gt=0)
    successful_inference: RatioMetric
    candidate_return: RatioMetric
    abstention: RatioMetric
    provider_failure: RatioMetric
    country_top1: RatioMetric
    recall_top1: dict[str, RatioMetric]
    top1_error: ErrorDistribution


class CoverageErrorPoint(EvaluationModel):
    selected_count: int = Field(ge=0)
    total_count: int = Field(gt=0)
    coverage: float = Field(ge=0, le=1)
    score_threshold: float | None = None
    mean_error_km: float | None = Field(default=None, ge=0)
    median_error_km: float | None = Field(default=None, ge=0)


class BenchmarkSummary(EvaluationModel):
    provider_id: str
    model_revision: str
    manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_count: int = Field(gt=0)
    successful_inference: RatioMetric
    candidate_return: RatioMetric
    abstention: RatioMetric
    provider_failure: RatioMetric
    country_top1: RatioMetric
    country_top5: RatioMetric
    region_top1: RatioMetric
    city_or_area_top1: RatioMetric
    recall_top1: dict[str, RatioMetric]
    recall_top5_oracle: dict[str, RatioMetric]
    top1_error: ErrorDistribution
    top5_oracle_error: ErrorDistribution
    latency_ms: LatencyDistribution
    uncertainty_coverage: RatioMetric
    cpu_gpu_split: dict[str, int]
    coverage_error_curve: tuple[CoverageErrorPoint, ...]
    breakdowns: dict[str, dict[str, GroupMetrics]]
    exclusions: dict[str, int]


class PerImageResult(EvaluationModel):
    image_asset_key: str
    source_record_id: str
    split: str
    scene_category: str
    continent: str
    country_code: str
    returned_candidates: int = Field(ge=0)
    abstained: bool
    failure_code: str | None
    device: str
    latency_ms: float = Field(ge=0)
    top1_raw_score: float | None
    top1_error_km: float | None = Field(default=None, ge=0)
    top5_oracle_error_km: float | None = Field(default=None, ge=0)
    country_top1_correct: bool | None
    country_top5_correct: bool | None
    region_top1_correct: bool | None
    city_or_area_top1_correct: bool | None
    uncertainty_contains_truth: bool | None
