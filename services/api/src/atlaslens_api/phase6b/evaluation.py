from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.evaluation.metrics import (
    error_distribution,
    haversine_km,
    latency_distribution,
    ratio,
)
from atlaslens_api.evaluation.models import (
    ErrorDistribution,
    LatencyDistribution,
    RatioMetric,
)

type AblationProfile = Literal[
    "geoclip_only",
    "geoclip_osv5m",
    "geoclip_plonk",
    "all_local_models",
    "all_local_models_ocr",
    "optional_cloud_assistance",
]

ABLATION_PROFILES: tuple[AblationProfile, ...] = (
    "geoclip_only",
    "geoclip_osv5m",
    "geoclip_plonk",
    "all_local_models",
    "all_local_models_ocr",
    "optional_cloud_assistance",
)
_RADII_KM = (1, 25, 100, 750)


class EvaluationError(ValueError):
    """Safe Phase 6B evaluation input error."""


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Phase6BManifestItem(EvaluationModel):
    image: str = Field(min_length=1, max_length=500)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    country: str = Field(min_length=2, max_length=80)
    city: str | None = Field(default=None, max_length=160)
    license: str = Field(min_length=1, max_length=500)

    @field_validator("latitude", "longitude")
    @classmethod
    def finite_coordinate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("coordinate must be finite")
        return value

    @field_validator("city", mode="before")
    @classmethod
    def blank_city(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


class Phase6BPrediction(EvaluationModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    country: str | None = Field(default=None, max_length=80)
    city: str | None = Field(default=None, max_length=160)

    @field_validator("latitude", "longitude")
    @classmethod
    def finite_coordinate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("coordinate must be finite")
        return value


class Phase6BProviderObservation(EvaluationModel):
    provider: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,79}$")
    status: Literal["completed", "skipped", "disabled", "failed", "timeout"]
    latency_ms: float = Field(ge=0)

    @field_validator("latency_ms")
    @classmethod
    def finite_latency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("latency must be finite")
        return value


class Phase6BCloudObservation(EvaluationModel):
    triggered: bool = False
    cache_hit: bool = False
    estimated_cost_usd: float = Field(default=0, ge=0)

    @field_validator("estimated_cost_usd")
    @classmethod
    def finite_cost(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("estimated cost must be finite")
        return value

    @model_validator(mode="after")
    def coherent_cache_state(self) -> Phase6BCloudObservation:
        if self.cache_hit and not self.triggered:
            raise ValueError("a cache hit requires a triggered cloud review")
        return self


class Phase6BEvaluationObservation(EvaluationModel):
    profile: AblationProfile
    image: str = Field(min_length=1, max_length=500)
    predictions: tuple[Phase6BPrediction, ...] = Field(default=(), max_length=100)
    providers: tuple[Phase6BProviderObservation, ...] = Field(default=(), max_length=20)
    cloud: Phase6BCloudObservation = Field(default_factory=Phase6BCloudObservation)

    @model_validator(mode="after")
    def unique_providers(self) -> Phase6BEvaluationObservation:
        names = [item.provider for item in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("provider observations must be unique")
        return self


class Phase6BProviderMetrics(EvaluationModel):
    success_rate: RatioMetric
    latency_ms: LatencyDistribution


class Phase6BFusionDelta(EvaluationModel):
    median_error_reduction_km: float | None = None
    within_100_km_rate_delta: float | None = None
    note: Literal["descriptive_only_not_a_benchmark_claim"] = (
        "descriptive_only_not_a_benchmark_claim"
    )


class Phase6BProfileMetrics(EvaluationModel):
    profile: AblationProfile
    manifest_sample_count: int = Field(gt=0)
    observed_sample_count: int = Field(ge=0)
    complete: bool
    prediction_return: RatioMetric
    top1_error: ErrorDistribution
    country_accuracy: RatioMetric
    city_accuracy: RatioMetric
    accuracy_within_km: dict[str, RatioMetric]
    top_k_success_within_km: dict[str, RatioMetric]
    providers: dict[str, Phase6BProviderMetrics]
    openai_trigger_rate: RatioMetric
    openai_cache_hit_rate: RatioMetric
    estimated_openai_cost_usd: float = Field(ge=0)
    fusion_delta_from_geoclip: Phase6BFusionDelta | None = None


class Phase6BEvaluationReport(EvaluationModel):
    manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(gt=0)
    minimum_improvement_claim_sample_count: int = Field(gt=0)
    improvement_claim_allowed: bool
    claim_note: Literal[
        "insufficient_sample_or_incomplete_profiles_no_improvement_claim",
        "sample_gate_met_measurements_still_require_dataset_review",
    ]
    profiles: tuple[Phase6BProfileMetrics, ...]


class Phase6BEvaluationHarness:
    """Aggregate real Phase 6B run records without downloading or invoking models."""

    def load_manifest(self, manifest_path: Path) -> tuple[tuple[Phase6BManifestItem, ...], str]:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError("phase6b_manifest_unreadable") from exc
        if not isinstance(payload, list) or not payload:
            raise EvaluationError("phase6b_manifest_empty_or_invalid")
        try:
            items = tuple(Phase6BManifestItem.model_validate(item) for item in payload)
        except (TypeError, ValueError) as exc:
            raise EvaluationError("phase6b_manifest_row_invalid") from exc

        root = manifest_path.resolve().parent
        seen: set[str] = set()
        canonical: list[dict[str, object]] = []
        for item in items:
            relative = Path(item.image)
            if relative.is_absolute() or ".." in relative.parts:
                raise EvaluationError("phase6b_manifest_image_path_unsafe")
            try:
                resolved = (root / relative).resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise EvaluationError("phase6b_manifest_image_unavailable") from exc
            if resolved.is_symlink() or not resolved.is_file():
                raise EvaluationError("phase6b_manifest_image_unsafe")
            normalized = relative.as_posix()
            if normalized in seen:
                raise EvaluationError("phase6b_manifest_duplicate_image")
            seen.add(normalized)
            try:
                with Image.open(resolved) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise EvaluationError("phase6b_manifest_image_invalid") from exc
            canonical.append(item.model_dump(mode="json"))
        fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return items, fingerprint

    def load_observations(self, path: Path) -> tuple[Phase6BEvaluationObservation, ...]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError("phase6b_results_unreadable") from exc
        if isinstance(payload, dict):
            payload = payload.get("records")
        if not isinstance(payload, list):
            raise EvaluationError("phase6b_results_invalid")
        try:
            return tuple(Phase6BEvaluationObservation.model_validate(item) for item in payload)
        except (TypeError, ValueError) as exc:
            raise EvaluationError("phase6b_result_row_invalid") from exc

    def evaluate(
        self,
        manifest: tuple[Phase6BManifestItem, ...],
        observations: tuple[Phase6BEvaluationObservation, ...],
        *,
        manifest_fingerprint: str,
        minimum_claim_samples: int = 100,
    ) -> Phase6BEvaluationReport:
        if minimum_claim_samples <= 0:
            raise EvaluationError("phase6b_minimum_claim_samples_invalid")
        truth = {item.image: item for item in manifest}
        grouped: defaultdict[
            AblationProfile, list[Phase6BEvaluationObservation]
        ] = defaultdict(list)
        seen: set[tuple[AblationProfile, str]] = set()
        for observation in observations:
            if observation.image not in truth:
                raise EvaluationError("phase6b_result_image_not_in_manifest")
            key = (observation.profile, observation.image)
            if key in seen:
                raise EvaluationError("phase6b_duplicate_result")
            seen.add(key)
            grouped[observation.profile].append(observation)

        summaries: list[Phase6BProfileMetrics] = []
        for profile in ABLATION_PROFILES:
            summaries.append(self._summarize(profile, grouped[profile], truth, len(manifest)))

        baseline = summaries[0]
        summaries = [
            item
            if item.profile == "geoclip_only"
            else item.model_copy(
                update={"fusion_delta_from_geoclip": self._fusion_delta(baseline, item)}
            )
            for item in summaries
        ]
        complete_profiles = all(item.complete for item in summaries)
        claim_allowed = len(manifest) >= minimum_claim_samples and complete_profiles
        return Phase6BEvaluationReport(
            manifest_fingerprint=manifest_fingerprint,
            sample_count=len(manifest),
            minimum_improvement_claim_sample_count=minimum_claim_samples,
            improvement_claim_allowed=claim_allowed,
            claim_note=(
                "sample_gate_met_measurements_still_require_dataset_review"
                if claim_allowed
                else "insufficient_sample_or_incomplete_profiles_no_improvement_claim"
            ),
            profiles=tuple(summaries),
        )

    @staticmethod
    def write_report(report: Phase6BEvaluationReport, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.parent.is_symlink() or not output_path.parent.is_dir():
            raise EvaluationError("phase6b_evaluation_output_unsafe")
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(report.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output_path)
        except OSError as exc:
            raise EvaluationError("phase6b_evaluation_output_failed") from exc

    @staticmethod
    def _summarize(
        profile: AblationProfile,
        observations: list[Phase6BEvaluationObservation],
        truth: dict[str, Phase6BManifestItem],
        manifest_count: int,
    ) -> Phase6BProfileMetrics:
        errors: list[float] = []
        oracle_errors: list[float] = []
        country_denominator = city_denominator = 0
        country_correct = city_correct = 0
        provider_rows: defaultdict[str, list[Phase6BProviderObservation]] = defaultdict(list)
        cloud_triggered = cloud_cache_hits = 0
        cloud_cost = 0.0

        for observation in observations:
            expected = truth[observation.image]
            for provider in observation.providers:
                provider_rows[provider.provider].append(provider)
            cloud_triggered += int(observation.cloud.triggered)
            cloud_cache_hits += int(observation.cloud.cache_hit)
            cloud_cost += observation.cloud.estimated_cost_usd
            if not observation.predictions:
                continue
            top1 = observation.predictions[0]
            errors.append(
                haversine_km(
                    expected.latitude, expected.longitude, top1.latitude, top1.longitude
                )
            )
            oracle_errors.append(
                min(
                    haversine_km(
                        expected.latitude,
                        expected.longitude,
                        prediction.latitude,
                        prediction.longitude,
                    )
                    for prediction in observation.predictions
                )
            )
            country_denominator += 1
            country_correct += int(
                top1.country is not None
                and top1.country.casefold() == expected.country.casefold()
            )
            if expected.city is not None:
                city_denominator += 1
                city_correct += int(
                    top1.city is not None and top1.city.casefold() == expected.city.casefold()
                )

        observation_count = len(observations)
        providers = {
            name: Phase6BProviderMetrics(
                success_rate=ratio(
                    sum(item.status == "completed" for item in rows), len(rows)
                ),
                latency_ms=latency_distribution(item.latency_ms for item in rows),
            )
            for name, rows in sorted(provider_rows.items())
        }
        return Phase6BProfileMetrics(
            profile=profile,
            manifest_sample_count=manifest_count,
            observed_sample_count=observation_count,
            complete=observation_count == manifest_count,
            prediction_return=ratio(len(errors), observation_count),
            top1_error=error_distribution(errors),
            country_accuracy=ratio(country_correct, country_denominator),
            city_accuracy=ratio(city_correct, city_denominator),
            accuracy_within_km={
                str(radius_km): ratio(
                    sum(error <= radius_km for error in errors), observation_count
                )
                for radius_km in _RADII_KM
            },
            top_k_success_within_km={
                str(radius_km): ratio(
                    sum(error <= radius_km for error in oracle_errors), observation_count
                )
                for radius_km in _RADII_KM
            },
            providers=providers,
            openai_trigger_rate=ratio(cloud_triggered, observation_count),
            openai_cache_hit_rate=ratio(cloud_cache_hits, cloud_triggered),
            estimated_openai_cost_usd=cloud_cost,
        )

    @staticmethod
    def _fusion_delta(
        baseline: Phase6BProfileMetrics, current: Phase6BProfileMetrics
    ) -> Phase6BFusionDelta:
        baseline_median = baseline.top1_error.median_km
        current_median = current.top1_error.median_km
        baseline_recall = baseline.accuracy_within_km["100"].value
        current_recall = current.accuracy_within_km["100"].value
        return Phase6BFusionDelta(
            median_error_reduction_km=(
                baseline_median - current_median
                if baseline_median is not None and current_median is not None
                else None
            ),
            within_100_km_rate_delta=(
                current_recall - baseline_recall
                if baseline_recall is not None and current_recall is not None
                else None
            ),
        )
