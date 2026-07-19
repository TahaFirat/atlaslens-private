from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

from atlaslens_api.evaluation.models import (
    BenchmarkSummary,
    CoverageErrorPoint,
    ErrorDistribution,
    GroupMetrics,
    LatencyDistribution,
    PerImageResult,
    RatioMetric,
)

_RADII = (1, 25, 200, 750, 2500)


def haversine_km(left_lat: float, left_lon: float, right_lat: float, right_lon: float) -> float:
    lat1, lat2 = math.radians(left_lat), math.radians(right_lat)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(right_lon - left_lon)
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(
        delta_lon / 2
    ) ** 2
    return 6371.0088 * 2 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1 - value)))


def ratio(numerator: int, denominator: int) -> RatioMetric:
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator if denominator else None,
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def error_distribution(values: Iterable[float | None]) -> ErrorDistribution:
    present = [value for value in values if value is not None]
    return ErrorDistribution(
        denominator=len(present),
        mean_km=statistics.fmean(present) if present else None,
        median_km=statistics.median(present) if present else None,
        p75_km=_percentile(present, 0.75),
        p90_km=_percentile(present, 0.90),
        p95_km=_percentile(present, 0.95),
    )


def latency_distribution(values: Iterable[float]) -> LatencyDistribution:
    present = list(values)
    return LatencyDistribution(
        denominator=len(present),
        mean_ms=statistics.fmean(present) if present else None,
        median_ms=statistics.median(present) if present else None,
        p95_ms=_percentile(present, 0.95),
    )


def _recall(results: Sequence[PerImageResult], field: str) -> dict[str, RatioMetric]:
    return {
        str(radius): ratio(
            sum(
                1
                for item in results
                if (value := getattr(item, field)) is not None and value <= radius
            ),
            len(results),
        )
        for radius in _RADII
    }


def _group(results: Sequence[PerImageResult]) -> GroupMetrics:
    count = len(results)
    country_eligible = [item for item in results if item.country_top1_correct is not None]
    return GroupMetrics(
        image_count=count,
        successful_inference=ratio(sum(item.failure_code is None for item in results), count),
        candidate_return=ratio(sum(item.returned_candidates > 0 for item in results), count),
        abstention=ratio(sum(item.abstained for item in results), count),
        provider_failure=ratio(sum(item.failure_code is not None for item in results), count),
        country_top1=ratio(
            sum(item.country_top1_correct is True for item in country_eligible),
            len(country_eligible),
        ),
        recall_top1=_recall(results, "top1_error_km"),
        top1_error=error_distribution(item.top1_error_km for item in results),
    )


def _breakdown(
    results: Sequence[PerImageResult], key: Callable[[PerImageResult], str]
) -> dict[str, GroupMetrics]:
    grouped: defaultdict[str, list[PerImageResult]] = defaultdict(list)
    for result in results:
        grouped[key(result)].append(result)
    return {name: _group(items) for name, items in sorted(grouped.items())}


def _coverage_curve(results: Sequence[PerImageResult]) -> tuple[CoverageErrorPoint, ...]:
    scored = sorted(
        (
            item
            for item in results
            if item.top1_raw_score is not None and item.top1_error_km is not None
        ),
        key=lambda item: (-float(item.top1_raw_score or 0), item.image_asset_key),
    )
    points: list[CoverageErrorPoint] = []
    for target in (0.1, 0.25, 0.5, 0.75, 1.0):
        selected_count = min(len(scored), math.ceil(len(results) * target))
        selected = scored[:selected_count]
        errors = [float(item.top1_error_km) for item in selected if item.top1_error_km is not None]
        points.append(
            CoverageErrorPoint(
                selected_count=selected_count,
                total_count=len(results),
                coverage=selected_count / len(results),
                score_threshold=(
                    float(selected[-1].top1_raw_score)
                    if selected and selected[-1].top1_raw_score is not None
                    else None
                ),
                mean_error_km=statistics.fmean(errors) if errors else None,
                median_error_km=statistics.median(errors) if errors else None,
            )
        )
    return tuple(points)


def summarize(
    results: Sequence[PerImageResult],
    *,
    provider_id: str,
    model_revision: str,
    manifest_fingerprint: str,
) -> BenchmarkSummary:
    if not results:
        raise ValueError("benchmark results are empty")
    count = len(results)
    country_top1_eligible = [item for item in results if item.country_top1_correct is not None]
    country_top5_eligible = [item for item in results if item.country_top5_correct is not None]
    region_eligible = [item for item in results if item.region_top1_correct is not None]
    city_eligible = [item for item in results if item.city_or_area_top1_correct is not None]
    uncertainty = [item for item in results if item.uncertainty_contains_truth is not None]
    return BenchmarkSummary(
        provider_id=provider_id,
        model_revision=model_revision,
        manifest_fingerprint=manifest_fingerprint,
        image_count=count,
        successful_inference=ratio(sum(item.failure_code is None for item in results), count),
        candidate_return=ratio(sum(item.returned_candidates > 0 for item in results), count),
        abstention=ratio(sum(item.abstained for item in results), count),
        provider_failure=ratio(sum(item.failure_code is not None for item in results), count),
        country_top1=ratio(
            sum(item.country_top1_correct is True for item in country_top1_eligible),
            len(country_top1_eligible),
        ),
        country_top5=ratio(
            sum(item.country_top5_correct is True for item in country_top5_eligible),
            len(country_top5_eligible),
        ),
        region_top1=ratio(
            sum(item.region_top1_correct is True for item in region_eligible), len(region_eligible)
        ),
        city_or_area_top1=ratio(
            sum(item.city_or_area_top1_correct is True for item in city_eligible),
            len(city_eligible),
        ),
        recall_top1=_recall(results, "top1_error_km"),
        recall_top5_oracle=_recall(results, "top5_oracle_error_km"),
        top1_error=error_distribution(item.top1_error_km for item in results),
        top5_oracle_error=error_distribution(item.top5_oracle_error_km for item in results),
        latency_ms=latency_distribution(item.latency_ms for item in results),
        uncertainty_coverage=ratio(
            sum(item.uncertainty_contains_truth is True for item in uncertainty), len(uncertainty)
        ),
        cpu_gpu_split=dict(Counter(item.device for item in results)),
        coverage_error_curve=_coverage_curve(results),
        breakdowns={
            "scene_category": _breakdown(results, lambda item: item.scene_category),
            "continent": _breakdown(results, lambda item: item.continent),
            "country": _breakdown(results, lambda item: item.country_code),
        },
        exclusions={
            "top1_error_missing": sum(item.top1_error_km is None for item in results),
            "raw_score_missing": sum(item.top1_raw_score is None for item in results),
            "country_top1_label_missing": count - len(country_top1_eligible),
            "country_top5_label_missing": count - len(country_top5_eligible),
            "region_label_missing": count - len(region_eligible),
            "city_or_area_label_missing": count - len(city_eligible),
            "uncertainty_radius_missing": count - len(uncertainty),
        },
    )
