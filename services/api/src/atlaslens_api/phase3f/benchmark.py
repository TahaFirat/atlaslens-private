"""Calibration, aggregate benchmark metrics, and preregistered Phase 3F gates."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

Outcome = Literal["COMPLETE_BENCHMARK_ONLY", "COMPLETE_ACCEPTED_PRIVATE_PILOT"]
Role = Literal["calibration", "sealed_holdout", "ood_holdout"]

DISTANCE_THRESHOLDS_KM: Final = (1.0, 25.0, 50.0, 100.0, 250.0)
RECALL_RANKS: Final = (1, 5, 10)


@dataclass(frozen=True, slots=True)
class RetrievalSignals:
    similarity: float
    top1_top2_margin: float
    reference_density: int

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.similarity, self.top1_top2_margin)):
            raise ValueError("retrieval signals must be finite")
        if self.reference_density < 0:
            raise ValueError("reference density must be non-negative")


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    signals: RetrievalSignals
    city_correct: bool


@dataclass(frozen=True, slots=True)
class AbstentionThresholdLock:
    selection_lock_sha256: str
    split_lock_sha256: str
    minimum_similarity: float
    minimum_margin: float
    minimum_reference_density: int
    calibration_count: int
    calibration_coverage: float
    accepted_city_accuracy: float

    def accepts(self, signals: RetrievalSignals) -> bool:
        return (
            signals.similarity >= self.minimum_similarity
            and signals.top1_top2_margin >= self.minimum_margin
            and signals.reference_density >= self.minimum_reference_density
        )

    def document(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-phase3f-abstention-threshold-v1",
            "fit_split": "calibration_only",
            "holdout_observed_before_lock": False,
            "selection_lock_sha256": self.selection_lock_sha256,
            "split_lock_sha256": self.split_lock_sha256,
            "minimum_similarity": self.minimum_similarity,
            "minimum_top1_top2_margin": self.minimum_margin,
            "minimum_reference_density": self.minimum_reference_density,
            "calibration_count": self.calibration_count,
            "calibration_coverage": self.calibration_coverage,
            "accepted_city_accuracy": self.accepted_city_accuracy,
            "similarity_semantics": "uncalibrated_cosine_similarity_not_probability",
            "confidence": None,
        }

    @property
    def lock_sha256(self) -> str:
        return _canonical_sha256(self.document())


@dataclass(frozen=True, slots=True)
class RetrievalObservation:
    opaque_query_id: str
    role: Role
    city: str
    province: str
    predicted_cities: tuple[str, ...]
    predicted_provinces: tuple[str, ...]
    relevant_reference_rank: int | None
    geodesic_error_km: float | None
    signals: RetrievalSignals
    contributor_group_sha256: str
    sequence_group_sha256: str
    failure_category: str | None = None

    def __post_init__(self) -> None:
        if len(self.predicted_cities) < 3 or len(self.predicted_provinces) < 3:
            raise ValueError("top-three city and province predictions are required")
        if self.relevant_reference_rank is not None and self.relevant_reference_rank <= 0:
            raise ValueError("relevant reference rank must be positive")
        if self.geodesic_error_km is not None and (
            not math.isfinite(self.geodesic_error_km) or self.geodesic_error_km < 0
        ):
            raise ValueError("geodesic error must be finite and non-negative")
        for value in (self.contributor_group_sha256, self.sequence_group_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("group identifiers must be SHA-256")
        if self.failure_category is not None and self.failure_category not in {
            "visual_aliasing",
            "low_information",
            "coverage_gap",
            "season_or_viewpoint_shift",
            "other_redacted",
        }:
            raise ValueError("failure category must be redacted and preregistered")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    outcome: Outcome
    metrics: dict[str, object]
    gates: dict[str, bool]
    threshold_lock_sha256: str

    def document(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-phase3f-aggregate-benchmark-v1",
            "outcome": self.outcome,
            "threshold_lock_sha256": self.threshold_lock_sha256,
            "metrics": self.metrics,
            "activation_gates": self.gates,
            "raw_coordinates_included": False,
            "raw_paths_included": False,
            "confidence": None,
            "similarity_semantics": "uncalibrated_cosine_similarity_not_probability",
        }

    @property
    def result_sha256(self) -> str:
        return _canonical_sha256(self.document())


def fit_abstention_threshold(
    observations: Sequence[CalibrationObservation],
    *,
    selection_lock_sha256: str,
    split_lock_sha256: str,
    minimum_accepted_accuracy: float = 0.70,
    minimum_coverage: float = 0.25,
) -> AbstentionThresholdLock:
    """Fit only on calibration data; maximize coverage subject to preregistered accuracy."""

    if not observations:
        raise ValueError("calibration observations are required")
    if not 0.0 <= minimum_coverage <= 1.0 or not 0.0 <= minimum_accepted_accuracy <= 1.0:
        raise ValueError("calibration targets must be rates")
    similarities = sorted({item.signals.similarity for item in observations})
    margins = sorted({item.signals.top1_top2_margin for item in observations})
    densities = sorted({item.signals.reference_density for item in observations})
    candidates: list[tuple[float, float, int, float, float]] = []
    for similarity in similarities:
        for margin in margins:
            for density in densities:
                accepted = [
                    item
                    for item in observations
                    if item.signals.similarity >= similarity
                    and item.signals.top1_top2_margin >= margin
                    and item.signals.reference_density >= density
                ]
                coverage = len(accepted) / len(observations)
                accuracy = (
                    sum(item.city_correct for item in accepted) / len(accepted) if accepted else 0.0
                )
                if coverage >= minimum_coverage and accuracy >= minimum_accepted_accuracy:
                    candidates.append((similarity, margin, density, coverage, accuracy))
    if not candidates:
        strict_similarity = math.nextafter(max(similarities), math.inf)
        return AbstentionThresholdLock(
            selection_lock_sha256=selection_lock_sha256,
            split_lock_sha256=split_lock_sha256,
            minimum_similarity=strict_similarity,
            minimum_margin=math.nextafter(max(margins), math.inf),
            minimum_reference_density=max(densities) + 1,
            calibration_count=len(observations),
            calibration_coverage=0.0,
            accepted_city_accuracy=0.0,
        )
    selected = max(
        candidates,
        key=lambda item: (item[3], item[4], item[0], item[1], item[2]),
    )
    return AbstentionThresholdLock(
        selection_lock_sha256=selection_lock_sha256,
        split_lock_sha256=split_lock_sha256,
        minimum_similarity=selected[0],
        minimum_margin=selected[1],
        minimum_reference_density=selected[2],
        calibration_count=len(observations),
        calibration_coverage=selected[3],
        accepted_city_accuracy=selected[4],
    )


def evaluate_holdout_once(
    observations: Sequence[RetrievalObservation],
    *,
    threshold: AbstentionThresholdLock,
    selection_lock_sha256: str,
    split_lock_sha256: str,
    in_domain_cities: Sequence[str],
    ood_cities: Sequence[str],
    leakage_passed: bool,
    city_minimums_passed: bool,
    security_integrity_passed: bool,
    holdout_open_count_before: int,
) -> BenchmarkResult:
    if holdout_open_count_before != 0:
        raise RuntimeError("PHASE3F_HOLDOUT_ALREADY_OPENED")
    if threshold.selection_lock_sha256 != selection_lock_sha256:
        raise ValueError("threshold selection lock mismatch")
    if threshold.split_lock_sha256 != split_lock_sha256:
        raise ValueError("threshold split lock mismatch")
    if any(item.role == "calibration" for item in observations):
        raise ValueError("calibration rows cannot enter final holdout scoring")
    in_scope = set(in_domain_cities)
    ood_scope = set(ood_cities)
    if len(in_scope) < 5 or len(ood_scope) < 2 or in_scope & ood_scope:
        raise ValueError("benchmark city scopes are invalid")
    identity = [item for item in observations if item.role == "sealed_holdout"]
    ood = [item for item in observations if item.role == "ood_holdout"]
    if not identity or not ood:
        raise ValueError("both in-domain and OOD observations are required")
    if any(item.city not in in_scope for item in identity) or any(
        item.city not in ood_scope for item in ood
    ):
        raise ValueError("observation city is outside its locked scope")

    accepted_identity = [item for item in identity if threshold.accepts(item.signals)]
    accepted_ood = [item for item in ood if threshold.accepts(item.signals)]
    city_top1 = _rate(identity, lambda item: item.predicted_cities[0] == item.city)
    city_top3 = _rate(identity, lambda item: item.city in item.predicted_cities[:3])
    province_top1 = _rate(identity, lambda item: item.predicted_provinces[0] == item.province)
    province_top3 = _rate(identity, lambda item: item.province in item.predicted_provinces[:3])
    accepted_city_top1 = _rate(
        accepted_identity, lambda item: item.predicted_cities[0] == item.city
    )
    coverage = len(accepted_identity) / len(identity)
    abstention_rate = 1.0 - coverage
    ood_false_accept = len(accepted_ood) / len(ood)
    ood_abstention = 1.0 - ood_false_accept
    accepted_errors = [
        item.geodesic_error_km
        for item in accepted_identity
        if item.geodesic_error_km is not None
    ]
    all_errors = [
        item.geodesic_error_km for item in identity if item.geodesic_error_km is not None
    ]
    accepted_median = _median_or_none(accepted_errors)
    accepted_by_city = {
        city: sum(item.city == city for item in accepted_identity) for city in sorted(in_scope)
    }
    maximum_city_share = (
        max(accepted_by_city.values()) / len(accepted_identity) if accepted_identity else 1.0
    )
    recall = {
        f"recall_at_{rank}": _recall_at_rank(identity, rank)
        for rank in RECALL_RANKS
    }
    geographic = {
        f"within_{int(distance)}km_recall": _recall_within_distance(identity, distance)
        for distance in DISTANCE_THRESHOLDS_KM
    }
    metrics: dict[str, object] = {
        "query_count": len(identity),
        "ood_query_count": len(ood),
        "city_top1": city_top1,
        "city_top3": city_top3,
        "province_top1": province_top1,
        "province_top3": province_top3,
        **recall,
        **geographic,
        "median_geodesic_error_km": _median_or_none(all_errors),
        "p90_geodesic_error_km": _percentile_or_none(all_errors, 0.90),
        "accepted_median_geodesic_error_km": accepted_median,
        "calibration_coverage": threshold.calibration_coverage,
        "accepted_query_coverage": coverage,
        "accepted_city_top1": accepted_city_top1,
        "abstention_rate": abstention_rate,
        "ood_false_accept_rate": ood_false_accept,
        "ood_abstention_rate": ood_abstention,
        "maximum_accepted_city_share": maximum_city_share,
        "per_city": _per_city(identity, threshold),
        "contributor_breakdown": _group_breakdown(identity, "contributor_group_sha256"),
        "sequence_breakdown": _group_breakdown(identity, "sequence_group_sha256"),
        "failure_categories": _failure_categories(observations),
    }
    gates = {
        "minimum_in_domain_cities": len(in_scope) >= 5,
        "minimum_ood_cities": len(ood_scope) >= 2,
        "leakage_invariants": leakage_passed,
        "per_city_minimums": city_minimums_passed,
        "city_top1_at_least_0_50": city_top1 >= 0.50,
        "city_top3_at_least_0_75": city_top3 >= 0.75,
        "accepted_city_top1_at_least_0_70": accepted_city_top1 >= 0.70,
        "accepted_coverage_at_least_0_25": coverage >= 0.25,
        "ood_false_accept_at_most_0_15": ood_false_accept <= 0.15,
        "ood_abstention_at_least_0_70": ood_abstention >= 0.70,
        "accepted_median_error_at_most_75km": (
            accepted_median is not None and accepted_median <= 75.0
        ),
        "maximum_city_share_at_most_0_40": maximum_city_share <= 0.40,
        "secret_security_artifact_integrity": security_integrity_passed,
    }
    outcome: Outcome = (
        "COMPLETE_ACCEPTED_PRIVATE_PILOT"
        if all(gates.values())
        else "COMPLETE_BENCHMARK_ONLY"
    )
    return BenchmarkResult(
        outcome=outcome,
        metrics=metrics,
        gates=gates,
        threshold_lock_sha256=threshold.lock_sha256,
    )


def _rate(
    items: Sequence[RetrievalObservation],
    predicate: Callable[[RetrievalObservation], bool],
) -> float:
    if not items:
        return 0.0
    return sum(predicate(item) for item in items) / len(items)


def _recall_at_rank(items: Sequence[RetrievalObservation], rank: int) -> float:
    return _rate(
        items,
        lambda item: item.relevant_reference_rank is not None
        and item.relevant_reference_rank <= rank,
    )


def _recall_within_distance(
    items: Sequence[RetrievalObservation], distance_km: float
) -> float:
    return _rate(
        items,
        lambda item: item.geodesic_error_km is not None
        and item.geodesic_error_km <= distance_km,
    )


def _median_or_none(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _percentile_or_none(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def _per_city(
    items: Sequence[RetrievalObservation], threshold: AbstentionThresholdLock
) -> dict[str, object]:
    result: dict[str, object] = {}
    for city in sorted({item.city for item in items}):
        city_items = [item for item in items if item.city == city]
        accepted = [item for item in city_items if threshold.accepts(item.signals)]
        result[city] = {
            "count": len(city_items),
            "city_top1": _rate(city_items, lambda item: item.predicted_cities[0] == item.city),
            "city_top3": _rate(city_items, lambda item: item.city in item.predicted_cities[:3]),
            "accepted_count": len(accepted),
            "accepted_city_top1": _rate(
                accepted, lambda item: item.predicted_cities[0] == item.city
            ),
            "abstention_rate": 1.0 - len(accepted) / len(city_items),
        }
    return result


def _group_breakdown(items: Sequence[RetrievalObservation], field: str) -> dict[str, object]:
    grouped: dict[str, list[RetrievalObservation]] = {}
    for item in items:
        grouped.setdefault(str(getattr(item, field)), []).append(item)
    accuracies = [
        _rate(group, lambda item: item.predicted_cities[0] == item.city)
        for group in grouped.values()
    ]
    sizes = [len(group) for group in grouped.values()]
    return {
        "group_count": len(grouped),
        "minimum_group_size": min(sizes, default=0),
        "maximum_group_size": max(sizes, default=0),
        "median_group_accuracy": float(statistics.median(accuracies)) if accuracies else 0.0,
        "identifiers_published": False,
    }


def _failure_categories(items: Sequence[RetrievalObservation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if item.failure_category is not None:
            counts[item.failure_category] = counts.get(item.failure_category, 0) + 1
    return dict(sorted(counts.items()))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AbstentionThresholdLock",
    "BenchmarkResult",
    "CalibrationObservation",
    "RetrievalObservation",
    "RetrievalSignals",
    "evaluate_holdout_once",
    "fit_abstention_threshold",
]
