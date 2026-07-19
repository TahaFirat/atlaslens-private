"""Locked geographic benchmark and query-only operator smoke for Phase 3B3."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from atlaslens_api.corpus_index.artifacts import atomic_write_json, sha256_json
from atlaslens_api.corpus_index.benchmark import haversine_distance_m
from atlaslens_api.corpus_index.descriptors import DescriptorDataset
from atlaslens_api.mapillary_demo.indexing import (
    MapillaryDemoSearchHit,
    MapillaryPrivateDemoDescriptorProvider,
    PrivateDemoMegaLocExecutor,
    PublishedMapillaryDemoIndex,
)
from atlaslens_api.mapillary_demo.selection import (
    POSITIVE_THRESHOLDS_M,
    LockedMapillarySplit,
    SelectedMapillaryAsset,
)


@dataclass(frozen=True, slots=True)
class GeographicRecall:
    threshold_m: int
    query_count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float

    def to_json(self) -> dict[str, int | float]:
        return {
            "threshold_m": self.threshold_m,
            "query_count": self.query_count,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
        }


@dataclass(frozen=True, slots=True)
class QueryBenchmarkDetail:
    query_id: str
    abstained: bool
    top1_geodesic_error_m: float | None
    references_within_25m: int
    references_within_100m: int
    references_within_500m: int
    references_within_1000m: int
    contributor_relation: str
    capture_year_relation: str
    failure_group: str

    def to_json(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "abstained": self.abstained,
            "top1_geodesic_error_m": self.top1_geodesic_error_m,
            "reference_density": {
                "25m": self.references_within_25m,
                "100m": self.references_within_100m,
                "500m": self.references_within_500m,
                "1000m": self.references_within_1000m,
            },
            "contributor_relation": self.contributor_relation,
            "capture_year_relation": self.capture_year_relation,
            "failure_group": self.failure_group,
        }


@dataclass(frozen=True, slots=True)
class MapillaryBenchmarkResult:
    selection_lock_sha256: str
    query_count: int
    geographic_recall: dict[str, GeographicRecall]
    median_geodesic_error_m: float | None
    p90_geodesic_error_m: float | None
    city_accuracy: float
    province_accuracy: float
    abstention_coverage: float
    abstained_queries: int
    failure_counts: dict[str, int]
    failure_examples: dict[str, tuple[str, ...]]
    contributor_groups: dict[str, dict[str, float | int]]
    capture_year_groups: dict[str, dict[str, float | int]]
    density_groups: dict[str, int]
    query_details: tuple[QueryBenchmarkDetail, ...]
    result_sha256: str

    def result_basis(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-mapillary-geographic-benchmark-v1",
            "selection_lock_sha256": self.selection_lock_sha256,
            "query_count": self.query_count,
            "geographic_recall": {
                key: value.to_json() for key, value in sorted(self.geographic_recall.items())
            },
            "median_geodesic_error_m": self.median_geodesic_error_m,
            "p90_geodesic_error_m": self.p90_geodesic_error_m,
            "city_accuracy": self.city_accuracy,
            "province_accuracy": self.province_accuracy,
            "abstention_coverage": self.abstention_coverage,
            "abstained_queries": self.abstained_queries,
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "failure_examples": {
                key: list(value) for key, value in sorted(self.failure_examples.items())
            },
            "contributor_groups": dict(sorted(self.contributor_groups.items())),
            "capture_year_groups": dict(sorted(self.capture_year_groups.items())),
            "density_groups": dict(sorted(self.density_groups.items())),
            "query_details": [detail.to_json() for detail in self.query_details],
            "semantics": (
                "locked corpus-specific geographic retrieval; no probability, "
                "nationwide accuracy or production claim"
            ),
        }

    def to_json(self) -> dict[str, object]:
        return {**self.result_basis(), "result_sha256": self.result_sha256}

    def verify(self) -> None:
        if sha256_json(self.result_basis()) != self.result_sha256:
            raise ValueError("Mapillary benchmark result checksum mismatch")


def evaluate_mapillary_holdout(
    index: PublishedMapillaryDemoIndex,
    holdout_descriptors: DescriptorDataset,
    split: LockedMapillarySplit,
    *,
    expected_selection_lock_sha256: str,
    abstain_if_cosine_distance_gt: float | None = None,
) -> MapillaryBenchmarkResult:
    """Evaluate fixed top-k geography without treating proximity as leakage."""

    split.verify()
    if split.lock_sha256 != expected_selection_lock_sha256:
        raise ValueError("locked split hash mismatch")
    if index.split_lock_sha256 != expected_selection_lock_sha256:
        raise ValueError("index was not built for this locked holdout")
    if holdout_descriptors.spec != index.index.spec:
        raise ValueError("holdout descriptor specification is incompatible")
    if holdout_descriptors.source_policy_hash != index.source_policy_sha256:
        raise ValueError("holdout source policy is incompatible")
    if abstain_if_cosine_distance_gt is not None and (
        not math.isfinite(abstain_if_cosine_distance_gt)
        or not 0.0 <= abstain_if_cosine_distance_gt <= 2.0
    ):
        raise ValueError("abstention distance must be in [0, 2]")
    vector_by_id = {
        asset.asset_id: holdout_descriptors.vectors[position]
        for position, asset in enumerate(holdout_descriptors.assets)
    }
    if set(vector_by_id) != {asset.asset_id for asset in split.holdout}:
        raise ValueError("holdout descriptor inventory does not match the locked split")

    outcomes: list[tuple[SelectedMapillaryAsset, tuple[MapillaryDemoSearchHit, ...], bool]] = []
    details: list[QueryBenchmarkDetail] = []
    errors: list[float] = []
    city_correct = 0
    province_correct = 0
    failures: defaultdict[str, list[str]] = defaultdict(list)
    contributor_outcomes: defaultdict[str, list[bool]] = defaultdict(list)
    year_outcomes: defaultdict[str, list[bool]] = defaultdict(list)
    density_groups: defaultdict[str, int] = defaultdict(int)
    references = tuple(
        record for record in index.attribution.values() if record.split == "reference"
    )
    query_by_id = {asset.asset_id: asset for asset in split.holdout}

    for query_id in sorted(query_by_id):
        query = query_by_id[query_id]
        hits = index.search(vector_by_id[query_id], top_k=10)
        abstained = not hits or (
            abstain_if_cosine_distance_gt is not None
            and hits[0].cosine_distance > abstain_if_cosine_distance_gt
        )
        outcomes.append((query, hits, abstained))
        densities = {
            int(threshold): sum(
                haversine_distance_m(
                    query.latitude,
                    query.longitude,
                    reference.latitude,
                    reference.longitude,
                )
                <= threshold
                for reference in references
            )
            for threshold in POSITIVE_THRESHOLDS_M
        }
        density_groups[_density_bucket(densities[1000])] += 1
        top1_error: float | None = None
        contributor_relation = "no_result"
        year_relation = "no_result"
        if not abstained and hits:
            top = hits[0]
            top1_error = haversine_distance_m(
                query.latitude,
                query.longitude,
                top.latitude,
                top.longitude,
            )
            errors.append(top1_error)
            city_correct += int(top.city.casefold() == query.city.casefold())
            province_correct += int(top.province_code == query.province_code)
            contributor_relation = _contributor_relation(query.creator_id, top.creator_id)
            year_relation = _year_relation(query.captured_at.year, top.captured_at)
        success_100 = top1_error is not None and top1_error <= 100.0
        contributor_outcomes[contributor_relation].append(success_100)
        year_outcomes[year_relation].append(success_100)
        failure_group = _failure_group(abstained, top1_error)
        failures[failure_group].append(query_id)
        details.append(
            QueryBenchmarkDetail(
                query_id=query_id,
                abstained=abstained,
                top1_geodesic_error_m=top1_error,
                references_within_25m=densities[25],
                references_within_100m=densities[100],
                references_within_500m=densities[500],
                references_within_1000m=densities[1000],
                contributor_relation=contributor_relation,
                capture_year_relation=year_relation,
                failure_group=failure_group,
            )
        )

    recalls = {
        str(int(threshold)): _geographic_recall(outcomes, threshold)
        for threshold in POSITIVE_THRESHOLDS_M
    }
    query_count = len(outcomes)
    if query_count == 0:
        raise ValueError("locked Mapillary holdout is empty")
    abstained_count = sum(abstained for _, _, abstained in outcomes)
    provisional = MapillaryBenchmarkResult(
        selection_lock_sha256=expected_selection_lock_sha256,
        query_count=query_count,
        geographic_recall=recalls,
        median_geodesic_error_m=float(np.median(errors)) if errors else None,
        p90_geodesic_error_m=float(np.percentile(errors, 90)) if errors else None,
        city_accuracy=city_correct / query_count,
        province_accuracy=province_correct / query_count,
        abstention_coverage=(query_count - abstained_count) / query_count,
        abstained_queries=abstained_count,
        failure_counts={key: len(value) for key, value in sorted(failures.items())},
        failure_examples={key: tuple(value[:5]) for key, value in sorted(failures.items())},
        contributor_groups=_group_metrics(contributor_outcomes),
        capture_year_groups=_group_metrics(year_outcomes),
        density_groups=dict(sorted(density_groups.items())),
        query_details=tuple(details),
        result_sha256="0" * 64,
    )
    result = MapillaryBenchmarkResult(
        selection_lock_sha256=provisional.selection_lock_sha256,
        query_count=provisional.query_count,
        geographic_recall=provisional.geographic_recall,
        median_geodesic_error_m=provisional.median_geodesic_error_m,
        p90_geodesic_error_m=provisional.p90_geodesic_error_m,
        city_accuracy=provisional.city_accuracy,
        province_accuracy=provisional.province_accuracy,
        abstention_coverage=provisional.abstention_coverage,
        abstained_queries=provisional.abstained_queries,
        failure_counts=provisional.failure_counts,
        failure_examples=provisional.failure_examples,
        contributor_groups=provisional.contributor_groups,
        capture_year_groups=provisional.capture_year_groups,
        density_groups=provisional.density_groups,
        query_details=provisional.query_details,
        result_sha256=sha256_json(provisional.result_basis()),
    )
    result.verify()
    return result


def write_benchmark_result(path: Path, result: MapillaryBenchmarkResult) -> None:
    result.verify()
    atomic_write_json(path, result.to_json())


@dataclass(frozen=True, slots=True)
class OperatorSmokeResult:
    claimed_city: str
    candidates: tuple[MapillaryDemoSearchHit, ...]
    abstained: bool
    claimed_city_appears: bool
    exact_ground_truth_available: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "claimed_city": self.claimed_city,
            "candidates": [
                {
                    "rank": hit.rank,
                    "cosine_distance": hit.cosine_distance,
                    "cosine_similarity": hit.cosine_similarity,
                    "mapillary_image_id": hit.mapillary_image_id,
                    "city": hit.city,
                    "province": hit.province,
                    "latitude": hit.latitude,
                    "longitude": hit.longitude,
                    "creator_id": hit.creator_id,
                    "captured_at": hit.captured_at,
                    "attribution_text": hit.attribution_text,
                    "source_page_url": hit.source_page_url,
                    "license_identifier": hit.license_identifier,
                    "license_url": hit.license_url,
                    "ground_truth_distance_m": None,
                }
                for hit in self.candidates
            ],
            "abstained": self.abstained,
            "claimed_city_appears": self.claimed_city_appears,
            "exact_ground_truth_available": self.exact_ground_truth_available,
            "semantics": "query-only smoke; operator city claim is not exact ground truth",
        }


def run_operator_query_smoke(
    image_path: Path,
    executor: PrivateDemoMegaLocExecutor,
    index: PublishedMapillaryDemoIndex,
    *,
    claimed_city: str,
    abstain_if_cosine_distance_gt: float | None,
) -> OperatorSmokeResult:
    """Describe one operator-authorized image without copies or persisted records."""

    if not claimed_city.strip():
        raise ValueError("claimed city must not be blank")
    if not image_path.is_file() or image_path.is_symlink():
        raise ValueError("operator image must be a regular non-symlink file")
    if image_path.stat().st_size <= 0 or image_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("operator image size is outside the private-demo bound")
    if abstain_if_cosine_distance_gt is not None and (
        not math.isfinite(abstain_if_cosine_distance_gt)
        or not 0.0 <= abstain_if_cosine_distance_gt <= 2.0
    ):
        raise ValueError("abstention distance must be in [0, 2]")
    provider = MapillaryPrivateDemoDescriptorProvider(
        executor,
        source_policy_sha256=index.source_policy_sha256,
    )
    matrix = provider.describe_batch((image_path,))
    if matrix.shape != (1, index.index.spec.dimension):
        raise ValueError("operator descriptor shape is incompatible")
    candidates = index.search(matrix[0], top_k=5)
    abstained = not candidates or (
        abstain_if_cosine_distance_gt is not None
        and candidates[0].cosine_distance > abstain_if_cosine_distance_gt
    )
    return OperatorSmokeResult(
        claimed_city=claimed_city,
        candidates=candidates,
        abstained=abstained,
        claimed_city_appears=any(
            candidate.city.casefold() == claimed_city.casefold() for candidate in candidates
        ),
    )


def _geographic_recall(
    outcomes: Sequence[tuple[SelectedMapillaryAsset, tuple[MapillaryDemoSearchHit, ...], bool]],
    threshold_m: float,
) -> GeographicRecall:
    count = len(outcomes)

    def recall_at(limit: int) -> float:
        if not count:
            return 0.0
        matches = 0
        for query, hits, abstained in outcomes:
            if abstained:
                continue
            matches += int(
                any(
                    haversine_distance_m(
                        query.latitude,
                        query.longitude,
                        hit.latitude,
                        hit.longitude,
                    )
                    <= threshold_m
                    for hit in hits[:limit]
                )
            )
        return matches / count

    return GeographicRecall(
        threshold_m=int(threshold_m),
        query_count=count,
        recall_at_1=recall_at(1),
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
    )


def _group_metrics(
    outcomes: dict[str, list[bool]] | defaultdict[str, list[bool]],
) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "query_count": len(values),
            "recall_at_1_100m": sum(values) / len(values) if values else 0.0,
        }
        for key, values in sorted(outcomes.items())
    }


def _contributor_relation(query: str | None, reference: str | None) -> str:
    if query is None or reference is None:
        return "unknown"
    return "same_contributor" if query == reference else "different_contributor"


def _year_relation(query_year: int, reference_timestamp: str) -> str:
    try:
        reference_year = datetime.fromisoformat(reference_timestamp).year
    except ValueError:
        return "unknown"
    return "same_capture_year" if query_year == reference_year else "different_capture_year"


def _failure_group(abstained: bool, top1_error: float | None) -> str:
    if abstained or top1_error is None:
        return "abstained"
    if top1_error <= 25.0:
        return "within_25m"
    if top1_error <= 100.0:
        return "within_100m"
    if top1_error <= 500.0:
        return "within_500m"
    if top1_error <= 1_000.0:
        return "within_1000m"
    return "beyond_1000m"


def _density_bucket(count: int) -> str:
    if count == 0:
        return "zero_references_within_1000m"
    if count <= 5:
        return "one_to_five_references_within_1000m"
    if count <= 20:
        return "six_to_twenty_references_within_1000m"
    return "more_than_twenty_references_within_1000m"


__all__ = [
    "GeographicRecall",
    "MapillaryBenchmarkResult",
    "OperatorSmokeResult",
    "QueryBenchmarkDetail",
    "evaluate_mapillary_holdout",
    "run_operator_query_smoke",
    "write_benchmark_result",
]
