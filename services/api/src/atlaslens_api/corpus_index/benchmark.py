"""Locked, leakage-gated retrieval benchmark metrics."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from atlaslens_api.corpus_index.artifacts import sha256_bytes, sha256_json
from atlaslens_api.corpus_index.errors import BenchmarkLeakageError, HoldoutIntegrityError
from atlaslens_api.corpus_index.index import PublishedCorpusIndex
from atlaslens_api.corpus_index.models import AssetProvenance, FloatVector, normalize_vector

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    """One locked query with hidden truth supplied only to offline evaluation."""

    query_id: str
    descriptor: FloatVector
    relevant_asset_ids: tuple[str, ...]
    content_sha256: str
    holdout_asset_id: str
    source_id: str
    contributor_id: str
    capture_run_id: str
    sequence_id: str
    sampling_cell: str
    province: str
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must not be blank")
        for field, value in (
            ("holdout_asset_id", self.holdout_asset_id),
            ("source_id", self.source_id),
            ("contributor_id", self.contributor_id),
            ("capture_run_id", self.capture_run_id),
            ("sequence_id", self.sequence_id),
            ("sampling_cell", self.sampling_cell),
            ("province", self.province),
        ):
            if not value.strip():
                raise ValueError(f"{field} must not be blank")
        if not self.relevant_asset_ids or any(
            not asset_id.strip() for asset_id in self.relevant_asset_ids
        ):
            raise ValueError("each benchmark query needs at least one relevant asset ID")
        if len(set(self.relevant_asset_ids)) != len(self.relevant_asset_ids):
            raise ValueError("relevant asset IDs must be unique")
        if not _SHA256.fullmatch(self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if not math.isfinite(self.latitude) or not -90.0 <= self.latitude <= 90.0:
            raise ValueError("benchmark latitude is invalid")
        if not math.isfinite(self.longitude) or not -180.0 <= self.longitude <= 180.0:
            raise ValueError("benchmark longitude is invalid")
        object.__setattr__(self, "descriptor", normalize_vector(self.descriptor))

    def lock_record(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "holdout_asset_id": self.holdout_asset_id,
            "descriptor_sha256": sha256_bytes(self.descriptor.tobytes(order="C")),
            "descriptor_dimension": int(self.descriptor.shape[0]),
            "relevant_asset_ids": sorted(self.relevant_asset_ids),
            "content_sha256": self.content_sha256,
            "province": self.province,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source_id": self.source_id,
            "contributor_id": self.contributor_id,
            "capture_run_id": self.capture_run_id,
            "sequence_id": self.sequence_id,
            "sampling_cell": self.sampling_cell,
            "split": "holdout",
        }


@dataclass(frozen=True, slots=True)
class LeakageViolation:
    query_id: str
    reference_asset_id: str
    kind: str


LeakageHook = Callable[
    [Sequence[AssetProvenance], Sequence[BenchmarkQuery]], Sequence[LeakageViolation]
]


@dataclass(frozen=True, slots=True)
class RecallMetrics:
    query_count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float

    def to_json(self) -> dict[str, float | int]:
        return {
            "query_count": self.query_count,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    holdout_hash: str
    query_count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    province_recall: dict[str, RecallMetrics]
    median_haversine_error_m: float | None
    p90_haversine_error_m: float | None
    geodesic_error_count: int
    abstention_coverage: float
    abstained_queries: int

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-locked-benchmark-result-v1",
            "holdout_hash": self.holdout_hash,
            "query_count": self.query_count,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "province_recall": {
                province: metrics.to_json()
                for province, metrics in sorted(self.province_recall.items())
            },
            "median_haversine_error_m": self.median_haversine_error_m,
            "p90_haversine_error_m": self.p90_haversine_error_m,
            "geodesic_error_count": self.geodesic_error_count,
            "abstention_coverage": self.abstention_coverage,
            "abstained_queries": self.abstained_queries,
            "accuracy_semantics": "synthetic or corpus-specific retrieval metrics; not confidence",
        }


def compute_locked_holdout_hash(queries: Sequence[BenchmarkQuery]) -> str:
    """Hash the full ordered benchmark contract, including descriptor bytes by digest."""

    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError("benchmark query IDs must be unique")
    return sha256_json(
        {
            "schema": "atlaslens-locked-holdout-v1",
            "queries": [
                query.lock_record() for query in sorted(queries, key=lambda item: item.query_id)
            ],
        }
    )


def verify_locked_holdout_hash(
    queries: Sequence[BenchmarkQuery], expected_hash: str
) -> None:
    if not _SHA256.fullmatch(expected_hash):
        raise HoldoutIntegrityError("expected holdout hash is invalid")
    if compute_locked_holdout_hash(queries) != expected_hash:
        raise HoldoutIntegrityError("locked holdout hash mismatch")


def detect_prebenchmark_leakage(
    references: Sequence[AssetProvenance],
    queries: Sequence[BenchmarkQuery],
    *,
    minimum_spatial_separation_m: float = 1_000.0,
) -> tuple[LeakageViolation, ...]:
    """Detect identity/lineage leakage and optional positive spatial separation."""

    _require_positive_spatial_separation(minimum_spatial_separation_m)
    violations: set[tuple[str, str, str]] = set()
    for query in queries:
        for reference in references:
            kinds: list[str] = []
            if query.content_sha256 == reference.content_sha256:
                kinds.append("content_sha256")
            if query.source_id == reference.source_id:
                kinds.append("source_id")
            for kind, query_value, reference_value in (
                ("contributor_id", query.contributor_id, reference.contributor_id),
                ("capture_run_id", query.capture_run_id, reference.capture_run_id),
                ("sequence_id", query.sequence_id, reference.sequence_id),
                ("sampling_cell", query.sampling_cell, reference.sampling_cell),
            ):
                if query_value is not None and query_value == reference_value:
                    kinds.append(kind)
            if (
                reference.latitude is not None
                and reference.longitude is not None
                and haversine_distance_m(
                    query.latitude,
                    query.longitude,
                    reference.latitude,
                    reference.longitude,
                )
                < minimum_spatial_separation_m
            ):
                kinds.append("spatial_separation")
            violations.update((query.query_id, reference.asset_id, kind) for kind in kinds)
    return tuple(
        LeakageViolation(*values)
        for values in sorted(violations, key=lambda item: (item[0], item[1], item[2]))
    )


def evaluate_locked_holdout(
    index: PublishedCorpusIndex,
    queries: Sequence[BenchmarkQuery],
    *,
    expected_holdout_hash: str,
    expected_split_lock_hash: str,
    leakage_hook: LeakageHook = detect_prebenchmark_leakage,
    abstain_if_distance_gt: float | None = None,
    minimum_spatial_separation_m: float = 1_000.0,
) -> BenchmarkResult:
    """Verify the lock and leakage gate before computing retrieval-only metrics."""

    if not queries:
        raise ValueError("locked benchmark must contain at least one query")
    _require_positive_spatial_separation(minimum_spatial_separation_m)
    verify_locked_holdout_hash(queries, expected_holdout_hash)
    if index.locked_holdout_hash != expected_holdout_hash:
        raise HoldoutIntegrityError("index was not built for this locked holdout")
    if not _SHA256.fullmatch(expected_split_lock_hash):
        raise HoldoutIntegrityError("expected split lock hash is invalid")
    if index.split_lock_hash != expected_split_lock_hash:
        raise HoldoutIntegrityError("index was not built for this split lock")
    if abstain_if_distance_gt is not None and (
        not math.isfinite(abstain_if_distance_gt) or not 0.0 <= abstain_if_distance_gt <= 2.0
    ):
        raise ValueError("abstention distance must be finite and between zero and two")
    for query in queries:
        if query.descriptor.shape != (index.spec.dimension,):
            raise HoldoutIntegrityError("holdout descriptor dimension mismatch")
    violations = list(
        detect_prebenchmark_leakage(
            index.provenance,
            queries,
            minimum_spatial_separation_m=minimum_spatial_separation_m,
        )
    )
    if leakage_hook is not detect_prebenchmark_leakage:
        violations.extend(leakage_hook(index.provenance, queries))
    if violations:
        raise BenchmarkLeakageError(
            f"pre-benchmark leakage gate rejected {len(violations)} relationship(s)"
        )

    outcomes: list[tuple[BenchmarkQuery, tuple[str, ...], bool]] = []
    errors: list[float] = []
    for query in sorted(queries, key=lambda item: item.query_id):
        hits = index.search(query.descriptor, top_k=10)
        abstained = not hits or (
            abstain_if_distance_gt is not None
            and hits[0].cosine_distance > abstain_if_distance_gt
        )
        ranked_ids = () if abstained else tuple(hit.reference_asset_id for hit in hits)
        outcomes.append((query, ranked_ids, abstained))
        if (
            not abstained
            and hits[0].latitude is not None
            and hits[0].longitude is not None
        ):
            errors.append(
                haversine_distance_m(
                    query.latitude,
                    query.longitude,
                    hits[0].latitude,
                    hits[0].longitude,
                )
            )

    recalls = _recall_metrics(outcomes)
    by_province: defaultdict[str, list[tuple[BenchmarkQuery, tuple[str, ...], bool]]] = (
        defaultdict(list)
    )
    for outcome in outcomes:
        by_province[outcome[0].province].append(outcome)
    province_recall = {
        province: _recall_metrics(province_outcomes)
        for province, province_outcomes in by_province.items()
    }
    abstained_count = sum(1 for _, _, abstained in outcomes if abstained)
    return BenchmarkResult(
        holdout_hash=expected_holdout_hash,
        query_count=len(outcomes),
        recall_at_1=recalls.recall_at_1,
        recall_at_5=recalls.recall_at_5,
        recall_at_10=recalls.recall_at_10,
        province_recall=province_recall,
        median_haversine_error_m=float(np.median(errors)) if errors else None,
        p90_haversine_error_m=float(np.percentile(errors, 90)) if errors else None,
        geodesic_error_count=len(errors),
        abstention_coverage=(len(outcomes) - abstained_count) / len(outcomes),
        abstained_queries=abstained_count,
    )


def _recall_metrics(
    outcomes: Sequence[tuple[BenchmarkQuery, tuple[str, ...], bool]],
) -> RecallMetrics:
    count = len(outcomes)
    if not count:
        return RecallMetrics(0, 0.0, 0.0, 0.0)

    def recalled_at(limit: int) -> float:
        matched = sum(
            bool(set(query.relevant_asset_ids).intersection(ranked_ids[:limit]))
            for query, ranked_ids, _ in outcomes
        )
        return matched / count

    return RecallMetrics(count, recalled_at(1), recalled_at(5), recalled_at(10))


def _require_positive_spatial_separation(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("minimum spatial separation must be finite and strictly positive")


def haversine_distance_m(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return WGS84-sphere great-circle distance in metres."""

    radius_m = 6_371_008.8
    lat_a = math.radians(latitude_a)
    lat_b = math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius_m * math.asin(min(1.0, math.sqrt(haversine)))
