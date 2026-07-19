from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from atlaslens_api.providers.base import GlobalPredictionHypothesis
from atlaslens_api.reranking.geo import geodesic_km, percentile, spherical_center
from atlaslens_api.schemas import GeoClipClusterSummary


@dataclass(frozen=True, slots=True)
class GeoClipCandidateCluster:
    """One geodesic cluster of correlated GeoCLIP gallery hypotheses."""

    summary: GeoClipClusterSummary
    latitude: float
    longitude: float
    dispersion_km: float
    members: tuple[GlobalPredictionHypothesis, ...]


class GeoClipCandidateClusterer:
    """Small deterministic connected-component clusterer for bounded Top-K output.

    Cluster support is an uncalibrated relative ordering feature. It combines
    best/mean gallery similarity, cluster density and reciprocal-rank support;
    it is never a location probability or confidence.
    """

    version = "phase6a-geoclip-clustering-v1"

    def __init__(self, *, radius_km: float = 40.0, max_clusters: int = 50) -> None:
        if not math.isfinite(radius_km) or not 0 < radius_km <= 2_000:
            raise ValueError("cluster radius must be positive and finite")
        if not 1 <= max_clusters <= 100:
            raise ValueError("max clusters must be between 1 and 100")
        self._radius_km = radius_km
        self._max_clusters = max_clusters

    def cluster(
        self, hypotheses: Sequence[GlobalPredictionHypothesis]
    ) -> tuple[GeoClipCandidateCluster, ...]:
        ordered = sorted(hypotheses, key=lambda item: (item.original_rank, item.rank))
        if not ordered:
            return ()
        parents = list(range(len(ordered)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        for left in range(len(ordered)):
            left_point = (ordered[left].latitude, ordered[left].longitude)
            for right in range(left + 1, len(ordered)):
                right_point = (ordered[right].latitude, ordered[right].longitude)
                if geodesic_km(left_point, right_point) <= self._radius_km:
                    union(left, right)

        grouped: dict[int, list[GlobalPredictionHypothesis]] = defaultdict(list)
        for index, hypothesis in enumerate(ordered):
            grouped[find(index)].append(hypothesis)

        score_ceiling = max(item.raw_score for item in ordered)
        reciprocal_rank_total = sum(1.0 / item.original_rank for item in ordered)
        clusters = [
            self._build(
                members,
                total_count=len(ordered),
                score_ceiling=score_ceiling,
                reciprocal_rank_total=reciprocal_rank_total,
            )
            for members in grouped.values()
        ]
        clusters.sort(
            key=lambda item: (
                -item.summary.cluster_support,
                min(item.summary.member_ranks),
                item.summary.cluster_id,
            )
        )
        return tuple(clusters[: self._max_clusters])

    def _build(
        self,
        members: list[GlobalPredictionHypothesis],
        *,
        total_count: int,
        score_ceiling: float,
        reciprocal_rank_total: float,
    ) -> GeoClipCandidateCluster:
        members.sort(key=lambda item: (item.original_rank, item.rank))
        center = spherical_center([(item.latitude, item.longitude) for item in members])
        distances = [geodesic_km(center, (item.latitude, item.longitude)) for item in members]
        raw_scores = [item.raw_score for item in members]
        maximum = max(raw_scores)
        mean = sum(raw_scores) / len(raw_scores)
        safe_ceiling = max(score_ceiling, 1e-12)
        best_support = min(1.0, maximum / safe_ceiling)
        mean_support = min(1.0, mean / safe_ceiling)
        density_support = len(members) / total_count
        rank_support = (
            sum(1.0 / item.original_rank for item in members) / reciprocal_rank_total
            if reciprocal_rank_total > 0
            else 0.0
        )
        support = min(
            1.0,
            0.45 * best_support
            + 0.20 * mean_support
            + 0.20 * density_support
            + 0.15 * rank_support,
        )
        stable_payload = "|".join(
            f"{item.original_rank}:{item.latitude:.7f}:{item.longitude:.7f}" for item in members
        ).encode("utf-8")
        cluster_id = f"geoclip-{hashlib.sha256(stable_payload).hexdigest()[:24]}"
        first = members[0]
        return GeoClipCandidateCluster(
            summary=GeoClipClusterSummary(
                cluster_id=cluster_id,
                member_count=len(members),
                member_ranks=[item.original_rank for item in members],
                max_raw_similarity=maximum,
                mean_raw_similarity=mean,
                raw_score_type=first.score_type,
                cluster_support=support,
            ),
            latitude=center[0],
            longitude=center[1],
            dispersion_km=percentile(distances, 0.80) if distances else 0.0,
            members=tuple(members),
        )
