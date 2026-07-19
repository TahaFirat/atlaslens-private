from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

from atlaslens_api.reranking.config import ClusteringConfig
from atlaslens_api.reranking.geo import geodesic_km, median, percentile, spherical_center
from atlaslens_api.reranking.models import (
    CandidateHypothesis,
    HypothesisClassification,
    HypothesisMember,
)
from atlaslens_api.retrieval.models import RetrievalHit


class CandidateHypothesisBuilder:
    def __init__(self, config: ClusteringConfig) -> None:
        self._config = config

    def build(self, hits: Sequence[RetrievalHit]) -> list[CandidateHypothesis]:
        unique = self._unique_hits(hits)
        if not unique:
            return []
        components = self._components(unique)
        hypotheses = [self._hypothesis(component) for component in components]
        hypotheses.sort(key=lambda item: (-len(item.supporting_hit_ids), item.id))
        return hypotheses[: self._config.max_hypotheses]

    @staticmethod
    def _unique_hits(hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
        ordered = sorted(hits, key=lambda hit: (hit.distance, str(hit.metadata.image_id)))
        unique: dict[object, RetrievalHit] = {}
        for hit in ordered:
            unique.setdefault(hit.metadata.image_id, hit)
        return list(unique.values())

    def _components(self, hits: list[RetrievalHit]) -> list[list[RetrievalHit]]:
        parents = list(range(len(hits)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        for left in range(len(hits)):
            for right in range(left + 1, len(hits)):
                left_point = (hits[left].metadata.latitude, hits[left].metadata.longitude)
                right_point = (hits[right].metadata.latitude, hits[right].metadata.longitude)
                if geodesic_km(left_point, right_point) <= self._config.radius_km:
                    union(left, right)

        grouped: dict[int, list[RetrievalHit]] = defaultdict(list)
        for index, hit in enumerate(hits):
            grouped[find(index)].append(hit)
        components = list(grouped.values())
        for component in components:
            component.sort(key=lambda hit: (hit.distance, str(hit.metadata.image_id)))
        components.sort(key=lambda group: str(min(hit.metadata.image_id for hit in group)))
        return components

    def _hypothesis(self, hits: list[RetrievalHit]) -> CandidateHypothesis:
        medoid = min(
            hits,
            key=lambda candidate: (
                sum(
                    geodesic_km(
                        (candidate.metadata.latitude, candidate.metadata.longitude),
                        (other.metadata.latitude, other.metadata.longitude),
                    )
                    for other in hits
                ),
                str(candidate.metadata.image_id),
            ),
        )
        medoid_point = (medoid.metadata.latitude, medoid.metadata.longitude)
        radial = [
            geodesic_km(medoid_point, (hit.metadata.latitude, hit.metadata.longitude))
            for hit in hits
        ]
        radial_median = median(radial)
        mad = median([abs(value - radial_median) for value in radial])
        robust_limit = min(
            self._config.radius_km,
            max(self._config.uncertainty_floor_km, radial_median + 3.0 * mad),
        )
        outlier_ids = {
            hit.metadata.image_id
            for hit, distance in zip(hits, radial, strict=True)
            if len(hits) >= 4 and distance > robust_limit
        }

        seen_hashes: set[str] = set()
        per_source: dict[str, int] = defaultdict(int)
        supporting: list[RetrievalHit] = []
        roles: dict[object, str] = {}
        for hit in hits:
            metadata = hit.metadata
            if metadata.image_id in outlier_ids:
                roles[metadata.image_id] = "outlier"
            elif metadata.hash in seen_hashes:
                roles[metadata.image_id] = "duplicate_hash"
            elif per_source[metadata.source] >= self._config.max_hits_per_source:
                roles[metadata.image_id] = "source_limited"
            else:
                roles[metadata.image_id] = "supporting"
                supporting.append(hit)
                seen_hashes.add(metadata.hash)
                per_source[metadata.source] += 1
        if not supporting:
            supporting = [medoid]
            roles[medoid.metadata.image_id] = "supporting"
            outlier_ids.discard(medoid.metadata.image_id)

        center = spherical_center(
            [(hit.metadata.latitude, hit.metadata.longitude) for hit in supporting]
        )
        dispersions = [
            geodesic_km(center, (hit.metadata.latitude, hit.metadata.longitude))
            for hit in supporting
        ]
        uncertainty = max(
            self._config.uncertainty_floor_km,
            percentile(dispersions, 0.80) * self._config.dispersion_multiplier,
        )
        contributing_sources = {hit.metadata.source for hit in supporting}
        classification = (
            HypothesisClassification.MULTI_SOURCE
            if len(contributing_sources) >= 2
            else HypothesisClassification.SINGLE_SOURCE
            if len(supporting) >= 2
            else HypothesisClassification.SINGLETON
        )
        members = tuple(
            HypothesisMember(
                hit_id=hit.metadata.image_id,
                index_id=hit.metadata.index_id,
                latitude=hit.metadata.latitude,
                longitude=hit.metadata.longitude,
                distance=hit.distance,
                source=hit.metadata.source,
                license=hit.metadata.license,
                content_hash=hit.metadata.hash,
                embedding_provider=hit.metadata.embedding_provider,
                embedding_version=hit.metadata.embedding_version,
                role=roles[hit.metadata.image_id],
            )
            for hit in hits
        )
        hit_ids = tuple(member.hit_id for member in members)
        supporting_ids = tuple(member.hit_id for member in members if member.role == "supporting")
        outlier_hit_ids = tuple(member.hit_id for member in members if member.role == "outlier")
        suppressed_ids = tuple(
            member.hit_id
            for member in members
            if member.role in {"duplicate_hash", "source_limited"}
        )
        stable_payload = ",".join(sorted(str(hit_id) for hit_id in hit_ids)).encode()
        stable = hashlib.sha256(stable_payload).hexdigest()
        return CandidateHypothesis(
            id=f"hypothesis-{stable[:20]}",
            latitude=center[0],
            longitude=center[1],
            uncertainty_radius_km=uncertainty,
            uncertainty_basis="phase4.robust_geodesic_dispersion",
            classification=classification,
            hit_ids=hit_ids,
            supporting_hit_ids=supporting_ids,
            suppressed_hit_ids=suppressed_ids,
            outlier_hit_ids=outlier_hit_ids,
            members=members,
            source_count=len({member.source for member in members}),
            hash_count=len({member.content_hash for member in members}),
            contributing_source_count=len({hit.metadata.source for hit in supporting}),
            contributing_hash_count=len({hit.metadata.hash for hit in supporting}),
        )
