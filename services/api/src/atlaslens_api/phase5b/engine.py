from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence

from atlaslens_api.fusion import CandidateBatch, CandidateDraft
from atlaslens_api.phase5b.config import Phase5BRerankConfig
from atlaslens_api.phase5b.models import (
    EvidenceHypothesis,
    Phase5BCluster,
    Phase5BEngineResult,
    Phase5BIntegrationResult,
)
from atlaslens_api.reranking.geo import geodesic_km, percentile, spherical_center
from atlaslens_api.schemas import (
    GeoPoint,
    MapConstraintSummary,
    ModelPredictionDiagnostics,
    Phase5BAssessment,
    Phase5BDiagnostics,
    Phase5BScoreContribution,
    PlaceEvidenceSummary,
    Provenance,
    ProviderRunDiagnostic,
    ReferenceIndexDiagnostic,
    RetrievalMatchSummary,
)


class Phase5BEvidenceReranker:
    """Deterministic evidence clustering and relative ranking.

    Inputs are already validated, source-specific hypotheses. This engine performs
    no network access and never treats its score as a probability or confidence.
    """

    version = "phase5b-v1"

    def __init__(self, config: Phase5BRerankConfig) -> None:
        self._config = config

    def rerank(self, hypotheses: Sequence[EvidenceHypothesis]) -> Phase5BEngineResult:
        unique = self._bounded_unique(hypotheses)
        if not unique:
            return Phase5BEngineResult(
                clusters=(), abstained=True, abstention_reason="phase5b.no_hypotheses"
            )
        components = self._components(unique)[: self._config.clustering.max_clusters]
        provisional = [self._build_cluster(component) for component in components]
        provisional.sort(
            key=lambda item: (
                -item.assessment.relative_rank_score,
                -item.assessment.provider_diversity,
                -item.assessment.source_diversity,
                item.original_best_rank,
                item.id,
            )
        )
        ranked: list[Phase5BCluster] = []
        for rank, cluster in enumerate(provisional[: self._config.scoring.max_candidates], 1):
            movement = list(cluster.assessment.movement_reasons)
            if rank < cluster.original_best_rank:
                movement.append("phase5b.rank_increased.evidence_support")
            elif rank > cluster.original_best_rank:
                movement.append("phase5b.rank_decreased.relative_support")
            else:
                movement.append("phase5b.rank_preserved")
            ranked.append(
                cluster.model_copy(
                    update={
                        "rank": rank,
                        "assessment": cluster.assessment.model_copy(
                            update={"movement_reasons": list(dict.fromkeys(movement))[:24]}
                        ),
                    }
                )
            )
        viable = [
            item
            for item in ranked
            if item.assessment.classification != "contradicted"
            and item.assessment.relative_rank_score
            >= self._config.scoring.minimum_relative_score
        ]
        if not viable:
            reason = (
                "phase5b.all_candidates_contradicted"
                if ranked and all(
                    item.assessment.classification == "contradicted" for item in ranked
                )
                else "phase5b.insufficient_relative_support"
            )
            return Phase5BEngineResult(
                clusters=tuple(ranked), abstained=True, abstention_reason=reason
            )
        return Phase5BEngineResult(clusters=tuple(ranked), abstained=False)

    def rank_candidates(
        self,
        hypotheses: Sequence[EvidenceHypothesis],
        *,
        providers: Sequence[ProviderRunDiagnostic] = (),
        reference_index: ReferenceIndexDiagnostic | None = None,
        partial_failures: Sequence[str] = (),
    ) -> Phase5BIntegrationResult:
        """Return fusion-ready drafts without depending on concrete evidence providers."""

        engine_result = self.rerank(hypotheses)
        by_id = {item.id: item for item in hypotheses}
        drafts: list[CandidateDraft] = []
        for cluster in engine_result.viable_clusters:
            if cluster.assessment.relative_rank_score < self._config.scoring.minimum_relative_score:
                continue
            members = [by_id[item_id] for item_id in cluster.member_ids if item_id in by_id]
            if not members:
                continue
            evidence_ids = list(
                dict.fromkeys(
                    evidence_id for member in members for evidence_id in member.evidence_ids
                )
            )
            provenance: list[Provenance] = []
            seen_provenance: set[str] = set()
            for item in [
                *(entry for member in members for entry in member.provenance),
                Provenance(
                    provider_id="phase5b-evidence-engine",
                    provider_kind="candidate_reranking",
                    provider_version=self.version,
                    execution_boundary="local",
                    output_schema_version="phase5b-assessment-v1",
                ),
            ]:
                key = item.model_dump_json()
                if key not in seen_provenance:
                    seen_provenance.add(key)
                    provenance.append(item)
            place = (
                cluster.assessment.place_matches[0]
                if cluster.assessment.place_matches
                else None
            )
            drafts.append(
                CandidateDraft(
                    id=f"candidate-{cluster.id}",
                    center=GeoPoint(latitude=cluster.latitude, longitude=cluster.longitude),
                    radius_km=cluster.uncertainty_radius_km,
                    uncertainty_basis=cluster.uncertainty_basis,
                    confidence=None,
                    confidence_kind="uncalibrated_score",
                    confidence_basis="phase5b.no_probability_before_calibration",
                    granularity=self._granularity(place),
                    country_code=place.country_code if place is not None else None,
                    label=place.normalized_name if place is not None else None,
                    source="phase5b-evidence-engine",
                    evidence_ids=evidence_ids,
                    evidence_summary="evidence.phase5b_combined_evidence",
                    provenance=provenance,
                    verification_status=(
                        "unverified_model"
                        if cluster.assessment.classification == "model_only"
                        else "corroborated"
                    ),
                    verified=False,
                    phase5b_assessment=cluster.assessment,
                    model_prediction=(
                        self._model_prediction(members)
                        if cluster.assessment.classification == "model_only"
                        else None
                    ),
                )
            )
        diagnostics = Phase5BDiagnostics(
            providers=list(providers)[:16],
            reference_index=reference_index,
            partial_failures=list(dict.fromkeys(partial_failures))[:24],
        )
        return Phase5BIntegrationResult(
            batch=CandidateBatch(
                provider_id="phase5b-evidence-engine", candidates=tuple(drafts)
            ),
            diagnostics=diagnostics,
            engine_result=engine_result,
        )

    @staticmethod
    def _model_prediction(
        members: Sequence[EvidenceHypothesis],
    ) -> ModelPredictionDiagnostics | None:
        predictions = [
            member.model_prediction
            for member in members
            if member.model_prediction is not None
        ]
        if not predictions:
            return None
        return min(predictions, key=lambda item: (item.original_rank, -item.raw_score))

    def _bounded_unique(
        self, hypotheses: Sequence[EvidenceHypothesis]
    ) -> list[EvidenceHypothesis]:
        by_id: dict[str, EvidenceHypothesis] = {}
        for hypothesis in hypotheses:
            current = by_id.get(hypothesis.id)
            if current is None or self._member_key(hypothesis) < self._member_key(current):
                by_id[hypothesis.id] = hypothesis
        return sorted(by_id.values(), key=self._member_key)[
            : self._config.clustering.max_input_hypotheses
        ]

    def _components(
        self, hypotheses: list[EvidenceHypothesis]
    ) -> list[list[EvidenceHypothesis]]:
        parents = list(range(len(hypotheses)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        for left in range(len(hypotheses)):
            for right in range(left + 1, len(hypotheses)):
                if geodesic_km(
                    (hypotheses[left].latitude, hypotheses[left].longitude),
                    (hypotheses[right].latitude, hypotheses[right].longitude),
                ) <= self._config.clustering.radius_km:
                    union(left, right)
        grouped: dict[int, list[EvidenceHypothesis]] = defaultdict(list)
        for index, hypothesis in enumerate(hypotheses):
            grouped[find(index)].append(hypothesis)
        components = [sorted(items, key=self._member_key) for items in grouped.values()]
        components.sort(key=lambda items: (min(item.original_rank for item in items), items[0].id))
        return components

    def _build_cluster(self, component: list[EvidenceHypothesis]) -> Phase5BCluster:
        retained, suppressed_count = self._suppress(component)
        center = spherical_center([(item.latitude, item.longitude) for item in retained])
        distances = [
            geodesic_km(center, (item.latitude, item.longitude)) for item in retained
        ]
        dispersion = percentile(distances, 0.75) if distances else 0.0
        places = self._place_matches(retained)
        retrieval = self._retrieval_matches(retained)
        map_observations = self._map_observations(retained)
        provider_diversity = len({item.provider_id for item in retained})
        source_diversity = len({item.source_id for item in retained})
        map_support = self._map_support(map_observations)
        contradiction_strength = max(
            [
                item.strength
                for hypothesis in retained
                for item in hypothesis.contradictions
            ]
            + [
                item.reliability
                for item in map_observations
                if item.status == "contradicted"
            ]
            + [0.0]
        )
        features = {
            "model_prior": max(
                [item.raw_score for item in retained if item.source_kind == "global_model"]
                + [0.0]
            ),
            "place_support": self._place_support(places),
            "retrieval_similarity": max(
                [max(0.0, item.relative_similarity) for item in retrieval] + [0.0]
            ),
            "retrieval_compactness": (
                1.0 / (1.0 + dispersion / self._config.clustering.radius_km)
                if len(retrieval) >= 2
                else 0.0
            ),
            "map_support": map_support,
            "provider_diversity": min(
                1.0,
                provider_diversity
                / self._config.scoring.provider_diversity_normalizer,
            ),
            "source_diversity": min(
                1.0,
                source_diversity / self._config.scoring.source_diversity_normalizer,
            ),
        }
        breakdown = [
            Phase5BScoreContribution(
                feature=name,
                raw_value=value,
                weight=self._config.scoring.weights[name],
                contribution=value * self._config.scoring.weights[name],
                reason_code=f"phase5b.feature.{name}",
            )
            for name, value in features.items()
        ]
        if contradiction_strength > 0:
            penalty = -contradiction_strength * self._config.scoring.contradiction_penalty
            breakdown.append(
                Phase5BScoreContribution(
                    feature="contradiction_penalty",
                    raw_value=contradiction_strength,
                    weight=-self._config.scoring.contradiction_penalty,
                    contribution=penalty,
                    reason_code="phase5b.feature.contradiction_penalty",
                )
            )
        score = min(1.0, max(0.0, sum(item.contribution for item in breakdown)))
        classification = self._classification(
            retained, places, retrieval, map_support, contradiction_strength
        )
        radius = self._uncertainty_radius(
            retained, classification=classification, dispersion=dispersion
        )
        supports = list(
            dict.fromkeys(
                [support for item in retained for support in item.supports]
                + (["phase5b.support.place"] if places else [])
                + (["phase5b.support.retrieval"] if retrieval else [])
                + (["phase5b.support.map"] if map_support > 0 else [])
            )
        )[:24]
        contradictions = list(
            dict.fromkeys(
                [
                    contradiction.reason_code
                    for item in retained
                    for contradiction in item.contradictions
                ]
                + [
                    f"phase5b.map.{item.map_feature}.contradicted"
                    for item in map_observations
                    if item.status == "contradicted"
                ]
            )
        )[:24]
        limitations = ["phase5b.relative_rank_is_not_probability"]
        if not places:
            limitations.append("phase5b.place_evidence_unavailable")
        if not retrieval:
            limitations.append("phase5b.retrieval_evidence_unavailable")
        if not map_observations or all(item.status == "unknown" for item in map_observations):
            limitations.append("phase5b.map_evidence_unavailable")
        if suppressed_count:
            limitations.append("phase5b.repeated_source_members_suppressed")
        cluster_id = self._cluster_id(retained)
        assessment = Phase5BAssessment(
            classification=classification,
            relative_rank_score=score,
            score_breakdown=breakdown,
            provider_diversity=provider_diversity,
            source_diversity=source_diversity,
            place_matches=places,
            retrieval_matches=retrieval,
            map_observations=map_observations,
            supports=supports,
            contradictions=contradictions,
            limitations=limitations[:24],
        )
        return Phase5BCluster(
            id=cluster_id,
            rank=1,
            original_best_rank=min(item.original_rank for item in retained),
            latitude=center[0],
            longitude=center[1],
            uncertainty_radius_km=radius,
            member_ids=tuple(sorted(item.id for item in retained)),
            suppressed_member_count=suppressed_count,
            assessment=assessment,
        )

    def _suppress(
        self, members: list[EvidenceHypothesis]
    ) -> tuple[list[EvidenceHypothesis], int]:
        retained: list[EvidenceHypothesis] = []
        provider_counts: dict[str, int] = defaultdict(int)
        source_counts: dict[str, int] = defaultdict(int)
        family_counts: dict[str, int] = defaultdict(int)
        content_hash_counts: dict[str, int] = defaultdict(int)
        for item in sorted(members, key=self._member_key):
            family = item.capture_family_id
            content_hash = item.content_hash
            if (
                provider_counts[item.provider_id]
                >= self._config.suppression.max_members_per_provider
            ):
                continue
            if source_counts[item.source_id] >= self._config.suppression.max_members_per_source:
                continue
            if (
                family is not None
                and family_counts[family]
                >= self._config.suppression.max_members_per_capture_family
            ):
                continue
            if (
                content_hash is not None
                and content_hash_counts[content_hash]
                >= self._config.suppression.max_members_per_content_hash
            ):
                continue
            retained.append(item)
            provider_counts[item.provider_id] += 1
            source_counts[item.source_id] += 1
            if family is not None:
                family_counts[family] += 1
            if content_hash is not None:
                content_hash_counts[content_hash] += 1
        if not retained:
            retained = [min(members, key=self._member_key)]
        return retained, len(members) - len(retained)

    @staticmethod
    def _member_key(item: EvidenceHypothesis) -> tuple[float, int, str, str, str]:
        return -item.raw_score, item.original_rank, item.provider_id, item.source_id, item.id

    @staticmethod
    def _place_matches(items: Sequence[EvidenceHypothesis]) -> list[PlaceEvidenceSummary]:
        unique: dict[tuple[object, ...], PlaceEvidenceSummary] = {}
        for match in (match for item in items for match in item.place_matches):
            key = (
                match.normalized_name.casefold(),
                match.country_code,
                round(match.center.latitude, 5),
                round(match.center.longitude, 5),
                match.source,
            )
            current = unique.get(key)
            if current is None or (
                match.evidence_strength,
                match.text_similarity,
                -match.ambiguity_count,
            ) > (
                current.evidence_strength,
                current.text_similarity,
                -current.ambiguity_count,
            ):
                unique[key] = match
        return sorted(
            unique.values(),
            key=lambda item: (
                -item.evidence_strength,
                -item.text_similarity,
                item.ambiguity_count,
                item.normalized_name.casefold(),
            ),
        )[:12]

    @staticmethod
    def _retrieval_matches(
        items: Sequence[EvidenceHypothesis],
    ) -> list[RetrievalMatchSummary]:
        unique: dict[str, RetrievalMatchSummary] = {}
        for match in (match for item in items for match in item.retrieval_matches):
            current = unique.get(match.reference_id)
            if current is None or (match.relative_similarity, -match.distance, match.source) > (
                current.relative_similarity,
                -current.distance,
                current.source,
            ):
                unique[match.reference_id] = match
        return sorted(
            unique.values(),
            key=lambda item: (-item.relative_similarity, item.distance, item.reference_id),
        )[:16]

    @staticmethod
    def _map_observations(
        items: Sequence[EvidenceHypothesis],
    ) -> list[MapConstraintSummary]:
        unique: dict[tuple[str, str, str], MapConstraintSummary] = {}
        for observation in (
            observation for item in items for observation in item.map_observations
        ):
            key = (observation.provider, observation.clue, observation.map_feature)
            current = unique.get(key)
            if current is None or observation.reliability > current.reliability:
                unique[key] = observation
        return sorted(
            unique.values(),
            key=lambda item: (
                item.provider,
                item.clue,
                item.map_feature,
                item.status,
            ),
        )[:32]

    @staticmethod
    def _place_support(matches: Sequence[PlaceEvidenceSummary]) -> float:
        return max(
            [
                match.evidence_strength
                * match.text_similarity
                / math.sqrt(match.ambiguity_count)
                for match in matches
            ]
            + [0.0]
        )

    @staticmethod
    def _map_support(observations: Sequence[MapConstraintSummary]) -> float:
        supported = [item.reliability for item in observations if item.status == "supported"]
        return sum(supported) / len(supported) if supported else 0.0

    def _classification(
        self,
        retained: Sequence[EvidenceHypothesis],
        places: Sequence[PlaceEvidenceSummary],
        retrieval: Sequence[RetrievalMatchSummary],
        map_support: float,
        contradiction_strength: float,
    ) -> str:
        if contradiction_strength >= self._config.scoring.contradiction_threshold:
            return "contradicted"
        independent = int(bool(places)) + int(bool(retrieval)) + int(map_support > 0)
        if independent >= 2:
            return "multi_source_supported"
        if places:
            return "place_supported"
        if retrieval:
            return "retrieval_supported"
        if map_support > 0:
            return "map_supported"
        del retained
        return "model_only"

    def _uncertainty_radius(
        self,
        retained: Sequence[EvidenceHypothesis],
        *,
        classification: str,
        dispersion: float,
    ) -> float:
        kinds = {item.source_kind for item in retained}
        if "ocr_place" in kinds and (
            "visual_retrieval" in kinds or "landmark_research" in kinds
        ):
            floor = self._config.uncertainty.multi_source_floor_km
        elif "visual_retrieval" in kinds or "landmark_research" in kinds:
            floor = self._config.uncertainty.retrieval_supported_floor_km
        elif "ocr_place" in kinds or "gazetteer" in kinds:
            floor = self._config.uncertainty.place_supported_floor_km
        else:
            floor = self._config.uncertainty.model_only_floor_km
        if classification == "contradicted":
            floor = max(floor, min(item.uncertainty_radius_km for item in retained))
        return max(
            floor,
            dispersion * self._config.uncertainty.dispersion_multiplier,
            min(item.uncertainty_radius_km for item in retained),
        )

    @staticmethod
    def _cluster_id(items: Sequence[EvidenceHypothesis]) -> str:
        payload = "|".join(sorted(item.id for item in items)).encode()
        return f"phase5b-cluster-{hashlib.sha256(payload).hexdigest()[:20]}"

    @staticmethod
    def _granularity(place: PlaceEvidenceSummary | None) -> str:
        if place is None:
            return "broad_area"
        if place.match_type == "country":
            return "country"
        if place.match_type == "region":
            return "region"
        return "city"
