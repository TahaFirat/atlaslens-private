from __future__ import annotations

from atlaslens_api.candidate_pipeline.models import (
    Contradiction,
    FinalCandidateAssessment,
    FinalClassification,
    GeometryClassification,
    GeometrySignal,
    MapConstraintSignal,
    ReferenceAttribution,
    SignalClassification,
)
from atlaslens_api.reranking.models import CandidateHypothesis, RerankResult


class ConservativeVerificationPolicy:
    def assess(
        self,
        hypothesis: CandidateHypothesis,
        rerank: RerankResult,
        map_observations: tuple[MapConstraintSignal, ...],
        geometry_results: tuple[GeometrySignal, ...],
    ) -> FinalCandidateAssessment:
        contradictions = tuple(
            [
                Contradiction(
                    source="map",
                    reason_code=signal.observation_code,
                    strength=signal.strength,
                )
                for signal in map_observations
                if signal.classification == SignalClassification.CONTRADICTS
                and signal.strength > 0
            ]
            + [
                Contradiction(
                    source="geometry",
                    reason_code=signal.reason_code,
                    strength=signal.strength,
                )
                for signal in geometry_results
                if signal.classification == GeometryClassification.REJECTED and signal.strength > 0
            ]
        )
        geometry_supported = any(
            signal.classification == GeometryClassification.SUPPORTED
            for signal in geometry_results
        )
        map_supported = any(
            signal.classification == SignalClassification.SUPPORTS for signal in map_observations
        )
        if contradictions:
            classification = FinalClassification.CONTRADICTED
        elif geometry_supported and map_supported:
            classification = FinalClassification.MULTI_SOURCE_SUPPORTED
        elif geometry_supported:
            classification = FinalClassification.GEOMETRY_SUPPORTED
        elif map_supported:
            classification = FinalClassification.MAP_SUPPORTED
        elif any(
            signal.classification == GeometryClassification.INCONCLUSIVE
            for signal in geometry_results
        ) or any(
            signal.classification == SignalClassification.INCONCLUSIVE
            for signal in map_observations
        ):
            classification = FinalClassification.ABSTAINED
        else:
            classification = FinalClassification.RETRIEVAL_ONLY

        members = {member.hit_id: member for member in hypothesis.members}
        attributions = tuple(
            ReferenceAttribution(
                hit_id=hit_id,
                source=members[hit_id].source,
                license=members[hit_id].license,
                content_hash=members[hit_id].content_hash,
            )
            for hit_id in rerank.contributing_hit_ids
        )
        provenance = tuple(
            sorted(
                {
                    "phase4-v1",
                    *(
                        f"embedding:{member.embedding_provider}:{member.embedding_version}"
                        for member in hypothesis.members
                    ),
                    *(
                        f"map:{signal.provider_id}:{signal.provider_version}"
                        for signal in map_observations
                    ),
                    *(
                        f"geometry:{signal.provider_id}:{signal.provider_version}"
                        for signal in geometry_results
                    ),
                }
            )
        )
        limitations = ["phase4.relative_rank_is_not_calibrated_probability"]
        if not map_observations or all(
            item.classification == SignalClassification.UNAVAILABLE for item in map_observations
        ):
            limitations.append("phase4.map_constraints_unavailable")
        if not geometry_results or all(
            item.classification == GeometryClassification.UNAVAILABLE for item in geometry_results
        ):
            limitations.append("phase4.geometric_verification_unavailable")
        return FinalCandidateAssessment(
            hypothesis_id=hypothesis.id,
            classification=classification,
            relative_rank_score=rerank.relative_rank_score,
            score_breakdown=rerank.score_breakdown,
            source_diversity=rerank.source_diversity,
            contributing_hit_ids=rerank.contributing_hit_ids,
            latitude=hypothesis.latitude,
            longitude=hypothesis.longitude,
            uncertainty_radius_km=hypothesis.uncertainty_radius_km,
            map_observations=map_observations,
            geometry_results=geometry_results,
            contradictions=contradictions,
            reference_attributions=attributions,
            provenance=provenance,
            limitations=tuple(limitations),
        )
