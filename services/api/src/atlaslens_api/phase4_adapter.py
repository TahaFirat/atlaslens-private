from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from atlaslens_api.candidate_pipeline.models import (
    FinalCandidateAssessment,
    FinalClassification,
)
from atlaslens_api.schemas import (
    GeometryVerificationSummary,
    MapConstraintSummary,
    Phase4Assessment,
    Phase4ScoreContribution,
    ReferenceAttribution,
)

_CLASSIFICATIONS = {
    FinalClassification.GEOMETRY_SUPPORTED: "geometry_supported",
    FinalClassification.MAP_SUPPORTED: "map_supported",
    FinalClassification.MULTI_SOURCE_SUPPORTED: "multi_source_supported",
    FinalClassification.RETRIEVAL_ONLY: "retrieval_only",
    FinalClassification.ABSTAINED: "abstained",
    FinalClassification.CONTRADICTED: "contradicted",
}


def to_phase4_assessment(
    assessment: FinalCandidateAssessment,
    *,
    map_observations: Sequence[MapConstraintSummary] = (),
    geometry_results: Sequence[GeometryVerificationSummary] = (),
    attribution_by_hit: Mapping[UUID, str] | None = None,
) -> Phase4Assessment:
    """Adapt the internal Phase 4 result to the additive public API contract.

    Geometry support is deliberately exposed as visual consistency only. It does
    not set or imply a calibrated probability or geographic verification.
    """

    known_hits = set(assessment.contributing_hit_ids)
    public_hit_ids = {str(hit_id) for hit_id in known_hits}
    if any(item.reference_id not in public_hit_ids for item in geometry_results):
        raise ValueError("geometry result references a non-contributing retrieval hit")
    attributions = attribution_by_hit or {}
    return Phase4Assessment(
        classification=_CLASSIFICATIONS[assessment.classification],
        relative_rank_score=assessment.relative_rank_score,
        reranker_version=assessment.reranker_version,
        score_breakdown=[
            Phase4ScoreContribution(
                feature=item.feature,
                raw_value=item.raw,
                weight=item.weight,
                contribution=item.contribution,
                reason_code=item.reason,
            )
            for item in assessment.score_breakdown
        ],
        source_diversity=assessment.source_diversity,
        contributing_retrieval_hit_ids=[str(hit_id) for hit_id in assessment.contributing_hit_ids],
        map_observations=list(map_observations),
        geometry_results=list(geometry_results),
        contradictions=[item.reason_code for item in assessment.contradictions],
        reference_attributions=[
            ReferenceAttribution(
                reference_id=str(item.hit_id),
                source=item.source,
                license=item.license,
                attribution=attributions.get(
                    item.hit_id, f"Source: {item.source}; license: {item.license}"
                ),
                display_allowed=False,
            )
            for item in assessment.reference_attributions
        ],
        limitations=[
            *assessment.limitations,
            "phase4.geometry_is_visual_consistency_not_geographic_proof",
        ],
    )
