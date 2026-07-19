from __future__ import annotations

from uuid import UUID

from atlaslens_api.candidate_pipeline.models import (
    FinalCandidateAssessment,
    FinalClassification,
    ReferenceAttribution,
)
from atlaslens_api.phase4_adapter import to_phase4_assessment
from atlaslens_api.reranking.models import ScoreBreakdown


def test_adapter_preserves_uncalibrated_semantics_without_geographic_proof() -> None:
    hit_id = UUID("00000000-0000-4000-8000-000000000001")
    internal = FinalCandidateAssessment(
        hypothesis_id="hypothesis-1",
        classification=FinalClassification.GEOMETRY_SUPPORTED,
        relative_rank_score=0.7,
        score_breakdown=(
            ScoreBreakdown(
                feature="compactness",
                raw=0.7,
                weight=1,
                contribution=0.7,
                reason="phase4.robust_geodesic_compactness",
            ),
        ),
        source_diversity=1,
        contributing_hit_ids=(hit_id,),
        latitude=41,
        longitude=29,
        uncertainty_radius_km=2,
        map_observations=(),
        geometry_results=(),
        contradictions=(),
        reference_attributions=(
            ReferenceAttribution(
                hit_id=hit_id,
                source="licensed-source",
                license="CC-BY-4.0",
                content_hash="0" * 64,
            ),
        ),
        provenance=("phase4-v1",),
        limitations=("phase4.relative_rank_is_not_calibrated_probability",),
    )

    public = to_phase4_assessment(internal)

    assert public.classification == "geometry_supported"
    assert public.score_semantics == "uncalibrated_relative_rank"
    assert "phase4.geometry_is_visual_consistency_not_geographic_proof" in public.limitations
    assert all(not item.display_allowed for item in public.reference_attributions)
