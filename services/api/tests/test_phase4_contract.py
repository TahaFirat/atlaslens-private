from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlaslens_api.schemas import (
    GeometryVerificationSummary,
    MapConstraintSummary,
    Phase4Assessment,
    Phase4ScoreContribution,
    ReferenceAttribution,
)


def assessment() -> Phase4Assessment:
    return Phase4Assessment(
        classification="multi_source_supported",
        relative_rank_score=0.72,
        reranker_version="phase4-v1",
        score_breakdown=[
            Phase4ScoreContribution(
                feature="source_diversity",
                raw_value=1.0,
                weight=0.2,
                contribution=0.2,
                reason_code="rerank.source_diversity",
            )
        ],
        source_diversity=2,
        contributing_retrieval_hit_ids=["hit-a", "hit-b"],
        map_observations=[
            MapConstraintSummary(
                clue="urban_scene",
                map_feature="dense_roads",
                status="supported",
                reliability=0.7,
                query_radius_km=2.0,
                provider="offline-fixture",
                limitation="fixture-only",
            )
        ],
        geometry_results=[
            GeometryVerificationSummary(
                reference_id="ref-opaque-a",
                provider="orb-ransac-v1",
                status="supported",
                query_keypoints=120,
                reference_keypoints=130,
                raw_matches=80,
                filtered_matches=42,
                inliers=25,
                inlier_ratio=0.59,
                query_coverage=0.31,
                reference_coverage=0.29,
                residual_error_px=1.8,
                robust_model_type="homography",
                limitations=[],
                runtime_ms=12,
            )
        ],
        contradictions=[],
        reference_attributions=[
            ReferenceAttribution(
                reference_id="ref-opaque-a",
                source="licensed-fixture",
                license="CC BY 4.0",
                attribution="Fixture author",
                display_allowed=False,
            )
        ],
        limitations=["uncalibrated"],
    )


def test_phase4_contract_keeps_relative_score_and_safe_summaries_distinct() -> None:
    value = assessment()
    payload = value.model_dump(mode="json")

    assert payload["score_semantics"] == "uncalibrated_relative_rank"
    assert payload["relative_rank_score"] == pytest.approx(0.72)
    assert payload["geometry_results"][0]["status"] == "supported"
    assert payload["reference_attributions"][0]["display_allowed"] is False
    rendered = value.model_dump_json().lower()
    assert "descriptor" not in rendered
    assert "absolute_path" not in rendered


def test_phase4_contract_rejects_duplicate_hit_or_reference_ids() -> None:
    value = assessment().model_dump(mode="python")
    value["contributing_retrieval_hit_ids"] = ["hit-a", "hit-a"]
    with pytest.raises(ValidationError, match="retrieval hit ids must be unique"):
        Phase4Assessment.model_validate(value)

    value = assessment().model_dump(mode="python")
    value["reference_attributions"].append(value["reference_attributions"][0])
    with pytest.raises(ValidationError, match="reference attribution ids must be unique"):
        Phase4Assessment.model_validate(value)
