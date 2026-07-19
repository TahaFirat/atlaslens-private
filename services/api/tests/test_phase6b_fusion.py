from __future__ import annotations

from pathlib import Path

import pytest

from atlaslens_api.phase6b.fusion import (
    Phase6BFusedCandidate,
    Phase6BGeographicFusionEngine,
    load_phase6b_fusion_config,
)
from atlaslens_api.phase6b.models import GeographicCandidate, GeographicProviderResult
from atlaslens_api.schemas import GeoPoint, PlaceEvidenceSummary


def result(
    provider: str,
    model_id: str,
    family: str,
    points: list[tuple[float, float, float | None]],
) -> GeographicProviderResult:
    semantics = "sample_density" if provider == "plonk" else "similarity"
    if provider == "osv5m-baseline":
        semantics = "direct_regression"
    return GeographicProviderResult.model_validate(
        {
            "provider": provider,
            "model_id": model_id,
            "model_revision": "test-revision",
            "source_family": family,
            "status": "completed",
            "device": "cpu",
            "duration_ms": 1,
            "score_semantics": semantics,
            "candidates": [
                GeographicCandidate(
                    candidate_id=f"{provider}-{index}",
                    latitude=latitude,
                    longitude=longitude,
                    raw_score=raw_score,
                    provider_rank=index,
                    sample_support=3 if provider == "plonk" else 1,
                )
                for index, (latitude, longitude, raw_score) in enumerate(points, start=1)
            ],
        }
    )


def contribution(candidate: Phase6BFusedCandidate, name: str) -> float:
    return next(item.contribution for item in candidate.contributions if item.name == name)


def test_independent_family_agreement_outweighs_same_family_duplicate_support() -> None:
    engine = Phase6BGeographicFusionEngine()
    independent = engine.fuse(
        (
            result("geoclip-global-v1", "GeoCLIP", "mp16_family", [(41.0, 29.0, 0.99)]),
            result(
                "plonk",
                "nicolas-dufour/PLONK_YFCC",
                "yfcc_family",
                [(41.01, 29.01, 0.2)],
            ),
        )
    ).candidates[0]
    correlated = engine.fuse(
        (
            result(
                "osv5m-baseline",
                "osv5m/baseline",
                "osv5m_family",
                [(41.0, 29.0, None)],
            ),
            result(
                "plonk",
                "nicolas-dufour/PLONK_OSV_5M",
                "osv5m_family",
                [(41.01, 29.01, 0.9)],
            ),
        )
    ).candidates[0]
    assert independent.independent_family_count == 2
    assert independent.same_family_duplicate_support == 0
    assert correlated.independent_family_count == 1
    assert correlated.same_family_duplicate_support == 1
    assert contribution(independent, "independent_model_agreement") > 0
    assert contribution(correlated, "independent_model_agreement") == 0
    assert contribution(correlated, "same_family_support") > 0
    assert independent.relative_rank_score > correlated.relative_rank_score


def test_raw_provider_scores_are_never_averaged_or_used_by_fusion() -> None:
    engine = Phase6BGeographicFusionEngine()
    low = engine.fuse((result("geoclip-global-v1", "GeoCLIP", "mp16_family", [(10, 20, 0.01)]),))
    high = engine.fuse((result("geoclip-global-v1", "GeoCLIP", "mp16_family", [(10, 20, 0.99)]),))
    assert low == high


def test_failure_is_neutral_and_three_way_disagreement_remains_explainable() -> None:
    failed = GeographicProviderResult(
        provider="osv5m-baseline",
        model_id="osv5m/baseline",
        model_revision="revision",
        source_family="osv5m_family",
        status="failed",
        device="cpu",
        duration_ms=2,
        score_semantics="direct_regression",
        reason_code="worker_unavailable",
    )
    fused = Phase6BGeographicFusionEngine().fuse(
        (
            result("geoclip-global-v1", "GeoCLIP", "mp16_family", [(0, 0, 0.4)]),
            result(
                "plonk",
                "nicolas-dufour/PLONK_YFCC",
                "yfcc_family",
                [(45, 45, 0.4)],
            ),
            result("other-model", "other", "inat_family", [(-45, -45, 0.4)]),
            failed,
        )
    )
    assert len(fused.candidates) == 3
    assert fused.abstained is False
    assert "osv5m-baseline:worker_unavailable" in fused.provider_failures
    assert all(candidate.independent_family_count == 1 for candidate in fused.candidates)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((10.0, 179.9), (10.0, -179.9)),
        ((89.9, 0.0), (89.9, 90.0)),
    ],
)
def test_fusion_clusters_dateline_and_polar_neighbors(
    left: tuple[float, float], right: tuple[float, float]
) -> None:
    fused = Phase6BGeographicFusionEngine().fuse(
        (
            result(
                "geoclip-global-v1",
                "GeoCLIP",
                "mp16_family",
                [(left[0], left[1], 0.5)],
            ),
            result(
                "osv5m-baseline",
                "osv5m/baseline",
                "osv5m_family",
                [(right[0], right[1], None)],
            ),
        )
    )
    assert len(fused.candidates) == 1
    assert fused.candidates[0].provider_count == 2
    if abs(left[1]) > 170:
        assert abs(abs(fused.candidates[0].longitude) - 180) < 0.2


def test_duplicate_points_from_one_provider_do_not_create_provider_diversity() -> None:
    fused = Phase6BGeographicFusionEngine().fuse(
        (
            result(
                "geoclip-global-v1",
                "GeoCLIP",
                "mp16_family",
                [(41.0, 29.0, 0.9), (41.0, 29.0, 0.8)],
            ),
        )
    )
    assert fused.candidates[0].provider_count == 1
    assert fused.candidates[0].independent_family_count == 1
    assert fused.candidates[0].same_family_duplicate_support == 0


def place(latitude: float, longitude: float) -> PlaceEvidenceSummary:
    return PlaceEvidenceSummary(
        matched_entity="Test city",
        normalized_name="Test city",
        country_code="TR",
        center=GeoPoint(latitude=latitude, longitude=longitude),
        match_type="city",
        text_similarity=0.95,
        ambiguity_count=1,
        evidence_strength=0.9,
        source="local-test-gazetteer",
        dataset_version="test-v1",
        license="test-only",
    )


def test_ocr_agreement_and_contradiction_are_separate_explicit_components() -> None:
    provider = result("geoclip-global-v1", "GeoCLIP", "mp16_family", [(41.0, 29.0, 0.5)])
    engine = Phase6BGeographicFusionEngine()
    supported = engine.fuse((provider,), ocr_places=(place(41.01, 29.01),)).candidates[0]
    contradicted = engine.fuse((provider,), ocr_places=(place(-30, -60),)).candidates[0]
    assert supported.ocr_agreement is True
    assert supported.ocr_contradiction is False
    assert contradicted.ocr_agreement is False
    assert contradicted.ocr_contradiction is True
    assert contribution(supported, "ocr_place_agreement") > 0
    assert contribution(contradicted, "ocr_contradiction_penalty") < 0


def test_checked_in_fusion_config_is_valid_and_deterministic() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_phase6b_fusion_config(root / "config" / "reranking" / "phase6b-v1.json")
    engine = Phase6BGeographicFusionEngine(config)
    providers = (
        result("geoclip-global-v1", "GeoCLIP", "mp16_family", [(41, 29, 0.5)]),
        result(
            "plonk",
            "nicolas-dufour/PLONK_YFCC",
            "yfcc_family",
            [(41.01, 29.01, 0.5)],
        ),
    )
    assert engine.fuse(providers).model_dump_json() == engine.fuse(providers).model_dump_json()
