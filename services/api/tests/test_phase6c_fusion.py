from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlaslens_api.phase6c.fusion import (
    G3VerificationEvidence,
    GeoCLIPHierarchicalEvidence,
    GeoCLIPOriginalEvidence,
    MegaLocRetrievalEvidence,
    OCRPlaceMatchEvidence,
    OSVDirectRegressionEvidence,
    Phase6CAblationSelection,
    Phase6CEvidenceCandidate,
    Phase6CFusionConfig,
    Phase6CGeographicFusionEngine,
    PlonkOSVSampleEvidence,
    load_phase6c_fusion_config,
)


def candidate(
    candidate_id: str,
    latitude: float,
    longitude: float,
    *,
    rank: int = 1,
    raw_value: float | None = 0.5,
    support: int = 1,
    radius_km: float = 25.0,
    country_code: str | None = None,
    mode_id: str | None = None,
    retrieval_cluster_id: str | None = None,
    verification_target_ids: tuple[str, ...] = (),
) -> Phase6CEvidenceCandidate:
    return Phase6CEvidenceCandidate(
        candidate_id=candidate_id,
        latitude=latitude,
        longitude=longitude,
        raw_value=raw_value,
        provider_rank=rank,
        sample_support=support,
        uncertainty_radius_km=radius_km,
        provenance=f"test-provenance:{candidate_id}",
        country_code=country_code,
        mode_id=mode_id,
        retrieval_cluster_id=retrieval_cluster_id,
        verification_target_ids=verification_target_ids,
    )


def geoclip(
    points: list[tuple[float, float, float]],
    *,
    hierarchical: bool = False,
) -> GeoCLIPOriginalEvidence | GeoCLIPHierarchicalEvidence:
    candidates = tuple(
        candidate(
            f"geo-{index}",
            latitude,
            longitude,
            rank=index,
            raw_value=raw,
            country_code="TR" if latitude > 35 else None,
            mode_id=f"mode-{index}" if hierarchical else None,
        )
        for index, (latitude, longitude, raw) in enumerate(points, start=1)
    )
    values = {
        "provider": "geoclip-hierarchical" if hierarchical else "geoclip-original",
        "model_id": "geoclip-test",
        "model_revision": "test-revision",
        "candidates": candidates,
    }
    if hierarchical:
        return GeoCLIPHierarchicalEvidence.model_validate(values)
    return GeoCLIPOriginalEvidence.model_validate(values)


def osv(latitude: float, longitude: float) -> OSVDirectRegressionEvidence:
    return OSVDirectRegressionEvidence(
        provider="osv5m",
        model_id="osv5m-test",
        model_revision="test-revision",
        candidates=(candidate("osv-1", latitude, longitude, raw_value=None, radius_km=80),),
    )


def plonk_osv(latitude: float, longitude: float) -> PlonkOSVSampleEvidence:
    return PlonkOSVSampleEvidence(
        provider="plonk-osv",
        model_id="plonk-osv-test",
        model_revision="test-revision",
        candidates=(candidate("plonk-osv-1", latitude, longitude, support=12),),
    )


def ocr(latitude: float, longitude: float) -> OCRPlaceMatchEvidence:
    return OCRPlaceMatchEvidence(
        provider="local-ocr-place",
        model_id="gazetteer-test",
        model_revision="test-revision",
        candidates=(candidate("ocr-1", latitude, longitude, raw_value=0.9, radius_km=40),),
    )


def test_source_family_and_correlation_group_are_distinct_safety_dimensions() -> None:
    engine = Phase6CGeographicFusionEngine()
    correlated = engine.fuse((osv(41.0, 29.0), plonk_osv(41.01, 29.01))).candidates[0]

    assert correlated.source_family_count == 2
    assert correlated.independent_source_family_count == 1
    assert correlated.correlated_source_family_count == 1
    assert correlated.publication_eligible is False
    assert correlated.confidence is None
    assert correlated.calibrated is False

    independent = engine.fuse(
        (
            geoclip([(41.005, 29.005, 0.2)]),
            osv(41.0, 29.0),
            plonk_osv(41.01, 29.01),
        )
    ).candidates[0]
    assert independent.source_family_count == 3
    assert independent.independent_source_family_count == 2
    assert independent.correlated_source_family_count == 1
    assert independent.publication_eligible is True
    assert independent.publication_basis == "at_least_two_independent_source_families"
    assert independent.uncertainty_radius_km >= 80


def test_incompatible_raw_provider_values_never_change_fusion_rank_or_contributions() -> None:
    engine = Phase6CGeographicFusionEngine()
    low = engine.fuse(
        (
            geoclip([(10.0, 20.0, -0.99), (-30.0, 100.0, 0.99)]),
            ocr(10.01, 20.01),
        )
    )
    high = engine.fuse(
        (
            geoclip([(10.0, 20.0, 0.99), (-30.0, 100.0, -0.99)]),
            OCRPlaceMatchEvidence(
                provider="local-ocr-place",
                model_id="gazetteer-test",
                model_revision="test-revision",
                candidates=(
                    candidate("ocr-1", 10.01, 20.01, raw_value=0.01, radius_km=40),
                ),
            ),
        )
    )

    assert [item.cluster_id for item in low.candidates] == [
        item.cluster_id for item in high.candidates
    ]
    assert [item.relative_rank_score for item in low.candidates] == [
        item.relative_rank_score for item in high.candidates
    ]
    assert [item.contributions for item in low.candidates] == [
        item.contributions for item in high.candidates
    ]
    assert low.candidates[0].members[0].score_semantics.endswith("not_confidence")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((10.0, 179.9), (10.0, -179.9)),
        ((89.9, 0.0), (89.9, 90.0)),
    ],
)
def test_geodesic_clustering_handles_dateline_and_poles(
    left: tuple[float, float], right: tuple[float, float]
) -> None:
    fused = Phase6CGeographicFusionEngine().fuse(
        (
            geoclip([(left[0], left[1], 0.5)]),
            ocr(right[0], right[1]),
        )
    )
    assert len(fused.candidates) == 1
    assert fused.candidates[0].independent_source_family_count == 2
    if abs(left[1]) > 170:
        assert abs(abs(fused.candidates[0].longitude) - 180) < 0.2


def test_complete_link_clustering_does_not_merge_a_transitive_geographic_chain() -> None:
    config = Phase6CFusionConfig(cluster_radius_km=75)
    fused = Phase6CGeographicFusionEngine(config).fuse(
        (geoclip([(0, 0, 0.5), (0, 0.6, 0.4), (0, 1.2, 0.3)]),)
    )
    assert fused.input_cluster_count == 2


def test_diversity_top_k_preserves_distant_modes_and_explains_rank_movement() -> None:
    config = Phase6CFusionConfig(
        cluster_radius_km=10,
        diversity_radius_km=300,
        maximum_candidates_per_diversity_region=2,
        max_candidates=4,
    )
    fused = Phase6CGeographicFusionEngine(config).fuse(
        (
            geoclip(
                [
                    (0.0, 0.0, 0.9),
                    (0.0, 0.2, 0.8),
                    (0.0, 0.4, 0.7),
                    (15.0, 15.0, 0.6),
                    (-25.0, 80.0, 0.5),
                    (55.0, -120.0, 0.4),
                ],
                hierarchical=True,
            ),
        )
    )

    assert len(fused.candidates) == 4
    assert sum(abs(item.latitude) < 1 for item in fused.candidates) <= 2
    assert any(item.pre_diversity_rank > item.final_rank for item in fused.candidates)
    assert all(item.movement_reasons for item in fused.candidates)
    assert any(
        "diversity" in reason
        for item in fused.candidates
        for reason in item.movement_reasons
    )
    assert fused.suppressed_cluster_count == 2


def test_failures_and_abstention_are_explicit() -> None:
    failed = GeoCLIPOriginalEvidence(
        provider="geoclip-original",
        model_id="geoclip-test",
        model_revision="test-revision",
        status="failed",
        reason_code="model_load_failed",
    )
    fused = Phase6CGeographicFusionEngine().fuse((failed,))
    assert fused.abstained is True
    assert fused.abstention_reason == "no_completed_evidence"
    assert fused.candidates == ()
    assert fused.provider_failures[0].reason_code == "model_load_failed"

    excluded = Phase6CGeographicFusionEngine().fuse(
        (geoclip([(10, 20, 0.5)]),),
        ablation=Phase6CAblationSelection(
            profile_id="retrieval_only_test",
            allowed_evidence_kinds=("megaloc_retrieval",),
        ),
    )
    assert excluded.abstained is True
    assert excluded.abstention_reason == "ablation_excluded_all_evidence"
    assert excluded.ablation.excluded_evidence_count == 1


def test_ablation_profile_records_exactly_which_evidence_was_removed() -> None:
    config = Phase6CFusionConfig()
    evidence = (geoclip([(41, 29, 0.5)]), ocr(41.01, 29.01))
    original_only = Phase6CGeographicFusionEngine(config).fuse(
        evidence,
        ablation=config.ablation("geoclip_original_only"),
    )
    final = Phase6CGeographicFusionEngine(config).fuse(evidence)

    assert original_only.ablation.profile_id == "geoclip_original_only"
    assert original_only.ablation.included_evidence_kinds == ("geoclip_original",)
    assert original_only.ablation.excluded_evidence_kinds == ("ocr_place_match",)
    assert original_only.candidates[0].publication_eligible is False
    assert final.candidates[0].publication_eligible is True
    with pytest.raises(ValueError, match="original fusion policy"):
        config.ablation("phase6b_baseline")


def test_typed_evidence_rejects_missing_type_specific_provenance_links() -> None:
    with pytest.raises(ValidationError, match="retrieval cluster"):
        MegaLocRetrievalEvidence(
            provider="megaloc",
            model_id="megaloc-test",
            model_revision="test-revision",
            candidates=(candidate("mega-1", 1, 2),),
        )
    with pytest.raises(ValidationError, match="existing candidate"):
        G3VerificationEvidence(
            provider="g3",
            model_id="g3-test",
            model_revision="test-revision",
            candidates=(candidate("g3-1", 1, 2),),
        )
    with pytest.raises(ValidationError, match="search mode"):
        GeoCLIPHierarchicalEvidence(
            provider="geoclip-hierarchical",
            model_id="geoclip-test",
            model_revision="test-revision",
            candidates=(candidate("hier-1", 1, 2),),
        )
    with pytest.raises(ValidationError, match="scoreless"):
        OSVDirectRegressionEvidence(
            provider="osv5m",
            model_id="osv5m-test",
            model_revision="test-revision",
            candidates=(candidate("osv-1", 1, 2, raw_value=0.5),),
        )


def test_every_score_component_is_bounded_and_fully_explained() -> None:
    fused = Phase6CGeographicFusionEngine().fuse(
        (geoclip([(41, 29, 0.5)]), ocr(41.01, 29.01))
    )
    contributions = fused.candidates[0].contributions
    assert {item.name for item in contributions} == {
        "within_provider_rank",
        "sample_density",
        "provider_diversity",
        "independent_family_corroboration",
        "correlated_source_support",
        "geographic_spread_penalty",
    }
    for item in contributions:
        assert item.source
        assert item.reason
        assert item.correlation_group
        assert math.isfinite(item.raw_value)
        assert 0 <= item.normalized_value <= 1
        assert item.contribution == pytest.approx(item.normalized_value * item.weight)
    independent = next(
        item for item in contributions if item.name == "independent_family_corroboration"
    )
    assert independent.independent is True
    assert all("raw_provider" not in item.name for item in contributions)


def test_checked_in_phase6c_config_is_valid_and_contains_no_holdout_special_case() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "config" / "reranking" / "phase6c-v1.json"
    config = load_phase6c_fusion_config(path)
    assert config.version == "phase6c-v1"
    assert config.source_correlation_groups["osv5m_family"] == (
        config.source_correlation_groups["plonk_osv_family"]
    )
    assert config.source_correlation_groups["g3_family"] == (
        config.source_correlation_groups["geoclip_mp16_family"]
    )
    assert set(config.ablation_profiles["phase6c_final"]) == {
        "geoclip_original",
        "geoclip_hierarchical",
        "osv_direct_regression",
        "plonk_samples",
        "megaloc_retrieval",
        "g3_verification",
        "ocr_place_match",
        "openai_review",
    }
    source = Path(
        root / "services" / "api" / "src" / "atlaslens_api" / "phase6c" / "fusion.py"
    ).read_text(encoding="utf-8").lower()
    assert "kayseri" not in source
