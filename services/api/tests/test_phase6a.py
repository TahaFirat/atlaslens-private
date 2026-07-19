from __future__ import annotations

from atlaslens_api.global_prediction.clustering import GeoClipCandidateClusterer
from atlaslens_api.global_prediction.reverse_geocoding import ReverseGeocodedCluster
from atlaslens_api.phase6a import (
    Phase6AHybridConfig,
    Phase6AHybridEvidenceEngine,
    default_phase6a_config_path,
)
from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    OCRResult,
    ProviderDescriptor,
)
from atlaslens_api.schemas import GeoPoint, PlaceEvidenceSummary, ProviderRunDiagnostic


def hypothesis(
    rank: int,
    latitude: float,
    longitude: float,
    score: float,
) -> GlobalPredictionHypothesis:
    return GlobalPredictionHypothesis(
        rank=rank,
        original_rank=rank,
        latitude=latitude,
        longitude=longitude,
        raw_score=score,
        score_type="uncalibrated_gallery_softmax",
        normalization_method="softmax_over_fixed_gallery",
        calibration_state="uncalibrated",
        limitations=["test.relative_only"],
    )


def descriptor(identifier: str, kind: str) -> ProviderDescriptor:
    return ProviderDescriptor(
        id=identifier,
        kind=kind,
        version="test-v1",
        execution_boundary="local",
        criticality="optional",
        available=True,
    )


def place(latitude: float, longitude: float, *, strength: float = 0.95) -> PlaceEvidenceSummary:
    return PlaceEvidenceSummary(
        matched_entity="redacted-public-place",
        normalized_name="Reviewed Public Place",
        country_code="TR",
        region="Reviewed Region",
        center=GeoPoint(latitude=latitude, longitude=longitude),
        match_type="city",
        text_similarity=0.96,
        ambiguity_count=1,
        evidence_strength=strength,
        source="installed-test-gazetteer",
        dataset_version="test-v1",
        license="test-license",
    )


def test_clustering_nearby_duplicates_distant_and_empty() -> None:
    clusterer = GeoClipCandidateClusterer(radius_km=40)
    assert clusterer.cluster([]) == ()
    clusters = clusterer.cluster(
        [
            hypothesis(1, 40.0, 29.0, 0.9),
            hypothesis(2, 40.0, 29.0, 0.8),
            hypothesis(3, 40.1, 29.1, 0.7),
            hypothesis(4, -20.0, 130.0, 0.6),
        ]
    )
    assert len(clusters) == 2
    assert clusters[0].summary.member_count == 3
    assert clusters[0].summary.member_ranks == [1, 2, 3]
    assert clusters[0].summary.max_raw_similarity == 0.9
    assert clusters[0].summary.score_semantics == "uncalibrated_relative_rank"
    assert clusters[1].summary.member_count == 1


def test_clustering_handles_dateline_and_poles() -> None:
    clusters = GeoClipCandidateClusterer(radius_km=40).cluster(
        [
            hypothesis(1, 0.0, 179.9, 0.9),
            hypothesis(2, 0.0, -179.9, 0.8),
            hypothesis(3, 89.9, 45.0, 0.7),
            hypothesis(4, 89.9, -45.0, 0.6),
        ]
    )
    assert sorted(item.summary.member_count for item in clusters) == [2, 2]
    assert all(-180 <= item.longitude <= 180 for item in clusters)


def _engine() -> Phase6AHybridEvidenceEngine:
    return Phase6AHybridEvidenceEngine(Phase6AHybridConfig.from_path(default_phase6a_config_path()))


def _diagnostics() -> list[ProviderRunDiagnostic]:
    return [
        ProviderRunDiagnostic(
            provider_id="geoclip-global-v1",
            provider_type="global_geolocation",
            status="succeeded",
            duration_ms=1,
            device="cpu",
            offline=True,
        )
    ]


def test_geoclip_only_ranking_is_explainable_and_uncalibrated() -> None:
    clusters = GeoClipCandidateClusterer(radius_km=40).cluster(
        [
            hypothesis(1, 40.0, 29.0, 0.9),
            hypothesis(2, 40.1, 29.1, 0.8),
            hypothesis(3, -20.0, 130.0, 0.7),
        ]
    )
    result = _engine().rank(
        [ReverseGeocodedCluster(cluster=item, place=None) for item in clusters],
        geoclip_descriptor=descriptor("geoclip-global-v1", "global_geolocation"),
        ocr_descriptor=descriptor("local-ocr", "ocr"),
        ocr=None,
        quality=None,
        segmentation=None,
        providers=_diagnostics(),
    )
    assert result.batch.candidates
    first = result.batch.candidates[0]
    assert first.confidence is None
    assert first.confidence_assessment is not None
    assert first.confidence_assessment.score is None
    assert not first.confidence_assessment.calibrated
    assert first.phase5b_assessment is not None
    assert first.phase5b_assessment.reranker_version == "phase6a-v1"
    assert first.phase5b_assessment.score_breakdown[0].feature == "geoclip_cluster"


def test_strong_real_ocr_place_match_reranks_near_cluster() -> None:
    clusters = GeoClipCandidateClusterer(radius_km=40).cluster(
        [
            hypothesis(1, 0.0, 0.0, 0.9),
            hypothesis(2, 38.72, 35.48, 0.7),
            hypothesis(3, 38.73, 35.49, 0.6),
        ]
    )
    result = _engine().rank(
        [ReverseGeocodedCluster(cluster=item, place=None) for item in clusters],
        geoclip_descriptor=descriptor("geoclip-global-v1", "global_geolocation"),
        ocr_descriptor=descriptor("local-ocr", "ocr"),
        ocr=OCRResult(
            redacted_snippets=[],
            blocks=[],
            place_matches=[place(38.72, 35.48)],
        ),
        quality=None,
        segmentation=None,
        providers=_diagnostics(),
    )
    first = result.batch.candidates[0]
    assert first.center.latitude > 30
    assert first.verification_status == "corroborated"
    assert first.phase5b_assessment is not None
    assert "phase6a.support.ocr_place_match" in first.phase5b_assessment.supports
    assert any(
        item.feature == "ocr_place_match" and item.contribution > 0
        for item in first.phase5b_assessment.score_breakdown
    )


def test_weak_ocr_is_bounded_and_segmentation_has_zero_geographic_weight() -> None:
    cluster = GeoClipCandidateClusterer(radius_km=40).cluster([hypothesis(1, 10.0, 20.0, 0.9)])[0]
    result = _engine().rank(
        [ReverseGeocodedCluster(cluster=cluster, place=None)],
        geoclip_descriptor=descriptor("geoclip-global-v1", "global_geolocation"),
        ocr_descriptor=descriptor("local-ocr", "ocr"),
        ocr=OCRResult(
            redacted_snippets=[], blocks=[], place_matches=[place(10.0, 20.0, strength=0.2)]
        ),
        quality=None,
        segmentation=None,
        providers=_diagnostics(),
    )
    assessment = result.batch.candidates[0].phase5b_assessment
    assert assessment is not None
    segmentation = next(
        item
        for item in assessment.score_breakdown
        if item.feature == "segmentation_descriptive_only"
    )
    assert segmentation.weight == 0
    assert segmentation.contribution == 0
    assert "phase6a.support.ocr_place_match" not in assessment.supports
    assert result.batch.candidates[0].confidence is None


def test_strong_contradiction_filters_all_candidates_without_confidence_claim() -> None:
    cluster = GeoClipCandidateClusterer(radius_km=40).cluster([hypothesis(1, 0.0, 0.0, 0.9)])[0]

    result = _engine().rank(
        [ReverseGeocodedCluster(cluster=cluster, place=None)],
        geoclip_descriptor=descriptor("geoclip-global-v1", "global_geolocation"),
        ocr_descriptor=descriptor("local-ocr", "ocr"),
        ocr=OCRResult(
            redacted_snippets=[],
            blocks=[],
            place_matches=[place(40.0, 29.0)],
        ),
        quality=None,
        segmentation=None,
        providers=_diagnostics(),
    )

    assert result.batch.candidates == ()
    assert "warning.confidence_not_calibrated" not in result.warnings
