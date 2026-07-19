from __future__ import annotations

from atlaslens_api.fusion import DeterministicCandidateFusionService
from atlaslens_api.global_prediction import GlobalPredictionCandidateProvider
from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    ProviderDescriptor,
)


def descriptor() -> ProviderDescriptor:
    return ProviderDescriptor(
        id="real-provider-test-double",
        kind="global_geolocation",
        version="test",
        execution_boundary="local",
        criticality="optional",
        available=True,
        model_name="test-only-model",
    )


def hypothesis(
    rank: int, latitude: float, longitude: float, score: float
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
        limitations=["test.fixture_only"],
    )


def result(hypotheses: list[GlobalPredictionHypothesis]) -> GlobalPredictionResult:
    return GlobalPredictionResult(
        provider_id=descriptor().id,
        model_name="test-only-model",
        model_revision="test-revision",
        implementation_revision="test-implementation",
        device="cpu",
        dtype="float32",
        inference_ms=10,
        hypotheses=hypotheses,
    )


def test_model_candidates_are_deduplicated_bounded_and_explicitly_uncalibrated() -> None:
    synthesis = GlobalPredictionCandidateProvider().synthesize(
        result(
            [
                hypothesis(1, 10.0, 179.9, 0.5),
                hypothesis(2, 10.0, -179.9, 0.3),
                hypothesis(3, 40.0, 30.0, 0.1),
                hypothesis(4, -20.0, 120.0, 0.06),
                hypothesis(5, 55.0, -70.0, 0.03),
                hypothesis(6, 0.0, 0.0, 0.01),
            ]
        ),
        descriptor(),
    )
    assert len(synthesis.batch.candidates) == 5
    candidate = synthesis.batch.candidates[0]
    assert candidate.confidence is None
    assert candidate.radius_km >= 750
    assert candidate.granularity == "broad_area"
    assert candidate.verification_status == "unverified_model"
    assert candidate.verified is False
    assert candidate.model_prediction is not None
    assert candidate.model_prediction.calibration_state == "uncalibrated"
    assert candidate.model_prediction.place_label is not None
    assert candidate.model_prediction.place_label.source == "coordinate_fallback"
    assert candidate.phase4_assessment is not None
    assert candidate.phase4_assessment.classification == "model_only"
    assert candidate.phase4_assessment.reranker_version == "phase4-model-only-v1"
    assert candidate.phase4_assessment.contributing_retrieval_hit_ids == []
    assert all(item.confidence is None for item in synthesis.evidence)


def test_stable_order_and_ids_repeat_for_same_provider_output() -> None:
    hypotheses = [
        hypothesis(1, 1.0, 1.0, 0.6),
        hypothesis(2, 20.0, 20.0, 0.3),
        hypothesis(3, -20.0, -20.0, 0.1),
    ]
    provider = GlobalPredictionCandidateProvider()
    first = provider.synthesize(result(hypotheses), descriptor())
    second = provider.synthesize(result(hypotheses), descriptor())
    assert [item.id for item in first.batch.candidates] == [
        item.id for item in second.batch.candidates
    ]


def test_synthesis_preserves_provider_top_three_after_provider_deduplication() -> None:
    synthesized = GlobalPredictionCandidateProvider().synthesize(
        result(
            [
                hypothesis(1, 0.0, 0.0, 0.5),
                hypothesis(2, 0.0, 0.3, 0.3),
                hypothesis(3, 0.0, 0.6, 0.2),
            ]
        ),
        descriptor(),
    )
    assert len(synthesized.batch.candidates) == 3
    fused = DeterministicCandidateFusionService().fuse(
        list(synthesized.evidence), [synthesized.batch]
    )
    assert len(fused.candidates) == 3


def test_invalid_provider_provenance_or_softmax_is_rejected() -> None:
    provider = GlobalPredictionCandidateProvider()
    mismatched = result([hypothesis(1, 0, 0, 0.5)]).model_copy(
        update={"provider_id": "wrong-provider"}
    )
    try:
        provider.synthesize(mismatched, descriptor())
    except ValueError as exc:
        assert "provenance" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched provenance was accepted")

    invalid_score = result([hypothesis(1, 0, 0, 1.5)])
    try:
        provider.synthesize(invalid_score, descriptor())
    except ValueError as exc:
        assert "softmax" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid score was accepted")
