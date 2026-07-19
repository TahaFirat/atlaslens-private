from __future__ import annotations

from atlaslens_api.global_prediction import GlobalPredictionCandidateProvider
from atlaslens_api.global_prediction.synthesis import GlobalPredictionSynthesis
from atlaslens_api.phase5b.integration import Phase5BSourceAdapter
from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    OCRResult,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.schemas import GeoPoint, PlaceEvidenceSummary


def global_descriptor(*, available: bool = True) -> ProviderDescriptor:
    return ProviderDescriptor(
        id="geoclip-global-v1",
        kind="global_geolocation",
        version="1.2.0",
        execution_boundary="local",
        criticality="optional",
        available=available,
        unavailable_reason_code=None if available else "model_not_installed",
        model_name="GeoCLIP",
    )


def ocr_descriptor(*, available: bool = True) -> ProviderDescriptor:
    return ProviderDescriptor(
        id="rapidocr-ppocrv6-local",
        kind="ocr",
        version="3.9.1",
        execution_boundary="local",
        criticality="optional",
        available=available,
        unavailable_reason_code=None if available else "missing_dependency",
        model_name="PP-OCRv6-small+script-profiles",
    )


def global_result() -> GlobalPredictionResult:
    return GlobalPredictionResult(
        provider_id="geoclip-global-v1",
        model_name="GeoCLIP",
        model_revision="official-model-v1",
        implementation_revision="official-code-v1",
        device="cuda",
        dtype="float32",
        inference_ms=75,
        hypotheses=[
            GlobalPredictionHypothesis(
                rank=1,
                original_rank=12,
                latitude=41.01,
                longitude=28.97,
                raw_score=0.12,
                score_type="uncalibrated_gallery_softmax",
                normalization_method="softmax_over_fixed_gallery",
                calibration_state="uncalibrated",
                limitations=["test.global_score_is_not_probability"],
            )
        ],
    )


def synthesis() -> GlobalPredictionSynthesis:
    return GlobalPredictionCandidateProvider().synthesize(
        global_result(), global_descriptor()
    )


def place(
    name: str,
    *,
    latitude: float,
    longitude: float,
    strength: float = 0.9,
    ambiguity: int = 1,
    match_type: str = "city",
) -> PlaceEvidenceSummary:
    return PlaceEvidenceSummary(
        matched_entity=name,
        normalized_name=name.casefold(),
        country_code="TR",
        region="Public Region",
        center=GeoPoint(latitude=latitude, longitude=longitude),
        match_type=match_type,  # type: ignore[arg-type]
        text_similarity=0.95,
        ambiguity_count=ambiguity,
        evidence_strength=strength,
        source="GeoNames",
        dataset_version="atlaslens-geonames-forward-v2",
        license="CC BY 4.0",
    )


def ocr_result(*matches: PlaceEvidenceSummary) -> OCRResult:
    return OCRResult(
        redacted_snippets=["RAW PRIVATE OCR SENTINEL MUST NOT CROSS ADAPTER"],
        place_matches=list(matches),
    )


def test_adapter_combines_safe_place_and_existing_global_sources() -> None:
    global_value = global_result()
    adapted = Phase5BSourceAdapter().adapt(
        global_descriptor=global_descriptor(),
        ocr_descriptor=ocr_descriptor(),
        global_synthesis=GlobalPredictionCandidateProvider().synthesize(
            global_value, global_descriptor()
        ),
        global_outcome=ProviderOutcome.succeeded(global_value),
        ocr_result=ocr_result(
            place("İstanbul", latitude=41.0082, longitude=28.9784)
        ),
        global_duration_ms=81,
        ocr_duration_ms=120,
        ocr_device="cuda",
    )
    assert {item.source_kind for item in adapted.hypotheses} == {
        "global_model",
        "ocr_place",
    }
    global_hypothesis = next(
        item for item in adapted.hypotheses if item.source_kind == "global_model"
    )
    place_hypothesis = next(
        item for item in adapted.hypotheses if item.source_kind == "ocr_place"
    )
    assert global_hypothesis.uncertainty_radius_km >= 750
    assert global_hypothesis.place_matches == ()
    assert place_hypothesis.uncertainty_radius_km == 75
    assert place_hypothesis.place_matches[0].matched_entity == "İstanbul"
    assert len(adapted.evidence) == 2
    assert [item.status for item in adapted.providers] == ["succeeded", "succeeded"]
    assert adapted.providers[0].duration_ms == 81
    assert adapted.providers[1].device == "cuda"
    rendered = repr(adapted)
    assert "RAW PRIVATE OCR SENTINEL" not in rendered
    assert "İstanbul" in rendered


def test_place_ids_and_order_are_deterministic_and_duplicates_are_suppressed() -> None:
    istanbul = place("İstanbul", latitude=41.0082, longitude=28.9784, strength=0.8)
    ankara = place("Ankara", latitude=39.9334, longitude=32.8597, strength=0.9)
    adapter = Phase5BSourceAdapter()
    first = adapter.adapt(
        global_descriptor=global_descriptor(available=False),
        ocr_descriptor=ocr_descriptor(),
        ocr_result=ocr_result(istanbul, ankara, istanbul),
    )
    second = adapter.adapt(
        global_descriptor=global_descriptor(available=False),
        ocr_descriptor=ocr_descriptor(),
        ocr_result=ocr_result(istanbul, ankara),
    )
    assert [item.id for item in first.hypotheses] == [
        item.id for item in second.hypotheses
    ]
    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]
    place_hypotheses = [
        item for item in first.hypotheses if item.source_kind == "ocr_place"
    ]
    assert len(place_hypotheses) == 2
    assert place_hypotheses[0].place_matches[0].matched_entity == "Ankara"


def test_ambiguity_expands_conservative_place_radius() -> None:
    unambiguous = Phase5BSourceAdapter().adapt(
        global_descriptor=global_descriptor(available=False),
        ocr_descriptor=ocr_descriptor(),
        ocr_result=ocr_result(
            place("Paris", latitude=48.85, longitude=2.35, ambiguity=1)
        ),
    )
    ambiguous = Phase5BSourceAdapter().adapt(
        global_descriptor=global_descriptor(available=False),
        ocr_descriptor=ocr_descriptor(),
        ocr_result=ocr_result(
            place("Paris", latitude=48.85, longitude=2.35, ambiguity=9)
        ),
    )
    assert unambiguous.hypotheses[0].uncertainty_radius_km == 75
    assert ambiguous.hypotheses[0].uncertainty_radius_km == 225


def test_empty_and_unavailable_sources_degrade_without_exception() -> None:
    adapted = Phase5BSourceAdapter().adapt(
        global_descriptor=global_descriptor(available=False),
        ocr_descriptor=ocr_descriptor(available=False),
    )
    assert adapted.hypotheses == ()
    assert adapted.evidence == ()
    assert [item.status for item in adapted.providers] == ["skipped", "skipped"]
    assert [item.reason_code for item in adapted.providers] == [
        "model_not_installed",
        "missing_dependency",
    ]
    assert adapted.partial_failures == (
        "phase5b.global.model_not_installed",
        "phase5b.ocr.missing_dependency",
    )


def test_abstention_is_empty_but_not_reported_as_provider_failure() -> None:
    global_abstention: ProviderOutcome[GlobalPredictionResult] = ProviderOutcome.abstained()
    ocr_abstention: ProviderOutcome[OCRResult] = ProviderOutcome.abstained()
    adapted = Phase5BSourceAdapter().adapt(
        global_descriptor=global_descriptor(),
        ocr_descriptor=ocr_descriptor(),
        global_outcome=global_abstention,
        ocr_outcome=ocr_abstention,
    )
    assert [item.status for item in adapted.providers] == ["abstained", "abstained"]
    assert adapted.partial_failures == ()
    assert adapted.hypotheses == ()


def test_invalid_global_draft_is_safely_rejected_and_reported() -> None:
    valid = synthesis()
    invalid = GlobalPredictionSynthesis(
        evidence=(),
        batch=valid.batch,
        warnings=valid.warnings,
    )
    adapted = Phase5BSourceAdapter().adapt(
        global_descriptor=global_descriptor(),
        ocr_descriptor=ocr_descriptor(available=False),
        global_synthesis=invalid,
    )
    assert not any(item.source_kind == "global_model" for item in adapted.hypotheses)
    assert adapted.evidence == ()
    assert "phase5b.global.invalid_candidate_draft" in adapted.partial_failures


def test_global_provider_mismatch_is_rejected_without_cross_provenance() -> None:
    valid = synthesis()
    wrong_descriptor = global_descriptor().model_copy(update={"id": "other-global"})
    adapted = Phase5BSourceAdapter().adapt(
        global_descriptor=wrong_descriptor,
        ocr_descriptor=ocr_descriptor(available=False),
        global_synthesis=valid,
    )
    assert not any(item.source_kind == "global_model" for item in adapted.hypotheses)
    assert adapted.evidence == ()
    assert "phase5b.global.invalid_candidate_draft" in adapted.partial_failures


def test_failed_ocr_outcome_ignores_accidentally_supplied_result() -> None:
    failed: ProviderOutcome[OCRResult] = ProviderOutcome.failed(
        "inference_timeout", retryable=False, attempts=1, duration_ms=99
    )
    adapted = Phase5BSourceAdapter().adapt(
        global_descriptor=global_descriptor(available=False),
        ocr_descriptor=ocr_descriptor(),
        ocr_result=ocr_result(
            place("Must Not Appear", latitude=41.0, longitude=29.0)
        ),
        ocr_outcome=failed,
    )
    assert not any(item.source_kind == "ocr_place" for item in adapted.hypotheses)
    assert all("Must Not Appear" not in repr(item) for item in adapted.evidence)
    assert adapted.providers[1].status == "failed"
    assert adapted.providers[1].duration_ms == 99
    assert "phase5b.ocr.inference_timeout" in adapted.partial_failures
