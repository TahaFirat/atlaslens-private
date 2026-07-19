from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlaslens_api.schemas import Candidate, Phase5BDiagnostics


def model_candidate() -> dict[str, object]:
    return {
        "id": "candidate-geoclip-1",
        "rank": 1,
        "center": {"latitude": 21.9, "longitude": -159.5},
        "geometry": {"type": "Point", "coordinates": [-159.5, 21.9]},
        "radius_km": 750,
        "uncertainty_basis": "phase5.uncalibrated_model_dispersion_floor",
        "confidence": None,
        "confidence_kind": "uncalibrated_score",
        "confidence_basis": "phase5.no_calibrated_confidence",
        "granularity": "broad_area",
        "country_code": None,
        "label": None,
        "source": "geoclip-global-v1",
        "evidence_ids": ["evidence-geoclip-1"],
        "evidence_summary": "evidence.global_model_prediction",
        "provenance": [
            {
                "provider_id": "geoclip-global-v1",
                "provider_kind": "global_geolocation",
                "provider_version": "1.0.0",
                "execution_boundary": "local",
                "model_name": "GeoCLIP",
                "output_schema_version": "phase5-global-prediction-v1",
            }
        ],
        "verification_status": "unverified_model",
        "verified": False,
        "phase4_assessment": {
            "classification": "model_only",
            "relative_rank_score": 0.7,
            "score_semantics": "uncalibrated_relative_rank",
            "reranker_version": "phase4-v1",
            "score_breakdown": [],
            "source_diversity": 1,
            "contributing_retrieval_hit_ids": [],
            "map_observations": [],
            "geometry_results": [],
            "contradictions": [],
            "reference_attributions": [],
            "limitations": ["phase5.model_prediction_is_not_geographic_proof"],
        },
        "model_prediction": {
            "provider_id": "geoclip-global-v1",
            "model_name": "GeoCLIP",
            "model_revision": "1.2.0",
            "implementation_revision": "official-pypi-1.2.0",
            "device": "cuda:0",
            "dtype": "float32",
            "raw_score": 0.2,
            "score_type": "uncalibrated_gallery_softmax",
            "normalization_method": "softmax_over_fixed_gallery",
            "calibration_state": "uncalibrated",
            "original_rank": 1,
            "inference_ms": 100,
            "external_transfer": False,
            "limitations": ["phase5.fixed_gallery_not_probability"],
        },
    }


def test_model_candidate_keeps_null_confidence_and_explicit_raw_score() -> None:
    candidate = Candidate.model_validate(model_candidate())

    assert candidate.confidence is None
    assert candidate.model_prediction is not None
    assert candidate.model_prediction.raw_score == 0.2
    assert candidate.phase4_assessment is not None
    assert candidate.phase4_assessment.classification == "model_only"


def test_phase6b_reranker_version_remains_contract_compatible() -> None:
    diagnostics = Phase5BDiagnostics(
        reranker_version="phase6b-v1",
        providers=[],
        partial_failures=[],
    )

    assert diagnostics.reranker_version == "phase6b-v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [("confidence", 0.2), ("radius_km", 749.9), ("verified", True)],
)
def test_model_candidate_rejects_false_precision(field: str, value: object) -> None:
    payload = model_candidate()
    payload[field] = value

    with pytest.raises(ValidationError):
        Candidate.model_validate(payload)
