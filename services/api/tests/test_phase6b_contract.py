from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlaslens_api.inference.models import (
    InferenceCandidate,
    InferenceProvenance,
    InferenceResult,
)
from atlaslens_api.phase6b.geoclip import normalize_geoclip_result
from atlaslens_api.phase6b.models import GeographicCandidate, GeographicProviderResult


def candidate(**updates: object) -> GeographicCandidate:
    values: dict[str, object] = {
        "candidate_id": "test-point",
        "latitude": 10.0,
        "longitude": 190.0,
        "raw_score": 0.8,
        "provider_rank": 1,
        "sample_support": 1,
    }
    values.update(updates)
    return GeographicCandidate.model_validate(values)


def test_candidate_normalizes_longitude_and_rejects_non_finite_values() -> None:
    assert candidate().longitude == -170.0
    with pytest.raises(ValidationError):
        candidate(latitude=float("nan"))
    with pytest.raises(ValidationError):
        candidate(longitude=float("inf"))
    with pytest.raises(ValidationError):
        candidate(raw_score=float("nan"))


def test_provider_contract_enforces_outcome_semantics_and_bounds() -> None:
    result = GeographicProviderResult(
        provider="osv5m-baseline",
        model_id="osv5m/baseline",
        model_revision="revision-1",
        source_family="osv5m_family",
        status="completed",
        device="cpu",
        duration_ms=12,
        score_semantics="direct_regression",
        candidates=(candidate(raw_score=None),),
    )
    assert "confidence" not in result.model_dump()
    with pytest.raises(ValidationError):
        GeographicProviderResult.model_validate(
            {
                **result.model_dump(),
                "status": "failed",
                "reason_code": "worker_failed",
            }
        )
    with pytest.raises(ValidationError):
        GeographicProviderResult(
            provider="plonk",
            model_id="nicolas-dufour/PLONK_YFCC",
            model_revision="revision-1",
            source_family="yfcc_family",
            status="failed",
            device="cpu",
            duration_ms=1,
            score_semantics="sample_density",
        )


def test_geoclip_adapter_preserves_raw_similarity_as_similarity_not_confidence() -> None:
    result = InferenceResult(
        provider_id="geoclip-global-v1",
        provider_revision="5.0.0",
        model_name="GeoCLIP",
        model_revision="model-revision",
        runtime_revision="runtime-revision",
        classification="real",
        status="succeeded",
        device="cuda:0",
        dtype="float32",
        runtime_ms=20,
        score_semantics="raw_gallery_similarity",
        normalization_method="stable_rank",
        calibration_state="uncalibrated",
        candidates=(
            InferenceCandidate(
                rank=1,
                original_rank=3,
                latitude=41.0,
                longitude=29.0,
                raw_score=0.77,
                limitations=("uncalibrated",),
            ),
        ),
        warnings=("warning.global_prediction_uncalibrated",),
        provenance=InferenceProvenance(
            provider_id="geoclip-global-v1",
            provider_revision="5.0.0",
            model_revision="model-revision",
            runtime_revision="runtime-revision",
            source_kind="verified_model",
        ),
    )
    normalized = normalize_geoclip_result(result)
    assert normalized.status == "completed"
    assert normalized.source_family == "mp16_family"
    assert normalized.score_semantics == "similarity"
    assert normalized.candidates[0].raw_score == 0.77
    assert "confidence" not in normalized.model_dump_json()
