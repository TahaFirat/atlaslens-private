from __future__ import annotations

import asyncio

import pytest

from atlaslens_api.inference.models import (
    InferenceCandidate,
    InferenceProvenance,
    InferenceResult,
)
from atlaslens_api.phase6b.ensemble import Phase6BLocalModelEnsemble
from atlaslens_api.phase6b.fusion import Phase6BGeographicFusionEngine
from atlaslens_api.phase6b.models import GeographicCandidate, GeographicProviderResult


def provider_result(
    provider: str, model: str, family: str, latitude: float
) -> GeographicProviderResult:
    semantics = "direct_regression" if provider == "osv5m-baseline" else "sample_density"
    return GeographicProviderResult.model_validate(
        {
            "provider": provider,
            "model_id": model,
            "model_revision": "revision",
            "source_family": family,
            "status": "completed",
            "device": "cpu",
            "duration_ms": 1,
            "score_semantics": semantics,
            "candidates": (
                GeographicCandidate(
                    candidate_id=f"{provider}-one",
                    latitude=latitude,
                    longitude=29,
                    raw_score=None if semantics == "direct_regression" else 1,
                    provider_rank=1,
                ),
            ),
        }
    )


class StubOSV:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def predict(
        self, image_bytes: bytes, *, cancellation: asyncio.Event | None = None
    ) -> GeographicProviderResult:
        self.calls.append("osv")
        return provider_result("osv5m-baseline", "osv5m/baseline", "osv5m_family", 41.01)


class StubPlonk:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def predict(self, image_bytes: bytes, **kwargs: object) -> GeographicProviderResult:
        self.calls.append("plonk")
        return provider_result("plonk", "nicolas-dufour/PLONK_YFCC", "yfcc_family", 41.02)


def geoclip_result() -> InferenceResult:
    return InferenceResult(
        provider_id="geoclip-global-v1",
        provider_revision="5.0.0",
        model_name="GeoCLIP",
        model_revision="model-revision",
        runtime_revision="runtime-revision",
        classification="real",
        status="succeeded",
        device="cuda",
        dtype="float32",
        runtime_ms=1,
        score_semantics="raw_gallery_similarity",
        normalization_method="stable_rank",
        calibration_state="uncalibrated",
        candidates=(
            InferenceCandidate(
                rank=1,
                original_rank=1,
                latitude=41,
                longitude=29,
                raw_score=0.5,
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


@pytest.mark.asyncio
async def test_local_ensemble_runs_heavy_models_sequentially_and_fuses_all() -> None:
    calls: list[str] = []
    ensemble = Phase6BLocalModelEnsemble(
        osv5m=StubOSV(calls),
        plonk=StubPlonk(calls),  # type: ignore[arg-type]
        fusion=Phase6BGeographicFusionEngine(),
    )
    result = await ensemble.run(
        b"licensed-image",
        geoclip=geoclip_result(),
        scene=None,
        ocr=None,
        cancellation=asyncio.Event(),
    )
    assert calls == ["osv", "plonk"]
    assert [item.provider for item in result.model_predictions] == [
        "geoclip-global-v1",
        "osv5m-baseline",
        "plonk",
    ]
    assert result.fusion.candidates[0].provider_count == 3
    assert result.fusion.candidates[0].independent_family_count == 3
