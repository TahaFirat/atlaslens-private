from __future__ import annotations

from fastapi.testclient import TestClient

from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    GlobalProviderStatus,
    InvocationContext,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.storage import LocalImageHandle
from conftest import image_bytes, upload, wait_for_terminal


class OperationalGlobalProvider:
    descriptor = ProviderDescriptor(
        id="phase5-operational-test",
        kind="global_geolocation",
        version="test-only",
        execution_boundary="local",
        criticality="optional",
        available=True,
        model_name="test-only",
    )

    def status(self) -> GlobalProviderStatus:
        return GlobalProviderStatus(
            status="ready",
            installed=True,
            verified=True,
            model_name="test-only",
            model_revision="test-only",
            device="cpu",
            calibration_state="uncalibrated",
        )

    async def predict(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[GlobalPredictionResult]:
        coordinates = [
            (40.0, -74.0),
            (51.5, -0.1),
            (35.7, 139.7),
            (-33.9, 151.2),
            (-23.6, -46.6),
        ]
        hypotheses = [
            GlobalPredictionHypothesis(
                rank=rank,
                original_rank=rank,
                latitude=latitude,
                longitude=longitude,
                raw_score=0.25 / rank,
                score_type="uncalibrated_gallery_softmax",
                normalization_method="softmax_over_fixed_gallery",
                calibration_state="uncalibrated",
                limitations=["test_only_provider"],
            )
            for rank, (latitude, longitude) in enumerate(coordinates, start=1)
        ]
        return ProviderOutcome.succeeded(
            GlobalPredictionResult(
                provider_id=self.descriptor.id,
                model_name="test-only",
                model_revision="test-only",
                implementation_revision="test-only",
                device="cpu",
                dtype="float32",
                inference_ms=2,
                hypotheses=hypotheses,
            )
        )


def test_real_application_path_exposes_status_and_model_candidates(client_factory) -> None:
    client: TestClient = client_factory(
        global_model_enabled=True,
        global_provider=OperationalGlobalProvider(),
    )

    capability = client.get("/api/v1/capabilities").json()["providers"][
        "global_geolocation"
    ]
    assert capability["installed"] is True
    assert capability["verified"] is True
    assert capability["operational_status"] == "ready"
    assert capability["device"] == "cpu"
    assert capability["calibration_state"] == "uncalibrated"

    accepted = upload(client, image_bytes("JPEG"))
    assert accepted.status_code == 202
    analysis = wait_for_terminal(client, accepted.json()["id"])

    assert len(analysis["candidates"]) == 5
    assert all(candidate["confidence"] is None for candidate in analysis["candidates"])
    assert all(candidate["radius_km"] >= 750 for candidate in analysis["candidates"])
    assert all(candidate["verified"] is False for candidate in analysis["candidates"])
    assert all(
        candidate["model_prediction"]["score_type"]
        == "uncalibrated_gallery_softmax"
        for candidate in analysis["candidates"]
    )
