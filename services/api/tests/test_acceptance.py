from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from atlaslens_api.acceptance import run_acceptance
from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    GlobalProviderStatus,
    InvocationContext,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.storage import LocalImageHandle


class TestGlobalProvider:
    descriptor = ProviderDescriptor(
        id="test-global",
        kind="global_geolocation",
        version="test",
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
            model_revision="test",
            device="cpu",
            calibration_state="uncalibrated",
        )

    async def predict(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[GlobalPredictionResult]:
        hypotheses = [
            GlobalPredictionHypothesis(
                rank=index,
                original_rank=index,
                latitude=float(index),
                longitude=float(index * 10),
                raw_score=0.1 / index,
                score_type="uncalibrated_gallery_softmax",
                normalization_method="softmax_over_fixed_gallery",
                calibration_state="uncalibrated",
                limitations=["test_only"],
            )
            for index in range(1, 6)
        ]
        return ProviderOutcome.succeeded(
            GlobalPredictionResult(
                provider_id=self.descriptor.id,
                model_name="test-only",
                model_revision="test",
                implementation_revision="test",
                device="cpu",
                dtype="float32",
                inference_ms=1,
                hypotheses=hypotheses,
            )
        )


@pytest.mark.asyncio
async def test_acceptance_processes_five_non_exif_images_without_modifying_them(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    before: dict[str, bytes] = {}
    for index in range(5):
        path = images / f"private-name-{index}.jpg"
        Image.new("RGB", (32, 32), color=(index, 40, 80)).save(path, format="JPEG")
        before[path.name] = path.read_bytes()

    report = await run_acceptance(images, tmp_path / "reports", TestGlobalProvider())

    assert report["status"] == "passed"
    assert report["summary"] == {
        "discovered_images": 5,
        "valid_images": 5,
        "non_exif_images": 5,
        "images_with_candidates": 5,
        "non_empty_candidate_rate": 1.0,
        "median_runtime_ms": report["summary"]["median_runtime_ms"],
        "p95_runtime_ms": report["summary"]["p95_runtime_ms"],
    }
    assert all(path.read_bytes() == before[path.name] for path in images.iterdir())
    serialized = (tmp_path / "reports" / "acceptance.json").read_text(encoding="utf-8")
    assert "private-name" not in serialized
    assert '"external_transfer": false' in serialized
