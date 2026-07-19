from __future__ import annotations

import asyncio
import base64
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from atlaslens_api.config import Settings
from atlaslens_api.image_processing import CloudSafeDerivative, SafeImageProcessor
from atlaslens_api.providers.base import InvocationContext, OutcomeStatus
from atlaslens_api.providers.exif import extract_exif
from atlaslens_api.providers.ocr import redact_ocr_text
from atlaslens_api.providers.openai_vision import OpenAIVisionClueProvider
from atlaslens_api.providers.quality import analyze_quality
from atlaslens_api.schemas import AnalysisMode
from atlaslens_api.storage import LocalImageHandle, LocalTemporaryStorage
from conftest import gps_jpeg, image_bytes


def context(mode: AnalysisMode = AnalysisMode.CLOUD_ASSISTED) -> InvocationContext:
    return InvocationContext(
        analysis_id=uuid4(),
        request_id="provider-test-request",
        mode=mode,
        cloud_consent=mode == AnalysisMode.CLOUD_ASSISTED,
        deadline=datetime.now(UTC) + timedelta(seconds=10),
        cancellation=asyncio.Event(),
    )


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(41.015137, 28.97953), (-33.86882, 151.209296), (40.7128, -74.006), (-34.6, -58.4)],
)
def test_exif_gps_all_hemispheres(tmp_path: Path, latitude: float, longitude: float) -> None:
    path = tmp_path / "gps.jpg"
    path.write_bytes(gps_jpeg(latitude, longitude))
    result = extract_exif(path)
    assert result is not None
    assert result.latitude == pytest.approx(latitude, abs=1e-5)
    assert result.longitude == pytest.approx(longitude, abs=1e-5)
    assert result.altitude_m == pytest.approx(12.5)


def test_exif_without_gps_abstains(tmp_path: Path) -> None:
    path = tmp_path / "plain.jpg"
    path.write_bytes(image_bytes())
    assert extract_exif(path) is None


def test_quality_metrics_are_real_and_bounded(tmp_path: Path) -> None:
    dark = tmp_path / "dark.png"
    dark.write_bytes(image_bytes("PNG", color=(0, 0, 0), size=(100, 100)))
    result = analyze_quality(dark)
    assert 0 <= result.blur_score <= 1
    assert result.brightness_score == 0
    assert "quality.severe_underexposure" in result.warnings
    assert "quality.severe_blur" in result.warnings


def test_ocr_redacts_obvious_sensitive_tokens() -> None:
    raw = "Contact jane@example.com or +90 555 123 45 67\nPlate AB123CD\nCafe Bosphorus"
    snippets = redact_ocr_text(raw)
    rendered = " ".join(snippets)
    assert "jane@example.com" not in rendered
    assert "555 123" not in rendered
    assert "AB123CD" not in rendered
    assert "[redacted-email]" in rendered
    assert "[redacted-phone]" in rendered
    assert "Cafe Bosphorus" in rendered


class FakeResponses:
    def __init__(self, output: object, *, delay: float = 0) -> None:
        self.output = output
        self.delay = delay
        self.calls = 0

    async def parse(self, **_: object) -> object:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(output_parsed=self.output)


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def derivative() -> CloudSafeDerivative:
    return CloudSafeDerivative(
        data_url="data:image/jpeg;base64,AA==",
        byte_count=1,
        width=1,
        height=1,
    )


@pytest.mark.asyncio
async def test_structured_cloud_result_is_capped_and_broad() -> None:
    responses = FakeResponses(
        {
            "abstained": False,
            "hypotheses": [
                {
                    "label": "Istanbul region",
                    "latitude": 41.0,
                    "longitude": 29.0,
                    "radius_km": 2,
                    "confidence": 0.91,
                    "granularity": "exact_metadata",
                    "country_code": "tr",
                    "visible_clues": ["coastal terrain", "urban ferry"],
                }
            ],
        }
    )
    provider = OpenAIVisionClueProvider(
        api_key="test-key",
        model="test-model",
        timeout_seconds=1,
        client=FakeClient(responses),
    )
    outcome = await provider.extract(derivative(), context())
    assert outcome.status == OutcomeStatus.SUCCEEDED
    assert outcome.value is not None
    hypothesis = outcome.value.hypotheses[0]
    assert hypothesis.confidence == 0.40
    assert hypothesis.radius_km == 25
    assert hypothesis.granularity == "broad_area"
    assert hypothesis.country_code == "TR"
    assert responses.calls == 1


@pytest.mark.asyncio
async def test_invalid_cloud_schema_degrades_without_candidate() -> None:
    responses = FakeResponses({"abstained": False, "hypotheses": [{"label": "missing"}]})
    provider = OpenAIVisionClueProvider(
        api_key="test-key",
        model="test-model",
        timeout_seconds=1,
        client=FakeClient(responses),
    )
    outcome = await provider.extract(derivative(), context())
    assert outcome.status == OutcomeStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.code == "invalid_output"
    assert outcome.failure.retryable is False


@pytest.mark.asyncio
async def test_cloud_timeout_retries_are_bounded() -> None:
    responses = FakeResponses({}, delay=0.05)
    provider = OpenAIVisionClueProvider(
        api_key="test-key",
        model="test-model",
        timeout_seconds=0.005,
        client=FakeClient(responses),
    )
    outcome = await provider.extract(derivative(), context())
    assert outcome.status == OutcomeStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.code == "timeout"
    assert outcome.failure.attempts == 3
    assert responses.calls == 3


@pytest.mark.asyncio
async def test_local_only_never_calls_cloud_client() -> None:
    responses = FakeResponses({})
    provider = OpenAIVisionClueProvider(
        api_key="test-key",
        model="test-model",
        timeout_seconds=1,
        client=FakeClient(responses),
    )
    outcome = await provider.extract(derivative(), context(AnalysisMode.LOCAL_ONLY))
    assert outcome.status == OutcomeStatus.SKIPPED
    assert outcome.failure is not None
    assert outcome.failure.code == "privacy_denied"
    assert responses.calls == 0


@pytest.mark.asyncio
async def test_cloud_derivative_is_metadata_stripped_and_bounded(tmp_path: Path) -> None:
    input_path = tmp_path / "source.jpg"
    input_path.write_bytes(gps_jpeg(41.0, 29.0, size=(2000, 1000)))
    storage = LocalTemporaryStorage(tmp_path / "storage")
    await storage.initialize()
    processor = SafeImageProcessor(
        Settings(_env_file=None, temp_storage_dir=tmp_path / "storage"), storage
    )
    result = await processor.cloud_derivative(
        LocalImageHandle(key="0" * 32 + ".jpg", path=input_path)
    )
    payload = base64.b64decode(result.data_url.split(",", maxsplit=1)[1])
    with Image.open(io.BytesIO(payload)) as decoded:
        assert not decoded.getexif()
        assert max(decoded.size) <= 1568
