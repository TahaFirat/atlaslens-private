from __future__ import annotations

import asyncio
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from atlaslens_api.phase6b.ocr import (
    PaddleOCRProvider,
    PreferredOCRProvider,
    RawPaddleOCRLine,
    extract_ocr_pattern_signals,
)
from atlaslens_api.providers.base import (
    InvocationContext,
    OCRResult,
    OutcomeStatus,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.schemas import AnalysisMode, GeoPoint, PlaceEvidenceSummary
from atlaslens_api.storage import LocalImageHandle


def context() -> InvocationContext:
    return InvocationContext(
        analysis_id=uuid4(),
        request_id="phase6b-ocr-test",
        mode=AnalysisMode.LOCAL_ONLY,
        cloud_consent=False,
        deadline=datetime.now(UTC) + timedelta(seconds=5),
        cancellation=asyncio.Event(),
    )


def image_handle(tmp_path: Path, *, orientation: int | None = None) -> LocalImageHandle:
    path = tmp_path / "ocr.jpg"
    image = Image.new("RGB", (100, 50), "white")
    if orientation is None:
        image.save(path)
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(path, exif=exif)
    return LocalImageHandle(key="ocr.upload", path=path)


def line(
    text: str,
    *,
    confidence: float = 0.9,
    polygon: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = ((5, 5), (90, 5), (90, 25), (5, 25)),
    crop_id: str = "standard",
) -> RawPaddleOCRLine:
    return RawPaddleOCRLine(
        text=text,
        confidence=confidence,
        polygon=polygon,
        crop_id=crop_id,
    )


class StubPaddleWorker:
    def __init__(self, lines: tuple[RawPaddleOCRLine, ...]) -> None:
        self.lines = lines
        self.received_size: tuple[int, int] | None = None
        self.scales: tuple[float, ...] | None = None
        self.closed = False

    async def infer(
        self,
        image_bytes: bytes,
        *,
        scales: tuple[float, ...],
        timeout_seconds: float,
        cancellation: asyncio.Event,
    ) -> tuple[RawPaddleOCRLine, ...]:
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.received_size = image.size
        self.scales = scales
        assert timeout_seconds > 0
        assert not cancellation.is_set()
        return self.lines

    async def close(self) -> None:
        self.closed = True


class StubPlaceService:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def resolve(self, observations: object) -> tuple[PlaceEvidenceSummary, ...]:
        self.texts = [item.text for item in observations]  # type: ignore[union-attr]
        return (
            PlaceEvidenceSummary(
                matched_entity="İstanbul",
                normalized_name="İstanbul",
                country_code="TR",
                center=GeoPoint(latitude=41.0, longitude=29.0),
                match_type="city",
                text_similarity=0.95,
                ambiguity_count=1,
                evidence_strength=0.9,
                source="local-test-gazetteer",
                dataset_version="test-v1",
                license="test-only",
            ),
        )


@pytest.mark.asyncio
async def test_paddle_success_preserves_turkish_and_deduplicates_overlapping_boxes(
    tmp_path: Path,
) -> None:
    worker = StubPaddleWorker(
        (
            line("İstanbul Şişli", confidence=0.91),
            line("İSTANBUL ŞİŞLİ", confidence=0.80, crop_id="enlarged"),
            line("Çığ öyküsü, ıslak güneş", polygon=((5, 28), (95, 28), (95, 45), (5, 45))),
            line("$$$$"),
            line("AAAAAA"),
            line("weak text", confidence=0.2),
        )
    )
    place_service = StubPlaceService()
    provider = PaddleOCRProvider(
        enabled=True,
        worker=worker,
        artifact_verified=True,
        provider_version="3.7.0",
        place_service=place_service,  # type: ignore[arg-type]
    )
    outcome = await provider.extract(image_handle(tmp_path), context())
    assert outcome.status == OutcomeStatus.SUCCEEDED
    assert outcome.value is not None
    assert len(outcome.value.blocks) == 2
    first = outcome.value.blocks[0]
    assert first.redacted_text == "İstanbul Şişli"
    assert first.normalized_text == "istanbul şişli"
    assert outcome.value.blocks[1].redacted_text == "Çığ öyküsü, ıslak güneş"
    assert outcome.value.place_matches[0].country_code == "TR"
    assert place_service.texts == ["İstanbul Şişli", "Çığ öyküsü, ıslak güneş"]
    assert worker.scales == (1.0, 1.5)


@pytest.mark.asyncio
async def test_exif_orientation_is_applied_before_worker_and_metadata_is_stripped(
    tmp_path: Path,
) -> None:
    worker = StubPaddleWorker((line("Ankara", polygon=((1, 1), (40, 1), (40, 20), (1, 20))),))
    provider = PaddleOCRProvider(
        enabled=True,
        worker=worker,
        artifact_verified=True,
        provider_version="3.7.0",
    )
    outcome = await provider.extract(image_handle(tmp_path, orientation=6), context())
    assert outcome.status == OutcomeStatus.SUCCEEDED
    assert worker.received_size == (50, 100)


class StubOCRProvider:
    def __init__(
        self,
        provider_id: str,
        outcome: ProviderOutcome[OCRResult],
        *,
        available: bool,
    ) -> None:
        self.outcome = outcome
        self.calls = 0
        self.descriptor = ProviderDescriptor(
            id=provider_id,
            kind="ocr",
            version="test",
            execution_boundary="local",
            criticality="optional",
            available=available,
            unavailable_reason_code=None if available else "missing_dependency",
        )

    async def extract(
        self, handle: LocalImageHandle, invocation: InvocationContext
    ) -> ProviderOutcome[OCRResult]:
        self.calls += 1
        return self.outcome


@pytest.mark.asyncio
async def test_rapidocr_fallback_and_fully_disabled_state(tmp_path: Path) -> None:
    paddle = StubOCRProvider(
        "paddle",
        ProviderOutcome.skipped("missing_dependency"),
        available=False,
    )
    rapid = StubOCRProvider(
        "rapid",
        ProviderOutcome.succeeded(OCRResult(redacted_snippets=["Ankara"])),
        available=True,
    )
    preferred = PreferredOCRProvider(paddle, rapid)
    result = await preferred.extract(image_handle(tmp_path), context())
    assert result.status == OutcomeStatus.SUCCEEDED
    assert rapid.calls == 1
    disabled = PreferredOCRProvider(
        paddle,
        StubOCRProvider(
            "rapid-disabled",
            ProviderOutcome.skipped("missing_dependency"),
            available=False,
        ),
    )
    disabled_result = await disabled.extract(image_handle(tmp_path), context())
    assert disabled.descriptor.available is False
    assert disabled_result.status == OutcomeStatus.SKIPPED


def test_pattern_extraction_is_bounded_and_does_not_treat_any_number_as_place() -> None:
    signals = extract_ocr_pattern_signals(
        "example.com +90 555 123 45 67 Atatürk Caddesi No: 42; random 987654"
    )
    assert {(item.kind, item.value) for item in signals} == {
        ("domain", "example.com"),
        ("phone_country_prefix", "+90"),
        ("road_number", "42"),
    }
