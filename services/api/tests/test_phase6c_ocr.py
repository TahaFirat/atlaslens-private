from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from atlaslens_api.phase6b.ocr import PreferredOCRProvider
from atlaslens_api.phase6c.ocr import (
    NormalizedImageBox,
    OCRCropPass,
    OCRSegmentationRegion,
    Phase6COCRConfig,
    assess_ocr_country_support,
    build_ocr_crop_plan,
    build_place_token_evidence,
    load_phase6c_ocr_config,
    normalize_turkish_latin,
    to_phase6c_fusion_evidence,
)
from atlaslens_api.providers.base import (
    InvocationContext,
    OCRBlock,
    OCRPolygonPoint,
    OCRResult,
    OutcomeStatus,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.schemas import AnalysisMode, GeoPoint, PlaceEvidenceSummary
from atlaslens_api.storage import LocalImageHandle


def _context(*, seconds: float = 2.0) -> InvocationContext:
    return InvocationContext(
        analysis_id=uuid4(),
        request_id="phase6c-ocr-test",
        mode=AnalysisMode.LOCAL_ONLY,
        cloud_consent=False,
        deadline=datetime.now(UTC) + timedelta(seconds=seconds),
        cancellation=asyncio.Event(),
    )


def _block(
    text: str,
    *,
    provider: str = "paddleocr-ppocr-local",
    profile: str = "paddle-full-0",
    confidence: float = 0.91,
    sensitive: bool = False,
) -> OCRBlock:
    points = (
        OCRPolygonPoint(x=0.1, y=0.1),
        OCRPolygonPoint(x=0.8, y=0.1),
        OCRPolygonPoint(x=0.8, y=0.3),
        OCRPolygonPoint(x=0.1, y=0.3),
    )
    return OCRBlock(
        redacted_text=text,
        normalized_text="typed-redacted-input",
        script="latin",
        language_hints=["und-Latn"],
        confidence=confidence,
        bounding_polygon=points,
        provider=provider,
        profile=profile,
        sensitive_content=sensitive,
    )


def _place(
    name: str,
    *,
    latitude: float,
    longitude: float,
    country_code: str = "TR",
    match_type: str = "city",
    similarity: float = 0.96,
    ambiguity: int = 1,
    strength: float = 0.84,
) -> PlaceEvidenceSummary:
    return PlaceEvidenceSummary(
        matched_entity=name,
        normalized_name=name,
        country_code=country_code,
        region=None,
        center=GeoPoint(latitude=latitude, longitude=longitude),
        match_type=match_type,
        text_similarity=similarity,
        ambiguity_count=ambiguity,
        evidence_strength=strength,
        source="GeoNames",
        dataset_version="test-v1",
        license="CC BY 4.0",
    )


def test_checked_in_config_reserves_fallback_budget_and_rejects_starvation() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_phase6c_ocr_config(root / "config" / "ocr" / "phase6c-v1.json")
    assert config.paddle_attempt_timeout_seconds + config.rapidocr_reserved_seconds <= 45
    assert config.max_targeted_crops + config.max_rotated_crops + 1 <= 17
    with pytest.raises(ValidationError, match="RapidOCR reserve"):
        Phase6COCRConfig(
            total_timeout_seconds=10,
            paddle_attempt_timeout_seconds=9,
            rapidocr_reserved_seconds=2,
        )


def test_crop_plan_has_one_full_pass_and_bounded_targeted_rotated_passes() -> None:
    config = Phase6COCRConfig(max_targeted_crops=3, max_rotated_crops=2)
    regions = tuple(
        OCRSegmentationRegion(
            label=label,
            bounds=NormalizedImageBox(
                x_min=0.05 + index * 0.12,
                y_min=0.1,
                x_max=0.13 + index * 0.12,
                y_max=0.3,
            ),
            segmentation_score=0.9 - index * 0.05,
        )
        for index, label in enumerate(
            (
                "Signage - Advertisement",
                "Signage - Information",
                "Signage - Store",
                "Traffic Sign - Direction",
                "Lane Marking - Text",
            )
        )
    )
    plan = build_ocr_crop_plan(regions, config)
    assert sum(item.crop_kind == "full_image" for item in plan.passes) == 1
    assert len(plan.passes) == 1 + 3 + 2
    assert all(item.provenance for item in plan.passes)
    assert {item.rotation_degrees for item in plan.passes} == {-90, 0, 90}
    assert all(
        item.segmentation_label is not None
        for item in plan.passes
        if item.crop_kind == "segmentation_region"
    )


def test_turkish_normalization_keeps_dotted_and_dotless_i_and_diacritics() -> None:
    assert normalize_turkish_latin("IĞDIR İZMİR ÇĞÖŞÜ ıi") == "ığdır izmir çğöşü ıi"


def test_place_tokens_have_exact_alias_fuzzy_and_independence_semantics() -> None:
    config = Phase6COCRConfig()
    crop = OCRCropPass(
        crop_id="full-0",
        crop_kind="full_image",
        bounds=NormalizedImageBox(x_min=0, y_min=0, x_max=1, y_max=1),
        scales=(1.0, 1.5),
        rotation_degrees=0,
        provenance="full-image",
    )
    result = OCRResult(
        redacted_snippets=["IĞDIR Belediyesi", "CAG UNIVERSITESI", "Ankaraa"],
        blocks=[
            _block("IĞDIR Belediyesi"),
            _block("CAG UNIVERSITESI", confidence=0.88),
            _block("Ankaraa Büyükşehir", confidence=0.77),
        ],
        place_matches=[
            _place("Iğdır", latitude=39.92, longitude=44.04),
            _place(
                "Çağ Üniversitesi",
                latitude=37.02,
                longitude=35.0,
                match_type="public_institution",
                similarity=0.95,
            ),
            _place(
                "Ankara",
                latitude=39.93,
                longitude=32.86,
                similarity=0.93,
                ambiguity=2,
            ),
        ],
    )
    evidence = build_place_token_evidence(
        result,
        config,
        crop_passes_by_profile={"paddle-full-0": crop},
    )
    by_name = {item.matched_entity: item for item in evidence}
    assert by_name["Iğdır"].match_semantics == "exact"
    assert by_name["Iğdır"].independent_support_eligible is True
    assert by_name["Çağ Üniversitesi"].match_semantics == "alias"
    assert by_name["Çağ Üniversitesi"].source.crop.scales == (1.0, 1.5)
    assert by_name["Ankara"].match_semantics == "fuzzy"
    assert by_name["Ankara"].independent_support_eligible is False
    assert by_name["Ankara"].location_confidence is None
    assert by_name["Ankara"].engine_confidence_semantics == "uncalibrated_ocr_engine_score"
    assert "Ankaraa" not in repr(by_name["Ankara"])
    batch = to_phase6c_fusion_evidence(
        evidence,
        provider="phase6c-ocr-place",
        model_id="paddle-rapid-local",
        model_revision="phase6c-v1",
        duration_ms=12,
    )
    assert batch.status == "completed"
    assert len(batch.candidates) == 2
    assert all(item.confidence is None and item.calibrated is False for item in batch.candidates)
    assert all(item.raw_value == 0.84 for item in batch.candidates)


def test_sensitive_or_unassociated_text_cannot_create_place_evidence() -> None:
    result = OCRResult(
        redacted_snippets=["[redacted-phone] İzmir"],
        blocks=[_block("[redacted-phone] İzmir", sensitive=True)],
        place_matches=[_place("İzmir", latitude=38.42, longitude=27.14)],
    )
    assert build_place_token_evidence(result, Phase6COCRConfig()) == ()
    unrelated = result.model_copy(
        update={"blocks": [_block("ordinary storefront")], "redacted_snippets": []}
    )
    assert build_place_token_evidence(unrelated, Phase6COCRConfig()) == ()


def test_country_support_and_contradiction_use_only_independent_place_tokens() -> None:
    result = OCRResult(
        redacted_snippets=["İzmir", "Boston", "Ankaraa"],
        blocks=[_block("İzmir"), _block("Boston"), _block("Ankaraa")],
        place_matches=[
            _place("İzmir", latitude=38.42, longitude=27.14),
            _place("Boston", latitude=42.36, longitude=-71.06, country_code="US"),
            _place(
                "Ankara",
                latitude=39.93,
                longitude=32.86,
                ambiguity=2,
                similarity=0.93,
            ),
        ],
    )
    evidence = build_place_token_evidence(result, Phase6COCRConfig())
    assessment = assess_ocr_country_support(evidence, "tr")
    assert assessment.independent_support is True
    assert assessment.contradiction_present is True
    assert len(assessment.neutral_evidence_ids) == 1


def test_fusion_adapter_abstains_without_specific_unambiguous_token() -> None:
    ambiguous = OCRResult(
        redacted_snippets=["Ankaraa"],
        blocks=[_block("Ankaraa")],
        place_matches=[
            _place(
                "Ankara",
                latitude=39.93,
                longitude=32.86,
                ambiguity=2,
                similarity=0.93,
            )
        ],
    )
    evidence = build_place_token_evidence(ambiguous, Phase6COCRConfig())
    batch = to_phase6c_fusion_evidence(
        evidence,
        provider="phase6c-ocr-place",
        model_id="paddle-rapid-local",
        model_revision="phase6c-v1",
        duration_ms=12,
    )
    assert batch.status == "abstained"
    assert batch.reason_code == "no_specific_unambiguous_place_token"


class _StubOCRProvider:
    def __init__(
        self,
        provider_id: str,
        *,
        delay: float,
        outcome: ProviderOutcome[OCRResult],
    ) -> None:
        self.delay = delay
        self.outcome = outcome
        self.calls = 0
        self.cancelled = False
        self.descriptor = ProviderDescriptor(
            id=provider_id,
            kind="ocr",
            version="test",
            execution_boundary="local",
            criticality="optional",
            available=True,
        )

    async def extract(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[OCRResult]:
        self.calls += 1
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.cancelled = True
        return self.outcome


@pytest.mark.asyncio
async def test_paddle_timeout_leaves_a_real_rapidocr_fallback_budget(tmp_path: Path) -> None:
    handle = LocalImageHandle(key="test", path=tmp_path / "unused.jpg")
    paddle = _StubOCRProvider(
        "paddle",
        delay=5,
        outcome=ProviderOutcome.abstained(),
    )
    rapid = _StubOCRProvider(
        "rapid",
        delay=0,
        outcome=ProviderOutcome.succeeded(OCRResult(redacted_snippets=["İzmir"])),
    )
    preferred = PreferredOCRProvider(
        paddle,
        rapid,
        total_timeout_seconds=0.3,
        paddle_attempt_timeout_seconds=0.1,
        rapidocr_reserved_seconds=0.15,
    )
    started = time.monotonic()
    outcome = await preferred.extract(handle, _context())
    assert time.monotonic() - started < 0.4
    assert outcome.status == OutcomeStatus.SUCCEEDED
    assert paddle.cancelled is True
    assert rapid.calls == 1


@pytest.mark.asyncio
async def test_preferred_ocr_bounds_the_complete_slow_fallback_chain(tmp_path: Path) -> None:
    handle = LocalImageHandle(key="test", path=tmp_path / "unused.jpg")
    paddle = _StubOCRProvider(
        "paddle",
        delay=5,
        outcome=ProviderOutcome.abstained(),
    )
    rapid = _StubOCRProvider(
        "rapid",
        delay=5,
        outcome=ProviderOutcome.abstained(),
    )
    preferred = PreferredOCRProvider(
        paddle,
        rapid,
        total_timeout_seconds=0.2,
        paddle_attempt_timeout_seconds=0.05,
        rapidocr_reserved_seconds=0.1,
    )
    started = time.monotonic()
    outcome = await preferred.extract(handle, _context())
    assert time.monotonic() - started < 0.3
    assert outcome.status == OutcomeStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.subreason_code == "rapidocr_fallback_budget_exhausted"


@pytest.mark.asyncio
async def test_preferred_ocr_is_promptly_cancellable(tmp_path: Path) -> None:
    handle = LocalImageHandle(key="test", path=tmp_path / "unused.jpg")
    paddle = _StubOCRProvider(
        "paddle",
        delay=5,
        outcome=ProviderOutcome.abstained(),
    )
    rapid = _StubOCRProvider(
        "rapid",
        delay=0,
        outcome=ProviderOutcome.abstained(),
    )
    invocation = _context()
    preferred = PreferredOCRProvider(
        paddle,
        rapid,
        total_timeout_seconds=0.5,
        paddle_attempt_timeout_seconds=0.25,
        rapidocr_reserved_seconds=0.2,
    )
    task = asyncio.create_task(preferred.extract(handle, invocation))
    await asyncio.sleep(0.02)
    invocation.cancellation.set()
    outcome = await asyncio.wait_for(task, timeout=0.2)
    assert outcome.status == OutcomeStatus.SKIPPED
    assert paddle.cancelled is True
    assert rapid.calls == 0
