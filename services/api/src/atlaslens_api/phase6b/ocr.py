from __future__ import annotations

import asyncio
import io
import math
import re
import sqlite3
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol

from PIL import Image, ImageOps, UnidentifiedImageError

from atlaslens_api.place_evidence.models import OCRTextObservation
from atlaslens_api.place_evidence.normalization import (
    detect_script,
    language_hints,
    normalize_search_text,
    sanitize_unicode,
)
from atlaslens_api.place_evidence.service import PlaceEvidenceService
from atlaslens_api.providers.base import (
    InvocationContext,
    OCRBlock,
    OCRPolygonPoint,
    OCRProvider,
    OCRResult,
    OutcomeStatus,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.providers.ocr import redact_sensitive_text
from atlaslens_api.storage import LocalImageHandle

_DOMAIN = re.compile(r"(?<![\w@])(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b", re.IGNORECASE)
_PHONE_PREFIX = re.compile(r"(?<!\w)\+(\d{1,3})[\s().-]+\d")
_ROAD_NUMBER = re.compile(
    r"\b(?:road|street|avenue|sokak|sokağı|cadde|caddesi|no)\s*[:.#-]?\s*(\d{1,5})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RawPaddleOCRLine:
    text: str
    confidence: float
    polygon: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    crop_id: str = "standard"


@dataclass(frozen=True, slots=True)
class OCRPatternSignal:
    kind: str
    value: str


class PaddleOCRWorker(Protocol):
    """Process-isolated official PaddleOCR/PP-OCR worker boundary."""

    async def infer(
        self,
        image_bytes: bytes,
        *,
        scales: tuple[float, ...],
        timeout_seconds: float,
        cancellation: asyncio.Event,
    ) -> Sequence[RawPaddleOCRLine]: ...

    async def close(self) -> None: ...


def extract_ocr_pattern_signals(text: str) -> tuple[OCRPatternSignal, ...]:
    """Extract bounded public pattern hints without retaining full phone numbers."""

    normalized = sanitize_unicode(text, max_length=240)
    signals: list[OCRPatternSignal] = []
    signals.extend(
        OCRPatternSignal(kind="domain", value=match.group(1).casefold()[:120])
        for match in _DOMAIN.finditer(normalized)
    )
    signals.extend(
        OCRPatternSignal(kind="phone_country_prefix", value=f"+{match.group(1)}")
        for match in _PHONE_PREFIX.finditer(normalized)
    )
    signals.extend(
        OCRPatternSignal(kind="road_number", value=match.group(1))
        for match in _ROAD_NUMBER.finditer(normalized)
    )
    unique: dict[tuple[str, str], OCRPatternSignal] = {}
    for item in signals:
        unique[(item.kind, item.value)] = item
    return tuple(unique.values())[:12]


class PaddleOCRProvider:
    """Offline, orientation-corrected PP-OCR adapter with privacy-first output."""

    def __init__(
        self,
        *,
        enabled: bool,
        worker: PaddleOCRWorker | None,
        artifact_verified: bool,
        place_service: PlaceEvidenceService | None = None,
        provider_version: str = "unprepared",
        model_name: str = "PP-OCR-mobile",
        device: str = "cpu",
        timeout_seconds: float = 15.0,
        scales: tuple[float, ...] = (1.0, 1.5),
        minimum_confidence: float = 0.45,
        duplicate_iou: float = 0.55,
        max_input_bytes: int = 20 * 1024 * 1024,
        max_decoded_pixels: int = 16_000_000,
        max_side: int = 4096,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("PaddleOCR device must be cpu or cuda")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("PaddleOCR timeout must be positive")
        if (
            not scales
            or len(scales) > 3
            or any(not math.isfinite(value) or not 0.5 <= value <= 3 for value in scales)
        ):
            raise ValueError("PaddleOCR scales are invalid")
        if not 0 <= minimum_confidence <= 1 or not 0 <= duplicate_iou <= 1:
            raise ValueError("PaddleOCR thresholds must be within [0, 1]")
        if max_input_bytes <= 0 or max_decoded_pixels <= 0 or max_side <= 0:
            raise ValueError("PaddleOCR input bounds must be positive")
        if not enabled:
            reason = "disabled"
        elif worker is None:
            reason = "missing_dependency"
        elif not artifact_verified:
            reason = "weights_incomplete"
        else:
            reason = None
        self._worker = worker if reason is None else None
        self._place_service = place_service
        self._device = device
        self._timeout = timeout_seconds
        self._scales = scales
        self._minimum_confidence = minimum_confidence
        self._duplicate_iou = duplicate_iou
        self._max_input_bytes = max_input_bytes
        self._max_decoded_pixels = max_decoded_pixels
        self._max_side = max_side
        self.descriptor = ProviderDescriptor(
            id="paddleocr-ppocr-local",
            kind="ocr",
            version=provider_version,
            execution_boundary="local",
            criticality="optional",
            available=reason is None,
            unavailable_reason_code=reason,
            model_name=model_name,
        )

    async def extract(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[OCRResult]:
        if not self.descriptor.available or self._worker is None:
            return ProviderOutcome.skipped(self.descriptor.unavailable_reason_code or "unavailable")
        if context.cancellation.is_set():
            return ProviderOutcome.skipped("unavailable")
        started = time.monotonic()
        try:
            image_bytes, width, height = await asyncio.to_thread(self._read_oriented_image, handle)
        except (OSError, ValueError, UnidentifiedImageError):
            return ProviderOutcome.failed(
                "unsupported_input",
                retryable=False,
                attempts=1,
                duration_ms=_elapsed_ms(started),
                subreason_code="paddleocr_image_bounds_or_decode_invalid",
            )
        try:
            remaining = (context.deadline - datetime.now(UTC)).total_seconds()
            timeout = min(self._timeout, max(0.05, remaining))
            async with asyncio.timeout(timeout):
                lines = await self._worker.infer(
                    image_bytes,
                    scales=self._scales,
                    timeout_seconds=timeout,
                    cancellation=context.cancellation,
                )
        except TimeoutError:
            return ProviderOutcome.failed(
                "inference_timeout",
                retryable=False,
                attempts=1,
                duration_ms=_elapsed_ms(started),
                subreason_code="paddleocr_worker_timeout",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ProviderOutcome.failed(
                "internal_provider_error",
                retryable=False,
                attempts=1,
                duration_ms=_elapsed_ms(started),
                subreason_code="paddleocr_worker_failed",
            )
        finally:
            del image_bytes
        blocks, observations = self._normalize_lines(lines, width=width, height=height)
        if not blocks:
            return ProviderOutcome.abstained()
        place_matches = []
        if self._place_service is not None and observations:
            try:
                place_matches = list(
                    await asyncio.to_thread(self._place_service.resolve, observations)
                )
            except (OSError, ValueError, sqlite3.Error):
                place_matches = []
        observations.clear()
        return ProviderOutcome.succeeded(
            OCRResult(
                redacted_snippets=list(dict.fromkeys(block.redacted_text for block in blocks))[:5],
                blocks=blocks[:32],
                place_matches=place_matches[:12],
            )
        )

    async def close(self) -> None:
        if self._worker is not None:
            await self._worker.close()

    def safe_status(self) -> dict[str, str | bool | int]:
        return {
            "provider_id": self.descriptor.id,
            "available": self.descriptor.available,
            "reason_code": self.descriptor.unavailable_reason_code or "ready",
            "device": self._device,
            "offline": True,
            "scale_count": len(self._scales),
        }

    def _read_oriented_image(self, handle: LocalImageHandle) -> tuple[bytes, int, int]:
        size = handle.path.stat().st_size
        if not 0 < size <= self._max_input_bytes:
            raise ValueError("PaddleOCR image byte bound exceeded")
        with Image.open(handle.path) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            width, height = oriented.size
            if (
                width <= 0
                or height <= 0
                or max(width, height) > self._max_side
                or width * height > self._max_decoded_pixels
            ):
                raise ValueError("PaddleOCR image dimensions exceed bounds")
            output = io.BytesIO()
            oriented.save(output, format="PNG", optimize=False)
        image_bytes = output.getvalue()
        if not image_bytes or len(image_bytes) > self._max_input_bytes * 4:
            raise ValueError("PaddleOCR oriented derivative exceeds bounds")
        return image_bytes, width, height

    def _normalize_lines(
        self,
        lines: Sequence[RawPaddleOCRLine],
        *,
        width: int,
        height: int,
    ) -> tuple[list[OCRBlock], list[OCRTextObservation]]:
        selected: list[tuple[OCRBlock, OCRTextObservation, tuple[float, float, float, float]]] = []
        for line in lines[:128]:
            raw = sanitize_unicode(line.text)
            if (
                not raw
                or not math.isfinite(line.confidence)
                or not self._minimum_confidence <= line.confidence <= 1
                or self._looks_like_gibberish(raw)
            ):
                continue
            points: list[OCRPolygonPoint] = []
            for x, y in line.polygon:
                if (
                    not math.isfinite(x)
                    or not math.isfinite(y)
                    or not 0 <= x <= width
                    or not 0 <= y <= height
                ):
                    points = []
                    break
                points.append(OCRPolygonPoint(x=x / width, y=y / height))
            if len(points) != 4:
                continue
            redacted = redact_sensitive_text(raw)
            if not redacted.text:
                continue
            normalized = normalize_search_text(redacted.text)
            if not normalized:
                continue
            script = detect_script(raw)
            hints = language_hints(script, f"paddle-{line.crop_id}-latin")
            block = OCRBlock(
                redacted_text=redacted.text,
                normalized_text=normalized[:240],
                script=script,
                language_hints=hints,
                confidence=line.confidence,
                bounding_polygon=(points[0], points[1], points[2], points[3]),
                provider=self.descriptor.id,
                profile=f"paddle-{line.crop_id}"[:80],
                sensitive_content=redacted.sensitive,
            )
            observation = OCRTextObservation(
                text=raw,
                confidence=line.confidence,
                script=script,
                language_hints=tuple(hints),
            )
            bounds = self._bounds(points)
            duplicate = next(
                (
                    index
                    for index, (other, _, other_bounds) in enumerate(selected)
                    if other.normalized_text == block.normalized_text
                    and self._iou(bounds, other_bounds) >= self._duplicate_iou
                ),
                None,
            )
            if duplicate is None:
                selected.append((block, observation, bounds))
            elif block.confidence > selected[duplicate][0].confidence:
                selected[duplicate] = (block, observation, bounds)
        selected.sort(
            key=lambda item: (
                item[2][1],
                item[2][0],
                -item[0].confidence,
                item[0].normalized_text,
            )
        )
        # Raw OCR text remains only in ephemeral observations used by the local
        # place service. Sensitive observations never become place queries.
        return (
            [item[0] for item in selected[:32]],
            [item[1] for item in selected[:32] if not item[0].sensitive_content],
        )

    @staticmethod
    def _looks_like_gibberish(text: str) -> bool:
        alphanumeric = [character for character in text if character.isalnum()]
        if len(alphanumeric) < 2:
            return True
        visible = [character for character in text if not character.isspace()]
        if not visible or len(alphanumeric) / len(visible) < 0.45:
            return True
        folded = [unicodedata.normalize("NFKC", item).casefold() for item in alphanumeric]
        return len(folded) >= 5 and len(set(folded)) == 1

    @staticmethod
    def _bounds(points: Sequence[OCRPolygonPoint]) -> tuple[float, float, float, float]:
        return (
            min(item.x for item in points),
            min(item.y for item in points),
            max(item.x for item in points),
            max(item.y for item in points),
        )

    @staticmethod
    def _iou(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        intersection = intersection_width * intersection_height
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0


class PreferredOCRProvider:
    """PaddleOCR -> existing RapidOCR -> disabled selection without hidden success."""

    def __init__(
        self,
        paddle: OCRProvider,
        rapid: OCRProvider,
        *,
        total_timeout_seconds: float = 45.0,
        paddle_attempt_timeout_seconds: float = 28.0,
        rapidocr_reserved_seconds: float = 16.0,
    ) -> None:
        if (
            not math.isfinite(total_timeout_seconds)
            or not math.isfinite(paddle_attempt_timeout_seconds)
            or not math.isfinite(rapidocr_reserved_seconds)
            or total_timeout_seconds <= 0
            or paddle_attempt_timeout_seconds <= 0
            or rapidocr_reserved_seconds <= 0
            or paddle_attempt_timeout_seconds + rapidocr_reserved_seconds
            > total_timeout_seconds
        ):
            raise ValueError("preferred OCR timeout allocation is invalid")
        self._paddle = paddle
        self._rapid = rapid
        self._total_timeout = total_timeout_seconds
        self._paddle_attempt_timeout = paddle_attempt_timeout_seconds
        self._rapidocr_reserved = rapidocr_reserved_seconds
        available = paddle.descriptor.available or rapid.descriptor.available
        reason = None if available else "missing_dependency"
        self.descriptor = ProviderDescriptor(
            id="phase6b-preferred-local-ocr",
            kind="ocr",
            version="phase6b-v1",
            execution_boundary="local",
            criticality="optional",
            available=available,
            unavailable_reason_code=reason,
            model_name="PaddleOCR-with-RapidOCR-fallback",
        )

    async def extract(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[OCRResult]:
        if not self.descriptor.available:
            return ProviderOutcome.skipped(self.descriptor.unavailable_reason_code or "unavailable")
        started = time.monotonic()
        context_remaining = (context.deadline - datetime.now(UTC)).total_seconds()
        total_budget = min(self._total_timeout, max(0.0, context_remaining))
        if total_budget <= 0 or context.cancellation.is_set():
            return ProviderOutcome.skipped("unavailable")
        if self._rapid.descriptor.available:
            rapid_reserve = min(self._rapidocr_reserved, total_budget * 0.4)
            paddle_budget = min(
                total_budget,
                self._paddle_attempt_timeout,
                max(0.05, total_budget - rapid_reserve),
            )
        else:
            paddle_budget = total_budget
        paddle = await self._bounded_extract(
            self._paddle,
            handle,
            context,
            timeout_seconds=paddle_budget,
            timeout_subreason="paddleocr_attempt_budget_exhausted",
        )
        if paddle.status == OutcomeStatus.SUCCEEDED:
            return paddle
        if context.cancellation.is_set():
            return paddle
        remaining = self._total_timeout - (time.monotonic() - started)
        context_remaining = (context.deadline - datetime.now(UTC)).total_seconds()
        rapid_budget = min(remaining, context_remaining)
        if rapid_budget <= 0:
            return paddle
        rapid = await self._bounded_extract(
            self._rapid,
            handle,
            context,
            timeout_seconds=rapid_budget,
            timeout_subreason="rapidocr_fallback_budget_exhausted",
        )
        if rapid.status in {OutcomeStatus.SUCCEEDED, OutcomeStatus.ABSTAINED}:
            return rapid
        if paddle.status == OutcomeStatus.ABSTAINED:
            return paddle
        if rapid.status == OutcomeStatus.SKIPPED and paddle.status == OutcomeStatus.FAILED:
            return paddle
        return rapid

    @staticmethod
    async def _bounded_extract(
        provider: OCRProvider,
        handle: LocalImageHandle,
        context: InvocationContext,
        *,
        timeout_seconds: float,
        timeout_subreason: str,
    ) -> ProviderOutcome[OCRResult]:
        """Run one OCR attempt within its own deadline and the shared cancellation event."""

        started = time.monotonic()
        attempt_context = replace(
            context,
            deadline=min(
                context.deadline,
                datetime.now(UTC) + timedelta(seconds=timeout_seconds),
            ),
        )
        provider_task = asyncio.create_task(provider.extract(handle, attempt_context))
        cancellation_task = asyncio.create_task(context.cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {provider_task, cancellation_task},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_task in done and context.cancellation.is_set():
                provider_task.cancel()
                await asyncio.gather(provider_task, return_exceptions=True)
                return ProviderOutcome.skipped("unavailable")
            if provider_task in done:
                return provider_task.result()
            provider_task.cancel()
            await asyncio.gather(provider_task, return_exceptions=True)
            if context.cancellation.is_set():
                return ProviderOutcome.skipped("unavailable")
            return ProviderOutcome.failed(
                "inference_timeout",
                retryable=False,
                attempts=1,
                duration_ms=_elapsed_ms(started),
                subreason_code=timeout_subreason,
            )
        finally:
            cancellation_task.cancel()
            await asyncio.gather(cancellation_task, return_exceptions=True)

    def safe_status(self) -> dict[str, str | bool]:
        selected = (
            self._paddle.descriptor.id
            if self._paddle.descriptor.available
            else self._rapid.descriptor.id
            if self._rapid.descriptor.available
            else "disabled"
        )
        return {
            "provider_id": self.descriptor.id,
            "available": self.descriptor.available,
            "selected_provider": selected,
            "paddle_available": self._paddle.descriptor.available,
            "rapid_available": self._rapid.descriptor.available,
        }


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
