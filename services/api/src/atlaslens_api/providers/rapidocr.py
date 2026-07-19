from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import math
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

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
    OCRResult,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.providers.ocr import redact_sensitive_text
from atlaslens_api.providers.rapidocr_worker import (
    ProcessRapidOCRWorker,
    RapidOCRWorker,
    RapidOCRWorkerFailure,
    RawOCRLine,
    ResolvedRapidOCRProfile,
)
from atlaslens_api.storage import LocalImageHandle

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,79}$")
_SAFE_LANGUAGE = re.compile(r"^[a-z0-9_-]{1,40}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RapidOCRProfileConfig:
    profile_id: str
    detector_file: str
    recognizer_file: str
    classifier_file: str | None = None
    detector_language: str = "multi"
    recognizer_language: str = "multi"
    ocr_version: str = "PP-OCRv6"
    model_type: str = "small"


class _ReceiptModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class RapidOCRArtifactReceipt(_ReceiptModel):
    path: str = Field(min_length=1, max_length=240)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(gt=0, le=512 * 1024 * 1024)


class RapidOCRProfileReceipt(_ReceiptModel):
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,79}$")
    detector_file: str = Field(min_length=1, max_length=240)
    recognizer_file: str = Field(min_length=1, max_length=240)
    classifier_file: str | None = Field(default=None, max_length=240)
    detector_language: str = Field(default="multi", max_length=40)
    recognizer_language: str = Field(default="multi", max_length=40)
    ocr_version: str = Field(default="PP-OCRv6", max_length=40)
    model_type: str = Field(default="small", max_length=40)


class RapidOCRInstallationReceipt(_ReceiptModel):
    schema_version: str = Field(pattern=r"^atlaslens-rapidocr-v1$")
    provider_version: str = Field(pattern=r"^3\.9\.1$")
    runtime_family: str = Field(pattern=r"^onnxruntime-(gpu|cpu)$")
    runtime_version: str = Field(pattern=r"^1\.27\.0$")
    profiles: tuple[RapidOCRProfileReceipt, ...] = Field(min_length=1, max_length=4)
    artifacts: tuple[RapidOCRArtifactReceipt, ...] = Field(min_length=2, max_length=12)


@dataclass(frozen=True, slots=True, repr=False)
class RapidOCRRuntimeConfig:
    model_root: Path
    profiles: tuple[RapidOCRProfileConfig, ...]
    device: str = "cpu"
    verified: bool = False

    def __repr__(self) -> str:
        return (
            "RapidOCRRuntimeConfig(model_root=<private>, "
            f"profiles={len(self.profiles)}, device={self.device!r}, verified={self.verified!r})"
        )

    def resolve_profiles(self) -> tuple[ResolvedRapidOCRProfile, ...]:
        if not self.verified:
            raise ValueError("RapidOCR installation is not verified")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("RapidOCR device must be cpu or cuda")
        if not 1 <= len(self.profiles) <= 4:
            raise ValueError("RapidOCR requires between one and four profiles")
        requested_root = self.model_root.expanduser()
        if requested_root.is_symlink():
            raise ValueError("RapidOCR model root cannot be a symlink")
        root = requested_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("RapidOCR model root must be a directory")
        resolved: list[ResolvedRapidOCRProfile] = []
        seen: set[str] = set()
        for profile in self.profiles:
            if not _PROFILE_ID.fullmatch(profile.profile_id) or profile.profile_id in seen:
                raise ValueError("RapidOCR profile id is invalid or duplicated")
            if not all(
                _SAFE_LANGUAGE.fullmatch(value)
                for value in (
                    profile.detector_language,
                    profile.recognizer_language,
                    profile.ocr_version,
                    profile.model_type,
                )
            ):
                raise ValueError("RapidOCR profile metadata is invalid")
            seen.add(profile.profile_id)
            detector = self._resolve_file(root, profile.detector_file)
            recognizer = self._resolve_file(root, profile.recognizer_file)
            classifier = (
                self._resolve_file(root, profile.classifier_file)
                if profile.classifier_file is not None
                else None
            )
            resolved.append(
                ResolvedRapidOCRProfile(
                    profile_id=profile.profile_id,
                    detector_path=str(detector),
                    recognizer_path=str(recognizer),
                    classifier_path=str(classifier) if classifier is not None else None,
                    detector_language=profile.detector_language,
                    recognizer_language=profile.recognizer_language,
                    ocr_version=profile.ocr_version,
                    model_type=profile.model_type,
                )
            )
        return tuple(resolved)

    @staticmethod
    def _resolve_file(root: Path, relative_name: str) -> Path:
        pure = PurePosixPath(relative_name)
        if (
            not relative_name
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in relative_name
        ):
            raise ValueError("RapidOCR artifact path is unsafe")
        requested = root.joinpath(*pure.parts)
        if requested.is_symlink():
            raise ValueError("RapidOCR artifact symlinks are not accepted")
        path = requested.resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise ValueError("RapidOCR artifact is outside the installation root")
        size = path.stat().st_size
        if not 0 < size <= 512 * 1024 * 1024:
            raise ValueError("RapidOCR artifact size is invalid")
        return path


def load_verified_rapidocr_runtime(
    model_root: Path, *, device: str
) -> RapidOCRRuntimeConfig | None:
    """Load only a fully receipted local installation; never discover or download."""

    try:
        requested_root = model_root.expanduser()
        if requested_root.is_symlink():
            return None
        root = requested_root.resolve(strict=True)
        if not root.is_dir():
            return None
        receipt_path = root / "receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            return None
        receipt = RapidOCRInstallationReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
        if device == "cuda" and receipt.runtime_family != "onnxruntime-gpu":
            return None
        artifact_by_path = {item.path: item for item in receipt.artifacts}
        if len(artifact_by_path) != len(receipt.artifacts):
            return None
        referenced = {
            path
            for profile in receipt.profiles
            for path in (
                profile.detector_file,
                profile.recognizer_file,
                profile.classifier_file,
            )
            if path is not None
        }
        if referenced != set(artifact_by_path):
            return None
        for relative, artifact in artifact_by_path.items():
            path = RapidOCRRuntimeConfig._resolve_file(root, relative)
            if path.stat().st_size != artifact.size_bytes:
                return None
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                return None
        profiles = tuple(
            RapidOCRProfileConfig(
                profile_id=item.profile_id,
                detector_file=item.detector_file,
                recognizer_file=item.recognizer_file,
                classifier_file=item.classifier_file,
                detector_language=item.detector_language,
                recognizer_language=item.recognizer_language,
                ocr_version=item.ocr_version,
                model_type=item.model_type,
            )
            for item in receipt.profiles
        )
        runtime = RapidOCRRuntimeConfig(
            model_root=root, profiles=profiles, device=device, verified=True
        )
        runtime.resolve_profiles()
        return runtime
    except (OSError, ValueError):
        return None


def _dependencies_available() -> bool:
    return all(
        importlib.util.find_spec(module) is not None
        for module in ("rapidocr", "onnxruntime", "cv2", "numpy")
    )


class RapidOCRProvider:
    """Offline local OCR with process-kill timeout and pre-persistence redaction."""

    def __init__(
        self,
        *,
        enabled: bool,
        runtime: RapidOCRRuntimeConfig | None,
        place_service: PlaceEvidenceService | None = None,
        timeout_seconds: float = 12.0,
        max_input_bytes: int = 20 * 1024 * 1024,
        max_decoded_pixels: int = 16_000_000,
        max_side: int = 4096,
        worker: RapidOCRWorker | None = None,
        dependency_probe: Callable[[], bool] = _dependencies_available,
    ) -> None:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("RapidOCR timeout must be positive and finite")
        if max_input_bytes <= 0 or max_decoded_pixels <= 0 or max_side <= 0:
            raise ValueError("RapidOCR image bounds must be positive")
        self._timeout = timeout_seconds
        self._max_input_bytes = max_input_bytes
        self._max_decoded_pixels = max_decoded_pixels
        self._max_side = max_side
        self._place_service = place_service
        self._worker: RapidOCRWorker | None = None
        self._resolved_profiles: tuple[ResolvedRapidOCRProfile, ...] = ()
        reason: str | None = None
        if not enabled:
            reason = "disabled"
        elif runtime is None:
            reason = "model_not_installed"
        elif not runtime.verified:
            reason = "weights_incomplete"
        elif not dependency_probe():
            reason = "missing_dependency"
        else:
            try:
                self._resolved_profiles = runtime.resolve_profiles()
            except (OSError, ValueError):
                reason = "weights_incomplete"
            else:
                self._worker = worker or ProcessRapidOCRWorker(
                    self._resolved_profiles, device=runtime.device
                )
        self._device = runtime.device if runtime is not None else "cpu"
        self.descriptor = ProviderDescriptor(
            id="rapidocr-ppocrv6-local",
            kind="ocr",
            version="3.9.1",
            execution_boundary="local",
            criticality="optional",
            available=reason is None,
            unavailable_reason_code=reason,
            model_name="PP-OCRv6-small+script-profiles",
        )

    async def extract(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[OCRResult]:
        if not self.descriptor.available or self._worker is None:
            return ProviderOutcome.skipped(
                self.descriptor.unavailable_reason_code or "unavailable"
            )
        if context.cancellation.is_set():
            return ProviderOutcome.skipped("unavailable")
        started = time.monotonic()
        try:
            image_bytes, width, height = await asyncio.to_thread(
                self._read_bounded_image, handle.path
            )
        except (OSError, UnidentifiedImageError, ValueError):
            return ProviderOutcome.failed(
                "unsupported_input",
                retryable=False,
                attempts=1,
                duration_ms=int((time.monotonic() - started) * 1000),
                subreason_code="ocr_image_bounds_or_decode_invalid",
            )
        remaining = (context.deadline - datetime.now(UTC)).total_seconds()
        timeout = min(self._timeout, max(0.05, remaining))
        try:
            batch = await self._worker.infer(
                image_bytes,
                timeout_seconds=timeout,
                cancellation=context.cancellation,
            )
            del image_bytes
        except RapidOCRWorkerFailure as exc:
            if exc.code == "cancelled":
                return ProviderOutcome.skipped("unavailable")
            failure_code = (
                "inference_timeout" if exc.code == "timeout" else "internal_provider_error"
            )
            return ProviderOutcome.failed(
                failure_code,
                retryable=False,
                attempts=1,
                duration_ms=int((time.monotonic() - started) * 1000),
                subreason_code=self._safe_subreason(exc.code),
            )
        blocks, observations = self._normalize_lines(batch.lines, width=width, height=height)
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
        snippets = list(dict.fromkeys(block.redacted_text for block in blocks))[:5]
        return ProviderOutcome.succeeded(
            OCRResult(
                redacted_snippets=snippets,
                blocks=blocks[:32],
                place_matches=place_matches[:12],
            )
        )

    async def close(self) -> None:
        if self._worker is not None:
            await self._worker.close()

    def safe_status(self) -> dict[str, object]:
        return {
            "provider_id": self.descriptor.id,
            "available": self.descriptor.available,
            "reason_code": self.descriptor.unavailable_reason_code,
            "device": self._device,
            "offline": True,
            "profiles": len(self._resolved_profiles),
        }

    def _read_bounded_image(self, path: Path) -> tuple[bytes, int, int]:
        size = path.stat().st_size
        if not 0 < size <= self._max_input_bytes:
            raise ValueError("image byte bound exceeded")
        with Image.open(path) as image:
            width, height = image.size
            if (
                width <= 0
                or height <= 0
                or max(width, height) > self._max_side
                or width * height > self._max_decoded_pixels
            ):
                raise ValueError("image dimension bound exceeded")
        return path.read_bytes(), width, height

    def _normalize_lines(
        self, lines: tuple[RawOCRLine, ...], *, width: int, height: int
    ) -> tuple[list[OCRBlock], list[OCRTextObservation]]:
        selected: dict[tuple[object, ...], tuple[OCRBlock, OCRTextObservation]] = {}
        for line in lines[:128]:
            raw = sanitize_unicode(line.text)
            if not raw or not math.isfinite(line.confidence) or not 0 <= line.confidence <= 1:
                continue
            points: list[OCRPolygonPoint] = []
            valid = True
            for x, y in line.polygon:
                if (
                    not math.isfinite(x)
                    or not math.isfinite(y)
                    or not 0 <= x <= width
                    or not 0 <= y <= height
                ):
                    valid = False
                    break
                points.append(OCRPolygonPoint(x=x / width, y=y / height))
            if not valid or len(points) != 4:
                continue
            redacted = redact_sensitive_text(raw)
            if not redacted.text:
                continue
            script = detect_script(raw)
            hints = language_hints(script, line.profile_id)
            normalized_safe = normalize_search_text(redacted.text) or redacted.text
            block = OCRBlock(
                redacted_text=redacted.text,
                normalized_text=normalized_safe[:240],
                script=script,
                language_hints=hints,
                confidence=line.confidence,
                bounding_polygon=(points[0], points[1], points[2], points[3]),
                provider=self.descriptor.id,
                profile=line.profile_id,
                sensitive_content=redacted.sensitive,
            )
            observation = OCRTextObservation(
                text=raw,
                confidence=line.confidence,
                script=script,
                language_hints=tuple(hints),
            )
            key = (
                block.normalized_text,
                *(round(point.x, 3) for point in points),
                *(round(point.y, 3) for point in points),
            )
            previous = selected.get(key)
            if previous is None or block.confidence > previous[0].confidence:
                selected[key] = (block, observation)
            if len(selected) >= 32:
                break
        ranked = sorted(
            selected.values(),
            key=lambda pair: (
                min(point.y for point in pair[0].bounding_polygon),
                min(point.x for point in pair[0].bounding_polygon),
                -pair[0].confidence,
                pair[0].normalized_text,
            ),
        )
        return [pair[0] for pair in ranked], [pair[1] for pair in ranked]

    @staticmethod
    def _safe_subreason(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
        return (normalized or "rapidocr_worker_failed")[:120]
