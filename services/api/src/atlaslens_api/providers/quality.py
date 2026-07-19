from __future__ import annotations

import asyncio
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from atlaslens_api.providers.base import InvocationContext, ProviderDescriptor, ProviderOutcome
from atlaslens_api.schemas import QualitySummary
from atlaslens_api.storage import LocalImageHandle


def analyze_quality(path: Path) -> QualitySummary:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = laplacian_variance / (laplacian_variance + 100.0)
    brightness_score = float(gray.mean()) / 255.0
    contrast_score = min(float(gray.std()) / 64.0, 1.0)
    megapixels = (gray.shape[0] * gray.shape[1]) / 1_000_000
    resolution_score = min(megapixels / 2.0, 1.0)

    warnings: list[str] = []
    if blur_score < 0.12:
        warnings.append("quality.severe_blur")
    if brightness_score < 0.10:
        warnings.append("quality.severe_underexposure")
    elif brightness_score > 0.90:
        warnings.append("quality.severe_overexposure")
    if contrast_score < 0.10:
        warnings.append("quality.low_contrast")
    if resolution_score < 0.15:
        warnings.append("quality.low_resolution")

    return QualitySummary(
        blur_score=round(max(0.0, min(blur_score, 1.0)), 4),
        brightness_score=round(max(0.0, min(brightness_score, 1.0)), 4),
        contrast_score=round(max(0.0, min(contrast_score, 1.0)), 4),
        resolution_score=round(max(0.0, min(resolution_score, 1.0)), 4),
        warnings=warnings,
    )


class OpenCVImageQualityProvider:
    descriptor = ProviderDescriptor(
        id="opencv-quality",
        kind="image_quality",
        version="1.0.0",
        execution_boundary="local",
        criticality="required",
        available=True,
    )

    async def analyze(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[QualitySummary]:
        started = time.monotonic()
        if context.cancellation.is_set():
            return ProviderOutcome.skipped("unavailable")
        try:
            result = await asyncio.to_thread(analyze_quality, handle.path)
        except (OSError, ValueError, cv2.error):
            return ProviderOutcome.failed(
                "internal_provider_error",
                retryable=False,
                attempts=1,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        return ProviderOutcome.succeeded(result)
