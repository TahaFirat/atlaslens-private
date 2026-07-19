from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from atlaslens_api.inference.models import InferenceResult
from atlaslens_api.phase6b.fusion import (
    Phase6BFusionResult,
    Phase6BGeographicFusionEngine,
)
from atlaslens_api.phase6b.geoclip import normalize_geoclip_result
from atlaslens_api.phase6b.models import GeographicProviderResult
from atlaslens_api.phase6b.plonk import PlonkModelKind
from atlaslens_api.providers.base import OCRResult
from atlaslens_api.schemas import SceneSegmentationSummary


class OSV5MProviderLike(Protocol):
    async def predict(
        self,
        image_bytes: bytes,
        *,
        cancellation: asyncio.Event | None = None,
    ) -> GeographicProviderResult: ...


class PlonkProviderLike(Protocol):
    async def predict(
        self,
        image_bytes: bytes,
        *,
        scene: SceneSegmentationSummary | None,
        override: PlonkModelKind | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> GeographicProviderResult: ...


@dataclass(frozen=True, slots=True)
class Phase6BLocalEnsembleResult:
    model_predictions: tuple[GeographicProviderResult, ...]
    fusion: Phase6BFusionResult


class Phase6BLocalModelEnsemble:
    """Small integration seam; heavy providers execute sequentially by design."""

    def __init__(
        self,
        *,
        osv5m: OSV5MProviderLike,
        plonk: PlonkProviderLike,
        fusion: Phase6BGeographicFusionEngine,
    ) -> None:
        self._osv5m = osv5m
        self._plonk = plonk
        self._fusion = fusion

    async def run(
        self,
        image_bytes: bytes,
        *,
        geoclip: InferenceResult,
        scene: SceneSegmentationSummary | None,
        ocr: OCRResult | None,
        cancellation: asyncio.Event,
        plonk_override: PlonkModelKind | None = None,
    ) -> Phase6BLocalEnsembleResult:
        if cancellation.is_set():
            raise asyncio.CancelledError
        geoclip_result = normalize_geoclip_result(geoclip)
        osv5m_result = await self._osv5m.predict(image_bytes, cancellation=cancellation)
        if cancellation.is_set():
            raise asyncio.CancelledError
        plonk_result = await self._plonk.predict(
            image_bytes,
            scene=scene,
            override=plonk_override,
            cancellation=cancellation,
        )
        predictions = (geoclip_result, osv5m_result, plonk_result)
        fused = self._fusion.fuse(
            predictions,
            ocr_places=ocr.place_matches if ocr is not None else (),
        )
        return Phase6BLocalEnsembleResult(
            model_predictions=predictions,
            fusion=fused,
        )
