from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from uuid import uuid4

from PIL import Image

from atlaslens_api.evaluation.models import CandidatePrediction, ProviderPrediction
from atlaslens_api.fusion import haversine_km
from atlaslens_api.gazetteer import GazetteerResolver
from atlaslens_api.inference.models import (
    GeolocationInferenceProvider,
    InferenceRequest,
)
from atlaslens_api.providers.base import (
    GlobalGeolocationProvider,
    InvocationContext,
    OutcomeStatus,
)
from atlaslens_api.schemas import AnalysisMode, GeoPoint
from atlaslens_api.storage import LocalImageHandle


class GlobalProviderEvaluationAdapter:
    """Synchronous benchmark adapter for the real typed global-provider contract."""

    def __init__(
        self,
        provider: GlobalGeolocationProvider,
        *,
        gazetteer: GazetteerResolver | None = None,
        timeout_seconds: float = 120,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("evaluation provider timeout must be positive")
        self._provider = provider
        self._gazetteer = gazetteer
        self._timeout = timeout_seconds
        self.provider_id = provider.descriptor.id
        self.model_revision = provider.status().model_revision
        self._runner = asyncio.Runner(loop_factory=asyncio.new_event_loop)
        self._closed = False

    def predict(self, image_path: Path) -> ProviderPrediction:
        if self._closed:
            raise RuntimeError("evaluation adapter is closed")
        return self._runner.run(self._predict(image_path))

    def close(self) -> None:
        if not self._closed:
            self._runner.close()
            self._closed = True

    async def _predict(self, image_path: Path) -> ProviderPrediction:
        context = InvocationContext(
            analysis_id=uuid4(),
            request_id="benchmark",
            mode=AnalysisMode.LOCAL_ONLY,
            cloud_consent=False,
            deadline=datetime.now(UTC) + timedelta(seconds=self._timeout),
            cancellation=asyncio.Event(),
        )
        outcome = await self._provider.predict(
            LocalImageHandle(key="evaluation.asset", path=image_path), context
        )
        if outcome.status == OutcomeStatus.SUCCEEDED and outcome.value is not None:
            result = outcome.value
            candidates: list[CandidatePrediction] = []
            for hypothesis in result.hypotheses:
                point = GeoPoint(
                    latitude=hypothesis.latitude, longitude=hypothesis.longitude
                )
                dispersion = median(
                    [
                        haversine_km(
                            point,
                            GeoPoint(
                                latitude=other.latitude,
                                longitude=other.longitude,
                            ),
                        )
                        for other in result.hypotheses
                        if other is not hypothesis
                    ]
                    or [0.0]
                )
                place = (
                    self._gazetteer.resolve(hypothesis.latitude, hypothesis.longitude)
                    if self._gazetteer is not None
                    else None
                )
                candidates.append(
                    CandidatePrediction(
                        rank=hypothesis.rank,
                        latitude=hypothesis.latitude,
                        longitude=hypothesis.longitude,
                        raw_score=hypothesis.raw_score,
                        score_type=hypothesis.score_type,
                        country_code=place.country_code if place else None,
                        region=place.region if place else None,
                        city_or_area=place.city if place else None,
                        uncertainty_radius_km=max(750.0, dispersion),
                    )
                )
            return ProviderPrediction(
                candidates=tuple(candidates),
                latency_ms=result.inference_ms,
                device=_device(result.device),
            )
        if outcome.status == OutcomeStatus.ABSTAINED:
            return ProviderPrediction(abstained=True, latency_ms=0, device="other")
        failure = outcome.failure
        return ProviderPrediction(
            failure_code=failure.code if failure is not None else "provider_unavailable",
            latency_ms=failure.duration_ms if failure is not None else 0,
            device="other",
        )


class InferenceProviderEvaluationAdapter:
    """Synchronous benchmark boundary for a verified real inference provider."""

    def __init__(
        self,
        provider: GeolocationInferenceProvider,
        *,
        gazetteer: GazetteerResolver | None = None,
        timeout_seconds: float = 120,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("evaluation provider timeout must be positive")
        status = provider.status()
        if status.classification != "real":
            raise ValueError("simulated providers cannot enter evaluation")
        if not status.available or status.model_revision is None:
            raise ValueError("evaluation provider is unavailable")
        self._provider = provider
        self._gazetteer = gazetteer
        self._timeout = timeout_seconds
        self.provider_id = status.provider_id
        self.model_revision = status.model_revision
        self._runner = asyncio.Runner(loop_factory=asyncio.new_event_loop)
        self._closed = False

    def predict(self, image_path: Path) -> ProviderPrediction:
        if self._closed:
            raise RuntimeError("evaluation adapter is closed")
        return self._runner.run(self._predict(image_path))

    def close(self) -> None:
        if not self._closed:
            self._runner.close()
            self._closed = True

    async def _predict(self, image_path: Path) -> ProviderPrediction:
        def dimensions() -> tuple[int, int]:
            with Image.open(image_path) as image:
                return int(image.width), int(image.height)

        width, height = await asyncio.to_thread(dimensions)
        declared_size = image_path.stat().st_size
        cancellation = asyncio.Event()
        result = await self._provider.infer(
            InferenceRequest(
                image_handle=LocalImageHandle(key="evaluation.asset", path=image_path),
                image_bytes=None,
                width=width,
                height=height,
                exif=None,
                analysis_mode=AnalysisMode.LOCAL_ONLY,
                top_k=5,
                cancellation=cancellation,
                deadline=datetime.now(UTC) + timedelta(seconds=self._timeout),
                trace_id=f"benchmark-{uuid4().hex}",
                max_input_bytes=max(1, declared_size),
            )
        )
        if result.classification != "real":
            return ProviderPrediction(
                failure_code="simulated_provider_rejected",
                latency_ms=result.runtime_ms,
                device="other",
            )
        if result.status == "succeeded" and result.score_semantics is not None:
            candidates: list[CandidatePrediction] = []
            for hypothesis in result.candidates:
                point = GeoPoint(
                    latitude=hypothesis.latitude, longitude=hypothesis.longitude
                )
                dispersion = median(
                    [
                        haversine_km(
                            point,
                            GeoPoint(
                                latitude=other.latitude,
                                longitude=other.longitude,
                            ),
                        )
                        for other in result.candidates
                        if other is not hypothesis
                    ]
                    or [0.0]
                )
                place = (
                    self._gazetteer.resolve(hypothesis.latitude, hypothesis.longitude)
                    if self._gazetteer is not None
                    else None
                )
                candidates.append(
                    CandidatePrediction(
                        rank=hypothesis.rank,
                        latitude=hypothesis.latitude,
                        longitude=hypothesis.longitude,
                        raw_score=hypothesis.raw_score,
                        score_type=result.score_semantics,
                        country_code=place.country_code if place else None,
                        region=place.region if place else None,
                        city_or_area=place.city if place else None,
                        uncertainty_radius_km=max(750.0, dispersion),
                    )
                )
            return ProviderPrediction(
                candidates=tuple(candidates),
                latency_ms=result.runtime_ms,
                device=_device(result.device or "other"),
            )
        if result.status == "abstained":
            return ProviderPrediction(
                abstained=True,
                latency_ms=result.runtime_ms,
                device=_device(result.device or "other"),
            )
        return ProviderPrediction(
            failure_code=(
                result.failure.code if result.failure is not None else "provider_unavailable"
            ),
            latency_ms=result.runtime_ms,
            device=_device(result.device or "other"),
        )


def _device(value: str) -> str:
    normalized = value.lower()
    if normalized.startswith("cuda"):
        return "cuda"
    if normalized.startswith("cpu"):
        return "cpu"
    return "other"
