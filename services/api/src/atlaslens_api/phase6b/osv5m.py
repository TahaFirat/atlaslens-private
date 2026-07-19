from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Callable
from typing import Literal, Protocol

from atlaslens_api.phase6b.models import (
    GeographicCandidate,
    GeographicProviderCapability,
    GeographicProviderResult,
    normalize_longitude,
)
from atlaslens_api.phase6b.scheduler import (
    DeviceName,
    HeavyModelScheduler,
    ScheduledExecution,
)


class OSV5MWorker(Protocol):
    """Local worker seam around the official ``models.huggingface.Geolocalizer``.

    Implementations live in the pinned isolated OSV-5M environment. ``load``
    must use a prepared local ``osv5m/baseline`` snapshot and the official
    model-provided ``transform``; it must never call the Hub at request time.
    ``predict_radians`` returns one ``(latitude, longitude)`` pair in radians,
    matching the official repository contract.
    """

    async def load(self, device: DeviceName) -> None: ...

    async def predict_radians(self, image_bytes: bytes, *, device: DeviceName) -> object: ...

    async def unload(self, device: DeviceName) -> None: ...


class InvalidOSV5MOutput(ValueError):
    pass


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def _as_python(value: object) -> object:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def convert_osv5m_radians(value: object) -> tuple[float, float]:
    """Convert the official Bx2 lat/lon-radian output to validated degrees."""

    raw = _as_python(value)
    if isinstance(raw, tuple | list) and len(raw) == 1:
        raw = raw[0]
    if not isinstance(raw, tuple | list) or len(raw) != 2:
        raise InvalidOSV5MOutput("OSV-5M output must contain one latitude/longitude pair")
    latitude_raw, longitude_raw = raw
    if (
        isinstance(latitude_raw, bool)
        or isinstance(longitude_raw, bool)
        or not isinstance(latitude_raw, int | float)
        or not isinstance(longitude_raw, int | float)
    ):
        raise InvalidOSV5MOutput("OSV-5M coordinates must be numeric")
    latitude_radians = float(latitude_raw)
    longitude_radians = float(longitude_raw)
    if not math.isfinite(latitude_radians) or not math.isfinite(longitude_radians):
        raise InvalidOSV5MOutput("OSV-5M coordinates must be finite")
    latitude = math.degrees(latitude_radians)
    longitude = normalize_longitude(math.degrees(longitude_radians))
    if not -90 <= latitude <= 90:
        raise InvalidOSV5MOutput("OSV-5M latitude is outside WGS84")
    return latitude, longitude


class OSV5MProvider:
    provider_id = "osv5m-baseline"
    source_family: Literal["osv5m_family"] = "osv5m_family"

    def __init__(
        self,
        *,
        enabled: bool,
        worker: OSV5MWorker | None,
        scheduler: HeavyModelScheduler,
        model_id: str = "osv5m/baseline",
        model_revision: str = "unprepared",
        source_revision: str = "unprepared",
        device: Literal["auto", "cuda", "cpu"] = "auto",
        cuda_available: Callable[[], bool] = _cuda_available,
        timeout_seconds: float = 45.0,
        max_input_bytes: int = 20 * 1024 * 1024,
        estimated_vram_mb: int = 3_500,
        allow_cpu_fallback: bool = True,
    ) -> None:
        if not model_id or len(model_id) > 160:
            raise ValueError("OSV-5M model id must be bounded")
        if not model_revision or not source_revision:
            raise ValueError("OSV-5M revisions must be explicit")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("OSV-5M timeout must be positive")
        if max_input_bytes <= 0:
            raise ValueError("OSV-5M input bound must be positive")
        if device == "cuda" and not cuda_available():
            selected_device: DeviceName = "cpu"
            device_warning = "warning.osv5m.cuda_unavailable_cpu_selected"
        elif device == "auto":
            selected_device = "cuda" if cuda_available() else "cpu"
            device_warning = None
        else:
            selected_device = device
            device_warning = None
        self._enabled = enabled
        self._worker = worker
        self._scheduler = scheduler
        self._model_id = model_id
        self._model_revision = model_revision
        self._source_revision = source_revision
        self._device = selected_device
        self._device_warning = device_warning
        self._timeout = timeout_seconds
        self._max_input_bytes = max_input_bytes
        self._estimated_vram_mb = estimated_vram_mb
        self._allow_cpu_fallback = allow_cpu_fallback

    def status(self) -> GeographicProviderCapability:
        if not self._enabled:
            state: Literal["ready", "disabled", "not_installed"] = "disabled"
            reason = "disabled"
        elif self._worker is None:
            state = "not_installed"
            reason = "isolated_worker_not_installed"
        elif self._model_revision == "unprepared" or self._source_revision == "unprepared":
            state = "not_installed"
            reason = "model_artifact_not_verified"
        else:
            state = "ready"
            reason = None
        return GeographicProviderCapability(
            provider=self.provider_id,
            available=state == "ready",
            state=state,
            reason_code=reason,
            model_id=self._model_id,
            model_revision=self._model_revision,
            source_revision=self._source_revision,
            device=self._device,
        )

    async def predict(
        self,
        image_bytes: bytes,
        *,
        cancellation: asyncio.Event | None = None,
    ) -> GeographicProviderResult:
        capability = self.status()
        if not capability.available or self._worker is None:
            return self._unavailable(capability)
        worker = self._worker
        if type(image_bytes) is not bytes or not 0 < len(image_bytes) <= self._max_input_bytes:
            return self._failed("unsupported_input", duration_ms=0)
        if cancellation is not None and cancellation.is_set():
            return self._failed("cancelled", duration_ms=0, status="skipped")
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._timeout):
                execution: ScheduledExecution[object] = await self._scheduler.execute(
                    model_name=self._model_id,
                    requested_device=self._device,
                    estimated_vram_mb=self._estimated_vram_mb,
                    load=worker.load,
                    infer=lambda device: worker.predict_radians(image_bytes, device=device),
                    unload=worker.unload,
                    allow_cpu_fallback=self._allow_cpu_fallback,
                )
                latitude, longitude = convert_osv5m_radians(execution.value)
        except TimeoutError:
            return self._failed(
                "inference_timeout",
                duration_ms=_elapsed_ms(started),
                status="timeout",
            )
        except asyncio.CancelledError:
            raise
        except InvalidOSV5MOutput:
            return self._failed("invalid_model_output", duration_ms=_elapsed_ms(started))
        except Exception as exc:
            reason = (
                "cuda_out_of_memory"
                if "out of memory" in str(exc).casefold()
                else "worker_unavailable"
            )
            return self._failed(reason, duration_ms=_elapsed_ms(started))
        stable = hashlib.sha256(f"{latitude:.7f}|{longitude:.7f}".encode()).hexdigest()[:24]
        warnings = ["warning.osv5m.direct_regression_not_confidence"]
        if self._device_warning is not None:
            warnings.append(self._device_warning)
        if execution.diagnostic.cpu_fallback_used:
            warnings.append("warning.osv5m.cuda_oom_cpu_fallback")
        return GeographicProviderResult(
            provider=self.provider_id,
            model_id=self._model_id,
            model_revision=self._model_revision,
            source_family=self.source_family,
            status="completed",
            device=execution.diagnostic.actual_device,
            duration_ms=_elapsed_ms(started),
            score_semantics="direct_regression",
            candidates=(
                GeographicCandidate(
                    candidate_id=f"osv5m-{stable}",
                    latitude=latitude,
                    longitude=longitude,
                    raw_score=None,
                    provider_rank=1,
                    sample_support=1,
                    metadata={
                        "coordinate_order": "latitude_longitude",
                        "provider_output_units": "radians",
                    },
                ),
            ),
            warnings=tuple(warnings),
            diagnostics={
                "source_revision": self._source_revision,
                "cpu_fallback_used": execution.diagnostic.cpu_fallback_used,
                "dataset_required": False,
            },
        )

    def _unavailable(self, capability: GeographicProviderCapability) -> GeographicProviderResult:
        return GeographicProviderResult(
            provider=self.provider_id,
            model_id=self._model_id,
            model_revision=self._model_revision,
            source_family=self.source_family,
            status="disabled" if capability.state == "disabled" else "skipped",
            device=self._device,
            duration_ms=0,
            score_semantics="direct_regression",
            reason_code=capability.reason_code or "unavailable",
            diagnostics={"source_revision": self._source_revision},
        )

    def _failed(
        self,
        reason: str,
        *,
        duration_ms: int,
        status: Literal["failed", "timeout", "skipped"] = "failed",
    ) -> GeographicProviderResult:
        return GeographicProviderResult(
            provider=self.provider_id,
            model_id=self._model_id,
            model_revision=self._model_revision,
            source_family=self.source_family,
            status=status,
            device=self._device,
            duration_ms=duration_ms,
            score_semantics="direct_regression",
            reason_code=reason,
            diagnostics={"source_revision": self._source_revision},
        )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
