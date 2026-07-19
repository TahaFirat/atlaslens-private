from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

type DeviceName = Literal["cuda", "cpu"]
type MaybeAwaitable[T] = T | Awaitable[T]


@dataclass(frozen=True, slots=True)
class HeavyModelDiagnostic:
    model_name: str
    requested_device: DeviceName
    actual_device: DeviceName
    estimated_vram_mb: int
    load_duration_ms: int
    inference_duration_ms: int
    unload_duration_ms: int
    status: Literal["succeeded", "failed", "oom_cpu_fallback"]
    cpu_fallback_used: bool


@dataclass(frozen=True, slots=True)
class ScheduledExecution[T]:
    value: T
    diagnostic: HeavyModelDiagnostic


@dataclass(slots=True)
class _ResidentModel:
    unload: Callable[[DeviceName], MaybeAwaitable[None]]
    estimated_vram_mb: int


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


async def _resolve[T](value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    return isinstance(exc, RuntimeError) and "cuda out of memory" in str(exc).casefold()


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


class HeavyModelScheduler:
    """Serialize heavy CUDA stages and bound resident models.

    The scheduler owns only execution/resource policy. Provider workers retain
    model-specific loading logic. CUDA cache clearing occurs solely after an
    actually resident model has been released.
    """

    def __init__(
        self,
        *,
        max_heavy_concurrency: int = 1,
        max_resident_models: int = 1,
        cuda_oom_probe: Callable[[BaseException], bool] = _is_cuda_oom,
        empty_cuda_cache: Callable[[], None] = _empty_cuda_cache,
        history_size: int = 64,
    ) -> None:
        if not 1 <= max_heavy_concurrency <= 8:
            raise ValueError("heavy-model concurrency must be between 1 and 8")
        if not 1 <= max_resident_models <= 8:
            raise ValueError("resident-model limit must be between 1 and 8")
        if not 1 <= history_size <= 512:
            raise ValueError("scheduler history size must be between 1 and 512")
        self._semaphore = asyncio.Semaphore(max_heavy_concurrency)
        self._max_resident = max_resident_models
        self._resident: OrderedDict[str, _ResidentModel] = OrderedDict()
        self._state_lock = asyncio.Lock()
        self._cuda_oom_probe = cuda_oom_probe
        self._empty_cuda_cache = empty_cuda_cache
        self._history: deque[HeavyModelDiagnostic] = deque(maxlen=history_size)
        self._active_cuda = 0
        self._max_observed_cuda = 0

    async def execute[T](
        self,
        *,
        model_name: str,
        requested_device: DeviceName,
        estimated_vram_mb: int,
        load: Callable[[DeviceName], MaybeAwaitable[None]],
        infer: Callable[[DeviceName], MaybeAwaitable[T]],
        unload: Callable[[DeviceName], MaybeAwaitable[None]],
        allow_cpu_fallback: bool = True,
    ) -> ScheduledExecution[T]:
        if not model_name or len(model_name) > 160:
            raise ValueError("model name must be bounded")
        if estimated_vram_mb < 0 or not math.isfinite(float(estimated_vram_mb)):
            raise ValueError("estimated VRAM must be non-negative")
        if requested_device == "cpu":
            return await self._execute_cpu(
                model_name=model_name,
                estimated_vram_mb=estimated_vram_mb,
                load=load,
                infer=infer,
                unload=unload,
                fallback=False,
            )

        async with self._semaphore:
            self._active_cuda += 1
            self._max_observed_cuda = max(self._max_observed_cuda, self._active_cuda)
            load_ms = 0
            unload_ms = 0
            inference_started = 0.0
            try:
                load_ms, eviction_unload_ms = await self._ensure_resident(
                    model_name=model_name,
                    estimated_vram_mb=estimated_vram_mb,
                    load=load,
                    unload=unload,
                )
                unload_ms += eviction_unload_ms
                inference_started = time.monotonic()
                value = await _resolve(infer("cuda"))
                inference_ms = _elapsed_ms(inference_started)
                await self._touch(model_name)
                diagnostic = HeavyModelDiagnostic(
                    model_name=model_name,
                    requested_device="cuda",
                    actual_device="cuda",
                    estimated_vram_mb=estimated_vram_mb,
                    load_duration_ms=load_ms,
                    inference_duration_ms=inference_ms,
                    unload_duration_ms=unload_ms,
                    status="succeeded",
                    cpu_fallback_used=False,
                )
                self._history.append(diagnostic)
                return ScheduledExecution(value=value, diagnostic=diagnostic)
            except asyncio.CancelledError:
                await self._release(model_name)
                raise
            except Exception as exc:
                inference_ms = _elapsed_ms(inference_started) if inference_started else 0
                released_ms = await self._release(model_name)
                unload_ms += released_ms
                if not self._cuda_oom_probe(exc) or not allow_cpu_fallback:
                    self._history.append(
                        HeavyModelDiagnostic(
                            model_name=model_name,
                            requested_device="cuda",
                            actual_device="cuda",
                            estimated_vram_mb=estimated_vram_mb,
                            load_duration_ms=load_ms,
                            inference_duration_ms=inference_ms,
                            unload_duration_ms=unload_ms,
                            status="failed",
                            cpu_fallback_used=False,
                        )
                    )
                    raise
                cpu = await self._execute_cpu(
                    model_name=model_name,
                    estimated_vram_mb=estimated_vram_mb,
                    load=load,
                    infer=infer,
                    unload=unload,
                    fallback=True,
                )
                combined = HeavyModelDiagnostic(
                    model_name=model_name,
                    requested_device="cuda",
                    actual_device="cpu",
                    estimated_vram_mb=estimated_vram_mb,
                    load_duration_ms=load_ms + cpu.diagnostic.load_duration_ms,
                    inference_duration_ms=inference_ms + cpu.diagnostic.inference_duration_ms,
                    unload_duration_ms=unload_ms + cpu.diagnostic.unload_duration_ms,
                    status="oom_cpu_fallback",
                    cpu_fallback_used=True,
                )
                self._history.append(combined)
                return ScheduledExecution(value=cpu.value, diagnostic=combined)
            finally:
                self._active_cuda -= 1

    async def close(self) -> None:
        async with self._semaphore:
            async with self._state_lock:
                residents = list(reversed(self._resident.items()))
                self._resident.clear()
            released = False
            for _, resident in residents:
                await _resolve(resident.unload("cuda"))
                released = True
            if released:
                self._empty_cuda_cache()

    def diagnostics(self) -> tuple[HeavyModelDiagnostic, ...]:
        return tuple(self._history)

    def safe_status(self) -> dict[str, int | tuple[str, ...]]:
        return {
            "active_cuda": self._active_cuda,
            "max_observed_cuda": self._max_observed_cuda,
            "resident_count": len(self._resident),
            "resident_models": tuple(self._resident),
        }

    async def _execute_cpu[T](
        self,
        *,
        model_name: str,
        estimated_vram_mb: int,
        load: Callable[[DeviceName], MaybeAwaitable[None]],
        infer: Callable[[DeviceName], MaybeAwaitable[T]],
        unload: Callable[[DeviceName], MaybeAwaitable[None]],
        fallback: bool,
    ) -> ScheduledExecution[T]:
        load_started = time.monotonic()
        await _resolve(load("cpu"))
        load_ms = _elapsed_ms(load_started)
        inference_started = time.monotonic()
        try:
            value = await _resolve(infer("cpu"))
            inference_ms = _elapsed_ms(inference_started)
        except BaseException:
            await _resolve(unload("cpu"))
            raise
        unload_started = time.monotonic()
        await _resolve(unload("cpu"))
        unload_ms = _elapsed_ms(unload_started)
        diagnostic = HeavyModelDiagnostic(
            model_name=model_name,
            requested_device="cuda" if fallback else "cpu",
            actual_device="cpu",
            estimated_vram_mb=estimated_vram_mb,
            load_duration_ms=load_ms,
            inference_duration_ms=inference_ms,
            unload_duration_ms=unload_ms,
            status="oom_cpu_fallback" if fallback else "succeeded",
            cpu_fallback_used=fallback,
        )
        if not fallback:
            self._history.append(diagnostic)
        return ScheduledExecution(value=value, diagnostic=diagnostic)

    async def _ensure_resident(
        self,
        *,
        model_name: str,
        estimated_vram_mb: int,
        load: Callable[[DeviceName], MaybeAwaitable[None]],
        unload: Callable[[DeviceName], MaybeAwaitable[None]],
    ) -> tuple[int, int]:
        async with self._state_lock:
            if model_name in self._resident:
                self._resident.move_to_end(model_name)
                return 0, 0
            evicted: list[_ResidentModel] = []
            while len(self._resident) >= self._max_resident:
                _, resident = self._resident.popitem(last=False)
                evicted.append(resident)
            unload_ms = 0
            for resident in evicted:
                started = time.monotonic()
                await _resolve(resident.unload("cuda"))
                unload_ms += _elapsed_ms(started)
            if evicted:
                self._empty_cuda_cache()
            started = time.monotonic()
            await _resolve(load("cuda"))
            load_ms = _elapsed_ms(started)
            self._resident[model_name] = _ResidentModel(
                unload=unload,
                estimated_vram_mb=estimated_vram_mb,
            )
            return load_ms, unload_ms

    async def _touch(self, model_name: str) -> None:
        async with self._state_lock:
            if model_name in self._resident:
                self._resident.move_to_end(model_name)

    async def _release(self, model_name: str) -> int:
        async with self._state_lock:
            resident = self._resident.pop(model_name, None)
        if resident is None:
            return 0
        started = time.monotonic()
        await _resolve(resident.unload("cuda"))
        duration = _elapsed_ms(started)
        self._empty_cuda_cache()
        return duration
