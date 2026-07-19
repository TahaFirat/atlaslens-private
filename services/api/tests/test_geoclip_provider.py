from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest

from atlaslens_api.model_management.errors import ModelManagementError
from atlaslens_api.providers.base import InvocationContext, OutcomeStatus
from atlaslens_api.providers.geoclip import GeoCLIPGlobalGeolocationProvider, select_device
from atlaslens_api.schemas import AnalysisMode
from atlaslens_api.storage import LocalImageHandle
from test_model_management import build_installed_management


class FakeRuntime:
    device = "cpu"
    dtype = "float32"

    def __init__(self, coordinates: list[list[float]] | None = None, *, delay: float = 0) -> None:
        self.coordinates = coordinates or [[index * 2.0, index * 3.0] for index in range(20)]
        self.delay = delay

    def predict(self, _image_payload: bytes, top_k: int) -> tuple[list[list[float]], list[float]]:
        time.sleep(self.delay)
        coordinates = self.coordinates[:top_k]
        return coordinates, [1 / (index + 2) for index in range(len(coordinates))]


class OutOfMemoryRuntime(FakeRuntime):
    device = "cuda"

    def predict(self, _image_payload: bytes, top_k: int) -> tuple[list[list[float]], list[float]]:
        raise RuntimeError("CUDA out of memory")


class PayloadLifetimeRuntime(FakeRuntime):
    def __init__(self, *, delay: float = 0) -> None:
        super().__init__(delay=delay)
        self.work_completed = Event()

    def predict(self, image_payload: bytes, top_k: int) -> tuple[list[list[float]], list[float]]:
        assert image_payload == b"private"
        time.sleep(self.delay)
        self.work_completed.set()
        coordinates = self.coordinates[:top_k]
        return coordinates, [1 / (index + 2) for index in range(len(coordinates))]


class TopKRecordingRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__([[-70.0 + index * 2.5, -175.0 + index * 7.0] for index in range(50)])
        self.requested_top_k: int | None = None

    def predict(self, image_payload: bytes, top_k: int) -> tuple[list[list[float]], list[float]]:
        self.requested_top_k = top_k
        return super().predict(image_payload, top_k)


def _context(*, cancelled: bool = False, deadline_seconds: float = 5) -> InvocationContext:
    cancellation = asyncio.Event()
    if cancelled:
        cancellation.set()
    return InvocationContext(
        analysis_id=uuid4(),
        request_id="safe-request",
        mode=AnalysisMode.LOCAL_ONLY,
        cloud_consent=False,
        deadline=datetime.now(UTC) + timedelta(seconds=deadline_seconds),
        cancellation=cancellation,
    )


def _handle(tmp_path: Path) -> LocalImageHandle:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"private")
    return LocalImageHandle(key="fixture.upload", path=image)


@pytest.mark.asyncio
async def test_real_provider_boundary_returns_stable_uncalibrated_top_five(tmp_path: Path) -> None:
    management = build_installed_management(tmp_path)
    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: FakeRuntime(),
        device_selector=lambda: "cpu",
    )
    outcome = await provider.predict(_handle(tmp_path), _context())
    assert outcome.status == OutcomeStatus.SUCCEEDED
    assert outcome.value is not None
    assert len(outcome.value.hypotheses) == 5
    assert [item.rank for item in outcome.value.hypotheses] == [1, 2, 3, 4, 5]
    assert all(item.calibration_state == "uncalibrated" for item in outcome.value.hypotheses)
    assert all(
        item.score_type == "uncalibrated_gallery_softmax" for item in outcome.value.hypotheses
    )
    assert outcome.value.external_transfer is False


@pytest.mark.asyncio
async def test_phase6a_internal_top_fifty_preserves_raw_ranks(tmp_path: Path) -> None:
    management = build_installed_management(tmp_path)
    runtime = TopKRecordingRuntime()
    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: runtime,
        device_selector=lambda: "cpu",
        internal_top_k=50,
    )
    outcome = await provider.predict(_handle(tmp_path), _context())
    assert outcome.status == OutcomeStatus.SUCCEEDED
    assert outcome.value is not None
    assert runtime.requested_top_k == 50
    assert len(outcome.value.hypotheses) == 50
    assert [item.original_rank for item in outcome.value.hypotheses] == list(range(1, 51))


@pytest.mark.asyncio
async def test_nearby_predictions_are_deduplicated_preserving_original_rank(tmp_path: Path) -> None:
    management = build_installed_management(tmp_path)
    coordinates = [[0, 0], [0.01, 0.01], [2, 2], [5, 5], [8, 8], [11, 11], [14, 14]]
    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: FakeRuntime(coordinates),
        device_selector=lambda: "cpu",
    )
    outcome = await provider.predict(_handle(tmp_path), _context())
    assert outcome.value is not None
    assert [item.original_rank for item in outcome.value.hypotheses][:2] == [1, 3]


@pytest.mark.asyncio
async def test_loading_is_single_flight_and_model_is_reused(tmp_path: Path) -> None:
    management = build_installed_management(tmp_path)
    loads = 0

    def loader(_snapshot: Path, _assets: Path, _device: str) -> FakeRuntime:
        nonlocal loads
        loads += 1
        return FakeRuntime()

    provider = GeoCLIPGlobalGeolocationProvider(
        management, model_loader=loader, device_selector=lambda: "cpu"
    )
    handle = _handle(tmp_path)
    await asyncio.gather(
        provider.predict(handle, _context()),
        provider.predict(handle, _context()),
    )
    assert loads == 1
    assert provider.status().verified is True


@pytest.mark.asyncio
async def test_successive_predictions_reuse_runtime_and_remain_deterministic(
    tmp_path: Path,
) -> None:
    management = build_installed_management(tmp_path)
    runtime = FakeRuntime()
    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: runtime,
        device_selector=lambda: "cpu",
    )
    handle = _handle(tmp_path)

    first = await provider.predict(handle, _context())
    second = await provider.predict(handle, _context())

    assert first.status == second.status == OutcomeStatus.SUCCEEDED
    assert first.value is not None and second.value is not None
    assert [
        (item.original_rank, item.latitude, item.longitude, item.raw_score)
        for item in first.value.hypotheses
    ] == [
        (item.original_rank, item.latitude, item.longitude, item.raw_score)
        for item in second.value.hypotheses
    ]
    assert provider._model is runtime


@pytest.mark.asyncio
async def test_invalid_output_and_timeout_are_safe_failures(tmp_path: Path) -> None:
    management = build_installed_management(tmp_path)
    invalid = FakeRuntime([[math.nan, index] for index in range(5)])
    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: invalid,
        device_selector=lambda: "cpu",
    )
    outcome = await provider.predict(_handle(tmp_path), _context())
    assert outcome.failure is not None and outcome.failure.code == "invalid_model_output"
    assert outcome.failure.subreason_code == "all_coordinates_non_finite"
    diagnostics = provider.diagnostics()
    assert diagnostics["output_subreason_code"] == "all_coordinates_non_finite"
    assert str(tmp_path) not in json.dumps(diagnostics, default=str)

    slow = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: FakeRuntime(delay=0.05),
        device_selector=lambda: "cpu",
        timeout_seconds=0.01,
    )
    timeout = await slow.predict(_handle(tmp_path), _context())
    assert timeout.failure is not None and timeout.failure.code == "inference_timeout"


@pytest.mark.parametrize(
    "coordinate",
    [[91, 0], [-91, 0], [math.inf, 0], [0, math.nan]],
)
@pytest.mark.asyncio
async def test_one_invalid_wgs84_coordinate_does_not_erase_valid_candidates(
    tmp_path: Path, coordinate: list[float]
) -> None:
    management = build_installed_management(tmp_path)
    runtime = FakeRuntime([coordinate, [2, 2], [4, 4], [6, 6], [8, 8]])
    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: runtime,
        device_selector=lambda: "cpu",
    )
    outcome = await provider.predict(_handle(tmp_path), _context())
    assert outcome.status == OutcomeStatus.SUCCEEDED
    assert outcome.value is not None
    assert len(outcome.value.hypotheses) == 4
    assert [item.original_rank for item in outcome.value.hypotheses] == [2, 3, 4, 5]


@pytest.mark.asyncio
async def test_longitude_is_normalized_across_the_antimeridian(tmp_path: Path) -> None:
    management = build_installed_management(tmp_path)
    runtime = FakeRuntime([[0, 181], [20, 20], [40, 40], [60, 60], [-20, -20]])
    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: runtime,
        device_selector=lambda: "cpu",
    )
    outcome = await provider.predict(_handle(tmp_path), _context())
    assert outcome.value is not None
    assert outcome.value.hypotheses[0].longitude == -179


@pytest.mark.asyncio
async def test_timeout_returns_after_model_owns_an_immutable_memory_copy(
    tmp_path: Path,
) -> None:
    management = build_installed_management(tmp_path)
    runtime = PayloadLifetimeRuntime(delay=0)
    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: runtime,
        device_selector=lambda: "cpu",
        timeout_seconds=1,
    )
    handle = _handle(tmp_path)
    warmed = await provider.predict(handle, _context())
    assert warmed.status == OutcomeStatus.SUCCEEDED
    runtime.delay = 0.05
    runtime.work_completed.clear()
    started = time.monotonic()
    outcome = await provider.predict(handle, _context(deadline_seconds=0.01))
    elapsed = time.monotonic() - started
    assert outcome.failure is not None and outcome.failure.code == "inference_timeout"
    assert elapsed < 0.04
    handle.path.unlink()
    assert await asyncio.to_thread(runtime.work_completed.wait, 1)


@pytest.mark.asyncio
async def test_not_installed_and_cancelled_do_not_load(tmp_path: Path) -> None:
    management = build_installed_management(tmp_path)
    management.remove(confirmed=True)
    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: FakeRuntime(),
        device_selector=lambda: "cpu",
    )
    missing = await provider.predict(_handle(tmp_path), _context())
    assert missing.failure is not None and missing.failure.code == "model_not_installed"
    cancelled = await provider.predict(_handle(tmp_path), _context(cancelled=True))
    assert cancelled.status == OutcomeStatus.SKIPPED


@pytest.mark.asyncio
async def test_cuda_oom_releases_model_and_clears_cache(tmp_path: Path) -> None:
    management = build_installed_management(tmp_path)
    cleanups = 0

    def cleanup() -> None:
        nonlocal cleanups
        cleanups += 1

    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: OutOfMemoryRuntime(),
        device_selector=lambda: "cuda",
        cuda_cleanup=cleanup,
    )
    outcome = await provider.predict(_handle(tmp_path), _context())
    assert outcome.failure is not None and outcome.failure.code == "cuda_out_of_memory"
    assert cleanups == 1
    assert provider.status().status == "failed"
    assert provider.status().verified is False


@pytest.mark.asyncio
async def test_cuda_oom_during_load_clears_cache(tmp_path: Path) -> None:
    management = build_installed_management(tmp_path)
    cleanups = 0

    def cleanup() -> None:
        nonlocal cleanups
        cleanups += 1

    def loader(_snapshot: Path, _assets: Path, _device: str) -> FakeRuntime:
        raise RuntimeError("CUDA out of memory")

    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=loader,
        device_selector=lambda: "cuda",
        cuda_cleanup=cleanup,
    )
    outcome = await provider.predict(_handle(tmp_path), _context())
    assert outcome.failure is not None and outcome.failure.code == "cuda_out_of_memory"
    assert cleanups == 1


@pytest.mark.asyncio
async def test_initialization_failure_is_not_mislabeled_as_missing_model(
    tmp_path: Path,
) -> None:
    management = build_installed_management(tmp_path)

    def loader(_snapshot: Path, _assets: Path, _device: str) -> FakeRuntime:
        raise RuntimeError("native initialization failed")

    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=loader,
        device_selector=lambda: "cpu",
    )
    outcome = await provider.predict(_handle(tmp_path), _context())
    assert outcome.failure is not None
    assert outcome.failure.code == "internal_provider_error"
    assert provider.status().reason_code == "provider_initialization_failed"


def test_explicit_cpu_and_invalid_device_selection() -> None:
    assert select_device("cpu") == "cpu"
    with pytest.raises(ModelManagementError, match="unsupported_device"):
        select_device("tpu")
