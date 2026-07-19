from __future__ import annotations

import asyncio
import math

import pytest

from atlaslens_api.phase6b.osv5m import (
    InvalidOSV5MOutput,
    OSV5MProvider,
    convert_osv5m_radians,
)
from atlaslens_api.phase6b.scheduler import HeavyModelScheduler


class StubOSVWorker:
    def __init__(self, output: object, *, delay: float = 0.0, oom_on_cuda: bool = False) -> None:
        self.output = output
        self.delay = delay
        self.oom_on_cuda = oom_on_cuda
        self.calls: list[tuple[str, str]] = []

    async def load(self, device: str) -> None:
        self.calls.append(("load", device))

    async def predict_radians(self, image_bytes: bytes, *, device: str) -> object:
        assert image_bytes == b"licensed-image"
        self.calls.append(("predict", device))
        if self.oom_on_cuda and device == "cuda":
            raise RuntimeError("CUDA out of memory")
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.output

    async def unload(self, device: str) -> None:
        self.calls.append(("unload", device))


def provider(worker: StubOSVWorker | None, **updates: object) -> OSV5MProvider:
    values: dict[str, object] = {
        "enabled": True,
        "worker": worker,
        "scheduler": HeavyModelScheduler(),
        "model_revision": "hf-revision",
        "source_revision": "git-revision",
        "device": "cpu",
        "cuda_available": lambda: True,
    }
    values.update(updates)
    return OSV5MProvider(**values)  # type: ignore[arg-type]


def test_official_radian_coordinate_order_is_converted_to_degrees() -> None:
    latitude, longitude = convert_osv5m_radians([[math.radians(38.72), math.radians(215.48)]])
    assert latitude == pytest.approx(38.72)
    assert longitude == pytest.approx(-144.52)


@pytest.mark.parametrize(
    "value",
    [
        [[float("nan"), 0.0]],
        [[math.radians(91), 0.0]],
        [[0.0, 1.0, 2.0]],
        ["not", "numbers"],
    ],
)
def test_invalid_coordinate_output_is_rejected(value: object) -> None:
    with pytest.raises(InvalidOSV5MOutput):
        convert_osv5m_radians(value)


@pytest.mark.asyncio
async def test_cpu_provider_returns_one_direct_coordinate_without_fake_top_k() -> None:
    worker = StubOSVWorker([[math.radians(41.01), math.radians(28.97)]])
    result = await provider(worker).predict(b"licensed-image")
    assert result.status == "completed"
    assert result.device == "cpu"
    assert result.score_semantics == "direct_regression"
    assert len(result.candidates) == 1
    assert result.candidates[0].raw_score is None
    assert result.candidates[0].sample_support == 1
    assert worker.calls == [
        ("load", "cpu"),
        ("predict", "cpu"),
        ("unload", "cpu"),
    ]


@pytest.mark.asyncio
async def test_cuda_path_runs_under_scheduler() -> None:
    worker = StubOSVWorker([[0.1, 0.2]])
    scheduler = HeavyModelScheduler()
    result = await OSV5MProvider(
        enabled=True,
        worker=worker,
        scheduler=scheduler,
        model_revision="hf-revision",
        source_revision="git-revision",
        device="cuda",
        cuda_available=lambda: True,
    ).predict(b"licensed-image")
    assert result.status == "completed"
    assert result.device == "cuda"
    assert worker.calls[:2] == [("load", "cuda"), ("predict", "cuda")]
    assert scheduler.safe_status()["resident_count"] == 1
    await scheduler.close()


@pytest.mark.asyncio
async def test_missing_worker_is_transparently_capability_disabled() -> None:
    instance = provider(None)
    status = instance.status()
    result = await instance.predict(b"licensed-image")
    assert status.state == "not_installed"
    assert status.reason_code == "isolated_worker_not_installed"
    assert result.status == "skipped"
    assert result.reason_code == "isolated_worker_not_installed"


@pytest.mark.asyncio
async def test_timeout_is_safe_and_does_not_raise() -> None:
    worker = StubOSVWorker([[0.1, 0.2]], delay=0.1)
    result = await provider(worker, timeout_seconds=0.01).predict(b"licensed-image")
    assert result.status == "timeout"
    assert result.reason_code == "inference_timeout"


@pytest.mark.asyncio
async def test_cuda_oom_retries_once_on_cpu() -> None:
    worker = StubOSVWorker([[0.1, 0.2]], oom_on_cuda=True)
    result = await provider(worker, device="cuda").predict(b"licensed-image")
    assert result.status == "completed"
    assert result.device == "cpu"
    assert "warning.osv5m.cuda_oom_cpu_fallback" in result.warnings
    assert [call for call in worker.calls if call[0] == "predict"] == [
        ("predict", "cuda"),
        ("predict", "cpu"),
    ]
