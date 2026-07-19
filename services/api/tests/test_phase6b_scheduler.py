from __future__ import annotations

import asyncio

import pytest

from atlaslens_api.phase6b.scheduler import HeavyModelScheduler


@pytest.mark.asyncio
async def test_heavy_cuda_inference_is_serial_by_default() -> None:
    scheduler = HeavyModelScheduler(max_heavy_concurrency=1, max_resident_models=1)
    active = 0
    maximum = 0

    async def run(name: str) -> str:
        async def infer(device: str) -> str:
            nonlocal active, maximum
            assert device == "cuda"
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            return name

        execution = await scheduler.execute(
            model_name=name,
            requested_device="cuda",
            estimated_vram_mb=100,
            load=lambda _device: None,
            infer=infer,
            unload=lambda _device: None,
        )
        return execution.value

    assert await asyncio.gather(run("one"), run("two")) == ["one", "two"]
    assert maximum == 1
    assert scheduler.safe_status()["max_observed_cuda"] == 1
    await scheduler.close()


@pytest.mark.asyncio
async def test_resident_limit_evicts_lru_and_clears_cache_only_after_release() -> None:
    unloaded: list[str] = []
    cache_clears = 0

    def empty_cache() -> None:
        nonlocal cache_clears
        cache_clears += 1

    scheduler = HeavyModelScheduler(
        max_resident_models=1,
        empty_cuda_cache=empty_cache,
    )

    async def execute(name: str) -> None:
        await scheduler.execute(
            model_name=name,
            requested_device="cuda",
            estimated_vram_mb=100,
            load=lambda _device: None,
            infer=lambda _device: None,
            unload=lambda _device: unloaded.append(name),
        )

    await execute("one")
    await execute("one")
    assert unloaded == []
    assert cache_clears == 0
    await execute("two")
    assert unloaded == ["one"]
    assert cache_clears == 1
    await scheduler.close()
    assert unloaded == ["one", "two"]
    assert cache_clears == 2


@pytest.mark.asyncio
async def test_cuda_oom_releases_model_and_retries_cpu_once() -> None:
    calls: list[tuple[str, str]] = []
    cache_clears = 0

    def empty_cache() -> None:
        nonlocal cache_clears
        cache_clears += 1

    async def infer(device: str) -> str:
        calls.append(("infer", device))
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return "cpu-result"

    scheduler = HeavyModelScheduler(empty_cuda_cache=empty_cache)
    execution = await scheduler.execute(
        model_name="large-model",
        requested_device="cuda",
        estimated_vram_mb=4_000,
        load=lambda device: calls.append(("load", device)),
        infer=infer,
        unload=lambda device: calls.append(("unload", device)),
        allow_cpu_fallback=True,
    )
    assert execution.value == "cpu-result"
    assert execution.diagnostic.status == "oom_cpu_fallback"
    assert execution.diagnostic.actual_device == "cpu"
    assert execution.diagnostic.cpu_fallback_used is True
    assert calls == [
        ("load", "cuda"),
        ("infer", "cuda"),
        ("unload", "cuda"),
        ("load", "cpu"),
        ("infer", "cpu"),
        ("unload", "cpu"),
    ]
    assert cache_clears == 1
