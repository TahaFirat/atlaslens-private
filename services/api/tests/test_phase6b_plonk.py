from __future__ import annotations

import asyncio

import pytest

from atlaslens_api.phase6b.plonk import (
    PlonkModelRouter,
    PlonkProvider,
    PlonkWorkerOutput,
    cluster_plonk_samples,
)
from atlaslens_api.phase6b.scheduler import HeavyModelScheduler
from atlaslens_api.schemas import SceneGroupSummary, SceneSegmentationSummary


def scene(**groups: float) -> SceneSegmentationSummary:
    return SceneSegmentationSummary(
        provider="segformer-b2-local",
        device="cuda",
        inference_ms=20,
        image_width=640,
        image_height=640,
        semantic_label_names_available=True,
        dominant_classes=[],
        scene_groups=[
            SceneGroupSummary(name=name, pixel_ratio=value, percentage=value * 100)
            for name, value in groups.items()
        ],
    )


def test_router_selects_one_specialization_from_scene_groups() -> None:
    router = PlonkModelRouter()
    street = router.route(scene(road_surface=0.22, sidewalk=0.08, built_environment=0.25))
    nature = router.route(
        scene(vegetation=0.58, terrain=0.20, road_surface=0.01, built_environment=0.02)
    )
    mixed = router.route(scene(build_environment=0.30, vegetation=0.20))
    assert street.kind == "osv5m"
    assert street.model_id == "nicolas-dufour/PLONK_OSV_5M"
    assert nature.kind == "inat"
    assert nature.model_id == "nicolas-dufour/PLONK_iNaturalist"
    assert mixed.kind == "yfcc"
    assert router.route(None).kind == "yfcc"


def test_router_diagnostic_override_is_explicit() -> None:
    route = PlonkModelRouter().route(scene(road_surface=0.5), override="inat")
    assert route.kind == "inat"
    assert route.explicit_override is True
    assert route.reason_code == "phase6b.plonk.explicit_diagnostic_override"


def test_sample_clustering_preserves_density_and_handles_dateline() -> None:
    candidates = cluster_plonk_samples(
        ((10.0, 179.8), (10.0, -179.9), (10.1, 179.9), (-30.0, 20.0)),
        radius_km=100,
    )
    assert len(candidates) == 2
    assert candidates[0].sample_support == 3
    assert candidates[0].raw_score == pytest.approx(0.75)
    assert abs(abs(candidates[0].longitude) - 180) < 0.3


class StubPlonkWorker:
    def __init__(self, output: object, *, delay: float = 0.0) -> None:
        self.output = output
        self.delay = delay
        self.loaded: list[str] = []
        self.sampled: list[str] = []
        self.unloaded: list[str] = []

    async def load(self, model_id: str, device: str) -> None:
        self.loaded.append(model_id)

    async def sample(
        self,
        image_bytes: bytes,
        *,
        model_id: str,
        sample_count: int,
        device: str,
    ) -> PlonkWorkerOutput:
        assert image_bytes == b"licensed-image"
        assert sample_count == 4
        self.sampled.append(model_id)
        if self.delay:
            await asyncio.sleep(self.delay)
        return PlonkWorkerOutput(self.output, localizability=0.25)

    async def unload(self, model_id: str, device: str) -> None:
        self.unloaded.append(model_id)


def provider(worker: StubPlonkWorker, **updates: object) -> PlonkProvider:
    ids = {
        "nicolas-dufour/PLONK_OSV_5M": "osv-revision",
        "nicolas-dufour/PLONK_YFCC": "yfcc-revision",
        "nicolas-dufour/PLONK_iNaturalist": "inat-revision",
    }
    values: dict[str, object] = {
        "enabled": True,
        "worker": worker,
        "scheduler": HeavyModelScheduler(),
        "model_revisions": ids,
        "source_revision": "diff-plonk-0.4",
        "device": "cpu",
        "sample_count": 4,
    }
    values.update(updates)
    return PlonkProvider(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_provider_invokes_only_the_routed_model() -> None:
    worker = StubPlonkWorker([[41.0, 29.0], [41.1, 29.1], [40.9, 28.9], [0.0, 0.0]])
    result = await provider(worker).predict(
        b"licensed-image",
        scene=scene(road_surface=0.25, sidewalk=0.1, built_environment=0.3),
    )
    assert result.status == "completed"
    assert result.model_id == "nicolas-dufour/PLONK_OSV_5M"
    assert result.source_family == "osv5m_family"
    assert worker.loaded == [result.model_id]
    assert worker.sampled == [result.model_id]
    assert worker.unloaded == [result.model_id]
    assert result.diagnostics["raw_localizability"] == 0.25


@pytest.mark.asyncio
async def test_empty_or_invalid_samples_fail_without_fabrication() -> None:
    worker = StubPlonkWorker([[float("nan"), 0], [91, 0], ["x", "y"], [1]])
    result = await provider(worker).predict(b"licensed-image", scene=None)
    assert result.status == "failed"
    assert result.reason_code == "invalid_model_output"
    assert result.candidates == ()


@pytest.mark.asyncio
async def test_timeout_is_neutral_and_does_not_run_a_second_specialization() -> None:
    worker = StubPlonkWorker([[1.0, 2.0]] * 4, delay=0.1)
    result = await provider(worker, timeout_seconds=0.01).predict(b"licensed-image", scene=None)
    assert result.status == "timeout"
    assert result.reason_code == "inference_timeout"
    assert worker.sampled == ["nicolas-dufour/PLONK_YFCC"]
