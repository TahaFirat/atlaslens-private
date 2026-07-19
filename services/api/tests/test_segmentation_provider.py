from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import torch
from PIL import Image

from atlaslens_api.providers.base import InvocationContext, OutcomeStatus
from atlaslens_api.schemas import AnalysisMode
from atlaslens_api.segmentation import (
    HuggingFaceSegmentationRuntime,
    RawSegmentationPrediction,
    SegFormerSceneProvider,
    SegmentationProviderError,
)
from atlaslens_api.storage import LocalImageHandle


def prepared_model(root: Path, *, semantic_labels: bool = True) -> Path:
    root.mkdir()
    metadata = {
        "schema_version": "atlaslens-segmentation-deployment-v1",
        "model_family": "segformer",
        "variant": "b2",
        "source_checkpoint": "last_checkpoint.pt",
        "source_checkpoint_sha256": "a" * 64,
        "source_checkpoint_size_bytes": 439_721_578,
        "weight_source": "ema",
        "base_model": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        "num_labels": 2,
        "image_size": 640,
        "training_epoch": 12,
        "best_miou": 0.24783799030684367,
        "semantic_label_names_available": semantic_labels,
        "label_mapping_source": "test-fixture",
        "prepared_at": "2026-07-14T10:00:00Z",
    }
    (root / "deployment_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"safe-test-placeholder")
    return root


def image_handle(root: Path, *, size: tuple[int, int] = (4, 3)) -> LocalImageHandle:
    path = root / f"{uuid4().hex}.png"
    Image.new("RGB", size, (100, 120, 140)).save(path, format="PNG")
    return LocalImageHandle(key=path.name, path=path)


def context() -> InvocationContext:
    return InvocationContext(
        analysis_id=uuid4(),
        request_id=uuid4().hex,
        mode=AnalysisMode.LOCAL_ONLY,
        cloud_consent=False,
        deadline=datetime.now(UTC) + timedelta(seconds=5),
        cancellation=asyncio.Event(),
    )


class FakeRuntime:
    def __init__(
        self,
        *,
        device: str,
        semantic_labels: bool,
        fail: bool = False,
        delay: float = 0,
    ) -> None:
        self.device = device
        self.num_labels = 2
        self.id2label = {0: "Road", 1: "Building"}
        self.semantic_label_names_available = semantic_labels
        self.fail = fail
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.maximum_active = 0
        self._guard = threading.Lock()

    def predict(self, image_payload: bytes) -> RawSegmentationPrediction:
        assert image_payload.startswith(b"\x89PNG")
        with self._guard:
            self.calls += 1
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise SegmentationProviderError("inference_failed")
            return RawSegmentationPrediction(
                image_width=4,
                image_height=3,
                class_pixel_counts=(8, 4),
                inference_ms=9,
            )
        finally:
            with self._guard:
                self.active -= 1


async def test_disabled_and_unavailable_providers_never_load(tmp_path: Path) -> None:
    loader_calls = 0

    def loader(*_: object) -> FakeRuntime:
        nonlocal loader_calls
        loader_calls += 1
        return FakeRuntime(device="cpu", semantic_labels=True)

    missing = tmp_path / "missing-model"
    handle = LocalImageHandle(key="missing.png", path=tmp_path / "missing.png")
    disabled = SegFormerSceneProvider(
        enabled=False,
        model_directory=missing,
        runtime_loader=loader,
    )
    unavailable = SegFormerSceneProvider(
        enabled=True,
        model_directory=missing,
        runtime_loader=loader,
    )

    assert disabled.status().status == "disabled"
    assert unavailable.status().status == "not_installed"
    assert (await disabled.analyze(handle, context())).status == OutcomeStatus.SKIPPED
    outcome = await unavailable.analyze(handle, context())
    assert outcome.status == OutcomeStatus.SKIPPED
    assert outcome.failure is not None
    assert outcome.failure.code == "model_not_installed"
    assert loader_calls == 0


async def test_cpu_runtime_loads_once_and_one_slot_serializes_inference(
    tmp_path: Path,
) -> None:
    model = prepared_model(tmp_path / "prepared")
    handle = image_handle(tmp_path)
    runtime = FakeRuntime(device="cpu", semantic_labels=True, delay=0.03)
    loader_calls: list[tuple[Path, str]] = []

    def loader(path: Path, device: str, _metadata: object) -> FakeRuntime:
        loader_calls.append((path, device))
        return runtime

    provider = SegFormerSceneProvider(
        enabled=True,
        model_directory=model,
        requested_device="auto",
        runtime_loader=loader,
        device_selector=lambda requested: "cpu" if requested == "auto" else requested,
    )

    before = provider.status()
    assert before.status == "ready"
    assert before.prepared and before.usable and not before.loaded
    first, second = await asyncio.gather(
        provider.analyze(handle, context()),
        provider.analyze(handle, context()),
    )

    assert first.status == OutcomeStatus.SUCCEEDED
    assert second.status == OutcomeStatus.SUCCEEDED
    assert first.value is not None
    assert first.value.device == "cpu"
    assert first.value.inference_ms == 9
    assert first.value.scene_groups["road_surface"] == pytest.approx(8 / 12, abs=1e-6)
    assert len(loader_calls) == 1
    assert loader_calls[0][1] == "cpu"
    assert runtime.calls == 2
    assert runtime.maximum_active == 1
    after = provider.status()
    assert after.loaded and after.weight_source == "ema"
    assert after.checkpoint_sha256 == "a" * 64


async def test_mocked_cuda_selection_and_generic_label_fallback(tmp_path: Path) -> None:
    model = prepared_model(tmp_path / "prepared", semantic_labels=False)
    handle = image_handle(tmp_path)
    selected: list[str] = []

    def loader(_path: Path, device: str, _metadata: object) -> FakeRuntime:
        selected.append(device)
        return FakeRuntime(device="cuda", semantic_labels=False)

    provider = SegFormerSceneProvider(
        enabled=True,
        model_directory=model,
        requested_device="cuda",
        runtime_loader=loader,
        device_selector=lambda requested: requested,
    )
    outcome = await provider.analyze(handle, context())

    assert outcome.status == OutcomeStatus.SUCCEEDED
    assert outcome.value is not None
    assert selected == ["cuda"]
    assert outcome.value.device == "cuda"
    assert not outcome.value.semantic_label_names_available
    assert outcome.value.scene_groups == {}
    assert outcome.value.scene_tags == ()
    assert outcome.value.warnings == ("segmentation.semantic_label_names_unavailable",)


async def test_runtime_failure_is_a_safe_optional_provider_outcome(tmp_path: Path) -> None:
    model = prepared_model(tmp_path / "prepared")
    handle = image_handle(tmp_path)
    runtime = FakeRuntime(device="cpu", semantic_labels=True, fail=True)
    provider = SegFormerSceneProvider(
        enabled=True,
        model_directory=model,
        runtime_loader=lambda *_: runtime,
        device_selector=lambda _: "cpu",
    )

    outcome = await provider.analyze(handle, context())

    assert outcome.status == OutcomeStatus.FAILED
    assert outcome.failure is not None
    assert outcome.failure.code == "invalid_model_output"
    assert outcome.failure.subreason_code == "inference_failed"
    assert "prepared" not in repr(outcome)


class FakeProcessor:
    def __call__(self, *, images: Image.Image, return_tensors: str) -> dict[str, Any]:
        assert images.size == (2, 2)
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32)}


class FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.device: str | None = None

    def eval(self) -> None:
        self.eval_called = True

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self

    def __call__(self, **_: object) -> SimpleNamespace:
        logits = torch.tensor(
            [
                [
                    [[2.0, 0.0], [0.0, 2.0]],
                    [[0.0, 2.0], [2.0, 0.0]],
                ]
            ]
        )
        return SimpleNamespace(logits=logits)


def test_huggingface_runtime_uses_eval_inference_mode_and_original_dimensions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.png"
    Image.new("RGB", (2, 2), (50, 60, 70)).save(path, format="PNG")
    model = FakeModel()
    runtime = HuggingFaceSegmentationRuntime(
        model=model,
        processor=FakeProcessor(),
        torch_module=torch,
        functional=torch.nn.functional,
        device="cpu",
        num_labels=2,
        id2label={0: "Road", 1: "Building"},
        semantic_label_names_available=True,
    )

    output = runtime.predict(path.read_bytes())

    assert model.eval_called
    assert model.device == "cpu"
    assert output.image_width == 2
    assert output.image_height == 2
    assert output.class_pixel_counts == (2, 2)
    assert not hasattr(output, "mask")
