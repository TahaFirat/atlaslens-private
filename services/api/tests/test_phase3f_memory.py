from __future__ import annotations

import hashlib
import importlib.util
import inspect
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from atlaslens_api.phase3f import cloud_job, training
from atlaslens_api.phase3f.memory import (
    BASELINE_MAX_ATTENTION_TOKENS,
    BASELINE_MAX_INPUT_SHAPE,
    BASELINE_MICROBATCH,
    EFFECTIVE_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    PRODUCTION_MEMORY_EVIDENCE_SCHEMA,
    TRAINING_INPUT_SHAPE,
    TRAINING_MICROBATCH,
    MeasuredMemoryRequirement,
    StageOOMError,
    require_production_memory_evidence,
)
from atlaslens_api.phase3f.pipeline import MODEL_SHA256
from atlaslens_api.phase3f.training_recovery import classify_remote_failure

ROOT = Path(__file__).resolve().parents[3]


class _OOM(RuntimeError):
    pass


class _FakeCuda:
    OutOfMemoryError = _OOM

    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    @staticmethod
    def memory_allocated() -> int:
        return 3 * 1024**3

    @staticmethod
    def memory_reserved() -> int:
        return 4 * 1024**3

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_device_properties(_device: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=8 * 1024**3)


class _FakeTensor:
    def __init__(self, marker: int) -> None:
        self.marker = marker
        self.shape = (3, 420, 560)


class _FakeBatch:
    def __init__(self, tensors: list[_FakeTensor]) -> None:
        self.tensors = tensors

    def to(self, _device: str) -> _FakeBatch:
        return self


class _FakeOutput:
    def __init__(self, matrix: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        self.matrix = matrix

    def float(self) -> _FakeOutput:
        return self

    def cpu(self) -> _FakeOutput:
        return self

    def numpy(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        return self.matrix


class _FakeModel:
    def __init__(self, maximum_batch: int) -> None:
        self.maximum_batch = maximum_batch
        self.eval_called = False

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def __call__(self, batch: _FakeBatch) -> _FakeOutput:
        if len(batch.tensors) > self.maximum_batch:
            raise _OOM("CUDA out of memory. Tried to allocate 1.00 GiB")
        matrix = np.zeros((len(batch.tensors), 8448), dtype=np.float32)
        for row, tensor in enumerate(batch.tensors):
            matrix[row, tensor.marker] = 1.0
        return _FakeOutput(matrix)


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()

    @staticmethod
    def stack(tensors: list[_FakeTensor]) -> _FakeBatch:
        return _FakeBatch(tensors)


def _runtime(maximum_batch: int) -> cloud_job.MegaLocRuntime:
    runtime = object.__new__(cloud_job.MegaLocRuntime)
    runtime._torch = _FakeTorch()  # type: ignore[attr-defined]  # noqa: SLF001
    runtime._model = _FakeModel(maximum_batch)  # type: ignore[attr-defined]  # noqa: SLF001
    runtime._last_describe_diagnostic = None  # type: ignore[attr-defined]  # noqa: SLF001
    runtime._tensor = lambda path: _FakeTensor(int(path.name))  # type: ignore[method-assign]
    return runtime


def test_baseline_oom_halves_microbatch_and_preserves_order() -> None:
    runtime = _runtime(maximum_batch=2)

    matrix = runtime.describe(
        [Path(str(index)) for index in range(6)],
        batch_size=4,
        stage="baseline",
    )

    assert np.argmax(matrix, axis=1).tolist() == list(range(6))
    assert runtime._model.eval_called is True  # noqa: SLF001
    assert runtime._last_describe_diagnostic == {  # noqa: SLF001
        "stage": "baseline",
        "requested_microbatch": 4,
        "minimum_microbatch": 2,
        "oom_recoveries": 1,
        "row_count": 6,
        "dtype": "float32",
        "model_mode": "eval",
        "inference_mode": True,
    }
    assert runtime._torch.cuda.empty_cache_calls == 1  # noqa: SLF001


@pytest.mark.parametrize(
    ("stage", "stage_code"),
    (
        ("baseline", "REMOTE_TRAINING_CUDA_OOM_BASELINE"),
        ("validation", "REMOTE_TRAINING_CUDA_OOM_VALIDATION"),
        ("holdout", "REMOTE_TRAINING_CUDA_OOM_HOLDOUT"),
    ),
)
def test_batch_one_oom_is_terminal_and_stage_bound(
    stage: str, stage_code: str
) -> None:
    runtime = _runtime(maximum_batch=0)

    with pytest.raises(StageOOMError) as error:
        runtime.describe(  # type: ignore[arg-type]
            [Path("0")], batch_size=1, stage=stage
        )

    assert error.value.code == "REMOTE_TRAINING_CUDA_OOM_BATCH_ONE"
    assert error.value.stage_error_code == stage_code
    assert error.value.input_shape == (1, 3, 420, 560)
    assert error.value.requested_bytes == 1024**3


def test_microbatch_numerical_equivalence_is_exact_for_eval_model() -> None:
    runtime = _runtime(maximum_batch=4)
    paths = [Path(str(index)) for index in range(4)]

    bulk = runtime.describe(paths, batch_size=4, stage="baseline")
    micro = runtime.describe(paths, batch_size=1, stage="baseline")

    assert bulk.shape == micro.shape == (4, 8448)
    assert np.array_equal(bulk, micro)
    assert np.min(np.sum(bulk * micro, axis=1)) == pytest.approx(1.0)


def _evidence(*, peak_reserved: int = 4 * 1024**3) -> dict[str, object]:
    peak_allocated = peak_reserved - 1024**3
    requirement = MeasuredMemoryRequirement.from_peaks(
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )
    peak = {
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }
    return {
        "schema": PRODUCTION_MEMORY_EVIDENCE_SCHEMA,
        "run_id": "a" * 32,
        "readiness_sha256": "b" * 64,
        "sealed_assets_sha256": "c" * 64,
        "checksum_inventory_sha256": "d" * 64,
        "model_sha256": MODEL_SHA256,
        "gpu_name": "local-test-gpu",
        "gpu_total_bytes": max(8 * 1024**3, peak_reserved),
        "production_input_shape": list(BASELINE_MAX_INPUT_SHAPE),
        "training_input_shape": list(TRAINING_INPUT_SHAPE),
        "attention_token_count": BASELINE_MAX_ATTENTION_TOKENS,
        "baseline_microbatch": BASELINE_MICROBATCH,
        "training_microbatch": TRAINING_MICROBATCH,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "full_reference_descriptor_count": 450,
        "validation_descriptor_count": 150,
        "stage_peaks": {
            name: peak
            for name in ("baseline", "train", "validation", "holdout_probe", "resume")
        },
        "required_gpu_vram_gib": requirement.required_vram_gib,
        "numerical_equivalence": {
            "shape_equal": True,
            "order_equal": True,
            "cosine_min": 0.99999,
            "linf_max": 0.00001,
        },
        "local_cuda_production_shape_passed": True,
        "checkpoint_roundtrip_passed": True,
        "locked_holdout_access_count": 0,
        "dataset_write_count": 0,
        "network_calls": 0,
        "runpod_api_calls": 0,
        "cloud_mutations": 0,
        "nonfinite_loss_count": 0,
        "model_parameters_changed": True,
        "temporary_cleanup_verified": True,
        "secrets_included": False,
    }


def test_production_evidence_is_hash_bound_and_requires_16_gib() -> None:
    measured = require_production_memory_evidence(
        _evidence(),
        run_id="a" * 32,
        readiness_sha256="b" * 64,
        sealed_assets_sha256="c" * 64,
        checksum_inventory_sha256="d" * 64,
        model_sha256=MODEL_SHA256,
    )

    assert measured.requirement.required_vram_gib == 16
    assert measured.requirement.required_headroom_bytes == 2 * 1024**3


def _load_control() -> ModuleType:
    path = ROOT / "scripts" / "phase3f" / "end_to_end.py"
    spec = importlib.util.spec_from_file_location("phase3f_memory_control", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys_modules = __import__("sys").modules
    sys_modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_training_memory_plan_is_read_only_and_selects_48_gib_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = _load_control()
    repository = tmp_path / "repository"
    inventory = tmp_path / "runtime" / ("a" * 32) / "sealed-acquisition"
    inventory.mkdir(parents=True)
    inventory_payload = b"{}\n"
    (inventory / "checksum-inventory.json").write_bytes(inventory_payload)
    monkeypatch.setattr(
        control,
        "resume_plan",
        lambda *_args: {
            "run_id": "a" * 32,
            "asset_count": 830,
            "readiness_sha256": "b" * 64,
            "sealed_assets_sha256": "c" * 64,
        },
    )
    document = _evidence(peak_reserved=30 * 1024**3)
    document["checksum_inventory_sha256"] = hashlib.sha256(
        inventory_payload
    ).hexdigest()
    evidence = (
        repository
        / ".local"
        / "phase3f"
        / "verification"
        / ("a" * 32)
        / "production-memory-smoke.json"
    )
    evidence.parent.mkdir(parents=True)
    evidence.write_text(__import__("json").dumps(document), encoding="utf-8")

    plan = control.training_memory_plan(
        repository,
        tmp_path / "runtime",
        tmp_path / "cloud",
        max_gpu_hourly_usd=Decimal("0.50"),
    )

    assert plan["required_gpu_vram_gib"] == 48
    assert plan["allowed_gpu_classes"] == [
        {"provider_memory_gib": 48, "class": "A40_A6000_FALLBACK"}
    ]
    assert plan["ready_for_cloud"] is True
    assert plan["create_attempts"] == plan["runpod_api_calls"] == 0
    assert plan["network_calls"] == plan["cloud_mutations"] == 0
    assert plan["dataset_writes"] == 0

    price_blocked = control.training_memory_plan(
        repository,
        tmp_path / "runtime",
        tmp_path / "cloud",
        max_gpu_hourly_usd=Decimal("0.51"),
    )
    assert price_blocked["ready_for_cloud"] is False
    assert price_blocked["local_blockers"] == ["GPU_MEMORY_PLAN_UNSATISFIED"]
    assert price_blocked["create_attempts"] == 0


@pytest.mark.parametrize(
    "code",
    (
        "REMOTE_TRAINING_CUDA_OOM_BASELINE",
        "REMOTE_TRAINING_CUDA_OOM_TRAIN",
        "REMOTE_TRAINING_CUDA_OOM_VALIDATION",
        "REMOTE_TRAINING_CUDA_OOM_HOLDOUT",
        "REMOTE_TRAINING_CUDA_OOM_BATCH_ONE",
    ),
)
def test_typed_oom_codes_are_not_masked(code: str) -> None:
    diagnostic = classify_remote_failure(1, explicit_code=code)
    assert diagnostic.failure_code == code


def test_training_gradient_cache_preserves_effective_loss_and_step_boundary() -> None:
    source = inspect.getsource(training.TrainableMegaLocRuntime._effective_batch_step)
    train_source = inspect.getsource(training.TrainableMegaLocRuntime.train)
    checkpoint_source = inspect.getsource(training.TrainableMegaLocRuntime._write_checkpoint)

    assert "self._torch.cat(cached, dim=0)" in source
    assert source.count("self._metric_loss(cached_embeddings, labels)") == 1
    assert source.count("scaler.step(optimizer)") == 1
    assert source.count("scheduler.step()") == 1
    assert "scaled_loss = loss /" not in source
    assert "batch_size=EFFECTIVE_BATCH_SIZE" in train_source
    assert '"accumulation_boundary": True' in checkpoint_source
    assert '"effective_batch_size": EFFECTIVE_BATCH_SIZE' in checkpoint_source


def test_supervisor_memory_gate_precedes_api_and_create() -> None:
    source = (ROOT / "scripts" / "phase3f" / "supervisor.py").read_text(
        encoding="utf-8"
    )
    execute = source[source.index("def _run_execute(") : source.index("def main(")]

    assert execute.index("_require_production_memory_plan(") < execute.index(
        "billing_client.inventory()"
    )
    assert execute.index("min_gpu_memory_gb=minimum_gpu_memory_gib") < execute.index(
        "select_gpu_offer"
    )
    assert execute.index("select_gpu_offer") < execute.index("session.execute")
    assert '"GPU_MEMORY_PLAN_UNSATISFIED"' in execute
    assert '"NVIDIA A40"' in source and '"NVIDIA RTX A6000"' in source
