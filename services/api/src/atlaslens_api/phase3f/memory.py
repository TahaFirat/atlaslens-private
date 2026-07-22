"""Measured, fail-closed Phase 3F CUDA memory contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, Literal, cast

BASELINE_MICROBATCH: Final = 4
TRAINING_MICROBATCH: Final = 4
EFFECTIVE_BATCH_SIZE: Final = 16
GRADIENT_ACCUMULATION_STEPS: Final = EFFECTIVE_BATCH_SIZE // TRAINING_MICROBATCH
BASELINE_MAX_INPUT_SHAPE: Final = (3, 420, 560)
TRAINING_INPUT_SHAPE: Final = (3, 392, 392)
BASELINE_MAX_ATTENTION_TOKENS: Final = 1 + (420 // 14) * (560 // 14)
TRAINING_ATTENTION_TOKENS: Final = 1 + (392 // 14) * (392 // 14)
MINIMUM_CLOUD_VRAM_GIB: Final = 16
FALLBACK_CLOUD_VRAM_GIB: Final = 48
MINIMUM_HEADROOM_BYTES: Final = 2 * 1024**3
MAX_GPU_HOURLY_USD: Final = Decimal("0.50")
NUMERICAL_COSINE_MINIMUM: Final = 0.9999
NUMERICAL_LINF_TOLERANCE: Final = 2e-4
PRODUCTION_MEMORY_EVIDENCE_SCHEMA: Final = (
    "atlaslens-phase3f-production-memory-smoke-v1"
)

OOMStage = Literal["baseline", "train", "validation", "holdout"]

_STAGE_CODES: Final[dict[OOMStage, str]] = {
    "baseline": "REMOTE_TRAINING_CUDA_OOM_BASELINE",
    "train": "REMOTE_TRAINING_CUDA_OOM_TRAIN",
    "validation": "REMOTE_TRAINING_CUDA_OOM_VALIDATION",
    "holdout": "REMOTE_TRAINING_CUDA_OOM_HOLDOUT",
}
_REQUESTED = re.compile(r"Tried to allocate ([0-9]+(?:\.[0-9]+)?) (GiB|MiB|KiB)")


def stage_oom_code(stage: OOMStage) -> str:
    return _STAGE_CODES[stage]


def _requested_bytes(exc: BaseException) -> int | None:
    match = _REQUESTED.search(str(exc))
    if match is None:
        return None
    multiplier = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * multiplier)


class StageOOMError(RuntimeError):
    """Sanitized terminal OOM with a stable stage and allocator snapshot."""

    def __init__(
        self,
        *,
        stage: OOMStage,
        batch_size: int,
        input_shape: tuple[int, ...],
        torch_module: Any,
        source: BaseException,
    ) -> None:
        self.code = (
            "REMOTE_TRAINING_CUDA_OOM_BATCH_ONE"
            if batch_size == 1
            else stage_oom_code(stage)
        )
        self.stage_error_code = stage_oom_code(stage)
        self.stage = stage
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.requested_bytes = _requested_bytes(source)
        self.allocated_bytes = int(torch_module.cuda.memory_allocated())
        self.reserved_bytes = int(torch_module.cuda.memory_reserved())
        self.total_bytes = int(
            torch_module.cuda.get_device_properties(
                torch_module.cuda.current_device()
            ).total_memory
        )
        super().__init__(self.code)

    def diagnostics(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "stage_error_code": self.stage_error_code,
            "batch_size": self.batch_size,
            "input_shape": list(self.input_shape),
            "requested_bytes": self.requested_bytes,
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "total_bytes": self.total_bytes,
            "batch_one": self.batch_size == 1,
        }


class MemoryPlanError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class MeasuredMemoryRequirement:
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    required_headroom_bytes: int
    required_vram_gib: int

    @classmethod
    def from_peaks(
        cls,
        *,
        peak_allocated_bytes: int,
        peak_reserved_bytes: int,
    ) -> MeasuredMemoryRequirement:
        if peak_allocated_bytes <= 0 or peak_reserved_bytes < peak_allocated_bytes:
            raise ValueError("GPU_MEMORY_MEASUREMENT_INVALID")
        headroom = max(MINIMUM_HEADROOM_BYTES, math.ceil(peak_reserved_bytes * 0.25))
        measured_gib = math.ceil((peak_reserved_bytes + headroom) / 1024**3)
        required = next(
            (
                size
                for size in (MINIMUM_CLOUD_VRAM_GIB, 24, FALLBACK_CLOUD_VRAM_GIB)
                if size >= measured_gib
            ),
            measured_gib,
        )
        return cls(
            peak_allocated_bytes=peak_allocated_bytes,
            peak_reserved_bytes=peak_reserved_bytes,
            required_headroom_bytes=headroom,
            required_vram_gib=required,
        )


@dataclass(frozen=True, slots=True)
class ProductionMemoryEvidence:
    gpu_name: str
    gpu_total_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    requirement: MeasuredMemoryRequirement
    checkpoint_roundtrip_passed: bool
    numerical_cosine_min: float
    numerical_linf_max: float


def _evidence_require(condition: bool) -> None:
    if not condition:
        raise MemoryPlanError("LOCAL_CUDA_PRODUCTION_SHAPE_INVALID")


def require_production_memory_evidence(
    document: Mapping[str, object],
    *,
    run_id: str,
    readiness_sha256: str,
    sealed_assets_sha256: str,
    checksum_inventory_sha256: str,
    model_sha256: str,
) -> ProductionMemoryEvidence:
    """Validate the offline, hash-bound production-shape CUDA receipt."""
    _evidence_require(
        document.get("schema") == PRODUCTION_MEMORY_EVIDENCE_SCHEMA
        and document.get("run_id") == run_id
        and document.get("readiness_sha256") == readiness_sha256
        and document.get("sealed_assets_sha256") == sealed_assets_sha256
        and document.get("checksum_inventory_sha256")
        == checksum_inventory_sha256
        and document.get("model_sha256") == model_sha256
        and document.get("local_cuda_production_shape_passed") is True
        and document.get("full_reference_descriptor_count") == 450
        and document.get("validation_descriptor_count") == 150
        and document.get("locked_holdout_access_count") == 0
        and document.get("dataset_write_count") == 0
        and document.get("network_calls") == 0
        and document.get("runpod_api_calls") == 0
        and document.get("cloud_mutations") == 0
        and document.get("nonfinite_loss_count") == 0
        and document.get("model_parameters_changed") is True
        and document.get("checkpoint_roundtrip_passed") is True
        and document.get("temporary_cleanup_verified") is True
        and document.get("secrets_included") is False
        and document.get("baseline_microbatch") == BASELINE_MICROBATCH
        and document.get("training_microbatch") == TRAINING_MICROBATCH
        and document.get("gradient_accumulation_steps")
        == GRADIENT_ACCUMULATION_STEPS
        and document.get("effective_batch_size") == EFFECTIVE_BATCH_SIZE
        and document.get("production_input_shape")
        == list(BASELINE_MAX_INPUT_SHAPE)
        and document.get("training_input_shape") == list(TRAINING_INPUT_SHAPE)
        and document.get("attention_token_count")
        == BASELINE_MAX_ATTENTION_TOKENS
    )
    gpu_name = document.get("gpu_name")
    gpu_total_bytes = document.get("gpu_total_bytes")
    stages = document.get("stage_peaks")
    numerical = document.get("numerical_equivalence")
    _evidence_require(
        isinstance(gpu_name, str)
        and 1 <= len(gpu_name) <= 128
        and isinstance(gpu_total_bytes, int)
        and not isinstance(gpu_total_bytes, bool)
        and gpu_total_bytes > 0
        and isinstance(stages, Mapping)
        and isinstance(numerical, Mapping)
    )
    typed_stages = cast(Mapping[str, object], stages)
    typed_numerical = cast(Mapping[str, object], numerical)
    typed_gpu_total_bytes = cast(int, gpu_total_bytes)
    allocated: list[int] = []
    reserved: list[int] = []
    for stage in ("baseline", "train", "validation", "holdout_probe", "resume"):
        raw_row = typed_stages.get(stage)
        _evidence_require(isinstance(raw_row, Mapping))
        row = cast(Mapping[str, object], raw_row)
        stage_allocated = row.get("peak_allocated_bytes")
        stage_reserved = row.get("peak_reserved_bytes")
        _evidence_require(
            isinstance(stage_allocated, int)
            and not isinstance(stage_allocated, bool)
            and stage_allocated > 0
            and isinstance(stage_reserved, int)
            and not isinstance(stage_reserved, bool)
            and stage_reserved >= stage_allocated
            and stage_reserved <= typed_gpu_total_bytes
        )
        allocated.append(cast(int, stage_allocated))
        reserved.append(cast(int, stage_reserved))
    cosine = typed_numerical.get("cosine_min")
    linf = typed_numerical.get("linf_max")
    _evidence_require(
        isinstance(cosine, int | float)
        and not isinstance(cosine, bool)
        and float(cosine) >= NUMERICAL_COSINE_MINIMUM
        and isinstance(linf, int | float)
        and not isinstance(linf, bool)
        and float(linf) <= NUMERICAL_LINF_TOLERANCE
        and typed_numerical.get("shape_equal") is True
        and typed_numerical.get("order_equal") is True
    )
    requirement = MeasuredMemoryRequirement.from_peaks(
        peak_allocated_bytes=max(allocated),
        peak_reserved_bytes=max(reserved),
    )
    _evidence_require(document.get("required_gpu_vram_gib") == requirement.required_vram_gib)
    return ProductionMemoryEvidence(
        gpu_name=cast(str, gpu_name),
        gpu_total_bytes=typed_gpu_total_bytes,
        peak_allocated_bytes=max(allocated),
        peak_reserved_bytes=max(reserved),
        requirement=requirement,
        checkpoint_roundtrip_passed=True,
        numerical_cosine_min=float(cast(float, cosine)),
        numerical_linf_max=float(cast(float, linf)),
    )


__all__ = [
    "BASELINE_MAX_ATTENTION_TOKENS",
    "BASELINE_MAX_INPUT_SHAPE",
    "BASELINE_MICROBATCH",
    "EFFECTIVE_BATCH_SIZE",
    "FALLBACK_CLOUD_VRAM_GIB",
    "GRADIENT_ACCUMULATION_STEPS",
    "MAX_GPU_HOURLY_USD",
    "MemoryPlanError",
    "MeasuredMemoryRequirement",
    "NUMERICAL_COSINE_MINIMUM",
    "NUMERICAL_LINF_TOLERANCE",
    "PRODUCTION_MEMORY_EVIDENCE_SCHEMA",
    "ProductionMemoryEvidence",
    "StageOOMError",
    "TRAINING_ATTENTION_TOKENS",
    "TRAINING_INPUT_SHAPE",
    "TRAINING_MICROBATCH",
    "require_production_memory_evidence",
    "stage_oom_code",
]
