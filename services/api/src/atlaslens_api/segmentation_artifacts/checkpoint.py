from __future__ import annotations

import hashlib
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

_HASH_CHUNK_BYTES = 1024 * 1024
_CLASSIFIER_WEIGHT_SUFFIX = "decode_head.classifier.weight"
_CLASSIFIER_BIAS_SUFFIX = "decode_head.classifier.bias"


class SegmentationCheckpointError(ValueError):
    """A stable, tensor-free checkpoint validation error."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        message = code if detail is None else f"{code}: {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class StateDictionarySummary:
    tensor_count: int
    classifier_output_count: int
    classifier_weight_key: str
    classifier_bias_key: str

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "tensor_count": self.tensor_count,
            "classifier_output_count": self.classifier_output_count,
            "classifier_weight_key": self.classifier_weight_key,
            "classifier_bias_key": self.classifier_bias_key,
        }


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    path: Path
    size_bytes: int
    sha256: str
    top_level_keys: tuple[str, ...]
    epoch: int | None
    best_miou: float | None
    patience: int | None
    training_config: dict[str, JsonValue]
    ema: StateDictionarySummary | None
    student: StateDictionarySummary | None
    inferred_num_labels: int
    inferred_model_family: str | None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "top_level_keys": list(self.top_level_keys),
            "epoch": self.epoch,
            "best_miou": self.best_miou,
            "patience": self.patience,
            "training_config": self.training_config,
            "ema_exists": self.ema is not None,
            "student_exists": self.student is not None,
            "ema": None if self.ema is None else self.ema.as_dict(),
            "student": None if self.student is None else self.student.as_dict(),
            "inferred_segmentation_classifier_output_count": self.inferred_num_labels,
            "inferred_model_family": self.inferred_model_family,
        }


@dataclass(frozen=True, slots=True)
class LoadedSegmentationCheckpoint:
    inspection: CheckpointInspection
    ema_state: dict[str, torch.Tensor] | None
    student_state: dict[str, torch.Tensor] | None

    @property
    def selected_weight_source(self) -> str:
        return "ema" if self.ema_state is not None else "student"

    @property
    def selected_state_dict(self) -> dict[str, torch.Tensor]:
        if self.ema_state is not None:
            return self.ema_state
        if self.student_state is not None:
            return self.student_state
        raise SegmentationCheckpointError("missing_inference_weights")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise SegmentationCheckpointError("checkpoint_unreadable") from exc
    return digest.hexdigest()


def _json_value(value: object, *, field: str) -> JsonValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SegmentationCheckpointError("invalid_training_config", field)
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item, field=field) for item in value]
    if isinstance(value, dict):
        converted: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SegmentationCheckpointError("invalid_training_config", field)
            converted[key] = _json_value(item, field=f"{field}.{key}")
        return converted
    raise SegmentationCheckpointError("invalid_training_config", field)


def _optional_nonnegative_int(payload: dict[object, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SegmentationCheckpointError("invalid_checkpoint_scalar", key)
    return value


def _optional_metric(payload: dict[object, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SegmentationCheckpointError("invalid_checkpoint_scalar", key)
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise SegmentationCheckpointError("invalid_checkpoint_scalar", key)
    return converted


def _state_dict(
    payload: dict[object, object], key: str
) -> tuple[dict[str, torch.Tensor] | None, StateDictionarySummary | None]:
    value = payload.get(key)
    if value is None:
        return None, None
    if not isinstance(value, dict) or not value:
        raise SegmentationCheckpointError("invalid_state_dict", key)

    state: dict[str, torch.Tensor] = {}
    for parameter_name, tensor in value.items():
        if not isinstance(parameter_name, str) or not isinstance(tensor, torch.Tensor):
            raise SegmentationCheckpointError("invalid_state_dict", key)
        state[parameter_name] = tensor

    weight_keys = sorted(
        name for name in state if name.casefold().endswith(_CLASSIFIER_WEIGHT_SUFFIX)
    )
    bias_keys = sorted(name for name in state if name.casefold().endswith(_CLASSIFIER_BIAS_SUFFIX))
    if len(weight_keys) != 1 or len(bias_keys) != 1:
        raise SegmentationCheckpointError("classifier_keys_ambiguous", key)
    weight = state[weight_keys[0]]
    bias = state[bias_keys[0]]
    if (
        weight.ndim < 1
        or bias.ndim != 1
        or weight.shape[0] <= 0
        or weight.shape[0] != bias.shape[0]
    ):
        raise SegmentationCheckpointError("classifier_dimensions_inconsistent", key)
    weight_prefix = weight_keys[0][: -len("weight")]
    bias_prefix = bias_keys[0][: -len("bias")]
    if weight_prefix != bias_prefix:
        raise SegmentationCheckpointError("classifier_dimensions_inconsistent", key)
    summary = StateDictionarySummary(
        tensor_count=len(state),
        classifier_output_count=int(weight.shape[0]),
        classifier_weight_key=weight_keys[0],
        classifier_bias_key=bias_keys[0],
    )
    return state, summary


def _model_family(
    training_config: dict[str, JsonValue], state: dict[str, torch.Tensor]
) -> str | None:
    base = training_config.get("student_checkpoint")
    if isinstance(base, str) and "segformer" in base.casefold():
        return "segformer"
    if any(name.casefold().startswith(("segformer.", "decode_head.")) for name in state):
        return "segformer"
    return None


def load_trusted_checkpoint(path: Path) -> LoadedSegmentationCheckpoint:
    """Load a trusted operator checkpoint on CPU and retain inference weights only."""
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise SegmentationCheckpointError("checkpoint_unavailable")
    resolved = expanded.resolve()
    try:
        before = resolved.stat()
        raw: Any = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
        )
        after = resolved.stat()
    except (
        OSError,
        RuntimeError,
        EOFError,
        ValueError,
        TypeError,
        pickle.UnpicklingError,
    ) as exc:
        raise SegmentationCheckpointError("checkpoint_load_failed") from exc
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise SegmentationCheckpointError("checkpoint_changed_during_read")
    if not isinstance(raw, dict) or not raw:
        raise SegmentationCheckpointError("invalid_checkpoint", "top level must be a mapping")
    if any(not isinstance(key, str) for key in raw):
        raise SegmentationCheckpointError("invalid_checkpoint", "top-level keys must be strings")

    training_value = raw.get("training_config", {})
    if training_value is None:
        training_value = {}
    if not isinstance(training_value, dict):
        raise SegmentationCheckpointError("invalid_training_config")
    converted_training = _json_value(training_value, field="training_config")
    if not isinstance(converted_training, dict):
        raise SegmentationCheckpointError("invalid_training_config")
    image_size = converted_training.get("image_size")
    if image_size is not None and (
        isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0
    ):
        raise SegmentationCheckpointError("invalid_training_config", "image_size")
    base_model = converted_training.get("student_checkpoint")
    if base_model is not None and (not isinstance(base_model, str) or not base_model.strip()):
        raise SegmentationCheckpointError("invalid_training_config", "student_checkpoint")

    ema_state, ema_summary = _state_dict(raw, "ema")
    student_state, student_summary = _state_dict(raw, "student")
    if ema_state is None and student_state is None:
        raise SegmentationCheckpointError("missing_inference_weights")
    if (
        ema_state is not None
        and student_state is not None
        and (
            ema_state.keys() != student_state.keys()
            or any(ema_state[name].shape != student_state[name].shape for name in ema_state)
        )
    ):
        raise SegmentationCheckpointError("ema_student_state_mismatch")
    counts = {
        summary.classifier_output_count
        for summary in (ema_summary, student_summary)
        if summary is not None
    }
    if len(counts) != 1:
        raise SegmentationCheckpointError("classifier_dimensions_inconsistent", "ema/student")
    inferred_num_labels = counts.pop()
    selected = ema_state if ema_state is not None else student_state
    assert selected is not None
    inspection = CheckpointInspection(
        path=resolved,
        size_bytes=after.st_size,
        sha256=_sha256(resolved),
        top_level_keys=tuple(sorted(raw)),
        epoch=_optional_nonnegative_int(raw, "epoch"),
        best_miou=_optional_metric(raw, "best_miou"),
        patience=_optional_nonnegative_int(raw, "patience"),
        training_config=converted_training,
        ema=ema_summary,
        student=student_summary,
        inferred_num_labels=inferred_num_labels,
        inferred_model_family=_model_family(converted_training, selected),
    )
    return LoadedSegmentationCheckpoint(
        inspection=inspection,
        ema_state=ema_state,
        student_state=student_state,
    )


def inspect_checkpoint(path: Path) -> CheckpointInspection:
    return load_trusted_checkpoint(path).inspection
