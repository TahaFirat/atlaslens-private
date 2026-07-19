from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch

from atlaslens_api.segmentation_artifacts.checkpoint import (
    SegmentationCheckpointError,
    inspect_checkpoint,
    load_trusted_checkpoint,
)
from atlaslens_api.segmentation_artifacts.preparation import prepare_segmentation_model


def _state_dict(num_labels: int = 3, *, marker: float = 0.0) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        {
            "segformer.encoder.patch_embeddings.0.proj.weight": torch.full((2, 3, 1, 1), marker),
            "decode_head.classifier.weight": torch.full((num_labels, 4, 1, 1), marker),
            "decode_head.classifier.bias": torch.full((num_labels,), marker),
        }
    )


def _checkpoint(
    path: Path,
    *,
    ema: bool = True,
    student: bool = True,
    num_labels: int = 3,
) -> Path:
    payload: dict[str, object] = {
        "epoch": 12,
        "best_miou": 0.25,
        "patience": 1,
        "training_config": {
            "student_checkpoint": "nvidia/segformer-b2-test",
            "image_size": 640,
        },
    }
    if ema:
        payload["ema"] = _state_dict(num_labels, marker=2.0)
    if student:
        payload["student"] = _state_dict(num_labels, marker=1.0)
    torch.save(payload, path)
    return path


class _FakeModel:
    def __init__(self, *, fail_load: bool = False) -> None:
        self.fail_load = fail_load
        self.loaded_marker: float | None = None

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return _state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], *, strict: bool) -> object:
        assert strict is True
        if self.fail_load:
            raise RuntimeError("Missing key(s): secret.weight; Unexpected key(s): old.weight")
        self.loaded_marker = float(state_dict["decode_head.classifier.bias"][0])
        return object()

    def save_pretrained(self, directory: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        (directory / "config.json").write_text("{}\n", encoding="utf-8")
        (directory / "model.safetensors").write_bytes(b"safe-test-weights")


class _FakeProcessor:
    size: object = None

    def save_pretrained(self, directory: Path) -> None:
        (directory / "preprocessor_config.json").write_text(
            json.dumps({"size": self.size}), encoding="utf-8"
        )


class _Factories:
    def __init__(self, *, fail_load: bool = False) -> None:
        self.calls = 0
        self.model = _FakeModel(fail_load=fail_load)
        self.processor = _FakeProcessor()

    def model_factory(self, _source: str, **kwargs: Any) -> _FakeModel:
        assert kwargs["local_files_only"] is True
        assert kwargs["ignore_mismatched_sizes"] is True
        self.calls += 1
        return self.model

    def processor_factory(self, _source: str, **kwargs: Any) -> _FakeProcessor:
        assert kwargs["local_files_only"] is True
        return self.processor


def test_valid_ema_checkpoint_is_inspected_without_tensor_values(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "checkpoint.pt")

    loaded = load_trusted_checkpoint(path)
    public = loaded.inspection.as_dict()

    assert loaded.selected_weight_source == "ema"
    assert loaded.inspection.ema is not None
    assert loaded.inspection.ema.tensor_count == 3
    assert loaded.inspection.inferred_num_labels == 3
    assert loaded.inspection.inferred_model_family == "segformer"
    assert public["ema_exists"] is True
    assert "Tensor" not in json.dumps(public)


def test_valid_student_only_checkpoint_is_supported(tmp_path: Path) -> None:
    loaded = load_trusted_checkpoint(_checkpoint(tmp_path / "checkpoint.pt", ema=False))

    assert loaded.selected_weight_source == "student"
    assert loaded.inspection.ema is None
    assert loaded.inspection.student is not None


def test_missing_weight_keys_and_invalid_checkpoint_fail_cleanly(tmp_path: Path) -> None:
    missing = tmp_path / "missing-weights.pt"
    torch.save({"training_config": {}}, missing)
    invalid = tmp_path / "invalid.pt"
    invalid.write_bytes(b"not a checkpoint")

    with pytest.raises(SegmentationCheckpointError, match="missing_inference_weights"):
        inspect_checkpoint(missing)
    with pytest.raises(SegmentationCheckpointError, match="checkpoint_load_failed"):
        inspect_checkpoint(invalid)


def test_classifier_dimensions_are_inferred_and_validated(tmp_path: Path) -> None:
    valid = inspect_checkpoint(_checkpoint(tmp_path / "valid.pt", num_labels=7))
    inconsistent = _checkpoint(tmp_path / "inconsistent.pt")
    payload = torch.load(inconsistent, map_location="cpu", weights_only=True)
    payload["ema"]["decode_head.classifier.bias"] = torch.zeros(4)
    torch.save(payload, inconsistent)

    assert valid.inferred_num_labels == 7
    with pytest.raises(SegmentationCheckpointError, match="classifier_dimensions_inconsistent"):
        inspect_checkpoint(inconsistent)


def test_ema_and_student_must_have_compatible_state_shapes(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "incompatible.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["student"].pop("segformer.encoder.patch_embeddings.0.proj.weight")
    torch.save(payload, checkpoint)

    with pytest.raises(SegmentationCheckpointError, match="ema_student_state_mismatch"):
        inspect_checkpoint(checkpoint)


def test_checkpoint_sha256_is_exact(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "checkpoint.pt")

    inspection = inspect_checkpoint(path)

    assert inspection.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inspection.size_bytes == path.stat().st_size


def test_preparation_prefers_ema_writes_safe_metadata_and_is_idempotent(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt")
    output = tmp_path / "models" / "segformer"
    factories = _Factories()

    first = prepare_segmentation_model(
        checkpoint,
        output,
        repository_root=tmp_path,
        model_factory=factories.model_factory,
        processor_factory=factories.processor_factory,
    )
    second = prepare_segmentation_model(
        checkpoint,
        output,
        repository_root=tmp_path,
        model_factory=factories.model_factory,
        processor_factory=factories.processor_factory,
    )

    assert first.changed is True
    assert second.changed is False
    assert factories.calls == 1
    assert factories.model.loaded_marker == 2.0
    assert factories.processor.size == {"height": 640, "width": 640}
    assert first.metadata["weight_source"] == "ema"
    assert first.metadata["state_dict_adapter"] == "identity"
    assert (
        first.metadata["source_checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert first.metadata["semantic_label_names_available"] is False
    assert (output / "model.safetensors").is_file()
    assert not (output / "pytorch_model.bin").exists()


def test_preparation_uses_exact_operator_label_config(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt")
    labels = tmp_path / "config_v2.0.json"
    labels.write_text(
        json.dumps({"labels": [{"name": "Road"}, {"name": "Sky"}, {"name": "Car"}]}),
        encoding="utf-8",
    )
    factories = _Factories()

    result = prepare_segmentation_model(
        checkpoint,
        tmp_path / "model",
        repository_root=tmp_path,
        model_factory=factories.model_factory,
        processor_factory=factories.processor_factory,
    )

    assert result.metadata["semantic_label_names_available"] is True
    assert result.metadata["label_mapping_source"] == "repository:config_v2.0.json"


def test_preparation_adapts_reviewed_modular_segformer_names_then_loads_strictly(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for source in ("ema", "student"):
        state = payload[source]
        state["segformer.stages.0.patch_embeddings.proj.weight"] = state.pop(
            "segformer.encoder.patch_embeddings.0.proj.weight"
        )
    torch.save(payload, checkpoint)
    factories = _Factories()

    result = prepare_segmentation_model(
        checkpoint,
        tmp_path / "model",
        repository_root=tmp_path,
        model_factory=factories.model_factory,
        processor_factory=factories.processor_factory,
    )

    assert result.metadata["state_dict_adapter"] == ("transformers_modular_segformer_to_legacy_v1")
    assert factories.model.loaded_marker == 2.0


def test_strict_state_dict_failure_is_bounded_and_does_not_publish(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt")
    output = tmp_path / "model"
    factories = _Factories(fail_load=True)

    with pytest.raises(SegmentationCheckpointError, match="strict_state_dict_load_failed"):
        prepare_segmentation_model(
            checkpoint,
            output,
            repository_root=tmp_path,
            model_factory=factories.model_factory,
            processor_factory=factories.processor_factory,
        )

    assert not output.exists()
