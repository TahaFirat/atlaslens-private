from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from atlaslens_api.segmentation_artifacts.checkpoint import (
    JsonValue,
    LoadedSegmentationCheckpoint,
    SegmentationCheckpointError,
    load_trusted_checkpoint,
)

_DEFAULT_BASE_MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
_DEFAULT_MODEL_NAME = "atlaslens-segformer-b2-v4"
_METADATA_FILE = "deployment_metadata.json"
_MAX_LABEL_CONFIG_BYTES = 2 * 1024 * 1024
_GENERIC_LABEL = re.compile(r"class_\d+", re.IGNORECASE)


class _Model(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], *, strict: bool) -> object: ...

    def save_pretrained(self, directory: Path, *, safe_serialization: bool) -> None: ...


class _Processor(Protocol):
    size: object

    def save_pretrained(self, directory: Path) -> None: ...


class _ModelFactory(Protocol):
    def __call__(
        self,
        source: str,
        *,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        ignore_mismatched_sizes: bool,
        local_files_only: bool,
    ) -> _Model: ...


class _ProcessorFactory(Protocol):
    def __call__(self, source: str, *, local_files_only: bool) -> _Processor: ...


@dataclass(frozen=True, slots=True)
class LabelMapping:
    id2label: dict[int, str]
    semantic_names_available: bool
    source: str

    @property
    def label2id(self) -> dict[str, int]:
        return {name: class_id for class_id, name in self.id2label.items()}


@dataclass(frozen=True, slots=True)
class PreparationResult:
    output_directory: Path
    metadata: dict[str, JsonValue]
    changed: bool


def default_output_directory(repository_root: Path) -> Path:
    configured = os.environ.get("ATLAS_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser() / "segmentation" / _DEFAULT_MODEL_NAME
    return repository_root / ".local" / "models" / _DEFAULT_MODEL_NAME


def _read_json_bytes(payload: bytes, *, code: str) -> object:
    if not payload or len(payload) > _MAX_LABEL_CONFIG_BYTES:
        raise SegmentationCheckpointError(code)
    try:
        return json.loads(payload.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SegmentationCheckpointError(code) from exc


def _read_json_file(path: Path, *, code: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise SegmentationCheckpointError(code)
    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_LABEL_CONFIG_BYTES:
            raise SegmentationCheckpointError(code)
        return _read_json_bytes(path.read_bytes(), code=code)
    except OSError as exc:
        raise SegmentationCheckpointError(code) from exc


def _normalize_mapping(raw: object, *, num_labels: int, source: str) -> LabelMapping:
    semantic_names_available = True
    names: dict[int, str] = {}
    if isinstance(raw, dict) and "labels" in raw:
        labels = raw["labels"]
        if not isinstance(labels, list):
            raise SegmentationCheckpointError("invalid_label_mapping", source)
        for class_id, item in enumerate(labels):
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise SegmentationCheckpointError("invalid_label_mapping", source)
            names[class_id] = item["name"].strip()
    elif isinstance(raw, dict) and "id2label" in raw:
        mapping = raw["id2label"]
        if not isinstance(mapping, dict):
            raise SegmentationCheckpointError("invalid_label_mapping", source)
        for raw_id, raw_name in mapping.items():
            try:
                class_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise SegmentationCheckpointError("invalid_label_mapping", source) from exc
            if not isinstance(raw_name, str):
                raise SegmentationCheckpointError("invalid_label_mapping", source)
            names[class_id] = raw_name.strip()
        marker = raw.get("semantic_label_names_available")
        if marker is not None:
            if not isinstance(marker, bool):
                raise SegmentationCheckpointError("invalid_label_mapping", source)
            semantic_names_available = marker
    elif isinstance(raw, list):
        for class_id, raw_name in enumerate(raw):
            if not isinstance(raw_name, str):
                raise SegmentationCheckpointError("invalid_label_mapping", source)
            names[class_id] = raw_name.strip()
    else:
        raise SegmentationCheckpointError("invalid_label_mapping", source)

    if set(names) != set(range(num_labels)) or any(not name for name in names.values()):
        raise SegmentationCheckpointError("label_count_mismatch", source)
    if len(set(names.values())) != num_labels:
        raise SegmentationCheckpointError("duplicate_label_names", source)
    if all(_GENERIC_LABEL.fullmatch(name) for name in names.values()):
        semantic_names_available = False
    return LabelMapping(
        id2label=dict(sorted(names.items())),
        semantic_names_available=semantic_names_available,
        source=source,
    )


def _mapping_from_zip(path: Path, *, num_labels: int) -> LabelMapping:
    if path.is_symlink() or not path.is_file():
        raise SegmentationCheckpointError("mapillary_zip_unavailable")
    try:
        with zipfile.ZipFile(path) as archive:
            matches = sorted(
                (
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and Path(info.filename.replace("\\", "/")).name.casefold() == "config_v2.0.json"
                ),
                key=lambda info: (len(info.filename), info.filename.casefold()),
            )
            if not matches:
                raise SegmentationCheckpointError("mapillary_config_missing_in_zip")
            member = matches[0]
            if member.flag_bits & 0x1 or not 0 < member.file_size <= _MAX_LABEL_CONFIG_BYTES:
                raise SegmentationCheckpointError("unsafe_mapillary_config_member")
            with archive.open(member, "r") as source:
                payload = source.read(_MAX_LABEL_CONFIG_BYTES + 1)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SegmentationCheckpointError("invalid_mapillary_zip") from exc
    raw = _read_json_bytes(payload, code="invalid_mapillary_config")
    return _normalize_mapping(raw, num_labels=num_labels, source="operator_mapillary_v2_config")


def _repository_label_candidates(repository_root: Path, output_directory: Path) -> tuple[Path, ...]:
    return (
        output_directory / "config.json",
        repository_root / "config_v2.0.json",
        repository_root / "assets" / "config_v2.0.json",
        repository_root / "config" / "mapillary" / "config_v2.0.json",
        repository_root / "services" / "api" / "assets" / "config_v2.0.json",
        repository_root / "config" / "segmentation" / "labels.json",
        repository_root
        / "services"
        / "api"
        / "src"
        / "atlaslens_api"
        / "resources"
        / "segmentation_labels.json",
    )


def resolve_label_mapping(
    *,
    repository_root: Path,
    output_directory: Path,
    num_labels: int,
    label_config: Path | None,
    mapillary_zip: Path | None,
) -> LabelMapping:
    for candidate in _repository_label_candidates(repository_root, output_directory):
        if candidate.is_file() and not candidate.is_symlink():
            raw = _read_json_file(candidate, code="invalid_label_mapping")
            try:
                relative = candidate.resolve().relative_to(repository_root.resolve())
                source = f"repository:{relative.as_posix()}"
            except ValueError:
                source = "existing_exported_model_config"
            try:
                return _normalize_mapping(
                    raw,
                    num_labels=num_labels,
                    source=source,
                )
            except SegmentationCheckpointError as exc:
                if exc.code != "label_count_mismatch":
                    raise
    if label_config is not None:
        raw = _read_json_file(label_config.expanduser(), code="label_config_unavailable")
        return _normalize_mapping(
            raw,
            num_labels=num_labels,
            source="operator_label_config",
        )
    if mapillary_zip is not None:
        return _mapping_from_zip(mapillary_zip.expanduser(), num_labels=num_labels)
    return LabelMapping(
        id2label={class_id: f"class_{class_id}" for class_id in range(num_labels)},
        semantic_names_available=False,
        source="generic_fallback",
    )


def _variant(base_model: str) -> str | None:
    match = re.search(r"segformer[-_/](b\d+)", base_model, re.IGNORECASE)
    return None if match is None else match.group(1).casefold()


def _existing_result(
    output_directory: Path,
    *,
    checkpoint: LoadedSegmentationCheckpoint,
) -> PreparationResult | None:
    if output_directory.is_symlink():
        raise SegmentationCheckpointError("unsafe_output_directory")
    if not output_directory.exists():
        return None
    if not output_directory.is_dir():
        raise SegmentationCheckpointError("output_directory_conflict")
    metadata_path = output_directory / _METADATA_FILE
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise SegmentationCheckpointError("output_directory_conflict")
    raw = _read_json_file(metadata_path, code="invalid_deployment_metadata")
    if not isinstance(raw, dict):
        raise SegmentationCheckpointError("invalid_deployment_metadata")
    expected = {
        "schema_version": "atlaslens-segmentation-deployment-v1",
        "source_checkpoint_sha256": checkpoint.inspection.sha256,
        "weight_source": checkpoint.selected_weight_source,
        "num_labels": checkpoint.inspection.inferred_num_labels,
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise SegmentationCheckpointError("output_checkpoint_mismatch")
    required = ("config.json", "preprocessor_config.json")
    if any(not (output_directory / name).is_file() for name in required):
        raise SegmentationCheckpointError("prepared_model_incomplete")
    if not tuple(output_directory.glob("*.safetensors")):
        raise SegmentationCheckpointError("prepared_model_incomplete")
    metadata: dict[str, JsonValue] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise SegmentationCheckpointError("invalid_deployment_metadata")
        if value is None or isinstance(value, bool | int | float | str):
            metadata[key] = value
        else:
            raise SegmentationCheckpointError("invalid_deployment_metadata")
    return PreparationResult(output_directory=output_directory, metadata=metadata, changed=False)


def _default_factories() -> tuple[_ModelFactory, _ProcessorFactory]:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    return (
        cast(_ModelFactory, AutoModelForSemanticSegmentation.from_pretrained),
        cast(_ProcessorFactory, AutoImageProcessor.from_pretrained),
    )


def _load_runtime_objects(
    *,
    base_source: str,
    mapping: LabelMapping,
    num_labels: int,
    model_factory: _ModelFactory,
    processor_factory: _ProcessorFactory,
) -> tuple[_Model, _Processor]:
    try:
        model = model_factory(
            base_source,
            num_labels=num_labels,
            id2label=mapping.id2label,
            label2id=mapping.label2id,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        processor = processor_factory(base_source, local_files_only=True)
    except OSError as exc:
        raise SegmentationCheckpointError(
            "base_model_unavailable_offline",
            "install the reviewed base model in the local Hugging Face cache "
            "or pass --base-model-dir",
        ) from exc
    return model, processor


def _translate_modular_segformer_key(key: str) -> str:
    """Translate the reviewed modular SegFormer naming used by the Colab trainer.

    The adapter changes names only. A later exact key-set and tensor-shape check
    is mandatory before strict loading, so unknown architecture variants fail.
    """

    stage = re.fullmatch(r"segformer\.stages\.(\d+)\.(.+)", key)
    if stage is None:
        if key.startswith("decode_head.linear_projections."):
            return key.replace("decode_head.linear_projections.", "decode_head.linear_c.", 1)
        return key
    stage_index, remainder = stage.groups()
    if remainder.startswith("patch_embeddings."):
        suffix = remainder.removeprefix("patch_embeddings.")
        return f"segformer.encoder.patch_embeddings.{stage_index}.{suffix}"
    if remainder.startswith("layer_norm."):
        suffix = remainder.removeprefix("layer_norm.")
        return f"segformer.encoder.layer_norm.{stage_index}.{suffix}"
    block = re.fullmatch(r"blocks\.(\d+)\.(.+)", remainder)
    if block is None:
        return key
    block_index, suffix = block.groups()
    replacements = (
        ("layernorm_before.", "layer_norm_1."),
        ("attention.q_proj.", "attention.self.query."),
        ("attention.k_proj.", "attention.self.key."),
        ("attention.v_proj.", "attention.self.value."),
        ("attention.o_proj.", "attention.output.dense."),
        (
            "attention.sequence_reduction.sequence_reduction.",
            "attention.self.sr.",
        ),
        ("attention.sequence_reduction.layer_norm.", "attention.self.layer_norm."),
        ("layernorm_after.", "layer_norm_2."),
        ("mlp.fc1.", "mlp.dense1."),
        ("mlp.dwconv.dwconv.", "mlp.dwconv.dwconv."),
        ("mlp.fc2.", "mlp.dense2."),
    )
    translated = suffix
    for old, new in replacements:
        if suffix.startswith(old):
            translated = new + suffix.removeprefix(old)
            break
    return f"segformer.encoder.block.{stage_index}.{block_index}.{translated}"


def _shape(value: object) -> tuple[int, ...] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError, OverflowError):
        return None


def _state_dict_for_strict_load(
    model: _Model, state_dict: Mapping[str, Any]
) -> tuple[Mapping[str, Any], str]:
    expected = model.state_dict()
    if set(state_dict) == set(expected):
        candidate: Mapping[str, Any] = state_dict
        adapter = "identity"
    else:
        translated: dict[str, Any] = {}
        for key, value in state_dict.items():
            new_key = _translate_modular_segformer_key(key)
            if new_key in translated:
                raise SegmentationCheckpointError(
                    "state_dict_key_adapter_failed", f"duplicate:{new_key[:160]}"
                )
            translated[new_key] = value
        candidate = translated
        adapter = "transformers_modular_segformer_to_legacy_v1"
    missing = sorted(set(expected) - set(candidate))
    unexpected = sorted(set(candidate) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(candidate)
        if _shape(expected[key]) != _shape(candidate[key])
    )
    if missing or unexpected or mismatched:
        detail = "; ".join(
            (
                f"missing={missing[:8]}",
                f"unexpected={unexpected[:8]}",
                f"shape_mismatch={mismatched[:8]}",
            )
        )[:800]
        raise SegmentationCheckpointError("state_dict_key_adapter_failed", detail)
    return candidate, adapter


def prepare_segmentation_model(
    checkpoint_path: Path,
    output_directory: Path,
    *,
    repository_root: Path,
    base_model_directory: Path | None = None,
    label_config: Path | None = None,
    mapillary_zip: Path | None = None,
    model_factory: _ModelFactory | None = None,
    processor_factory: _ProcessorFactory | None = None,
) -> PreparationResult:
    checkpoint = load_trusted_checkpoint(checkpoint_path)
    output = output_directory.expanduser().resolve()
    existing = _existing_result(output, checkpoint=checkpoint)
    if existing is not None:
        return existing

    inspection = checkpoint.inspection
    configured_base = inspection.training_config.get("student_checkpoint")
    base_model = (
        configured_base.strip()
        if isinstance(configured_base, str) and configured_base.strip()
        else _DEFAULT_BASE_MODEL
    )
    base_source = base_model
    if base_model_directory is not None:
        local_base = base_model_directory.expanduser()
        if local_base.is_symlink() or not local_base.is_dir():
            raise SegmentationCheckpointError("base_model_directory_unavailable")
        base_source = str(local_base.resolve())

    mapping = resolve_label_mapping(
        repository_root=repository_root,
        output_directory=output,
        num_labels=inspection.inferred_num_labels,
        label_config=label_config,
        mapillary_zip=mapillary_zip,
    )
    if (model_factory is None) != (processor_factory is None):
        raise SegmentationCheckpointError("incomplete_runtime_factory")
    if model_factory is None or processor_factory is None:
        model_factory, processor_factory = _default_factories()
    model, processor = _load_runtime_objects(
        base_source=base_source,
        mapping=mapping,
        num_labels=inspection.inferred_num_labels,
        model_factory=model_factory,
        processor_factory=processor_factory,
    )
    adapted_state_dict, state_dict_adapter = _state_dict_for_strict_load(
        model, checkpoint.selected_state_dict
    )
    try:
        model.load_state_dict(adapted_state_dict, strict=True)
    except RuntimeError as exc:
        detail = str(exc).replace("\n", " ")[:500]
        raise SegmentationCheckpointError("strict_state_dict_load_failed", detail) from exc

    image_size_value = inspection.training_config.get("image_size")
    image_size = image_size_value if isinstance(image_size_value, int) else None
    if image_size is not None:
        processor.size = {"height": image_size, "width": image_size}
    metadata: dict[str, JsonValue] = {
        "schema_version": "atlaslens-segmentation-deployment-v1",
        "model_family": inspection.inferred_model_family or "unknown",
        "variant": _variant(base_model),
        "source_checkpoint": inspection.path.name,
        "source_checkpoint_sha256": inspection.sha256,
        "source_checkpoint_size_bytes": inspection.size_bytes,
        "weight_source": checkpoint.selected_weight_source,
        "state_dict_adapter": state_dict_adapter,
        "base_model": base_model,
        "num_labels": inspection.inferred_num_labels,
        "image_size": image_size,
        "training_epoch": inspection.epoch,
        "best_miou": inspection.best_miou,
        "semantic_label_names_available": mapping.semantic_names_available,
        "label_mapping_source": mapping.source,
        "prepared_at": datetime.now(UTC).isoformat(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.preparing-", dir=output.parent
    ) as temporary_name:
        staging = Path(temporary_name)
        model.save_pretrained(staging, safe_serialization=True)
        processor.save_pretrained(staging)
        if not (staging / "config.json").is_file():
            raise SegmentationCheckpointError("prepared_model_incomplete", "config.json")
        if not (staging / "preprocessor_config.json").is_file():
            raise SegmentationCheckpointError(
                "prepared_model_incomplete", "preprocessor_config.json"
            )
        if not tuple(staging.glob("*.safetensors")) or tuple(staging.glob("*.bin")):
            raise SegmentationCheckpointError("safe_serialization_failed")
        metadata_path = staging / _METADATA_FILE
        with metadata_path.open("x", encoding="utf-8", newline="\n") as target:
            json.dump(metadata, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        try:
            os.replace(staging, output)
        except OSError as exc:
            raise SegmentationCheckpointError("atomic_model_publish_failed") from exc
    return PreparationResult(output_directory=output, metadata=metadata, changed=True)
