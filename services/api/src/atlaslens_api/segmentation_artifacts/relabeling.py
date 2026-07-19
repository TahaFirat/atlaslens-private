from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError
from safetensors import SafetensorError, safe_open

from atlaslens_api.segmentation.models import DeploymentMetadata
from atlaslens_api.segmentation_artifacts.checkpoint import SegmentationCheckpointError

_ARCHITECTURE = "SegformerForSemanticSegmentation"
_LABEL_COUNT = 124
_TRAINING_IMAGE_SIZE = 640
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_WEIGHT_INDEX_BYTES = 2 * 1024 * 1024
_CLASSIFIER_WEIGHT_SUFFIX = "decode_head.classifier.weight"
_CLASSIFIER_BIAS_SUFFIX = "decode_head.classifier.bias"
_LABEL_CONFIG_CANDIDATES = (
    Path("assets/mapillary/segformer_mapillary_config.json"),
    Path("assets/mapillary/config.json"),
    Path("config.json"),
)


@dataclass(frozen=True, slots=True)
class MapillaryLabelMapping:
    id2label: dict[int, str]
    label2id: dict[str, int]
    source: str


@dataclass(frozen=True, slots=True)
class LabelRestorationResult:
    model_directory: Path
    label_config: Path
    label_mapping_source: str
    classifier_output_count: int
    training_image_size: int
    weights_sha256: dict[str, str]
    changed: bool


def _read_json(path: Path, *, code: str, maximum_bytes: int = _MAX_JSON_BYTES) -> object:
    try:
        if path.is_symlink() or not path.is_file():
            raise SegmentationCheckpointError(code)
        size = path.stat().st_size
        if not 0 < size <= maximum_bytes:
            raise SegmentationCheckpointError(code)
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except SegmentationCheckpointError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SegmentationCheckpointError(code) from exc


def _repository_root(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise SegmentationCheckpointError("repository_root_unavailable")
    try:
        return expanded.resolve(strict=True)
    except OSError as exc:
        raise SegmentationCheckpointError("repository_root_unavailable") from exc


def resolve_mapillary_label_config(
    repository_root: Path,
    *,
    label_config: Path | None = None,
) -> tuple[Path, str]:
    root = _repository_root(repository_root)
    candidates = (label_config,) if label_config is not None else _LABEL_CONFIG_CANDIDATES
    for relative_or_absolute in candidates:
        assert relative_or_absolute is not None
        candidate = (
            relative_or_absolute.expanduser()
            if relative_or_absolute.is_absolute()
            else root / relative_or_absolute
        )
        try:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise SegmentationCheckpointError("mapillary_label_config_outside_repository") from exc
        except OSError:
            continue
        return resolved, relative.as_posix()
    raise SegmentationCheckpointError("mapillary_label_config_unavailable")


def load_mapillary_label_mapping(path: Path, *, source: str) -> MapillaryLabelMapping:
    raw = _read_json(path, code="invalid_mapillary_label_config")
    if not isinstance(raw, dict):
        raise SegmentationCheckpointError("invalid_mapillary_label_config")
    architectures = raw.get("architectures")
    if architectures != [_ARCHITECTURE]:
        raise SegmentationCheckpointError("incompatible_mapillary_architecture")

    raw_id2label = raw.get("id2label")
    raw_label2id = raw.get("label2id")
    expected_ids = {str(class_id) for class_id in range(_LABEL_COUNT)}
    if (
        not isinstance(raw_id2label, dict)
        or set(raw_id2label) != expected_ids
        or not isinstance(raw_label2id, dict)
        or len(raw_label2id) != _LABEL_COUNT
    ):
        raise SegmentationCheckpointError("invalid_mapillary_label_mapping")

    id2label: dict[int, str] = {}
    for class_id in range(_LABEL_COUNT):
        name = raw_id2label[str(class_id)]
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 200:
            raise SegmentationCheckpointError("invalid_mapillary_label_mapping")
        id2label[class_id] = name.strip()
    if len(set(id2label.values())) != _LABEL_COUNT:
        raise SegmentationCheckpointError("invalid_mapillary_label_mapping")

    expected_inverse = {name: class_id for class_id, name in id2label.items()}
    for name, class_id in raw_label2id.items():
        if (
            not isinstance(name, str)
            or type(class_id) is not int
            or expected_inverse.get(name) != class_id
        ):
            raise SegmentationCheckpointError("mapillary_label_mapping_not_inverse")
    if set(raw_label2id) != set(expected_inverse):
        raise SegmentationCheckpointError("mapillary_label_mapping_not_inverse")
    return MapillaryLabelMapping(
        id2label=id2label,
        label2id=expected_inverse,
        source=source,
    )


def _weight_files(directory: Path) -> tuple[Path, ...]:
    single = directory / "model.safetensors"
    if single.is_file() and not single.is_symlink():
        return (single,)
    index = directory / "model.safetensors.index.json"
    raw = _read_json(
        index,
        code="prepared_safetensors_unavailable",
        maximum_bytes=_MAX_WEIGHT_INDEX_BYTES,
    )
    weight_map = raw.get("weight_map") if isinstance(raw, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise SegmentationCheckpointError("prepared_safetensors_unavailable")
    filenames = set(weight_map.values())
    if any(
        not isinstance(filename, str)
        or PurePosixPath(filename).name != filename
        or "\\" in filename
        or not filename.endswith(".safetensors")
        for filename in filenames
    ):
        raise SegmentationCheckpointError("prepared_safetensors_unavailable")
    files = tuple(sorted((directory / str(filename) for filename in filenames), key=str))
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise SegmentationCheckpointError("prepared_safetensors_unavailable")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise SegmentationCheckpointError("prepared_safetensors_unreadable") from exc
    return digest.hexdigest()


def _weight_receipts(files: tuple[Path, ...]) -> dict[str, str]:
    return {path.name: _sha256(path) for path in files}


def _classifier_output_count(files: tuple[Path, ...]) -> int:
    weight_shapes: list[tuple[str, tuple[int, ...]]] = []
    bias_shapes: list[tuple[str, tuple[int, ...]]] = []
    try:
        for path in files:
            with safe_open(  # type: ignore[no-untyped-call]
                path, framework="pt", device="cpu"
            ) as tensors:
                for key in tensors.keys():  # noqa: SIM118 - safe_open is not iterable
                    if key.endswith(_CLASSIFIER_WEIGHT_SUFFIX):
                        weight_shapes.append((key, tuple(tensors.get_slice(key).get_shape())))
                    elif key.endswith(_CLASSIFIER_BIAS_SUFFIX):
                        bias_shapes.append((key, tuple(tensors.get_slice(key).get_shape())))
    except (OSError, RuntimeError, SafetensorError) as exc:
        raise SegmentationCheckpointError("prepared_safetensors_invalid") from exc
    if len(weight_shapes) != 1 or len(bias_shapes) != 1:
        raise SegmentationCheckpointError("prepared_classifier_keys_ambiguous")
    weight_key, weight_shape = weight_shapes[0]
    bias_key, bias_shape = bias_shapes[0]
    if (
        not weight_shape
        or len(bias_shape) != 1
        or weight_shape[0] != bias_shape[0]
        or weight_key[: -len("weight")] != bias_key[: -len("bias")]
    ):
        raise SegmentationCheckpointError("prepared_classifier_dimensions_inconsistent")
    return weight_shape[0]


def _validated_prepared_files(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[Path, ...]]:
    model_config = _read_json(directory / "config.json", code="prepared_model_config_invalid")
    processor_config = _read_json(
        directory / "preprocessor_config.json",
        code="prepared_processor_config_invalid",
    )
    raw_metadata = _read_json(
        directory / "deployment_metadata.json",
        code="invalid_deployment_metadata",
    )
    if not all(isinstance(item, dict) for item in (model_config, processor_config, raw_metadata)):
        raise SegmentationCheckpointError("prepared_model_config_invalid")
    assert isinstance(model_config, dict)
    assert isinstance(processor_config, dict)
    assert isinstance(raw_metadata, dict)
    if model_config.get("model_type") != "segformer" or model_config.get("architectures") != [
        _ARCHITECTURE
    ]:
        raise SegmentationCheckpointError("prepared_model_architecture_mismatch")
    try:
        metadata = DeploymentMetadata.model_validate(raw_metadata)
    except ValidationError as exc:
        raise SegmentationCheckpointError("invalid_deployment_metadata") from exc
    if metadata.num_labels != _LABEL_COUNT:
        raise SegmentationCheckpointError("prepared_num_labels_mismatch")
    if metadata.image_size != _TRAINING_IMAGE_SIZE:
        raise SegmentationCheckpointError("training_image_size_mismatch")
    size = processor_config.get("size")
    if not isinstance(size, dict) or size != {
        "height": _TRAINING_IMAGE_SIZE,
        "width": _TRAINING_IMAGE_SIZE,
    }:
        raise SegmentationCheckpointError("training_image_size_mismatch")
    return model_config, processor_config, raw_metadata, _weight_files(directory)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _stage_bytes(path: Path, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _restore_bytes(path: Path, payload: bytes) -> None:
    staged = _stage_bytes(path, payload)
    os.replace(staged, path)


def restore_mapillary_labels(
    model_directory: Path,
    *,
    repository_root: Path,
    label_config: Path | None = None,
) -> LabelRestorationResult:
    requested = model_directory.expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise SegmentationCheckpointError("prepared_model_unavailable")
    try:
        directory = requested.resolve(strict=True)
    except OSError as exc:
        raise SegmentationCheckpointError("prepared_model_unavailable") from exc
    config_path, source = resolve_mapillary_label_config(
        repository_root,
        label_config=label_config,
    )
    mapping = load_mapillary_label_mapping(config_path, source=source)
    model_config, _processor_config, metadata, weight_files = _validated_prepared_files(directory)
    classifier_output_count = _classifier_output_count(weight_files)
    if classifier_output_count != _LABEL_COUNT:
        raise SegmentationCheckpointError("prepared_classifier_output_mismatch")
    weight_receipts = _weight_receipts(weight_files)

    updated_config = dict(model_config)
    updated_config["id2label"] = {
        str(class_id): mapping.id2label[class_id] for class_id in range(_LABEL_COUNT)
    }
    updated_config["label2id"] = mapping.label2id
    updated_metadata = dict(metadata)
    updated_metadata.update(
        {
            "semantic_label_names_available": True,
            "label_mapping_source": mapping.source,
            "num_labels": _LABEL_COUNT,
        }
    )
    try:
        DeploymentMetadata.model_validate(updated_metadata)
    except ValidationError as exc:
        raise SegmentationCheckpointError("invalid_updated_deployment_metadata") from exc

    changed = updated_config != model_config or updated_metadata != metadata
    if changed:
        model_config_path = directory / "config.json"
        metadata_path = directory / "deployment_metadata.json"
        try:
            original_config = model_config_path.read_bytes()
            original_metadata = metadata_path.read_bytes()
            staged_config = _stage_bytes(model_config_path, _json_bytes(updated_config))
            staged_metadata = _stage_bytes(metadata_path, _json_bytes(updated_metadata))
            os.replace(staged_config, model_config_path)
            try:
                os.replace(staged_metadata, metadata_path)
            except OSError:
                _restore_bytes(model_config_path, original_config)
                raise
        except OSError as exc:
            raise SegmentationCheckpointError("atomic_label_restoration_failed") from exc
        finally:
            if "staged_config" in locals():
                staged_config.unlink(missing_ok=True)
            if "staged_metadata" in locals():
                staged_metadata.unlink(missing_ok=True)

    if _weight_receipts(weight_files) != weight_receipts:
        if changed:
            try:
                _restore_bytes(directory / "config.json", original_config)
                _restore_bytes(directory / "deployment_metadata.json", original_metadata)
            except OSError as exc:
                raise SegmentationCheckpointError("label_restoration_rollback_failed") from exc
        raise SegmentationCheckpointError("weights_changed_during_label_restoration")
    return LabelRestorationResult(
        model_directory=directory,
        label_config=config_path,
        label_mapping_source=mapping.source,
        classifier_output_count=classifier_output_count,
        training_image_size=_TRAINING_IMAGE_SIZE,
        weights_sha256=weight_receipts,
        changed=changed,
    )
