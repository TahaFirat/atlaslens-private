from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from atlaslens_api.segmentation import (
    RawSegmentationPrediction,
    aggregate_segmentation,
    default_scene_group_config_path,
    load_scene_group_config,
)
from atlaslens_api.segmentation_artifacts import (
    SegmentationCheckpointError,
    load_mapillary_label_mapping,
    resolve_mapillary_label_config,
    restore_mapillary_labels,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REVIEWED_LABEL_CONFIG = REPOSITORY_ROOT / "assets" / "mapillary" / "segformer_mapillary_config.json"
EXPECTED_LABELS = {
    21: "Road",
    27: "Building",
    59: "Mountain",
    61: "Sky",
    64: "Vegetation",
    78: "Signage - Advertisement",
    83: "Signage - Store",
    88: "Utility Pole",
    99: "Traffic Sign - Direction (Front)",
    108: "Car",
}


def _reviewed_payload() -> dict[str, Any]:
    payload = json.loads(REVIEWED_LABEL_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _repository(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:
    root = tmp_path / "repository"
    target = root / "assets" / "mapillary" / "segformer_mapillary_config.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(payload or _reviewed_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def _prepared_model(tmp_path: Path, *, classifier_outputs: int = 124) -> Path:
    model = tmp_path / "model"
    model.mkdir(parents=True)
    generic = {str(class_id): f"class_{class_id}" for class_id in range(124)}
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["SegformerForSemanticSegmentation"],
                "id2label": generic,
                "image_size": 777,
                "label2id": {name: int(class_id) for class_id, name in generic.items()},
                "model_type": "segformer",
                "torch_dtype": "float16",
                "transformers_version": "preserve-existing-runtime-version",
            }
        ),
        encoding="utf-8",
    )
    (model / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "do_resize": True,
                "size": {"height": 640, "width": 640},
            }
        ),
        encoding="utf-8",
    )
    (model / "deployment_metadata.json").write_text(
        json.dumps(
            {
                "base_model": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
                "best_miou": 0.24783799030684367,
                "image_size": 640,
                "label_mapping_source": "generic_fallback",
                "model_family": "segformer",
                "num_labels": 124,
                "prepared_at": "2026-07-14T12:11:09+00:00",
                "schema_version": "atlaslens-segmentation-deployment-v1",
                "semantic_label_names_available": False,
                "source_checkpoint": "last_checkpoint.pt",
                "source_checkpoint_sha256": "a" * 64,
                "source_checkpoint_size_bytes": 439_721_578,
                "state_dict_adapter": "transformers_modular_segformer_to_legacy_v1",
                "training_epoch": 12,
                "variant": "b2",
                "weight_source": "ema",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "decode_head.classifier.bias": torch.zeros(classifier_outputs),
            "decode_head.classifier.weight": torch.zeros(classifier_outputs, 1, 1, 1),
            "segformer.encoder.patch_embeddings.0.proj.weight": torch.zeros(1, 1, 1, 1),
        },
        model / "model.safetensors",
    )
    return model


def test_reviewed_mapillary_mapping_has_exact_inverse_and_required_class_names() -> None:
    mapping = load_mapillary_label_mapping(
        REVIEWED_LABEL_CONFIG,
        source="assets/mapillary/segformer_mapillary_config.json",
    )

    assert len(mapping.id2label) == 124
    assert set(mapping.id2label) == set(range(124))
    assert mapping.label2id == {name: class_id for class_id, name in mapping.id2label.items()}
    for class_id, expected_name in EXPECTED_LABELS.items():
        assert mapping.id2label[class_id] == expected_name
        assert mapping.label2id[expected_name] == class_id


def test_restoration_changes_only_labels_and_metadata_and_preserves_safe_weights(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    model = _prepared_model(tmp_path)
    weights = model / "model.safetensors"
    before_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    before_mtime = weights.stat().st_mtime_ns

    result = restore_mapillary_labels(model, repository_root=repository)
    second = restore_mapillary_labels(model, repository_root=repository)

    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    processor = json.loads((model / "preprocessor_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((model / "deployment_metadata.json").read_text(encoding="utf-8"))
    assert result.changed is True
    assert second.changed is False
    assert result.classifier_output_count == 124
    assert result.training_image_size == 640
    assert result.label_mapping_source == "assets/mapillary/segformer_mapillary_config.json"
    assert metadata["semantic_label_names_available"] is True
    assert metadata["label_mapping_source"] == "assets/mapillary/segformer_mapillary_config.json"
    assert metadata["num_labels"] == 124
    assert metadata["image_size"] == 640
    assert processor["size"] == {"height": 640, "width": 640}
    assert config["image_size"] == 777
    assert config["transformers_version"] == "preserve-existing-runtime-version"
    assert config["torch_dtype"] == "float16"
    assert "dtype" not in config
    assert config["id2label"]["21"] == "Road"
    assert config["label2id"]["Road"] == 21
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == before_hash
    assert weights.stat().st_mtime_ns == before_mtime
    assert result.weights_sha256 == {"model.safetensors": before_hash}


def test_incorrect_inverse_mapping_is_rejected_before_model_changes(tmp_path: Path) -> None:
    payload = _reviewed_payload()
    payload["label2id"]["Road"] = 22
    repository = _repository(tmp_path, payload)
    model = _prepared_model(tmp_path)
    before = (model / "config.json").read_bytes()

    with pytest.raises(SegmentationCheckpointError, match="mapillary_label_mapping_not_inverse"):
        restore_mapillary_labels(model, repository_root=repository)

    assert (model / "config.json").read_bytes() == before


def test_incompatible_architecture_and_classifier_output_are_rejected(tmp_path: Path) -> None:
    payload = _reviewed_payload()
    payload["architectures"] = ["RemoteCustomModel"]
    repository = _repository(tmp_path, payload)
    model = _prepared_model(tmp_path)
    with pytest.raises(SegmentationCheckpointError, match="incompatible_mapillary_architecture"):
        restore_mapillary_labels(model, repository_root=repository)

    valid_repository = _repository(tmp_path / "valid")
    wrong_classifier = _prepared_model(tmp_path / "wrong", classifier_outputs=123)
    with pytest.raises(SegmentationCheckpointError, match="prepared_classifier_output_mismatch"):
        restore_mapillary_labels(wrong_classifier, repository_root=valid_repository)


def test_label_config_fallback_locations_are_checked_in_order(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    fallback = root / "assets" / "mapillary" / "config.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_text(json.dumps(_reviewed_payload()), encoding="utf-8")
    root_fallback = root / "config.json"
    root_fallback.write_text(json.dumps(_reviewed_payload()), encoding="utf-8")

    selected, source = resolve_mapillary_label_config(root)
    assert selected == fallback.resolve()
    assert source == "assets/mapillary/config.json"

    fallback.unlink()
    selected, source = resolve_mapillary_label_config(root)
    assert selected == root_fallback.resolve()
    assert source == "config.json"


def test_restored_names_drive_only_reviewed_scene_groups() -> None:
    mapping = load_mapillary_label_mapping(
        REVIEWED_LABEL_CONFIG,
        source="assets/mapillary/segformer_mapillary_config.json",
    )
    groups = load_scene_group_config(default_scene_group_config_path())
    configured_labels = {name for names in groups.groups.values() for name in names}
    assert configured_labels <= set(mapping.label2id)
    assert set(groups.groups) == {
        "road_surface",
        "built_environment",
        "vegetation",
        "terrain",
        "sky",
        "water",
        "vehicles",
        "people",
        "traffic_signs",
        "traffic_lights",
        "road_markings",
        "street_infrastructure",
    }

    counts = [0] * 124
    counts[21] = 40
    counts[27] = 25
    counts[61] = 15
    counts[64] = 20
    result = aggregate_segmentation(
        RawSegmentationPrediction(
            image_width=10,
            image_height=10,
            class_pixel_counts=tuple(counts),
            inference_ms=1,
        ),
        provider_id="atlaslens-segformer-b2-v4",
        device="cpu",
        id2label=mapping.id2label,
        semantic_label_names_available=True,
        minimum_class_ratio=0,
        scene_group_config=groups,
    )

    assert [item.class_name for item in result.dominant_classes[:4]] == [
        "Road",
        "Building",
        "Vegetation",
        "Sky",
    ]
    assert result.scene_groups["road_surface"] == 0.4
    assert result.scene_groups["built_environment"] == 0.25
    assert result.scene_groups["vegetation"] == 0.2
    assert result.scene_groups["sky"] == 0.15
    assert {tag.name for tag in result.scene_tags} >= {"urban", "road_heavy"}
