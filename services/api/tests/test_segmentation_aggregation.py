from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlaslens_api.segmentation import (
    RawSegmentationPrediction,
    aggregate_segmentation,
    default_scene_group_config_path,
    load_scene_group_config,
)


def prediction(*counts: int) -> RawSegmentationPrediction:
    assert sum(counts) == 100
    return RawSegmentationPrediction(
        image_width=10,
        image_height=10,
        class_pixel_counts=counts,
        inference_ms=17,
    )


def test_exact_labels_produce_pixel_ratios_reviewed_groups_and_explainable_tags() -> None:
    config = load_scene_group_config(default_scene_group_config_path())
    result = aggregate_segmentation(
        prediction(40, 30, 2, 20, 8),
        provider_id="atlaslens-segformer-b2-v4",
        device="cuda",
        id2label={
            0: "Road",
            1: "Building",
            2: "Vegetation",
            3: "Sky",
            4: "Unlabeled",
        },
        semantic_label_names_available=True,
        minimum_class_ratio=0.01,
        scene_group_config=config,
    )

    assert result.image_width == 10
    assert result.image_height == 10
    assert result.inference_ms == 17
    assert result.dominant_classes[0].class_name == "Road"
    assert result.dominant_classes[0].pixel_ratio == 0.4
    assert result.dominant_classes[0].percentage == 40.0
    assert result.scene_groups["road_surface"] == 0.4
    assert result.scene_groups["built_environment"] == 0.3
    assert result.scene_groups["vegetation"] == 0.02
    assert {tag.name for tag in result.scene_tags} == {
        "road_heavy",
        "urban",
        "vegetation_sparse",
    }
    assert all(
        tag.strength_semantics == "deterministic_heuristic_not_probability"
        for tag in result.scene_tags
    )
    serialized = result.model_dump_json().casefold()
    assert "kayseri" not in serialized
    assert "türkiye" not in serialized
    assert "country" not in serialized


def test_minimum_ratio_filters_negligible_classes_without_changing_raw_ids() -> None:
    config = load_scene_group_config(default_scene_group_config_path())
    result = aggregate_segmentation(
        prediction(98, 1, 1),
        provider_id="atlaslens-segformer-b2-v4",
        device="cpu",
        id2label={0: "Sky", 1: "Road", 2: "Building"},
        semantic_label_names_available=True,
        minimum_class_ratio=0.02,
        scene_group_config=config,
    )

    assert [(item.class_id, item.class_name) for item in result.dominant_classes] == [(0, "Sky")]
    assert result.scene_groups["road_surface"] == 0.01
    assert result.scene_groups["built_environment"] == 0.01


def test_generic_labels_never_enable_semantic_groups_or_tags() -> None:
    config = load_scene_group_config(default_scene_group_config_path())
    result = aggregate_segmentation(
        prediction(60, 40),
        provider_id="atlaslens-segformer-b2-v4",
        device="cpu",
        id2label={0: "Road", 1: "Building"},
        semantic_label_names_available=False,
        minimum_class_ratio=0,
        scene_group_config=config,
    )

    assert [item.class_name for item in result.dominant_classes] == ["class_0", "class_1"]
    assert result.scene_groups == {}
    assert result.scene_tags == ()
    assert result.warnings == ("segmentation.semantic_label_names_unavailable",)


def test_exact_label_mapping_must_cover_every_classifier_output() -> None:
    config = load_scene_group_config(default_scene_group_config_path())
    with pytest.raises(ValueError, match="cover every classifier output"):
        aggregate_segmentation(
            prediction(60, 40),
            provider_id="atlaslens-segformer-b2-v4",
            device="cpu",
            id2label={0: "Road"},
            semantic_label_names_available=True,
            minimum_class_ratio=0,
            scene_group_config=config,
        )


def test_scene_group_config_rejects_one_alias_in_multiple_groups(tmp_path: Path) -> None:
    path = tmp_path / "groups.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-scene-groups-v1",
                "source_label_set": "mapillary-vistas-v2",
                "groups": {"road_surface": ["Road"], "terrain": ["road"]},
                "tags": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple groups"):
        load_scene_group_config(path)
