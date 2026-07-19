from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from atlaslens_api.segmentation.models import (
    DominantClass,
    RawSegmentationPrediction,
    SceneGroupConfiguration,
    SceneTag,
    SceneTagRule,
    SegmentationResult,
)

_MAX_CONFIGURATION_BYTES = 256 * 1024
_LABEL_SEPARATOR = re.compile(r"[^\w]+", re.UNICODE)


def default_scene_group_config_path() -> Path:
    return Path(__file__).resolve().parents[5] / "config" / "segmentation" / "scene-groups-v1.json"


def normalize_label_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(part for part in _LABEL_SEPARATOR.split(normalized) if part)


def load_scene_group_config(path: Path) -> SceneGroupConfiguration:
    requested = path.expanduser()
    if requested.is_symlink() or not requested.is_file():
        raise ValueError("scene-group configuration is unavailable")
    try:
        payload = requested.read_bytes()
    except OSError as exc:
        raise ValueError("scene-group configuration is unreadable") from exc
    if not payload or len(payload) > _MAX_CONFIGURATION_BYTES:
        raise ValueError("scene-group configuration size is invalid")
    try:
        parsed = json.loads(payload)
        config = SceneGroupConfiguration.model_validate(parsed)
    except (UnicodeError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ValueError("scene-group configuration is invalid") from exc
    _validated_alias_map(config)
    return config


def _validated_alias_map(config: SceneGroupConfiguration) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for group, names in config.groups.items():
        for name in names:
            normalized = normalize_label_name(name)
            if not normalized:
                raise ValueError("scene-group label alias is empty after normalization")
            existing = aliases.get(normalized)
            if existing is not None and existing != group:
                raise ValueError("scene-group label alias belongs to multiple groups")
            aliases[normalized] = group
    return aliases


def _tag_strength(rule: SceneTagRule, ratios: Mapping[str, float]) -> float | None:
    if any(ratios.get(group, 0.0) < threshold for group, threshold in rule.minimums.items()):
        return None
    if any(ratios.get(group, 0.0) > threshold for group, threshold in rule.maximums.items()):
        return None

    components: list[float] = []
    for group, threshold in rule.minimums.items():
        ratio = ratios.get(group, 0.0)
        headroom = (ratio - threshold) / max(1e-12, 1.0 - threshold)
        components.append(0.5 + 0.5 * max(0.0, min(1.0, headroom)))
    for group, threshold in rule.maximums.items():
        ratio = ratios.get(group, 0.0)
        if threshold <= 0:
            components.append(1.0 if ratio <= 0 else 0.0)
        else:
            headroom = (threshold - ratio) / threshold
            components.append(0.5 + 0.5 * max(0.0, min(1.0, headroom)))
    if not components:
        return None
    return round(sum(components) / len(components), 4)


def aggregate_segmentation(
    prediction: RawSegmentationPrediction,
    *,
    provider_id: str,
    device: str,
    id2label: Mapping[int, str],
    semantic_label_names_available: bool,
    minimum_class_ratio: float,
    scene_group_config: SceneGroupConfiguration,
    maximum_dominant_classes: int = 20,
) -> SegmentationResult:
    if not 0 <= minimum_class_ratio <= 1:
        raise ValueError("minimum class ratio must be between zero and one")
    if not 1 <= maximum_dominant_classes <= 32:
        raise ValueError("dominant-class output bound must be between one and 32")
    if device not in {"cpu", "cuda"}:
        raise ValueError("segmentation device must be cpu or cuda")

    class_count = len(prediction.class_pixel_counts)
    expected_ids = set(range(class_count))
    if semantic_label_names_available:
        if set(id2label) != expected_ids or any(
            not isinstance(id2label[class_id], str) or not id2label[class_id].strip()
            for class_id in expected_ids
        ):
            raise ValueError("exact semantic labels must cover every classifier output")
        labels = {class_id: id2label[class_id].strip() for class_id in expected_ids}
    else:
        labels = {class_id: f"class_{class_id}" for class_id in expected_ids}

    total = prediction.image_width * prediction.image_height
    ratios = [count / total for count in prediction.class_pixel_counts]
    dominant = [
        DominantClass(
            class_id=class_id,
            class_name=labels[class_id],
            pixel_ratio=round(ratio, 6),
            percentage=round(ratio * 100.0, 2),
        )
        for class_id, ratio in sorted(enumerate(ratios), key=lambda item: (-item[1], item[0]))
        if ratio >= minimum_class_ratio and prediction.class_pixel_counts[class_id] > 0
    ][:maximum_dominant_classes]

    warnings: list[str] = []
    scene_groups: dict[str, float] = {}
    scene_tags: list[SceneTag] = []
    if semantic_label_names_available:
        aliases = _validated_alias_map(scene_group_config)
        counts_by_group = {group: 0 for group in scene_group_config.groups}
        for class_id, count in enumerate(prediction.class_pixel_counts):
            group = aliases.get(normalize_label_name(labels[class_id]))
            if group is not None:
                counts_by_group[group] += count
        scene_groups = {
            group: round(counts_by_group[group] / total, 6) for group in scene_group_config.groups
        }
        for rule in scene_group_config.tags:
            strength = _tag_strength(rule, scene_groups)
            if strength is not None:
                scene_tags.append(
                    SceneTag(
                        name=rule.name,
                        strength=strength,
                        reason=rule.reason,
                    )
                )
    else:
        warnings.append("segmentation.semantic_label_names_unavailable")

    return SegmentationResult(
        provider=provider_id,
        device=device,
        inference_ms=prediction.inference_ms,
        image_width=prediction.image_width,
        image_height=prediction.image_height,
        semantic_label_names_available=semantic_label_names_available,
        dominant_classes=tuple(dominant),
        scene_groups=scene_groups,
        scene_tags=tuple(scene_tags),
        warnings=tuple(warnings),
    )
