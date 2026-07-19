from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SAFE_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class SegmentationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class DeploymentMetadata(SegmentationModel):
    schema_version: Literal["atlaslens-segmentation-deployment-v1"]
    model_family: Literal["segformer"]
    variant: str = Field(min_length=1, max_length=40)
    source_checkpoint: str = Field(min_length=1, max_length=160)
    source_checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_checkpoint_size_bytes: int = Field(gt=0)
    weight_source: Literal["ema", "student"]
    state_dict_adapter: Literal["identity", "transformers_modular_segformer_to_legacy_v1"] = (
        "identity"
    )
    base_model: str = Field(min_length=1, max_length=300)
    num_labels: int = Field(gt=0, le=10_000)
    image_size: int | None = Field(default=None, gt=0, le=16_384)
    training_epoch: int | None = Field(default=None, ge=0)
    best_miou: float | None = Field(default=None, ge=0, le=1)
    semantic_label_names_available: bool
    label_mapping_source: str = Field(min_length=1, max_length=200)
    prepared_at: datetime

    @field_validator("source_checkpoint")
    @classmethod
    def checkpoint_is_a_basename(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("source checkpoint must be a basename")
        return value


class DominantClass(SegmentationModel):
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1, max_length=200)
    pixel_ratio: float = Field(ge=0, le=1)
    percentage: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def percentage_matches_ratio(self) -> DominantClass:
        if not math.isclose(
            self.percentage,
            self.pixel_ratio * 100.0,
            rel_tol=0,
            abs_tol=0.051,
        ):
            raise ValueError("dominant-class percentage does not match its pixel ratio")
        return self


class SceneTag(SegmentationModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    strength: float = Field(ge=0, le=1)
    strength_semantics: Literal["deterministic_heuristic_not_probability"] = (
        "deterministic_heuristic_not_probability"
    )
    reason: str = Field(min_length=1, max_length=240)


class SegmentationResult(SegmentationModel):
    status: Literal["completed"] = "completed"
    provider: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    device: Literal["cpu", "cuda"]
    inference_ms: int = Field(ge=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    semantic_label_names_available: bool
    dominant_classes: tuple[DominantClass, ...] = Field(max_length=32)
    scene_groups: dict[str, float] = Field(default_factory=dict, max_length=32)
    scene_tags: tuple[SceneTag, ...] = Field(default=(), max_length=16)
    warnings: tuple[str, ...] = Field(default=(), max_length=24)

    @field_validator("scene_groups")
    @classmethod
    def valid_scene_groups(cls, value: dict[str, float]) -> dict[str, float]:
        if any(
            not _SAFE_IDENTIFIER.fullmatch(name) or not math.isfinite(ratio) or not 0 <= ratio <= 1
            for name, ratio in value.items()
        ):
            raise ValueError("scene groups must have safe names and bounded finite ratios")
        return value

    @field_validator("warnings")
    @classmethod
    def unique_safe_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            not _SAFE_IDENTIFIER.fullmatch(item) for item in value
        ):
            raise ValueError("segmentation warnings must be unique safe identifiers")
        return value

    @model_validator(mode="after")
    def generic_labels_have_no_semantic_claims(self) -> SegmentationResult:
        if not self.semantic_label_names_available and (self.scene_groups or self.scene_tags):
            raise ValueError("generic labels cannot produce semantic scene groups or tags")
        return self


class SegmentationProviderStatus(SegmentationModel):
    provider_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    status: Literal[
        "ready",
        "not_installed",
        "incomplete",
        "loading",
        "failed",
        "disabled",
        "unavailable",
    ]
    enabled: bool
    installed: bool
    prepared: bool
    loaded: bool
    usable: bool
    device: Literal["cpu", "cuda"] | None = None
    weight_source: Literal["ema", "student"] | None = None
    num_labels: int | None = Field(default=None, gt=0, le=10_000)
    semantic_label_names_available: bool | None = None
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    load_error: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )

    @model_validator(mode="after")
    def status_flags_agree(self) -> SegmentationProviderStatus:
        if self.loaded and not self.prepared:
            raise ValueError("a loaded segmentation provider must be prepared")
        if self.usable and (not self.enabled or not self.prepared or self.device is None):
            raise ValueError("a usable segmentation provider must be enabled and prepared")
        return self


class SceneTagRule(SegmentationModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    reason: str = Field(min_length=1, max_length=240)
    minimums: dict[str, float] = Field(default_factory=dict, max_length=16)
    maximums: dict[str, float] = Field(default_factory=dict, max_length=16)

    @field_validator("minimums", "maximums")
    @classmethod
    def valid_thresholds(cls, value: dict[str, float]) -> dict[str, float]:
        if any(
            not _SAFE_IDENTIFIER.fullmatch(name)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
            for name, threshold in value.items()
        ):
            raise ValueError("scene-tag thresholds are invalid")
        return value

    @model_validator(mode="after")
    def has_a_condition(self) -> SceneTagRule:
        if not self.minimums and not self.maximums:
            raise ValueError("a scene-tag rule needs at least one condition")
        if set(self.minimums) & set(self.maximums):
            raise ValueError("a scene group cannot have both minimum and maximum conditions")
        return self


class SceneGroupConfiguration(SegmentationModel):
    schema_version: Literal["atlaslens-scene-groups-v1"]
    source_label_set: Literal["mapillary-vistas-v2"]
    groups: dict[str, tuple[str, ...]] = Field(min_length=1, max_length=32)
    tags: tuple[SceneTagRule, ...] = Field(default=(), max_length=16)

    @field_validator("groups")
    @classmethod
    def valid_groups(cls, value: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        if any(
            not _SAFE_IDENTIFIER.fullmatch(name)
            or not labels
            or len(labels) > 128
            or len(labels) != len(set(labels))
            or any(not label or len(label) > 200 for label in labels)
            for name, labels in value.items()
        ):
            raise ValueError("scene-group configuration is invalid")
        return value

    @model_validator(mode="after")
    def tag_groups_exist(self) -> SceneGroupConfiguration:
        known = set(self.groups)
        referenced = {
            group for tag in self.tags for group in (*tag.minimums.keys(), *tag.maximums.keys())
        }
        if not referenced <= known:
            raise ValueError("scene-tag rule references an unknown group")
        if len({tag.name for tag in self.tags}) != len(self.tags):
            raise ValueError("scene-tag names must be unique")
        return self


@dataclass(frozen=True, slots=True)
class RawSegmentationPrediction:
    image_width: int
    image_height: int
    class_pixel_counts: tuple[int, ...]
    inference_ms: int

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("segmentation dimensions must be positive")
        if self.inference_ms < 0:
            raise ValueError("segmentation inference timing must be non-negative")
        if not self.class_pixel_counts or any(
            type(count) is not int or count < 0 for count in self.class_pixel_counts
        ):
            raise ValueError("segmentation pixel counts must be non-negative integers")
        if sum(self.class_pixel_counts) != self.image_width * self.image_height:
            raise ValueError("segmentation pixel counts must cover the full image")

    def __repr__(self) -> str:
        return (
            "RawSegmentationPrediction("
            f"dimensions={self.image_width}x{self.image_height}, "
            f"classes={len(self.class_pixel_counts)}, inference_ms={self.inference_ms})"
        )
