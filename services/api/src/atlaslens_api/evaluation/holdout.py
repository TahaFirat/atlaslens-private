from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HoldoutManifestError(ValueError):
    """Safe validation error for private geolocation evaluation metadata."""


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class HoldoutEvaluationRecord(_EvaluationModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    path: str = Field(min_length=1, max_length=1_024)
    country: str = Field(pattern=r"^[A-Z]{2}$")
    city: str = Field(min_length=1, max_length=160)
    split: Literal["development", "validation", "final_holdout"]
    usage: Literal["development_only", "validation_only", "holdout_only"]
    allow_reference_index: Literal[False] = False
    allow_training: Literal[False] = False
    allow_prompt_ground_truth: Literal[False] = False
    source: str | None = Field(default=None, min_length=1, max_length=160)
    source_image_id: str | None = Field(default=None, min_length=1, max_length=200)
    capture_family_id: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_usage_and_source(self) -> HoldoutEvaluationRecord:
        expected_usage = {
            "development": "development_only",
            "validation": "validation_only",
            "final_holdout": "holdout_only",
        }[self.split]
        if self.usage != expected_usage:
            raise ValueError("evaluation usage must match its split")
        if (self.source is None) != (self.source_image_id is None):
            raise ValueError("source and source_image_id must be supplied together")
        return self


class HoldoutEvaluationManifest(_EvaluationModel):
    schema_version: Literal["atlaslens-geolocation-evaluation-v1"]
    records: tuple[HoldoutEvaluationRecord, ...] = Field(
        min_length=1,
        max_length=10_000,
    )


@dataclass(frozen=True, slots=True)
class PredictionInput:
    evaluation_id: str
    image_path: Path

    def __repr__(self) -> str:
        return f"PredictionInput(evaluation_id={self.evaluation_id!r}, image_path=<redacted>)"


@dataclass(frozen=True, slots=True)
class ValidatedHoldoutRecord:
    metadata: HoldoutEvaluationRecord
    image_path: Path
    content_sha256: str
    perceptual_hash: str

    def __repr__(self) -> str:
        return (
            f"ValidatedHoldoutRecord(id={self.metadata.id!r}, "
            "image_path=<redacted>, ground_truth=<evaluation-only>)"
        )


@dataclass(frozen=True, slots=True)
class ValidatedHoldoutManifest:
    records: tuple[ValidatedHoldoutRecord, ...]
    fingerprint: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _average_hash(path: Path) -> str:
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("L").resize(
                (8, 8), Image.Resampling.LANCZOS
            )
            pixels = list(image.getdata())
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise HoldoutManifestError("holdout_image_invalid") from exc
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return f"{value:016x}"


class HoldoutManifestLoader:
    """Load truth only inside evaluation code; prediction projection contains paths only."""

    def __init__(
        self,
        *,
        max_records: int = 10_000,
        max_manifest_bytes: int = 2 * 1024 * 1024,
        max_image_bytes: int = 50 * 1024 * 1024,
        max_decoded_pixels: int = 40_000_000,
        perceptual_cross_split_distance: int = 4,
    ) -> None:
        if not 1 <= max_records <= 10_000:
            raise ValueError("max_records must be in [1, 10000]")
        if min(max_manifest_bytes, max_image_bytes, max_decoded_pixels) < 1:
            raise ValueError("manifest and image bounds must be positive")
        if not 0 <= perceptual_cross_split_distance <= 16:
            raise ValueError("perceptual distance must be in [0, 16]")
        self._max_records = max_records
        self._max_manifest_bytes = max_manifest_bytes
        self._max_image_bytes = max_image_bytes
        self._max_decoded_pixels = max_decoded_pixels
        self._perceptual_cross_split_distance = perceptual_cross_split_distance

    def _payload(self, manifest_path: Path) -> dict[str, object]:
        expanded = manifest_path.expanduser()
        if expanded.is_symlink():
            raise HoldoutManifestError("holdout_manifest_unsafe")
        try:
            resolved = expanded.resolve(strict=True)
        except OSError as exc:
            raise HoldoutManifestError("holdout_manifest_unavailable") from exc
        if resolved.is_symlink() or not resolved.is_file():
            raise HoldoutManifestError("holdout_manifest_unsafe")
        if resolved.stat().st_size > self._max_manifest_bytes:
            raise HoldoutManifestError("holdout_manifest_too_large")
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HoldoutManifestError("holdout_manifest_unreadable") from exc
        if not isinstance(payload, dict):
            raise HoldoutManifestError("holdout_manifest_invalid")
        return payload

    def _image_path(self, manifest_path: Path, value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_path.resolve().parent / candidate
        if candidate.is_symlink():
            raise HoldoutManifestError("holdout_image_unsafe")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise HoldoutManifestError("holdout_image_unavailable") from exc
        if resolved.is_symlink() or not resolved.is_file():
            raise HoldoutManifestError("holdout_image_unsafe")
        size = resolved.stat().st_size
        if size < 1 or size > self._max_image_bytes:
            raise HoldoutManifestError("holdout_image_size_invalid")
        try:
            with Image.open(resolved) as image:
                width, height = image.size
                if width < 1 or height < 1 or width * height > self._max_decoded_pixels:
                    raise HoldoutManifestError("holdout_image_dimensions_invalid")
                image.verify()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise HoldoutManifestError("holdout_image_invalid") from exc
        return resolved

    def load_prediction_inputs(self, manifest_path: Path) -> tuple[PredictionInput, ...]:
        """Project only opaque IDs and image paths before any inference process starts."""

        payload = self._payload(manifest_path)
        if payload.get("schema_version") != "atlaslens-geolocation-evaluation-v1":
            raise HoldoutManifestError("holdout_manifest_schema_invalid")
        raw_records = payload.get("records")
        if (
            not isinstance(raw_records, list)
            or not raw_records
            or len(raw_records) > self._max_records
        ):
            raise HoldoutManifestError("holdout_manifest_count_invalid")
        inputs: list[PredictionInput] = []
        seen_ids: set[str] = set()
        seen_paths: set[Path] = set()
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise HoldoutManifestError("holdout_manifest_row_invalid")
            evaluation_id = raw.get("id")
            path_value = raw.get("path")
            if not isinstance(evaluation_id, str) or not isinstance(path_value, str):
                raise HoldoutManifestError("holdout_manifest_row_invalid")
            if (
                not evaluation_id
                or len(evaluation_id) > 128
                or any(
                    character
                    not in (
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                        "0123456789._:-"
                    )
                    for character in evaluation_id
                )
            ):
                raise HoldoutManifestError("holdout_manifest_id_invalid")
            image_path = self._image_path(manifest_path, path_value)
            if evaluation_id in seen_ids or image_path in seen_paths:
                raise HoldoutManifestError("holdout_manifest_duplicate_input")
            seen_ids.add(evaluation_id)
            seen_paths.add(image_path)
            inputs.append(PredictionInput(evaluation_id=evaluation_id, image_path=image_path))
        return tuple(inputs)

    def load(self, manifest_path: Path) -> ValidatedHoldoutManifest:
        """Load and validate ground truth. Call only after prediction has completed."""

        try:
            manifest = HoldoutEvaluationManifest.model_validate(self._payload(manifest_path))
        except ValueError as exc:
            raise HoldoutManifestError("holdout_manifest_row_invalid") from exc
        if len(manifest.records) > self._max_records:
            raise HoldoutManifestError("holdout_manifest_count_invalid")

        validated: list[ValidatedHoldoutRecord] = []
        ids: set[str] = set()
        paths: set[Path] = set()
        hash_owner: dict[str, str] = {}
        source_owner: dict[tuple[str, str], str] = {}
        family_splits: defaultdict[str, set[str]] = defaultdict(set)
        perceptual_rows: list[tuple[HoldoutEvaluationRecord, int]] = []
        for record in manifest.records:
            image_path = self._image_path(manifest_path, record.path)
            if record.id in ids or image_path in paths:
                raise HoldoutManifestError("holdout_manifest_duplicate_input")
            ids.add(record.id)
            paths.add(image_path)
            content_sha256 = _sha256(image_path)
            if content_sha256 in hash_owner:
                raise HoldoutManifestError("holdout_manifest_duplicate_content")
            hash_owner[content_sha256] = record.id
            perceptual_hash = _average_hash(image_path)
            perceptual_rows.append((record, int(perceptual_hash, 16)))
            if record.source is not None and record.source_image_id is not None:
                source_key = (record.source.casefold(), record.source_image_id)
                if source_key in source_owner:
                    raise HoldoutManifestError("holdout_manifest_duplicate_source_image")
                source_owner[source_key] = record.id
            if record.capture_family_id is not None:
                family_splits[record.capture_family_id].add(record.split)
            validated.append(
                ValidatedHoldoutRecord(
                    metadata=record,
                    image_path=image_path,
                    content_sha256=content_sha256,
                    perceptual_hash=perceptual_hash,
                )
            )

        if any(len(splits) > 1 for splits in family_splits.values()):
            raise HoldoutManifestError("holdout_capture_family_crosses_splits")
        for index, (left, left_hash) in enumerate(perceptual_rows):
            for right, right_hash in perceptual_rows[index + 1 :]:
                if (
                    left.split != right.split
                    and (left_hash ^ right_hash).bit_count()
                    <= self._perceptual_cross_split_distance
                ):
                    raise HoldoutManifestError("holdout_perceptual_duplicate_crosses_splits")

        canonical = manifest.model_dump(mode="json")
        fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return ValidatedHoldoutManifest(records=tuple(validated), fingerprint=fingerprint)
