from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from atlaslens_api.evaluation.models import (
    EvaluationRecord,
    ManifestValidationReport,
    ValidatedEvaluationAsset,
    ValidatedManifest,
)

_HEADERS = tuple(EvaluationRecord.model_fields)
_NEAR_DUPLICATE_HASH_DISTANCE = 4


class EvaluationManifestError(ValueError):
    pass


def _average_hash(path: Path) -> str:
    try:
        with Image.open(path) as image:
            pixels = list(
                image.convert("L")
                .resize((8, 8), Image.Resampling.LANCZOS)
                .getdata()
            )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise EvaluationManifestError("evaluation asset is not a valid image") from exc
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return f"{value:016x}"


class EvaluationManifestLoader:
    def __init__(
        self,
        *,
        allowed_licenses: frozenset[str],
        max_asset_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        if not allowed_licenses or any(not item.strip() for item in allowed_licenses):
            raise ValueError("allowed_licenses must be an explicit non-empty allowlist")
        if max_asset_bytes < 1:
            raise ValueError("max_asset_bytes must be positive")
        self._allowed_licenses = allowed_licenses
        self._max_asset_bytes = max_asset_bytes

    def load(self, manifest_path: Path, asset_root: Path) -> ValidatedManifest:
        try:
            root = asset_root.resolve(strict=True)
        except OSError as exc:
            raise EvaluationManifestError("evaluation asset root is unavailable") from exc
        if not root.is_dir():
            raise EvaluationManifestError("evaluation asset root is unavailable")
        try:
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != _HEADERS:
                    raise EvaluationManifestError("evaluation manifest headers are invalid")
                raw_rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise EvaluationManifestError("evaluation manifest is unreadable") from exc
        if not raw_rows:
            raise EvaluationManifestError("evaluation manifest is empty")

        assets: list[ValidatedEvaluationAsset] = []
        hash_owner: dict[str, str] = {}
        perceptual_rows: list[tuple[EvaluationRecord, int]] = []
        source_rows: dict[tuple[str, str], EvaluationRecord] = {}
        family_splits: defaultdict[str, set[str]] = defaultdict(set)
        cell_splits: defaultdict[str, set[str]] = defaultdict(set)
        for row_number, raw in enumerate(raw_rows, 2):
            try:
                record = EvaluationRecord.model_validate(raw)
            except ValidationError as exc:
                raise EvaluationManifestError(f"invalid evaluation row {row_number}") from exc
            if record.license not in self._allowed_licenses:
                raise EvaluationManifestError(f"unapproved license at row {row_number}")
            relative = Path(record.local_reference)
            if relative.is_absolute() or ".." in relative.parts:
                raise EvaluationManifestError(f"unsafe evaluation path at row {row_number}")
            try:
                candidate = (root / relative).resolve(strict=True)
                candidate.relative_to(root)
            except (OSError, ValueError) as exc:
                raise EvaluationManifestError(
                    f"unsafe evaluation path at row {row_number}"
                ) from exc
            if not candidate.is_file():
                raise EvaluationManifestError(f"evaluation asset is not a file at row {row_number}")
            if candidate.stat().st_size > self._max_asset_bytes:
                raise EvaluationManifestError(f"evaluation asset is oversized at row {row_number}")
            with candidate.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != record.content_sha256:
                raise EvaluationManifestError(f"content hash mismatch at row {row_number}")
            if digest in hash_owner:
                raise EvaluationManifestError("duplicate evaluation content hash")
            hash_owner[digest] = record.image_asset_key
            computed_perceptual_hash = _average_hash(candidate)
            if (
                record.perceptual_hash is not None
                and record.perceptual_hash != computed_perceptual_hash
            ):
                raise EvaluationManifestError(
                    f"perceptual hash mismatch at row {row_number}"
                )
            perceptual_rows.append((record, int(computed_perceptual_hash, 16)))
            source_key = (record.source, record.source_record_id)
            existing = source_rows.get(source_key)
            if existing is not None and existing != record:
                raise EvaluationManifestError("conflicting source record metadata")
            source_rows[source_key] = record
            family_splits[record.capture_family_id].add(record.split)
            cell_splits[record.geographic_cell].add(record.split)
            assets.append(ValidatedEvaluationAsset(record=record, path=candidate))

        if any(len(splits) > 1 for splits in family_splits.values()):
            raise EvaluationManifestError("capture family crosses evaluation splits")
        for index, (first, first_hash) in enumerate(perceptual_rows):
            for second, second_hash in perceptual_rows[index + 1 :]:
                if (
                    first.split != second.split
                    and (first_hash ^ second_hash).bit_count()
                    <= _NEAR_DUPLICATE_HASH_DISTANCE
                ):
                    raise EvaluationManifestError(
                        "near-duplicate perceptual hash crosses evaluation splits"
                    )
        warnings = tuple(
            sorted(
                f"geographic_cell_cross_split:{cell}"
                for cell, splits in cell_splits.items()
                if len(splits) > 1
            )
        )
        canonical = [asset.record.model_dump(mode="json") for asset in assets]
        fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        report = ManifestValidationReport(
            fingerprint=fingerprint,
            image_count=len(assets),
            split_counts=dict(Counter(asset.record.split for asset in assets)),
            continent_counts=dict(Counter(asset.record.continent for asset in assets)),
            country_counts=dict(Counter(asset.record.country_code for asset in assets)),
            scene_counts=dict(Counter(asset.record.scene_category for asset in assets)),
            warnings=warnings,
        )
        return ValidatedManifest(assets=tuple(assets), report=report)
