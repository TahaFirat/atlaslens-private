from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path
from typing import Literal, cast

from PIL import Image, UnidentifiedImageError

from atlaslens_api.retrieval.errors import ManifestValidationError
from atlaslens_api.retrieval.models import ManifestRecord

MANIFEST_COLUMNS = (
    "image_path",
    "latitude",
    "longitude",
    "country",
    "region",
    "city",
    "license",
    "source",
    "Notes",
)
EXTENDED_MANIFEST_COLUMNS = MANIFEST_COLUMNS + (
    "asset_key",
    "source_record_id",
    "source_url",
    "license_url",
    "attribution",
    "display_allowed",
    "capture_type",
    "capture_family_id",
    "coordinate_kind",
    "coordinate_uncertainty_m",
    "captured_at",
    "heading_degrees",
    "perceptual_hash",
    "terms_version",
    "continent",
    "geographic_cell",
)
_SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
_OPAQUE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PERCEPTUAL_HASH = re.compile(r"^[0-9a-f]{16}$")
_COORDINATE_KINDS = {
    "operator_provided",
    "camera_raw",
    "map_matched",
    "object",
    "manual",
    "unknown",
}


def hash_image(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coordinate(value: str, *, latitude: bool, row_number: int) -> float:
    try:
        coordinate = float(value)
    except ValueError as exc:
        raise ManifestValidationError(f"row {row_number}: malformed coordinate") from exc
    limit = 90.0 if latitude else 180.0
    if not math.isfinite(coordinate) or coordinate < -limit or coordinate > limit:
        raise ManifestValidationError(f"row {row_number}: coordinate is outside WGS84 bounds")
    return coordinate


def _optional(value: str, limit: int, row_number: int) -> str | None:
    stripped = value.strip()
    if len(stripped) > limit:
        raise ManifestValidationError(f"row {row_number}: metadata field is too long")
    return stripped or None


def _optional_float(
    value: str,
    *,
    row_number: int,
    minimum_exclusive: float | None = None,
    maximum_exclusive: float | None = None,
) -> float | None:
    if not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ManifestValidationError(f"row {row_number}: malformed numeric metadata") from exc
    if (
        not math.isfinite(parsed)
        or (minimum_exclusive is not None and parsed <= minimum_exclusive)
        or (maximum_exclusive is not None and parsed >= maximum_exclusive)
    ):
        raise ManifestValidationError(f"row {row_number}: numeric metadata is out of range")
    return parsed


def validate_manifest(manifest_path: Path, input_root: Path) -> list[ManifestRecord]:
    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise ManifestValidationError("input root is not a directory")
    try:
        handle = manifest_path.expanduser().resolve().open(
            "r", encoding="utf-8-sig", newline=""
        )
    except (OSError, UnicodeError) as exc:
        raise ManifestValidationError("manifest cannot be read as UTF-8") from exc

    records: list[ManifestRecord] = []
    try:
        with handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ManifestValidationError("manifest is empty") from exc
            if len(header) != len(set(header)):
                raise ManifestValidationError("manifest contains duplicate headers")
            columns = tuple(header)
            if columns not in {MANIFEST_COLUMNS, EXTENDED_MANIFEST_COLUMNS}:
                raise ManifestValidationError("manifest headers do not match the required schema")
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(columns):
                    raise ManifestValidationError(f"row {row_number}: incorrect field count")
                values = dict(zip(columns, row, strict=True))
                if any(len(value) > 4096 for value in row) or len(values["Notes"]) > 2000:
                    raise ManifestValidationError(f"row {row_number}: field is too long")
                relative = values["image_path"].strip()
                if not relative or len(relative) > 1024:
                    raise ManifestValidationError(f"row {row_number}: invalid image path")
                image_path = (root / relative).resolve()
                if not image_path.is_relative_to(root) or not image_path.is_file():
                    raise ManifestValidationError(
                        f"row {row_number}: image must be a regular file below input root"
                    )
                license_value = values["license"].strip()
                source = values["source"].strip()
                if not license_value or not source:
                    raise ManifestValidationError(
                        f"row {row_number}: license and source are required"
                    )
                if len(license_value) > 500 or len(source) > 500:
                    raise ManifestValidationError(f"row {row_number}: metadata field is too long")
                try:
                    with Image.open(image_path) as image:
                        if image.format not in _SUPPORTED_FORMATS:
                            raise ManifestValidationError(
                                f"row {row_number}: unsupported image format"
                            )
                        image.verify()
                except (OSError, UnidentifiedImageError) as exc:
                    raise ManifestValidationError(f"row {row_number}: invalid image") from exc
                asset_key = values.get("asset_key", "").strip() or None
                if asset_key is not None and not _OPAQUE_KEY.fullmatch(asset_key):
                    raise ManifestValidationError(f"row {row_number}: invalid opaque asset key")
                perceptual_hash = values.get("perceptual_hash", "").strip() or None
                if perceptual_hash is not None and not _PERCEPTUAL_HASH.fullmatch(
                    perceptual_hash
                ):
                    raise ManifestValidationError(f"row {row_number}: invalid perceptual hash")
                coordinate_kind = (
                    values.get("coordinate_kind", "").strip() or "operator_provided"
                )
                if coordinate_kind not in _COORDINATE_KINDS:
                    raise ManifestValidationError(f"row {row_number}: invalid coordinate kind")
                display_text = values.get("display_allowed", "false").strip().lower()
                if display_text not in {"true", "false", "1", "0"}:
                    raise ManifestValidationError(f"row {row_number}: invalid display policy")
                capture_type = values.get("capture_type", "").strip() or "user_provided"
                attribution = (
                    values.get("attribution", "").strip() or "attribution unavailable"
                )
                if len(capture_type) > 40 or len(attribution) > 500:
                    raise ManifestValidationError(f"row {row_number}: metadata field is too long")
                records.append(
                    ManifestRecord(
                        image_path=image_path,
                        latitude=_coordinate(
                            values["latitude"].strip(), latitude=True, row_number=row_number
                        ),
                        longitude=_coordinate(
                            values["longitude"].strip(),
                            latitude=False,
                            row_number=row_number,
                        ),
                        country=_optional(values["country"], 120, row_number),
                        region=_optional(values["region"], 160, row_number),
                        city=_optional(values["city"], 160, row_number),
                        license=license_value,
                        source=source,
                        content_hash=hash_image(image_path),
                        asset_key=asset_key,
                        source_record_id=_optional(
                            values.get("source_record_id", ""), 160, row_number
                        ),
                        source_url=_optional(values.get("source_url", ""), 500, row_number),
                        license_url=_optional(
                            values.get("license_url", ""), 500, row_number
                        ),
                        attribution=attribution,
                        display_allowed=display_text in {"true", "1"},
                        capture_type=capture_type,
                        capture_family_id=_optional(
                            values.get("capture_family_id", ""), 160, row_number
                        ),
                        coordinate_kind=cast(
                            Literal[
                                "operator_provided",
                                "camera_raw",
                                "map_matched",
                                "object",
                                "manual",
                                "unknown",
                            ],
                            coordinate_kind,
                        ),
                        coordinate_uncertainty_m=_optional_float(
                            values.get("coordinate_uncertainty_m", ""),
                            row_number=row_number,
                            minimum_exclusive=0,
                        ),
                        captured_at=_optional(values.get("captured_at", ""), 80, row_number),
                        heading_degrees=_optional_float(
                            values.get("heading_degrees", ""),
                            row_number=row_number,
                            minimum_exclusive=-1e-12,
                            maximum_exclusive=360,
                        ),
                        perceptual_hash=perceptual_hash,
                        terms_version=_optional(
                            values.get("terms_version", ""), 80, row_number
                        ),
                        continent=_optional(values.get("continent", ""), 40, row_number),
                        geographic_cell=_optional(
                            values.get("geographic_cell", ""), 100, row_number
                        ),
                    )
                )
    except UnicodeError as exc:
        raise ManifestValidationError("manifest cannot be read as UTF-8") from exc
    except csv.Error as exc:
        raise ManifestValidationError("manifest contains malformed CSV") from exc
    return records
