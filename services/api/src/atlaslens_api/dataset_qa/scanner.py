from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

from atlaslens_api.dataset_qa.models import (
    DatasetQAError,
    DatasetQAPolicy,
    DatasetQARun,
    utc_now,
)
from atlaslens_api.schemas import (
    DatasetQAIssueView,
    DatasetQAReport,
    DatasetQAReportSummary,
)

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_MASK_SUFFIXES = frozenset({".png", ".tif", ".tiff", ".bmp"})
_SAFE_KEY = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


@dataclass(frozen=True, slots=True)
class _Asset:
    key: str
    image_path: Path
    mask_path: Path | None
    latitude: float | None
    longitude: float | None
    split: str | None
    capture_family: str | None
    sequence: str | None
    labels: tuple[str | None, ...]
    country: str | None


@dataclass(frozen=True, slots=True)
class _Scanned:
    asset: _Asset
    content_sha256: str
    perceptual_hash: str
    width: int
    height: int
    image_format: str
    mask_sha256: str | None
    mask_classes: tuple[int, ...]


class _Issues:
    def __init__(self, limit: int) -> None:
        self.items: list[DatasetQAIssueView] = []
        self.errors = 0
        self.warnings = 0
        self._limit = limit
        self.truncated = False

    def add(
        self,
        code: str,
        severity: str,
        asset_key: str,
        *,
        field: str | None = None,
        metrics: dict[str, str] | None = None,
    ) -> None:
        if severity == "error":
            self.errors += 1
        elif severity == "warning":
            self.warnings += 1
        if len(self.items) >= self._limit:
            self.truncated = True
            return
        self.items.append(
            DatasetQAIssueView(
                code=code,
                severity=severity,
                asset_key=asset_key,
                field=field,
                message_key=f"dataset_qa.{code}",
                safe_metrics=metrics or {},
            )
        )


def _root(path: Path, code: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise DatasetQAError(code)
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise DatasetQAError(code) from exc
    if not resolved.is_dir():
        raise DatasetQAError(code)
    return resolved


def _contained(base: Path, relative_value: str, code: str) -> Path:
    relative = PurePosixPath(relative_value.replace("\\", "/"))
    if not relative_value.strip() or relative.is_absolute() or ".." in relative.parts:
        raise DatasetQAError(code)
    candidate = base.joinpath(*relative.parts)
    cursor = base
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise DatasetQAError(code)
    try:
        candidate.resolve(strict=False).relative_to(base)
    except (OSError, ValueError) as exc:
        raise DatasetQAError(code) from exc
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque_key(value: str) -> str:
    stripped = value.strip()
    if stripped and _SAFE_KEY.fullmatch(stripped):
        return stripped
    return "asset-" + hashlib.sha256(value.encode()).hexdigest()[:24]


def _derived_key(value: str) -> str:
    return "asset-" + hashlib.sha256(value.encode()).hexdigest()[:24]


def _coordinate(
    latitude_value: str | None,
    longitude_value: str | None,
    key: str,
    issues: _Issues,
) -> tuple[float | None, float | None]:
    if not (latitude_value or "").strip() and not (longitude_value or "").strip():
        return None, None
    if not (latitude_value or "").strip() or not (longitude_value or "").strip():
        issues.add("gps.incomplete", "error", key, field="coordinates")
        return None, None
    try:
        latitude = float(str(latitude_value).strip())
        longitude = float(str(longitude_value).strip())
    except ValueError:
        issues.add("gps.malformed", "error", key, field="coordinates")
        return None, None
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        issues.add("gps.non_finite", "error", key, field="coordinates")
        return None, None
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        if -180 <= latitude <= 180 and -90 <= longitude <= 90:
            issues.add("gps.possibly_swapped", "warning", key, field="coordinates")
        issues.add("gps.out_of_range", "error", key, field="coordinates")
        return None, None
    if latitude == 0 and longitude == 0:
        issues.add("gps.zero_zero", "warning", key, field="coordinates")
    return latitude, longitude


def _read_manifest(
    manifest: Path,
    *,
    image_root: Path,
    asset_root: Path,
    mask_root: Path | None,
    policy: DatasetQAPolicy,
    issues: _Issues,
) -> list[_Asset]:
    if manifest.is_symlink() or not manifest.is_file():
        raise DatasetQAError("qa_manifest_unavailable")
    if manifest.stat().st_size > policy.max_manifest_bytes:
        raise DatasetQAError("qa_manifest_too_large")
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            headers = tuple(reader.fieldnames or ())
            if not headers or len(headers) != len(set(headers)):
                raise DatasetQAError("qa_manifest_headers_invalid")
            image_field = "image_path" if "image_path" in headers else "local_reference"
            if image_field not in headers:
                raise DatasetQAError("qa_manifest_image_field_missing")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetQAError("qa_manifest_unreadable") from exc
    if len(rows) > policy.max_records:
        raise DatasetQAError("qa_record_limit_exceeded")

    assets: list[_Asset] = []
    for row_number, row in enumerate(rows, 2):
        if any(len(value or "") > 4096 for value in row.values()):
            raise DatasetQAError("qa_manifest_field_too_long")
        relative = (row.get(image_field) or "").strip()
        declared_key = row.get("asset_key") or row.get("image_asset_key") or ""
        key = _opaque_key(str(declared_key)) if declared_key.strip() else _derived_key(relative)
        image_path = _contained(asset_root, relative, "qa_image_path_unsafe")
        try:
            image_path.resolve(strict=False).relative_to(image_root)
        except (OSError, ValueError) as exc:
            raise DatasetQAError("qa_image_outside_images_root") from exc
        mask_value = (row.get("mask_path") or row.get("segmentation_mask") or "").strip()
        mask_path: Path | None = None
        if mask_value:
            mask_path = _contained(mask_root or asset_root, mask_value, "qa_mask_path_unsafe")
        elif mask_root is not None:
            image_relative = image_path.resolve(strict=False).relative_to(image_root)
            mask_path = _contained(
                mask_root,
                image_relative.with_suffix(".png").as_posix(),
                "qa_mask_path_unsafe",
            )
        latitude, longitude = _coordinate(
            row.get("latitude") or row.get("true_latitude"),
            row.get("longitude") or row.get("true_longitude"),
            key,
            issues,
        )
        labels = (
            (row.get("class_id") or "").strip() or None,
            (row.get("label") or "").strip() or None,
            (row.get("country_code") or "").strip() or None,
            (row.get("region") or "").strip() or None,
            (row.get("city") or row.get("city_or_area") or "").strip() or None,
            str(latitude) if latitude is not None else None,
            str(longitude) if longitude is not None else None,
        )
        assets.append(
            _Asset(
                key=key,
                image_path=image_path,
                mask_path=mask_path,
                latitude=latitude,
                longitude=longitude,
                split=(row.get("split") or "").strip() or None,
                capture_family=(row.get("capture_family_id") or "").strip() or None,
                sequence=(row.get("sequence_id") or row.get("sequence") or "").strip() or None,
                labels=labels,
                country=(row.get("country_code") or row.get("country") or "").strip() or None,
            )
        )
        if not relative:
            issues.add("manifest.image_reference_missing", "error", key, field=image_field)
        if row_number > policy.max_records + 1:
            raise DatasetQAError("qa_record_limit_exceeded")
    return assets


def _discover_images(root: Path, policy: DatasetQAPolicy) -> list[_Asset]:
    selected: list[Path] = []
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            if (directory_path / name).is_symlink():
                raise DatasetQAError("qa_images_symlink_rejected")
        for name in files:
            path = directory_path / name
            if path.is_symlink():
                raise DatasetQAError("qa_images_symlink_rejected")
            if path.suffix.casefold() in _IMAGE_SUFFIXES:
                selected.append(path)
    if len(selected) > policy.max_records:
        raise DatasetQAError("qa_record_limit_exceeded")
    return [
        _Asset(
            key=_derived_key(path.relative_to(root).as_posix()),
            image_path=path,
            mask_path=None,
            latitude=None,
            longitude=None,
            split=None,
            capture_family=None,
            sequence=None,
            labels=(None,) * 7,
            country=None,
        )
        for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix())
    ]


def _gps_from_exif(exif: Any) -> tuple[float, float] | None:
    try:
        gps = exif.get_ifd(0x8825) if 0x8825 in exif else None
        if not gps:
            return None
        lat_values, lon_values = gps.get(2), gps.get(4)
        lat_ref, lon_ref = str(gps.get(1, "N")), str(gps.get(3, "E"))
        if not lat_values or not lon_values:
            return None

        def decimal(values: Any) -> float:
            return float(values[0]) + float(values[1]) / 60 + float(values[2]) / 3600

        latitude = decimal(lat_values) * (-1 if lat_ref.upper().startswith("S") else 1)
        longitude = decimal(lon_values) * (-1 if lon_ref.upper().startswith("W") else 1)
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            return None
        return latitude, longitude
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lat2 = math.radians(left[0]), math.radians(right[0])
    dlat = lat2 - lat1
    dlon = math.radians(right[1] - left[1])
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.atan2(math.sqrt(value), math.sqrt(max(0, 1 - value)))


def _distance_bucket(distance: float) -> str:
    if distance < 1:
        return "under_1_km"
    if distance < 10:
        return "1_to_10_km"
    if distance < 100:
        return "10_to_100_km"
    return "100_km_or_more"


def _scan_mask(
    asset: _Asset,
    image_size: tuple[int, int],
    allowed_class_ids: frozenset[int],
    policy: DatasetQAPolicy,
    issues: _Issues,
    class_pixels: Counter[str],
) -> tuple[str | None, tuple[int, ...], bool]:
    path = asset.mask_path
    if path is None:
        return None, (), False
    if path.is_symlink():
        raise DatasetQAError("qa_mask_path_unsafe")
    if not path.is_file():
        issues.add("mask.missing", "error", asset.key, field="mask")
        return None, (), False
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        issues.add("mask.empty_file", "error", asset.key, field="mask")
        return _sha256(path), (), True
    if size_bytes > policy.max_file_bytes:
        issues.add("mask.file_too_large", "error", asset.key, field="mask")
        return _sha256(path), (), True
    digest = _sha256(path)
    try:
        with Image.open(path) as source:
            width, height = source.size
            if (
                width < 1
                or height < 1
                or width > policy.max_side
                or height > policy.max_side
                or width * height > policy.max_decoded_pixels
            ):
                issues.add("mask.dimensions_invalid", "error", asset.key, field="mask")
                return digest, (), True
            if (width, height) != image_size:
                issues.add("mask.dimension_mismatch", "error", asset.key, field="mask")
            if source.mode == "P":
                palette = source.getpalette()
                if palette is None or len(palette) < 3 or len(palette) % 3:
                    issues.add("mask.palette_invalid", "error", asset.key, field="mask")
            elif source.mode not in {"1", "L", "I", "I;16"}:
                issues.add("mask.class_representation_invalid", "error", asset.key, field="mask")
                return digest, (), True
            source.load()
            values = np.asarray(source, dtype=np.int64)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        del exc
        issues.add("mask.corrupt", "error", asset.key, field="mask")
        return digest, (), True
    if values.ndim != 2 or values.size == 0:
        issues.add("mask.corrupt", "error", asset.key, field="mask")
        return digest, (), True
    unique, counts = np.unique(values, return_counts=True)
    classes = tuple(int(item) for item in unique)
    if any(item < 0 or item > 65_535 for item in classes):
        issues.add("mask.class_id_out_of_range", "error", asset.key, field="class_id")
    invalid = sorted(set(classes) - allowed_class_ids) if allowed_class_ids else []
    if invalid:
        issues.add(
            "mask.invalid_class_id",
            "error",
            asset.key,
            field="class_id",
            metrics={"invalid_class_count": str(len(invalid))},
        )
    for class_id, count in zip(classes, counts, strict=True):
        class_pixels[str(class_id)] += int(count)
    foreground = values != 0
    foreground_count = int(np.count_nonzero(foreground))
    coverage = foreground_count / int(values.size)
    if foreground_count == 0:
        issues.add("mask.all_background", "warning", asset.key, field="mask")
        issues.add("mask.empty_foreground", "warning", asset.key, field="mask")
    elif coverage < policy.minimum_foreground_coverage:
        issues.add("mask.coverage_too_low", "warning", asset.key, field="mask")
    elif coverage > policy.maximum_foreground_coverage:
        issues.add("mask.coverage_too_high", "warning", asset.key, field="mask")
    tiny = 0
    for class_id in classes:
        if class_id == 0:
            continue
        components, _, stats, _ = cv2.connectedComponentsWithStats(
            np.asarray(values == class_id, dtype=np.uint8), connectivity=8
        )
        if components > 1:
            tiny += int(
                np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] < policy.tiny_component_pixels)
            )
    if tiny:
        issues.add(
            "mask.tiny_components",
            "warning",
            asset.key,
            field="mask",
            metrics={"component_count": str(tiny)},
        )
    return digest, classes, True


def _scan_image(
    asset: _Asset,
    allowed_class_ids: frozenset[int],
    policy: DatasetQAPolicy,
    issues: _Issues,
    class_pixels: Counter[str],
) -> tuple[_Scanned | None, bool]:
    path = asset.image_path
    if path.is_symlink():
        raise DatasetQAError("qa_image_path_unsafe")
    if not path.is_file():
        issues.add("image.missing", "error", asset.key, field="image")
        return None, False
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        issues.add("image.empty_file", "error", asset.key, field="image")
        return None, True
    digest = _sha256(path)
    if size_bytes > policy.max_file_bytes:
        issues.add("image.file_too_large", "error", asset.key, field="image")
        return None, True
    if path.suffix.casefold() not in _IMAGE_SUFFIXES:
        issues.add("image.format_unsupported", "error", asset.key, field="image")
        return None, True
    try:
        with Image.open(path) as verifier:
            width, height = verifier.size
            image_format = (verifier.format or "unknown").lower()
            if (
                width < 1
                or height < 1
                or width > policy.max_side
                or height > policy.max_side
                or width * height > policy.max_decoded_pixels
            ):
                issues.add("image.dimensions_invalid", "error", asset.key, field="image")
                return None, True
            verifier.verify()
        with Image.open(path) as source:
            exif_gps = _gps_from_exif(source.getexif())
            source.load()
            prepared = ImageOps.exif_transpose(source).convert("L")
            quality = prepared.copy()
            quality.thumbnail((512, 512), Image.Resampling.LANCZOS)
            values = np.asarray(quality, dtype=np.uint8)
            mean = float(ImageStat.Stat(quality).mean[0])
            stddev = float(ImageStat.Stat(quality).stddev[0])
            blur = float(cv2.Laplacian(values, cv2.CV_64F).var())
            small = prepared.resize((8, 8), Image.Resampling.LANCZOS)
            pixels = list(small.getdata())
            average = sum(pixels) / len(pixels)
            perceptual = 0
            for pixel in pixels:
                perceptual = (perceptual << 1) | int(pixel >= average)
            prepared.close()
            quality.close()
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
        issues.add("image.corrupt_or_truncated", "error", asset.key, field="image")
        return None, True
    if stddev <= policy.near_blank_stddev:
        issues.add("image.near_blank", "warning", asset.key, field="image")
    if blur < policy.blur_variance_threshold:
        issues.add("image.blur", "warning", asset.key, field="image")
    if mean <= policy.underexposed_mean:
        issues.add("image.underexposed", "warning", asset.key, field="image")
    elif mean >= policy.overexposed_mean:
        issues.add("image.overexposed", "warning", asset.key, field="image")
    if exif_gps is not None and asset.latitude is not None and asset.longitude is not None:
        distance = _haversine_km((asset.latitude, asset.longitude), exif_gps)
        if distance > policy.exif_conflict_km:
            issues.add(
                "gps.exif_conflict",
                "warning",
                asset.key,
                field="coordinates",
                metrics={"distance_bucket": _distance_bucket(distance)},
            )
    mask_sha, mask_classes, mask_scanned = _scan_mask(
        asset,
        (width, height),
        allowed_class_ids,
        policy,
        issues,
        class_pixels,
    )
    return (
        _Scanned(
            asset=asset,
            content_sha256=digest,
            perceptual_hash=f"{perceptual:016x}",
            width=width,
            height=height,
            image_format=image_format,
            mask_sha256=mask_sha,
            mask_classes=mask_classes,
        ),
        mask_scanned,
    )


class _BKNode:
    def __init__(self, value: int, item: _Scanned) -> None:
        self.value = value
        self.items = [item]
        self.children: dict[int, _BKNode] = {}

    def add(self, value: int, item: _Scanned) -> None:
        distance = (self.value ^ value).bit_count()
        if distance == 0:
            self.items.append(item)
            return
        child = self.children.get(distance)
        if child is None:
            self.children[distance] = _BKNode(value, item)
        else:
            child.add(value, item)

    def search(self, value: int, radius: int) -> Iterable[_Scanned]:
        distance = (self.value ^ value).bit_count()
        if distance <= radius:
            yield from self.items
        for edge, child in self.children.items():
            if distance - radius <= edge <= distance + radius:
                yield from child.search(value, radius)


def _related_issue(
    left: _Scanned,
    right: _Scanned,
    issues: _Issues,
    *,
    duplicate_code: str,
) -> None:
    issues.add(
        duplicate_code,
        "warning",
        right.asset.key,
        metrics={"related_asset_key": left.asset.key},
    )
    if left.asset.labels != right.asset.labels:
        issues.add(
            "labels.conflicting_duplicate",
            "error",
            right.asset.key,
            field="labels",
            metrics={"related_asset_key": left.asset.key},
        )
    if left.asset.split and right.asset.split and left.asset.split != right.asset.split:
        issues.add(
            "leakage.duplicate_cross_split",
            "error",
            right.asset.key,
            field="split",
            metrics={"related_asset_key": left.asset.key},
        )


def _duplicates(scanned: list[_Scanned], policy: DatasetQAPolicy, issues: _Issues) -> None:
    exact: dict[str, _Scanned] = {}
    tree: _BKNode | None = None
    for item in sorted(scanned, key=lambda value: value.asset.key):
        existing = exact.get(item.content_sha256)
        if existing is not None:
            _related_issue(existing, item, issues, duplicate_code="duplicate.exact")
        else:
            exact[item.content_sha256] = item
        value = int(item.perceptual_hash, 16)
        if tree is not None:
            matches = sorted(
                tree.search(value, policy.perceptual_distance), key=lambda x: x.asset.key
            )
            for match in matches:
                if match.content_sha256 != item.content_sha256:
                    _related_issue(
                        match,
                        item,
                        issues,
                        duplicate_code="duplicate.perceptual",
                    )
                    break
            tree.add(value, item)
        else:
            tree = _BKNode(value, item)


def _group_leakage(scanned: list[_Scanned], issues: _Issues, field: str, code: str) -> None:
    groups: defaultdict[str, list[_Scanned]] = defaultdict(list)
    for item in scanned:
        value = getattr(item.asset, field)
        if value:
            groups[str(value)].append(item)
    for items in groups.values():
        ordered = sorted(items, key=lambda value: value.asset.key)
        splits = {item.asset.split for item in ordered if item.asset.split}
        if len(splits) > 1:
            for item in ordered[1:]:
                issues.add(
                    code,
                    "error",
                    item.asset.key,
                    field="split",
                    metrics={"related_asset_key": ordered[0].asset.key},
                )


def _check_status(issues: list[DatasetQAIssueView], prefixes: tuple[str, ...]) -> str:
    selected = [item for item in issues if item.code.startswith(prefixes)]
    if any(item.severity == "error" for item in selected):
        return "failed"
    if any(item.severity == "warning" for item in selected):
        return "warning"
    return "passed"


class DatasetQAScanner:
    def __init__(
        self,
        policy: DatasetQAPolicy | None = None,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.policy = policy or DatasetQAPolicy()
        self._clock = clock

    def scan(
        self,
        *,
        images: Path,
        masks: Path | None = None,
        manifest: Path | None = None,
        asset_root: Path | None = None,
        report_id: str,
        allowed_class_ids: frozenset[int] = frozenset(),
        contact_sheet: bool = False,
    ) -> DatasetQARun:
        if not _SAFE_KEY.fullmatch(report_id):
            raise DatasetQAError("qa_report_id_invalid")
        if any(item < 0 or item > 65_535 for item in allowed_class_ids):
            raise DatasetQAError("qa_class_id_invalid")
        image_root = _root(images, "qa_images_root_invalid")
        selected_asset_root = (
            _root(asset_root, "qa_asset_root_invalid") if asset_root else image_root
        )
        mask_root = _root(masks, "qa_masks_root_invalid") if masks else None
        issues = _Issues(self.policy.max_issues)
        if manifest is not None and manifest.expanduser().is_symlink():
            raise DatasetQAError("qa_manifest_symlink_rejected")
        assets = (
            _read_manifest(
                manifest.expanduser(),
                image_root=image_root,
                asset_root=selected_asset_root,
                mask_root=mask_root,
                policy=self.policy,
                issues=issues,
            )
            if manifest is not None
            else _discover_images(image_root, self.policy)
        )
        if manifest is None and mask_root is not None:
            assets = [
                _Asset(
                    key=item.key,
                    image_path=item.image_path,
                    mask_path=_contained(
                        mask_root,
                        item.image_path.relative_to(image_root).with_suffix(".png").as_posix(),
                        "qa_mask_path_unsafe",
                    ),
                    latitude=item.latitude,
                    longitude=item.longitude,
                    split=item.split,
                    capture_family=item.capture_family,
                    sequence=item.sequence,
                    labels=item.labels,
                    country=item.country,
                )
                for item in assets
            ]
        if not assets:
            raise DatasetQAError("qa_dataset_empty")
        seen_keys: set[str] = set()
        for asset in assets:
            if asset.key in seen_keys:
                issues.add("manifest.duplicate_asset_key", "error", asset.key, field="asset_key")
            seen_keys.add(asset.key)

        scanned: list[_Scanned] = []
        scanned_images = 0
        scanned_masks = 0
        class_pixels: Counter[str] = Counter()
        for asset in sorted(assets, key=lambda value: value.key):
            result, mask_scanned = _scan_image(
                asset, allowed_class_ids, self.policy, issues, class_pixels
            )
            scanned_images += int(asset.image_path.is_file())
            scanned_masks += int(mask_scanned)
            if result is not None:
                scanned.append(result)
        _duplicates(scanned, self.policy, issues)
        _group_leakage(scanned, issues, "capture_family", "leakage.capture_family_cross_split")
        _group_leakage(scanned, issues, "sequence", "leakage.sequence_cross_split")

        ordered_issues = sorted(
            issues.items,
            key=lambda item: (item.asset_key, item.code, item.field or "", item.severity),
        )
        distributions: dict[str, dict[str, int]] = {
            "image_format": dict(sorted(Counter(item.image_format for item in scanned).items())),
            "split": dict(
                sorted(Counter(item.asset.split or "unspecified" for item in scanned).items())
            ),
            "country": dict(
                sorted(Counter(item.asset.country or "unspecified" for item in scanned).items())
            ),
            "mask_class_pixels": dict(sorted(class_pixels.items())),
            "issue_code": dict(sorted(Counter(item.code for item in ordered_issues).items())),
        }
        canonical = [
            {
                "asset_key": item.asset.key,
                "content_sha256": item.content_sha256,
                "perceptual_hash": item.perceptual_hash,
                "dimensions": [item.width, item.height],
                "mask_sha256": item.mask_sha256,
                "mask_classes": item.mask_classes,
                "split": item.asset.split,
                "capture_family": item.asset.capture_family,
                "sequence": item.asset.sequence,
                "labels": item.asset.labels,
            }
            for item in sorted(scanned, key=lambda value: value.asset.key)
        ]
        fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        has_masks = mask_root is not None or any(item.mask_path is not None for item in assets)
        has_coordinates = any(
            item.latitude is not None and item.longitude is not None for item in assets
        )
        dataset_type = (
            "mixed"
            if has_masks and has_coordinates
            else "segmentation"
            if has_masks
            else "geolocation"
        )
        limitations = [
            "country_boundary_checks_unavailable",
            "quality_thresholds_are_diagnostic_not_training_repairs",
            "dataset_qa_does_not_establish_model_accuracy",
            "exif_absence_is_neutral",
        ]
        if issues.truncated:
            limitations.append("issue_list_truncated_at_policy_limit")
        outputs = ["report.json", "issues.csv", "summary.md", "report.html"]
        if contact_sheet:
            outputs.append("contact-sheet.jpg")
        checks: dict[str, Any] = {
            "manifest": (
                _check_status(ordered_issues, ("manifest.",))
                if manifest is not None
                else "unavailable"
            ),
            "images": _check_status(ordered_issues, ("image.",)),
            "masks": _check_status(ordered_issues, ("mask.",)) if has_masks else "unavailable",
            "gps": (
                _check_status(ordered_issues, ("gps.",))
                if has_coordinates or any(item.code.startswith("gps.") for item in ordered_issues)
                else "unavailable"
            ),
            "duplicates": _check_status(ordered_issues, ("duplicate.", "labels.")),
            "leakage": _check_status(ordered_issues, ("leakage.",)),
            "country_boundary": "unavailable",
        }
        report = DatasetQAReport(
            summary=DatasetQAReportSummary(
                report_id=report_id,
                dataset_fingerprint=fingerprint,
                dataset_type=dataset_type,
                created_at=self._clock(),
                scanned_images=scanned_images,
                scanned_masks=scanned_masks,
                error_count=issues.errors,
                warning_count=issues.warnings,
            ),
            checks=checks,
            distributions=distributions,
            issues=ordered_issues,
            outputs=outputs,
            limitations=limitations,
        )
        sheet_sources = tuple(
            item.asset.image_path
            for item in sorted(scanned, key=lambda value: value.asset.key)[
                : self.policy.contact_sheet_limit
            ]
        )
        return DatasetQARun(report=report, source_images=sheet_sources)
