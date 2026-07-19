from __future__ import annotations

import csv
import io
import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from PIL import ExifTags, Image, UnidentifiedImageError

from atlaslens_api.corpus_pipeline import (
    ImageIdentity,
    inspect_image,
    sha256_file,
)
from atlaslens_api.corpus_pipeline.safety import read_bounded_bytes, resolve_contained_file

from .errors import CaptureImportError, ffmpeg_not_available
from .models import CapturePlan, TrackPoint


@dataclass(frozen=True, slots=True)
class ImageSource:
    ordinal: int
    path: Path
    identity: ImageIdentity
    timestamp: datetime
    embedded_point: TrackPoint | None


def parse_aware_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CaptureImportError("timestamp_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CaptureImportError("timestamp_timezone_required")
    return parsed.astimezone(UTC)


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_gpx_track(input_root: Path, locator: str, plan: CapturePlan) -> tuple[TrackPoint, ...]:
    path = resolve_contained_file(input_root, locator)
    payload = read_bounded_bytes(path, max_bytes=64 * 1024 * 1024, error_prefix="gpx")
    upper = payload.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise CaptureImportError("gpx_unsafe_xml_rejected")
    try:
        root = ET.fromstring(payload)  # noqa: S314 - bounded input rejects DTD/entity
    except ET.ParseError as exc:
        raise CaptureImportError("gpx_xml_invalid") from exc
    points: list[TrackPoint] = []
    for element in root.iter():
        if _xml_local_name(element.tag) != "trkpt":
            continue
        if len(points) >= plan.max_track_points:
            raise CaptureImportError("track_point_limit_exceeded")
        try:
            latitude = float(element.attrib["lat"])
            longitude = float(element.attrib["lon"])
        except (KeyError, ValueError) as exc:
            raise CaptureImportError("gpx_coordinate_invalid") from exc
        values = {_xml_local_name(child.tag): (child.text or "").strip() for child in element}
        time_value = values.get("time")
        if not time_value:
            raise CaptureImportError("gpx_timestamp_missing")
        accuracy = plan.default_coordinate_accuracy_m
        if values.get("accuracy"):
            try:
                accuracy = float(values["accuracy"])
            except ValueError as exc:
                raise CaptureImportError("gpx_accuracy_invalid") from exc
        heading: float | None = None
        heading_value = values.get("course") or values.get("heading")
        if heading_value:
            try:
                heading = float(heading_value)
            except ValueError as exc:
                raise CaptureImportError("gpx_heading_invalid") from exc
        try:
            points.append(
                TrackPoint(
                    timestamp=parse_aware_timestamp(time_value),
                    latitude=latitude,
                    longitude=longitude,
                    accuracy_m=accuracy,
                    heading_degrees=heading,
                )
            )
        except ValueError as exc:
            raise CaptureImportError("gpx_point_invalid") from exc
    return tuple(points)


def read_csv_track(input_root: Path, locator: str, plan: CapturePlan) -> tuple[TrackPoint, ...]:
    path = resolve_contained_file(input_root, locator)
    payload = read_bounded_bytes(path, max_bytes=64 * 1024 * 1024, error_prefix="gps_csv")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise CaptureImportError("gps_csv_encoding_invalid") from exc
    reader = csv.DictReader(io.StringIO(text))
    required = {"timestamp", "latitude", "longitude"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise CaptureImportError("gps_csv_headers_invalid")
    points: list[TrackPoint] = []
    try:
        for row in reader:
            if len(points) >= plan.max_track_points:
                raise CaptureImportError("track_point_limit_exceeded")
            points.append(
                TrackPoint(
                    timestamp=parse_aware_timestamp(row["timestamp"]),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    accuracy_m=float(
                        row.get("accuracy_m") or plan.default_coordinate_accuracy_m
                    ),
                    heading_degrees=(
                        float(row["heading_degrees"])
                        if row.get("heading_degrees")
                        else None
                    ),
                )
            )
    except CaptureImportError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureImportError("gps_csv_row_invalid") from exc
    return tuple(points)


def _rational(value: object) -> float:
    try:
        return float(cast(Any, value))  # Pillow IFDRational and numeric values
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise CaptureImportError("exif_gps_invalid") from exc


def _dms_to_degrees(value: object, reference: object) -> float:
    if not isinstance(value, tuple | list) or len(value) != 3:
        raise CaptureImportError("exif_gps_invalid")
    degrees = _rational(value[0]) + _rational(value[1]) / 60 + _rational(value[2]) / 3600
    ref = reference.decode("ascii", "strict") if isinstance(reference, bytes) else str(reference)
    if ref.upper() in {"S", "W"}:
        degrees = -degrees
    elif ref.upper() not in {"N", "E"}:
        raise CaptureImportError("exif_gps_invalid")
    return degrees


def _localize_exif_time(value: str, offset: str | None, timezone_name: str | None) -> datetime:
    try:
        naive = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError as exc:
        raise CaptureImportError("exif_timestamp_invalid") from exc
    if offset:
        try:
            return datetime.fromisoformat(
                naive.strftime("%Y-%m-%dT%H:%M:%S") + offset
            ).astimezone(UTC)
        except ValueError as exc:
            raise CaptureImportError("exif_timezone_invalid") from exc
    if timezone_name is None:
        raise CaptureImportError("capture_timezone_required")
    zone = ZoneInfo(timezone_name)
    first = naive.replace(tzinfo=zone, fold=0)
    second = naive.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != second.utcoffset():
        raise CaptureImportError("capture_local_time_ambiguous")
    round_trip = first.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
    if round_trip != naive:
        raise CaptureImportError("capture_local_time_nonexistent")
    return first.astimezone(UTC)


def _image_exif(path: Path, plan: CapturePlan) -> tuple[datetime, TrackPoint | None]:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except (UnidentifiedImageError, OSError, ValueError, KeyError) as exc:
        raise CaptureImportError("image_metadata_invalid") from exc
    timestamp_raw = exif_ifd.get(36867) or exif.get(306)
    offset_raw = exif_ifd.get(36881) or exif.get(36880)
    if timestamp_raw is None:
        raise CaptureImportError("image_capture_timestamp_missing")
    timestamp_text = (
        timestamp_raw.decode("utf-8", "strict")
        if isinstance(timestamp_raw, bytes)
        else str(timestamp_raw)
    )
    offset_text = (
        offset_raw.decode("ascii", "strict")
        if isinstance(offset_raw, bytes)
        else str(offset_raw)
        if offset_raw is not None
        else None
    )
    timestamp = _localize_exif_time(timestamp_text, offset_text, plan.capture_timezone)
    if not gps:
        return timestamp, None
    try:
        latitude = _dms_to_degrees(gps[2], gps[1])
        longitude = _dms_to_degrees(gps[4], gps[3])
        heading = _rational(gps[17]) if 17 in gps else None
        point = TrackPoint(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            accuracy_m=plan.default_coordinate_accuracy_m,
            heading_degrees=heading,
        )
    except (KeyError, ValueError) as exc:
        raise CaptureImportError("exif_gps_invalid") from exc
    return timestamp, point


def read_image_sources(input_root: Path, plan: CapturePlan) -> tuple[ImageSource, ...]:
    sources: list[ImageSource] = []
    for ordinal, locator in enumerate(plan.media_locators):
        path = resolve_contained_file(input_root, locator)
        try:
            identity = inspect_image(
                path,
                max_bytes=plan.max_media_bytes,
                max_pixels=plan.max_image_pixels,
            )
        except Exception as exc:
            if isinstance(exc, CaptureImportError):
                raise
            raise CaptureImportError("capture_image_invalid") from exc
        if identity.mime_type not in {"image/jpeg", "image/png"}:
            raise CaptureImportError("capture_image_format_unsupported")
        if plan.frame_timestamps is not None:
            timestamp = plan.frame_timestamps[ordinal]
            embedded = None
        elif plan.input_mode in {"ordered_frames_gpx", "ordered_frames_csv"}:
            assert plan.capture_started_at is not None
            assert plan.frame_interval_seconds is not None
            timestamp = plan.capture_started_at + timedelta(
                seconds=ordinal * plan.frame_interval_seconds
            )
            embedded = None
        else:
            timestamp, embedded = _image_exif(path, plan)
        sources.append(
            ImageSource(
                ordinal=ordinal,
                path=path,
                identity=identity,
                timestamp=timestamp,
                embedded_point=embedded,
            )
        )
    return tuple(sources)


def validate_ffmpeg(executable: str | None) -> Path:
    if executable is None:
        raise ffmpeg_not_available()
    supplied = Path(executable)
    try:
        if (
            not supplied.is_absolute()
            or supplied.name.casefold() not in {"ffmpeg", "ffmpeg.exe"}
            or supplied.is_symlink()
            or not supplied.is_file()
        ):
            raise ffmpeg_not_available()
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ffmpeg_not_available() from exc
    if os.name != "nt" and not os.access(resolved, os.X_OK):
        raise ffmpeg_not_available()
    return resolved


def video_sha256(input_root: Path, plan: CapturePlan) -> tuple[Path, str]:
    assert plan.video_locator is not None
    video = resolve_contained_file(input_root, plan.video_locator)
    if video.suffix.casefold() not in {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}:
        raise CaptureImportError("video_container_unsupported")
    digest, _ = sha256_file(video, max_bytes=plan.max_video_bytes)
    return video, digest


def extract_video_frame(
    executable: Path,
    video: Path,
    output: Path,
    *,
    offset_seconds: float,
    timeout_seconds: float,
) -> None:
    input_formats = {
        ".avi": "avi",
        ".m4v": "mov",
        ".mkv": "matroska",
        ".mov": "mov",
        ".mp4": "mov",
        ".webm": "matroska",
    }
    input_format = input_formats.get(video.suffix.casefold())
    if input_format is None:
        raise CaptureImportError("video_container_unsupported")
    command = [
        str(executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-ss",
        f"{offset_seconds:.6f}",
        "-f",
        input_format,
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-map_metadata",
        "-1",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "image2",
        "-vcodec",
        "png",
        "-y",
        str(output),
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(  # noqa: S603 - executable is explicit and validated
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureImportError("ffmpeg_frame_extraction_failed") from exc
    if completed.returncode != 0 or not output.is_file() or output.is_symlink():
        raise CaptureImportError("ffmpeg_frame_extraction_failed")


__all__ = [
    "ImageSource",
    "extract_video_frame",
    "parse_aware_timestamp",
    "read_csv_track",
    "read_gpx_track",
    "read_image_sources",
    "validate_ffmpeg",
    "video_sha256",
]
