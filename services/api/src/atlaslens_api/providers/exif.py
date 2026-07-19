from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

from PIL import ExifTags, Image

from atlaslens_api.providers.base import (
    ExifResult,
    InvocationContext,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.storage import LocalImageHandle


def _number(value: object) -> float:
    if isinstance(value, Fraction):
        return float(value)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        denominator = float(value.denominator)
        if denominator == 0:
            raise ValueError("zero denominator")
        return float(value.numerator) / denominator
    if isinstance(value, tuple) and len(value) == 2:
        denominator = float(value[1])
        if denominator == 0:
            raise ValueError("zero denominator")
        return float(value[0]) / denominator
    return float(value)  # type: ignore[arg-type]


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore").strip("\x00 ")
    return str(value).strip()


def _coordinate(parts: object, reference: object, positive: str, negative: str) -> float:
    if not isinstance(parts, Sequence) or isinstance(parts, str | bytes) or len(parts) != 3:
        raise ValueError("invalid coordinate")
    degrees, minutes, seconds = (_number(part) for part in parts)
    if 60 <= seconds < 60.000001:
        minutes += 1
        seconds = 0
    if 60 <= minutes < 60.000001:
        degrees += 1
        minutes = 0
    if degrees < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("invalid coordinate component")
    result = degrees + minutes / 60 + seconds / 3600
    hemisphere = _text(reference).upper()
    if hemisphere == negative:
        result = -result
    elif hemisphere != positive:
        raise ValueError("invalid hemisphere")
    return result


def _gps_by_name(gps: Mapping[object, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in gps.items():
        name = ExifTags.GPSTAGS.get(key, key) if isinstance(key, int) else key
        result[str(name)] = value
    return result


def extract_exif(path: Path) -> ExifResult | None:
    with Image.open(path) as image:
        exif = image.getexif()
        if not exif:
            return None
        orientation_raw = exif.get(274)
        orientation = int(orientation_raw) if orientation_raw is not None else None
        try:
            gps_raw = exif.get_ifd(34853)
        except (KeyError, TypeError, AttributeError):
            gps_raw = exif.get(34853)
        if not isinstance(gps_raw, Mapping) or not gps_raw:
            return None
        gps = _gps_by_name(gps_raw)
        required = {"GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef"}
        if not required <= gps.keys():
            return None
        latitude = _coordinate(gps["GPSLatitude"], gps["GPSLatitudeRef"], "N", "S")
        longitude = _coordinate(gps["GPSLongitude"], gps["GPSLongitudeRef"], "E", "W")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("coordinate out of bounds")

        altitude: float | None = None
        if "GPSAltitude" in gps:
            altitude = _number(gps["GPSAltitude"])
            reference = gps.get("GPSAltitudeRef", 0)
            if reference in (1, b"\x01"):
                altitude = -altitude

        captured_at: str | None = None
        date_stamp = gps.get("GPSDateStamp")
        time_stamp = gps.get("GPSTimeStamp")
        if date_stamp is not None and isinstance(time_stamp, Sequence) and len(time_stamp) == 3:
            try:
                hours, minutes, seconds = (_number(part) for part in time_stamp)
                captured_at = (
                    f"{_text(date_stamp).replace(':', '-')}"
                    f"T{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}Z"
                )
            except (TypeError, ValueError):
                captured_at = None

        return ExifResult(
            latitude=latitude,
            longitude=longitude,
            altitude_m=altitude,
            captured_at=captured_at,
            orientation=orientation,
        )


class PillowExifProvider:
    descriptor = ProviderDescriptor(
        id="pillow-exif",
        kind="metadata",
        version="1.0.0",
        execution_boundary="local",
        criticality="optional",
        available=True,
    )

    async def extract(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[ExifResult]:
        started = time.monotonic()
        if context.cancellation.is_set():
            return ProviderOutcome.skipped("unavailable")
        try:
            result = await asyncio.to_thread(extract_exif, handle.path)
        except (OSError, TypeError, ValueError):
            duration = int((time.monotonic() - started) * 1000)
            return ProviderOutcome.failed(
                "invalid_output", retryable=False, attempts=1, duration_ms=duration
            )
        if result is None:
            return ProviderOutcome.abstained()
        return ProviderOutcome.succeeded(result)
