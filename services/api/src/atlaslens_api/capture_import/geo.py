from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from datetime import datetime

from atlaslens_api.reranking.geo import geodesic_km

from .errors import CaptureImportError
from .models import CaptureSampleResult, SyncedFrame, TrackPoint


def geodesic_m(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return 1000.0 * geodesic_km(left, right)


def validate_track(
    points: Sequence[TrackPoint],
    *,
    max_speed_kmh: float,
    max_route_distance_km: float,
) -> tuple[TrackPoint, ...]:
    if not points:
        raise CaptureImportError("track_empty")
    total_m = 0.0
    previous = points[0]
    for current in points[1:]:
        seconds = (current.timestamp - previous.timestamp).total_seconds()
        if seconds <= 0:
            raise CaptureImportError("track_timestamps_not_strictly_increasing")
        distance_m = geodesic_m(
            (previous.latitude, previous.longitude),
            (current.latitude, current.longitude),
        )
        speed_kmh = distance_m / seconds * 3.6
        if speed_kmh > max_speed_kmh:
            raise CaptureImportError("track_speed_limit_exceeded")
        total_m += distance_m
        if total_m > max_route_distance_km * 1000.0:
            raise CaptureImportError("track_distance_limit_exceeded")
        previous = current
    return tuple(points)


def _interpolate_longitude(left: float, right: float, fraction: float) -> float:
    delta = ((right - left + 180.0) % 360.0) - 180.0
    value = left + delta * fraction
    return ((value + 180.0) % 360.0) - 180.0


def _interpolate_heading(
    left: float | None,
    right: float | None,
    fraction: float,
) -> float | None:
    if left is None or right is None:
        return left if fraction < 0.5 else right
    delta = ((right - left + 180.0) % 360.0) - 180.0
    return (left + delta * fraction) % 360.0


def interpolate_track(
    points: Sequence[TrackPoint],
    timestamp: datetime,
    *,
    max_gap_seconds: float,
) -> TrackPoint:
    if not points:
        raise CaptureImportError("track_empty")
    times = [point.timestamp for point in points]
    position = bisect.bisect_left(times, timestamp)
    if position < len(points) and points[position].timestamp == timestamp:
        return points[position]
    if position == 0 or position == len(points):
        raise CaptureImportError("frame_timestamp_outside_track")
    left = points[position - 1]
    right = points[position]
    span = (right.timestamp - left.timestamp).total_seconds()
    if span <= 0 or span > max_gap_seconds:
        raise CaptureImportError("track_interpolation_gap_exceeded")
    fraction = (timestamp - left.timestamp).total_seconds() / span
    return TrackPoint(
        timestamp=timestamp,
        latitude=left.latitude + (right.latitude - left.latitude) * fraction,
        longitude=_interpolate_longitude(left.longitude, right.longitude, fraction),
        accuracy_m=max(left.accuracy_m, right.accuracy_m),
        heading_degrees=_interpolate_heading(
            left.heading_degrees,
            right.heading_degrees,
            fraction,
        ),
    )


def sample_frames(
    frames: Sequence[SyncedFrame],
    *,
    plan_sha256: str,
    distance_m: int,
    stationary_radius_m: float,
) -> CaptureSampleResult:
    if distance_m not in {25, 50, 100}:
        raise CaptureImportError("sample_distance_not_allowed")
    ordered = tuple(sorted(frames, key=lambda item: (item.capture_timestamp, item.source_ordinal)))
    if not ordered:
        raise CaptureImportError("synchronized_frames_empty")
    selected = [ordered[0]]
    movement_anchor = ordered[0]
    accumulated_m = 0.0
    stationary_count = 0
    distance_excluded = 0
    for frame in ordered[1:]:
        segment_m = geodesic_m(
            (movement_anchor.latitude, movement_anchor.longitude),
            (frame.latitude, frame.longitude),
        )
        if segment_m <= stationary_radius_m:
            stationary_count += 1
            continue
        movement_anchor = frame
        accumulated_m += segment_m
        if accumulated_m + 1e-9 < distance_m:
            distance_excluded += 1
            continue
        selected.append(frame)
        accumulated_m %= float(distance_m)
    return CaptureSampleResult(
        plan_sha256=plan_sha256,
        distance_m=distance_m,
        input_count=len(ordered),
        sampled_count=len(selected),
        stationary_deduplicated_count=stationary_count,
        distance_excluded_count=distance_excluded,
        frames=tuple(selected),
    )


def bearing_degrees(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float | None:
    if geodesic_m(left, right) < 0.01:
        return None
    lat1, lat2 = math.radians(left[0]), math.radians(right[0])
    delta_lon = math.radians(((right[1] - left[1] + 180.0) % 360.0) - 180.0)
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2
    ) * math.cos(delta_lon)
    return math.degrees(math.atan2(y, x)) % 360.0


__all__ = [
    "bearing_degrees",
    "geodesic_m",
    "interpolate_track",
    "sample_frames",
    "validate_track",
]
