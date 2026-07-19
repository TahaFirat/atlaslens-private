from __future__ import annotations

import math
from collections.abc import Sequence

EARTH_RADIUS_KM = 6371.0088


def geodesic_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    delta_latitude = lat2 - lat1
    delta_longitude = math.radians(((right[1] - left[1] + 180.0) % 360.0) - 180.0)
    value = (
        math.sin(delta_latitude / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(value)))


def spherical_center(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        raise ValueError("at least one point is required")
    x = y = z = 0.0
    for latitude, longitude in points:
        lat = math.radians(latitude)
        lon = math.radians(longitude)
        x += math.cos(lat) * math.cos(lon)
        y += math.cos(lat) * math.sin(lon)
        z += math.sin(lat)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        return min(points)
    horizontal = math.hypot(x, y)
    latitude = math.degrees(math.atan2(z, horizontal))
    longitude = math.degrees(math.atan2(y, x)) if horizontal > 1e-12 else min(points)[1]
    if longitude == -180.0:
        longitude = 180.0
    return latitude, longitude


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]
