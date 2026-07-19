from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from atlaslens_api.reranking.geo import EARTH_RADIUS_KM, geodesic_km

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


@dataclass(frozen=True, slots=True)
class ScoredPoint:
    latitude: float
    longitude: float
    score: float
    source_id: str
    level: str
    resolution_km: float


def normalize_longitude(value: float) -> float:
    normalized = ((value + 180.0) % 360.0) - 180.0
    return 180.0 if normalized == -180.0 and value > 0 else normalized


def fibonacci_global_grid(count: int) -> tuple[tuple[float, float], ...]:
    """Return approximately equal-area sphere samples, including near-polar cells."""

    if not 32 <= count <= 100_000:
        raise ValueError("global grid count is outside the bounded range")
    points: list[tuple[float, float]] = []
    for index in range(count):
        z = 1.0 - (2.0 * index + 1.0) / count
        latitude = math.degrees(math.asin(max(-1.0, min(1.0, z))))
        longitude = normalize_longitude(math.degrees(index * _GOLDEN_ANGLE))
        points.append((latitude, longitude))
    return tuple(points)


def approximate_grid_resolution_km(count: int) -> float:
    surface_area = 4.0 * math.pi * EARTH_RADIUS_KM * EARTH_RADIUS_KM
    return math.sqrt(surface_area / count)


def destination_point(
    origin: tuple[float, float], *, bearing_degrees: float, distance_km: float
) -> tuple[float, float]:
    if not 0 <= distance_km <= math.pi * EARTH_RADIUS_KM:
        raise ValueError("geodesic refinement distance is invalid")
    latitude, longitude = map(math.radians, origin)
    bearing = math.radians(bearing_degrees)
    angular = distance_km / EARTH_RADIUS_KM
    target_latitude = math.asin(
        math.sin(latitude) * math.cos(angular)
        + math.cos(latitude) * math.sin(angular) * math.cos(bearing)
    )
    target_longitude = longitude + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(latitude),
        math.cos(angular) - math.sin(latitude) * math.sin(target_latitude),
    )
    return math.degrees(target_latitude), normalize_longitude(math.degrees(target_longitude))


def geodesic_refinement_points(
    anchors: Sequence[tuple[float, float]],
    *,
    radius_km: float,
    bearings: int = 8,
) -> tuple[tuple[float, float], ...]:
    if not anchors or len(anchors) > 32:
        raise ValueError("refinement anchors are outside the bounded range")
    if not 1 <= bearings <= 24:
        raise ValueError("refinement bearing count is outside the bounded range")
    points: list[tuple[float, float]] = []
    for anchor in anchors:
        points.append(anchor)
        for index in range(bearings):
            points.append(
                destination_point(
                    anchor,
                    bearing_degrees=index * 360.0 / bearings,
                    distance_km=radius_km,
                )
            )
    unique: dict[tuple[int, int], tuple[float, float]] = {}
    for latitude, longitude in points:
        key = (round(latitude * 1_000_000), round(longitude * 1_000_000))
        unique.setdefault(key, (latitude, longitude))
    return tuple(unique.values())


def select_geographically_diverse(
    points: Sequence[ScoredPoint],
    *,
    limit: int,
    minimum_distance_km: float,
) -> tuple[ScoredPoint, ...]:
    if limit < 1 or minimum_distance_km < 0:
        raise ValueError("diversity policy is invalid")
    ordered = sorted(points, key=lambda item: (-item.score, item.source_id))
    selected: list[ScoredPoint] = []
    for point in ordered:
        coordinate = (point.latitude, point.longitude)
        if all(
            geodesic_km(coordinate, (other.latitude, other.longitude)) >= minimum_distance_km
            for other in selected
        ):
            selected.append(point)
        if len(selected) == limit:
            return tuple(selected)
    for point in ordered:
        if point not in selected:
            selected.append(point)
        if len(selected) == limit:
            break
    return tuple(selected)
