from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol

from atlaslens_api.gazetteer.models import GazetteerMetadata, ResolvedPlace
from atlaslens_api.place_evidence.models import ForwardPlaceMatch
from atlaslens_api.place_evidence.normalization import normalize_search_text, secondary_search_key

EARTH_RADIUS_KM = 6371.0088


class GazetteerResolver(Protocol):
    def resolve(self, latitude: float, longitude: float) -> ResolvedPlace: ...


class ForwardGazetteerResolver(Protocol):
    def search(
        self, text: str, *, limit: int = 8, min_similarity: float = 0.78
    ) -> tuple[ForwardPlaceMatch, ...]: ...


def _distance_km(
    latitude: float, longitude: float, other_latitude: float, other_longitude: float
) -> float:
    lat1, lat2 = math.radians(latitude), math.radians(other_latitude)
    delta_latitude = lat2 - lat1
    delta_longitude = math.radians(
        ((other_longitude - longitude + 180.0) % 360.0) - 180.0
    )
    value = (
        math.sin(delta_latitude / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(value)))


class CoordinateFallbackResolver:
    """Honest label fallback; it never invents a place or administrative area."""

    def resolve(self, latitude: float, longitude: float) -> ResolvedPlace:
        return ResolvedPlace(
            label=f"{latitude:.3f}, {longitude:.3f}",
            distance_km=0.0,
            source="coordinate_fallback",
            dataset_version="not_applicable",
            license="not_applicable",
        )


class SQLiteGazetteerResolver:
    """Read-only resolver for an explicitly installed GeoNames-style SQLite artifact."""

    def __init__(
        self,
        database_path: Path,
        metadata: GazetteerMetadata,
        *,
        search_radius_km: float = 100.0,
        fallback: GazetteerResolver | None = None,
    ) -> None:
        if search_radius_km <= 0 or not math.isfinite(search_radius_km):
            raise ValueError("search radius must be positive and finite")
        requested_path = database_path.expanduser()
        if requested_path.is_symlink():
            raise ValueError("gazetteer database symlinks are not accepted")
        path = requested_path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("gazetteer database must be a regular installed file")
        self._database_uri = f"file:{path.as_posix()}?mode=ro"
        self._metadata = metadata
        self._radius_km = search_radius_km
        self._fallback = fallback or CoordinateFallbackResolver()

    def resolve(self, latitude: float, longitude: float) -> ResolvedPlace:
        latitude_delta = self._radius_km / 111.32
        longitude_scale = max(0.01, abs(math.cos(math.radians(latitude))))
        longitude_delta = min(180.0, self._radius_km / (111.32 * longitude_scale))
        with closing(sqlite3.connect(self._database_uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT name, country_code, country, region, city, latitude, longitude
                FROM places
                WHERE latitude BETWEEN ? AND ?
                  AND MIN(
                        ABS(longitude - ?),
                        360.0 - ABS(longitude - ?)
                      ) <= ?
                ORDER BY ((latitude - ?) * (latitude - ?))
                       + MIN(ABS(longitude - ?), 360.0 - ABS(longitude - ?))
                       * MIN(ABS(longitude - ?), 360.0 - ABS(longitude - ?)),
                         population DESC,
                         name ASC
                LIMIT 256
                """,
                (
                    max(-90.0, latitude - latitude_delta),
                    min(90.0, latitude + latitude_delta),
                    longitude,
                    longitude,
                    longitude_delta,
                    latitude,
                    latitude,
                    longitude,
                    longitude,
                    longitude,
                    longitude,
                ),
            ).fetchall()
        if not rows:
            return self._fallback.resolve(latitude, longitude)
        nearest = min(
            rows,
            key=lambda row: (
                _distance_km(latitude, longitude, row["latitude"], row["longitude"]),
                str(row["name"]),
            ),
        )
        distance = _distance_km(
            latitude,
            longitude,
            float(nearest["latitude"]),
            float(nearest["longitude"]),
        )
        if distance > self._radius_km:
            return self._fallback.resolve(latitude, longitude)
        label = str(nearest["name"])
        return ResolvedPlace(
            label=label,
            country_code=nearest["country_code"],
            country=nearest["country"],
            region=nearest["region"],
            city=nearest["city"],
            distance_km=distance,
            source=self._metadata.source,
            dataset_version=self._metadata.version,
            license=self._metadata.license,
        )


class SQLiteForwardGazetteerResolver:
    """Read-only, bounded name lookup over the additive GeoNames v2 artifact."""

    def __init__(self, database_path: Path, metadata: GazetteerMetadata) -> None:
        requested_path = database_path.expanduser()
        if requested_path.is_symlink():
            raise ValueError("gazetteer database symlinks are not accepted")
        path = requested_path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("gazetteer database must be a regular installed file")
        self._database_uri = f"file:{path.as_posix()}?mode=ro"
        self.metadata = metadata

    def search(
        self, text: str, *, limit: int = 8, min_similarity: float = 0.78
    ) -> tuple[ForwardPlaceMatch, ...]:
        if not 1 <= limit <= 32:
            raise ValueError("forward gazetteer limit must be between 1 and 32")
        if not 0 <= min_similarity <= 1 or not math.isfinite(min_similarity):
            raise ValueError("minimum text similarity must be finite and bounded")
        normalized = normalize_search_text(text)
        if len(normalized) < 2 or len(normalized) > 200:
            return ()
        secondary = secondary_search_key(normalized)
        prefix = self._escaped_prefix(normalized[: min(4, len(normalized))])
        with closing(sqlite3.connect(self._database_uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT p.geoname_id, p.name, p.country_code, p.region, p.city,
                       p.latitude, p.longitude, p.population,
                       a.normalized_name, a.secondary_key, a.script,
                       a.language_hint, a.match_type
                FROM aliases AS a
                JOIN places AS p ON p.geoname_id = a.place_id
                WHERE a.normalized_name = ?
                   OR a.secondary_key = ?
                   OR a.normalized_name LIKE ? ESCAPE '\\'
                ORDER BY
                    CASE
                        WHEN a.normalized_name = ? THEN 0
                        WHEN a.secondary_key = ? THEN 1
                        ELSE 2
                    END,
                    p.population DESC,
                    p.geoname_id ASC
                LIMIT 256
                """,
                (normalized, secondary, f"{prefix}%", normalized, secondary),
            ).fetchall()
        best_by_place: dict[int, tuple[sqlite3.Row, float]] = {}
        for row in rows:
            similarity = max(
                SequenceMatcher(None, normalized, str(row["normalized_name"])).ratio(),
                SequenceMatcher(None, secondary, str(row["secondary_key"])).ratio(),
            )
            if similarity < min_similarity:
                continue
            place_id = int(row["geoname_id"])
            previous = best_by_place.get(place_id)
            if previous is None or similarity > previous[1]:
                best_by_place[place_id] = (row, similarity)
        ambiguity_count = len(best_by_place)
        ranked = sorted(
            best_by_place.values(),
            key=lambda item: (-item[1], -int(item[0]["population"]), int(item[0]["geoname_id"])),
        )[:limit]
        return tuple(
            ForwardPlaceMatch(
                geoname_id=int(row["geoname_id"]),
                matched_entity=str(row["name"]),
                normalized_name=str(row["normalized_name"]),
                country_code=row["country_code"],
                region=row["region"],
                city=row["city"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                match_type=str(row["match_type"]),
                text_similarity=similarity,
                ambiguity_count=max(1, ambiguity_count),
                population=int(row["population"]),
                script=str(row["script"]),
                language_hint=row["language_hint"],
            )
            for row, similarity in ranked
        )

    @staticmethod
    def _escaped_prefix(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
