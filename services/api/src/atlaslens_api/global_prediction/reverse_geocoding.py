from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from atlaslens_api.gazetteer import GazetteerResolver, ResolvedPlace
from atlaslens_api.global_prediction.clustering import GeoClipCandidateCluster


@dataclass(frozen=True, slots=True)
class ReverseGeocodedCluster:
    cluster: GeoClipCandidateCluster
    place: ResolvedPlace | None
    warning: str | None = None


class SQLiteReverseGeocodeCache:
    """Small persistent cache for bounded cluster-centroid name lookups."""

    def __init__(
        self,
        path: Path,
        *,
        positive_ttl: timedelta = timedelta(days=30),
        negative_ttl: timedelta = timedelta(days=1),
    ) -> None:
        self._path = path.expanduser().resolve()
        self._positive_ttl = positive_ttl
        self._negative_ttl = negative_ttl
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._initialize_sync)

    async def get(self, latitude: float, longitude: float) -> ResolvedPlace | None | object:
        async with self._lock:
            return await asyncio.to_thread(self._get_sync, self._key(latitude, longitude))

    async def put(self, latitude: float, longitude: float, place: ResolvedPlace | None) -> None:
        async with self._lock:
            await asyncio.to_thread(self._put_sync, self._key(latitude, longitude), place)

    @staticmethod
    def _key(latitude: float, longitude: float) -> str:
        normalized_longitude = ((longitude + 180.0) % 360.0) - 180.0
        return f"{latitude:.4f},{normalized_longitude:.4f}"

    def _initialize_sync(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reverse_geocode_cache (
                    coordinate_key TEXT PRIMARY KEY,
                    payload_json TEXT,
                    expires_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _get_sync(self, key: str) -> ResolvedPlace | None | object:
        if not self._path.is_file():
            return _CACHE_MISS
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute(
                "SELECT payload_json, expires_at FROM reverse_geocode_cache "
                "WHERE coordinate_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return _CACHE_MISS
            try:
                expires_at = datetime.fromisoformat(str(row[1]))
            except ValueError:
                return _CACHE_MISS
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= datetime.now(UTC):
                connection.execute(
                    "DELETE FROM reverse_geocode_cache WHERE coordinate_key = ?", (key,)
                )
                connection.commit()
                return _CACHE_MISS
            payload = row[0]
            if payload is None:
                return None
            try:
                return ResolvedPlace.model_validate_json(str(payload))
            except ValueError:
                return _CACHE_MISS

    def _put_sync(self, key: str, place: ResolvedPlace | None) -> None:
        ttl = self._positive_ttl if place is not None else self._negative_ttl
        expires_at = (datetime.now(UTC) + ttl).isoformat()
        payload = None if place is None else place.model_dump_json()
        with closing(sqlite3.connect(self._path)) as connection:
            connection.execute(
                """
                INSERT INTO reverse_geocode_cache(coordinate_key, payload_json, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(coordinate_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    expires_at = excluded.expires_at
                """,
                (key, payload, expires_at),
            )
            connection.commit()


_CACHE_MISS = object()


class CachedClusterReverseGeocoder:
    """Names only bounded cluster centroids; it never creates or scores a prediction."""

    provider_id = "local-geonames-reverse-geocoder"

    def __init__(
        self,
        resolver: GazetteerResolver | None,
        cache: SQLiteReverseGeocodeCache,
        *,
        top_k: int = 10,
        timeout_seconds: float = 2.0,
    ) -> None:
        if not 1 <= top_k <= 50:
            raise ValueError("reverse-geocode top-k must be between 1 and 50")
        if timeout_seconds <= 0:
            raise ValueError("reverse-geocode timeout must be positive")
        self._resolver = resolver
        self._cache = cache
        self._top_k = top_k
        self._timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return self._resolver is not None

    @property
    def maximum_duration_seconds(self) -> float:
        """Bound used by the pipeline's outer cancellation/deadline guard."""

        return self._top_k * self._timeout_seconds

    async def enrich(
        self, clusters: Sequence[GeoClipCandidateCluster]
    ) -> tuple[ReverseGeocodedCluster, ...]:
        cache_available = True
        try:
            await self._cache.initialize()
        except (OSError, sqlite3.Error):
            cache_available = False
        enriched: list[ReverseGeocodedCluster] = []
        for index, cluster in enumerate(clusters):
            if index >= self._top_k:
                enriched.append(ReverseGeocodedCluster(cluster=cluster, place=None))
                continue
            if self._resolver is None:
                enriched.append(
                    ReverseGeocodedCluster(
                        cluster=cluster,
                        place=None,
                        warning="provider.reverse_geocoding.unavailable",
                    )
                )
                continue
            warning = None if cache_available else "provider.reverse_geocoding.cache_unavailable"
            if cache_available:
                try:
                    cached = await self._cache.get(cluster.latitude, cluster.longitude)
                except (OSError, sqlite3.Error):
                    cache_available = False
                    cached = _CACHE_MISS
                    warning = "provider.reverse_geocoding.cache_unavailable"
            else:
                cached = _CACHE_MISS
            if cached is not _CACHE_MISS:
                enriched.append(
                    ReverseGeocodedCluster(
                        cluster=cluster,
                        place=cached if isinstance(cached, ResolvedPlace) else None,
                        warning=warning,
                    )
                )
                continue
            try:
                place = await asyncio.wait_for(
                    asyncio.to_thread(self._resolver.resolve, cluster.latitude, cluster.longitude),
                    timeout=self._timeout_seconds,
                )
            except (OSError, RuntimeError, sqlite3.Error, TimeoutError, ValueError):
                enriched.append(
                    ReverseGeocodedCluster(
                        cluster=cluster,
                        place=None,
                        warning="provider.reverse_geocoding.failed",
                    )
                )
                continue
            usable = place if place.source != "coordinate_fallback" else None
            if cache_available:
                try:
                    await self._cache.put(cluster.latitude, cluster.longitude, usable)
                except (OSError, sqlite3.Error):
                    cache_available = False
                    warning = "provider.reverse_geocoding.cache_unavailable"
            enriched.append(
                ReverseGeocodedCluster(
                    cluster=cluster,
                    place=usable,
                    warning=warning,
                )
            )
        return tuple(enriched)
