from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
import socket
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol
from urllib.parse import urlencode, urlsplit

from atlaslens_api.constraints.cache import MapFeatureCache
from atlaslens_api.constraints.models import MapClue, MapConstraintObservation


class MapTransport(Protocol):
    async def post(
        self,
        url: str,
        *,
        body: bytes,
        headers: dict[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
        allowed_ip_addresses: frozenset[str],
    ) -> bytes: ...


class MapProviderError(RuntimeError):
    pass


Resolver = Callable[[str], Sequence[str]]
Sleeper = Callable[[float], Awaitable[None]]


def _system_resolver(host: str) -> Sequence[str]:
    return tuple(
        {str(item[4][0]) for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    )


def validate_public_https_url(
    url: str, *, allowed_hosts: set[str], resolver: Resolver
) -> frozenset[str]:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or host not in {item.lower() for item in allowed_hosts}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("map endpoint must be an allowlisted HTTPS origin")
    addresses = resolver(host)
    if not addresses:
        raise ValueError("map endpoint host did not resolve")
    validated: set[str] = set()
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("map endpoint resolved to a non-public address")
        validated.add(str(ip))
    return frozenset(validated)


class OverpassMapConstraintProvider:
    provider_id = "osm_overpass"
    available = True

    def __init__(
        self,
        transport: MapTransport,
        *,
        endpoint: str = "https://overpass-api.de/api/interpreter",
        allowed_hosts: set[str] | None = None,
        user_agent: str = "AtlasLens/0.1 map-constraints (operator-contact-required)",
        timeout_seconds: float = 5,
        max_response_bytes: int = 1_000_000,
        max_retries: int = 1,
        min_request_interval_seconds: float = 1,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 60,
        cache: MapFeatureCache[frozenset[str]] | None = None,
        resolver: Resolver = _system_resolver,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        if not user_agent.strip() or "contact-required" in user_agent:
            raise ValueError("a descriptive User-Agent with operator contact is required")
        if timeout_seconds <= 0 or max_response_bytes < 1 or not 0 <= max_retries <= 2:
            raise ValueError("invalid bounded request configuration")
        if min_request_interval_seconds < 0 or circuit_failure_threshold < 1:
            raise ValueError("invalid rate or circuit configuration")
        hosts = allowed_hosts or {"overpass-api.de"}
        self._allowed_ips = validate_public_https_url(
            endpoint, allowed_hosts=hosts, resolver=resolver
        )
        self._transport = transport
        self._endpoint = endpoint
        self._headers = {
            "User-Agent": user_agent,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        self._timeout = timeout_seconds
        self._max_bytes = max_response_bytes
        self._max_retries = max_retries
        self._min_interval = min_request_interval_seconds
        self._circuit_threshold = circuit_failure_threshold
        self._circuit_cooldown = circuit_cooldown_seconds
        self._cache = cache or MapFeatureCache()
        self._clock = clock
        self._sleep = sleeper
        self._next_request_at = 0.0
        self._failures = 0
        self._circuit_open_until = 0.0

    async def evaluate(
        self,
        clues: Sequence[MapClue],
        *,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> list[MapConstraintObservation]:
        if not clues:
            return []
        if not math.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError("invalid latitude")
        if not math.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError("invalid longitude")
        if not math.isfinite(radius_km) or not 0 < radius_km <= 100:
            raise ValueError("invalid query radius")
        key = self._cache_key(latitude, longitude, radius_km)
        features = self._cache.get(key)
        limitation = "OSM coverage and freshness vary; absence is not treated as contradiction"
        if features is None:
            try:
                features = await self._fetch(latitude, longitude, radius_km)
                self._cache.put(key, features)
            except MapProviderError as exc:
                return self._unknown(clues, radius_km, str(exc))
        return [
            MapConstraintObservation(
                clue=clue.clue,
                map_feature=clue.map_feature,
                status=(
                    "supported"
                    if clue.expected_present and clue.map_feature in features
                    else "contradicted"
                    if not clue.expected_present and clue.map_feature in features
                    else "neutral"
                ),
                reliability=0.7 if clue.map_feature in features else 0.0,
                query_radius_km=radius_km,
                provider=self.provider_id,
                limitation=limitation,
            )
            for clue in clues
        ]

    def _unknown(
        self, clues: Sequence[MapClue], radius_km: float, limitation: str
    ) -> list[MapConstraintObservation]:
        return [
            MapConstraintObservation(
                clue=clue.clue,
                map_feature=clue.map_feature,
                status="unknown",
                reliability=0,
                query_radius_km=radius_km,
                provider=self.provider_id,
                limitation=limitation,
            )
            for clue in clues
        ]

    def _cache_key(self, latitude: float, longitude: float, radius_km: float) -> str:
        value = f"{latitude:.4f}|{longitude:.4f}|{radius_km:.2f}"
        return hashlib.sha256(value.encode()).hexdigest()

    async def _fetch(self, latitude: float, longitude: float, radius_km: float) -> frozenset[str]:
        now = self._clock()
        if now < self._circuit_open_until:
            raise MapProviderError("map provider circuit is open")
        delay = self._next_request_at - now
        if delay > 0:
            await self._sleep(delay)
        radius_m = min(round(radius_km * 1000), 100_000)
        query = (
            "[out:json][timeout:5];(nwr(around:"
            f"{radius_m},{latitude:.6f},{longitude:.6f}"
            ")[highway];nwr(around:"
            f"{radius_m},{latitude:.6f},{longitude:.6f}"
            ")[building];nwr(around:"
            f"{radius_m},{latitude:.6f},{longitude:.6f}"
            ")[natural];nwr(around:"
            f"{radius_m},{latitude:.6f},{longitude:.6f}"
            ")[waterway];nwr(around:"
            f"{radius_m},{latitude:.6f},{longitude:.6f}"
            ")[railway];nwr(around:"
            f"{radius_m},{latitude:.6f},{longitude:.6f}"
            ")[aeroway];);out tags 500;"
        )
        body = urlencode({"data": query}).encode()
        for attempt in range(self._max_retries + 1):
            self._next_request_at = self._clock() + self._min_interval
            try:
                payload = await self._transport.post(
                    self._endpoint,
                    body=body,
                    headers=self._headers,
                    timeout_seconds=self._timeout,
                    max_response_bytes=self._max_bytes,
                    allowed_ip_addresses=self._allowed_ips,
                )
                if len(payload) > self._max_bytes:
                    raise MapProviderError("map response exceeded the size limit")
                features = self._parse(payload)
                self._failures = 0
                return features
            except (TimeoutError, OSError, MapProviderError, ValueError, RecursionError):
                if attempt < self._max_retries:
                    await self._sleep(min(0.25 * (attempt + 1), 0.5))
                    continue
        self._failures += 1
        if self._failures >= self._circuit_threshold:
            self._circuit_open_until = self._clock() + self._circuit_cooldown
        raise MapProviderError("map provider request failed safely")

    @staticmethod
    def _parse(payload: bytes) -> frozenset[str]:
        data = json.loads(payload)
        if not isinstance(data, dict) or not isinstance(data.get("elements"), list):
            raise MapProviderError("map response was malformed")
        features: set[str] = set()
        for element in data["elements"][:500]:
            if not isinstance(element, dict):
                continue
            tags = element.get("tags")
            if not isinstance(tags, dict):
                continue
            if "highway" in tags:
                features.add("roads")
            if "building" in tags:
                features.add("buildings")
            if "railway" in tags:
                features.add("railway")
            if tags.get("aeroway") in {"aerodrome", "terminal", "runway"}:
                features.add("airport")
            if tags.get("bridge") not in {None, "no"}:
                features.add("bridge")
            if tags.get("tunnel") not in {None, "no"}:
                features.add("tunnel")
            if "waterway" in tags or tags.get("natural") in {"water", "coastline", "bay"}:
                features.add("water")
        return frozenset(features)
