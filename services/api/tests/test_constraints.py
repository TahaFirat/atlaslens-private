from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from atlaslens_api.constraints import (
    DisabledMapConstraintProvider,
    FixtureMapConstraintProvider,
    MapClue,
    MapFeatureCache,
)
from atlaslens_api.constraints.overpass import (
    OverpassMapConstraintProvider,
    validate_public_https_url,
)


class FakeTransport:
    def __init__(self, responses: Sequence[bytes | Exception]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.headers: dict[str, str] = {}

    async def post(
        self,
        url: str,
        *,
        body: bytes,
        headers: dict[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
        allowed_ip_addresses: frozenset[str],
    ) -> bytes:
        assert url.startswith("https://")
        assert body.startswith(b"data=")
        assert b"out%3Ajson" in body
        assert timeout_seconds > 0
        assert max_response_bytes > 0
        assert all(not address.startswith("127.") for address in allowed_ip_addresses)
        self.calls += 1
        self.headers = headers
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def public_resolver(_host: str) -> Sequence[str]:
    return ["93.184.216.34"]


def provider(transport: FakeTransport, **kwargs: object) -> OverpassMapConstraintProvider:
    return OverpassMapConstraintProvider(
        transport,
        endpoint="https://maps.example/api",
        allowed_hosts={"maps.example"},
        user_agent="AtlasLens/0.1 map-constraints ops@example.invalid",
        resolver=public_resolver,
        min_request_interval_seconds=0,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_disabled_and_fixture_provider_classifications() -> None:
    clues = [
        MapClue(clue="visible coast", map_feature="water"),
        MapClue(clue="no railway visible", map_feature="railway", expected_present=False),
        MapClue(clue="mountain terrain", map_feature="terrain"),
    ]
    disabled = await DisabledMapConstraintProvider().evaluate(
        clues, latitude=1, longitude=2, radius_km=3
    )
    assert {item.status for item in disabled} == {"unknown"}

    fixture = FixtureMapConstraintProvider(
        {"water", "railway"}, assessed_features={"water", "railway"}
    )
    observed = await fixture.evaluate(clues, latitude=1, longitude=2, radius_km=3)
    assert [item.status for item in observed] == ["supported", "contradicted", "neutral"]
    assert all(item.query_radius_km == 3 for item in observed)


@pytest.mark.asyncio
async def test_overpass_parses_only_requested_clues_and_caches() -> None:
    payload = json.dumps(
        {"elements": [{"type": "way", "tags": {"highway": "primary", "bridge": "yes"}}]}
    ).encode()
    transport = FakeTransport([payload])
    map_provider = provider(transport)
    clues = [MapClue(clue="road scene", map_feature="roads")]
    first = await map_provider.evaluate(clues, latitude=12, longitude=34, radius_km=2)
    second = await map_provider.evaluate(clues, latitude=12, longitude=34, radius_km=2)
    assert first == second
    assert first[0].status == "supported"
    assert transport.calls == 1
    assert "User-Agent" in transport.headers


@pytest.mark.asyncio
async def test_overpass_absence_is_neutral_and_unrelated_features_are_not_returned() -> None:
    transport = FakeTransport([b'{"elements":[]}'])
    observed = await provider(transport).evaluate(
        [MapClue(clue="coastal appearance", map_feature="water")],
        latitude=0,
        longitude=0,
        radius_km=1,
    )
    assert len(observed) == 1
    assert observed[0].status == "neutral"
    assert observed[0].reliability == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, expected_text",
    [
        (TimeoutError(), "failed safely"),
        (b"not-json", "failed safely"),
        (b'{"wrong":[]}', "failed safely"),
        (b"x" * 101, "failed safely"),
    ],
)
async def test_overpass_timeout_malformed_and_oversized_are_unknown(
    response: bytes | Exception, expected_text: str
) -> None:
    transport = FakeTransport([response])
    map_provider = provider(transport, max_response_bytes=100, max_retries=0)
    observed = await map_provider.evaluate(
        [MapClue(clue="road scene", map_feature="roads")],
        latitude=1,
        longitude=1,
        radius_km=1,
    )
    assert observed[0].status == "unknown"
    assert expected_text in (observed[0].limitation or "")


def test_overpass_url_validation_rejects_ssrf_and_unsafe_urls() -> None:
    with pytest.raises(ValueError, match="allowlisted HTTPS"):
        validate_public_https_url(
            "http://maps.example/api", allowed_hosts={"maps.example"}, resolver=public_resolver
        )
    with pytest.raises(ValueError, match="non-public"):
        validate_public_https_url(
            "https://maps.example/api",
            allowed_hosts={"maps.example"},
            resolver=lambda _host: ["127.0.0.1"],
        )
    with pytest.raises(ValueError, match="allowlisted HTTPS"):
        validate_public_https_url(
            "https://user@maps.example/api",
            allowed_hosts={"maps.example"},
            resolver=public_resolver,
        )


@pytest.mark.asyncio
async def test_rate_limit_waits_and_circuit_breaker_short_circuits() -> None:
    now = [10.0]
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    transport = FakeTransport([TimeoutError(), TimeoutError()])
    map_provider = OverpassMapConstraintProvider(
        transport,
        endpoint="https://maps.example/api",
        allowed_hosts={"maps.example"},
        user_agent="AtlasLens/0.1 map-constraints ops@example.invalid",
        resolver=public_resolver,
        clock=lambda: now[0],
        sleeper=sleep,
        min_request_interval_seconds=2,
        max_retries=0,
        circuit_failure_threshold=2,
        circuit_cooldown_seconds=30,
    )
    clue = [MapClue(clue="road scene", map_feature="roads")]
    first = await map_provider.evaluate(clue, latitude=1, longitude=1, radius_km=1)
    second = await map_provider.evaluate(clue, latitude=2, longitude=2, radius_km=1)
    assert first[0].status == second[0].status == "unknown"
    assert sleeps == [2]
    third = await map_provider.evaluate(clue, latitude=3, longitude=3, radius_km=1)
    assert third[0].status == "unknown"
    assert transport.calls == 2


def test_ttl_lru_cache_expires_and_evicts() -> None:
    now = [0.0]
    cache: MapFeatureCache[int] = MapFeatureCache(
        max_entries=2, ttl_seconds=5, clock=lambda: now[0]
    )
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert cache.get("b") is None
    now[0] = 6
    assert cache.get("a") is None
