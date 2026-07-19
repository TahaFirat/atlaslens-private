from __future__ import annotations

import ssl
from collections.abc import Sequence

import pytest

from atlaslens_api.constraints.https_transport import PinnedHttpsMapTransport
from atlaslens_api.constraints.models import MapClue
from atlaslens_api.constraints.overpass import OverpassMapConstraintProvider


class FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def settimeout(self, value: float | None) -> None:
        self.timeouts.append(value)


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        content_length: str | None = None,
    ) -> None:
        self.status = status
        self._payload = payload
        self._offset = 0
        self._content_length = content_length
        self.closed = False
        self.read_calls = 0

    def getheader(self, name: str, default: str | None = None) -> str | None:
        if name.casefold() == "content-length":
            return self._content_length
        return default

    def read(self, amount: int | None = None) -> bytes:
        self.read_calls += 1
        if self._offset >= len(self._payload):
            return b""
        end = len(self._payload) if amount is None else self._offset + amount
        chunk = self._payload[self._offset : end]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(
        self,
        response: FakeResponse,
        *,
        request_error: BaseException | None = None,
    ) -> None:
        self.sock = FakeSocket()
        self.response = response
        self.request_error = request_error
        self.closed = False
        self.requests: list[tuple[str, str, bytes | None, dict[str, str] | None]] = []

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        *,
        encode_chunked: bool = False,
    ) -> None:
        assert encode_chunked is False
        self.requests.append((method, url, body, headers))
        if self.request_error is not None:
            raise self.request_error

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


class QueueConnectionFactory:
    def __init__(self, connections: Sequence[FakeConnection]) -> None:
        self.connections = list(connections)
        self.calls: list[tuple[str, str, float, ssl.SSLContext]] = []

    def __call__(
        self,
        host: str,
        ip_address: str,
        timeout_seconds: float,
        ssl_context: ssl.SSLContext,
    ) -> FakeConnection:
        self.calls.append((host, ip_address, timeout_seconds, ssl_context))
        return self.connections.pop(0)


def request_headers(user_agent: str = "AtlasLens/0.1 maps ops@example.com") -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Content-Type": "application/x-www-form-urlencoded",
    }


@pytest.mark.asyncio
async def test_transport_is_inert_until_called_and_pins_ip_with_verified_tls() -> None:
    response = FakeResponse(b'{"elements":[]}', content_length="15")
    connection = FakeConnection(response)
    factory = QueueConnectionFactory([connection])
    transport = PinnedHttpsMapTransport(connection_factory=factory)
    assert factory.calls == []

    payload = await transport.post(
        "https://maps.example/api/interpreter",
        body=b"data=query",
        headers=request_headers(),
        timeout_seconds=5,
        max_response_bytes=1_000,
        allowed_ip_addresses=frozenset({"93.184.216.34"}),
    )

    assert payload == b'{"elements":[]}'
    host, address, timeout, context = factory.calls[0]
    assert host == "maps.example"
    assert address == "93.184.216.34"
    assert 0 < timeout <= 5
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert connection.requests[0][0:3] == ("POST", "/api/interpreter", b"data=query")
    sent_headers = connection.requests[0][3]
    assert sent_headers is not None
    assert sent_headers["Accept-Encoding"] == "identity"
    assert response.closed and connection.closed
    assert connection.sock.timeouts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "limit"),
    [
        (FakeResponse(b"ignored", content_length="1001"), 1_000),
        (FakeResponse(b"x" * 1_001), 1_000),
        (FakeResponse(b"redirect", status=302), 1_000),
    ],
)
async def test_transport_rejects_oversized_and_non_success_responses(
    response: FakeResponse, limit: int
) -> None:
    connection = FakeConnection(response)
    transport = PinnedHttpsMapTransport(
        connection_factory=QueueConnectionFactory([connection])
    )

    with pytest.raises(OSError, match="bounded HTTPS request failed"):
        await transport.post(
            "https://maps.example/api",
            body=b"data=query",
            headers=request_headers(),
            timeout_seconds=5,
            max_response_bytes=limit,
            allowed_ip_addresses=frozenset({"93.184.216.34"}),
        )

    assert connection.closed and response.closed
    if response.getheader("Content-Length") == "1001":
        assert response.read_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "headers", "addresses", "body"),
    [
        (
            "http://maps.example/api",
            request_headers(),
            frozenset({"93.184.216.34"}),
            b"data=query",
        ),
        (
            "https://maps.example/api",
            request_headers("AtlasLens without contact"),
            frozenset({"93.184.216.34"}),
            b"data=query",
        ),
        (
            "https://maps.example/api",
            request_headers(),
            frozenset({"127.0.0.1"}),
            b"data=query",
        ),
        (
            "https://maps.example/api",
            {**request_headers(), "Host": "internal.example"},
            frozenset({"93.184.216.34"}),
            b"data=query",
        ),
        (
            "https://maps.example/api",
            request_headers(),
            frozenset({"93.184.216.34"}),
            b"x" * 129_000,
        ),
    ],
    ids=["http", "missing-contact", "private-ip", "host-header", "oversized-body"],
)
async def test_transport_rejects_unsafe_request_inputs(
    url: str,
    headers: dict[str, str],
    addresses: frozenset[str],
    body: bytes,
) -> None:
    factory = QueueConnectionFactory([])
    transport = PinnedHttpsMapTransport(connection_factory=factory)

    with pytest.raises(ValueError):
        await transport.post(
            url,
            body=body,
            headers=headers,
            timeout_seconds=5,
            max_response_bytes=1_000,
            allowed_ip_addresses=addresses,
        )
    assert factory.calls == []


@pytest.mark.asyncio
async def test_existing_provider_retry_policy_handles_transport_timeout_without_network() -> None:
    first = FakeConnection(FakeResponse(b""), request_error=TimeoutError())
    second = FakeConnection(FakeResponse(b'{"elements":[]}', content_length="15"))
    factory = QueueConnectionFactory([first, second])
    transport = PinnedHttpsMapTransport(connection_factory=factory)

    async def no_wait(_delay: float) -> None:
        return None

    provider = OverpassMapConstraintProvider(
        transport,
        endpoint="https://maps.example/api",
        allowed_hosts={"maps.example"},
        user_agent="AtlasLens/0.1 maps ops@example.com",
        resolver=lambda _host: ["93.184.216.34"],
        sleeper=no_wait,
        min_request_interval_seconds=0,
        max_retries=1,
    )
    observations = await provider.evaluate(
        [MapClue(clue="road scene", map_feature="roads")],
        latitude=41,
        longitude=29,
        radius_km=1,
    )

    assert observations[0].status == "neutral"
    assert len(factory.calls) == 2
    assert first.closed and second.closed
