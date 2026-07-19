from __future__ import annotations

import asyncio
import http.client
import ipaddress
import math
import re
import socket
import ssl
import time
from collections.abc import Callable
from typing import Protocol, cast
from urllib.parse import SplitResult, urlsplit

_CONTACT_PATTERN = re.compile(
    r"(?:https://[^\s()]+|mailto:[^\s()@]+@[^\s()@]+|[^\s()@]+@[^\s()@]+)",
    re.IGNORECASE,
)
_RESERVED_REQUEST_HEADERS = frozenset(
    {"connection", "content-length", "host", "proxy-authorization", "transfer-encoding"}
)
_READ_CHUNK_BYTES = 64 * 1024


class _TimeoutSocket(Protocol):
    def settimeout(self, value: float | None) -> None: ...


class _HttpResponse(Protocol):
    status: int

    def getheader(self, name: str, default: str | None = None) -> str | None: ...

    def read(self, amount: int | None = None) -> bytes: ...

    def close(self) -> None: ...


class _HttpsConnection(Protocol):
    sock: _TimeoutSocket | None

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        *,
        encode_chunked: bool = False,
    ) -> None: ...

    def getresponse(self) -> _HttpResponse: ...

    def close(self) -> None: ...


class HttpsConnectionFactory(Protocol):
    def __call__(
        self,
        host: str,
        ip_address: str,
        timeout_seconds: float,
        ssl_context: ssl.SSLContext,
    ) -> _HttpsConnection: ...


class _PinnedHttpsConnection(http.client.HTTPSConnection):
    """Connect to a validated IP while retaining hostname TLS verification."""

    def __init__(
        self,
        host: str,
        ip_address: str,
        timeout_seconds: float,
        ssl_context: ssl.SSLContext,
    ) -> None:
        super().__init__(host, port=443, timeout=timeout_seconds, context=ssl_context)
        self._pinned_ip_address = ip_address
        self._pinned_ssl_context = ssl_context

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_ip_address, self.port),
            timeout=self.timeout,
            source_address=None,
        )
        try:
            self.sock = self._pinned_ssl_context.wrap_socket(
                raw_socket, server_hostname=self.host
            )
        except BaseException:
            raw_socket.close()
            raise


def _default_connection_factory(
    host: str,
    ip_address: str,
    timeout_seconds: float,
    ssl_context: ssl.SSLContext,
) -> _HttpsConnection:
    return cast(
        _HttpsConnection,
        _PinnedHttpsConnection(host, ip_address, timeout_seconds, ssl_context),
    )


class PinnedHttpsMapTransport:
    """Bounded, direct HTTPS transport for the explicit Overpass provider seam.

    The caller supplies IP addresses previously resolved and validated by the
    provider. Connections are made directly to those addresses, so environment
    proxies and a second DNS lookup cannot redirect the request. TLS SNI and
    certificate validation still use the allowlisted endpoint hostname.

    Constructing this object performs no DNS lookup or network request.
    """

    def __init__(
        self,
        *,
        max_timeout_seconds: float = 15.0,
        max_response_bytes: int = 2_000_000,
        max_request_bytes: int = 128_000,
        max_ip_addresses: int = 8,
        ssl_context: ssl.SSLContext | None = None,
        connection_factory: HttpsConnectionFactory = _default_connection_factory,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(max_timeout_seconds) or not 0 < max_timeout_seconds <= 60:
            raise ValueError("HTTPS timeout bound must be finite and at most 60 seconds")
        if not 1 <= max_response_bytes <= 10_000_000:
            raise ValueError("HTTPS response byte bound is invalid")
        if not 1 <= max_request_bytes <= 1_000_000:
            raise ValueError("HTTPS request byte bound is invalid")
        if not 1 <= max_ip_addresses <= 16:
            raise ValueError("HTTPS address bound is invalid")
        context = ssl_context or ssl.create_default_context()
        if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
            raise ValueError("HTTPS transport requires hostname and certificate verification")
        self._max_timeout = max_timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._max_request_bytes = max_request_bytes
        self._max_ip_addresses = max_ip_addresses
        self._ssl_context = context
        self._connection_factory = connection_factory
        self._clock = clock

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
        return await asyncio.to_thread(
            self._post_sync,
            url,
            body,
            headers,
            timeout_seconds,
            max_response_bytes,
            allowed_ip_addresses,
        )

    def _post_sync(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
        allowed_ip_addresses: frozenset[str],
    ) -> bytes:
        parsed = self._validate_request(
            url,
            body=body,
            headers=headers,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )
        addresses = self._validated_addresses(allowed_ip_addresses)
        safe_headers = dict(headers)
        safe_headers.setdefault("Accept", "application/json")
        safe_headers["Accept-Encoding"] = "identity"
        deadline = self._clock() + timeout_seconds
        last_error: BaseException | None = None
        for address in addresses:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError("bounded HTTPS request timed out") from last_error
            try:
                return self._post_to_address(
                    parsed,
                    address=address,
                    body=body,
                    headers=safe_headers,
                    deadline=deadline,
                    max_response_bytes=max_response_bytes,
                )
            except (OSError, http.client.HTTPException) as exc:
                last_error = exc
        raise OSError("bounded HTTPS request failed") from last_error

    def _post_to_address(
        self,
        parsed: SplitResult,
        *,
        address: str,
        body: bytes,
        headers: dict[str, str],
        deadline: float,
        max_response_bytes: int,
    ) -> bytes:
        host = parsed.hostname
        if host is None:  # guarded by _validate_request
            raise ValueError("HTTPS endpoint host is required")
        connection = self._connection_factory(
            host,
            address,
            self._remaining(deadline),
            self._ssl_context,
        )
        try:
            target = parsed.path or "/"
            connection.request("POST", target, body=body, headers=headers, encode_chunked=False)
            self._set_socket_timeout(connection, deadline)
            response = connection.getresponse()
            try:
                if response.status != 200:
                    raise OSError("HTTPS endpoint returned a non-success status")
                self._validate_content_length(response, max_response_bytes)
                payload = bytearray()
                while True:
                    self._set_socket_timeout(connection, deadline)
                    read_size = min(
                        _READ_CHUNK_BYTES,
                        max_response_bytes - len(payload) + 1,
                    )
                    chunk = response.read(read_size)
                    if not chunk:
                        return bytes(payload)
                    payload.extend(chunk)
                    if len(payload) > max_response_bytes:
                        raise OSError("HTTPS response exceeded the byte limit")
            finally:
                response.close()
        finally:
            connection.close()

    def _validate_request(
        self,
        url: str,
        *,
        body: bytes,
        headers: dict[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> SplitResult:
        parsed = urlsplit(url)
        host = parsed.hostname
        if (
            parsed.scheme != "https"
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
            or parsed.query
            or parsed.fragment
            or any(ord(character) <= 32 or ord(character) == 127 for character in parsed.path)
        ):
            raise ValueError("HTTPS transport requires a safe HTTPS origin and path")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= self._max_timeout:
            raise ValueError("HTTPS request timeout exceeds the configured bound")
        if not 1 <= max_response_bytes <= self._max_response_bytes:
            raise ValueError("HTTPS response limit exceeds the configured bound")
        if len(body) > self._max_request_bytes:
            raise ValueError("HTTPS request body exceeds the configured bound")
        user_agent = next(
            (value for key, value in headers.items() if key.casefold() == "user-agent"),
            "",
        )
        if not user_agent or len(user_agent) > 300 or not _CONTACT_PATTERN.search(user_agent):
            raise ValueError("HTTPS transport requires a contactable User-Agent")
        for key, value in headers.items():
            normalized = key.casefold()
            if (
                not key
                or normalized in _RESERVED_REQUEST_HEADERS
                or len(key) > 100
                or len(value) > 2_000
                or any(character in key or character in value for character in "\r\n")
            ):
                raise ValueError("HTTPS request contains an unsafe header")
        return parsed

    def _validated_addresses(self, addresses: frozenset[str]) -> tuple[str, ...]:
        if not addresses or len(addresses) > self._max_ip_addresses:
            raise ValueError("HTTPS transport requires a bounded validated address set")
        parsed_addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
        for value in addresses:
            if "%" in value:
                raise ValueError("scoped IP addresses are not allowed")
            address = ipaddress.ip_address(value)
            if not address.is_global:
                raise ValueError("HTTPS transport refuses non-public addresses")
            parsed_addresses.append(address)
        parsed_addresses.sort(key=lambda item: (item.version, int(item)))
        return tuple(str(item) for item in parsed_addresses)

    def _set_socket_timeout(self, connection: _HttpsConnection, deadline: float) -> None:
        remaining = self._remaining(deadline)
        if connection.sock is None:
            raise OSError("HTTPS connection did not expose a socket")
        connection.sock.settimeout(remaining)

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise TimeoutError("bounded HTTPS request timed out")
        return remaining

    @staticmethod
    def _validate_content_length(response: _HttpResponse, max_response_bytes: int) -> None:
        content_length = response.getheader("Content-Length")
        if content_length is None:
            return
        try:
            parsed_length = int(content_length)
        except ValueError as exc:
            raise OSError("HTTPS response contained an invalid content length") from exc
        if parsed_length < 0 or parsed_length > max_response_bytes:
            raise OSError("HTTPS response exceeded the byte limit")
