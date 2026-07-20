"""Official Mapillary Graph API v4 client with hard pilot safety limits."""

from __future__ import annotations

import email.utils
import hashlib
import json
import math
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import SecretStr, ValidationError

from .errors import (
    MapillaryApiError,
    MapillaryLimitError,
    MapillarySafetyError,
    MapillaryTokenError,
)
from .models import (
    MAPILLARY_API_BASE_URL,
    MAPILLARY_API_FIELDS,
    BoundingBox,
    ClientLimits,
    GeoPoint,
    ImageMetadata,
)

_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "access_token",
        "appsecret_proof",
        "authorization",
        "client_secret",
        "sig",
        "signature",
        "token",
    }
)
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_ALLOWED_GRAPH_PATHS = frozenset({"/images", "/images/"})
_DEFAULT_MEDIA_HOST_SUFFIXES = (".fbcdn.net", ".fbsbx.com")


@dataclass(frozen=True)
class RemoteImage:
    """API image metadata plus an intentionally ephemeral signed media URL."""

    metadata: ImageMetadata
    thumbnail_url: SecretStr | None = field(default=None, repr=False)


@dataclass(frozen=True)
class PageResult:
    images: tuple[RemoteImage, ...]
    next_url: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class PaginationProgress:
    """Secret-free completed-page state for an external private checkpoint."""

    box_index: int
    next_url: str | None = field(default=None, repr=False)
    images: tuple[RemoteImage, ...] = ()
    visited_page_sha256: tuple[str, ...] = ()


TokenCheckStatus = Literal["configured", "invalid", "unavailable"]


def validate_access_token(value: str | None) -> SecretStr:
    """Validate a runtime value without returning or embedding it in an error."""

    if value is None:
        raise MapillaryTokenError("mapillary_token_not_configured")
    selected = value.strip()
    if not selected or selected.casefold() == "aldigin_token":
        raise MapillaryTokenError("mapillary_token_not_configured")
    if len(selected) < 8 or any(character.isspace() for character in selected):
        raise MapillaryTokenError("mapillary_token_invalid")
    return SecretStr(selected)


def redact_url(value: str) -> str:
    """Return a URL that is safe for structured diagnostics."""

    try:
        parsed = urlsplit(value)
        parsed_port = parsed.port
    except ValueError:
        return "[invalid-url]"
    query = [
        (key, "[REDACTED]" if key.casefold() in _SENSITIVE_QUERY_KEYS else item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    hostname = parsed.hostname or ""
    port = f":{parsed_port}" if parsed_port is not None else ""
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, urlencode(query), ""))


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Project headers without credentials."""

    return {
        key: "[REDACTED]" if key.casefold() in {"authorization", "proxy-authorization"} else value
        for key, value in headers.items()
    }


def validated_next_url(value: str) -> str:
    """Validate official paging.next and strip any embedded credential."""

    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise MapillarySafetyError("mapillary_paging_url_invalid") from None
    if (
        parsed.scheme not in {"", "https"}
        or (parsed.scheme == "https" and parsed.hostname != "graph.mapillary.com")
        or (parsed.scheme == "" and (parsed.hostname is not None or parsed.netloc))
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.path not in _ALLOWED_GRAPH_PATHS
    ):
        raise MapillarySafetyError("mapillary_paging_url_invalid")
    query: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        normalized = key.casefold()
        if normalized in _SENSITIVE_QUERY_KEYS:
            continue
        if normalized in seen:
            raise MapillarySafetyError("mapillary_paging_query_duplicate")
        seen.add(normalized)
        query.append((key, item))
    return urlunsplit(
        ("https", "graph.mapillary.com", parsed.path, urlencode(sorted(query)), "")
    )


def _official_graph_url(path_or_url: str) -> str:
    if path_or_url.startswith("/"):
        if path_or_url not in _ALLOWED_GRAPH_PATHS and not (
            len(path_or_url) > 1 and path_or_url[1:].replace("_", "").replace("-", "").isalnum()
        ):
            raise MapillarySafetyError("mapillary_graph_path_invalid")
        return f"{MAPILLARY_API_BASE_URL}{path_or_url}"
    return validated_next_url(path_or_url)


def _retry_after_seconds(value: str | None, now: datetime) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        seconds = (parsed - now).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


class MapillaryClient:
    """Small synchronous client; httpx.Client is safe for bounded worker sharing."""

    def __init__(
        self,
        access_token: str,
        *,
        limits: ClientLimits | None = None,
        http_client: httpx.Client | None = None,
        cancel_event: threading.Event | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        media_host_suffixes: Sequence[str] = _DEFAULT_MEDIA_HOST_SUFFIXES,
        counter_observer: Callable[[int, int, int], None] | None = None,
    ) -> None:
        self._token = validate_access_token(access_token)
        self.limits = limits or ClientLimits()
        self._http = http_client or httpx.Client(
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_http = http_client is None
        self._cancel_event = cancel_event or threading.Event()
        self._sleep = sleep
        self._jitter = jitter
        self._now = now
        self._media_host_suffixes = tuple(suffix.casefold() for suffix in media_host_suffixes)
        self._counter_observer = counter_observer
        if not self._media_host_suffixes or any(
            not suffix.startswith(".") for suffix in self._media_host_suffixes
        ):
            raise MapillarySafetyError("mapillary_media_host_policy_invalid")
        self._lock = threading.Lock()
        self._request_count = 0
        self._page_count = 0
        self._rejected_item_count = 0

    def __enter__(self) -> MapillaryClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    @property
    def page_count(self) -> int:
        with self._lock:
            return self._page_count

    @property
    def rejected_item_count(self) -> int:
        with self._lock:
            return self._rejected_item_count

    def add_historical_counts(
        self,
        *,
        request_count: int,
        page_count: int,
        rejected_item_count: int,
    ) -> None:
        """Conservatively add validated prior-process counters before resuming."""

        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (request_count, page_count, rejected_item_count)
        ):
            raise MapillaryLimitError("mapillary_resume_counter_invalid")
        with self._lock:
            if self._request_count + request_count > self.limits.request_cap:
                raise MapillaryLimitError("mapillary_request_cap_reached")
            if self._page_count + page_count > self.limits.page_cap:
                raise MapillaryLimitError("mapillary_page_cap_reached")
            self._request_count += request_count
            self._page_count += page_count
            self._rejected_item_count += rejected_item_count

    def _reserve_request(self) -> None:
        with self._lock:
            if self._request_count >= self.limits.request_cap:
                raise MapillaryLimitError("mapillary_request_cap_reached")
            self._request_count += 1
            counters = (
                self._request_count,
                self._page_count,
                self._rejected_item_count,
            )
        if self._counter_observer is not None:
            self._counter_observer(*counters)

    def _reserve_page(self) -> None:
        with self._lock:
            if self._page_count >= self.limits.page_cap:
                raise MapillaryLimitError("mapillary_page_cap_reached")
            self._page_count += 1
            counters = (
                self._request_count,
                self._page_count,
                self._rejected_item_count,
            )
        if self._counter_observer is not None:
            self._counter_observer(*counters)

    def _require_page_capacity(self) -> None:
        with self._lock:
            if self._page_count >= self.limits.page_cap:
                raise MapillaryLimitError("mapillary_page_cap_reached")

    def _record_rejected_item(self) -> None:
        with self._lock:
            self._rejected_item_count += 1
            counters = (
                self._request_count,
                self._page_count,
                self._rejected_item_count,
            )
        if self._counter_observer is not None:
            self._counter_observer(*counters)

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise MapillaryLimitError("mapillary_operation_cancelled")

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"OAuth {self._token.get_secret_value()}",
            "User-Agent": "AtlasLens-Private-Mapillary-Pilot/1",
        }

    def _delay(self, response: httpx.Response, retry_number: int) -> None:
        retry_after = _retry_after_seconds(response.headers.get("Retry-After"), self._now())
        if retry_after is None:
            base = min(
                self.limits.backoff_cap_seconds,
                self.limits.backoff_base_seconds * (2**retry_number),
            )
            retry_after = base + self._jitter(0.0, base * 0.25)
        delay = min(retry_after, self.limits.backoff_cap_seconds)
        self._sleep(delay)
        self._check_cancelled()

    def _get(
        self,
        path_or_url: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> httpx.Response:
        if not path_or_url.startswith("/") and params is not None:
            raise MapillarySafetyError("mapillary_paging_params_duplicate")
        url = _official_graph_url(path_or_url)
        for retry_number in range(self.limits.retry_cap + 1):
            self._check_cancelled()
            self._reserve_request()
            retry_response: httpx.Response | None = None
            bounded_response: httpx.Response | None = None
            try:
                with self._http.stream(
                    "GET",
                    url,
                    params=params,
                    headers=self._auth_headers(),
                    timeout=self.limits.timeout_seconds,
                ) as streamed:
                    if streamed.status_code in _RETRYABLE_STATUS:
                        if retry_number >= self.limits.retry_cap:
                            code = (
                                "mapillary_api_rate_limit_retry_exhausted"
                                if streamed.status_code == 429
                                else "mapillary_api_server_retry_exhausted"
                            )
                            retry_after = (
                                _retry_after_seconds(
                                    streamed.headers.get("Retry-After"), self._now()
                                )
                                if streamed.status_code == 429
                                else None
                            )
                            raise MapillaryApiError(
                                code,
                                retry_after_seconds=retry_after,
                            )
                        retry_response = httpx.Response(
                            streamed.status_code,
                            headers=streamed.headers,
                        )
                    else:
                        if streamed.status_code == 401:
                            raise MapillaryTokenError("mapillary_token_rejected")
                        if streamed.status_code == 403:
                            raise MapillaryApiError("mapillary_permission_denied")
                        if streamed.status_code == 400:
                            raise MapillaryApiError("mapillary_api_bad_request")
                        if streamed.status_code == 404:
                            raise MapillaryApiError("mapillary_image_not_found")
                        if streamed.is_redirect:
                            raise MapillarySafetyError("mapillary_graph_redirect_refused")
                        if not 200 <= streamed.status_code < 300:
                            raise MapillaryApiError("mapillary_api_request_failed")
                        content_length = streamed.headers.get("Content-Length")
                        if content_length is not None:
                            try:
                                declared_size = int(content_length)
                            except ValueError:
                                raise MapillaryApiError("mapillary_api_header_invalid") from None
                            if declared_size < 0:
                                raise MapillaryApiError("mapillary_api_header_invalid")
                            if declared_size > self.limits.response_byte_cap:
                                raise MapillaryLimitError("mapillary_response_byte_cap_reached")
                        chunks: list[bytes] = []
                        size = 0
                        for chunk in streamed.iter_bytes():
                            self._check_cancelled()
                            size += len(chunk)
                            if size > self.limits.response_byte_cap:
                                raise MapillaryLimitError("mapillary_response_byte_cap_reached")
                            chunks.append(chunk)
                        decoded_headers = [
                            (key, value)
                            for key, value in streamed.headers.multi_items()
                            if key.casefold()
                            not in {"content-encoding", "content-length", "transfer-encoding"}
                        ]
                        bounded_response = httpx.Response(
                            streamed.status_code,
                            headers=decoded_headers,
                            content=b"".join(chunks),
                            request=streamed.request,
                        )
            except httpx.TimeoutException:
                if retry_number >= self.limits.retry_cap:
                    raise MapillaryApiError("mapillary_api_timeout") from None
                synthetic = httpx.Response(503)
                self._delay(synthetic, retry_number)
                continue
            except (httpx.NetworkError, httpx.RemoteProtocolError):
                if retry_number >= self.limits.retry_cap:
                    raise MapillaryApiError(
                        "mapillary_api_transport_retry_exhausted"
                    ) from None
                self._delay(httpx.Response(503), retry_number)
                continue
            except httpx.HTTPError:
                raise MapillaryApiError("mapillary_api_transport_failed") from None
            if retry_response is not None:
                self._delay(retry_response, retry_number)
                continue
            if bounded_response is None:
                raise MapillaryApiError("mapillary_api_request_failed")
            return bounded_response
        raise MapillaryApiError("mapillary_api_retry_exhausted")

    def _json_object(
        self,
        path_or_url: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> dict[str, Any]:
        response = self._get(path_or_url, params=params)
        try:
            value = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MapillaryApiError("mapillary_api_json_invalid") from exc
        if not isinstance(value, dict):
            raise MapillaryApiError("mapillary_api_json_invalid")
        return value

    def check_token(self) -> TokenCheckStatus:
        """Return status only; no token fragment or server body can escape."""

        try:
            self._json_object(
                "/images",
                params={
                    "bbox": "35.4400,38.7200,35.4410,38.7210",
                    "fields": "id",
                    "limit": 1,
                },
            )
        except MapillaryTokenError:
            return "invalid"
        except (MapillaryApiError, MapillaryLimitError):
            return "unavailable"
        return "configured"

    def iter_images(
        self,
        boxes: Sequence[BoundingBox],
        *,
        include_thumbnail: bool,
        item_cap: int | None = None,
        resume_box_index: int = 0,
        resume_next_url: str | None = None,
        resume_seen_image_ids: Sequence[str] = (),
        resume_visited_page_sha256: Sequence[str] = (),
        page_observer: Callable[[PaginationProgress], None] | None = None,
    ) -> Iterator[RemoteImage]:
        """Yield deduplicated images from bounded boxes and validated pagination."""

        requested_cap = self.limits.metadata_item_cap if item_cap is None else item_cap
        selected_cap = min(requested_cap, self.limits.metadata_item_cap)
        if selected_cap <= 0:
            raise MapillaryLimitError("mapillary_item_cap_invalid")
        if not 0 <= resume_box_index <= len(boxes):
            raise MapillarySafetyError("mapillary_resume_box_invalid")
        if resume_next_url is not None and resume_box_index >= len(boxes):
            raise MapillarySafetyError("mapillary_resume_page_invalid")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in resume_visited_page_sha256
        ):
            raise MapillarySafetyError("mapillary_resume_page_invalid")
        emitted = len(resume_seen_image_ids)
        if emitted > selected_cap:
            raise MapillaryLimitError("mapillary_item_cap_reached")
        seen = set(resume_seen_image_ids)
        if len(seen) != emitted:
            raise MapillarySafetyError("mapillary_resume_image_duplicate")
        fields = MAPILLARY_API_FIELDS if include_thumbnail else MAPILLARY_API_FIELDS[:-1]
        for box_index, box in enumerate(boxes):
            if box_index < resume_box_index:
                continue
            resumed_box = box_index == resume_box_index
            next_url = (
                validated_next_url(resume_next_url)
                if resumed_box and resume_next_url is not None
                else None
            )
            visited_pages = set(resume_visited_page_sha256 if resumed_box else ())
            params: Mapping[str, str | int] | None = None
            if next_url is None:
                params = {
                    "bbox": box.as_query_value(),
                    "fields": ",".join(fields),
                    "limit": self.limits.page_size,
                }
            while True:
                self._require_page_capacity()
                payload = self._json_object(next_url or "/images", params=params)
                self._reserve_page()
                params = None
                page = self._parse_page(payload, include_thumbnail=include_thumbnail)
                page_images: list[RemoteImage] = []
                for image in page.images:
                    image_id = image.metadata.mapillary_image_id
                    if image_id in seen:
                        continue
                    if emitted >= selected_cap:
                        raise MapillaryLimitError("mapillary_item_cap_reached")
                    seen.add(image_id)
                    emitted += 1
                    page_images.append(image)
                next_box_index = box_index + 1
                next_visited: set[str] = set()
                if page.next_url is not None:
                    identity = hashlib.sha256(page.next_url.encode("utf-8")).hexdigest()
                    if identity in visited_pages:
                        raise MapillarySafetyError("mapillary_paging_loop_detected")
                    visited_pages.add(identity)
                    next_box_index = box_index
                    next_visited = visited_pages
                if page_observer is not None:
                    page_observer(
                        PaginationProgress(
                            box_index=next_box_index,
                            next_url=page.next_url,
                            images=tuple(page_images),
                            visited_page_sha256=tuple(sorted(next_visited)),
                        )
                    )
                yield from page_images
                if page.next_url is None:
                    break
                next_url = page.next_url

    def _parse_page(self, payload: Mapping[str, Any], *, include_thumbnail: bool) -> PageResult:
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise MapillaryApiError("mapillary_api_page_invalid")
        images: list[RemoteImage] = []
        for row in rows:
            if not isinstance(row, dict):
                self._record_rejected_item()
                continue
            try:
                images.append(_parse_remote_image(row, include_thumbnail=include_thumbnail))
            except (KeyError, TypeError, ValueError, ValidationError):
                self._record_rejected_item()
                continue
        paging = payload.get("paging")
        next_url: str | None = None
        if paging is not None:
            if not isinstance(paging, dict):
                raise MapillaryApiError("mapillary_api_page_invalid")
            value = paging.get("next")
            if value is not None:
                if not isinstance(value, str):
                    raise MapillaryApiError("mapillary_api_page_invalid")
                next_url = validated_next_url(value)
        return PageResult(images=tuple(images), next_url=next_url)

    def image_exists(self, image_id: str) -> bool:
        if not image_id or not image_id.replace("_", "").replace("-", "").isalnum():
            raise MapillarySafetyError("mapillary_image_id_invalid")
        try:
            payload = self._json_object(f"/{image_id}", params={"fields": "id"})
        except MapillaryApiError as exc:
            if exc.code == "mapillary_image_not_found":
                return False
            raise
        return payload.get("id") == image_id

    def download_thumbnail(self, signed_url: str, *, max_bytes: int) -> tuple[bytes, str]:
        """Download an ephemeral signed image without sending Graph authorization."""

        url = self._validated_media_url(signed_url)
        selected_cap = min(max_bytes, self.limits.max_image_bytes)
        if selected_cap <= 0:
            raise MapillaryLimitError("mapillary_image_byte_cap_invalid")
        for retry_number in range(self.limits.retry_cap + 1):
            self._check_cancelled()
            self._reserve_request()
            try:
                with self._http.stream(
                    "GET",
                    url,
                    headers={
                        "Accept": "image/jpeg,image/png,image/webp",
                        "User-Agent": "AtlasLens-Private-Mapillary-Pilot/1",
                    },
                    timeout=self.limits.timeout_seconds,
                ) as response:
                    if response.status_code in _RETRYABLE_STATUS:
                        if retry_number >= self.limits.retry_cap:
                            code = (
                                "mapillary_media_rate_limit_retry_exhausted"
                                if response.status_code == 429
                                else "mapillary_media_server_retry_exhausted"
                            )
                            retry_after = (
                                _retry_after_seconds(
                                    response.headers.get("Retry-After"), self._now()
                                )
                                if response.status_code == 429
                                else None
                            )
                            raise MapillaryApiError(
                                code,
                                retry_after_seconds=retry_after,
                            )
                        self._delay(response, retry_number)
                        continue
                    if response.is_redirect:
                        raise MapillarySafetyError("mapillary_media_redirect_refused")
                    if response.status_code in {401, 403}:
                        raise MapillaryTokenError("mapillary_media_authorization_rejected")
                    if response.status_code in {404, 410}:
                        raise MapillaryApiError("mapillary_media_unavailable")
                    if not 200 <= response.status_code < 300:
                        raise MapillaryApiError("mapillary_media_request_failed")
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            if int(content_length) > selected_cap:
                                raise MapillaryLimitError("mapillary_image_byte_cap_reached")
                        except ValueError as exc:
                            raise MapillaryApiError("mapillary_media_header_invalid") from exc
                    media_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
                        raise MapillaryApiError("mapillary_media_type_invalid")
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        self._check_cancelled()
                        size += len(chunk)
                        if size > selected_cap:
                            raise MapillaryLimitError("mapillary_image_byte_cap_reached")
                        chunks.append(chunk)
                    if size == 0:
                        raise MapillaryApiError("mapillary_media_empty")
                    return b"".join(chunks), media_type
            except httpx.TimeoutException:
                if retry_number >= self.limits.retry_cap:
                    raise MapillaryApiError("mapillary_media_timeout") from None
                self._delay(httpx.Response(503), retry_number)
            except (httpx.NetworkError, httpx.RemoteProtocolError):
                if retry_number >= self.limits.retry_cap:
                    raise MapillaryApiError(
                        "mapillary_media_transport_retry_exhausted"
                    ) from None
                self._delay(httpx.Response(503), retry_number)
            except httpx.HTTPError:
                raise MapillaryApiError("mapillary_media_transport_failed") from None
        raise MapillaryApiError("mapillary_media_retry_exhausted")

    def _validated_media_url(self, value: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            raise MapillarySafetyError("mapillary_media_url_invalid") from None
        hostname = (parsed.hostname or "").casefold()
        if (
            parsed.scheme != "https"
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or port not in {None, 443}
            or parsed.fragment
            or not any(hostname.endswith(suffix) for suffix in self._media_host_suffixes)
        ):
            raise MapillarySafetyError("mapillary_media_url_invalid")
        return value


def _object_identifier(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        identifier = value.get("id")
        if isinstance(identifier, str):
            return identifier
    raise ValueError("object identifier is invalid")


def _captured_at(value: object) -> datetime:
    if isinstance(value, bool):
        raise ValueError("capture timestamp is invalid")
    if isinstance(value, int | float):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1_000.0
        return datetime.fromtimestamp(seconds, tz=UTC)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("capture timestamp is naive")
        return parsed
    raise ValueError("capture timestamp is invalid")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("dimension is invalid")
    return value


def _parse_remote_image(row: Mapping[str, Any], *, include_thumbnail: bool) -> RemoteImage:
    geometry = row["computed_geometry"]
    if not isinstance(geometry, dict) or geometry.get("type") != "Point":
        raise ValueError("computed geometry is invalid")
    coordinates = geometry.get("coordinates")
    if (
        not isinstance(coordinates, list | tuple)
        or len(coordinates) < 2
        or isinstance(coordinates[0], bool)
        or isinstance(coordinates[1], bool)
        or not isinstance(coordinates[0], int | float)
        or not isinstance(coordinates[1], int | float)
    ):
        raise ValueError("computed geometry coordinates are invalid")
    compass = row.get("computed_compass_angle", row.get("compass_angle"))
    if compass is not None and (isinstance(compass, bool) or not isinstance(compass, int | float)):
        raise ValueError("compass angle is invalid")
    metadata = ImageMetadata(
        mapillary_image_id=str(row["id"]),
        computed_geometry=GeoPoint(coordinates=(float(coordinates[0]), float(coordinates[1]))),
        captured_at=_captured_at(row["captured_at"]),
        compass_angle=float(compass) % 360.0 if compass is not None else None,
        sequence_id=_object_identifier(row.get("sequence")),
        creator_id=_object_identifier(row.get("creator")),
        width_px=_optional_int(row.get("width")),
        height_px=_optional_int(row.get("height")),
    )
    thumbnail: SecretStr | None = None
    if include_thumbnail:
        value = row.get("thumb_1024_url")
        if not isinstance(value, str) or not value:
            raise ValueError("thumbnail URL is missing")
        thumbnail = SecretStr(value)
    return RemoteImage(metadata=metadata, thumbnail_url=thumbnail)


__all__ = [
    "MapillaryClient",
    "PageResult",
    "PaginationProgress",
    "RemoteImage",
    "TokenCheckStatus",
    "redact_headers",
    "redact_url",
    "validate_access_token",
    "validated_next_url",
]
