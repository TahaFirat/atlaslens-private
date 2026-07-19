"""Bounded async transport for NVIDIA's OpenAI-compatible chat endpoint."""

from __future__ import annotations

import asyncio
import email.utils
import json
import math
import random
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from .errors import (
    NvidiaAuthenticationError,
    NvidiaCancelledError,
    NvidiaCircuitOpenError,
    NvidiaConfigurationError,
    NvidiaModelUnavailableError,
    NvidiaRateLimitError,
    NvidiaResponseError,
    NvidiaTimeoutError,
    NvidiaTransportError,
)
from .preprocessing import PreparedImageSet
from .profiles import (
    NVIDIA_ACTIVE_VISION_MODEL,
    NVIDIA_VISION_MODEL_PROFILES,
    NvidiaVisionModelProfile,
    get_nvidia_vision_model_profile,
)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Backward-compatible public name; it now identifies the recommended active profile.
NVIDIA_VISION_MODEL = NVIDIA_ACTIVE_VISION_MODEL

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_SENSITIVE_HEADER_NAMES = frozenset(
    {
        "authorization",
        "cookie",
        "proxy-authorization",
        "set-cookie",
        "x-api-key",
    }
)


def validate_nvidia_base_url(value: str) -> str:
    """Allow only the exact official HTTPS v1 API origin and path."""

    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError:
        raise NvidiaConfigurationError("nvidia_base_url_invalid") from None
    if (
        parsed.scheme.casefold() != "https"
        or (parsed.hostname or "").casefold() != "integrate.api.nvidia.com"
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise NvidiaConfigurationError("nvidia_base_url_invalid")
    return urlunsplit(("https", "integrate.api.nvidia.com", "/v1", "", ""))


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a safe diagnostic projection without authentication values."""

    return {
        name: (
            "[REDACTED]"
            if name.casefold() in _SENSITIVE_HEADER_NAMES
            or any(marker in name.casefold() for marker in ("secret", "token", "credential"))
            else value
        )
        for name, value in headers.items()
    }


class NvidiaVisionClientConfig(BaseModel):
    """Fail-closed transport policy; it intentionally contains no credential."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    base_url: str = NVIDIA_BASE_URL
    model: str = NVIDIA_VISION_MODEL
    connect_timeout_seconds: float = Field(default=10, gt=0, le=30)
    read_timeout_seconds: float = Field(default=75, gt=0, le=120)
    write_timeout_seconds: float = Field(default=20, gt=0, le=60)
    pool_timeout_seconds: float = Field(default=10, gt=0, le=30)
    total_timeout_seconds: float = Field(default=90, gt=0, le=120)
    max_output_tokens: int = Field(default=2_048, ge=128, le=4_096)
    response_byte_cap: int = Field(default=1_048_576, ge=16_384, le=2_097_152)
    request_byte_cap: int = Field(default=16_777_216, ge=1_048_576, le=16_777_216)
    concurrency_limit: Literal[1] = 1
    max_transport_retries: int = Field(default=0, ge=0, le=2)
    backoff_base_seconds: float = Field(default=0.5, ge=0, le=5)
    backoff_cap_seconds: float = Field(default=8, ge=0, le=30)
    circuit_failure_threshold: int = Field(default=3, ge=1, le=10)
    circuit_cooldown_seconds: float = Field(default=60, gt=0, le=600)
    status_poll_interval_seconds: float = Field(default=0.5, ge=0, le=5)
    max_status_polls: int = Field(default=120, ge=1, le=240)

    @field_validator("base_url")
    @classmethod
    def official_base_only(cls, value: str) -> str:
        return validate_nvidia_base_url(value)

    @field_validator("model")
    @classmethod
    def exact_model_only(cls, value: str) -> str:
        if value not in NVIDIA_VISION_MODEL_PROFILES:
            raise ValueError("nvidia vision model is not authorized")
        return value

    @model_validator(mode="after")
    def profile_bounds(self) -> NvidiaVisionClientConfig:
        profile = self.profile
        if self.total_timeout_seconds > profile.maximum_total_timeout_seconds:
            raise ValueError("nvidia vision timeout exceeds model profile maximum")
        if self.max_output_tokens > profile.max_output_tokens:
            raise ValueError("nvidia vision max tokens exceed model profile maximum")
        return self

    @property
    def profile(self) -> NvidiaVisionModelProfile:
        return get_nvidia_vision_model_profile(self.model)


@dataclass(frozen=True, slots=True)
class NvidiaVisionResponse:
    """Private raw completion envelope; model content is omitted from repr."""

    request_id: str
    model: str
    content: str = field(repr=False)
    input_tokens: int | None = None
    output_tokens: int | None = None
    network_attempts: int = 1
    status_poll_attempts: int = 0

    def __repr__(self) -> str:
        return (
            f"NvidiaVisionResponse(request_id=<redacted>, model={self.model!r}, "
            "content=<redacted>, "
            f"input_tokens={self.input_tokens!r}, output_tokens={self.output_tokens!r}, "
            f"network_attempts={self.network_attempts!r}, "
            f"status_poll_attempts={self.status_poll_attempts!r})"
        )


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


def _bounded_token(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10_000_000:
        raise NvidiaResponseError("nvidia_response_usage_invalid")
    return value


def _normalize_json_object_content(
    value: object,
    *,
    profile: NvidiaVisionModelProfile,
) -> str:
    """Accept one JSON object, with only the profile's narrow fence compatibility."""

    if not isinstance(value, str):
        raise NvidiaResponseError("nvidia_response_content_invalid")
    selected = value.strip()
    if not selected:
        raise NvidiaResponseError("nvidia_response_content_invalid")
    if profile.output_mode.allow_single_json_fence:
        lines = selected.splitlines()
        if (
            len(lines) >= 3
            and lines[0].strip() == "```json"
            and lines[-1].strip() == "```"
        ):
            selected = "\n".join(lines[1:-1]).strip()
    if not selected:
        raise NvidiaResponseError("nvidia_response_content_invalid")
    try:
        parsed = json.loads(selected)
    except (json.JSONDecodeError, RecursionError):
        raise NvidiaResponseError("nvidia_response_content_invalid") from None
    if not isinstance(parsed, dict):
        raise NvidiaResponseError("nvidia_response_content_invalid")
    return selected


class NvidiaVisionTransport:
    """One-at-a-time bounded transport with retry and process-local circuit state."""

    def __init__(
        self,
        config: NvidiaVisionClientConfig,
        *,
        api_key: SecretStr,
        http_client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        secret = api_key.get_secret_value()
        if not secret or len(secret) < 8 or any(character.isspace() for character in secret):
            raise NvidiaConfigurationError("nvidia_api_key_invalid")
        self.config = config
        self._api_key = api_key
        self._timeout = httpx.Timeout(
            connect=config.connect_timeout_seconds,
            read=config.read_timeout_seconds,
            write=config.write_timeout_seconds,
            pool=config.pool_timeout_seconds,
        )
        self._http = http_client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=self._timeout,
            trust_env=False,
        )
        self._owns_http = http_client is None
        self._sleep = sleep
        self._jitter = jitter
        self._monotonic = monotonic
        self._now = now
        self._semaphore = asyncio.Semaphore(config.concurrency_limit)
        self._circuit_lock = threading.Lock()
        self._failure_count = 0
        self._circuit_open_until: float | None = None

    def __repr__(self) -> str:
        return (
            f"NvidiaVisionTransport(model={self.config.model!r}, "
            f"base_url={self.config.base_url!r}, api_key=<redacted>)"
        )

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    @property
    def circuit_open(self) -> bool:
        with self._circuit_lock:
            open_until = self._circuit_open_until
            return open_until is not None and self._monotonic() < open_until

    def _check_circuit(self) -> None:
        with self._circuit_lock:
            open_until = self._circuit_open_until
            if open_until is None:
                return
            if self._monotonic() < open_until:
                raise NvidiaCircuitOpenError("nvidia_circuit_open")
            # Permit exactly one semaphore-serialized half-open probe.  A failed
            # probe immediately reopens the circuit; a success resets it.
            self._circuit_open_until = None
            self._failure_count = self.config.circuit_failure_threshold - 1

    def _record_success(self) -> None:
        with self._circuit_lock:
            self._failure_count = 0
            self._circuit_open_until = None

    def _record_transport_failure(self) -> bool:
        with self._circuit_lock:
            self._failure_count += 1
            if self._failure_count < self.config.circuit_failure_threshold:
                return False
            self._circuit_open_until = (
                self._monotonic() + self.config.circuit_cooldown_seconds
            )
            return True

    def _headers(self, request_id: str, *, has_body: bool = True) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "User-Agent": "AtlasLens-NVIDIA-Vision/1",
            "X-Request-ID": request_id,
        }
        if has_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _request_bytes(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: PreparedImageSet | None,
    ) -> bytes:
        if (
            not system_prompt.strip()
            or len(system_prompt) > 16_000
            or not user_prompt.strip()
            or len(user_prompt) > 64_000
            or "\x00" in system_prompt
            or "\x00" in user_prompt
        ):
            raise NvidiaConfigurationError("nvidia_prompt_bounds_invalid")
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if images is not None:
            if not 1 <= len(images.images) <= 4:
                raise NvidiaConfigurationError("nvidia_image_count_invalid")
            user_content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": image.data_url()},
                }
                for image in images.images
            )
        profile = self.config.profile
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.config.max_output_tokens,
            "chat_template_kwargs": {
                "enable_thinking": profile.output_mode.enable_thinking
            },
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "stream": False,
        }
        optional_sampling = {
            "top_k": profile.top_k,
            "presence_penalty": profile.presence_penalty,
            "repetition_penalty": profile.repetition_penalty,
            "seed": profile.seed,
        }
        payload.update(
            {name: value for name, value in optional_sampling.items() if value is not None}
        )
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > self.config.request_byte_cap:
            raise NvidiaConfigurationError("nvidia_request_byte_cap_reached")
        return encoded

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: PreparedImageSet | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> NvidiaVisionResponse:
        """Run one logical completion; retries cover transport failures only."""

        if cancellation is not None and cancellation.is_set():
            raise NvidiaCancelledError("nvidia_request_cancelled")
        request_id = f"nvr-{uuid4().hex[:16]}"
        request_bytes = self._request_bytes(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )
        operation = asyncio.create_task(
            self._bounded_complete(request_id=request_id, request_bytes=request_bytes)
        )
        cancel_waiter: asyncio.Task[bool] | None = None
        wait_for: set[asyncio.Task[Any]] = {operation}
        if cancellation is not None:
            cancel_waiter = asyncio.create_task(cancellation.wait())
            wait_for.add(cancel_waiter)
        try:
            done, pending = await asyncio.wait(
                wait_for,
                timeout=self.config.total_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            operation.cancel()
            if cancel_waiter is not None:
                cancel_waiter.cancel()
            await asyncio.gather(operation, *(wait_for - {operation}), return_exceptions=True)
            raise
        if (
            cancel_waiter is not None
            and cancel_waiter in done
            and cancellation is not None
            and cancellation.is_set()
        ):
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            for task in pending - {operation}:
                task.cancel()
            raise NvidiaCancelledError("nvidia_request_cancelled")
        if operation not in done:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            remaining = pending - {operation}
            for task in remaining:
                task.cancel()
            if remaining:
                await asyncio.gather(*remaining, return_exceptions=True)
            raise NvidiaTimeoutError("nvidia_total_timeout")
        if cancel_waiter is not None:
            cancel_waiter.cancel()
            await asyncio.gather(cancel_waiter, return_exceptions=True)
        return await operation

    async def _bounded_complete(
        self,
        *,
        request_id: str,
        request_bytes: bytes,
    ) -> NvidiaVisionResponse:
        async with self._semaphore:
            self._check_circuit()
            try:
                response = await self._request_with_retries(
                    request_id=request_id,
                    request_bytes=request_bytes,
                )
            except (NvidiaRateLimitError, NvidiaAuthenticationError, NvidiaModelUnavailableError):
                raise
            except NvidiaTransportError:
                if self._record_transport_failure():
                    raise NvidiaCircuitOpenError("nvidia_circuit_open") from None
                raise
            self._record_success()
            return response

    async def _request_with_retries(
        self,
        *,
        request_id: str,
        request_bytes: bytes,
    ) -> NvidiaVisionResponse:
        endpoint = f"{self.config.base_url}/chat/completions"
        network_attempts = 0
        for retry_number in range(self.config.max_transport_retries + 1):
            retry_status: int | None = None
            retry_headers: httpx.Headers | None = None
            try:
                network_attempts += 1
                async with self._http.stream(
                    "POST",
                    endpoint,
                    headers=self._headers(request_id),
                    content=request_bytes,
                    timeout=self._timeout,
                    follow_redirects=False,
                ) as response:
                    if response.status_code in _RETRYABLE_STATUS:
                        retry_status = response.status_code
                        retry_headers = response.headers
                    elif response.status_code == 202:
                        pending_id = await self._pending_request_id(response)
                        return await self._poll_pending(
                            pending_id=pending_id,
                            request_id=request_id,
                            network_attempts=network_attempts,
                            initial_retry_after=_retry_after_seconds(
                                response.headers.get("Retry-After"), self._now()
                            ),
                        )
                    else:
                        return await self._consume_response(
                            response,
                            request_id=request_id,
                            network_attempts=network_attempts,
                            status_poll_attempts=0,
                        )
            except httpx.TimeoutException:
                if retry_number >= self.config.max_transport_retries:
                    raise NvidiaTimeoutError("nvidia_transport_timeout") from None
                await self._backoff(retry_number, retry_after=None)
                continue
            except httpx.TransportError:
                if retry_number >= self.config.max_transport_retries:
                    raise NvidiaTransportError("nvidia_transport_retry_exhausted") from None
                await self._backoff(retry_number, retry_after=None)
                continue

            if retry_status is None:
                raise NvidiaTransportError("nvidia_transport_failed")
            retry_after = _retry_after_seconds(
                retry_headers.get("Retry-After") if retry_headers is not None else None,
                self._now(),
            )
            if retry_number >= self.config.max_transport_retries:
                if retry_status == 429:
                    raise NvidiaRateLimitError(
                        "nvidia_rate_limit_retry_exhausted",
                        retry_after_seconds=(
                            min(retry_after, self.config.backoff_cap_seconds)
                            if retry_after is not None
                            else None
                        ),
                    )
                if retry_status == 408:
                    raise NvidiaTimeoutError("nvidia_upstream_timeout")
                raise NvidiaTransportError("nvidia_upstream_server_error")
            await self._backoff(retry_number, retry_after=retry_after)
        raise NvidiaTransportError("nvidia_transport_retry_exhausted")

    async def _pending_request_id(self, response: httpx.Response) -> UUID:
        value = response.headers.get("NVCF-REQID")
        if value is None:
            payload = await self._read_json_payload(response)
            if not isinstance(payload, dict):
                raise NvidiaResponseError("nvidia_pending_envelope_invalid")
            value = payload.get("requestId")
        return self._parse_pending_request_id(value)

    @staticmethod
    def _parse_pending_request_id(value: object) -> UUID:
        if not isinstance(value, str) or len(value) > 36:
            raise NvidiaResponseError("nvidia_pending_request_id_invalid")
        try:
            return UUID(value)
        except ValueError:
            raise NvidiaResponseError("nvidia_pending_request_id_invalid") from None

    async def _poll_pending(
        self,
        *,
        pending_id: UUID,
        request_id: str,
        network_attempts: int,
        initial_retry_after: float | None,
    ) -> NvidiaVisionResponse:
        endpoint = f"{self.config.base_url}/status/{pending_id}"
        delay = (
            min(initial_retry_after, self.config.backoff_cap_seconds)
            if initial_retry_after is not None
            else self.config.status_poll_interval_seconds
        )
        for poll_number in range(1, self.config.max_status_polls + 1):
            if delay > 0:
                await self._sleep(delay)
            try:
                network_attempts += 1
                async with self._http.stream(
                    "GET",
                    endpoint,
                    headers=self._headers(request_id, has_body=False),
                    timeout=self._timeout,
                    follow_redirects=False,
                ) as response:
                    if response.status_code == 202:
                        repeated_value = response.headers.get("NVCF-REQID")
                        if repeated_value is not None and (
                            self._parse_pending_request_id(repeated_value) != pending_id
                        ):
                            raise NvidiaResponseError("nvidia_pending_request_id_mismatch")
                        retry_after = _retry_after_seconds(
                            response.headers.get("Retry-After"), self._now()
                        )
                        delay = (
                            min(retry_after, self.config.backoff_cap_seconds)
                            if retry_after is not None
                            else self.config.status_poll_interval_seconds
                        )
                        continue
                    return await self._consume_response(
                        response,
                        request_id=request_id,
                        network_attempts=network_attempts,
                        status_poll_attempts=poll_number,
                    )
            except httpx.TimeoutException:
                raise NvidiaTimeoutError("nvidia_status_poll_transport_timeout") from None
            except httpx.TransportError:
                raise NvidiaTransportError("nvidia_status_poll_transport_failed") from None
        raise NvidiaTimeoutError("nvidia_status_poll_timeout")

    async def _backoff(self, retry_number: int, *, retry_after: float | None) -> None:
        if retry_after is None:
            base = min(
                self.config.backoff_cap_seconds,
                self.config.backoff_base_seconds * (2**retry_number),
            )
            delay = base + self._jitter(0.0, base * 0.25)
        else:
            delay = retry_after
        await self._sleep(min(max(delay, 0.0), self.config.backoff_cap_seconds))

    async def _consume_response(
        self,
        response: httpx.Response,
        *,
        request_id: str,
        network_attempts: int,
        status_poll_attempts: int,
    ) -> NvidiaVisionResponse:
        if response.is_redirect:
            raise NvidiaTransportError("nvidia_redirect_refused")
        if response.status_code in {401, 403}:
            raise NvidiaAuthenticationError("nvidia_authentication_rejected")
        if response.status_code == 404:
            raise NvidiaModelUnavailableError("nvidia_model_or_endpoint_unavailable")
        if response.status_code in {413, 422}:
            raise NvidiaTransportError("nvidia_upstream_payload_rejected")
        if response.status_code >= 500:
            raise NvidiaTransportError("nvidia_upstream_server_error")
        if not 200 <= response.status_code < 300:
            raise NvidiaTransportError("nvidia_upstream_request_rejected")

        payload = await self._read_json_payload(response)
        return self._parse_envelope(
            payload,
            request_id=request_id,
            network_attempts=network_attempts,
            status_poll_attempts=status_poll_attempts,
        )

    async def _read_json_payload(self, response: httpx.Response) -> object:
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
        if content_type != "application/json":
            raise NvidiaResponseError("nvidia_response_content_type_invalid")
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError:
                raise NvidiaResponseError("nvidia_response_length_invalid") from None
            if declared_size < 0:
                raise NvidiaResponseError("nvidia_response_length_invalid")
            if declared_size > self.config.response_byte_cap:
                raise NvidiaResponseError("nvidia_response_byte_cap_reached")
        chunks: list[bytes] = []
        received = 0
        async for chunk in response.aiter_bytes():
            received += len(chunk)
            if received > self.config.response_byte_cap:
                raise NvidiaResponseError("nvidia_response_byte_cap_reached")
            chunks.append(chunk)
        try:
            return json.loads(b"".join(chunks))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise NvidiaResponseError("nvidia_response_json_invalid") from None

    def _parse_envelope(
        self,
        payload: object,
        *,
        request_id: str,
        network_attempts: int,
        status_poll_attempts: int,
    ) -> NvidiaVisionResponse:
        if not isinstance(payload, dict):
            raise NvidiaResponseError("nvidia_response_envelope_invalid")
        model = payload.get("model")
        if model != self.config.model:
            raise NvidiaResponseError("nvidia_response_model_mismatch")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise NvidiaResponseError("nvidia_response_choices_invalid")
        first = choices[0]
        if first.get("finish_reason") == "length":
            raise NvidiaResponseError("nvidia_response_truncated")
        if first.get("finish_reason") != "stop":
            raise NvidiaResponseError("nvidia_response_finish_reason_invalid")
        message = first.get("message")
        if not isinstance(message, dict):
            raise NvidiaResponseError("nvidia_response_message_invalid")
        content = _normalize_json_object_content(
            message.get("content"),
            profile=self.config.profile,
        )
        usage = payload.get("usage")
        input_tokens: int | None = None
        output_tokens: int | None = None
        if usage is not None:
            if not isinstance(usage, dict):
                raise NvidiaResponseError("nvidia_response_usage_invalid")
            input_tokens = _bounded_token(usage.get("prompt_tokens"))
            output_tokens = _bounded_token(usage.get("completion_tokens"))
        return NvidiaVisionResponse(
            request_id=request_id,
            model=model,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            network_attempts=network_attempts,
            status_poll_attempts=status_poll_attempts,
        )
