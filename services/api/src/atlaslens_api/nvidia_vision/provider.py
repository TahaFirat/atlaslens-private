"""Secret-safe NVIDIA provider and generic Pydantic JSON validation seam."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from .client import (
    NVIDIA_BASE_URL,
    NVIDIA_VISION_MODEL,
    NvidiaVisionClientConfig,
    NvidiaVisionTransport,
)
from .errors import (
    NvidiaAuthenticationError,
    NvidiaCancelledError,
    NvidiaCircuitOpenError,
    NvidiaConfigurationError,
    NvidiaModelUnavailableError,
    NvidiaProviderUnavailableError,
    NvidiaRateLimitError,
    NvidiaResponseError,
    NvidiaSchemaError,
    NvidiaTimeoutError,
    NvidiaTransportError,
)
from .preprocessing import PreparedImageSet

NvidiaProviderState = Literal[
    "disabled",
    "missing_credentials",
    "not_ready",
    "available",
    "rate_limited",
    "circuit_open",
    "failed",
]

class NvidiaCloudAuthorization(BaseModel):
    """Per-call privacy gate; construction alone can never authorize a transfer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["local_only", "cloud_assisted"]
    cloud_consent: bool


class NvidiaProviderStatus(BaseModel):
    """Secret-free capability state safe for API projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: NvidiaProviderState
    enabled: bool
    credentials_configured: bool
    model: str


@dataclass(frozen=True, slots=True)
class NvidiaStructuredResponse[ResponseT: BaseModel]:
    """Locally validated content plus non-sensitive upstream accounting."""

    request_id: str
    model: str
    value: ResponseT = field(repr=False)
    input_tokens: int | None = None
    output_tokens: int | None = None
    network_attempts: int = 1
    status_poll_attempts: int = 0

    def __repr__(self) -> str:
        return (
            "NvidiaStructuredResponse(request_id=<redacted>, "
            f"model={self.model!r}, value=<redacted>, "
            f"input_tokens={self.input_tokens!r}, output_tokens={self.output_tokens!r}, "
            f"network_attempts={self.network_attempts!r}, "
            f"status_poll_attempts={self.status_poll_attempts!r})"
        )


def _validated_secret(value: SecretStr | str | None) -> SecretStr | None:
    if value is None:
        return None
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    selected = raw.strip()
    placeholders = {
        "yeni_gercek_anahtar",
        "placeholder",
        "changeme",
        "your_nvidia_api_key",
    }
    if (
        not selected
        or selected.casefold() in placeholders
        or len(selected) < 8
        or any(character.isspace() for character in selected)
    ):
        return None
    return SecretStr(selected)


class NvidiaVisionProvider:
    """Optional cloud provider; construction and status checks never use the network."""

    def __init__(
        self,
        config: NvidiaVisionClientConfig,
        *,
        enabled: bool,
        api_key: SecretStr | str | None,
        http_client: httpx.AsyncClient | None = None,
        transport: NvidiaVisionTransport | None = None,
    ) -> None:
        self.config = config
        self._enabled = enabled
        self._secret = _validated_secret(api_key)
        self._state_lock = threading.Lock()
        self._terminal_failure: Literal["authentication", "model"] | None = None
        if not enabled:
            self._transport = None
            self._state: NvidiaProviderState = "disabled"
        elif self._secret is None:
            self._transport = None
            self._state = "missing_credentials"
        else:
            self._transport = transport or NvidiaVisionTransport(
                config,
                api_key=self._secret,
                http_client=http_client,
            )
            self._state = "not_ready"

    def __repr__(self) -> str:
        return (
            f"NvidiaVisionProvider(enabled={self._enabled!r}, model={self.config.model!r}, "
            f"state={self.status().state!r}, api_key=<redacted>)"
        )

    async def aclose(self) -> None:
        if self._transport is not None:
            await self._transport.aclose()

    def _set_state(self, state: NvidiaProviderState) -> None:
        with self._state_lock:
            self._state = state

    def status(self) -> NvidiaProviderStatus:
        with self._state_lock:
            state = self._state
        if (
            state == "circuit_open"
            and self._transport is not None
            and not self._transport.circuit_open
        ):
            state = "not_ready"
        return NvidiaProviderStatus(
            state=state,
            enabled=self._enabled,
            credentials_configured=self._secret is not None,
            model=self.config.model,
        )

    async def complete_json[ResponseT: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        authorization: NvidiaCloudAuthorization,
        images: PreparedImageSet | None = None,
        response_model: type[ResponseT],
        cancellation: asyncio.Event | None = None,
    ) -> NvidiaStructuredResponse[ResponseT]:
        """Perform one completion and validate it once, without semantic retry."""

        if not self._enabled:
            raise NvidiaProviderUnavailableError("nvidia_provider_disabled")
        if self._secret is None or self._transport is None:
            raise NvidiaProviderUnavailableError("nvidia_credentials_missing")
        if authorization.mode != "cloud_assisted":
            raise NvidiaProviderUnavailableError("nvidia_cloud_mode_required")
        if not authorization.cloud_consent:
            raise NvidiaProviderUnavailableError("nvidia_cloud_consent_required")
        if cancellation is not None and cancellation.is_set():
            raise NvidiaCancelledError("nvidia_request_cancelled")
        with self._state_lock:
            terminal_failure = self._terminal_failure
        if terminal_failure == "authentication":
            raise NvidiaAuthenticationError("nvidia_authentication_rejected")
        if terminal_failure == "model":
            raise NvidiaModelUnavailableError("nvidia_model_or_endpoint_unavailable")

        try:
            response = await self._transport.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=images,
                cancellation=cancellation,
            )
        except NvidiaRateLimitError:
            self._set_state("rate_limited")
            raise
        except NvidiaCircuitOpenError:
            self._set_state("circuit_open")
            raise
        except NvidiaModelUnavailableError:
            with self._state_lock:
                self._terminal_failure = "model"
            self._set_state("not_ready")
            raise
        except NvidiaCancelledError:
            raise
        except NvidiaAuthenticationError:
            with self._state_lock:
                self._terminal_failure = "authentication"
            self._set_state("failed")
            raise
        except (NvidiaTimeoutError, NvidiaTransportError, NvidiaResponseError):
            self._set_state("failed")
            raise

        try:
            parsed = response_model.model_validate_json(response.content)
        except (ValidationError, ValueError, TypeError):
            self._set_state("not_ready")
            raise NvidiaSchemaError("nvidia_structured_output_invalid") from None
        self._set_state("available")
        with self._state_lock:
            self._terminal_failure = None
        return NvidiaStructuredResponse(
            request_id=response.request_id,
            model=response.model,
            value=parsed,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            network_attempts=response.network_attempts,
            status_poll_attempts=response.status_poll_attempts,
        )


def build_nvidia_vision_provider(
    *,
    enabled: bool,
    api_key: SecretStr | str | None,
    base_url: str = NVIDIA_BASE_URL,
    model: str = NVIDIA_VISION_MODEL,
    total_timeout_seconds: float = 90,
    max_transport_retries: int = 0,
    http_client: httpx.AsyncClient | None = None,
) -> NvidiaVisionProvider:
    """Build the disabled-by-default provider without issuing a network request."""

    try:
        config = NvidiaVisionClientConfig(
            base_url=base_url,
            model=model,
            total_timeout_seconds=total_timeout_seconds,
            max_transport_retries=max_transport_retries,
        )
    except ValidationError as exc:
        raise NvidiaConfigurationError("nvidia_configuration_invalid") from exc
    return NvidiaVisionProvider(
        config,
        enabled=enabled,
        api_key=api_key,
        http_client=http_client,
    )
