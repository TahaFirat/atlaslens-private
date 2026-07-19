from __future__ import annotations

import io
import json
from typing import Any

import httpx
import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from atlaslens_api.nvidia_vision.client import (
    NVIDIA_VISION_MODEL,
    NvidiaVisionClientConfig,
    NvidiaVisionTransport,
    redact_headers,
)
from atlaslens_api.nvidia_vision.errors import (
    NvidiaAuthenticationError,
    NvidiaConfigurationError,
    NvidiaModelUnavailableError,
    NvidiaRateLimitError,
    NvidiaTimeoutError,
    NvidiaTransportError,
)
from atlaslens_api.nvidia_vision.preprocessing import PreparedImageSet, prepare_nvidia_images
from atlaslens_api.nvidia_vision.profiles import (
    NVIDIA_ACTIVE_VISION_MODEL,
    NVIDIA_DEPRECATED_VISION_MODEL,
    get_nvidia_vision_model_profile,
)
from atlaslens_api.nvidia_vision.provider import (
    NvidiaCloudAuthorization,
    NvidiaVisionProvider,
)

_TEST_KEY = "test-only-nvidia-profile-key"


class _ProbeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result: str


def _synthetic_images() -> PreparedImageSet:
    image = Image.new("RGB", (32, 32), color=(12, 34, 56))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=75)
    return prepare_nvidia_images(
        output.getvalue(),
        declared_media_type="image/jpeg",
        maximum_edge=256,
        crop_kinds=("full",),
    )


def _completion(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": '{"result":"ok"}'},
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }


def _authorization() -> NvidiaCloudAuthorization:
    return NvidiaCloudAuthorization(mode="cloud_assisted", cloud_consent=True)


def test_profiles_default_active_and_keep_deprecated_model_explicit() -> None:
    active = get_nvidia_vision_model_profile(NVIDIA_ACTIVE_VISION_MODEL)
    deprecated = get_nvidia_vision_model_profile(NVIDIA_DEPRECATED_VISION_MODEL)
    default_config = NvidiaVisionClientConfig()
    legacy_config = NvidiaVisionClientConfig(model=NVIDIA_DEPRECATED_VISION_MODEL)

    assert NVIDIA_VISION_MODEL == NVIDIA_ACTIVE_VISION_MODEL
    assert default_config.model == "qwen/qwen3.5-122b-a10b"
    assert default_config.profile is active
    assert active.recommended is True
    assert active.deprecated is False
    assert active.temperature == 0.7
    assert active.top_p == 0.8
    assert active.seed == 0
    assert active.output_mode.name == "strict_json_object_or_single_json_fence"
    assert active.output_mode.enable_thinking is False
    assert active.output_mode.allow_single_json_fence is True
    assert active.output_mode.use_reasoning_content is False
    assert legacy_config.profile is deprecated
    assert deprecated.recommended is False
    assert deprecated.deprecated is True
    assert deprecated.temperature == 0.7
    assert deprecated.top_p == 0.8
    assert deprecated.seed is None
    assert deprecated.output_mode.name == "strict_json_object"
    assert deprecated.output_mode.enable_thinking is False
    assert deprecated.output_mode.allow_single_json_fence is False
    assert deprecated.output_mode.use_reasoning_content is False


def test_profiles_fail_closed_and_bind_operator_bounds() -> None:
    config = NvidiaVisionClientConfig()

    assert config.max_transport_retries == 0
    assert config.concurrency_limit == 1
    assert config.total_timeout_seconds <= 120
    assert config.total_timeout_seconds <= config.profile.maximum_total_timeout_seconds
    assert config.max_output_tokens <= config.profile.max_output_tokens

    with pytest.raises(ValidationError, match="not authorized"):
        NvidiaVisionClientConfig(model="synthetic/unknown-vision-model")
    with pytest.raises(NvidiaConfigurationError, match="nvidia_vision_model_not_authorized"):
        get_nvidia_vision_model_profile("synthetic/unknown-vision-model")
    with pytest.raises(ValidationError, match="less than or equal to 120"):
        NvidiaVisionClientConfig(total_timeout_seconds=120.01)
    with pytest.raises(ValidationError, match="model profile maximum"):
        NvidiaVisionClientConfig(max_output_tokens=2_049)


@pytest.mark.asyncio
async def test_active_profile_binds_exact_request_and_response_identity() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(NVIDIA_ACTIVE_VISION_MODEL))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            NvidiaVisionClientConfig(),
            enabled=True,
            api_key=_TEST_KEY,
            http_client=http_client,
        )
        response = await provider.complete_json(
            system_prompt="Synthetic system instruction",
            user_prompt="Synthetic user instruction",
            images=_synthetic_images(),
            authorization=_authorization(),
            response_model=_ProbeResponse,
        )

    assert len(captured) == 1
    payload = captured[0]
    assert payload["model"] == NVIDIA_ACTIVE_VISION_MODEL
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["seed"] == 0
    assert payload["max_tokens"] == 2_048
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "top_k" not in payload
    assert "presence_penalty" not in payload
    assert "repetition_penalty" not in payload
    assert response.model == NVIDIA_ACTIVE_VISION_MODEL
    assert response.value.result == "ok"
    assert response.network_attempts == 1
    assert response.status_poll_attempts == 0


@pytest.mark.asyncio
async def test_deprecated_profile_is_explicit_and_never_falls_back_or_retries() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(503)

    config = NvidiaVisionClientConfig(model=NVIDIA_DEPRECATED_VISION_MODEL)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            config,
            enabled=True,
            api_key=_TEST_KEY,
            http_client=http_client,
        )
        with pytest.raises(NvidiaTransportError):
            await provider.complete_json(
                system_prompt="Synthetic system instruction",
                user_prompt="Synthetic user instruction",
                images=_synthetic_images(),
                authorization=_authorization(),
                response_model=_ProbeResponse,
            )

    assert config.max_transport_retries == 0
    assert len(captured) == 1
    payload = captured[0]
    assert payload["model"] == NVIDIA_DEPRECATED_VISION_MODEL
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["presence_penalty"] == 1.5
    assert payload["repetition_penalty"] == 1.0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "seed" not in payload
    assert "response_format" not in payload
    assert "json_schema" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_type", "error_code"),
    (
        (401, NvidiaAuthenticationError, "nvidia_authentication_rejected"),
        (403, NvidiaAuthenticationError, "nvidia_authentication_rejected"),
        (404, NvidiaModelUnavailableError, "nvidia_model_or_endpoint_unavailable"),
        (408, NvidiaTimeoutError, "nvidia_upstream_timeout"),
        (413, NvidiaTransportError, "nvidia_upstream_payload_rejected"),
        (422, NvidiaTransportError, "nvidia_upstream_payload_rejected"),
        (429, NvidiaRateLimitError, "nvidia_rate_limit_retry_exhausted"),
        (500, NvidiaTransportError, "nvidia_upstream_server_error"),
        (502, NvidiaTransportError, "nvidia_upstream_server_error"),
        (503, NvidiaTransportError, "nvidia_upstream_server_error"),
        (504, NvidiaTransportError, "nvidia_upstream_server_error"),
    ),
)
async def test_http_failures_are_classified_without_resend(
    status: int,
    error_type: type[Exception],
    error_code: str,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            NvidiaVisionClientConfig(),
            enabled=True,
            api_key=_TEST_KEY,
            http_client=http_client,
        )
        with pytest.raises(error_type) as raised:
            await provider.complete_json(
                system_prompt="Synthetic system instruction",
                user_prompt="Synthetic user instruction",
                authorization=_authorization(),
                response_model=_ProbeResponse,
            )

    assert calls == 1
    assert getattr(raised.value, "code", None) == error_code


@pytest.mark.asyncio
async def test_read_timeout_is_classified_without_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("synthetic private timeout detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            NvidiaVisionClientConfig(),
            enabled=True,
            api_key=_TEST_KEY,
            http_client=http_client,
        )
        with pytest.raises(NvidiaTimeoutError) as raised:
            await provider.complete_json(
                system_prompt="Synthetic system instruction",
                user_prompt="Synthetic user instruction",
                authorization=_authorization(),
                response_model=_ProbeResponse,
            )

    assert calls == 1
    assert raised.value.code == "nvidia_transport_timeout"
    assert "synthetic private timeout detail" not in repr(raised.value)


@pytest.mark.asyncio
async def test_active_profile_202_polling_preserves_model_and_does_not_repost() -> None:
    methods: list[str] = []
    pending_id = "64ae71b0-fd2a-4382-9b00-104f43ae5f00"

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, headers={"NVCF-REQID": pending_id})
        assert str(request.url).endswith(f"/status/{pending_id}")
        return httpx.Response(200, json=_completion(NVIDIA_ACTIVE_VISION_MODEL))

    config = NvidiaVisionClientConfig(status_poll_interval_seconds=0, max_status_polls=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            config,
            enabled=True,
            api_key=_TEST_KEY,
            http_client=http_client,
        )
        response = await provider.complete_json(
            system_prompt="Synthetic system instruction",
            user_prompt="Synthetic user instruction",
            authorization=_authorization(),
            response_model=_ProbeResponse,
        )

    assert methods == ["POST", "GET"]
    assert response.model == NVIDIA_ACTIVE_VISION_MODEL
    assert response.network_attempts == 2
    assert response.status_poll_attempts == 1


@pytest.mark.asyncio
async def test_transport_diagnostics_redact_secret_and_authorization_values() -> None:
    config = NvidiaVisionClientConfig()
    transport = NvidiaVisionTransport(config, api_key=SecretStr(_TEST_KEY))
    rendered = repr(transport)
    projected = redact_headers(
        {
            "Authorization": f"Bearer {_TEST_KEY}",
            "X-API-Key": _TEST_KEY,
            "X-Request-ID": "synthetic-safe-id",
        }
    )
    await transport.aclose()

    assert _TEST_KEY not in rendered
    assert projected["Authorization"] == "[REDACTED]"
    assert projected["X-API-Key"] == "[REDACTED]"
    assert _TEST_KEY not in repr(projected)
