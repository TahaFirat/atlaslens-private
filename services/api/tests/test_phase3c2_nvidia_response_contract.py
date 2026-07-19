from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from atlaslens_api.nvidia_vision.client import NvidiaVisionClientConfig
from atlaslens_api.nvidia_vision.errors import NvidiaResponseError, NvidiaSchemaError
from atlaslens_api.nvidia_vision.models import NvidiaGeolocationReasoning
from atlaslens_api.nvidia_vision.profiles import (
    NVIDIA_ACTIVE_VISION_MODEL,
    NVIDIA_DEPRECATED_VISION_MODEL,
)
from atlaslens_api.nvidia_vision.prompt import (
    NVIDIA_GEO_SYSTEM_PROMPT,
    build_nvidia_geo_user_prompt,
)
from atlaslens_api.nvidia_vision.provider import (
    NvidiaCloudAuthorization,
    NvidiaStructuredResponse,
    NvidiaVisionProvider,
)

_TEST_KEY = "test-only-response-contract-key"


def _valid_reasoning_payload() -> dict[str, Any]:
    return {
        "schema_version": "phase3c1-nvidia-geolocation-v1",
        "decision": "hypotheses",
        "observed_clues": [
            {
                "clue_id": "clue-1",
                "category": "architecture",
                "observation": "Synthetic fixture",
                "strength": "moderate",
            }
        ],
        "hypotheses": [
            {
                "hypothesis_id": "hypothesis-1",
                "label": "Synthetic test region",
                "latitude": 10.0,
                "longitude": 20.0,
                "uncertainty_radius_km": 25.0,
                "granularity": "region",
                "country_code": "ZZ",
                "confidence": 0.4,
                "confidence_semantics": "uncalibrated_model_self_assessment",
                "supporting_clue_ids": ["clue-1"],
                "contradicting_clue_ids": [],
                "limitations": ["Synthetic fixture only"],
            }
        ],
        "abstention_reason": None,
        "uncertainty_summary": "Synthetic uncertainty",
    }


def _envelope(
    content: object,
    *,
    model: str = NVIDIA_ACTIVE_VISION_MODEL,
    finish_reason: str = "stop",
    reasoning_content: object | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "model": model,
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 16},
    }


async def _complete(
    envelope: dict[str, Any],
    *,
    selected_model: str = NVIDIA_ACTIVE_VISION_MODEL,
) -> NvidiaStructuredResponse[NvidiaGeolocationReasoning]:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=envelope)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            NvidiaVisionClientConfig(model=selected_model),
            enabled=True,
            api_key=_TEST_KEY,
            http_client=http_client,
        )
        return await provider.complete_json(
            system_prompt="Synthetic system instruction",
            user_prompt="Synthetic user instruction",
            authorization=NvidiaCloudAuthorization(
                mode="cloud_assisted",
                cloud_consent=True,
            ),
            response_model=NvidiaGeolocationReasoning,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ("", " \r\n\t"))
async def test_plain_json_object_accepts_only_outer_whitespace(prefix: str) -> None:
    serialized = json.dumps(_valid_reasoning_payload())
    response = await _complete(_envelope(f"{prefix}{serialized}{prefix}"))

    assert response.model == NVIDIA_ACTIVE_VISION_MODEL
    assert response.value.decision == "hypotheses"
    assert response.network_attempts == 1
    assert response.status_poll_attempts == 0


@pytest.mark.asyncio
async def test_active_profile_accepts_one_conventional_complete_json_fence() -> None:
    serialized = json.dumps(_valid_reasoning_payload())
    response = await _complete(_envelope(f"```json\n{serialized}\n```"))

    assert response.value.decision == "hypotheses"


@pytest.mark.asyncio
async def test_deprecated_profile_keeps_strict_unfenced_historical_contract() -> None:
    serialized = json.dumps(_valid_reasoning_payload())
    with pytest.raises(NvidiaResponseError) as raised:
        await _complete(
            _envelope(
                f"```json\n{serialized}\n```",
                model=NVIDIA_DEPRECATED_VISION_MODEL,
            ),
            selected_model=NVIDIA_DEPRECATED_VISION_MODEL,
        )

    assert raised.value.code == "nvidia_response_content_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    (
        "",
        "   \r\n",
        None,
        "{}{}",
        '{"schema_version":"phase3c1-nvidia-geolocation-v1"} trailing prose',
        'prefix {"schema_version":"phase3c1-nvidia-geolocation-v1"}',
        "```JSON\n{}\n```",
        "```json\n{}\n```\n```json\n{}\n```",
        "[]",
        "null",
    ),
)
async def test_invalid_or_ambiguous_content_is_rejected_without_salvage(
    content: object,
) -> None:
    with pytest.raises(NvidiaResponseError) as raised:
        await _complete(_envelope(content))

    assert raised.value.code == "nvidia_response_content_invalid"
    assert repr(raised.value) == (
        "NvidiaResponseError(code='nvidia_response_content_invalid')"
    )


@pytest.mark.asyncio
async def test_reasoning_content_is_never_used_or_exposed() -> None:
    private_reasoning = "synthetic hidden reasoning must remain unused"
    reasoning_only = _envelope(None, reasoning_content=private_reasoning)
    with pytest.raises(NvidiaResponseError) as raised:
        await _complete(reasoning_only)

    assert raised.value.code == "nvidia_response_content_invalid"
    assert private_reasoning not in str(raised.value)
    assert private_reasoning not in repr(raised.value)

    public_content = json.dumps(_valid_reasoning_payload())
    response = await _complete(
        _envelope(public_content, reasoning_content=private_reasoning)
    )
    assert response.value.decision == "hypotheses"
    assert private_reasoning not in repr(response)


@pytest.mark.asyncio
async def test_truncation_and_wrong_model_identity_fail_closed() -> None:
    content = json.dumps(_valid_reasoning_payload())
    with pytest.raises(NvidiaResponseError) as truncated:
        await _complete(_envelope(content, finish_reason="length"))
    assert truncated.value.code == "nvidia_response_truncated"

    with pytest.raises(NvidiaResponseError) as mismatch:
        await _complete(_envelope(content, model=NVIDIA_DEPRECATED_VISION_MODEL))
    assert mismatch.value.code == "nvidia_response_model_mismatch"


@pytest.mark.asyncio
async def test_valid_json_with_invalid_candidate_coordinates_is_not_repaired() -> None:
    payload = _valid_reasoning_payload()
    payload["hypotheses"][0]["latitude"] = 91.0

    with pytest.raises(NvidiaSchemaError) as raised:
        await _complete(_envelope(json.dumps(payload)))

    assert raised.value.code == "nvidia_structured_output_invalid"
    assert "91" not in repr(raised.value)


def test_geolocation_prompt_requires_one_bare_json_object_without_hidden_reasoning() -> None:
    user_prompt = build_nvidia_geo_user_prompt()
    combined = f"{NVIDIA_GEO_SYSTEM_PROMPT}\n{user_prompt}"
    normalized_system_prompt = " ".join(NVIDIA_GEO_SYSTEM_PROMPT.split())

    assert "Return exactly one JSON object" in normalized_system_prompt
    assert "no Markdown fence, commentary, prefix, or suffix" in normalized_system_prompt
    assert "Do not expose chain of thought or hidden reasoning" in normalized_system_prompt
    assert user_prompt.count("JSON_SCHEMA=") == 1
    assert "```" not in combined
