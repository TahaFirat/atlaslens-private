from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from atlaslens_api.config import Settings
from atlaslens_api.image_processing import CloudSafeDerivative
from atlaslens_api.main import create_app
from atlaslens_api.nvidia_vision.cli import _run_blind_query, _safe_report
from atlaslens_api.nvidia_vision.cli import main as cli_main
from atlaslens_api.nvidia_vision.client import (
    NVIDIA_VISION_MODEL,
    NvidiaVisionClientConfig,
)
from atlaslens_api.nvidia_vision.errors import (
    NvidiaAuthenticationError,
    NvidiaConfigurationError,
    NvidiaPreprocessingError,
    NvidiaProviderUnavailableError,
    NvidiaResponseError,
    NvidiaTimeoutError,
    NvidiaTransportError,
)
from atlaslens_api.nvidia_vision.models import NvidiaGeolocationReasoning
from atlaslens_api.nvidia_vision.preprocessing import (
    PreparedImage,
    PreparedImageSet,
    prepare_nvidia_images,
)
from atlaslens_api.nvidia_vision.provider import (
    NvidiaCloudAuthorization,
    NvidiaProviderStatus,
    NvidiaStructuredResponse,
    NvidiaVisionProvider,
)
from atlaslens_api.providers.base import InvocationContext, OutcomeStatus, ProviderDescriptor
from atlaslens_api.providers.nvidia_vision import NvidiaVisionClueProvider
from atlaslens_api.schemas import AnalysisMode

_DUMMY_API_KEY = "test-only-nvidia-key-0001"
_PENDING_ID = UUID("15fa1541-f10a-4fa5-86f5-28b5cb331ea0")


def _reasoning_payload() -> dict[str, Any]:
    return {
        "schema_version": "phase3c1-nvidia-geolocation-v1",
        "decision": "hypotheses",
        "observed_clues": [
            {
                "clue_id": "clue-1",
                "category": "architecture",
                "observation": "Synthetic masonry fixture",
                "strength": "moderate",
            }
        ],
        "hypotheses": [
            {
                "hypothesis_id": "hypothesis-1",
                "label": "Synthetic test region",
                "latitude": 10.0,
                "longitude": 20.0,
                "uncertainty_radius_km": 8.0,
                "granularity": "region",
                "country_code": "ZZ",
                "confidence": 0.9,
                "confidence_semantics": "uncalibrated_model_self_assessment",
                "supporting_clue_ids": ["clue-1"],
                "contradicting_clue_ids": [],
                "limitations": ["Synthetic fixture; not real-world evidence"],
            }
        ],
        "abstention_reason": None,
        "uncertainty_summary": "Synthetic fixture with broad uncertainty",
    }


def _reasoning() -> NvidiaGeolocationReasoning:
    return _validate_reasoning(_reasoning_payload())


def _validate_reasoning(payload: dict[str, Any]) -> NvidiaGeolocationReasoning:
    return NvidiaGeolocationReasoning.model_validate_json(json.dumps(payload))


def _completion_envelope() -> dict[str, Any]:
    return {
        "model": NVIDIA_VISION_MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": _reasoning().model_dump_json()},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 34},
    }


def _jpeg_with_metadata() -> bytes:
    image = Image.effect_noise((1_024, 768), 96).convert("RGB")
    exif = Image.Exif()
    exif[0x010E] = "synthetic private metadata"
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90, exif=exif)
    return output.getvalue()


def _prepared_images() -> PreparedImageSet:
    return prepare_nvidia_images(
        _jpeg_with_metadata(),
        declared_media_type="image/jpeg",
        maximum_edge=1_280,
        crop_kinds=("full",),
    )


def _inject_jpeg_segment(jpeg: bytes, marker: int, content: bytes) -> bytes:
    length = len(content) + 2
    return jpeg[:2] + bytes((0xFF, marker)) + length.to_bytes(2, "big") + content + jpeg[2:]


def _cloud_derivative() -> CloudSafeDerivative:
    prepared = _prepared_images().images[0]
    return CloudSafeDerivative(
        data_url=prepared.data_url(),
        byte_count=prepared.byte_count,
        width=prepared.width,
        height=prepared.height,
    )


def _authorization(
    *, mode: str = "cloud_assisted", consent: bool = True
) -> NvidiaCloudAuthorization:
    return NvidiaCloudAuthorization(mode=mode, cloud_consent=consent)  # type: ignore[arg-type]


def _context(
    mode: AnalysisMode = AnalysisMode.CLOUD_ASSISTED,
    *,
    cloud_consent: bool = True,
) -> InvocationContext:
    return InvocationContext(
        analysis_id=uuid4(),
        request_id="synthetic-nvidia-test",
        mode=mode,
        cloud_consent=cloud_consent,
        deadline=datetime.now(UTC) + timedelta(seconds=10),
        cancellation=asyncio.Event(),
    )


def test_reasoning_models_are_strict_and_abstention_is_explicit() -> None:
    result = _reasoning()
    assert result.hypotheses[0].uncertainty_radius_km > 0
    assert result.hypotheses[0].confidence_semantics == "uncalibrated_model_self_assessment"
    assert "10.0" not in repr(result)
    assert "20.0" not in str(result)
    assert "Synthetic masonry fixture" not in repr(result)

    extra = _reasoning_payload()
    extra["hidden_reasoning"] = "must never be accepted"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _validate_reasoning(extra)

    abstention = _reasoning_payload()
    abstention.update(
        decision="abstain",
        hypotheses=[],
        abstention_reason="The synthetic evidence is insufficient",
    )
    parsed = _validate_reasoning(abstention)
    assert parsed.decision == "abstain"
    assert parsed.hypotheses == ()

    contradictory = _reasoning_payload()
    contradictory["decision"] = "abstain"
    contradictory["abstention_reason"] = "Cannot decide"
    with pytest.raises(ValidationError, match="no hypotheses"):
        _validate_reasoning(contradictory)

    unknown_reference = _reasoning_payload()
    unknown_reference["hypotheses"][0]["supporting_clue_ids"] = ["clue-2"]
    with pytest.raises(ValidationError, match="unknown clue"):
        _validate_reasoning(unknown_reference)

    coerced_number = _reasoning_payload()
    coerced_number["hypotheses"][0]["latitude"] = "10.0"
    with pytest.raises(ValidationError, match="valid number"):
        _validate_reasoning(coerced_number)


def test_preprocessing_is_bounded_metadata_free_and_constructor_guarded() -> None:
    source = _jpeg_with_metadata()
    prepared = prepare_nvidia_images(
        source,
        declared_media_type="image/jpeg",
        maximum_edge=1_280,
        crop_kinds=("full", "centre", "upper", "lower"),
    )

    assert tuple(item.kind for item in prepared.images) == (
        "full",
        "centre",
        "upper",
        "lower",
    )
    assert all(0 < item.byte_count <= 180_000 for item in prepared.images)
    assert prepared.total_bytes <= 4 * 180_000
    assert "synthetic private metadata" not in repr(prepared)
    assert prepared.cache_identity_sha256 not in repr(prepared)
    for item in prepared.images:
        assert "jpeg_bytes=<redacted>" in repr(item)
        with Image.open(io.BytesIO(item.jpeg_bytes)) as decoded:
            assert decoded.format == "JPEG"
            assert not decoded.getexif()
            assert not decoded.info.get("icc_profile")

    with Image.open(io.BytesIO(source)) as decoded:
        width, height = decoded.size
    with pytest.raises(NvidiaPreprocessingError, match="nvidia_image_derivative_invalid"):
        PreparedImage(kind="full", jpeg_bytes=source, width=width, height=height)

    safe = prepared.images[0]
    metadata_segments = (
        (0xE0, b"private-synthetic-app0"),
        (0xFE, b"synthetic JPEG comment"),
        (0xE1, b"http://ns.adobe.com/xap/1.0/\x00synthetic-xmp"),
        (0xE2, b"ICC_PROFILE\x00synthetic-icc"),
        (0xED, b"Photoshop 3.0\x00synthetic-iptc"),
    )
    for marker, content in metadata_segments:
        with pytest.raises(
            NvidiaPreprocessingError, match="nvidia_image_derivative_invalid"
        ):
            PreparedImage(
                kind="full",
                jpeg_bytes=_inject_jpeg_segment(safe.jpeg_bytes, marker, content),
                width=safe.width,
                height=safe.height,
            )


@pytest.mark.asyncio
async def test_authorization_gate_refuses_without_network() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            NvidiaVisionClientConfig(max_transport_retries=0),
            enabled=True,
            api_key=_DUMMY_API_KEY,
            http_client=http_client,
        )
        with pytest.raises(NvidiaProviderUnavailableError, match="nvidia_cloud_mode_required"):
            await provider.complete_json(
                system_prompt="Synthetic system prompt",
                user_prompt="Synthetic user prompt",
                authorization=_authorization(mode="local_only"),
                response_model=NvidiaGeolocationReasoning,
            )
        with pytest.raises(NvidiaProviderUnavailableError, match="nvidia_cloud_consent_required"):
            await provider.complete_json(
                system_prompt="Synthetic system prompt",
                user_prompt="Synthetic user prompt",
                authorization=_authorization(consent=False),
                response_model=NvidiaGeolocationReasoning,
            )

    assert calls == 0


@pytest.mark.asyncio
async def test_mock_transport_200_payload_and_schema_validation() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "https://integrate.api.nvidia.com/v1/chat/completions"
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_completion_envelope())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            NvidiaVisionClientConfig(max_transport_retries=0),
            enabled=True,
            api_key=_DUMMY_API_KEY,
            http_client=http_client,
        )
        response = await provider.complete_json(
            system_prompt="Synthetic system prompt",
            user_prompt="Synthetic user prompt",
            authorization=_authorization(),
            images=_prepared_images(),
            response_model=NvidiaGeolocationReasoning,
        )

    assert response.value.decision == "hypotheses"
    assert response.network_attempts == 1
    assert response.status_poll_attempts == 0
    assert len(captured) == 1
    payload = captured[0]
    assert "response_format" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
    encoded = image_url.removeprefix("data:image/jpeg;base64,")
    assert len(base64.b64decode(encoded, validate=True)) <= 180_000


@pytest.mark.asyncio
async def test_mock_transport_202_polls_bounded_status_endpoint() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, headers={"NVCF-REQID": str(_PENDING_ID)})
        assert request.url == f"https://integrate.api.nvidia.com/v1/status/{_PENDING_ID}"
        if methods.count("GET") == 1:
            return httpx.Response(202)
        return httpx.Response(200, json=_completion_envelope())

    config = NvidiaVisionClientConfig(
        max_transport_retries=0,
        status_poll_interval_seconds=0,
        max_status_polls=3,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            config,
            enabled=True,
            api_key=_DUMMY_API_KEY,
            http_client=http_client,
        )
        response = await provider.complete_json(
            system_prompt="Synthetic system prompt",
            user_prompt="Synthetic user prompt",
            authorization=_authorization(),
            response_model=NvidiaGeolocationReasoning,
        )

    assert methods == ["POST", "GET", "GET"]
    assert response.network_attempts == 3
    assert response.status_poll_attempts == 2


@pytest.mark.asyncio
async def test_202_body_request_id_fallback_and_poll_exhaustion_are_bounded() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, json={"requestId": str(_PENDING_ID)})
        return httpx.Response(202)

    config = NvidiaVisionClientConfig(
        max_transport_retries=0,
        status_poll_interval_seconds=0,
        max_status_polls=2,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            config,
            enabled=True,
            api_key=_DUMMY_API_KEY,
            http_client=http_client,
        )
        with pytest.raises(NvidiaTimeoutError, match="nvidia_status_poll_timeout"):
            await provider.complete_json(
                system_prompt="Synthetic system prompt",
                user_prompt="Synthetic user prompt",
                authorization=_authorization(),
                response_model=NvidiaGeolocationReasoning,
            )

    assert methods == ["POST", "GET", "GET"]


@pytest.mark.asyncio
async def test_authentication_failure_is_terminal_for_provider_instance() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            NvidiaVisionClientConfig(max_transport_retries=0),
            enabled=True,
            api_key=_DUMMY_API_KEY,
            http_client=http_client,
        )
        for _ in range(2):
            with pytest.raises(
                NvidiaAuthenticationError, match="nvidia_authentication_rejected"
            ):
                await provider.complete_json(
                    system_prompt="Synthetic system prompt",
                    user_prompt="Synthetic user prompt",
                    authorization=_authorization(),
                    response_model=NvidiaGeolocationReasoning,
                )

    assert calls == 1


class _StubReasoner:
    def __init__(self) -> None:
        self.calls = 0
        self.model = NVIDIA_VISION_MODEL

    def status(self) -> NvidiaProviderStatus:
        return NvidiaProviderStatus(
            state="not_ready",
            enabled=True,
            credentials_configured=True,
            model=self.model,
        )

    async def close(self) -> None:
        return None

    async def analyze(self, _payload: bytes, **_: object) -> NvidiaStructuredResponse[Any]:
        self.calls += 1
        return NvidiaStructuredResponse(
            request_id="synthetic-request",
            model=self.model,
            value=_reasoning(),
        )


@pytest.mark.asyncio
async def test_adapter_consent_gate_and_mapping_are_fail_closed() -> None:
    reasoner = _StubReasoner()
    provider = NvidiaVisionClueProvider(
        enabled=True,
        api_key=_DUMMY_API_KEY,
        reasoner=reasoner,  # type: ignore[arg-type]
    )

    denied = await provider.extract(
        CloudSafeDerivative(data_url="invalid", byte_count=1, width=1, height=1),
        _context(AnalysisMode.LOCAL_ONLY, cloud_consent=False),
    )
    assert denied.status == OutcomeStatus.SKIPPED
    assert denied.failure is not None
    assert denied.failure.code == "privacy_denied"
    assert reasoner.calls == 0

    mapped = await provider.extract(_cloud_derivative(), _context())
    assert mapped.status == OutcomeStatus.SUCCEEDED
    assert mapped.value is not None
    assert reasoner.calls == 1
    hypothesis = mapped.value.hypotheses[0]
    assert hypothesis.radius_km == 25.0
    assert hypothesis.confidence == 0.35
    assert hypothesis.granularity == "region"
    assert hypothesis.country_code == "ZZ"
    assert hypothesis.visible_clues == ["Synthetic masonry fixture"]


def test_settings_default_to_openai_and_reject_frontend_nvidia_key() -> None:
    safe_environment = {"LOCALAPPDATA": "D:\\geoSearch\\.settings-test"}
    with patch.object(os, "environ", safe_environment):
        settings = Settings(_env_file=None)
    assert settings.cloud_vision_provider == "openai"
    assert settings.nvidia_vision_enabled is False
    assert settings.nvidia_api_key is None
    assert settings.nvidia_vision_model == NVIDIA_VISION_MODEL

    with patch.object(
        os,
        "environ",
        {**safe_environment, "VITE_NVIDIA_API_KEY": "test-only-frontend-key"},
    ), pytest.raises(ValidationError, match="frontend-prefixed"):
        Settings(_env_file=None)


def test_nvidia_timeout_is_wired_through_the_normal_api_pipeline(tmp_path: Path) -> None:
    safe_environment = {"LOCALAPPDATA": str(tmp_path / "local-app-data")}
    with patch.object(os, "environ", safe_environment):
        settings = Settings(
            _env_file=None,
            database_url=f"sqlite:///{(tmp_path / 'atlaslens.db').as_posix()}",
            temp_storage_dir=tmp_path / "storage",
            cloud_vision_provider="nvidia",
            nvidia_vision_enabled=False,
            nvidia_vision_timeout_seconds=90,
            global_model_enabled=False,
            phase6b_enabled=False,
        )
        app = create_app(settings)

    assert app.state.services.pipeline._vision_timeout_seconds == 90
    assert app.state.services.vision_provider.descriptor.id == "nvidia-qwen-vision-reasoning"


class _ClosableVisionProvider:
    def __init__(self) -> None:
        self.closed = False
        self.descriptor = ProviderDescriptor(
            id="synthetic-closable-vision",
            kind="vision_language",
            version="test-only",
            execution_boundary="cloud",
            criticality="optional",
            available=False,
            unavailable_reason_code="disabled",
        )

    async def close(self) -> None:
        self.closed = True


def test_default_provider_compatibility_and_injected_vision_lifecycle(tmp_path: Path) -> None:
    safe_environment = {"LOCALAPPDATA": str(tmp_path / "local-app-data")}
    with patch.object(os, "environ", safe_environment):
        settings = Settings(
            _env_file=None,
            database_url=f"sqlite:///{(tmp_path / 'atlaslens.db').as_posix()}",
            temp_storage_dir=tmp_path / "storage",
            global_model_enabled=False,
            phase6b_enabled=False,
        )
        default_app = create_app(settings)
        assert default_app.state.services.vision_provider.descriptor.id == "openai-vision-clues"

        closable = _ClosableVisionProvider()
        injected_app = create_app(settings, vision_provider=closable)  # type: ignore[arg-type]
        with TestClient(injected_app, raise_server_exceptions=False):
            pass

    assert closable.closed is True


@pytest.mark.parametrize(
    "base_url",
    (
        "http://integrate.api.nvidia.com/v1",
        "https://integrate.api.nvidia.com/v1/chat/completions",
        "https://integrate.api.nvidia.com/v1?redirect=1",
        "https://integrate.api.nvidia.com.evil.example/v1",
    ),
)
def test_transport_accepts_only_the_exact_official_base_url(base_url: str) -> None:
    with pytest.raises(NvidiaConfigurationError, match="nvidia_base_url_invalid"):
        NvidiaVisionClientConfig(base_url=base_url)


@pytest.mark.asyncio
async def test_non_stop_completion_is_rejected_without_content_projection() -> None:
    envelope = _completion_envelope()
    envelope["choices"][0]["finish_reason"] = "content_filter"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=envelope)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = NvidiaVisionProvider(
            NvidiaVisionClientConfig(max_transport_retries=0),
            enabled=True,
            api_key=_DUMMY_API_KEY,
            http_client=http_client,
        )
        with pytest.raises(
            NvidiaResponseError, match="nvidia_response_finish_reason_invalid"
        ):
            await provider.complete_json(
                system_prompt="Synthetic system prompt",
                user_prompt="Synthetic user prompt",
                authorization=_authorization(),
                response_model=NvidiaGeolocationReasoning,
            )


@pytest.mark.asyncio
async def test_redirect_is_refused_even_with_permissive_injected_client() -> None:
    destinations: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        destinations.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://example.invalid/collect"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as http_client:
        provider = NvidiaVisionProvider(
            NvidiaVisionClientConfig(max_transport_retries=0),
            enabled=True,
            api_key=_DUMMY_API_KEY,
            http_client=http_client,
        )
        with pytest.raises(NvidiaTransportError, match="nvidia_redirect_refused"):
            await provider.complete_json(
                system_prompt="Synthetic system prompt",
                user_prompt="Synthetic user prompt",
                authorization=_authorization(),
                response_model=NvidiaGeolocationReasoning,
            )

    assert destinations == ["https://integrate.api.nvidia.com/v1/chat/completions"]


@pytest.mark.asyncio
async def test_blind_query_checks_credentials_before_reading_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    report = await _run_blind_query(Path("synthetic-file-must-not-be-read.jpg"))

    assert report == {
        "status": "error",
        "code": "nvidia_credentials_missing",
        "network_attempts": 0,
        "status_poll_attempts": 0,
    }


def test_blind_cli_requires_all_authorization_flags_and_omits_sensitive_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["blind-query", "synthetic-never-read.jpg", "--execute"]) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["network_attempts"] == 0

    structured = NvidiaStructuredResponse(
        request_id="synthetic-private-request-id",
        model=NVIDIA_VISION_MODEL,
        value=_reasoning(),
    )
    assert "synthetic-private-request-id" not in repr(structured)
    safe = _safe_report(structured)
    serialized = json.dumps(safe, sort_keys=True)
    assert safe["coordinates_omitted"] is True
    assert "latitude" not in serialized
    assert "longitude" not in serialized
    assert "synthetic-private-request-id" not in serialized
    assert "Synthetic masonry fixture" not in serialized
