from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from atlaslens_api.config import Settings
from atlaslens_api.main import _ocr_requested, _rapidocr_requested
from conftest import image_bytes


def test_cloud_assist_requires_explicit_cloud_mode(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyses",
        files={"image": ("fixture.jpg", image_bytes(), "image/jpeg")},
        data={
            "analysis_mode": "local_only",
            "cloud_processing_consent": "false",
            "authorization_acknowledged": "true",
            "allow_cloud_assist": "true",
        },
    )

    assert response.status_code == 400
    assert response.json()["code"] == "cloud_mode_required"


def test_phase6b_capabilities_are_concrete_and_secret_free(
    client_factory: Any, unused_tcp_port_factory: Any
) -> None:
    client = client_factory(
        phase6b_enabled=True,
        osv5m_worker_port=unused_tcp_port_factory(),
        plonk_worker_port=unused_tcp_port_factory(),
        paddleocr_worker_port=unused_tcp_port_factory(),
    )
    response = client.get("/api/v1/capabilities")
    body = response.json()

    assert response.status_code == 200
    assert body["providers"]["osv5m"]["operational_status"] == "worker_unreachable"
    assert body["providers"]["osv5m"]["execution_mode"] == "isolated_worker"
    assert body["providers"]["plonk"]["usable"] is False
    assert body["providers"]["paddleocr"]["weights_available"] is False
    assert body["providers"]["openai_geo_review"]["key_configured"] is False
    assert "OPENAI_API_KEY" not in response.text


def test_frontend_prefixed_openai_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VITE_OPENAI_API_KEY", "must-not-enter-a-frontend-build")

    with pytest.raises(ValidationError, match="frontend-prefixed"):
        Settings(_env_file=None)


def test_rapidocr_fallback_remains_requested_when_paddle_is_disabled() -> None:
    settings = Settings(
        _env_file=None,
        ocr_enabled=False,
        ocr_provider="rapidocr",
        phase6b_enabled=True,
        paddleocr_enabled=False,
        paddleocr_fallback_to_rapidocr=True,
        rapidocr_enabled=True,
    )

    assert _rapidocr_requested(settings) is True
    assert _ocr_requested(settings) is True

    without_fallback = settings.model_copy(
        update={"paddleocr_fallback_to_rapidocr": False}
    )
    assert _rapidocr_requested(without_fallback) is False
    assert _ocr_requested(without_fallback) is False
