from __future__ import annotations

import io
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import gps_jpeg, image_bytes, upload, wait_for_terminal


def test_health_readiness_capabilities_and_headers(client: TestClient) -> None:
    health = client.get("/api/v1/health", headers={"X-Request-ID": "request-test-123"})
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "version": "0.1.0"}
    assert health.headers["x-request-id"] == "request-test-123"
    assert health.headers["cache-control"] == "no-store"
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"

    readiness = client.get("/api/v1/ready")
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "ready"

    capabilities = client.get("/api/v1/capabilities").json()
    assert capabilities["supported_formats"] == ["jpeg", "png", "webp"]
    assert capabilities["enabled_analysis_modes"] == ["local_only"]
    assert capabilities["providers"]["cloud_vision"]["reason_code"] == "missing_secret"
    assert capabilities["providers"]["ocr"]["available"] is False
    assert capabilities["providers"]["visual_clues"]["provider_id"] == ("atlaslens-segformer-b2-v4")
    assert capabilities["providers"]["visual_clues"]["enabled"] is False
    assert capabilities["providers"]["visual_clues"]["execution_boundary"] == "local"
    assert capabilities["providers"]["place_research"]["provider_id"] == (
        "local-geonames-reverse-geocoder"
    )
    assert capabilities["providers"]["place_research"]["offline"] is True
    assert "path" not in capabilities["providers"]["visual_clues"]
    assert "checkpoint_sha256" not in capabilities["providers"]["visual_clues"]
    assert capabilities["retention"]["originals_deleted_after_analysis"] is True

    provider_ids = {
        item["provider_id"] for item in client.get("/api/v1/providers").json()["providers"]
    }
    assert "atlaslens-segformer-b2-v4" in provider_ids
    assert "local-geonames-reverse-geocoder" in provider_ids


@pytest.mark.parametrize(
    ("image_format", "filename", "content_type", "expected_format"),
    [
        ("JPEG", "valid.jpeg", "image/jpeg", "jpeg"),
        ("PNG", "valid.png", "image/png", "png"),
        ("WEBP", "valid.webp", "image/webp", "webp"),
    ],
)
def test_valid_uploads_complete_with_honest_abstention(
    client: TestClient,
    image_format: str,
    filename: str,
    content_type: str,
    expected_format: str,
) -> None:
    response = upload(
        client,
        image_bytes(image_format),
        filename=filename,
        content_type=content_type,
    )
    assert response.status_code == 202
    body = wait_for_terminal(client, response.json()["id"])
    assert body["status"] == "completed"
    assert body["image"]["format"] == expected_format
    assert len(body["image"]["sha256"]) == 64
    assert body["quality"] is not None
    assert body["candidates"] == []
    assert body["abstention"]["reason_code"] == "insufficient_geographic_evidence"


@pytest.mark.parametrize(
    ("payload", "filename", "content_type", "expected_code"),
    [
        (b"not-an-image", "bad.jpg", "image/jpeg", "unsupported_media_type"),
        (image_bytes("PNG"), "bad.png", "image/jpeg", "mime_mismatch"),
        (image_bytes("PNG"), "bad.jpg", "image/png", "extension_mismatch"),
        (b"\xff\xd8\xff\xe0" + b"truncated", "bad.jpg", "image/jpeg", "malformed_image"),
    ],
)
def test_upload_rejections_are_safe(
    client: TestClient,
    payload: bytes,
    filename: str,
    content_type: str,
    expected_code: str,
) -> None:
    response = upload(client, payload, filename=filename, content_type=content_type)
    assert response.status_code in {415, 422}
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == expected_code
    assert "traceback" not in response.text.lower()


def test_oversized_compressed_upload(client_factory: Any) -> None:
    limited = client_factory(max_upload_bytes=1024)
    response = upload(limited, b"\xff\xd8\xff" + b"x" * 2048)
    assert response.status_code == 413
    assert response.json()["code"] == "upload_too_large"


def test_excessive_decoded_pixels(client_factory: Any) -> None:
    limited = client_factory(max_decoded_pixels=100, max_image_dimension=100)
    response = upload(
        limited,
        image_bytes("PNG", size=(20, 20)),
        filename="large.png",
        content_type="image/png",
    )
    assert response.status_code == 413
    assert response.json()["code"] == "decoded_image_too_large"


def test_animated_webp_is_rejected(client: TestClient) -> None:
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (20, 20), color) for color in ("red", "blue")]
    frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:], duration=50)
    response = upload(
        client,
        buffer.getvalue(),
        filename="animated.webp",
        content_type="image/webp",
    )
    assert response.status_code == 422
    assert response.json()["code"] == "animated_image_not_supported"


def test_orientation_is_normalized(client: TestClient) -> None:
    response = upload(client, gps_jpeg(10, 20, orientation=6, size=(40, 20)))
    body = wait_for_terminal(client, response.json()["id"])
    assert body["image"]["orientation_normalized"] is True
    assert (body["image"]["width"], body["image"]["height"]) == (20, 40)


def test_exif_upload_produces_metadata_only_candidate(client: TestClient) -> None:
    response = upload(client, gps_jpeg(41.015137, 28.97953))
    body = wait_for_terminal(client, response.json()["id"])
    assert body["status"] == "completed"
    assert body["abstention"] is None
    candidate = body["candidates"][0]
    assert candidate["center"]["latitude"] == pytest.approx(41.015137, abs=1e-5)
    assert candidate["center"]["longitude"] == pytest.approx(28.97953, abs=1e-5)
    assert candidate["geometry"]["coordinates"] == pytest.approx([28.97953, 41.015137])
    assert candidate["granularity"] == "exact_metadata"
    assert candidate["verification_status"] == "metadata_only"
    assert candidate["verified"] is False
    assert candidate["radius_km"] == 1.0
    assert candidate["confidence_kind"] == "source_reliability"
    assert candidate["provenance"]
    evidence = {item["id"]: item for item in body["evidence"]}
    assert set(candidate["evidence_ids"]) <= evidence.keys()
    assert evidence["evidence-exif-gps"]["sensitive"] is True
    assert "41.015" not in evidence["evidence-exif-gps"]["display_value"]


def test_sse_replays_progress_and_terminal_event(client: TestClient) -> None:
    accepted = upload(client, gps_jpeg(35.0, -120.0)).json()
    wait_for_terminal(client, accepted["id"])
    response = client.get(accepted["events_url"])
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert "event: progress" in response.text
    assert "event: completed" in response.text
    assert '"event_type":"completed"' in response.text


def test_delete_is_idempotent_and_unknown_get_is_404(client: TestClient) -> None:
    accepted = upload(client, image_bytes()).json()
    wait_for_terminal(client, accepted["id"])
    first = client.delete(accepted["delete_url"])
    second = client.delete(accepted["delete_url"])
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"id": accepted["id"], "deleted": True}
    missing = client.get(accepted["status_url"])
    assert missing.status_code == 404
    assert missing.json()["code"] == "analysis_not_found"

    random_id = str(uuid4())
    assert client.get(f"/api/v1/analyses/{random_id}").status_code == 404
    assert client.delete(f"/api/v1/analyses/{random_id}").status_code == 200


def test_authorization_cloud_consent_and_missing_capability(client: TestClient) -> None:
    no_authorization = upload(client, image_bytes(), authorization="false")
    assert no_authorization.status_code == 400
    assert no_authorization.json()["code"] == "authorization_required"

    no_consent = upload(client, image_bytes(), mode="cloud_assisted", consent="false")
    assert no_consent.status_code == 400
    assert no_consent.json()["code"] == "cloud_consent_required"

    missing_provider = upload(client, image_bytes(), mode="cloud_assisted", consent="true")
    assert missing_provider.status_code == 422
    assert missing_provider.json()["code"] == "cloud_provider_unavailable"


def test_idempotency_key_reuses_analysis(client: TestClient) -> None:
    headers = {"Idempotency-Key": "same-request-123"}
    first = upload(client, image_bytes(), headers=headers)
    second = upload(client, image_bytes(color=(1, 2, 3)), headers=headers)
    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]

    invalid = upload(client, image_bytes(), headers={"Idempotency-Key": "short"})
    assert invalid.status_code == 400
    assert invalid.json()["code"] == "invalid_idempotency_key"


def test_strict_cors(client: TestClient) -> None:
    allowed = client.options(
        "/api/v1/analyses",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"

    hostile = client.get("/api/v1/health", headers={"Origin": "https://attacker.invalid"})
    assert "access-control-allow-origin" not in hostile.headers


def test_analysis_rate_limit_is_local_and_safe(client_factory: Any) -> None:
    limited = client_factory(analysis_rate_limit_per_minute=1)
    assert upload(limited, image_bytes()).status_code == 202
    blocked = upload(limited, image_bytes())
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "analysis_rate_limited"
    assert int(blocked.headers["retry-after"]) > 0
