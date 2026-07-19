from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from atlaslens_api.config import Settings
from atlaslens_api.image_processing import CloudSafeDerivative
from atlaslens_api.main import create_app
from atlaslens_api.providers.base import (
    InvocationContext,
    ProviderDescriptor,
    ProviderOutcome,
    VisionClueResult,
    VisionHypothesis,
)
from atlaslens_api.storage import LocalImageHandle
from conftest import gps_jpeg, image_bytes, upload, wait_for_terminal


class TrackingVisionProvider:
    descriptor = ProviderDescriptor(
        id="mock-vision",
        kind="vision_language",
        version="test",
        execution_boundary="cloud",
        criticality="optional",
        available=True,
        model_name="mock-model",
    )

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def extract(
        self, derivative: CloudSafeDerivative, context: InvocationContext
    ) -> ProviderOutcome[VisionClueResult]:
        self.calls += 1
        assert derivative.data_url.startswith("data:image/jpeg;base64,")
        assert context.cloud_consent is True
        if self.fail:
            return ProviderOutcome.failed(
                "invalid_output", retryable=False, attempts=1, duration_ms=1
            )
        return ProviderOutcome.succeeded(
            VisionClueResult(
                hypotheses=[
                    VisionHypothesis(
                        label="Broad coastal region",
                        latitude=36.5,
                        longitude=30.5,
                        radius_km=25,
                        confidence=0.4,
                        granularity="region",
                        country_code="TR",
                        visible_clues=["coastline", "mountain profile"],
                    )
                ]
            )
        )


class ExplodingQualityProvider:
    descriptor = ProviderDescriptor(
        id="exploding-quality",
        kind="image_quality",
        version="test",
        execution_boundary="local",
        criticality="required",
        available=True,
    )

    async def analyze(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[Any]:
        raise RuntimeError("SENSITIVE_FAILURE_SENTINEL")


class SlowQualityProvider:
    descriptor = ProviderDescriptor(
        id="slow-quality",
        kind="image_quality",
        version="test",
        execution_boundary="local",
        criticality="required",
        available=True,
    )

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def analyze(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[Any]:
        self.started.set()
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return ProviderOutcome.failed("timeout", retryable=False, attempts=1, duration_ms=200)


def test_local_only_pipeline_never_invokes_cloud(client_factory: Any) -> None:
    provider = TrackingVisionProvider()
    client = client_factory(vision_provider=provider)
    accepted = upload(client, image_bytes()).json()
    body = wait_for_terminal(client, accepted["id"])
    assert body["status"] == "completed"
    assert body["abstention"] is not None
    assert provider.calls == 0


def test_mock_cloud_pipeline_enforces_vision_candidate_policy(client_factory: Any) -> None:
    provider = TrackingVisionProvider()
    client = client_factory(vision_provider=provider)
    accepted = upload(
        client,
        image_bytes(),
        mode="cloud_assisted",
        consent="true",
    ).json()
    body = wait_for_terminal(client, accepted["id"])
    assert provider.calls == 1
    candidate = body["candidates"][0]
    assert candidate["verification_status"] == "unverified_model"
    assert candidate["verified"] is False
    assert candidate["confidence"] <= 0.40
    assert candidate["radius_km"] >= 25
    assert candidate["granularity"] != "exact_metadata"
    assert candidate["provenance"]
    assert candidate["evidence_ids"]


def test_invalid_cloud_output_degrades_to_abstention(client_factory: Any) -> None:
    provider = TrackingVisionProvider(fail=True)
    client = client_factory(vision_provider=provider)
    accepted = upload(
        client,
        image_bytes(),
        mode="cloud_assisted",
        consent="true",
    ).json()
    body = wait_for_terminal(client, accepted["id"])
    assert body["status"] == "completed"
    assert body["candidates"] == []
    assert body["abstention"] is not None
    assert "provider.cloud_vision.safe_degradation" in body["warnings"]


def test_temporary_files_clean_after_success(client: TestClient) -> None:
    accepted = upload(client, image_bytes()).json()
    wait_for_terminal(client, accepted["id"])
    storage_root = Path(client.app.state.services.storage.root)
    assert list(storage_root.iterdir()) == []


def test_temporary_files_clean_after_failure(client_factory: Any) -> None:
    client = client_factory(quality_provider=ExplodingQualityProvider())
    accepted = upload(client, image_bytes()).json()
    body = wait_for_terminal(client, accepted["id"])
    assert body["status"] == "failed"
    assert body["failure"]["code"] == "internal_analysis_error"
    storage_root = Path(client.app.state.services.storage.root)
    assert list(storage_root.iterdir()) == []


def test_delete_cleans_active_job_artifacts(client_factory: Any) -> None:
    provider = SlowQualityProvider()
    client = client_factory(quality_provider=provider)
    accepted = upload(client, image_bytes()).json()
    assert provider.started.wait(timeout=1)
    deleted = client.delete(accepted["delete_url"])
    assert deleted.status_code == 200
    assert provider.cancelled.wait(timeout=1)
    storage_root = Path(client.app.state.services.storage.root)
    assert list(storage_root.iterdir()) == []
    assert client.get(accepted["status_url"]).status_code == 404


def test_sse_emits_heartbeat_and_releases_connection(client_factory: Any) -> None:
    client = client_factory(
        quality_provider=SlowQualityProvider(), max_sse_connections_per_client=1
    )
    accepted = upload(client, image_bytes()).json()
    first = client.get(accepted["events_url"])
    assert first.status_code == 200
    assert "event: heartbeat" in first.text
    assert "event: completed" in first.text
    second = client.get(accepted["events_url"])
    assert second.status_code == 200
    assert "event: completed" in second.text


def test_in_memory_database_handles_concurrent_sse_poll_and_delete(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        database_url="sqlite:///:memory:",
        temp_storage_dir=tmp_path / "tmp",
        openai_api_key=None,
        ocr_enabled=False,
        sse_heartbeat_seconds=0.02,
        max_sse_lifetime_seconds=1,
    )
    provider = SlowQualityProvider()
    errors: list[BaseException] = []
    poll_statuses: list[int] = []
    stream_result: dict[str, Any] = {}
    stop_polling = threading.Event()

    with TestClient(
        create_app(settings, quality_provider=provider),
        raise_server_exceptions=False,
    ) as concurrent_client:
        accepted = upload(concurrent_client, image_bytes()).json()
        assert provider.started.wait(timeout=1)

        def consume_events() -> None:
            try:
                response = concurrent_client.get(accepted["events_url"])
                stream_result["status"] = response.status_code
                stream_result["body"] = response.text
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        def poll_status() -> None:
            try:
                while not stop_polling.is_set():
                    response = concurrent_client.get(accepted["status_url"])
                    poll_statuses.append(response.status_code)
                    if response.status_code == 404:
                        return
                    time.sleep(0.002)
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        stream_thread = threading.Thread(target=consume_events)
        poll_thread = threading.Thread(target=poll_status)
        stream_thread.start()
        poll_thread.start()
        time.sleep(0.05)

        deleted = concurrent_client.delete(accepted["delete_url"])
        stop_polling.set()
        stream_thread.join(timeout=2)
        poll_thread.join(timeout=2)

        assert deleted.status_code == 200
        assert errors == []
        assert not stream_thread.is_alive()
        assert not poll_thread.is_alive()
        assert poll_statuses
        assert set(poll_statuses) <= {200, 404}
        assert stream_result["status"] == 200
        assert "event: deleted" in stream_result["body"]
        assert "internal_server_error" not in stream_result["body"]
        assert concurrent_client.get(accepted["status_url"]).status_code == 404


def test_keep_uploads_retains_only_until_delete(client_factory: Any) -> None:
    client = client_factory(keep_uploads=True)
    accepted = upload(client, image_bytes()).json()
    wait_for_terminal(client, accepted["id"])
    storage_root = Path(client.app.state.services.storage.root)
    retained = list(storage_root.iterdir())
    assert len(retained) == 1
    assert retained[0].suffix == ".upload"
    client.delete(accepted["delete_url"])
    assert list(storage_root.iterdir()) == []


def test_sensitive_values_never_appear_in_logs(client_factory: Any, capfd: Any) -> None:
    secret = "sk-SUPERSECRET0123456789"
    client = client_factory(openai_api_key=secret)
    accepted = upload(
        client,
        gps_jpeg(41.123456, 29.654321),
        filename="private-person@example.com.jpg",
    ).json()
    wait_for_terminal(client, accepted["id"])
    captured = capfd.readouterr()
    rendered = captured.out + captured.err
    assert secret not in rendered
    assert "private-person@example.com" not in rendered
    assert "41.123456" not in rendered
    assert "29.654321" not in rendered
    assert "SENSITIVE_FAILURE_SENTINEL" not in rendered
