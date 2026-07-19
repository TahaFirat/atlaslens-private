from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from atlaslens_api.config import Settings
from atlaslens_api.main import create_app
from conftest import image_bytes, wait_for_terminal


def _submit(client: TestClient, *, idempotency_key: str) -> str:
    response = client.post(
        "/api/v1/analyses",
        headers={"Idempotency-Key": idempotency_key},
        files={"image": ("fixture.jpg", image_bytes(), "image/jpeg")},
        data={
            "analysis_mode": "local_only",
            "cloud_processing_consent": "false",
            "authorization_acknowledged": "true",
        },
    )
    assert response.status_code == 202
    return str(response.json()["id"])


def test_phase6c_capabilities_report_safe_provider_and_index_states(
    client_factory: Any,
    tmp_path: Any,
    unused_tcp_port: int,
) -> None:
    index_path = tmp_path / "missing-index"
    client = client_factory(
        phase6c_enabled=True,
        global_model_enabled=False,
        reference_index_path=index_path,
        megaloc_worker_port=unused_tcp_port,
    )

    capabilities = client.get("/api/v1/capabilities")
    providers = client.get("/api/v1/providers")

    assert capabilities.status_code == 200
    capability_body = capabilities.json()["providers"]
    assert capability_body["geoclip_hierarchical"]["available"] is False
    assert capability_body["megaloc"]["available"] is False
    assert capability_body["reference_index"]["operational_status"] == "unavailable"
    assert providers.status_code == 200
    provider_ids = {item["provider_id"] for item in providers.json()["providers"]}
    assert {
        "geoclip_hierarchical_search",
        "megaloc_retrieval",
        "turkiye_megaloc_reference_index",
    } <= provider_ids
    assert str(index_path) not in capabilities.text
    assert str(index_path) not in providers.text


def test_phase6c_idempotency_cannot_reuse_legacy_pipeline_result(tmp_path: Any) -> None:
    database_url = f"sqlite:///{(tmp_path / 'atlaslens.db').as_posix()}"
    common = {
        "database_url": database_url,
        "openai_api_key": None,
        "ocr_enabled": False,
        "global_model_enabled": False,
        "phase6b_enabled": False,
        "sse_heartbeat_seconds": 0.05,
        "max_sse_lifetime_seconds": 2,
    }
    legacy_settings = Settings(
        _env_file=None,
        temp_storage_dir=tmp_path / "legacy-tmp",
        **common,
    )
    with TestClient(create_app(legacy_settings), raise_server_exceptions=False) as legacy:
        legacy_id = _submit(legacy, idempotency_key="shared-idempotency-key")

    phase6c_settings = Settings(
        _env_file=None,
        temp_storage_dir=tmp_path / "phase6c-tmp",
        phase6c_enabled=True,
        reference_index_path=tmp_path / "missing-index",
        megaloc_worker_enabled=False,
        **common,
    )
    with TestClient(create_app(phase6c_settings), raise_server_exceptions=False) as phase6c:
        phase6c_id = _submit(phase6c, idempotency_key="shared-idempotency-key")
        repeated_id = _submit(phase6c, idempotency_key="shared-idempotency-key")
        completed = wait_for_terminal(phase6c, phase6c_id)

    assert phase6c_id != legacy_id
    assert repeated_id == phase6c_id
    assert completed["status"] == "completed"
    assert completed["pipeline_version"] == "phase6c-v1"
    assert completed["phase6c"]["pipeline_version"] == "phase6c-v1"
    assert completed["phase6c"]["cache_fingerprint"].startswith("phase6c-v1:")
    assert any(
        item["provider_id"] == "megaloc_retrieval"
        for item in completed["phase6c"]["providers"]
    )


def test_phase6c_retained_source_rerun_keeps_phase6c_pipeline_version(
    client_factory: Any,
    tmp_path: Any,
) -> None:
    client = client_factory(
        phase6c_enabled=True,
        keep_uploads=True,
        global_model_enabled=False,
        phase6b_enabled=False,
        reference_index_path=tmp_path / "missing-index",
        megaloc_worker_enabled=False,
    )
    original_id = _submit(client, idempotency_key="phase6c-rerun-source")
    original = wait_for_terminal(client, original_id)

    response = client.post(f"/api/v1/analyses/{original_id}/rerun")
    assert response.status_code == 202
    rerun_id = response.json()["id"]
    queued = client.get(f"/api/v1/analyses/{rerun_id}").json()
    rerun = wait_for_terminal(client, rerun_id)

    assert queued["pipeline_version"] == "phase6c-v1"
    assert rerun["pipeline_version"] == "phase6c-v1"
    assert rerun["phase6c"]["pipeline_version"] == "phase6c-v1"
    assert rerun["image"]["sha256"] == original["image"]["sha256"]
