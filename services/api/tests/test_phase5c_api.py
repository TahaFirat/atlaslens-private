from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pydantic import ValidationError

from atlaslens_api.config import Settings
from conftest import image_bytes, upload, wait_for_terminal


def test_history_lists_safe_summary_and_delete_removes_it(client: TestClient) -> None:
    accepted = upload(client, image_bytes())
    assert accepted.status_code == 202
    analysis_id = accepted.json()["id"]
    wait_for_terminal(client, analysis_id)

    history = client.get("/api/v1/analyses")
    assert history.status_code == 200
    item = history.json()["items"][0]
    assert item["id"] == analysis_id
    assert item["result_classification"] == "real"
    assert item["source_retained"] is False
    assert "filename" not in item
    assert client.post(f"/api/v1/analyses/{analysis_id}/rerun").status_code == 409

    assert client.delete(f"/api/v1/analyses/{analysis_id}").status_code == 200
    assert client.get("/api/v1/analyses").json()["total"] == 0


def test_retained_local_source_can_rerun_as_a_new_analysis(client_factory: Any) -> None:
    client = client_factory(keep_uploads=True)
    accepted = upload(client, image_bytes())
    analysis_id = accepted.json()["id"]
    wait_for_terminal(client, analysis_id)
    assert client.get("/api/v1/analyses").json()["items"][0]["source_retained"] is True

    rerun = client.post(f"/api/v1/analyses/{analysis_id}/rerun")
    assert rerun.status_code == 202
    rerun_id = rerun.json()["id"]
    assert rerun_id != analysis_id
    rerun_result = wait_for_terminal(client, rerun_id)
    assert rerun_result["image"]["sha256"] == wait_for_terminal(client, analysis_id)[
        "image"
    ]["sha256"]


def test_explicit_development_mock_is_watermarked_and_history_labeled(
    client_factory: Any,
) -> None:
    client = client_factory(
        app_env="development",
        enable_mock_inference=True,
        mock_scenario="turkiye_kayseri_demo",
    )
    accepted = upload(client, image_bytes())
    result = wait_for_terminal(client, accepted.json()["id"])

    assert result["result_classification"] == "simulated"
    assert result["simulation"] == {
        "scenario_id": "turkiye_kayseri_demo",
        "warning_key": "warning.simulated_development_result",
        "watermark": "SIMULATED DEVELOPMENT RESULT",
    }
    assert "warning.simulated_development_result" in result["warnings"]
    history = client.get("/api/v1/analyses?classification=simulated").json()
    assert history["total"] == 1
    assert history["items"][0]["result_classification"] == "simulated"


def test_provider_status_is_safe_and_mock_absent_by_default(client: TestClient) -> None:
    response = client.get("/api/v1/providers")
    assert response.status_code == 200
    providers = response.json()["providers"]
    assert providers
    assert all(item["classification"] == "real" for item in providers)
    assert all("path" not in item and "artifact_path" not in item for item in providers)


def test_operator_catalogs_are_disabled_by_default_and_empty_when_enabled(
    client_factory: Any, tmp_path: Path
) -> None:
    default_client = client_factory()
    assert default_client.get("/api/v1/models").status_code == 404
    assert default_client.get("/api/v1/evaluations").status_code == 404
    assert default_client.get("/api/v1/datasets/qa").status_code == 404

    evaluation_root = tmp_path / "evaluation-reports"
    qa_root = tmp_path / "qa-reports"
    evaluation_root.mkdir()
    qa_root.mkdir()
    operator_client = client_factory(
        app_env="development",
        operator_api_enabled=True,
        evaluation_report_dir=evaluation_root,
        dataset_qa_report_dir=qa_root,
    )
    models = operator_client.get("/api/v1/models")
    assert models.status_code == 200
    assert models.json()["models"][0]["model_id"] == "atlaslens-custom-geolocation"
    assert operator_client.get("/api/v1/evaluations").json() == {"reports": []}
    assert operator_client.get("/api/v1/datasets/qa").json() == {"reports": []}
    assert operator_client.get("/api/v1/evaluations/missing").status_code == 404
    assert operator_client.get("/api/v1/datasets/qa/missing").status_code == 404


def test_production_refuses_unauthenticated_operator_api() -> None:
    try:
        Settings(_env_file=None, app_env="production", operator_api_enabled=True)
    except ValidationError as error:
        assert "operator API is forbidden" in str(error)
    else:
        raise AssertionError("production operator API must fail configuration")
