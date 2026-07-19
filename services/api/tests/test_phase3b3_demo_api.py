from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from atlaslens_api.config import Settings
from atlaslens_api.image_processing import SafeImageProcessor
from atlaslens_api.main import create_app
from atlaslens_api.mapillary_demo.api import (
    MAPILLARY_DEMO_INDEX_VERSION,
    MEGALOC_PHASE3B2_ARTIFACT_SHA256,
    DisabledMapillaryDemoIndex,
    MapillaryDemoCandidateResponse,
    MapillaryDemoIndexStatus,
    MapillaryDemoRuntime,
    PublishedMapillaryDemoSearchIndex,
)
from atlaslens_api.phase6b.worker_client import WorkerHealth
from atlaslens_api.storage import LocalTemporaryStorage
from conftest import image_bytes

CONTRACT = Path(__file__).resolve().parents[3] / "packages" / "contracts" / "openapi.yaml"


def _descriptor() -> tuple[float, ...]:
    return (1.0,) + (0.0,) * 8_447


def _without_docs(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_docs(nested)
            for key, nested in value.items()
            if key not in {"description", "summary"}
        }
    if isinstance(value, list):
        return [_without_docs(nested) for nested in value]
    return value


def _health(*, ready: bool = True) -> WorkerHealth:
    return WorkerHealth(
        schema_version="atlaslens-worker-v1",
        provider="megaloc",
        provider_revision="phase3b3-test-source",
        model_revision="phase3b3-test-model",
        device="cpu",
        process_running=True,
        import_ok=ready,
        weights_available=ready,
        model_loaded=ready,
        load_verified=ready,
        real_inference_verified=ready,
        last_error=None,
    )


class _Worker:
    def __init__(self, *, ready: bool = True) -> None:
        self.worker_health = _health(ready=ready)
        self.calls: list[str] = []
        self.request_fingerprints: list[str] = []
        self.closed = False

    async def health(self, **_: Any) -> WorkerHealth:
        self.calls.append("health")
        return self.worker_health

    async def load(self, device: str, **_: Any) -> None:
        assert device == "cpu"
        self.calls.append("load")

    async def describe(self, image_bytes: bytes, *, device: str, **_: Any) -> tuple[float, ...]:
        assert device == "cpu"
        assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        self.calls.append("describe")
        self.request_fingerprints.append(hashlib.sha256(image_bytes).hexdigest())
        return _descriptor()

    async def unload(self, device: str, **_: Any) -> None:
        assert device == "cpu"
        self.calls.append("unload")

    async def close(self) -> None:
        self.closed = True


class _ReadyIndex:
    def status(self) -> MapillaryDemoIndexStatus:
        return MapillaryDemoIndexStatus(
            state="active",
            enabled=True,
            available=True,
            reason_code=None,
            city="Ankara",
            image_count=1_500,
            model_version="megaloc-phase3b2-mapillary-private-demo-v1",
            model_artifact_sha256=MEGALOC_PHASE3B2_ARTIFACT_SHA256,
            index_version=MAPILLARY_DEMO_INDEX_VERSION,
            index_checksum="a" * 64,
        )

    def search(
        self, descriptor: Sequence[float], *, top_k: int
    ) -> tuple[MapillaryDemoCandidateResponse, ...]:
        assert len(descriptor) == 8_448
        assert top_k == 5
        return (
            MapillaryDemoCandidateResponse(
                rank=1,
                cosine_similarity=0.75,
                cosine_distance=0.25,
                latitude=38.7225,
                longitude=35.4875,
                mapillary_image_id="stable-mapillary-id",
                contributor="mapillary-contributor-id",
                source_url="https://www.mapillary.com/app/?pKey=stable-mapillary-id",
                capture_date=date(2024, 5, 12),
                uncertainty_radius_m=1_000.0,
            ),
        )


class _MisorderedIndex(_ReadyIndex):
    def search(
        self, descriptor: Sequence[float], *, top_k: int
    ) -> tuple[MapillaryDemoCandidateResponse, ...]:
        assert len(descriptor) == 8_448
        assert top_k == 5
        return tuple(
            MapillaryDemoCandidateResponse(
                rank=rank,
                cosine_similarity=1.0 - distance,
                cosine_distance=distance,
                latitude=38.7 + rank / 100,
                longitude=35.4 + rank / 100,
                mapillary_image_id=f"test-reference-{rank}",
                contributor="test-contributor",
                source_url=f"https://www.mapillary.com/app/?pKey=test-reference-{rank}",
                capture_date=None,
                uncertainty_radius_m=1_000.0,
            )
            for rank, distance in ((1, 0.4), (2, 0.2))
        )


def _runtime(*, worker: _Worker | None = None) -> MapillaryDemoRuntime:
    return MapillaryDemoRuntime(
        index=_ReadyIndex(),
        worker=worker or _Worker(),
        device="cpu",
        timeout_seconds=5.0,
        top_k=5,
    )


async def test_candidate_rank_similarity_drift_fails_closed() -> None:
    worker = _Worker()
    runtime = MapillaryDemoRuntime(
        index=_MisorderedIndex(),
        worker=worker,
        device="cpu",
        timeout_seconds=5.0,
        top_k=5,
    )

    result = await runtime.query(
        image_bytes("PNG"), analysis_scope="ankara_reference_pilot"
    )

    assert result.status == "failed"
    assert result.reason_code == "invalid_candidate_ordering"
    assert result.abstained is True
    assert result.candidates == ()


def test_demo_is_disabled_by_default_and_does_not_disturb_existing_api(client: TestClient) -> None:
    status = client.get("/api/v1/mapillary-demo/status")
    assert status.status_code == 200
    assert status.json()["state"] == "disabled"
    assert client.get("/api/v1/health").status_code == 200
    capabilities = client.get("/api/v1/capabilities").json()
    assert capabilities["providers"]["turkiye_mapillary_demo"]["enabled"] is False


def test_demo_openapi_contract_matches_runtime(client: TestClient) -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    generated = client.app.openapi()  # type: ignore[attr-defined]
    expected_operations = {
        ("/api/v1/mapillary-demo/status", "get"): "getMapillaryDemoStatus",
        ("/api/v1/mapillary-demo/query", "post"): "queryMapillaryDemo",
    }
    for (path, method), operation_id in expected_operations.items():
        assert contract["paths"][path][method]["operationId"] == operation_id
        assert generated["paths"][path][method]["operationId"] == operation_id
    assert contract["paths"]["/api/v1/mapillary-demo/status"]["get"]["responses"][
        "403"
    ] == {"$ref": "#/components/responses/ProblemResponse"}
    generated_status = generated["paths"]["/api/v1/mapillary-demo/status"]["get"]
    assert generated_status["responses"]["403"]["content"][
        "application/problem+json"
    ]["schema"] == {"$ref": "#/components/schemas/ProblemDetails"}
    generated_query = generated["paths"]["/api/v1/mapillary-demo/query"]["post"]
    for status in ("403", "413", "415", "422"):
        assert contract["paths"]["/api/v1/mapillary-demo/query"]["post"]["responses"][status] == {
            "$ref": "#/components/responses/ProblemResponse"
        }
        assert generated_query["responses"][status]["content"][
            "application/problem+json"
        ]["schema"] == {"$ref": "#/components/schemas/ProblemDetails"}
    for schema_name in (
        "Body_queryMapillaryDemo",
        "MapillaryDemoCandidateResponse",
        "MapillaryDemoQueryResponse",
        "MapillaryDemoStatusResponse",
    ):
        assert _without_docs(contract["components"]["schemas"][schema_name]) == _without_docs(
            generated["components"]["schemas"][schema_name]
        )


def test_enabled_demo_fails_closed_when_index_is_not_ready_and_hides_token(
    client_factory: Any,
) -> None:
    runtime = MapillaryDemoRuntime(
        index=DisabledMapillaryDemoIndex(enabled=True, reason_code="demo_index_incompatible"),
        worker=None,
        device="cpu",
        timeout_seconds=5.0,
    )
    fake_token = "obviously-fake-mapillary-token-for-tests"
    client = client_factory(
        mapillary_demo_provider=runtime,
        mapillary_access_token=fake_token,
    )
    status = client.get("/api/v1/mapillary-demo/status")
    assert status.status_code == 200
    assert status.json()["state"] == "not_ready"
    assert status.json()["reason_code"] == "demo_index_incompatible"
    assert fake_token not in status.text
    capabilities = client.get("/api/v1/capabilities")
    assert fake_token not in capabilities.text
    readiness = client.get("/api/v1/ready")
    assert readiness.status_code == 503
    assert readiness.json()["checks"]["turkiye_mapillary_demo"] == "error"


def test_published_demo_requires_explicit_source_policy_and_selection_hashes(
    tmp_path: Path,
) -> None:
    index = PublishedMapillaryDemoSearchIndex(
        enabled=True,
        bundle_path=tmp_path / "bundle",
        expected_publication_sha256="a" * 64,
        expected_source_policy_sha256="",
        expected_selection_lock_sha256="b" * 64,
        uncertainty_radius_m=1_000.0,
    )
    assert index.status().reason_code == "source_policy_checksum_not_configured"


@pytest.mark.parametrize(
    "overrides",
    (
        {"app_env": "production"},
        {"app_env": "development", "api_host": "0.0.0.0"},
        {"app_env": "development", "phase6c_enabled": True},
    ),
)
def test_private_demo_refuses_production_public_bind_and_phase6c_coactivation(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            atlaslens_turkiye_demo_enabled=True,
            **overrides,
        )


def test_demo_rejects_non_loopback_client_even_when_configured_host_is_loopback(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "remote-query-storage"
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'atlaslens.db').as_posix()}",
        temp_storage_dir=storage,
        openai_api_key=None,
        global_model_enabled=False,
        phase6b_enabled=False,
    )
    with TestClient(
        create_app(settings),
        raise_server_exceptions=False,
        client=("203.0.113.10", 50_000),
    ) as client:
        status = client.get("/api/v1/mapillary-demo/status")
        query = client.post(
            "/api/v1/mapillary-demo/query",
            files={"image": ("private-query.png", image_bytes("PNG"), "image/png")},
            data={"authorization_acknowledged": "true"},
        )
    assert status.status_code == 403
    assert query.status_code == 403
    assert status.headers["content-type"].startswith("application/problem+json")
    assert query.headers["content-type"].startswith("application/problem+json")
    assert list(storage.iterdir()) == []


def test_not_ready_demo_abstains_before_storing_upload(
    monkeypatch: pytest.MonkeyPatch,
    client_factory: Any,
) -> None:
    async def forbidden_save(*_: Any, **__: Any) -> None:
        raise AssertionError("not-ready demo must not persist the upload")

    monkeypatch.setattr(LocalTemporaryStorage, "save_upload", forbidden_save)
    runtime = MapillaryDemoRuntime(
        index=DisabledMapillaryDemoIndex(enabled=True, reason_code="demo_index_incompatible"),
        worker=None,
        device="cpu",
        timeout_seconds=5.0,
    )
    client = client_factory(mapillary_demo_provider=runtime)
    response = client.post(
        "/api/v1/mapillary-demo/query",
        files={"image": ("private-query.png", image_bytes("PNG"), "image/png")},
        data={
            "authorization_acknowledged": "true",
            "analysis_scope": "ankara_reference_pilot",
        },
    )
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["reason_code"] == "demo_index_incompatible"


def test_generic_upload_abstains_on_ankara_only_coverage_before_storage_or_decode(
    monkeypatch: pytest.MonkeyPatch,
    client_factory: Any,
) -> None:
    async def forbidden(*_: Any, **__: Any) -> None:
        raise AssertionError("coverage-abstained upload must not be stored or decoded")

    monkeypatch.setattr(LocalTemporaryStorage, "save_upload", forbidden)
    monkeypatch.setattr(SafeImageProcessor, "prepare", forbidden)
    worker = _Worker()
    client = client_factory(mapillary_demo_provider=_runtime(worker=worker))

    response = client.post(
        "/api/v1/mapillary-demo/query",
        files={"image": ("generic-place.png", image_bytes("PNG"), "image/png")},
        data={"authorization_acknowledged": "true"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "abstained"
    assert body["analysis_scope"] == "generic_upload"
    assert body["coverage_status"] == "insufficient"
    assert body["reason_code"] == "reference_coverage_insufficient"
    assert body["abstention_reason"] == "reference_coverage_insufficient"
    assert body["candidates"] == []
    assert worker.request_fingerprints == []
    assert "load" not in worker.calls
    assert "describe" not in worker.calls


def test_pilot_provider_request_is_pixel_only_across_context_variants(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "private-query-storage"
    worker = _Worker()
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'atlaslens.db').as_posix()}",
        temp_storage_dir=storage,
        openai_api_key=None,
        global_model_enabled=False,
        phase6b_enabled=False,
        app_env="development",
        atlaslens_turkiye_demo_enabled=True,
    )
    entries = (
        (
            "neutral-name.png",
            {
                "case_title": "Context A",
                "locale": "tr",
                "demo": "investor",
                "caseId": "00000000-0000-4000-8000-000000000001",
                "source_context": "Context A",
            },
        ),
        (
            "different-name.png",
            {
                "case_title": "Context B",
                "locale": "en",
                "demo": "normal",
                "caseId": "00000000-0000-4000-8000-000000000002",
                "source_context": "Context B",
            },
        ),
    )
    bodies: list[dict[str, object]] = []
    with TestClient(
        create_app(settings, mapillary_demo_provider=_runtime(worker=worker)),
        raise_server_exceptions=False,
    ) as client:
        for filename, context in entries:
            response = client.post(
                "/api/v1/mapillary-demo/query",
                files={"image": (filename, image_bytes("PNG"), "image/png")},
                data={
                    "authorization_acknowledged": "true",
                    "analysis_scope": "ankara_reference_pilot",
                    **context,
                },
            )
            assert response.status_code == 200
            bodies.append(response.json())

    assert len(worker.request_fingerprints) == 2
    assert len(set(worker.request_fingerprints)) == 1
    assert bodies[0]["candidates"] == bodies[1]["candidates"]
    assert bodies[0]["analysis_scope"] == bodies[1]["analysis_scope"]
    assert list(storage.iterdir()) == []


def test_ready_demo_returns_attributed_uncalibrated_result_and_deletes_query_files(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "private-query-storage"
    database = tmp_path / "atlaslens.db"
    worker = _Worker()
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{database.as_posix()}",
        temp_storage_dir=storage,
        openai_api_key=None,
        global_model_enabled=False,
        phase6b_enabled=False,
        app_env="development",
        atlaslens_turkiye_demo_enabled=True,
    )
    with TestClient(
        create_app(settings, mapillary_demo_provider=_runtime(worker=worker)),
        raise_server_exceptions=False,
    ) as client:
        status = client.get("/api/v1/mapillary-demo/status")
        assert status.status_code == 200
        assert status.json()["state"] == "active"
        assert status.json()["city"] == "Ankara"
        response = client.post(
            "/api/v1/mapillary-demo/query",
            files={"image": ("private-query.png", image_bytes("PNG"), "image/png")},
            data={
                "authorization_acknowledged": "true",
                "analysis_scope": "ankara_reference_pilot",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["abstained"] is False
        assert body["analysis_scope"] == "ankara_reference_pilot"
        assert body["coverage_status"] == "pilot_eligible"
        assert body["coverage_label"] == "Ankara reference pilot"
        assert body["retrieval_scope"] == "ankara_reference_collection"
        assert body["similarity_semantics"] == "cosine_similarity_not_confidence"
        assert body["result_semantics"].endswith("not_general_geolocation")
        candidate = body["candidates"][0]
        assert candidate["rank"] == 1
        assert candidate["confidence"] is None
        assert candidate["confidence_semantics"] == "uncalibrated_unavailable"
        assert candidate["similarity_semantics"] == "cosine_similarity_not_confidence"
        assert candidate["uncertainty_radius_m"] > 0
        assert candidate["mapillary_image_id"] == "stable-mapillary-id"
        assert candidate["source_url"].startswith("https://www.mapillary.com/app/")
        assert candidate["license_identifier"] == "CC-BY-SA-4.0"
        assert "access_token" not in response.text.casefold()
        assert worker.calls[-3:] == ["load", "describe", "unload"]
        assert list(storage.iterdir()) == []


def test_raw_upload_deletion_is_attempted_when_normalized_deletion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "private-query-storage"
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'atlaslens.db').as_posix()}",
        temp_storage_dir=storage,
        openai_api_key=None,
        global_model_enabled=False,
        phase6b_enabled=False,
        app_env="development",
        atlaslens_turkiye_demo_enabled=True,
    )
    original_delete = LocalTemporaryStorage.delete
    delete_attempts: list[str | None] = []

    async def fail_normalized_delete(
        self: LocalTemporaryStorage, key: str | None
    ) -> None:
        delete_attempts.append(key)
        if key is not None and not key.endswith(".upload"):
            raise OSError("simulated normalized cleanup failure")
        await original_delete(self, key)

    monkeypatch.setattr(LocalTemporaryStorage, "delete", fail_normalized_delete)
    with TestClient(
        create_app(settings, mapillary_demo_provider=_runtime()),
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/api/v1/mapillary-demo/query",
            files={"image": ("private-query.png", image_bytes("PNG"), "image/png")},
            data={
                "authorization_acknowledged": "true",
                "analysis_scope": "ankara_reference_pilot",
            },
        )
    assert response.status_code == 500
    assert any(key is not None and key.endswith(".upload") for key in delete_attempts)
    assert list(storage.glob("*.upload")) == []


def test_query_requires_authorization_before_storing_or_inference(
    client_factory: Any,
) -> None:
    worker = _Worker()
    client = client_factory(mapillary_demo_provider=_runtime(worker=worker))
    response = client.post(
        "/api/v1/mapillary-demo/query",
        files={"image": ("private-query.png", image_bytes("PNG"), "image/png")},
        data={"authorization_acknowledged": "false"},
    )
    assert response.status_code == 422
    assert "describe" not in worker.calls
