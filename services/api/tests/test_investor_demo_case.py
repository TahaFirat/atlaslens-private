from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlaslens_api.case_routes import router
from atlaslens_api.cases import (
    AdjudicationDecision,
    CalibrationState,
    CaseInvestigationService,
    MediaStorageState,
    SQLAlchemyCaseRepository,
)
from atlaslens_api.database import create_database_engine
from atlaslens_api.demo_case import (
    INVESTOR_DEMO_ANALYSIS_ID,
    INVESTOR_DEMO_CASE_ID,
    INVESTOR_DEMO_CASE_TITLE,
    INVESTOR_DEMO_MEDIA_ID,
    InvestorDemoCaseError,
    LockedPilotAssetIdentity,
    RetainedPilotAsset,
    load_retained_pilot_asset,
    prepare_investor_demo_case,
)
from atlaslens_api.mapillary_demo.api import (
    MAPILLARY_DEMO_INDEX_VERSION,
    MEGALOC_PHASE3B2_ARTIFACT_SHA256,
    MapillaryDemoCandidateResponse,
    MapillaryDemoQueryResponse,
    MapillaryDemoStatusResponse,
)
from atlaslens_api.repository import SQLAlchemyAnalysisRepository

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
_PUBLICATION_SHA = "a" * 64
_SOURCE_ASSET_ID = "test-only-holdout-source"


class _Runtime:
    def __init__(self, *, available: bool = True) -> None:
        self.query_calls = 0
        self.last_payload: bytes | None = None
        self.last_scope: str | None = None
        self._available = available

    async def status(self) -> MapillaryDemoStatusResponse:
        if not self._available:
            return MapillaryDemoStatusResponse(
                state="not_ready",
                enabled=True,
                available=False,
                reason_code="test_worker_unavailable",
                image_count=0,
            )
        return MapillaryDemoStatusResponse(
            state="active",
            enabled=True,
            available=True,
            reason_code=None,
            city="Ankara",
            image_count=3,
            model_version="test-only-megaloc-revision",
            model_artifact_sha256=MEGALOC_PHASE3B2_ARTIFACT_SHA256,
            index_version=MAPILLARY_DEMO_INDEX_VERSION,
            index_checksum=_PUBLICATION_SHA,
        )

    async def query(
        self, image_bytes: bytes, *, analysis_scope: str
    ) -> MapillaryDemoQueryResponse:
        assert analysis_scope == "ankara_reference_pilot"
        self.last_scope = analysis_scope
        self.query_calls += 1
        self.last_payload = bytes(image_bytes)
        return MapillaryDemoQueryResponse(
            status="completed",
            reason_code=None,
            analysis_scope="ankara_reference_pilot",
            coverage_status="pilot_eligible",
            abstained=False,
            abstention_reason=None,
            city="Ankara",
            index_version=MAPILLARY_DEMO_INDEX_VERSION,
            candidates=tuple(
                MapillaryDemoCandidateResponse(
                    rank=rank,
                    cosine_similarity=1.0 - distance,
                    cosine_distance=distance,
                    latitude=10.0 + rank,
                    longitude=20.0 + rank,
                    mapillary_image_id=f"test-only-reference-{rank}",
                    contributor="Test-only contributor",
                    source_url=(
                        "https://www.mapillary.com/app/"
                        f"?pKey=test-only-reference-{rank}"
                    ),
                    capture_date=None,
                    uncertainty_radius_m=1_000.0,
                )
                for rank, distance in ((1, 0.77), (2, 0.88), (3, 0.95))
            ),
        )


def _retained_fixture(root: Path, *, creator_id: str | None = "test-only-creator") -> bytes:
    payload = b"test-only-normalized-image-fixture"
    digest = hashlib.sha256(payload).hexdigest()
    asset_dir = root / "retained"
    sidecar_dir = root / "retained-sidecars"
    asset_dir.mkdir(parents=True)
    sidecar_dir.mkdir(parents=True)
    (asset_dir / "query.jpg").write_bytes(payload)
    sidecar = {
        "mapillary_image_id": _SOURCE_ASSET_ID,
        "computed_geometry": {"type": "Point", "coordinates": [2.5, 1.25]},
        "captured_at": "2025-01-02T03:04:05Z",
        "compass_angle": 10.0,
        "sequence_id": "test-only-sequence",
        "creator_id": creator_id,
        "width_px": 64,
        "height_px": 48,
        "source_page_url": (
            "https://www.mapillary.com/app/?pKey=test-only-holdout-source"
        ),
        "attribution_text": "Test-only contributor / Mapillary / CC BY-SA 4.0",
        "license_identifier": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "acquired_at": "2026-01-02T03:04:05Z",
        "raw_sha256": "b" * 64,
        "normalized_sha256": digest,
        "relative_path": "retained/query.jpg",
        "byte_size": len(payload) + 1,
        "normalized_byte_size": len(payload),
        "reconciliation_state": "active",
    }
    (sidecar_dir / f"{_SOURCE_ASSET_ID}.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return payload


def _identity_inventory(
    payload: bytes,
) -> tuple[tuple[LockedPilotAssetIdentity, ...], dict[str, LockedPilotAssetIdentity]]:
    reference = LockedPilotAssetIdentity(
        source_asset_id="test-only-reference",
        raw_sha256="c" * 64,
        normalized_sha256="d" * 64,
        perceptual_hash="f" * 16,
        sequence_id="test-only-reference-sequence",
        creator_id="test-only-reference-creator",
    )
    holdout = LockedPilotAssetIdentity(
        source_asset_id=_SOURCE_ASSET_ID,
        raw_sha256="b" * 64,
        normalized_sha256=hashlib.sha256(payload).hexdigest(),
        perceptual_hash="0" * 16,
        sequence_id="test-only-sequence",
        creator_id="test-only-creator",
    )
    return (reference,), {_SOURCE_ASSET_ID: holdout}


def _load_locked_asset(root: Path, payload: bytes) -> RetainedPilotAsset:
    references, holdout = _identity_inventory(payload)
    return load_retained_pilot_asset(
        root,
        approved_root=root,
        reference_image_ids={item.source_asset_id for item in references},
        reference_assets=references,
        locked_holdout_assets=holdout,
    )


async def _services():
    engine = create_database_engine("sqlite:///:memory:")
    analysis_repository = SQLAlchemyAnalysisRepository(engine)
    case_repository = SQLAlchemyCaseRepository(engine)
    await analysis_repository.initialize()
    await case_repository.initialize()
    service = CaseInvestigationService(
        case_repository, analysis_repository, clock=lambda: _NOW
    )
    return service, analysis_repository, engine


def test_retained_pilot_gate_uses_rights_metadata_without_exposing_ground_truth(
    tmp_path: Path,
) -> None:
    payload = _retained_fixture(tmp_path)
    asset = _load_locked_asset(tmp_path, payload)

    assert asset.path.name == "query.jpg"
    assert asset.creator_id == "test-only-creator"
    assert asset.byte_size > 0
    assert not hasattr(asset, "computed_geometry")
    assert not hasattr(asset, "latitude")
    assert not hasattr(asset, "longitude")

    with pytest.raises(InvestorDemoCaseError) as leakage:
        load_retained_pilot_asset(
            tmp_path,
            approved_root=tmp_path,
            reference_image_ids={_SOURCE_ASSET_ID},
        )
    assert leakage.value.code == "demo_query_reference_leakage"


def test_retained_pilot_gate_binds_locked_holdout_and_rejects_content_leakage(
    tmp_path: Path,
) -> None:
    payload = _retained_fixture(tmp_path)
    references, holdout = _identity_inventory(payload)
    colliding_reference = replace(
        references[0],
        normalized_sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(InvestorDemoCaseError) as exact_leakage:
        load_retained_pilot_asset(
            tmp_path,
            approved_root=tmp_path,
            reference_image_ids={colliding_reference.source_asset_id},
            reference_assets=(colliding_reference,),
            locked_holdout_assets=holdout,
        )
    assert exact_leakage.value.code == "demo_query_reference_leakage"

    near_duplicate_reference = replace(
        references[0],
        perceptual_hash="0" * 15 + "1",
    )
    with pytest.raises(InvestorDemoCaseError) as near_leakage:
        load_retained_pilot_asset(
            tmp_path,
            approved_root=tmp_path,
            reference_image_ids={near_duplicate_reference.source_asset_id},
            reference_assets=(near_duplicate_reference,),
            locked_holdout_assets=holdout,
        )
    assert near_leakage.value.code == "demo_query_reference_leakage"

    shared_contributor_reference = replace(
        references[0], creator_id="test-only-creator"
    )
    with pytest.raises(InvestorDemoCaseError) as lineage_leakage:
        load_retained_pilot_asset(
            tmp_path,
            approved_root=tmp_path,
            reference_image_ids={shared_contributor_reference.source_asset_id},
            reference_assets=(shared_contributor_reference,),
            locked_holdout_assets=holdout,
        )
    assert lineage_leakage.value.code == "demo_query_reference_lineage_leakage"

    with pytest.raises(InvestorDemoCaseError) as unlocked:
        load_retained_pilot_asset(
            tmp_path,
            approved_root=tmp_path,
            reference_image_ids={references[0].source_asset_id},
            reference_assets=references,
            locked_holdout_assets={"different-holdout": next(iter(holdout.values()))},
        )
    assert unlocked.value.code == "demo_query_not_locked_holdout"


def test_retained_pilot_gate_fails_closed_without_complete_attribution(
    tmp_path: Path,
) -> None:
    _retained_fixture(tmp_path, creator_id=None)

    with pytest.raises(InvestorDemoCaseError) as missing:
        load_retained_pilot_asset(
            tmp_path, approved_root=tmp_path, reference_image_ids=set()
        )
    assert missing.value.code == "demo_attribution_incomplete"


async def test_prepare_creates_one_real_canonical_case_and_is_idempotent(
    tmp_path: Path,
) -> None:
    payload = _retained_fixture(tmp_path)
    asset = _load_locked_asset(tmp_path, payload)
    service, analysis_repository, engine = await _services()
    runtime = _Runtime()
    try:
        first = await prepare_investor_demo_case(
            service=service,
            analysis_repository=analysis_repository,
            runtime=runtime,
            asset=asset,
            publication_sha256=_PUBLICATION_SHA,
            clock=lambda: _NOW,
        )
        first_audit, first_audit_total = await service.list_audit_events(
            INVESTOR_DEMO_CASE_ID
        )
        second = await prepare_investor_demo_case(
            service=service,
            analysis_repository=analysis_repository,
            runtime=runtime,
            asset=asset,
            publication_sha256=_PUBLICATION_SHA,
            clock=lambda: _NOW,
        )

        assert first.status == "created"
        assert second.status == "existing"
        assert runtime.query_calls == 1
        assert runtime.last_payload == payload
        assert runtime.last_scope == "ankara_reference_pilot"
        cases, case_total = await service.list_cases()
        assert case_total == 1
        assert cases[0].id == INVESTOR_DEMO_CASE_ID
        assert cases[0].title == INVESTOR_DEMO_CASE_TITLE
        assert cases[0].authorization_attested is True
        assert cases[0].adjudication_count == 0

        media, media_total = await service.list_media(INVESTOR_DEMO_CASE_ID)
        assert media_total == 1
        assert media[0].id == INVESTOR_DEMO_MEDIA_ID
        assert media[0].analysis_id == INVESTOR_DEMO_ANALYSIS_ID
        assert media[0].storage_state is MediaStorageState.DELETED_AFTER_ANALYSIS
        assert media[0].original_filename_display == "mapillary-pilot-query.jpg"
        assert media[0].source_url == asset.source_page_url
        assert str(tmp_path) not in (media[0].source_description or "")

        app = FastAPI()
        app.state.services = SimpleNamespace(case_service=service)
        app.include_router(router)
        with TestClient(app, raise_server_exceptions=False) as client:
            media_response = client.get(
                f"/api/v1/cases/{INVESTOR_DEMO_CASE_ID}/media?limit=100&offset=0"
            )
            evidence_response = client.get(
                f"/api/v1/cases/{INVESTOR_DEMO_CASE_ID}/evidence?limit=100&offset=0"
            )
        assert media_response.status_code == 200
        assert media_response.json()["total"] == 1
        assert media_response.json()["items"][0]["source_url"] == asset.source_page_url
        assert evidence_response.status_code == 200
        public_context = evidence_response.json()["items"][0]["structured_payload"]
        assert public_context["analysis_scope"] == "ankara_reference_pilot"
        assert public_context["coverage_label"] == "Ankara reference pilot"
        assert public_context["result_semantics"].endswith("not_general_geolocation")
        assert public_context["similarity_semantics"] == (
            "cosine_similarity_not_confidence"
        )
        assert all(
            value is None or isinstance(value, str | int | float | bool)
            for value in public_context.values()
        )

        stored = await analysis_repository.get(INVESTOR_DEMO_ANALYSIS_ID)
        assert stored is not None
        assert stored.storage_key is None
        assert stored.analysis.analysis_mode.value == "local_only"
        assert stored.analysis.result_classification == "real"
        assert stored.analysis.simulation is None
        assert stored.analysis.image is None
        retrieval_context = stored.analysis.evidence[0].retrieval_context
        assert retrieval_context is not None
        assert retrieval_context.coverage_label == "Ankara reference pilot"
        assert retrieval_context.similarity_semantics == (
            "cosine_similarity_not_confidence"
        )
        assert stored.analysis.candidates[0].label is None
        assert stored.analysis.candidates[0].country_code is None
        assert stored.analysis.candidates[0].confidence is None
        assert stored.analysis.candidates[0].radius_km == 25.0
        assert stored.analysis.evidence[0].provenance.execution_boundary == "local"

        evidence, evidence_total = await service.list_evidence(INVESTOR_DEMO_CASE_ID)
        hypotheses, hypothesis_total = await service.list_hypotheses(
            INVESTOR_DEMO_CASE_ID
        )
        assert evidence_total == hypothesis_total == 3
        assert evidence[0].provider == "mapillary-private-demo-retrieval"
        assert evidence[0].structured_payload["analysis_scope"] == (
            "ankara_reference_pilot"
        )
        assert evidence[0].structured_payload["coverage_status"] == "pilot_eligible"
        assert evidence[0].structured_payload["similarity_semantics"] == (
            "cosine_similarity_not_confidence"
        )
        assert hypotheses[0].calibration_state is CalibrationState.UNCALIBRATED
        assert sorted(item.rank for item in hypotheses if item.rank is not None) == [
            1,
            2,
            3,
        ]
        assert hypotheses[0].confidence_label == "uncalibrated"
        assert hypotheses[0].uncertainty_radius_m == 25_000.0
        assert (await service.verify_audit_integrity(INVESTOR_DEMO_CASE_ID)).valid
        second_audit, second_audit_total = await service.list_audit_events(
            INVESTOR_DEMO_CASE_ID
        )
        assert second_audit_total == first_audit_total
        assert [event.id for event in second_audit] == [event.id for event in first_audit]

        safe_output = json.dumps(first.safe_payload(), sort_keys=True)
        assert "latitude" not in safe_output
        assert "longitude" not in safe_output
        assert "https://" not in safe_output
        assert str(tmp_path) not in safe_output
        assert _SOURCE_ASSET_ID not in safe_output
    finally:
        engine.dispose()


async def test_prepare_fails_closed_when_runtime_is_not_ready(tmp_path: Path) -> None:
    payload = _retained_fixture(tmp_path)
    asset = _load_locked_asset(tmp_path, payload)
    service, analysis_repository, engine = await _services()
    runtime = _Runtime(available=False)
    try:
        with pytest.raises(InvestorDemoCaseError) as unavailable:
            await prepare_investor_demo_case(
                service=service,
                analysis_repository=analysis_repository,
                runtime=runtime,
                asset=asset,
                publication_sha256=_PUBLICATION_SHA,
                clock=lambda: _NOW,
            )
        assert unavailable.value.code == "demo_runtime_not_ready"
        assert (await service.list_cases())[1] == 0
        assert runtime.query_calls == 0
    finally:
        engine.dispose()


async def test_prepare_rejects_same_hash_analysis_semantic_drift(tmp_path: Path) -> None:
    payload = _retained_fixture(tmp_path)
    asset = _load_locked_asset(tmp_path, payload)
    service, analysis_repository, engine = await _services()
    runtime = _Runtime()
    try:
        await prepare_investor_demo_case(
            service=service,
            analysis_repository=analysis_repository,
            runtime=runtime,
            asset=asset,
            publication_sha256=_PUBLICATION_SHA,
            clock=lambda: _NOW,
        )
        stored = await analysis_repository.get(INVESTOR_DEMO_ANALYSIS_ID)
        assert stored is not None
        altered_candidate = stored.analysis.candidates[0].model_copy(
            update={"country_code": "TR", "label": "Test-only injected truth"}
        )
        await analysis_repository.save(
            stored.analysis.model_copy(update={"candidates": [altered_candidate]})
        )

        with pytest.raises(InvestorDemoCaseError) as drift:
            await prepare_investor_demo_case(
                service=service,
                analysis_repository=analysis_repository,
                runtime=runtime,
                asset=asset,
                publication_sha256=_PUBLICATION_SHA,
                clock=lambda: _NOW,
            )
        assert drift.value.code == "demo_existing_analysis_conflict"
        assert runtime.query_calls == 1
    finally:
        engine.dispose()


async def test_prepare_refuses_to_erase_append_only_operator_history(
    tmp_path: Path,
) -> None:
    payload = _retained_fixture(tmp_path)
    asset = _load_locked_asset(tmp_path, payload)
    service, analysis_repository, engine = await _services()
    runtime = _Runtime()
    try:
        await prepare_investor_demo_case(
            service=service,
            analysis_repository=analysis_repository,
            runtime=runtime,
            asset=asset,
            publication_sha256=_PUBLICATION_SHA,
            clock=lambda: _NOW,
        )
        hypotheses, _ = await service.list_hypotheses(INVESTOR_DEMO_CASE_ID)
        await service.add_adjudication(
            INVESTOR_DEMO_CASE_ID,
            hypotheses[0].id,
            actor_id="test-only-operator",
            decision=AdjudicationDecision.NEEDS_MORE_EVIDENCE,
            rationale="Test-only review requires another independent source.",
        )

        with pytest.raises(InvestorDemoCaseError) as drift:
            await prepare_investor_demo_case(
                service=service,
                analysis_repository=analysis_repository,
                runtime=runtime,
                asset=asset,
                publication_sha256=_PUBLICATION_SHA,
                clock=lambda: _NOW,
            )
        assert drift.value.code == "demo_existing_case_conflict"
        assert runtime.query_calls == 1
        assert (await service.list_cases())[1] == 1
        assert (await service.verify_audit_integrity(INVESTOR_DEMO_CASE_ID)).valid
    finally:
        engine.dispose()
