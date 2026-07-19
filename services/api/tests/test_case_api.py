from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from atlaslens_api.case_routes import router
from atlaslens_api.case_schemas import EvidenceView
from atlaslens_api.cases.domain import (
    DEFAULT_WORKSPACE_ID,
    ActorType,
    AdjudicationDecision,
    AdjudicationRecord,
    AuditEventRecord,
    AuditIntegrityResult,
    CalibrationState,
    CaseMediaRecord,
    CaseNotFoundError,
    CasePurpose,
    CaseRecord,
    CaseRelationshipError,
    CaseSensitivity,
    CaseStatus,
    EvidenceRecord,
    EvidenceType,
    HypothesisOrigin,
    LocationHypothesisRecord,
    MaterializationResult,
    MediaSourceType,
    MediaStorageState,
    MediaType,
    OperatorCorrectionResult,
)
from atlaslens_api.errors import AppError

CASE_ID = UUID("10000000-0000-4000-8000-000000000001")
MEDIA_ID = UUID("20000000-0000-4000-8000-000000000001")
ANALYSIS_ID = UUID("30000000-0000-4000-8000-000000000001")
EVIDENCE_ID = UUID("40000000-0000-4000-8000-000000000001")
HYPOTHESIS_ID = UUID("50000000-0000-4000-8000-000000000001")
ADJUDICATION_ID = UUID("60000000-0000-4000-8000-000000000001")
AUDIT_ID = UUID("70000000-0000-4000-8000-000000000001")
NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


def _case() -> CaseRecord:
    return CaseRecord(
        id=CASE_ID,
        workspace_id=DEFAULT_WORKSPACE_ID,
        title="Synthetic investigation",
        description=None,
        purpose=CasePurpose.JOURNALISM,
        purpose_detail=None,
        source_context="Explicitly synthetic API fixture.",
        authorization_attested=True,
        status=CaseStatus.OPEN,
        sensitivity=CaseSensitivity.STANDARD,
        created_at=NOW,
        updated_at=NOW,
        closed_at=None,
        created_by_actor_id="local-analyst",
        retention_policy="metadata-default-v1",
        version=1,
        media_count=1,
        evidence_count=1,
        hypothesis_count=1,
        adjudication_count=1,
        latest_adjudication_decision=AdjudicationDecision.NEEDS_MORE_EVIDENCE,
    )


def _media() -> CaseMediaRecord:
    return CaseMediaRecord(
        id=MEDIA_ID,
        case_id=CASE_ID,
        analysis_id=None,
        media_type=MediaType.IMAGE,
        source_type=MediaSourceType.UPLOAD,
        original_filename_display="synthetic.jpg",
        mime_type="image/jpeg",
        byte_size=1234,
        sha256="a" * 64,
        captured_at=None,
        received_at=NOW,
        source_url=None,
        archive_url=None,
        source_description="Synthetic fixture metadata only.",
        authorization_attested=True,
        storage_state=MediaStorageState.DELETED_AFTER_ANALYSIS,
        analysis_status_at_link=None,
        analysis_created_at=None,
        analysis_expires_at=None,
        materialized_at=None,
        created_at=NOW,
    )


def _evidence() -> EvidenceRecord:
    return EvidenceRecord(
        id=EVIDENCE_ID,
        case_id=CASE_ID,
        media_id=MEDIA_ID,
        analysis_id=ANALYSIS_ID,
        evidence_type=EvidenceType.VISUAL_CLUE,
        provider="synthetic-test-provider",
        provider_family="synthetic-test-family",
        summary="Synthetic fixture evidence; not an OSINT result.",
        structured_payload={"fixture": True},
        provenance={"source": "synthetic_test"},
        observed_at=NOW,
        created_at=NOW,
        immutable_source_hash="b" * 64,
    )


def _hypothesis(*, operator: bool = False) -> LocationHypothesisRecord:
    return LocationHypothesisRecord(
        id=HYPOTHESIS_ID if not operator else UUID("50000000-0000-4000-8000-000000000002"),
        case_id=CASE_ID,
        media_id=MEDIA_ID,
        analysis_id=ANALYSIS_ID,
        origin=HypothesisOrigin.OPERATOR_CORRECTION if operator else HypothesisOrigin.MODEL,
        latitude=10.0,
        longitude=20.0,
        uncertainty_radius_m=25_000.0,
        country_code=None,
        region_name=None,
        locality_name=None,
        rank=None if operator else 1,
        confidence_label=None,
        calibration_state=(
            CalibrationState.NOT_APPLICABLE if operator else CalibrationState.UNCALIBRATED
        ),
        supporting_evidence_ids=(EVIDENCE_ID,),
        model_family_groups=() if operator else ("synthetic_test_family",),
        created_at=NOW,
        supersedes_hypothesis_id=HYPOTHESIS_ID if operator else None,
    )


def _adjudication(*, hypothesis_id: UUID = HYPOTHESIS_ID) -> AdjudicationRecord:
    return AdjudicationRecord(
        id=ADJUDICATION_ID,
        case_id=CASE_ID,
        hypothesis_id=hypothesis_id,
        actor_id="local-analyst",
        decision=AdjudicationDecision.NEEDS_MORE_EVIDENCE,
        rationale="Synthetic fixture requires corroboration.",
        created_at=NOW,
        supersedes_adjudication_id=None,
    )


class FakeCaseService:
    def __init__(self) -> None:
        self.case = _case()
        self.media = _media()
        self.evidence = _evidence()
        self.hypothesis = _hypothesis()
        self.adjudication = _adjudication()
        self.link_calls = 0
        self.materialize_calls = 0
        self.last_create_case: dict[str, Any] = {}
        self.last_create_media: dict[str, Any] = {}

    async def create_case(self, **kwargs: Any) -> CaseRecord:
        self.last_create_case = kwargs
        return self.case

    async def list_cases(self, **kwargs: Any) -> tuple[list[CaseRecord], int]:
        return [self.case], 1

    async def get_case(self, case_id: UUID, **kwargs: Any) -> CaseRecord:
        if case_id != CASE_ID:
            raise CaseNotFoundError("not found")
        return self.case

    async def update_case(self, case_id: UUID, **kwargs: Any) -> CaseRecord:
        if case_id != CASE_ID:
            raise CaseNotFoundError("not found")
        return replace(self.case, version=2, updated_at=NOW + timedelta(seconds=1))

    async def create_media(self, case_id: UUID, **kwargs: Any) -> CaseMediaRecord:
        if case_id != CASE_ID:
            raise CaseNotFoundError("not found")
        self.last_create_media = kwargs
        self.media = replace(
            self.media,
            original_filename_display=kwargs["original_filename_display"],
            source_type=kwargs["source_type"],
            storage_state=kwargs["storage_state"],
            source_url=kwargs["source_url"],
            archive_url=kwargs["archive_url"],
        )
        return self.media

    async def list_media(self, case_id: UUID, **kwargs: Any) -> tuple[list[CaseMediaRecord], int]:
        await self.get_case(case_id)
        return [self.media], 1

    async def link_analysis(
        self,
        case_id: UUID,
        analysis_id: UUID,
        *,
        media_id: UUID,
        **kwargs: Any,
    ) -> CaseMediaRecord:
        await self.get_case(case_id)
        if media_id != MEDIA_ID:
            raise CaseRelationshipError("relationship mismatch")
        self.link_calls += 1
        self.media = replace(
            self.media,
            analysis_id=analysis_id,
            analysis_status_at_link="completed",
            analysis_created_at=NOW,
            analysis_expires_at=NOW + timedelta(hours=1),
        )
        return self.media

    async def materialize_analysis(
        self, case_id: UUID, analysis_id: UUID, *, media_id: UUID, **kwargs: Any
    ) -> MaterializationResult:
        await self.get_case(case_id)
        if media_id != MEDIA_ID or analysis_id != ANALYSIS_ID:
            raise CaseRelationshipError("relationship mismatch")
        self.materialize_calls += 1
        return MaterializationResult(
            created=self.materialize_calls == 1,
            evidence=(self.evidence,),
            hypotheses=(self.hypothesis,),
        )

    async def list_evidence(self, case_id: UUID, **kwargs: Any) -> tuple[list[EvidenceRecord], int]:
        await self.get_case(case_id)
        return [self.evidence], 1

    async def list_hypotheses(
        self, case_id: UUID, **kwargs: Any
    ) -> tuple[list[LocationHypothesisRecord], int]:
        await self.get_case(case_id)
        return [self.hypothesis], 1

    async def list_adjudications(
        self, case_id: UUID, **kwargs: Any
    ) -> tuple[list[AdjudicationRecord], int]:
        await self.get_case(case_id)
        if kwargs.get("hypothesis_id") != self.hypothesis.id:
            return [], 0
        return [self.adjudication], 1

    async def add_adjudication(
        self, case_id: UUID, hypothesis_id: UUID, **kwargs: Any
    ) -> AdjudicationRecord:
        await self.get_case(case_id)
        if hypothesis_id != self.hypothesis.id:
            raise CaseRelationshipError("relationship mismatch")
        return replace(
            self.adjudication,
            decision=kwargs["decision"],
            rationale=kwargs["rationale"],
            supersedes_adjudication_id=kwargs["supersedes_adjudication_id"],
        )

    async def add_operator_correction(
        self, case_id: UUID, **kwargs: Any
    ) -> OperatorCorrectionResult:
        await self.get_case(case_id)
        operator = replace(
            _hypothesis(operator=True),
            latitude=kwargs["latitude"],
            longitude=kwargs["longitude"],
            uncertainty_radius_m=kwargs["uncertainty_radius_m"],
        )
        adjudication = replace(
            _adjudication(hypothesis_id=operator.id),
            decision=kwargs["decision"],
            rationale=kwargs["rationale"],
        )
        return OperatorCorrectionResult(hypothesis=operator, adjudication=adjudication)

    async def list_audit_events(
        self, case_id: UUID, **kwargs: Any
    ) -> tuple[list[AuditEventRecord], int]:
        await self.get_case(case_id)
        event = AuditEventRecord(
            id=AUDIT_ID,
            case_id=CASE_ID,
            sequence_number=1,
            event_type="case_created",
            actor_id="local-analyst",
            actor_type=ActorType.OPERATOR,
            payload={"fixture": True},
            created_at=NOW,
            previous_event_hash="0" * 64,
            event_hash="c" * 64,
        )
        return [event], 1

    async def verify_audit_integrity(self, case_id: UUID, **kwargs: Any) -> AuditIntegrityResult:
        await self.get_case(case_id)
        return AuditIntegrityResult(valid=True, event_count=1, verified_through_sequence=1)


def _problem(error: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        media_type="application/problem+json",
        content={
            "type": f"https://atlaslens.invalid/problems/{error.code}",
            "title": error.title,
            "status": error.status_code,
            "code": error.code,
            "message_key": error.message_key,
            "request_id": "test-request",
            "retry_after_seconds": error.retry_after_seconds,
        },
    )


@pytest.fixture
def case_client() -> tuple[TestClient, FakeCaseService]:
    service = FakeCaseService()
    app = FastAPI()
    app.state.services = SimpleNamespace(case_service=service)
    app.include_router(router)

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
        return _problem(error)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return _problem(AppError(422, "validation_error", "error.validation", "Validation failed"))

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, service


def _create_case_payload() -> dict[str, Any]:
    return {
        "title": "Synthetic investigation",
        "description": None,
        "purpose": "journalism",
        "purpose_detail": None,
        "source_context": "Explicitly synthetic test context.",
        "sensitivity": "standard",
        "authorization_attested": True,
        "retention_policy": "metadata-default-v1",
        "created_by_actor_id": "local-analyst",
    }


def _media_payload() -> dict[str, Any]:
    return {
        "actor_id": "local-analyst",
        "media_type": "image",
        "source_type": "upload",
        "original_filename_display": "synthetic.jpg",
        "mime_type": "image/jpeg",
        "byte_size": 1234,
        "sha256": "a" * 64,
        "source_description": "Explicitly synthetic metadata.",
        "authorization_attested": True,
        "storage_state": "deleted_after_analysis",
    }


def test_case_creation_requires_controls_and_uses_local_workspace(
    case_client: tuple[TestClient, FakeCaseService],
) -> None:
    client, service = case_client
    payload = _create_case_payload()

    response = client.post("/api/v1/cases", json=payload)

    assert response.status_code == 201
    assert response.json()["workspace_id"] == DEFAULT_WORKSPACE_ID
    assert service.last_create_case["workspace_id"] == DEFAULT_WORKSPACE_ID
    assert "workspace_id" not in payload

    other = {**payload, "purpose": "other", "purpose_detail": None}
    invalid_other = client.post("/api/v1/cases", json=other)
    assert invalid_other.status_code == 422
    assert invalid_other.json()["code"] == "validation_error"

    unattested = client.post("/api/v1/cases", json={**payload, "authorization_attested": False})
    assert unattested.status_code == 422

    client_workspace = client.post(
        "/api/v1/cases", json={**payload, "workspace_id": "another-workspace"}
    )
    assert client_workspace.status_code == 422


def test_case_get_patch_and_pagination_are_bounded(
    case_client: tuple[TestClient, FakeCaseService],
) -> None:
    client, _ = case_client

    listed = client.get("/api/v1/cases?limit=1&offset=0")
    assert listed.status_code == 200
    assert listed.json()["ordering"] == "updated_at_desc_id_desc"
    assert listed.json()["items"][0]["media_count"] == 1
    assert client.get("/api/v1/cases?limit=101").status_code == 422
    assert client.get("/api/v1/cases/not-a-uuid").status_code == 422

    empty_patch = client.patch(
        f"/api/v1/cases/{CASE_ID}",
        json={"actor_id": "local-analyst", "expected_version": 1},
    )
    assert empty_patch.status_code == 422

    patched = client.patch(
        f"/api/v1/cases/{CASE_ID}",
        json={
            "actor_id": "local-analyst",
            "expected_version": 1,
            "status": "under_review",
        },
    )
    assert patched.status_code == 200
    assert patched.json()["version"] == 2

    missing = client.get("/api/v1/cases/10000000-0000-4000-8000-000000000099")
    assert missing.status_code == 404
    assert missing.json()["code"] == "case_not_found"


def test_media_api_accepts_metadata_only_and_sanitizes_display_filename(
    case_client: tuple[TestClient, FakeCaseService],
) -> None:
    client, service = case_client
    payload = {
        **_media_payload(),
        "original_filename_display": "C:\\private\\unsafe?.jpg",
    }

    created = client.post(f"/api/v1/cases/{CASE_ID}/media", json=payload)

    assert created.status_code == 201
    assert created.json()["original_filename_display"] == "unsafe_.jpg"
    assert service.last_create_media["original_filename_display"] == "unsafe_.jpg"
    assert "image" not in service.last_create_media

    video = client.post(
        f"/api/v1/cases/{CASE_ID}/media",
        json={**_media_payload(), "media_type": "video"},
    )
    assert video.status_code == 422

    raw_bytes = client.post(
        f"/api/v1/cases/{CASE_ID}/media",
        json={**_media_payload(), "image_bytes": "prohibited"},
    )
    assert raw_bytes.status_code == 422

    credential_query = client.post(
        f"/api/v1/cases/{CASE_ID}/media",
        json={
            **_media_payload(),
            "source_type": "source_url",
            "source_url": "https://example.invalid/image.jpg?token=prohibited",
        },
    )
    assert credential_query.status_code == 422


def test_media_api_round_trips_canonical_mapillary_attribution_url(
    case_client: tuple[TestClient, FakeCaseService],
) -> None:
    client, service = case_client
    source_url = "https://www.mapillary.com/app/?pKey=test-only-reference"
    payload = {
        **_media_payload(),
        "source_type": "source_url",
        "source_url": source_url,
        "storage_state": "externally_managed",
    }

    created = client.post(f"/api/v1/cases/{CASE_ID}/media", json=payload)
    listed = client.get(f"/api/v1/cases/{CASE_ID}/media?limit=100&offset=0")

    assert created.status_code == 201
    assert created.json()["source_url"] == source_url
    assert service.last_create_media["source_url"] == source_url
    assert listed.status_code == 200
    assert listed.json()["items"][0]["source_url"] == source_url


@pytest.mark.parametrize(
    "source_url",
    [
        "https://example.invalid/source?view=attribution",
        "https://www.mapillary.com/app/?pKey=test-only&access_token=test-only",
        "https://www.mapillary.com/app/?pKey=test-only&signature=test-only",
        "https://www.mapillary.com/app/?pKey=test-only#fragment",
    ],
)
def test_media_api_rejects_noncanonical_query_or_fragment_urls(
    case_client: tuple[TestClient, FakeCaseService], source_url: str
) -> None:
    client, service = case_client

    response = client.post(
        f"/api/v1/cases/{CASE_ID}/media",
        json={
            **_media_payload(),
            "source_type": "source_url",
            "source_url": source_url,
        },
    )

    assert response.status_code == 422
    assert service.last_create_media == {}


def test_link_and_materialization_are_idempotent_and_relationships_fail_closed(
    case_client: tuple[TestClient, FakeCaseService],
) -> None:
    client, service = case_client
    link_payload = {"media_id": str(MEDIA_ID), "actor_id": "local-analyst"}

    first_link = client.post(
        f"/api/v1/cases/{CASE_ID}/analyses/{ANALYSIS_ID}/link", json=link_payload
    )
    second_link = client.post(
        f"/api/v1/cases/{CASE_ID}/analyses/{ANALYSIS_ID}/link", json=link_payload
    )
    assert first_link.status_code == second_link.status_code == 200
    assert first_link.json()["id"] == second_link.json()["id"]
    assert service.link_calls == 2

    wrong_media = client.post(
        f"/api/v1/cases/{CASE_ID}/analyses/{ANALYSIS_ID}/link",
        json={
            "media_id": "20000000-0000-4000-8000-000000000099",
            "actor_id": "local-analyst",
        },
    )
    assert wrong_media.status_code == 404
    assert wrong_media.json()["code"] == "case_relationship_invalid"

    materialize_payload = {"media_id": str(MEDIA_ID), "actor_id": "local-analyst"}
    first = client.post(
        f"/api/v1/cases/{CASE_ID}/analyses/{ANALYSIS_ID}/materialize-evidence",
        json=materialize_payload,
    )
    second = client.post(
        f"/api/v1/cases/{CASE_ID}/analyses/{ANALYSIS_ID}/materialize-evidence",
        json=materialize_payload,
    )
    assert first.status_code == second.status_code == 200
    assert first.json()["created"] is True
    assert first.json()["evidence_created"] == 1
    assert second.json()["created"] is False
    assert second.json()["already_materialized"] is True
    assert second.json()["evidence_created"] == 0


def test_hypothesis_review_preserves_model_output_and_appends_operator_correction(
    case_client: tuple[TestClient, FakeCaseService],
) -> None:
    client, _ = case_client

    hypotheses = client.get(f"/api/v1/cases/{CASE_ID}/hypotheses")
    assert hypotheses.status_code == 200
    model_hypothesis = hypotheses.json()["items"][0]
    assert model_hypothesis["origin"] == "model"
    assert model_hypothesis["calibration_state"] == "uncalibrated"
    assert "confidence" not in model_hypothesis
    assert model_hypothesis["adjudication_count"] == 1
    assert model_hypothesis["latest_adjudication"]["decision"] == "needs_more_evidence"

    adjudicated = client.post(
        f"/api/v1/cases/{CASE_ID}/hypotheses/{HYPOTHESIS_ID}/adjudications",
        json={
            "actor_id": "local-analyst",
            "decision": "rejected",
            "rationale": "Synthetic fixture is not corroborated.",
        },
    )
    assert adjudicated.status_code == 201
    assert adjudicated.json()["decision"] == "rejected"

    invalid_withdrawal = client.post(
        f"/api/v1/cases/{CASE_ID}/hypotheses/{HYPOTHESIS_ID}/adjudications",
        json={
            "actor_id": "local-analyst",
            "decision": "withdrawn",
            "rationale": "Synthetic withdrawal without a superseded record.",
        },
    )
    assert invalid_withdrawal.status_code == 422

    corrected = client.post(
        f"/api/v1/cases/{CASE_ID}/operator-hypotheses",
        json={
            "actor_id": "local-analyst",
            "media_id": str(MEDIA_ID),
            "analysis_id": str(ANALYSIS_ID),
            "supersedes_hypothesis_id": str(HYPOTHESIS_ID),
            "latitude": 11.0,
            "longitude": 21.0,
            "uncertainty_radius_m": 50_000,
            "supporting_evidence_ids": [str(EVIDENCE_ID)],
            "decision": "needs_more_evidence",
            "rationale": "Synthetic operator correction pending corroboration.",
        },
    )
    assert corrected.status_code == 201
    assert corrected.json()["hypothesis"]["origin"] == "operator_correction"
    assert corrected.json()["hypothesis"]["calibration_state"] == "not_applicable"
    assert corrected.json()["hypothesis"]["supersedes_hypothesis_id"] == str(HYPOTHESIS_ID)

    duplicate_evidence = client.post(
        f"/api/v1/cases/{CASE_ID}/operator-hypotheses",
        json={
            "actor_id": "local-analyst",
            "media_id": str(MEDIA_ID),
            "analysis_id": str(ANALYSIS_ID),
            "supersedes_hypothesis_id": str(HYPOTHESIS_ID),
            "latitude": 11.0,
            "longitude": 21.0,
            "uncertainty_radius_m": 50_000,
            "supporting_evidence_ids": [str(EVIDENCE_ID), str(EVIDENCE_ID)],
            "decision": "needs_more_evidence",
            "rationale": "Synthetic duplicate evidence list.",
        },
    )
    assert duplicate_evidence.status_code == 422
    assert client.delete(f"/api/v1/cases/{CASE_ID}/audit-events").status_code == 405


def test_evidence_and_audit_responses_remain_bounded_and_truthful(
    case_client: tuple[TestClient, FakeCaseService],
) -> None:
    client, _ = case_client

    evidence = client.get(f"/api/v1/cases/{CASE_ID}/evidence?limit=50")
    assert evidence.status_code == 200
    assert evidence.json()["items"][0]["structured_payload"] == {"fixture": True}
    assert "raw_ocr" not in evidence.text
    with pytest.raises(ValidationError):
        EvidenceView.model_validate(
            replace(_evidence(), structured_payload={"authorization_header": "prohibited"})
        )
    with pytest.raises(ValidationError):
        EvidenceView.model_validate(
            replace(_evidence(), structured_payload={"nested": {"fixture": True}})
        )

    events = client.get(f"/api/v1/cases/{CASE_ID}/audit-events")
    assert events.status_code == 200
    assert events.json()["ordering"] == "sequence_number_asc"
    assert events.json()["integrity_scope"] == "tamper_evident_application_history"

    integrity = client.get(f"/api/v1/cases/{CASE_ID}/audit-integrity")
    assert integrity.status_code == 200
    assert integrity.json()["valid"] is True
    assert integrity.json()["legally_certified_evidence"] is False


def test_main_application_wires_case_router_and_service(client: TestClient) -> None:
    created = client.post("/api/v1/cases", json=_create_case_payload())

    assert created.status_code == 201
    case_id = created.json()["id"]
    assert created.json()["workspace_id"] == DEFAULT_WORKSPACE_ID
    assert client.get(f"/api/v1/cases/{case_id}").status_code == 200
    listed = client.get("/api/v1/cases?limit=10&offset=0")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["items"]] == [case_id]
