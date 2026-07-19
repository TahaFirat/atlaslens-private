from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from atlaslens_api.cases import (
    DEFAULT_WORKSPACE_ID,
    AdjudicationDecision,
    AnalysisNotCompletedError,
    CalibrationState,
    CaseConflictError,
    CaseDomainError,
    CaseInvestigationService,
    CaseNotFoundError,
    CasePurpose,
    CaseRelationshipError,
    CaseSensitivity,
    CaseStatus,
    CaseUpdate,
    MediaSourceType,
    MediaStorageState,
    MediaType,
    SQLAlchemyCaseRepository,
)
from atlaslens_api.database import create_database_engine
from atlaslens_api.repository import SQLAlchemyAnalysisRepository
from atlaslens_api.schemas import (
    Analysis,
    AnalysisMode,
    AnalysisStatus,
    Candidate,
    Evidence,
    GeoJsonPoint,
    GeoPoint,
    ImageSummary,
    Progress,
    Provenance,
)

SYNTHETIC_SHA = "a" * 64


def _analysis(
    analysis_id: UUID,
    *,
    sha256: str = SYNTHETIC_SHA,
    status: AnalysisStatus = AnalysisStatus.COMPLETED,
) -> Analysis:
    provenance = Provenance(
        provider_id="synthetic-provider",
        provider_kind="synthetic-model-family",
        provider_version="test-v1",
        execution_boundary="local",
        model_name="synthetic-model",
        output_schema_version="test-v1",
    )
    evidence = Evidence(
        id="synthetic-ocr-evidence",
        type="ocr",
        label="Synthetic OCR fixture",
        display_value="REDACTED SYNTHETIC VALUE",
        confidence=0.2,
        confidence_basis="synthetic source reliability",
        source="synthetic fixture",
        sensitive=True,
        provenance=provenance,
    )
    candidate = Candidate(
        id="synthetic-candidate",
        rank=1,
        center=GeoPoint(latitude=39.0, longitude=35.0),
        geometry=GeoJsonPoint(coordinates=(35.0, 39.0)),
        radius_km=100.0,
        uncertainty_basis="synthetic uncalibrated fixture",
        confidence=None,
        confidence_kind="uncalibrated_score",
        confidence_basis="not a probability",
        granularity="region",
        country_code="TR",
        label="Synthetic broad area",
        source="synthetic fixture",
        evidence_ids=[evidence.id],
        evidence_summary="synthetic evidence",
        provenance=[provenance],
        verification_status="unverified_model",
        verified=False,
    )
    return Analysis(
        id=analysis_id,
        status=status,
        analysis_mode=AnalysisMode.LOCAL_ONLY,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        expires_at=datetime(2027, 1, 1, tzinfo=UTC),
        progress=Progress(stage=status.value, percent=100, message_key="synthetic"),
        image=ImageSummary(
            format="jpeg",
            width=64,
            height=48,
            megapixels=0.003,
            sha256=sha256,
            orientation_normalized=True,
            exif_present=False,
        ),
        evidence=[evidence],
        candidates=[candidate],
        warnings=[],
        timings_ms={},
        fusion_policy_version="synthetic-v1",
    )


async def _service(
    *, clock: Callable[[], datetime] | None = None
) -> tuple[
    CaseInvestigationService,
    SQLAlchemyCaseRepository,
    SQLAlchemyAnalysisRepository,
    Engine,
]:
    engine = create_database_engine("sqlite:///:memory:")
    analysis_repository = SQLAlchemyAnalysisRepository(engine)
    case_repository = SQLAlchemyCaseRepository(engine)
    await analysis_repository.initialize()
    await case_repository.initialize()
    service = (
        CaseInvestigationService(case_repository, analysis_repository)
        if clock is None
        else CaseInvestigationService(case_repository, analysis_repository, clock=clock)
    )
    return (
        service,
        case_repository,
        analysis_repository,
        engine,
    )


async def _case(
    service: CaseInvestigationService,
    *,
    sensitivity: CaseSensitivity = CaseSensitivity.STANDARD,
):
    return await service.create_case(
        title="Synthetic Phase 2 case",
        description="Synthetic-only fixture",
        purpose=CasePurpose.JOURNALISM,
        purpose_detail=None,
        sensitivity=sensitivity,
        source_context="Synthetic public-interest test context",
        authorization_attested=True,
        created_by_actor_id="synthetic-analyst",
        retention_policy="temporary",
    )


async def _media(
    service: CaseInvestigationService,
    case_id: UUID,
    *,
    sha256: str = SYNTHETIC_SHA,
    storage_state: MediaStorageState = MediaStorageState.DELETED_AFTER_ANALYSIS,
):
    return await service.create_media(
        case_id,
        actor_id="synthetic-analyst",
        media_type=MediaType.IMAGE,
        source_type=MediaSourceType.UPLOAD,
        original_filename_display=r"C:\private\synthetic?.jpg",
        mime_type="image/jpeg",
        byte_size=128,
        sha256=sha256,
        captured_at=None,
        received_at=datetime(2026, 1, 2, tzinfo=UTC),
        source_url=None,
        archive_url=None,
        source_description="Synthetic fixture metadata",
        authorization_attested=True,
        storage_state=storage_state,
    )


async def test_case_crud_media_pagination_and_workspace_scope() -> None:
    service, _, analysis_repository, _ = await _service()
    case = await _case(service)

    assert case.workspace_id == DEFAULT_WORKSPACE_ID
    assert case.version == 1
    with pytest.raises(CaseNotFoundError):
        await service.get_case(case.id, workspace_id="another-workspace")
    with pytest.raises(CaseDomainError, match="purpose_detail"):
        await service.update_case(
            case.id,
            actor_id="synthetic-analyst",
            update=CaseUpdate(purpose=CasePurpose.OTHER, purpose_detail=None),
        )

    updated = await service.update_case(
        case.id,
        actor_id="synthetic-analyst",
        update=CaseUpdate(status=CaseStatus.UNDER_REVIEW, expected_version=1),
    )
    assert updated.status is CaseStatus.UNDER_REVIEW
    assert updated.version == 2

    first = await _media(service, case.id)
    second = await _media(service, case.id, sha256="b" * 64)
    assert first.original_filename_display == "synthetic_.jpg"
    page, total = await service.list_media(case.id, limit=1, offset=1)
    all_media, _ = await service.list_media(case.id)
    assert total == 2 and len(page) == 1
    assert {item.id for item in all_media} == {first.id, second.id}
    assert page[0] == all_media[1]

    analysis = _analysis(UUID(int=100))
    await analysis_repository.create(analysis, storage_key=None, idempotency_hash=None)
    linked = await service.link_analysis(
        case.id, analysis.id, media_id=first.id, actor_id="synthetic-analyst"
    )
    repeated = await service.link_analysis(
        case.id, analysis.id, media_id=first.id, actor_id="synthetic-analyst"
    )
    assert linked == repeated

    duplicate_target = await _media(service, case.id)
    _, duplicate_events_before = await service.list_audit_events(case.id)
    with pytest.raises(CaseConflictError):
        await service.link_analysis(
            case.id,
            analysis.id,
            media_id=duplicate_target.id,
            actor_id="synthetic-analyst",
        )
    _, duplicate_events_after = await service.list_audit_events(case.id)
    assert duplicate_events_after == duplicate_events_before

    mismatched_analysis = _analysis(UUID(int=101), sha256="c" * 64)
    await analysis_repository.create(mismatched_analysis, storage_key=None, idempotency_hash=None)
    events_before, total_before = await service.list_audit_events(case.id)
    with pytest.raises(CaseRelationshipError) as error:
        await service.link_analysis(
            case.id,
            mismatched_analysis.id,
            media_id=second.id,
            actor_id="synthetic-analyst",
        )
    assert error.value.code == "analysis_media_hash_mismatch"
    events_after, total_after = await service.list_audit_events(case.id)
    assert total_after == total_before
    assert events_after == events_before


async def test_non_completed_analysis_is_not_materialized() -> None:
    service, _, analysis_repository, _ = await _service()
    case = await _case(service)
    media = await _media(service, case.id, storage_state=MediaStorageState.EPHEMERAL)
    queued = _analysis(UUID(int=102), status=AnalysisStatus.QUEUED)
    await analysis_repository.create(queued, storage_key=None, idempotency_hash=None)
    await service.link_analysis(case.id, queued.id, media_id=media.id, actor_id="synthetic-analyst")
    with pytest.raises(AnalysisNotCompletedError):
        await service.materialize_analysis(
            case.id, queued.id, media_id=media.id, actor_id="synthetic-analyst"
        )
    await analysis_repository.save(_analysis(queued.id, sha256="d" * 64))
    with pytest.raises(CaseRelationshipError) as mismatch:
        await service.materialize_analysis(
            case.id, queued.id, media_id=media.id, actor_id="synthetic-analyst"
        )
    assert mismatch.value.code == "analysis_media_hash_mismatch"
    evidence, total = await service.list_evidence(case.id)
    assert evidence == [] and total == 0


async def test_materialization_is_idempotent_private_and_retention_safe() -> None:
    service, repository, analysis_repository, _ = await _service()
    case = await _case(service, sensitivity=CaseSensitivity.CONFLICT_RELATED)
    media = await _media(service, case.id, storage_state=MediaStorageState.EPHEMERAL)
    analysis = _analysis(UUID(int=103))
    await analysis_repository.create(analysis, storage_key=None, idempotency_hash=None)
    await service.link_analysis(
        case.id, analysis.id, media_id=media.id, actor_id="synthetic-analyst"
    )

    first = await service.materialize_analysis(
        case.id, analysis.id, media_id=media.id, actor_id="synthetic-analyst"
    )
    second = await service.materialize_analysis(
        case.id, analysis.id, media_id=media.id, actor_id="synthetic-analyst"
    )
    assert first.created is True and second.created is False
    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]
    assert [item.id for item in first.hypotheses] == [item.id for item in second.hypotheses]
    assert len(first.evidence) == len(first.hypotheses) == 1
    persisted_evidence = first.evidence[0]
    assert persisted_evidence.summary == "Redacted OCR evidence available"
    serialized = json.dumps(
        {
            "payload": persisted_evidence.structured_payload,
            "provenance": persisted_evidence.provenance,
            "summary": persisted_evidence.summary,
        }
    )
    assert "REDACTED SYNTHETIC VALUE" not in serialized
    assert "display_value" not in serialized and "raw_ocr" not in serialized
    hypothesis = first.hypotheses[0]
    assert hypothesis.calibration_state is CalibrationState.UNCALIBRATED
    assert hypothesis.confidence_label == "uncalibrated"
    assert hypothesis.uncertainty_radius_m > 0

    unchanged = await analysis_repository.get(analysis.id)
    assert unchanged is not None
    assert unchanged.analysis.evidence[0].display_value == "REDACTED SYNTHETIC VALUE"
    assert unchanged.analysis.candidates[0].verified is False

    deleted = await analysis_repository.delete(analysis.id)
    assert deleted is not None
    after_retention = await service.materialize_analysis(
        case.id, analysis.id, media_id=media.id, actor_id="synthetic-analyst"
    )
    assert after_retention.created is False
    assert after_retention.hypotheses[0].id == hypothesis.id
    retained_media = (await service.list_media(case.id))[0][0]
    assert retained_media.analysis_id == analysis.id
    assert retained_media.analysis_status_at_link == "completed"
    assert retained_media.storage_state is MediaStorageState.DELETED_AFTER_ANALYSIS
    retention_events, _ = await service.list_audit_events(case.id)
    retention_event = next(
        event for event in retention_events if event.event_type == "analysis.materialized"
    )
    assert retention_event.payload["storage_state"] == "deleted_after_analysis"

    assert not hasattr(repository, "update_evidence")
    assert not hasattr(repository, "delete_evidence")
    assert not hasattr(repository, "update_hypothesis")
    assert not hasattr(repository, "delete_adjudication")
    assert not hasattr(repository, "delete_audit_event")


async def test_materialization_preserves_ephemeral_state_while_storage_key_exists() -> None:
    service, _, analysis_repository, _ = await _service()
    case = await _case(service)
    media = await _media(service, case.id, storage_state=MediaStorageState.EPHEMERAL)
    analysis = _analysis(UUID(int=105))
    await analysis_repository.create(
        analysis,
        storage_key="synthetic-retained-storage-key",
        idempotency_hash=None,
    )
    await service.link_analysis(
        case.id, analysis.id, media_id=media.id, actor_id="synthetic-analyst"
    )

    result = await service.materialize_analysis(
        case.id, analysis.id, media_id=media.id, actor_id="synthetic-analyst"
    )
    assert result.created is True
    persisted_media = (await service.list_media(case.id))[0][0]
    assert persisted_media.storage_state is MediaStorageState.EPHEMERAL
    events, _ = await service.list_audit_events(case.id)
    materialized_event = next(
        event for event in events if event.event_type == "analysis.materialized"
    )
    assert materialized_event.payload["storage_state"] == "ephemeral"


async def test_adjudications_and_operator_correction_are_append_only() -> None:
    constant_time = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    service, _, analysis_repository, _ = await _service(clock=lambda: constant_time)
    case = await _case(service)
    media = await _media(service, case.id)
    analysis = _analysis(UUID(int=104))
    await analysis_repository.create(analysis, storage_key=None, idempotency_hash=None)
    await service.link_analysis(
        case.id, analysis.id, media_id=media.id, actor_id="synthetic-analyst"
    )
    materialized = await service.materialize_analysis(
        case.id, analysis.id, media_id=media.id, actor_id="synthetic-analyst"
    )
    original = materialized.hypotheses[0]
    first = await service.add_adjudication(
        case.id,
        original.id,
        actor_id="synthetic-analyst",
        decision=AdjudicationDecision.NEEDS_MORE_EVIDENCE,
        rationale="Synthetic fixture requires another independent source.",
    )
    second = await service.add_adjudication(
        case.id,
        original.id,
        actor_id="synthetic-analyst",
        decision=AdjudicationDecision.REJECTED,
        rationale="Synthetic fixture conflicts with the reviewed clue.",
        supersedes_adjudication_id=first.id,
    )
    adjudications, total = await service.list_adjudications(case.id, hypothesis_id=original.id)
    assert total == 2
    assert [item.decision for item in adjudications] == [
        AdjudicationDecision.NEEDS_MORE_EVIDENCE,
        AdjudicationDecision.REJECTED,
    ]
    assert second.created_at > first.created_at
    assert second.supersedes_adjudication_id == first.id
    case_after_second = await service.get_case(case.id)
    assert case_after_second.latest_adjudication_decision is AdjudicationDecision.REJECTED

    correction = await service.add_operator_correction(
        case.id,
        actor_id="synthetic-analyst",
        media_id=media.id,
        analysis_id=analysis.id,
        supersedes_hypothesis_id=original.id,
        latitude=38.5,
        longitude=34.5,
        uncertainty_radius_m=2_500,
        country_code="TR",
        region_name="Synthetic region",
        locality_name="Synthetic locality",
        supporting_evidence_ids=[materialized.evidence[0].id],
        model_family_groups=[],
        decision=AdjudicationDecision.ACCEPTED,
        rationale="Synthetic operator correction for persistence testing.",
    )
    assert correction.hypothesis.id != original.id
    assert correction.hypothesis.supersedes_hypothesis_id == original.id
    assert correction.hypothesis.origin.value == "operator_correction"
    assert correction.adjudication.created_at > second.created_at
    assert correction.hypothesis.created_at == correction.adjudication.created_at
    all_adjudications, all_total = await service.list_adjudications(case.id)
    assert all_total == 3
    assert [item.decision for item in all_adjudications] == [
        AdjudicationDecision.NEEDS_MORE_EVIDENCE,
        AdjudicationDecision.REJECTED,
        AdjudicationDecision.ACCEPTED,
    ]
    case_after_correction = await service.get_case(case.id)
    assert case_after_correction.latest_adjudication_decision is AdjudicationDecision.ACCEPTED
    hypotheses, hypothesis_total = await service.list_hypotheses(case.id)
    assert hypothesis_total == 2
    assert hypotheses[0] == original
    assert hypotheses[0].latitude == 39.0

    integrity = await service.verify_audit_integrity(case.id)
    events, event_total = await service.list_audit_events(case.id)
    assert integrity.valid is True
    assert integrity.event_count == event_total
    assert [event.sequence_number for event in events] == list(range(1, event_total + 1))


async def test_audit_verifier_detects_payload_hash_and_order_tampering() -> None:
    service, _, _, engine = await _service()
    payload_case = await _case(service)
    hash_case = await _case(service)
    order_case = await _case(service)

    assert (await service.verify_audit_integrity(payload_case.id)).valid is True
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE case_audit_events SET payload=:payload "
                "WHERE case_id=:case_id AND sequence_number=1"
            ),
            {
                "case_id": str(payload_case.id),
                "payload": json.dumps({"case_id": str(payload_case.id), "status": "archived"}),
            },
        )
        connection.execute(
            text(
                "UPDATE case_audit_events SET event_hash=:event_hash "
                "WHERE case_id=:case_id AND sequence_number=1"
            ),
            {"case_id": str(hash_case.id), "event_hash": "f" * 64},
        )
        connection.execute(
            text(
                "UPDATE case_audit_events SET sequence_number=3 "
                "WHERE case_id=:case_id AND sequence_number=1"
            ),
            {"case_id": str(order_case.id)},
        )

    payload_result = await service.verify_audit_integrity(payload_case.id)
    hash_result = await service.verify_audit_integrity(hash_case.id)
    order_result = await service.verify_audit_integrity(order_case.id)
    assert payload_result.failure_code == "event_hash_mismatch"
    assert hash_result.failure_code == "event_hash_mismatch"
    assert order_result.failure_code == "sequence_mismatch"
