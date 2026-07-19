from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypeVar
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from atlaslens_api.database import Base

from .domain import (
    GENESIS_AUDIT_HASH,
    ActorType,
    AdjudicationDecision,
    AdjudicationRecord,
    AuditEventRecord,
    AuditIntegrityResult,
    CalibrationState,
    CaseConflictError,
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
    JsonValue,
    LocationHypothesisRecord,
    MaterializationResult,
    MediaSourceType,
    MediaStorageState,
    MediaType,
    OperatorCorrectionResult,
    as_utc,
    canonical_audit_hash,
    safe_json_object,
)
from .models import (
    AdjudicationRow,
    AuditEventRow,
    CaseMediaRow,
    CaseRow,
    EvidenceRecordRow,
    LocationHypothesisRow,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class EvidenceInsert:
    record: EvidenceRecord
    source_record_key: str


@dataclass(frozen=True, slots=True)
class HypothesisInsert:
    record: LocationHypothesisRecord
    source_record_key: str


def _validate_page(limit: int, offset: int) -> None:
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("invalid pagination")


def _utc(value: datetime) -> datetime:
    normalized = as_utc(value)
    if normalized is None:
        raise ValueError("timestamp is required")
    return normalized


def _to_case(
    row: CaseRow,
    *,
    media_count: int = 0,
    evidence_count: int = 0,
    hypothesis_count: int = 0,
    adjudication_count: int = 0,
    latest_decision: str | None = None,
) -> CaseRecord:
    return CaseRecord(
        id=UUID(row.id),
        workspace_id=row.workspace_id,
        title=row.title,
        description=row.description,
        purpose=CasePurpose(row.purpose),
        purpose_detail=row.purpose_detail,
        source_context=row.source_context,
        authorization_attested=row.authorization_attested,
        status=CaseStatus(row.status),
        sensitivity=CaseSensitivity(row.sensitivity),
        created_at=_utc(row.created_at),
        updated_at=_utc(row.updated_at),
        closed_at=as_utc(row.closed_at),
        created_by_actor_id=row.created_by_actor_id,
        retention_policy=row.retention_policy,
        version=row.version,
        media_count=media_count,
        evidence_count=evidence_count,
        hypothesis_count=hypothesis_count,
        adjudication_count=adjudication_count,
        latest_adjudication_decision=(
            None if latest_decision is None else AdjudicationDecision(latest_decision)
        ),
    )


def _to_media(row: CaseMediaRow) -> CaseMediaRecord:
    return CaseMediaRecord(
        id=UUID(row.id),
        case_id=UUID(row.case_id),
        analysis_id=None if row.analysis_id is None else UUID(row.analysis_id),
        media_type=MediaType(row.media_type),
        source_type=MediaSourceType(row.source_type),
        original_filename_display=row.original_filename_display,
        mime_type=row.mime_type,
        byte_size=row.byte_size,
        sha256=row.sha256,
        captured_at=as_utc(row.captured_at),
        received_at=_utc(row.received_at),
        source_url=row.source_url,
        archive_url=row.archive_url,
        source_description=row.source_description,
        authorization_attested=row.authorization_attested,
        storage_state=MediaStorageState(row.storage_state),
        analysis_status_at_link=row.analysis_status_at_link,
        analysis_created_at=as_utc(row.analysis_created_at),
        analysis_expires_at=as_utc(row.analysis_expires_at),
        materialized_at=as_utc(row.materialized_at),
        created_at=_utc(row.created_at),
    )


def _to_evidence(row: EvidenceRecordRow) -> EvidenceRecord:
    return EvidenceRecord(
        id=UUID(row.id),
        case_id=UUID(row.case_id),
        media_id=UUID(row.media_id),
        analysis_id=UUID(row.analysis_id),
        evidence_type=EvidenceType(row.evidence_type),
        provider=row.provider,
        provider_family=row.provider_family,
        summary=row.summary,
        structured_payload=safe_json_object(row.structured_payload),
        provenance=safe_json_object(row.provenance),
        observed_at=_utc(row.observed_at),
        created_at=_utc(row.created_at),
        immutable_source_hash=row.immutable_source_hash,
    )


def _to_hypothesis(row: LocationHypothesisRow) -> LocationHypothesisRecord:
    return LocationHypothesisRecord(
        id=UUID(row.id),
        case_id=UUID(row.case_id),
        media_id=UUID(row.media_id),
        analysis_id=UUID(row.analysis_id),
        origin=HypothesisOrigin(row.origin),
        latitude=row.latitude,
        longitude=row.longitude,
        uncertainty_radius_m=row.uncertainty_radius_m,
        country_code=row.country_code,
        region_name=row.region_name,
        locality_name=row.locality_name,
        rank=row.rank,
        confidence_label=row.confidence_label,
        calibration_state=CalibrationState(row.calibration_state),
        supporting_evidence_ids=tuple(UUID(item) for item in row.supporting_evidence_ids),
        model_family_groups=tuple(row.model_family_groups),
        created_at=_utc(row.created_at),
        supersedes_hypothesis_id=(
            None if row.supersedes_hypothesis_id is None else UUID(row.supersedes_hypothesis_id)
        ),
    )


def _to_adjudication(row: AdjudicationRow) -> AdjudicationRecord:
    return AdjudicationRecord(
        id=UUID(row.id),
        case_id=UUID(row.case_id),
        hypothesis_id=UUID(row.hypothesis_id),
        actor_id=row.actor_id,
        decision=AdjudicationDecision(row.decision),
        rationale=row.rationale,
        created_at=_utc(row.created_at),
        supersedes_adjudication_id=(
            None if row.supersedes_adjudication_id is None else UUID(row.supersedes_adjudication_id)
        ),
    )


def _to_audit(row: AuditEventRow) -> AuditEventRecord:
    return AuditEventRecord(
        id=UUID(row.id),
        case_id=UUID(row.case_id),
        sequence_number=row.sequence_number,
        event_type=row.event_type,
        actor_id=row.actor_id,
        actor_type=ActorType(row.actor_type),
        payload=safe_json_object(row.payload),
        created_at=_utc(row.created_at),
        previous_event_hash=row.previous_event_hash,
        event_hash=row.event_hash,
    )


class SQLAlchemyCaseRepository:
    """Single-workspace-ready persistence seam; not production tenancy isolation."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._sessions: sessionmaker[Session] = sessionmaker(bind=engine, expire_on_commit=False)
        self._operation_lock = asyncio.Lock()

    async def _run(self, operation: Callable[[], T]) -> T:
        async with self._operation_lock:
            return await asyncio.to_thread(operation)

    async def initialize(self) -> None:
        await self._run(lambda: Base.metadata.create_all(self._engine))

    @staticmethod
    def _case_for_workspace(
        session: Session, case_id: UUID, workspace_id: str, *, lock: bool = False
    ) -> CaseRow:
        statement = select(CaseRow).where(
            CaseRow.id == str(case_id), CaseRow.workspace_id == workspace_id
        )
        if lock:
            statement = statement.with_for_update()
        row = session.scalar(statement)
        if row is None:
            raise CaseNotFoundError("case not found")
        return row

    @staticmethod
    def _counts(session: Session, case_id: str) -> tuple[int, int, int, int, str | None]:
        media_count = session.scalar(
            select(func.count()).select_from(CaseMediaRow).where(CaseMediaRow.case_id == case_id)
        )
        evidence_count = session.scalar(
            select(func.count())
            .select_from(EvidenceRecordRow)
            .where(EvidenceRecordRow.case_id == case_id)
        )
        hypothesis_count = session.scalar(
            select(func.count())
            .select_from(LocationHypothesisRow)
            .where(LocationHypothesisRow.case_id == case_id)
        )
        adjudication_count = session.scalar(
            select(func.count())
            .select_from(AdjudicationRow)
            .where(AdjudicationRow.case_id == case_id)
        )
        latest_decision = session.scalar(
            select(AdjudicationRow.decision)
            .where(AdjudicationRow.case_id == case_id)
            .order_by(AdjudicationRow.created_at.desc(), AdjudicationRow.id.desc())
            .limit(1)
        )
        return (
            int(media_count or 0),
            int(evidence_count or 0),
            int(hypothesis_count or 0),
            int(adjudication_count or 0),
            latest_decision,
        )

    @classmethod
    def _case_record(cls, session: Session, row: CaseRow) -> CaseRecord:
        media, evidence, hypotheses, adjudications, latest = cls._counts(session, row.id)
        return _to_case(
            row,
            media_count=media,
            evidence_count=evidence,
            hypothesis_count=hypotheses,
            adjudication_count=adjudications,
            latest_decision=latest,
        )

    @staticmethod
    def _touch(row: CaseRow, timestamp: datetime) -> None:
        row.updated_at = timestamp
        row.version += 1

    @staticmethod
    def _monotonic_append_timestamp(row: CaseRow, requested: datetime) -> datetime:
        requested_utc = _utc(requested)
        current_utc = _utc(row.updated_at)
        if requested_utc > current_utc:
            return requested_utc
        return current_utc + timedelta(microseconds=1)

    @staticmethod
    def _append_audit(
        session: Session,
        *,
        case_id: UUID,
        event_type: str,
        actor_id: str,
        actor_type: ActorType,
        payload: dict[str, JsonValue],
        created_at: datetime,
    ) -> AuditEventRecord:
        safe_payload = safe_json_object(payload)
        previous = session.scalar(
            select(AuditEventRow)
            .where(AuditEventRow.case_id == str(case_id))
            .order_by(AuditEventRow.sequence_number.desc())
            .limit(1)
        )
        sequence_number = 1 if previous is None else previous.sequence_number + 1
        previous_hash = GENESIS_AUDIT_HASH if previous is None else previous.event_hash
        event_id = uuid4()
        event_hash = canonical_audit_hash(
            event_id=event_id,
            case_id=case_id,
            sequence_number=sequence_number,
            event_type=event_type,
            actor_id=actor_id,
            actor_type=actor_type,
            payload=safe_payload,
            created_at=created_at,
            previous_event_hash=previous_hash,
        )
        row = AuditEventRow(
            id=str(event_id),
            case_id=str(case_id),
            sequence_number=sequence_number,
            event_type=event_type,
            actor_id=actor_id,
            actor_type=actor_type.value,
            payload=safe_payload,
            created_at=created_at,
            previous_event_hash=previous_hash,
            event_hash=event_hash,
        )
        session.add(row)
        session.flush()
        return _to_audit(row)

    async def create_case(self, record: CaseRecord) -> CaseRecord:
        def operation() -> CaseRecord:
            row = CaseRow(
                id=str(record.id),
                workspace_id=record.workspace_id,
                title=record.title,
                description=record.description,
                purpose=record.purpose.value,
                purpose_detail=record.purpose_detail,
                source_context=record.source_context,
                authorization_attested=record.authorization_attested,
                status=record.status.value,
                sensitivity=record.sensitivity.value,
                created_at=record.created_at,
                updated_at=record.updated_at,
                closed_at=record.closed_at,
                created_by_actor_id=record.created_by_actor_id,
                retention_policy=record.retention_policy,
                version=record.version,
            )
            try:
                with self._sessions.begin() as session:
                    session.add(row)
                    session.flush()
                    self._append_audit(
                        session,
                        case_id=record.id,
                        event_type="case.created",
                        actor_id=record.created_by_actor_id,
                        actor_type=ActorType.OPERATOR,
                        payload={
                            "case_id": str(record.id),
                            "purpose": record.purpose.value,
                            "sensitivity": record.sensitivity.value,
                            "status": record.status.value,
                            "version": record.version,
                        },
                        created_at=record.created_at,
                    )
                    return self._case_record(session, row)
            except IntegrityError as exc:
                raise CaseConflictError("case already exists") from exc

        return await self._run(operation)

    async def get_case(self, case_id: UUID, *, workspace_id: str) -> CaseRecord:
        def operation() -> CaseRecord:
            with self._sessions() as session:
                row = self._case_for_workspace(session, case_id, workspace_id)
                return self._case_record(session, row)

        return await self._run(operation)

    async def list_cases(
        self, *, workspace_id: str, limit: int, offset: int
    ) -> tuple[list[CaseRecord], int]:
        _validate_page(limit, offset)

        def operation() -> tuple[list[CaseRecord], int]:
            with self._sessions() as session:
                total = session.scalar(
                    select(func.count())
                    .select_from(CaseRow)
                    .where(CaseRow.workspace_id == workspace_id)
                )
                rows = session.scalars(
                    select(CaseRow)
                    .where(CaseRow.workspace_id == workspace_id)
                    .order_by(CaseRow.updated_at.desc(), CaseRow.id.desc())
                    .limit(limit)
                    .offset(offset)
                ).all()
                return [self._case_record(session, row) for row in rows], int(total or 0)

        return await self._run(operation)

    async def update_case(
        self,
        replacement: CaseRecord,
        *,
        expected_version: int,
        actor_id: str,
    ) -> CaseRecord:
        def operation() -> CaseRecord:
            with self._sessions.begin() as session:
                row = self._case_for_workspace(
                    session, replacement.id, replacement.workspace_id, lock=True
                )
                if row.version != expected_version:
                    raise CaseConflictError("case version conflict", code="case_version_conflict")
                row.title = replacement.title
                row.description = replacement.description
                row.purpose = replacement.purpose.value
                row.purpose_detail = replacement.purpose_detail
                row.source_context = replacement.source_context
                row.status = replacement.status.value
                row.sensitivity = replacement.sensitivity.value
                row.closed_at = replacement.closed_at
                row.retention_policy = replacement.retention_policy
                self._touch(row, replacement.updated_at)
                self._append_audit(
                    session,
                    case_id=replacement.id,
                    event_type="case.updated",
                    actor_id=actor_id,
                    actor_type=ActorType.OPERATOR,
                    payload={
                        "case_id": str(replacement.id),
                        "status": replacement.status.value,
                        "sensitivity": replacement.sensitivity.value,
                        "version": row.version,
                    },
                    created_at=replacement.updated_at,
                )
                return self._case_record(session, row)

        return await self._run(operation)

    async def create_media(
        self, record: CaseMediaRecord, *, workspace_id: str, actor_id: str
    ) -> CaseMediaRecord:
        def operation() -> CaseMediaRecord:
            try:
                with self._sessions.begin() as session:
                    case_row = self._case_for_workspace(
                        session, record.case_id, workspace_id, lock=True
                    )
                    row = CaseMediaRow(
                        id=str(record.id),
                        case_id=str(record.case_id),
                        analysis_id=None if record.analysis_id is None else str(record.analysis_id),
                        media_type=record.media_type.value,
                        source_type=record.source_type.value,
                        original_filename_display=record.original_filename_display,
                        mime_type=record.mime_type,
                        byte_size=record.byte_size,
                        sha256=record.sha256,
                        captured_at=record.captured_at,
                        received_at=record.received_at,
                        source_url=record.source_url,
                        archive_url=record.archive_url,
                        source_description=record.source_description,
                        authorization_attested=record.authorization_attested,
                        storage_state=record.storage_state.value,
                        analysis_status_at_link=record.analysis_status_at_link,
                        analysis_created_at=record.analysis_created_at,
                        analysis_expires_at=record.analysis_expires_at,
                        materialized_at=record.materialized_at,
                        created_at=record.created_at,
                    )
                    session.add(row)
                    session.flush()
                    self._touch(case_row, record.created_at)
                    self._append_audit(
                        session,
                        case_id=record.case_id,
                        event_type="media.created",
                        actor_id=actor_id,
                        actor_type=ActorType.OPERATOR,
                        payload={
                            "case_id": str(record.case_id),
                            "media_id": str(record.id),
                            "media_type": record.media_type.value,
                            "source_type": record.source_type.value,
                            "storage_state": record.storage_state.value,
                        },
                        created_at=record.created_at,
                    )
                    return _to_media(row)
            except IntegrityError as exc:
                raise CaseConflictError("media or analysis link already exists") from exc

        return await self._run(operation)

    async def get_media(
        self, case_id: UUID, media_id: UUID, *, workspace_id: str
    ) -> CaseMediaRecord:
        def operation() -> CaseMediaRecord:
            with self._sessions() as session:
                self._case_for_workspace(session, case_id, workspace_id)
                row = session.scalar(
                    select(CaseMediaRow).where(
                        CaseMediaRow.id == str(media_id), CaseMediaRow.case_id == str(case_id)
                    )
                )
                if row is None:
                    raise CaseRelationshipError("media does not belong to case")
                return _to_media(row)

        return await self._run(operation)

    async def list_media(
        self, case_id: UUID, *, workspace_id: str, limit: int, offset: int
    ) -> tuple[list[CaseMediaRecord], int]:
        _validate_page(limit, offset)

        def operation() -> tuple[list[CaseMediaRecord], int]:
            with self._sessions() as session:
                self._case_for_workspace(session, case_id, workspace_id)
                condition = CaseMediaRow.case_id == str(case_id)
                total = session.scalar(
                    select(func.count()).select_from(CaseMediaRow).where(condition)
                )
                rows = session.scalars(
                    select(CaseMediaRow)
                    .where(condition)
                    .order_by(CaseMediaRow.created_at, CaseMediaRow.id)
                    .limit(limit)
                    .offset(offset)
                ).all()
                return [_to_media(row) for row in rows], int(total or 0)

        return await self._run(operation)

    async def media_for_analysis(
        self, case_id: UUID, analysis_id: UUID, *, workspace_id: str
    ) -> CaseMediaRecord:
        def operation() -> CaseMediaRecord:
            with self._sessions() as session:
                self._case_for_workspace(session, case_id, workspace_id)
                row = session.scalar(
                    select(CaseMediaRow).where(
                        CaseMediaRow.case_id == str(case_id),
                        CaseMediaRow.analysis_id == str(analysis_id),
                    )
                )
                if row is None:
                    raise CaseRelationshipError("analysis is not linked to this case")
                return _to_media(row)

        return await self._run(operation)

    async def link_analysis(
        self,
        case_id: UUID,
        media_id: UUID,
        analysis_id: UUID,
        *,
        workspace_id: str,
        actor_id: str,
        analysis_status: str,
        analysis_created_at: datetime,
        analysis_expires_at: datetime | None,
        linked_at: datetime,
    ) -> CaseMediaRecord:
        def operation() -> CaseMediaRecord:
            try:
                with self._sessions.begin() as session:
                    case_row = self._case_for_workspace(session, case_id, workspace_id, lock=True)
                    row = session.scalar(
                        select(CaseMediaRow).where(
                            CaseMediaRow.id == str(media_id),
                            CaseMediaRow.case_id == str(case_id),
                        )
                    )
                    if row is None:
                        raise CaseRelationshipError("media does not belong to case")
                    if row.analysis_id == str(analysis_id):
                        return _to_media(row)
                    if row.analysis_id is not None:
                        raise CaseConflictError(
                            "media is already linked to another analysis",
                            code="media_analysis_conflict",
                        )
                    existing = session.scalar(
                        select(CaseMediaRow.id).where(CaseMediaRow.analysis_id == str(analysis_id))
                    )
                    if existing is not None:
                        raise CaseConflictError(
                            "analysis is already linked to another media record",
                            code="analysis_already_linked",
                        )
                    row.analysis_id = str(analysis_id)
                    row.analysis_status_at_link = analysis_status
                    row.analysis_created_at = analysis_created_at
                    row.analysis_expires_at = analysis_expires_at
                    self._touch(case_row, linked_at)
                    self._append_audit(
                        session,
                        case_id=case_id,
                        event_type="analysis.linked",
                        actor_id=actor_id,
                        actor_type=ActorType.OPERATOR,
                        payload={
                            "analysis_id": str(analysis_id),
                            "case_id": str(case_id),
                            "media_id": str(media_id),
                            "status_at_link": analysis_status,
                        },
                        created_at=linked_at,
                    )
                    session.flush()
                    return _to_media(row)
            except IntegrityError as exc:
                raise CaseConflictError(
                    "analysis is already linked", code="analysis_already_linked"
                ) from exc

        return await self._run(operation)

    async def materialize_analysis(
        self,
        case_id: UUID,
        media_id: UUID,
        analysis_id: UUID,
        *,
        workspace_id: str,
        actor_id: str,
        evidence: Sequence[EvidenceInsert],
        hypotheses: Sequence[HypothesisInsert],
        analysis_storage_present: bool,
        materialized_at: datetime,
    ) -> MaterializationResult:
        def operation() -> MaterializationResult:
            with self._sessions.begin() as session:
                case_row = self._case_for_workspace(session, case_id, workspace_id, lock=True)
                media_row = session.scalar(
                    select(CaseMediaRow).where(
                        CaseMediaRow.id == str(media_id),
                        CaseMediaRow.case_id == str(case_id),
                    )
                )
                if media_row is None or media_row.analysis_id != str(analysis_id):
                    raise CaseRelationshipError("analysis/media/case relationship is invalid")
                if media_row.materialized_at is not None:
                    existing_evidence = session.scalars(
                        select(EvidenceRecordRow)
                        .where(
                            EvidenceRecordRow.case_id == str(case_id),
                            EvidenceRecordRow.media_id == str(media_id),
                            EvidenceRecordRow.analysis_id == str(analysis_id),
                        )
                        .order_by(EvidenceRecordRow.created_at, EvidenceRecordRow.id)
                    ).all()
                    existing_hypotheses = session.scalars(
                        select(LocationHypothesisRow)
                        .where(
                            LocationHypothesisRow.case_id == str(case_id),
                            LocationHypothesisRow.media_id == str(media_id),
                            LocationHypothesisRow.analysis_id == str(analysis_id),
                        )
                        .order_by(LocationHypothesisRow.created_at, LocationHypothesisRow.id)
                    ).all()
                    return MaterializationResult(
                        created=False,
                        evidence=tuple(_to_evidence(row) for row in existing_evidence),
                        hypotheses=tuple(_to_hypothesis(row) for row in existing_hypotheses),
                    )

                for item in evidence:
                    record = item.record
                    session.add(
                        EvidenceRecordRow(
                            id=str(record.id),
                            case_id=str(record.case_id),
                            media_id=str(record.media_id),
                            analysis_id=str(record.analysis_id),
                            evidence_type=record.evidence_type.value,
                            provider=record.provider,
                            provider_family=record.provider_family,
                            summary=record.summary,
                            structured_payload=record.structured_payload,
                            provenance=record.provenance,
                            observed_at=record.observed_at,
                            created_at=record.created_at,
                            immutable_source_hash=record.immutable_source_hash,
                            source_record_key=item.source_record_key,
                        )
                    )
                for hypothesis_item in hypotheses:
                    hypothesis_record = hypothesis_item.record
                    session.add(
                        LocationHypothesisRow(
                            id=str(hypothesis_record.id),
                            case_id=str(hypothesis_record.case_id),
                            media_id=str(hypothesis_record.media_id),
                            analysis_id=str(hypothesis_record.analysis_id),
                            origin=hypothesis_record.origin.value,
                            latitude=hypothesis_record.latitude,
                            longitude=hypothesis_record.longitude,
                            uncertainty_radius_m=hypothesis_record.uncertainty_radius_m,
                            country_code=hypothesis_record.country_code,
                            region_name=hypothesis_record.region_name,
                            locality_name=hypothesis_record.locality_name,
                            rank=hypothesis_record.rank,
                            confidence_label=hypothesis_record.confidence_label,
                            calibration_state=hypothesis_record.calibration_state.value,
                            supporting_evidence_ids=[
                                str(value) for value in hypothesis_record.supporting_evidence_ids
                            ],
                            model_family_groups=list(hypothesis_record.model_family_groups),
                            created_at=hypothesis_record.created_at,
                            supersedes_hypothesis_id=(
                                None
                                if hypothesis_record.supersedes_hypothesis_id is None
                                else str(hypothesis_record.supersedes_hypothesis_id)
                            ),
                            source_record_key=hypothesis_item.source_record_key,
                        )
                    )
                session.flush()
                if (
                    media_row.storage_state == MediaStorageState.EPHEMERAL.value
                    and not analysis_storage_present
                ):
                    media_row.storage_state = MediaStorageState.DELETED_AFTER_ANALYSIS.value
                media_row.materialized_at = materialized_at
                media_row.analysis_status_at_link = "completed"
                self._touch(case_row, materialized_at)
                self._append_audit(
                    session,
                    case_id=case_id,
                    event_type="analysis.materialized",
                    actor_id=actor_id,
                    actor_type=ActorType.SYSTEM,
                    payload={
                        "analysis_id": str(analysis_id),
                        "case_id": str(case_id),
                        "evidence_count": len(evidence),
                        "hypothesis_count": len(hypotheses),
                        "media_id": str(media_id),
                        "storage_state": media_row.storage_state,
                    },
                    created_at=materialized_at,
                )
                return MaterializationResult(
                    created=True,
                    evidence=tuple(item.record for item in evidence),
                    hypotheses=tuple(item.record for item in hypotheses),
                )

        return await self._run(operation)

    async def list_evidence(
        self,
        case_id: UUID,
        *,
        workspace_id: str,
        media_id: UUID | None,
        analysis_id: UUID | None,
        limit: int,
        offset: int,
    ) -> tuple[list[EvidenceRecord], int]:
        _validate_page(limit, offset)

        def operation() -> tuple[list[EvidenceRecord], int]:
            with self._sessions() as session:
                self._case_for_workspace(session, case_id, workspace_id)
                statement = select(EvidenceRecordRow).where(
                    EvidenceRecordRow.case_id == str(case_id)
                )
                count_statement = (
                    select(func.count())
                    .select_from(EvidenceRecordRow)
                    .where(EvidenceRecordRow.case_id == str(case_id))
                )
                if media_id is not None:
                    if (
                        session.scalar(
                            select(CaseMediaRow.id).where(
                                CaseMediaRow.id == str(media_id),
                                CaseMediaRow.case_id == str(case_id),
                            )
                        )
                        is None
                    ):
                        raise CaseRelationshipError("media does not belong to case")
                    statement = statement.where(EvidenceRecordRow.media_id == str(media_id))
                    count_statement = count_statement.where(
                        EvidenceRecordRow.media_id == str(media_id)
                    )
                if analysis_id is not None:
                    statement = statement.where(EvidenceRecordRow.analysis_id == str(analysis_id))
                    count_statement = count_statement.where(
                        EvidenceRecordRow.analysis_id == str(analysis_id)
                    )
                total = session.scalar(count_statement)
                rows = session.scalars(
                    statement.order_by(EvidenceRecordRow.created_at, EvidenceRecordRow.id)
                    .limit(limit)
                    .offset(offset)
                ).all()
                return [_to_evidence(row) for row in rows], int(total or 0)

        return await self._run(operation)

    async def list_hypotheses(
        self,
        case_id: UUID,
        *,
        workspace_id: str,
        media_id: UUID | None,
        analysis_id: UUID | None,
        limit: int,
        offset: int,
    ) -> tuple[list[LocationHypothesisRecord], int]:
        _validate_page(limit, offset)

        def operation() -> tuple[list[LocationHypothesisRecord], int]:
            with self._sessions() as session:
                self._case_for_workspace(session, case_id, workspace_id)
                statement = select(LocationHypothesisRow).where(
                    LocationHypothesisRow.case_id == str(case_id)
                )
                count_statement = (
                    select(func.count())
                    .select_from(LocationHypothesisRow)
                    .where(LocationHypothesisRow.case_id == str(case_id))
                )
                if media_id is not None:
                    if (
                        session.scalar(
                            select(CaseMediaRow.id).where(
                                CaseMediaRow.id == str(media_id),
                                CaseMediaRow.case_id == str(case_id),
                            )
                        )
                        is None
                    ):
                        raise CaseRelationshipError("media does not belong to case")
                    statement = statement.where(LocationHypothesisRow.media_id == str(media_id))
                    count_statement = count_statement.where(
                        LocationHypothesisRow.media_id == str(media_id)
                    )
                if analysis_id is not None:
                    statement = statement.where(
                        LocationHypothesisRow.analysis_id == str(analysis_id)
                    )
                    count_statement = count_statement.where(
                        LocationHypothesisRow.analysis_id == str(analysis_id)
                    )
                total = session.scalar(count_statement)
                rows = session.scalars(
                    statement.order_by(LocationHypothesisRow.created_at, LocationHypothesisRow.id)
                    .limit(limit)
                    .offset(offset)
                ).all()
                return [_to_hypothesis(row) for row in rows], int(total or 0)

        return await self._run(operation)

    async def get_hypothesis(
        self, case_id: UUID, hypothesis_id: UUID, *, workspace_id: str
    ) -> LocationHypothesisRecord:
        def operation() -> LocationHypothesisRecord:
            with self._sessions() as session:
                self._case_for_workspace(session, case_id, workspace_id)
                row = session.scalar(
                    select(LocationHypothesisRow).where(
                        LocationHypothesisRow.id == str(hypothesis_id),
                        LocationHypothesisRow.case_id == str(case_id),
                    )
                )
                if row is None:
                    raise CaseRelationshipError("hypothesis does not belong to case")
                return _to_hypothesis(row)

        return await self._run(operation)

    async def add_adjudication(
        self, record: AdjudicationRecord, *, workspace_id: str
    ) -> AdjudicationRecord:
        def operation() -> AdjudicationRecord:
            with self._sessions.begin() as session:
                case_row = self._case_for_workspace(
                    session, record.case_id, workspace_id, lock=True
                )
                hypothesis = session.scalar(
                    select(LocationHypothesisRow).where(
                        LocationHypothesisRow.id == str(record.hypothesis_id),
                        LocationHypothesisRow.case_id == str(record.case_id),
                    )
                )
                if hypothesis is None:
                    raise CaseRelationshipError("hypothesis does not belong to case")
                effective_created_at = self._monotonic_append_timestamp(case_row, record.created_at)
                if record.supersedes_adjudication_id is not None:
                    previous = session.scalar(
                        select(AdjudicationRow).where(
                            AdjudicationRow.id == str(record.supersedes_adjudication_id),
                            AdjudicationRow.case_id == str(record.case_id),
                            AdjudicationRow.hypothesis_id == str(record.hypothesis_id),
                        )
                    )
                    if previous is None:
                        raise CaseRelationshipError(
                            "superseded adjudication does not belong to hypothesis"
                        )
                row = AdjudicationRow(
                    id=str(record.id),
                    case_id=str(record.case_id),
                    hypothesis_id=str(record.hypothesis_id),
                    actor_id=record.actor_id,
                    decision=record.decision.value,
                    rationale=record.rationale,
                    created_at=effective_created_at,
                    supersedes_adjudication_id=(
                        None
                        if record.supersedes_adjudication_id is None
                        else str(record.supersedes_adjudication_id)
                    ),
                )
                session.add(row)
                session.flush()
                self._touch(case_row, effective_created_at)
                self._append_audit(
                    session,
                    case_id=record.case_id,
                    event_type="hypothesis.adjudicated",
                    actor_id=record.actor_id,
                    actor_type=ActorType.OPERATOR,
                    payload={
                        "adjudication_id": str(record.id),
                        "decision": record.decision.value,
                        "hypothesis_id": str(record.hypothesis_id),
                        "supersedes_adjudication_id": (
                            None
                            if record.supersedes_adjudication_id is None
                            else str(record.supersedes_adjudication_id)
                        ),
                    },
                    created_at=effective_created_at,
                )
                return _to_adjudication(row)

        return await self._run(operation)

    async def list_adjudications(
        self,
        case_id: UUID,
        *,
        workspace_id: str,
        hypothesis_id: UUID | None,
        limit: int,
        offset: int,
    ) -> tuple[list[AdjudicationRecord], int]:
        _validate_page(limit, offset)

        def operation() -> tuple[list[AdjudicationRecord], int]:
            with self._sessions() as session:
                self._case_for_workspace(session, case_id, workspace_id)
                statement = select(AdjudicationRow).where(AdjudicationRow.case_id == str(case_id))
                count_statement = (
                    select(func.count())
                    .select_from(AdjudicationRow)
                    .where(AdjudicationRow.case_id == str(case_id))
                )
                if hypothesis_id is not None:
                    if (
                        session.scalar(
                            select(LocationHypothesisRow.id).where(
                                LocationHypothesisRow.id == str(hypothesis_id),
                                LocationHypothesisRow.case_id == str(case_id),
                            )
                        )
                        is None
                    ):
                        raise CaseRelationshipError("hypothesis does not belong to case")
                    statement = statement.where(AdjudicationRow.hypothesis_id == str(hypothesis_id))
                    count_statement = count_statement.where(
                        AdjudicationRow.hypothesis_id == str(hypothesis_id)
                    )
                total = session.scalar(count_statement)
                rows = session.scalars(
                    statement.order_by(AdjudicationRow.created_at, AdjudicationRow.id)
                    .limit(limit)
                    .offset(offset)
                ).all()
                return [_to_adjudication(row) for row in rows], int(total or 0)

        return await self._run(operation)

    async def add_operator_correction(
        self,
        hypothesis: LocationHypothesisRecord,
        adjudication: AdjudicationRecord,
        *,
        workspace_id: str,
        source_record_key: str,
    ) -> OperatorCorrectionResult:
        def operation() -> OperatorCorrectionResult:
            with self._sessions.begin() as session:
                case_row = self._case_for_workspace(
                    session, hypothesis.case_id, workspace_id, lock=True
                )
                media = session.scalar(
                    select(CaseMediaRow).where(
                        CaseMediaRow.id == str(hypothesis.media_id),
                        CaseMediaRow.case_id == str(hypothesis.case_id),
                        CaseMediaRow.analysis_id == str(hypothesis.analysis_id),
                    )
                )
                if media is None:
                    raise CaseRelationshipError("analysis/media/case relationship is invalid")
                superseded = session.scalar(
                    select(LocationHypothesisRow).where(
                        LocationHypothesisRow.id == str(hypothesis.supersedes_hypothesis_id),
                        LocationHypothesisRow.case_id == str(hypothesis.case_id),
                        LocationHypothesisRow.media_id == str(hypothesis.media_id),
                        LocationHypothesisRow.analysis_id == str(hypothesis.analysis_id),
                    )
                )
                if superseded is None:
                    raise CaseRelationshipError("superseded hypothesis is invalid")
                requested_created_at = max(
                    _utc(hypothesis.created_at), _utc(adjudication.created_at)
                )
                effective_created_at = self._monotonic_append_timestamp(
                    case_row, requested_created_at
                )
                if hypothesis.supporting_evidence_ids:
                    found = session.scalar(
                        select(func.count())
                        .select_from(EvidenceRecordRow)
                        .where(
                            EvidenceRecordRow.id.in_(
                                [str(value) for value in hypothesis.supporting_evidence_ids]
                            ),
                            EvidenceRecordRow.case_id == str(hypothesis.case_id),
                            EvidenceRecordRow.media_id == str(hypothesis.media_id),
                        )
                    )
                    if int(found or 0) != len(set(hypothesis.supporting_evidence_ids)):
                        raise CaseRelationshipError("supporting evidence is invalid")
                hypothesis_row = LocationHypothesisRow(
                    id=str(hypothesis.id),
                    case_id=str(hypothesis.case_id),
                    media_id=str(hypothesis.media_id),
                    analysis_id=str(hypothesis.analysis_id),
                    origin=hypothesis.origin.value,
                    latitude=hypothesis.latitude,
                    longitude=hypothesis.longitude,
                    uncertainty_radius_m=hypothesis.uncertainty_radius_m,
                    country_code=hypothesis.country_code,
                    region_name=hypothesis.region_name,
                    locality_name=hypothesis.locality_name,
                    rank=hypothesis.rank,
                    confidence_label=hypothesis.confidence_label,
                    calibration_state=hypothesis.calibration_state.value,
                    supporting_evidence_ids=[
                        str(value) for value in hypothesis.supporting_evidence_ids
                    ],
                    model_family_groups=list(hypothesis.model_family_groups),
                    created_at=effective_created_at,
                    supersedes_hypothesis_id=str(hypothesis.supersedes_hypothesis_id),
                    source_record_key=source_record_key,
                )
                adjudication_row = AdjudicationRow(
                    id=str(adjudication.id),
                    case_id=str(adjudication.case_id),
                    hypothesis_id=str(adjudication.hypothesis_id),
                    actor_id=adjudication.actor_id,
                    decision=adjudication.decision.value,
                    rationale=adjudication.rationale,
                    created_at=effective_created_at,
                    supersedes_adjudication_id=None,
                )
                session.add_all((hypothesis_row, adjudication_row))
                session.flush()
                self._touch(case_row, effective_created_at)
                self._append_audit(
                    session,
                    case_id=hypothesis.case_id,
                    event_type="operator_correction.created",
                    actor_id=adjudication.actor_id,
                    actor_type=ActorType.OPERATOR,
                    payload={
                        "adjudication_id": str(adjudication.id),
                        "decision": adjudication.decision.value,
                        "hypothesis_id": str(hypothesis.id),
                        "supersedes_hypothesis_id": str(hypothesis.supersedes_hypothesis_id),
                    },
                    created_at=effective_created_at,
                )
                return OperatorCorrectionResult(
                    hypothesis=_to_hypothesis(hypothesis_row),
                    adjudication=_to_adjudication(adjudication_row),
                )

        return await self._run(operation)

    async def list_audit_events(
        self, case_id: UUID, *, workspace_id: str, limit: int, offset: int
    ) -> tuple[list[AuditEventRecord], int]:
        _validate_page(limit, offset)

        def operation() -> tuple[list[AuditEventRecord], int]:
            with self._sessions() as session:
                self._case_for_workspace(session, case_id, workspace_id)
                condition = AuditEventRow.case_id == str(case_id)
                total = session.scalar(
                    select(func.count()).select_from(AuditEventRow).where(condition)
                )
                rows = session.scalars(
                    select(AuditEventRow)
                    .where(condition)
                    .order_by(AuditEventRow.sequence_number)
                    .limit(limit)
                    .offset(offset)
                ).all()
                return [_to_audit(row) for row in rows], int(total or 0)

        return await self._run(operation)

    async def verify_audit_integrity(
        self, case_id: UUID, *, workspace_id: str
    ) -> AuditIntegrityResult:
        def operation() -> AuditIntegrityResult:
            with self._sessions() as session:
                self._case_for_workspace(session, case_id, workspace_id)
                rows = session.scalars(
                    select(AuditEventRow)
                    .where(AuditEventRow.case_id == str(case_id))
                    .order_by(AuditEventRow.sequence_number)
                ).all()
                previous_hash = GENESIS_AUDIT_HASH
                verified = 0
                for expected_sequence, row in enumerate(rows, start=1):
                    if row.sequence_number != expected_sequence:
                        return AuditIntegrityResult(
                            valid=False,
                            event_count=len(rows),
                            verified_through_sequence=verified,
                            failure_code="sequence_mismatch",
                            failure_sequence=expected_sequence,
                        )
                    if row.previous_event_hash != previous_hash:
                        return AuditIntegrityResult(
                            valid=False,
                            event_count=len(rows),
                            verified_through_sequence=verified,
                            failure_code="previous_hash_mismatch",
                            failure_sequence=expected_sequence,
                        )
                    try:
                        event = _to_audit(row)
                        expected_hash = canonical_audit_hash(
                            event_id=event.id,
                            case_id=event.case_id,
                            sequence_number=event.sequence_number,
                            event_type=event.event_type,
                            actor_id=event.actor_id,
                            actor_type=event.actor_type,
                            payload=event.payload,
                            created_at=event.created_at,
                            previous_event_hash=event.previous_event_hash,
                        )
                    except (TypeError, ValueError):
                        return AuditIntegrityResult(
                            valid=False,
                            event_count=len(rows),
                            verified_through_sequence=verified,
                            failure_code="invalid_payload",
                            failure_sequence=expected_sequence,
                        )
                    if row.event_hash != expected_hash:
                        return AuditIntegrityResult(
                            valid=False,
                            event_count=len(rows),
                            verified_through_sequence=verified,
                            failure_code="event_hash_mismatch",
                            failure_sequence=expected_sequence,
                        )
                    previous_hash = row.event_hash
                    verified = expected_sequence
                return AuditIntegrityResult(
                    valid=True,
                    event_count=len(rows),
                    verified_through_sequence=verified,
                )

        return await self._run(operation)
