from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import Select, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from atlaslens_api.database import AnalysisRow, Base
from atlaslens_api.schemas import Analysis, FailureSummary


class AnalysisNotFoundError(LookupError):
    pass


class DuplicateIdempotencyError(RuntimeError):
    pass


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StoredAnalysis:
    analysis: Analysis
    storage_key: str | None
    idempotency_hash: str | None


@dataclass(frozen=True, slots=True)
class AnalysisHistoryFilters:
    status: str | None = None
    classification: str | None = None
    provider: str | None = None
    search: str | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None


class AnalysisRepository(Protocol):
    async def initialize(self) -> None: ...

    async def is_ready(self) -> bool: ...

    async def create(
        self,
        analysis: Analysis,
        *,
        storage_key: str | None,
        idempotency_hash: str | None,
    ) -> None: ...

    async def get(self, analysis_id: UUID) -> StoredAnalysis | None: ...

    async def get_by_idempotency(self, idempotency_hash: str) -> StoredAnalysis | None: ...

    async def save(self, analysis: Analysis) -> None: ...

    async def clear_storage_key(self, analysis_id: UUID) -> None: ...

    async def delete(self, analysis_id: UUID) -> StoredAnalysis | None: ...

    async def list_expired(self, now: datetime) -> list[StoredAnalysis]: ...

    async def fail_incomplete(self) -> list[StoredAnalysis]: ...

    async def list_history(
        self, *, filters: AnalysisHistoryFilters, limit: int, offset: int
    ) -> tuple[list[StoredAnalysis], int]: ...


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _to_stored(row: AnalysisRow) -> StoredAnalysis:
    cloud_assist = row.cloud_assist_json
    if cloud_assist is not None and row.cloud_assist_cache_key is not None:
        cloud_assist = {**cloud_assist, "cache_key": row.cloud_assist_cache_key}
    analysis = Analysis.model_validate(
        {
            "id": row.id,
            "status": row.status,
            "analysis_mode": row.analysis_mode,
            "created_at": _as_utc(row.created_at),
            "expires_at": _as_utc(row.expires_at),
            "progress": {
                "stage": row.progress_stage,
                "percent": row.progress_percent,
                "message_key": row.progress_message_key,
            },
            "image": row.image_json,
            "quality": row.quality_json,
            "evidence": row.evidence_json,
            "candidates": row.candidates_json,
            "abstention": row.abstention_json,
            "warnings": row.warnings_json,
            "timings_ms": row.timings_json,
            "fusion_policy_version": row.fusion_policy_version,
            "pipeline_version": row.pipeline_version or "legacy-v1",
            "phase5b_diagnostics": row.phase5b_diagnostics_json,
            "scene_analysis": row.scene_analysis_json,
            "model_predictions": row.model_predictions_json,
            "fusion": row.phase6b_fusion_json,
            "ocr": row.phase6b_ocr_json,
            "cloud_assist": cloud_assist,
            "phase6c": row.phase6c_metadata_json,
            "result_classification": row.result_classification or "real",
            "simulation": row.simulation_json,
            "provider_comparisons": row.provider_comparisons_json or [],
            "failure": row.failure_json,
        }
    )
    return StoredAnalysis(
        analysis=analysis,
        storage_key=row.storage_key,
        idempotency_hash=row.idempotency_hash,
    )


def _dump_optional(value: BaseModel | None) -> dict[str, object] | None:
    if value is None:
        return None
    return value.model_dump(mode="json")


def _assign_analysis(row: AnalysisRow, analysis: Analysis) -> None:
    row.status = analysis.status.value
    row.analysis_mode = analysis.analysis_mode.value
    row.created_at = analysis.created_at
    row.expires_at = analysis.expires_at
    row.progress_stage = analysis.progress.stage
    row.progress_percent = analysis.progress.percent
    row.progress_message_key = analysis.progress.message_key
    row.image_json = _dump_optional(analysis.image)
    row.quality_json = _dump_optional(analysis.quality)
    row.evidence_json = [item.model_dump(mode="json") for item in analysis.evidence]
    row.candidates_json = [item.model_dump(mode="json") for item in analysis.candidates]
    row.abstention_json = _dump_optional(analysis.abstention)
    row.warnings_json = list(analysis.warnings)
    row.timings_json = dict(analysis.timings_ms)
    row.fusion_policy_version = analysis.fusion_policy_version
    row.pipeline_version = analysis.pipeline_version
    row.phase5b_diagnostics_json = _dump_optional(analysis.phase5b_diagnostics)
    row.scene_analysis_json = _dump_optional(analysis.scene_analysis)
    row.model_predictions_json = (
        None
        if analysis.model_predictions is None
        else {
            key: value.model_dump(mode="json") for key, value in analysis.model_predictions.items()
        }
    )
    row.phase6b_fusion_json = _dump_optional(analysis.fusion)
    row.phase6b_ocr_json = _dump_optional(analysis.ocr)
    row.cloud_assist_json = _dump_optional(analysis.cloud_assist)
    row.phase6c_metadata_json = _dump_optional(analysis.phase6c)
    row.cloud_assist_cache_key = (
        analysis.cloud_assist.cache_key if analysis.cloud_assist is not None else None
    )
    row.result_classification = analysis.result_classification
    row.simulation_json = _dump_optional(analysis.simulation)
    row.provider_comparisons_json = [
        item.model_dump(mode="json") for item in analysis.provider_comparisons
    ]
    row.failure_json = _dump_optional(analysis.failure)


class SQLAlchemyAnalysisRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._sessions: sessionmaker[Session] = sessionmaker(bind=engine, expire_on_commit=False)
        # SQLite permits only limited concurrent access, and shared in-memory
        # databases can return SQLITE_LOCKED immediately while a worker commits.
        # Serializing this repository's short transactions keeps SSE/poll/delete
        # reads from surfacing transient 500s without changing other backends.
        self._operation_lock = asyncio.Lock()

    async def _run(self, operation: Callable[[], T]) -> T:
        async with self._operation_lock:
            return await asyncio.to_thread(operation)

    async def initialize(self) -> None:
        database = getattr(getattr(self._engine, "url", None), "database", None)
        if database and database != ":memory:":
            parent = Path(database).expanduser().resolve().parent
            await self._run(lambda: parent.mkdir(parents=True, exist_ok=True))
        await self._run(lambda: Base.metadata.create_all(self._engine))

    async def is_ready(self) -> bool:
        def check() -> bool:
            try:
                with self._sessions() as session:
                    session.execute(text("SELECT 1"))
                return True
            except Exception:
                return False

        return await self._run(check)

    async def create(
        self,
        analysis: Analysis,
        *,
        storage_key: str | None,
        idempotency_hash: str | None,
    ) -> None:
        def operation() -> None:
            row = AnalysisRow(
                id=str(analysis.id),
                status=analysis.status.value,
                analysis_mode=analysis.analysis_mode.value,
                created_at=analysis.created_at,
                expires_at=analysis.expires_at,
                progress_stage=analysis.progress.stage,
                progress_percent=analysis.progress.percent,
                progress_message_key=analysis.progress.message_key,
                image_json=None,
                quality_json=None,
                evidence_json=[],
                candidates_json=[],
                abstention_json=None,
                warnings_json=[],
                timings_json={},
                fusion_policy_version=analysis.fusion_policy_version,
                pipeline_version=analysis.pipeline_version,
                phase5b_diagnostics_json=None,
                scene_analysis_json=None,
                model_predictions_json=None,
                phase6b_fusion_json=None,
                phase6b_ocr_json=None,
                cloud_assist_json=None,
                cloud_assist_cache_key=None,
                phase6c_metadata_json=None,
                result_classification=analysis.result_classification,
                simulation_json=None,
                provider_comparisons_json=[],
                failure_json=None,
                storage_key=storage_key,
                idempotency_hash=idempotency_hash,
            )
            _assign_analysis(row, analysis)
            try:
                with self._sessions.begin() as session:
                    session.add(row)
            except IntegrityError as exc:
                raise DuplicateIdempotencyError from exc

        await self._run(operation)

    async def get(self, analysis_id: UUID) -> StoredAnalysis | None:
        def operation() -> StoredAnalysis | None:
            with self._sessions() as session:
                row = session.get(AnalysisRow, str(analysis_id))
                return None if row is None else _to_stored(row)

        return await self._run(operation)

    async def get_by_idempotency(self, idempotency_hash: str) -> StoredAnalysis | None:
        def operation() -> StoredAnalysis | None:
            with self._sessions() as session:
                row = session.scalar(
                    select(AnalysisRow).where(AnalysisRow.idempotency_hash == idempotency_hash)
                )
                return None if row is None else _to_stored(row)

        return await self._run(operation)

    async def save(self, analysis: Analysis) -> None:
        def operation() -> None:
            with self._sessions.begin() as session:
                row = session.get(AnalysisRow, str(analysis.id))
                if row is None:
                    raise AnalysisNotFoundError
                _assign_analysis(row, analysis)

        await self._run(operation)

    async def clear_storage_key(self, analysis_id: UUID) -> None:
        def operation() -> None:
            with self._sessions.begin() as session:
                row = session.get(AnalysisRow, str(analysis_id))
                if row is not None:
                    row.storage_key = None

        await self._run(operation)

    async def delete(self, analysis_id: UUID) -> StoredAnalysis | None:
        def operation() -> StoredAnalysis | None:
            with self._sessions.begin() as session:
                row = session.get(AnalysisRow, str(analysis_id))
                if row is None:
                    return None
                stored = _to_stored(row)
                session.delete(row)
                return stored

        return await self._run(operation)

    async def list_expired(self, now: datetime) -> list[StoredAnalysis]:
        def operation() -> list[StoredAnalysis]:
            with self._sessions() as session:
                rows = session.scalars(
                    select(AnalysisRow).where(
                        AnalysisRow.expires_at.is_not(None), AnalysisRow.expires_at <= now
                    )
                ).all()
                return [_to_stored(row) for row in rows]

        return await self._run(operation)

    async def fail_incomplete(self) -> list[StoredAnalysis]:
        def operation() -> list[StoredAnalysis]:
            with self._sessions.begin() as session:
                rows = session.scalars(
                    select(AnalysisRow).where(AnalysisRow.status.in_(("queued", "processing")))
                ).all()
                stored = [_to_stored(row) for row in rows]
                for row in rows:
                    row.status = "failed"
                    row.progress_stage = "failed"
                    row.progress_percent = 100
                    row.progress_message_key = "progress.failed"
                    row.failure_json = FailureSummary(
                        code="job_interrupted",
                        message_key="error.job_interrupted",
                        retryable=True,
                    ).model_dump(mode="json")
            return stored

        return await self._run(operation)

    async def list_history(
        self, *, filters: AnalysisHistoryFilters, limit: int, offset: int
    ) -> tuple[list[StoredAnalysis], int]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("invalid history pagination")

        def operation() -> tuple[list[StoredAnalysis], int]:
            statement: Select[tuple[AnalysisRow]] = select(AnalysisRow)
            if filters.status is not None:
                statement = statement.where(AnalysisRow.status == filters.status)
            if filters.classification is not None:
                statement = statement.where(
                    AnalysisRow.result_classification == filters.classification
                )
            if filters.created_from is not None:
                statement = statement.where(AnalysisRow.created_at >= filters.created_from)
            if filters.created_to is not None:
                statement = statement.where(AnalysisRow.created_at <= filters.created_to)
            statement = statement.order_by(AnalysisRow.created_at.desc(), AnalysisRow.id.desc())
            with self._sessions() as session:
                stored = [_to_stored(row) for row in session.scalars(statement).all()]

            provider_filter = filters.provider.casefold() if filters.provider else None
            search_filter = filters.search.casefold() if filters.search else None

            def included(item: StoredAnalysis) -> bool:
                analysis = item.analysis
                provider_ids = {
                    provenance.provider_id
                    for candidate in analysis.candidates
                    for provenance in candidate.provenance
                } | {evidence.provenance.provider_id for evidence in analysis.evidence}
                if provider_filter is not None and all(
                    provider_filter not in provider_id.casefold() for provider_id in provider_ids
                ):
                    return False
                if search_filter is not None:
                    searchable = " ".join(
                        candidate.label or "" for candidate in analysis.candidates
                    ).casefold()
                    if search_filter not in searchable:
                        return False
                return True

            filtered = [item for item in stored if included(item)]
            return filtered[offset : offset + limit], len(filtered)

        return await self._run(operation)
