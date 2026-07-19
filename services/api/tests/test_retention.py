from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from atlaslens_api.cleanup import DefaultRetentionCleanupService
from atlaslens_api.database import create_database_engine
from atlaslens_api.repository import SQLAlchemyAnalysisRepository
from atlaslens_api.schemas import Analysis, AnalysisMode, AnalysisStatus, Progress
from atlaslens_api.storage import LocalTemporaryStorage


def queued_analysis(*, expired: bool) -> Analysis:
    now = datetime.now(UTC)
    return Analysis(
        id=uuid4(),
        status=AnalysisStatus.QUEUED,
        analysis_mode=AnalysisMode.LOCAL_ONLY,
        created_at=now,
        expires_at=now - timedelta(seconds=1) if expired else now + timedelta(hours=1),
        progress=Progress(stage="queued", percent=0, message_key="progress.queued"),
        evidence=[],
        candidates=[],
        warnings=[],
        timings_ms={},
        fusion_policy_version="phase1-fusion-v1",
    )


@pytest.mark.asyncio
async def test_ttl_cleanup_deletes_record_and_artifact(tmp_path: Path) -> None:
    engine = create_database_engine(f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}")
    repository = SQLAlchemyAnalysisRepository(engine)
    storage = LocalTemporaryStorage(tmp_path / "tmp")
    await repository.initialize()
    await storage.initialize()
    handle = await storage.create_temporary(".upload")
    handle.path.write_bytes(b"private")
    analysis = queued_analysis(expired=True)
    await repository.create(analysis, storage_key=handle.key, idempotency_hash=None)
    cleanup = DefaultRetentionCleanupService(repository, storage, ttl_seconds=1)
    await cleanup.run_once()
    assert await repository.get(analysis.id) is None
    assert not handle.path.exists()
    engine.dispose()


@pytest.mark.asyncio
async def test_startup_marks_non_durable_jobs_failed_and_removes_upload(tmp_path: Path) -> None:
    engine = create_database_engine(f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}")
    repository = SQLAlchemyAnalysisRepository(engine)
    storage = LocalTemporaryStorage(tmp_path / "tmp")
    await repository.initialize()
    await storage.initialize()
    handle = await storage.create_temporary(".upload")
    handle.path.write_bytes(b"private")
    analysis = queued_analysis(expired=False)
    await repository.create(analysis, storage_key=handle.key, idempotency_hash=None)
    cleanup = DefaultRetentionCleanupService(repository, storage, ttl_seconds=3600)
    await cleanup.startup_cleanup()
    stored = await repository.get(analysis.id)
    assert stored is not None
    assert stored.analysis.status == AnalysisStatus.FAILED
    assert stored.analysis.failure is not None
    assert stored.analysis.failure.code == "job_interrupted"
    assert stored.storage_key is None
    assert not handle.path.exists()
    engine.dispose()


@pytest.mark.asyncio
async def test_periodic_cleanup_runs_on_bounded_interval(tmp_path: Path) -> None:
    engine = create_database_engine(f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}")
    repository = SQLAlchemyAnalysisRepository(engine)
    storage = LocalTemporaryStorage(tmp_path / "tmp")
    await repository.initialize()
    await storage.initialize()
    analysis = queued_analysis(expired=True)
    await repository.create(analysis, storage_key=None, idempotency_hash=None)
    cleanup = DefaultRetentionCleanupService(
        repository, storage, ttl_seconds=1, interval_seconds=0.01
    )
    await cleanup.start()
    deadline = asyncio.get_running_loop().time() + 1.0
    while await repository.get(analysis.id) is not None:
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)
    await cleanup.stop()
    assert await repository.get(analysis.id) is None
    engine.dispose()
