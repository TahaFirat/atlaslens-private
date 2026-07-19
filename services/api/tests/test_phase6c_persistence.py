from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from atlaslens_api.database import create_database_engine
from atlaslens_api.repository import SQLAlchemyAnalysisRepository
from atlaslens_api.schemas import (
    Analysis,
    AnalysisMode,
    AnalysisStatus,
    Phase6CAnalysisSummary,
    Phase6CHierarchicalCandidateSummary,
    Phase6CLeakageAuditSummary,
    Phase6CProviderRunSummary,
    Progress,
)


def _analysis() -> Analysis:
    return Analysis(
        id=UUID(int=61),
        status=AnalysisStatus.QUEUED,
        analysis_mode=AnalysisMode.LOCAL_ONLY,
        created_at=datetime.now(UTC),
        expires_at=None,
        progress=Progress(stage="queued", percent=0, message_key="progress.queued"),
        evidence=[],
        candidates=[],
        warnings=[],
        timings_ms={},
        fusion_policy_version="phase6c-v1",
        pipeline_version="phase6c-v1",
        phase6c=Phase6CAnalysisSummary(
            reference_index_version=None,
            hierarchical_candidates=[
                Phase6CHierarchicalCandidateSummary(
                    candidate_id="hier-0123456789abcdef0123",
                    latitude=39.0,
                    longitude=35.0,
                    search_level="global",
                    provider_rank=1,
                    provider_score=0.12,
                    grid_resolution_km=400,
                    diversity_cluster="mode-01",
                )
            ],
            providers=[
                Phase6CProviderRunSummary(
                    provider_id="geoclip-hierarchical",
                    status="completed",
                    source_revision="geoclip-1.2.0",
                    model_revision="1.2.0",
                    duration_ms=20,
                    candidates_produced=1,
                )
            ],
            leakage_audit=Phase6CLeakageAuditSummary(
                status="not_run",
                audit_version="phase6c-leakage-v1",
                references_checked=0,
                references_excluded=0,
                reason_code="reference_index_unavailable",
            ),
            cache_fingerprint=f"phase6c-v1:{'a' * 64}",
        ),
    )


async def test_phase6c_pipeline_metadata_round_trips_without_paths_or_secrets() -> None:
    repository = SQLAlchemyAnalysisRepository(create_database_engine("sqlite:///:memory:"))
    await repository.initialize()
    analysis = _analysis()

    await repository.create(analysis, storage_key=None, idempotency_hash=None)
    stored = await repository.get(analysis.id)

    assert stored is not None
    assert stored.analysis.pipeline_version == "phase6c-v1"
    assert stored.analysis.phase6c == analysis.phase6c
    payload = stored.analysis.model_dump_json()
    assert "filesystem" not in payload and "access_token" not in payload


def _migration_config(database: Path) -> Config:
    root = Path(__file__).parents[1]
    config = Config(root / "alembic.ini")
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")
    return config


def _columns(database: Path) -> set[str]:
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        return {column["name"] for column in inspect(engine).get_columns("analyses")}
    finally:
        engine.dispose()


def test_phase6c_migration_preserves_legacy_rows_and_downgrades(tmp_path: Path) -> None:
    database = tmp_path / "phase6c-migration.sqlite3"
    config = _migration_config(database)

    command.upgrade(config, "0007")
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO analyses (id,status,analysis_mode,created_at,progress_stage,"
                    "progress_percent,progress_message_key,evidence_json,candidates_json,"
                    "warnings_json,timings_json,fusion_policy_version,result_classification) "
                    "VALUES ('legacy','queued','local_only',CURRENT_TIMESTAMP,'queued',0,"
                    "'progress.queued','[]','[]','[]','{}','phase6b-v1','real')"
                )
            )
    finally:
        engine.dispose()

    command.upgrade(config, "0008")
    assert {"pipeline_version", "phase6c_metadata_json"} <= _columns(database)
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        with engine.connect() as connection:
            value = connection.execute(
                text("SELECT pipeline_version FROM analyses WHERE id='legacy'")
            ).scalar_one()
        assert value == "legacy-v1"
    finally:
        engine.dispose()

    command.downgrade(config, "0007")
    assert not {"pipeline_version", "phase6c_metadata_json"}.intersection(
        _columns(database)
    )
