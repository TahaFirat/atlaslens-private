from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from atlaslens_api.database import create_database_engine
from atlaslens_api.repository import SQLAlchemyAnalysisRepository
from atlaslens_api.schemas import (
    Analysis,
    AnalysisMode,
    AnalysisStatus,
    Phase6BAgreementSummary,
    Phase6BCloudAssistSummary,
    Phase6BFusionSummary,
    Phase6BOCRSummary,
    Phase6BProviderPredictionSummary,
    Progress,
)


def _analysis() -> Analysis:
    return Analysis(
        id=UUID(int=60),
        status=AnalysisStatus.QUEUED,
        analysis_mode=AnalysisMode.LOCAL_ONLY,
        created_at=datetime.now(UTC),
        expires_at=None,
        progress=Progress(stage="queued", percent=0, message_key="progress.queued"),
        evidence=[],
        candidates=[],
        warnings=[],
        timings_ms={},
        fusion_policy_version="phase6b-v1",
        model_predictions={
            "osv5m": Phase6BProviderPredictionSummary(
                provider="osv5m-baseline",
                model_id="osv5m/baseline",
                model_revision="pinned-revision",
                source_family="osv5m_family",
                status="skipped",
                device="cpu",
                duration_ms=0,
                score_semantics="direct_regression",
                reason_code="isolated_worker_not_installed",
            )
        },
        fusion=Phase6BFusionSummary(
            agreement_summary=Phase6BAgreementSummary(
                provider_count=0,
                independent_family_count=0,
                same_family_duplicate_support=0,
                geographic_disagreement=False,
                ocr_agreement=False,
                ocr_contradiction=False,
            )
        ),
        ocr=Phase6BOCRSummary(
            provider="paddleocr-ppocr-local",
            status="skipped",
            reason_code="missing_dependency",
        ),
        cloud_assist=Phase6BCloudAssistSummary(
            allowed=False,
            triggered=False,
            status="skipped",
            model="gpt-5.6-luna",
            reason="cloud_consent_denied",
            cache_key=f"v1:{'a' * 64}",
        ),
    )


async def test_phase6b_summaries_and_private_cache_key_round_trip() -> None:
    repository = SQLAlchemyAnalysisRepository(create_database_engine("sqlite:///:memory:"))
    await repository.initialize()
    analysis = _analysis()

    await repository.create(analysis, storage_key=None, idempotency_hash=None)
    stored = await repository.get(analysis.id)

    assert stored is not None
    assert stored.analysis.model_predictions == analysis.model_predictions
    assert stored.analysis.fusion == analysis.fusion
    assert stored.analysis.ocr == analysis.ocr
    assert stored.analysis.cloud_assist == analysis.cloud_assist
    assert "cache_key" not in stored.analysis.model_dump(mode="json")["cloud_assist"]


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


def test_phase6b_migration_upgrades_and_downgrades_from_phase6a(tmp_path: Path) -> None:
    database = tmp_path / "phase6b-migration.sqlite3"
    config = _migration_config(database)
    expected = {
        "model_predictions_json",
        "phase6b_fusion_json",
        "phase6b_ocr_json",
        "cloud_assist_json",
        "cloud_assist_cache_key",
    }

    command.upgrade(config, "0006")
    assert not expected.intersection(_columns(database))

    command.upgrade(config, "0007")
    assert expected <= _columns(database)

    command.downgrade(config, "0006")
    assert not expected.intersection(_columns(database))
