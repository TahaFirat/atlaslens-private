from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
    Progress,
    SceneClassSummary,
    SceneSegmentationSummary,
)


def _analysis() -> Analysis:
    now = datetime.now(UTC)
    return Analysis(
        id=UUID(int=6),
        status=AnalysisStatus.QUEUED,
        analysis_mode=AnalysisMode.LOCAL_ONLY,
        created_at=now,
        expires_at=now + timedelta(hours=1),
        progress=Progress(stage="queued", percent=0, message_key="progress.queued"),
        evidence=[],
        candidates=[],
        warnings=[],
        timings_ms={},
        fusion_policy_version="phase6a-v1",
        scene_analysis=SceneSegmentationSummary(
            provider="segformer-scene-v1",
            device="cpu",
            inference_ms=12.5,
            image_width=640,
            image_height=480,
            semantic_label_names_available=False,
            dominant_classes=[
                SceneClassSummary(
                    class_id=7,
                    class_name="class_7",
                    pixel_ratio=0.625,
                    percentage=62.5,
                )
            ],
            warnings=["generic_label_names"],
        ),
    )


async def test_scene_analysis_round_trips_and_can_be_cleared() -> None:
    repository = SQLAlchemyAnalysisRepository(create_database_engine("sqlite:///:memory:"))
    await repository.initialize()
    analysis = _analysis()

    await repository.create(analysis, storage_key=None, idempotency_hash=None)
    stored = await repository.get(analysis.id)

    assert stored is not None
    assert stored.analysis.scene_analysis == analysis.scene_analysis

    await repository.save(analysis.model_copy(update={"scene_analysis": None}))
    cleared = await repository.get(analysis.id)
    assert cleared is not None
    assert cleared.analysis.scene_analysis is None


def _migration_config(database: Path) -> Config:
    root = Path(__file__).parents[1]
    config = Config(root / "alembic.ini")
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")
    return config


def _analysis_columns(database: Path) -> set[str]:
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        return {column["name"] for column in inspect(engine).get_columns("analyses")}
    finally:
        engine.dispose()


def test_scene_analysis_migration_upgrades_from_previous_head(tmp_path: Path) -> None:
    database = tmp_path / "phase6a-upgrade.sqlite3"
    config = _migration_config(database)

    command.upgrade(config, "0005")
    assert "scene_analysis_json" not in _analysis_columns(database)

    command.upgrade(config, "0006")
    assert "scene_analysis_json" in _analysis_columns(database)

    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        with engine.connect() as connection:
            version = connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
        assert version == "0006"
    finally:
        engine.dispose()


def test_scene_analysis_migration_downgrades_to_previous_head(tmp_path: Path) -> None:
    database = tmp_path / "phase6a-downgrade.sqlite3"
    config = _migration_config(database)

    command.upgrade(config, "0006")
    assert "scene_analysis_json" in _analysis_columns(database)

    command.downgrade(config, "0005")
    assert "scene_analysis_json" not in _analysis_columns(database)

    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        with engine.connect() as connection:
            version = connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
        assert version == "0005"
    finally:
        engine.dispose()
