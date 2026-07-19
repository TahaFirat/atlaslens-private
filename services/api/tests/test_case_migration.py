from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError
from sqlalchemy.schema import CreateTable

from atlaslens_api.cases.models import (
    AdjudicationRow,
    AuditEventRow,
    CaseMediaRow,
    CaseRow,
    EvidenceRecordRow,
    LocationHypothesisRow,
)
from atlaslens_api.database import create_database_engine

CASE_TABLES = {
    "cases",
    "case_media",
    "case_evidence_records",
    "case_location_hypotheses",
    "case_adjudications",
    "case_audit_events",
}


def _config(database: Path) -> Config:
    root = Path(__file__).parents[1]
    config = Config(root / "alembic.ini")
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")
    return config


def _tables(database: Path) -> set[str]:
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_0009_upgrades_a_fresh_database_and_downgrades_to_0008(tmp_path: Path) -> None:
    database = tmp_path / "fresh-phase2.sqlite3"
    config = _config(database)
    command.upgrade(config, "0009")

    assert _tables(database) >= CASE_TABLES
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        with engine.connect() as connection:
            revision = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            triggers = (
                connection.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='trigger' "
                        "AND name LIKE 'trg_case_%_no_%'"
                    )
                )
                .scalars()
                .all()
            )
        assert revision == "0009"
        assert len(triggers) == 8
    finally:
        engine.dispose()

    command.downgrade(config, "0008")
    remaining = _tables(database)
    assert not CASE_TABLES.intersection(remaining)
    assert "analyses" in remaining


def test_0009_preserves_legacy_analysis_and_guards_append_only_rows(tmp_path: Path) -> None:
    database = tmp_path / "legacy-phase2.sqlite3"
    config = _config(database)
    command.upgrade(config, "0008")
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO analyses (id,status,analysis_mode,created_at,progress_stage,"
                    "progress_percent,progress_message_key,evidence_json,candidates_json,"
                    "warnings_json,timings_json,fusion_policy_version,pipeline_version,"
                    "result_classification) VALUES "
                    "('legacy-phase2','queued','local_only',CURRENT_TIMESTAMP,'queued',0,"
                    "'synthetic','[]','[]','[]','{}','legacy-v1','legacy-v1','real')"
                )
            )
    finally:
        engine.dispose()

    command.upgrade(config, "0009")
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        with engine.begin() as connection:
            assert (
                connection.execute(
                    text("SELECT COUNT(*) FROM analyses WHERE id='legacy-phase2'")
                ).scalar_one()
                == 1
            )
            connection.execute(
                text(
                    "INSERT INTO cases "
                    "(id,workspace_id,title,purpose,source_context,authorization_attested,"
                    "status,sensitivity,created_at,updated_at,created_by_actor_id,"
                    "retention_policy,version) VALUES "
                    "('00000000-0000-0000-0000-000000000001','local-default','Synthetic',"
                    "'journalism','Synthetic context',1,'open','standard',CURRENT_TIMESTAMP,"
                    "CURRENT_TIMESTAMP,'synthetic-analyst','temporary',1)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO case_audit_events "
                    "(id,case_id,sequence_number,event_type,actor_id,actor_type,payload,"
                    "created_at,previous_event_hash,event_hash) VALUES "
                    "('00000000-0000-0000-0000-000000000002',"
                    "'00000000-0000-0000-0000-000000000001',1,'case.created',"
                    "'synthetic-analyst','operator','{}',CURRENT_TIMESTAMP,:previous,:event)"
                ),
                {"previous": "0" * 64, "event": "a" * 64},
            )
        with (
            pytest.raises(DBAPIError, match="append-only case history"),
            engine.begin() as connection,
        ):
            connection.execute(
                text("UPDATE case_audit_events SET event_hash=:value"),
                {"value": "b" * 64},
            )
    finally:
        engine.dispose()

    command.downgrade(config, "0008")
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    try:
        with engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT COUNT(*) FROM analyses WHERE id='legacy-phase2'")
                ).scalar_one()
                == 1
            )
    finally:
        engine.dispose()


def test_case_tables_compile_for_postgresql_without_native_only_types() -> None:
    tables = (
        CaseRow.__table__,
        CaseMediaRow.__table__,
        EvidenceRecordRow.__table__,
        LocationHypothesisRow.__table__,
        AdjudicationRow.__table__,
        AuditEventRow.__table__,
    )
    for table in tables:
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        assert f"CREATE TABLE {table.name}" in ddl
        assert "JSON" in ddl or table.name in {"cases", "case_media", "case_adjudications"}
