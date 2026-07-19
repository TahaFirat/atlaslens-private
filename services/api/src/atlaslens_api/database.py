from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Index, Integer, String, create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AnalysisRow(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    analysis_mode: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    progress_stage: Mapped[str] = mapped_column(String(64), nullable=False)
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False)
    progress_message_key: Mapped[str] = mapped_column(String(128), nullable=False)
    image_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    quality_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    evidence_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    candidates_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    abstention_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    warnings_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    timings_json: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False)
    fusion_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="legacy-v1", server_default="legacy-v1"
    )
    phase5b_diagnostics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    scene_analysis_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_predictions_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    phase6b_fusion_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    phase6b_ocr_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    cloud_assist_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    cloud_assist_cache_key: Mapped[str | None] = mapped_column(String(67), nullable=True)
    phase6c_metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    result_classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default="real", server_default="real", index=True
    )
    simulation_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    provider_comparisons_json: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    failure_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    idempotency_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    storage_key: Mapped[str | None] = mapped_column(String(80), nullable=True)

    __table_args__ = (Index("ix_analyses_expiry_status", "expires_at", "status"),)


def create_database_engine(database_url: str) -> Engine:
    if database_url.startswith("postgres://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgres://") :]
    elif database_url.startswith("postgresql://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgresql://") :]

    url = make_url(database_url)
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.get_backend_name() == "sqlite":
        kwargs["connect_args"] = {"check_same_thread": False}
        if url.database == ":memory:":
            # A StaticPool exposes one sqlite3 connection to every request thread.
            # Concurrent polling/SSE/delete calls can then use that connection at
            # the same time and trigger sqlite3.InterfaceError. A unique named
            # shared-cache database keeps in-memory isolation while allowing the
            # normal SQLAlchemy pool to provide one connection per concurrent use.
            memory_name = f"atlaslens_{uuid4().hex}"
            database_url = f"sqlite:///file:{memory_name}?mode=memory&cache=shared&uri=true"
    return create_engine(database_url, **kwargs)
