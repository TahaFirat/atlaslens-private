from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from atlaslens_api.database import Base


class CaseRow(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    purpose: Mapped[str] = mapped_column(String(48), nullable=False)
    purpose_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_context: Mapped[str] = mapped_column(Text, nullable=False)
    authorization_attested: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    sensitivity: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by_actor_id: Mapped[str] = mapped_column(String(160), nullable=False)
    retention_policy: Mapped[str] = mapped_column(String(160), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "purpose IN ('journalism','humanitarian','disaster_response','insurance',"
            "'authorized_security_research','other')",
            name="ck_cases_purpose",
        ),
        CheckConstraint(
            "purpose <> 'other' OR (purpose_detail IS NOT NULL "
            "AND length(trim(purpose_detail)) > 0)",
            name="ck_cases_other_detail",
        ),
        CheckConstraint(
            "status IN ('open','under_review','resolved','archived')",
            name="ck_cases_status",
        ),
        CheckConstraint(
            "sensitivity IN ('standard','sensitive','conflict_related')",
            name="ck_cases_sensitivity",
        ),
        CheckConstraint("authorization_attested = true", name="ck_cases_authorized"),
        CheckConstraint("version > 0", name="ck_cases_version_positive"),
        Index("ix_cases_workspace_updated", "workspace_id", "updated_at", "id"),
    )


class CaseMediaRow(Base):
    __tablename__ = "case_media"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False
    )
    # Existing retention may delete an analysis while case metadata remains.
    analysis_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    media_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source_type: Mapped[str] = mapped_column(String(24), nullable=False)
    original_filename_display: Mapped[str] = mapped_column(String(160), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(80), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    archive_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    authorization_attested: Mapped[bool] = mapped_column(Boolean, nullable=False)
    storage_state: Mapped[str] = mapped_column(String(32), nullable=False)
    analysis_status_at_link: Mapped[str | None] = mapped_column(String(24), nullable=True)
    analysis_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    analysis_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    materialized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("analysis_id", name="uq_case_media_analysis"),
        CheckConstraint("media_type = 'image'", name="ck_case_media_image_only"),
        CheckConstraint(
            "source_type IN ('upload','source_url','external_archive','other')",
            name="ck_case_media_source_type",
        ),
        CheckConstraint(
            "mime_type IN ('image/jpeg','image/png','image/webp')",
            name="ck_case_media_mime_type",
        ),
        CheckConstraint("byte_size > 0", name="ck_case_media_byte_size_positive"),
        CheckConstraint("length(sha256) = 64", name="ck_case_media_sha256_length"),
        CheckConstraint("authorization_attested = true", name="ck_case_media_authorized"),
        CheckConstraint(
            "storage_state IN ('ephemeral','deleted_after_analysis','unavailable',"
            "'externally_managed')",
            name="ck_case_media_storage_state",
        ),
        Index("ix_case_media_case_created", "case_id", "created_at", "id"),
    )


class EvidenceRecordRow(Base):
    __tablename__ = "case_evidence_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False
    )
    media_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("case_media.id", ondelete="RESTRICT"), nullable=False
    )
    analysis_id: Mapped[str] = mapped_column(String(36), nullable=False)
    evidence_type: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(160), nullable=False)
    provider_family: Mapped[str] = mapped_column(String(160), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    structured_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    immutable_source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_record_key: Mapped[str] = mapped_column(String(180), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "case_id",
            "media_id",
            "analysis_id",
            "source_record_key",
            name="uq_case_evidence_source",
        ),
        CheckConstraint(
            "evidence_type IN ('metadata','ocr','visual_clue','model_hypothesis',"
            "'retrieval_match','map_evidence','analyst_note')",
            name="ck_case_evidence_type",
        ),
        CheckConstraint("length(immutable_source_hash) = 64", name="ck_case_evidence_hash_length"),
        Index("ix_case_evidence_case_created", "case_id", "created_at", "id"),
        Index("ix_case_evidence_analysis", "analysis_id"),
    )


class LocationHypothesisRow(Base):
    __tablename__ = "case_location_hypotheses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False
    )
    media_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("case_media.id", ondelete="RESTRICT"), nullable=False
    )
    analysis_id: Mapped[str] = mapped_column(String(36), nullable=False)
    origin: Mapped[str] = mapped_column(String(32), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty_radius_m: Mapped[float] = mapped_column(Float, nullable=False)
    country_code: Mapped[str | None] = mapped_column(String(2), nullable=True)
    region_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    locality_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence_label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    calibration_state: Mapped[str] = mapped_column(String(24), nullable=False)
    supporting_evidence_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    model_family_groups: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    supersedes_hypothesis_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("case_location_hypotheses.id", ondelete="RESTRICT"), nullable=True
    )
    source_record_key: Mapped[str] = mapped_column(String(180), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "case_id",
            "media_id",
            "analysis_id",
            "source_record_key",
            name="uq_case_hypothesis_source",
        ),
        CheckConstraint(
            "origin IN ('model','operator_correction','imported')",
            name="ck_case_hypothesis_origin",
        ),
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="ck_case_hypothesis_lat"),
        CheckConstraint("longitude >= -180 AND longitude <= 180", name="ck_case_hypothesis_lon"),
        CheckConstraint("uncertainty_radius_m > 0", name="ck_case_hypothesis_uncertainty_positive"),
        CheckConstraint("rank IS NULL OR rank > 0", name="ck_case_hypothesis_rank_positive"),
        CheckConstraint(
            "calibration_state IN ('calibrated','uncalibrated','not_applicable')",
            name="ck_case_hypothesis_calibration",
        ),
        Index("ix_case_hypothesis_case_created", "case_id", "created_at", "id"),
        Index("ix_case_hypothesis_analysis", "analysis_id"),
    )


class AdjudicationRow(Base):
    __tablename__ = "case_adjudications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False
    )
    hypothesis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("case_location_hypotheses.id", ondelete="RESTRICT"), nullable=False
    )
    actor_id: Mapped[str] = mapped_column(String(160), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    supersedes_adjudication_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("case_adjudications.id", ondelete="RESTRICT"), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "decision IN ('accepted','rejected','needs_more_evidence','withdrawn')",
            name="ck_case_adjudication_decision",
        ),
        Index("ix_case_adjudication_case_created", "case_id", "created_at", "id"),
        Index("ix_case_adjudication_hypothesis", "hypothesis_id", "created_at", "id"),
    )


class AuditEventRow(Base):
    __tablename__ = "case_audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(120), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(160), nullable=False)
    actor_type: Mapped[str] = mapped_column(String(24), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    previous_event_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint("case_id", "sequence_number", name="uq_case_audit_sequence"),
        UniqueConstraint("event_hash", name="uq_case_audit_event_hash"),
        CheckConstraint("sequence_number > 0", name="ck_case_audit_sequence_positive"),
        CheckConstraint(
            "actor_type IN ('operator','system','model')", name="ck_case_audit_actor_type"
        ),
        CheckConstraint(
            "length(previous_event_hash) = 64", name="ck_case_audit_previous_hash_length"
        ),
        CheckConstraint("length(event_hash) = 64", name="ck_case_audit_hash_length"),
        Index("ix_case_audit_case_sequence", "case_id", "sequence_number"),
    )
