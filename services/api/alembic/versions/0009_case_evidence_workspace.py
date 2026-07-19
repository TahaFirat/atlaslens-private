"""Add the Phase 2 case and evidence investigation workspace.

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | None = None
depends_on: str | None = None

_APPEND_ONLY_TABLES = (
    "case_evidence_records",
    "case_location_hypotheses",
    "case_adjudications",
    "case_audit_events",
)


def _install_append_only_guards() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "sqlite":
        for table in _APPEND_ONLY_TABLES:
            for action in ("UPDATE", "DELETE"):
                trigger = f"trg_{table}_no_{action.casefold()}"
                op.execute(  # noqa: S608 - table/action are fixed migration constants.
                    sa.text(
                        f"CREATE TRIGGER {trigger} BEFORE {action} ON {table} "
                        "BEGIN SELECT RAISE(ABORT, 'append-only case history'); END"
                    )
                )
    elif dialect == "postgresql":
        op.execute(
            sa.text(
                "CREATE FUNCTION atlaslens_reject_case_history_mutation() "
                "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
                "RAISE EXCEPTION 'append-only case history'; END; $$"
            )
        )
        for table in _APPEND_ONLY_TABLES:
            trigger = f"trg_{table}_append_only"
            op.execute(  # noqa: S608 - table/trigger are fixed migration constants.
                sa.text(
                    f"CREATE TRIGGER {trigger} BEFORE UPDATE OR DELETE ON {table} "
                    "FOR EACH ROW EXECUTE FUNCTION atlaslens_reject_case_history_mutation()"
                )
            )


def _drop_append_only_guards() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "sqlite":
        for table in _APPEND_ONLY_TABLES:
            for action in ("update", "delete"):
                op.execute(sa.text(f"DROP TRIGGER IF EXISTS trg_{table}_no_{action}"))  # noqa: S608
    elif dialect == "postgresql":
        for table in _APPEND_ONLY_TABLES:
            op.execute(  # noqa: S608 - table/trigger are fixed migration constants.
                sa.text(f"DROP TRIGGER IF EXISTS trg_{table}_append_only ON {table}")
            )
        op.execute(sa.text("DROP FUNCTION IF EXISTS atlaslens_reject_case_history_mutation()"))


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("purpose", sa.String(length=48), nullable=False),
        sa.Column("purpose_detail", sa.Text(), nullable=True),
        sa.Column("source_context", sa.Text(), nullable=False),
        sa.Column("authorization_attested", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("sensitivity", sa.String(length=24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by_actor_id", sa.String(length=160), nullable=False),
        sa.Column("retention_policy", sa.String(length=160), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "purpose IN ('journalism','humanitarian','disaster_response','insurance',"
            "'authorized_security_research','other')",
            name="ck_cases_purpose",
        ),
        sa.CheckConstraint(
            "purpose <> 'other' OR (purpose_detail IS NOT NULL "
            "AND length(trim(purpose_detail)) > 0)",
            name="ck_cases_other_detail",
        ),
        sa.CheckConstraint(
            "status IN ('open','under_review','resolved','archived')",
            name="ck_cases_status",
        ),
        sa.CheckConstraint(
            "sensitivity IN ('standard','sensitive','conflict_related')",
            name="ck_cases_sensitivity",
        ),
        sa.CheckConstraint("authorization_attested = true", name="ck_cases_authorized"),
        sa.CheckConstraint("version > 0", name="ck_cases_version_positive"),
    )
    op.create_index("ix_cases_workspace_id", "cases", ["workspace_id"])
    op.create_index("ix_cases_workspace_updated", "cases", ["workspace_id", "updated_at", "id"])

    op.create_table(
        "case_media",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "case_id",
            sa.String(length=36),
            sa.ForeignKey("cases.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("analysis_id", sa.String(length=36), nullable=True),
        sa.Column("media_type", sa.String(length=16), nullable=False),
        sa.Column("source_type", sa.String(length=24), nullable=False),
        sa.Column("original_filename_display", sa.String(length=160), nullable=False),
        sa.Column("mime_type", sa.String(length=80), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("archive_url", sa.Text(), nullable=True),
        sa.Column("source_description", sa.Text(), nullable=True),
        sa.Column("authorization_attested", sa.Boolean(), nullable=False),
        sa.Column("storage_state", sa.String(length=32), nullable=False),
        sa.Column("analysis_status_at_link", sa.String(length=24), nullable=True),
        sa.Column("analysis_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("analysis_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("materialized_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("analysis_id", name="uq_case_media_analysis"),
        sa.CheckConstraint("media_type = 'image'", name="ck_case_media_image_only"),
        sa.CheckConstraint(
            "source_type IN ('upload','source_url','external_archive','other')",
            name="ck_case_media_source_type",
        ),
        sa.CheckConstraint(
            "mime_type IN ('image/jpeg','image/png','image/webp')",
            name="ck_case_media_mime_type",
        ),
        sa.CheckConstraint("byte_size > 0", name="ck_case_media_byte_size_positive"),
        sa.CheckConstraint("length(sha256) = 64", name="ck_case_media_sha256_length"),
        sa.CheckConstraint("authorization_attested = true", name="ck_case_media_authorized"),
        sa.CheckConstraint(
            "storage_state IN ('ephemeral','deleted_after_analysis','unavailable',"
            "'externally_managed')",
            name="ck_case_media_storage_state",
        ),
    )
    op.create_index("ix_case_media_case_created", "case_media", ["case_id", "created_at", "id"])

    op.create_table(
        "case_evidence_records",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "case_id",
            sa.String(length=36),
            sa.ForeignKey("cases.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "media_id",
            sa.String(length=36),
            sa.ForeignKey("case_media.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("evidence_type", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=160), nullable=False),
        sa.Column("provider_family", sa.String(length=160), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("structured_payload", sa.JSON(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("immutable_source_hash", sa.String(length=64), nullable=False),
        sa.Column("source_record_key", sa.String(length=180), nullable=False),
        sa.UniqueConstraint(
            "case_id",
            "media_id",
            "analysis_id",
            "source_record_key",
            name="uq_case_evidence_source",
        ),
        sa.CheckConstraint(
            "evidence_type IN ('metadata','ocr','visual_clue','model_hypothesis',"
            "'retrieval_match','map_evidence','analyst_note')",
            name="ck_case_evidence_type",
        ),
        sa.CheckConstraint(
            "length(immutable_source_hash) = 64", name="ck_case_evidence_hash_length"
        ),
    )
    op.create_index(
        "ix_case_evidence_case_created",
        "case_evidence_records",
        ["case_id", "created_at", "id"],
    )
    op.create_index("ix_case_evidence_analysis", "case_evidence_records", ["analysis_id"])

    op.create_table(
        "case_location_hypotheses",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "case_id",
            sa.String(length=36),
            sa.ForeignKey("cases.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "media_id",
            sa.String(length=36),
            sa.ForeignKey("case_media.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("origin", sa.String(length=32), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("uncertainty_radius_m", sa.Float(), nullable=False),
        sa.Column("country_code", sa.String(length=2), nullable=True),
        sa.Column("region_name", sa.String(length=160), nullable=True),
        sa.Column("locality_name", sa.String(length=160), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("confidence_label", sa.String(length=120), nullable=True),
        sa.Column("calibration_state", sa.String(length=24), nullable=False),
        sa.Column("supporting_evidence_ids", sa.JSON(), nullable=False),
        sa.Column("model_family_groups", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "supersedes_hypothesis_id",
            sa.String(length=36),
            sa.ForeignKey("case_location_hypotheses.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("source_record_key", sa.String(length=180), nullable=False),
        sa.UniqueConstraint(
            "case_id",
            "media_id",
            "analysis_id",
            "source_record_key",
            name="uq_case_hypothesis_source",
        ),
        sa.CheckConstraint(
            "origin IN ('model','operator_correction','imported')",
            name="ck_case_hypothesis_origin",
        ),
        sa.CheckConstraint("latitude >= -90 AND latitude <= 90", name="ck_case_hypothesis_lat"),
        sa.CheckConstraint("longitude >= -180 AND longitude <= 180", name="ck_case_hypothesis_lon"),
        sa.CheckConstraint(
            "uncertainty_radius_m > 0", name="ck_case_hypothesis_uncertainty_positive"
        ),
        sa.CheckConstraint("rank IS NULL OR rank > 0", name="ck_case_hypothesis_rank_positive"),
        sa.CheckConstraint(
            "calibration_state IN ('calibrated','uncalibrated','not_applicable')",
            name="ck_case_hypothesis_calibration",
        ),
    )
    op.create_index(
        "ix_case_hypothesis_case_created",
        "case_location_hypotheses",
        ["case_id", "created_at", "id"],
    )
    op.create_index("ix_case_hypothesis_analysis", "case_location_hypotheses", ["analysis_id"])

    op.create_table(
        "case_adjudications",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "case_id",
            sa.String(length=36),
            sa.ForeignKey("cases.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "hypothesis_id",
            sa.String(length=36),
            sa.ForeignKey("case_location_hypotheses.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("actor_id", sa.String(length=160), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "supersedes_adjudication_id",
            sa.String(length=36),
            sa.ForeignKey("case_adjudications.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.CheckConstraint(
            "decision IN ('accepted','rejected','needs_more_evidence','withdrawn')",
            name="ck_case_adjudication_decision",
        ),
    )
    op.create_index(
        "ix_case_adjudication_case_created",
        "case_adjudications",
        ["case_id", "created_at", "id"],
    )
    op.create_index(
        "ix_case_adjudication_hypothesis",
        "case_adjudications",
        ["hypothesis_id", "created_at", "id"],
    )

    op.create_table(
        "case_audit_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "case_id",
            sa.String(length=36),
            sa.ForeignKey("cases.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=120), nullable=False),
        sa.Column("actor_id", sa.String(length=160), nullable=False),
        sa.Column("actor_type", sa.String(length=24), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("previous_event_hash", sa.String(length=64), nullable=False),
        sa.Column("event_hash", sa.String(length=64), nullable=False),
        sa.UniqueConstraint("case_id", "sequence_number", name="uq_case_audit_sequence"),
        sa.UniqueConstraint("event_hash", name="uq_case_audit_event_hash"),
        sa.CheckConstraint("sequence_number > 0", name="ck_case_audit_sequence_positive"),
        sa.CheckConstraint(
            "actor_type IN ('operator','system','model')", name="ck_case_audit_actor_type"
        ),
        sa.CheckConstraint(
            "length(previous_event_hash) = 64", name="ck_case_audit_previous_hash_length"
        ),
        sa.CheckConstraint("length(event_hash) = 64", name="ck_case_audit_hash_length"),
    )
    op.create_index(
        "ix_case_audit_case_sequence",
        "case_audit_events",
        ["case_id", "sequence_number"],
    )
    _install_append_only_guards()


def downgrade() -> None:
    _drop_append_only_guards()
    op.drop_table("case_audit_events")
    op.drop_table("case_adjudications")
    op.drop_table("case_location_hypotheses")
    op.drop_table("case_evidence_records")
    op.drop_table("case_media")
    op.drop_table("cases")
