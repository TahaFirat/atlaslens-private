"""Create ephemeral analyses table.

Revision ID: 0001
Revises:
Create Date: 2026-07-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_EXPECTED_COLUMNS = {
    "id",
    "status",
    "analysis_mode",
    "created_at",
    "expires_at",
    "progress_stage",
    "progress_percent",
    "progress_message_key",
    "image_json",
    "quality_json",
    "evidence_json",
    "candidates_json",
    "abstention_json",
    "warnings_json",
    "timings_json",
    "fusion_policy_version",
    "failure_json",
    "idempotency_hash",
    "storage_key",
}


def _adopt_legacy_table() -> bool:
    inspector = sa.inspect(op.get_bind())
    if "analyses" not in inspector.get_table_names():
        return False
    actual = {column["name"] for column in inspector.get_columns("analyses")}
    missing = _EXPECTED_COLUMNS - actual
    if missing:
        raise RuntimeError(
            "existing analyses table is incompatible with migration 0001: "
            + ", ".join(sorted(missing))
        )
    indexes = {index["name"] for index in inspector.get_indexes("analyses")}
    required_indexes = {
        "ix_analyses_status": ["status"],
        "ix_analyses_expires_at": ["expires_at"],
        "ix_analyses_expiry_status": ["expires_at", "status"],
    }
    for name, columns in required_indexes.items():
        if name not in indexes:
            op.create_index(name, "analyses", columns, unique=False)
    return True


def upgrade() -> None:
    # Early AtlasLens runtime builds used ORM create_all before Alembic was
    # enforced. Adopt only the exact compatible legacy shape; never stamp an
    # unknown or partial schema silently.
    if _adopt_legacy_table():
        return
    op.create_table(
        "analyses",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("analysis_mode", sa.String(length=24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("progress_stage", sa.String(length=64), nullable=False),
        sa.Column("progress_percent", sa.Integer(), nullable=False),
        sa.Column("progress_message_key", sa.String(length=128), nullable=False),
        sa.Column("image_json", sa.JSON(), nullable=True),
        sa.Column("quality_json", sa.JSON(), nullable=True),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("candidates_json", sa.JSON(), nullable=False),
        sa.Column("abstention_json", sa.JSON(), nullable=True),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
        sa.Column("timings_json", sa.JSON(), nullable=False),
        sa.Column("fusion_policy_version", sa.String(length=64), nullable=False),
        sa.Column("failure_json", sa.JSON(), nullable=True),
        sa.Column("idempotency_hash", sa.String(length=64), nullable=True),
        sa.Column("storage_key", sa.String(length=80), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_hash"),
    )
    op.create_index("ix_analyses_status", "analyses", ["status"], unique=False)
    op.create_index("ix_analyses_expires_at", "analyses", ["expires_at"], unique=False)
    op.create_index("ix_analyses_expiry_status", "analyses", ["expires_at", "status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_analyses_expiry_status", table_name="analyses")
    op.drop_index("ix_analyses_expires_at", table_name="analyses")
    op.drop_index("ix_analyses_status", table_name="analyses")
    op.drop_table("analyses")
