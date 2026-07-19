"""Persist bounded Phase 6C pipeline metadata.

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("analyses")}
    if "pipeline_version" not in existing:
        op.add_column(
            "analyses",
            sa.Column(
                "pipeline_version",
                sa.String(length=64),
                nullable=False,
                server_default="legacy-v1",
            ),
        )
    if "phase6c_metadata_json" not in existing:
        op.add_column("analyses", sa.Column("phase6c_metadata_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("analyses")}
    if "phase6c_metadata_json" in existing:
        op.drop_column("analyses", "phase6c_metadata_json")
    if "pipeline_version" in existing:
        op.drop_column("analyses", "pipeline_version")
