"""Persist optional Phase 6A scene-analysis summaries.

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("analyses")}
    if "scene_analysis_json" not in existing:
        op.add_column(
            "analyses",
            sa.Column("scene_analysis_json", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("analyses")}
    if "scene_analysis_json" in existing:
        op.drop_column("analyses", "scene_analysis_json")
