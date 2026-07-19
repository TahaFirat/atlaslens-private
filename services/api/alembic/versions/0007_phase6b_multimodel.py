"""Persist optional bounded Phase 6B ensemble summaries.

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | None = None
depends_on: str | None = None

_COLUMNS = (
    "model_predictions_json",
    "phase6b_fusion_json",
    "phase6b_ocr_json",
    "cloud_assist_json",
)


def upgrade() -> None:
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("analyses")}
    for name in _COLUMNS:
        if name not in existing:
            op.add_column("analyses", sa.Column(name, sa.JSON(), nullable=True))
    if "cloud_assist_cache_key" not in existing:
        op.add_column(
            "analyses",
            sa.Column("cloud_assist_cache_key", sa.String(length=67), nullable=True),
        )


def downgrade() -> None:
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("analyses")}
    if "cloud_assist_cache_key" in existing:
        op.drop_column("analyses", "cloud_assist_cache_key")
    for name in reversed(_COLUMNS):
        if name in existing:
            op.drop_column("analyses", name)
