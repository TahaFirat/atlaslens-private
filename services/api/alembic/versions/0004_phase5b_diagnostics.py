"""Persist additive Phase 5B provider/index diagnostics.

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {column["name"] for column in inspector.get_columns("analyses")}
    if "phase5b_diagnostics_json" not in existing:
        op.add_column(
            "analyses",
            sa.Column("phase5b_diagnostics_json", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {column["name"] for column in inspector.get_columns("analyses")}
    if "phase5b_diagnostics_json" in existing:
        op.drop_column("analyses", "phase5b_diagnostics_json")
