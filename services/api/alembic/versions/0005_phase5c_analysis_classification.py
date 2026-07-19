"""Persist Phase 5C analysis classification and safe provider comparisons.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    connection = op.get_bind()
    existing = {column["name"] for column in sa.inspect(connection).get_columns("analyses")}
    if "result_classification" not in existing:
        op.add_column(
            "analyses",
            sa.Column(
                "result_classification",
                sa.String(length=16),
                nullable=False,
                server_default="real",
            ),
        )
        op.create_index(
            "ix_analyses_result_classification",
            "analyses",
            ["result_classification"],
            unique=False,
        )
    if "simulation_json" not in existing:
        op.add_column("analyses", sa.Column("simulation_json", sa.JSON(), nullable=True))
    if "provider_comparisons_json" not in existing:
        op.add_column(
            "analyses",
            sa.Column("provider_comparisons_json", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    connection = op.get_bind()
    existing = {column["name"] for column in sa.inspect(connection).get_columns("analyses")}
    if "provider_comparisons_json" in existing:
        op.drop_column("analyses", "provider_comparisons_json")
    if "simulation_json" in existing:
        op.drop_column("analyses", "simulation_json")
    if "result_classification" in existing:
        indexes = {index["name"] for index in sa.inspect(connection).get_indexes("analyses")}
        if "ix_analyses_result_classification" in indexes:
            op.drop_index("ix_analyses_result_classification", table_name="analyses")
        op.drop_column("analyses", "result_classification")
