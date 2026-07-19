"""Add licensed reference provenance without blobs or filesystem paths.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_COLUMNS: tuple[sa.Column[object], ...] = (
    sa.Column("asset_key", sa.String(length=128), nullable=True),
    sa.Column("source_record_id", sa.String(length=160), nullable=True),
    sa.Column("source_url", sa.String(length=500), nullable=True),
    sa.Column("license_url", sa.String(length=500), nullable=True),
    sa.Column(
        "attribution",
        sa.String(length=500),
        nullable=False,
        server_default="attribution unavailable",
    ),
    sa.Column("display_allowed", sa.Boolean(), nullable=False, server_default=sa.false()),
    sa.Column("capture_family_id", sa.String(length=160), nullable=True),
    sa.Column(
        "coordinate_kind",
        sa.String(length=32),
        nullable=False,
        server_default="operator_provided",
    ),
    sa.Column("coordinate_uncertainty_m", sa.Float(), nullable=True),
    sa.Column("captured_at", sa.String(length=80), nullable=True),
    sa.Column("heading_degrees", sa.Float(), nullable=True),
    sa.Column("perceptual_hash", sa.String(length=16), nullable=True),
    sa.Column("terms_version", sa.String(length=80), nullable=True),
    sa.Column("continent", sa.String(length=40), nullable=True),
    sa.Column("geographic_cell", sa.String(length=100), nullable=True),
)


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {column["name"] for column in inspector.get_columns("retrieval_images")}
    for column in _COLUMNS:
        if column.name not in existing:
            op.add_column("retrieval_images", column)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {column["name"] for column in inspector.get_columns("retrieval_images")}
    for column in reversed(_COLUMNS):
        if column.name in existing:
            op.drop_column("retrieval_images", column.name)
