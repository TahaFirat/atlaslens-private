"""Add licensed retrieval image metadata without blobs or paths.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_EXPECTED_COLUMNS = {
    "image_id",
    "index_id",
    "latitude",
    "longitude",
    "country",
    "region",
    "city",
    "source",
    "license",
    "capture_type",
    "hash",
    "embedding_provider",
    "embedding_version",
    "state",
}


def _adopt_legacy_table() -> bool:
    inspector = sa.inspect(op.get_bind())
    if "retrieval_images" not in inspector.get_table_names():
        return False
    actual = {column["name"] for column in inspector.get_columns("retrieval_images")}
    missing = _EXPECTED_COLUMNS - actual
    if missing:
        raise RuntimeError(
            "existing retrieval_images table is incompatible with migration 0002: "
            + ", ".join(sorted(missing))
        )
    indexes = {index["name"] for index in inspector.get_indexes("retrieval_images")}
    required_indexes = {
        "ix_retrieval_images_image_id": ["image_id"],
        "ix_retrieval_images_state": ["state"],
    }
    for name, columns in required_indexes.items():
        if name not in indexes:
            op.create_index(name, "retrieval_images", columns, unique=False)
    return True


def upgrade() -> None:
    if _adopt_legacy_table():
        return
    op.create_table(
        "retrieval_images",
        sa.Column("image_id", sa.String(length=36), nullable=False),
        sa.Column("index_id", sa.BigInteger(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("country", sa.String(length=120), nullable=True),
        sa.Column("region", sa.String(length=160), nullable=True),
        sa.Column("city", sa.String(length=160), nullable=True),
        sa.Column("source", sa.String(length=500), nullable=False),
        sa.Column("license", sa.String(length=500), nullable=False),
        sa.Column("capture_type", sa.String(length=40), nullable=False),
        sa.Column("hash", sa.String(length=64), nullable=False),
        sa.Column("embedding_provider", sa.String(length=80), nullable=False),
        sa.Column("embedding_version", sa.String(length=80), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.CheckConstraint(
            "latitude >= -90 AND latitude <= 90", name="ck_retrieval_latitude"
        ),
        sa.CheckConstraint(
            "longitude >= -180 AND longitude <= 180", name="ck_retrieval_longitude"
        ),
        sa.CheckConstraint("state IN ('pending', 'active')", name="ck_retrieval_state"),
        sa.PrimaryKeyConstraint("index_id"),
        sa.UniqueConstraint(
            "hash",
            "embedding_provider",
            "embedding_version",
            name="uq_retrieval_content_embedding",
        ),
    )
    op.create_index(
        "ix_retrieval_images_image_id", "retrieval_images", ["image_id"], unique=False
    )
    op.create_index("ix_retrieval_images_state", "retrieval_images", ["state"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_retrieval_images_state", table_name="retrieval_images")
    op.drop_index("ix_retrieval_images_image_id", table_name="retrieval_images")
    op.drop_table("retrieval_images")
