from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    Index,
    String,
    Table,
    UniqueConstraint,
    delete,
    func,
)
from sqlalchemy import select as sql_select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from atlaslens_api.database import Base
from atlaslens_api.retrieval.errors import MetadataConflictError
from atlaslens_api.retrieval.models import ImageMetadata


class RetrievalImageRow(Base):
    __tablename__ = "retrieval_images"

    index_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    image_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    country: Mapped[str | None] = mapped_column(String(120), nullable=True)
    region: Mapped[str | None] = mapped_column(String(160), nullable=True)
    city: Mapped[str | None] = mapped_column(String(160), nullable=True)
    source: Mapped[str] = mapped_column(String(500), nullable=False)
    license: Mapped[str] = mapped_column(String(500), nullable=False)
    capture_type: Mapped[str] = mapped_column(String(40), nullable=False)
    hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_provider: Mapped[str] = mapped_column(String(80), nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(80), nullable=False)
    asset_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_record_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    license_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    attribution: Mapped[str] = mapped_column(
        String(500), nullable=False, default="attribution unavailable"
    )
    display_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    capture_family_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    coordinate_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default="operator_provided"
    )
    coordinate_uncertainty_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[str | None] = mapped_column(String(80), nullable=True)
    heading_degrees: Mapped[float | None] = mapped_column(Float, nullable=True)
    perceptual_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)
    terms_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    continent: Mapped[str | None] = mapped_column(String(40), nullable=True)
    geographic_cell: Mapped[str | None] = mapped_column(String(100), nullable=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False)

    __table_args__ = (
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="ck_retrieval_latitude"),
        CheckConstraint(
            "longitude >= -180 AND longitude <= 180", name="ck_retrieval_longitude"
        ),
        CheckConstraint("state IN ('pending', 'active')", name="ck_retrieval_state"),
        UniqueConstraint(
            "hash",
            "embedding_provider",
            "embedding_version",
            name="uq_retrieval_content_embedding",
        ),
        Index("ix_retrieval_images_state", "state"),
    )


def _to_metadata(row: RetrievalImageRow) -> ImageMetadata:
    return ImageMetadata(
        image_id=row.image_id,
        index_id=row.index_id,
        latitude=row.latitude,
        longitude=row.longitude,
        country=row.country,
        region=row.region,
        city=row.city,
        source=row.source,
        license=row.license,
        capture_type=row.capture_type,
        hash=row.hash,
        embedding_provider=row.embedding_provider,
        embedding_version=row.embedding_version,
        asset_key=row.asset_key,
        source_record_id=row.source_record_id,
        source_url=row.source_url,
        license_url=row.license_url,
        attribution=row.attribution,
        display_allowed=row.display_allowed,
        capture_family_id=row.capture_family_id,
        coordinate_kind=cast(
            Literal[
                "operator_provided",
                "camera_raw",
                "map_matched",
                "object",
                "manual",
                "unknown",
            ],
            row.coordinate_kind,
        ),
        coordinate_uncertainty_m=row.coordinate_uncertainty_m,
        captured_at=row.captured_at,
        heading_degrees=row.heading_degrees,
        perceptual_hash=row.perceptual_hash,
        terms_version=row.terms_version,
        continent=row.continent,
        geographic_cell=row.geographic_cell,
    )


def _same_metadata(left: ImageMetadata, right: ImageMetadata) -> bool:
    return left.model_dump(exclude={"index_id"}) == right.model_dump(exclude={"index_id"})


class SQLAlchemyImageMetadataRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._sessions: sessionmaker[Session] = sessionmaker(bind=engine, expire_on_commit=False)

    def initialize(self) -> None:
        database = self._engine.url.database
        if (
            self._engine.url.get_backend_name() == "sqlite"
            and database
            and database != ":memory:"
        ):
            Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        cast(Table, RetrievalImageRow.__table__).create(self._engine, checkfirst=True)

    def close(self) -> None:
        self._engine.dispose()

    def next_index_ids(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("count must not be negative")
        with self._sessions() as session:
            maximum = session.scalar(sql_select(func.max(RetrievalImageRow.index_id)))
        start = 1 if maximum is None else int(maximum) + 1
        if start + count - 1 > 9_223_372_036_854_775_807:
            raise OverflowError("retrieval index id space is exhausted")
        return list(range(start, start + count))

    def stage(self, metadata: Sequence[ImageMetadata]) -> tuple[list[ImageMetadata], int]:
        staged: list[ImageMetadata] = []
        duplicates = 0
        seen: dict[tuple[str, str, str], ImageMetadata] = {}
        for item in metadata:
            key = (item.hash, item.embedding_provider, item.embedding_version)
            previous = seen.get(key)
            if previous is not None:
                if not _same_metadata(previous, item):
                    raise MetadataConflictError("duplicate content has conflicting metadata")
                duplicates += 1
                continue
            seen[key] = item

        with self._sessions.begin() as session:
            for item in seen.values():
                existing = session.scalar(
                    sql_select(RetrievalImageRow).where(
                        RetrievalImageRow.hash == item.hash,
                        RetrievalImageRow.embedding_provider == item.embedding_provider,
                        RetrievalImageRow.embedding_version == item.embedding_version,
                    )
                )
                if existing is not None:
                    if existing.state != "active" or not _same_metadata(
                        _to_metadata(existing), item
                    ):
                        raise MetadataConflictError("existing content has conflicting metadata")
                    duplicates += 1
                    continue
                session.add(
                    RetrievalImageRow(
                        image_id=str(item.image_id),
                        index_id=item.index_id,
                        latitude=item.latitude,
                        longitude=item.longitude,
                        country=item.country,
                        region=item.region,
                        city=item.city,
                        source=item.source,
                        license=item.license,
                        capture_type=item.capture_type,
                        hash=item.hash,
                        embedding_provider=item.embedding_provider,
                        embedding_version=item.embedding_version,
                        asset_key=item.asset_key,
                        source_record_id=item.source_record_id,
                        source_url=item.source_url,
                        license_url=item.license_url,
                        attribution=item.attribution,
                        display_allowed=item.display_allowed,
                        capture_family_id=item.capture_family_id,
                        coordinate_kind=item.coordinate_kind,
                        coordinate_uncertainty_m=item.coordinate_uncertainty_m,
                        captured_at=item.captured_at,
                        heading_degrees=item.heading_degrees,
                        perceptual_hash=item.perceptual_hash,
                        terms_version=item.terms_version,
                        continent=item.continent,
                        geographic_cell=item.geographic_cell,
                        state="pending",
                    )
                )
                staged.append(item)
        return staged, duplicates

    def activate(self, index_ids: Sequence[int]) -> None:
        if not index_ids:
            return
        with self._sessions.begin() as session:
            rows = session.scalars(
                sql_select(RetrievalImageRow).where(RetrievalImageRow.index_id.in_(index_ids))
            ).all()
            if len(rows) != len(set(index_ids)) or any(row.state != "pending" for row in rows):
                raise MetadataConflictError("pending metadata set changed before activation")
            for row in rows:
                row.state = "active"

    def abort(self, index_ids: Sequence[int]) -> None:
        if not index_ids:
            return
        with self._sessions.begin() as session:
            session.execute(
                delete(RetrievalImageRow).where(
                    RetrievalImageRow.index_id.in_(index_ids), RetrievalImageRow.state == "pending"
                )
            )

    def discard(self, index_ids: Sequence[int]) -> None:
        if not index_ids:
            return
        with self._sessions.begin() as session:
            session.execute(
                delete(RetrievalImageRow).where(RetrievalImageRow.index_id.in_(index_ids))
            )

    def get_by_index_ids(self, index_ids: Sequence[int]) -> dict[int, ImageMetadata]:
        if not index_ids:
            return {}
        with self._sessions() as session:
            rows = session.scalars(
                sql_select(RetrievalImageRow).where(
                    RetrievalImageRow.index_id.in_(index_ids),
                    RetrievalImageRow.state == "active",
                )
            ).all()
        return {row.index_id: _to_metadata(row) for row in rows}

    def active_index_ids(self) -> set[int]:
        with self._sessions() as session:
            values = session.scalars(
                sql_select(RetrievalImageRow.index_id).where(RetrievalImageRow.state == "active")
            ).all()
        return {int(value) for value in values}

    def active_embedding_specs(self) -> set[tuple[str, str]]:
        with self._sessions() as session:
            values = session.execute(
                sql_select(
                    RetrievalImageRow.embedding_provider, RetrievalImageRow.embedding_version
                )
                .where(RetrievalImageRow.state == "active")
                .distinct()
            ).all()
        return {(str(provider), str(version)) for provider, version in values}

    def pending_index_ids(self) -> list[int]:
        with self._sessions() as session:
            values = session.scalars(
                sql_select(RetrievalImageRow.index_id).where(RetrievalImageRow.state == "pending")
            ).all()
        return [int(value) for value in values]
