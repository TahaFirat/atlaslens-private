from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from atlaslens_api.retrieval.errors import (
    IndexIntegrityError,
    IndexNotFoundError,
    ProviderUnavailableError,
)
from atlaslens_api.retrieval.manifest import hash_image, validate_manifest
from atlaslens_api.retrieval.metadata import SQLAlchemyImageMetadataRepository
from atlaslens_api.retrieval.models import (
    Embedding,
    ImageMetadata,
    ImportSummary,
    RetrievalHit,
)
from atlaslens_api.retrieval.protocols import EmbeddingProvider, ImageIndex


def image_id_for_hash(content_hash: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"atlaslens:image:sha256:{content_hash}"))


class ManifestDatasetImporter:
    def __init__(
        self,
        provider: EmbeddingProvider,
        index: ImageIndex,
        metadata: SQLAlchemyImageMetadataRepository,
    ) -> None:
        self._provider = provider
        self._index = index
        self._metadata = metadata

    def create(self, manifest_path: Path, input_root: Path) -> ImportSummary:
        records = validate_manifest(manifest_path, input_root)
        if not records:
            return ImportSummary(validated=0, added=0, duplicates=0, index_size=self._index.size)
        if not self._provider.available:
            raise ProviderUnavailableError(
                self._provider.unavailable_reason or "embedding provider is unavailable"
            )
        if self._provider.spec != self._index.spec:
            raise ValueError("embedding provider specification does not match the index")
        self._metadata.initialize()
        verify_metadata_consistency(self._index, self._metadata, allow_empty_unpersisted=True)
        allocated = self._metadata.next_index_ids(len(records))
        proposed = [
            ImageMetadata(
                image_id=image_id_for_hash(record.content_hash),
                index_id=index_id,
                latitude=record.latitude,
                longitude=record.longitude,
                country=record.country,
                region=record.region,
                city=record.city,
                source=record.source,
                license=record.license,
                hash=record.content_hash,
                embedding_provider=self._provider.spec.provider,
                embedding_version=self._provider.spec.version,
                asset_key=record.asset_key,
                source_record_id=record.source_record_id,
                source_url=record.source_url,
                license_url=record.license_url,
                attribution=record.attribution,
                display_allowed=record.display_allowed,
                capture_family_id=record.capture_family_id,
                coordinate_kind=record.coordinate_kind,
                coordinate_uncertainty_m=record.coordinate_uncertainty_m,
                captured_at=record.captured_at,
                heading_degrees=record.heading_degrees,
                perceptual_hash=record.perceptual_hash,
                terms_version=record.terms_version,
                capture_type=record.capture_type,
                continent=record.continent,
                geographic_cell=record.geographic_cell,
            )
            for record, index_id in zip(records, allocated, strict=True)
        ]
        staged, duplicates = self._metadata.stage(proposed)
        staged_ids = [item.index_id for item in staged]
        record_by_hash = {record.content_hash: record for record in records}
        appended = False
        try:
            embedding_paths: list[Path] = []
            for item in staged:
                record = record_by_hash[item.hash]
                if hash_image(record.image_path) != item.hash:
                    raise IndexIntegrityError("an input image changed during import")
                embedding_paths.append(record.image_path)
            batch_embed = getattr(self._provider, "embed_many", None)
            embeddings = (
                batch_embed(embedding_paths)
                if callable(batch_embed)
                else [self._provider.embed(path) for path in embedding_paths]
            )
            if len(embeddings) != len(staged):
                raise IndexIntegrityError("embedding provider returned an invalid batch count")
            if staged:
                self._index.append(staged_ids, embeddings)
                appended = True
                self._index.persist()
                expected_ids = self._metadata.active_index_ids().union(staged_ids)
                if self._index.ids() != expected_ids:
                    raise IndexIntegrityError("FAISS and staged metadata ids do not match")
                self._metadata.activate(staged_ids)
            if self._index.ids() != self._metadata.active_index_ids():
                raise IndexIntegrityError("FAISS and metadata ids do not match")
        except Exception as original_error:
            rollback_error: Exception | None = None
            if appended:
                try:
                    self._index.remove(staged_ids)
                    self._index.persist()
                except Exception as exc:
                    rollback_error = exc
            self._metadata.discard(staged_ids)
            if rollback_error is not None:
                raise IndexIntegrityError("retrieval import rollback failed") from original_error
            raise
        return ImportSummary(
            validated=len(records),
            added=len(staged),
            duplicates=duplicates,
            index_size=self._index.size,
        )


class FaissNearestNeighborEngine:
    def __init__(
        self, index: ImageIndex, metadata: SQLAlchemyImageMetadataRepository
    ) -> None:
        self._index = index
        self._metadata = metadata

    def retrieve(self, embedding: Embedding, top_k: int) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        search_with_ties = getattr(self._index, "search_with_ties", None)
        matches = (
            search_with_ties(embedding, top_k)
            if callable(search_with_ties)
            else self._index.search(embedding, top_k)
        )
        rows = self._metadata.get_by_index_ids([match.index_id for match in matches])
        if len(rows) != len(matches):
            raise IndexIntegrityError("retrieval metadata is incomplete")
        hits = [
            RetrievalHit(
                distance=match.distance,
                provider=embedding.spec,
                metadata=rows[match.index_id],
            )
            for match in matches
        ]
        return sorted(hits, key=lambda hit: (hit.distance, str(hit.metadata.image_id)))[:top_k]


def verify_metadata_consistency(
    index: ImageIndex,
    metadata: SQLAlchemyImageMetadataRepository,
    *,
    allow_empty_unpersisted: bool = False,
) -> None:
    try:
        index.verify()
    except IndexNotFoundError:
        if not (allow_empty_unpersisted and index.size == 0 and not metadata.active_index_ids()):
            raise
    if index.ids() != metadata.active_index_ids():
        raise IndexIntegrityError("FAISS and metadata ids do not match")
    specs = metadata.active_embedding_specs()
    expected = {(index.spec.provider, index.spec.version)} if index.size else set()
    if specs != expected:
        raise IndexIntegrityError("embedding provider metadata does not match the index")


class RetrievalQueryService:
    """Image-facing query facade for a verified local FAISS/metadata pair."""

    def __init__(
        self,
        index: ImageIndex,
        metadata: SQLAlchemyImageMetadataRepository,
        provider: EmbeddingProvider,
    ) -> None:
        if provider.spec != index.spec:
            raise IndexIntegrityError("query provider specification does not match the index")
        verify_metadata_consistency(index, metadata)
        self._index = index
        self._metadata = metadata
        self._provider = provider
        self._engine = FaissNearestNeighborEngine(index, metadata)

    def retrieve_image(self, image_path: Path, top_k: int) -> list[RetrievalHit]:
        if not self._provider.available:
            raise ProviderUnavailableError(
                self._provider.unavailable_reason or "embedding provider is unavailable"
            )
        return self._engine.retrieve(self._provider.embed(image_path), top_k)

    def retrieve_image_bytes(self, payload: bytes, top_k: int) -> list[RetrievalHit]:
        """Query from immutable bytes when the concrete provider supports it."""

        if not self._provider.available:
            raise ProviderUnavailableError(
                self._provider.unavailable_reason or "embedding provider is unavailable"
            )
        embed_bytes = getattr(self._provider, "embed_bytes", None)
        if not callable(embed_bytes):
            raise ProviderUnavailableError("embedding provider does not accept immutable payloads")
        embedding = embed_bytes(payload)
        if not isinstance(embedding, Embedding):
            raise ProviderUnavailableError("embedding provider returned an invalid payload")
        return self._engine.retrieve(embedding, top_k)

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = self._index.diagnostics()
        result: dict[str, Any] = {
            "status": diagnostics.status,
            "index_size": diagnostics.index_size,
            "embedding_provider": diagnostics.embedding_provider,
            "embedding_version": diagnostics.embedding_version,
            "dimension": diagnostics.dimension,
            "storage_size_bytes": diagnostics.storage_size_bytes,
            "provider_available": self._provider.available,
        }
        directory = getattr(self._index, "directory", None)
        if isinstance(directory, Path):
            try:
                manifest = json.loads((directory / "index.json").read_text(encoding="utf-8"))
                checksum = manifest.get("index_sha256")
                if isinstance(checksum, str) and len(checksum) == 64:
                    result["checksum"] = checksum
                    result["index_id"] = "faiss-" + hashlib.sha256(
                        (
                            f"{diagnostics.embedding_provider}\x1f"
                            f"{diagnostics.embedding_version}\x1f{checksum}"
                        ).encode()
                    ).hexdigest()[:24]
            except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
                pass
        return result

    def close(self) -> None:
        self._metadata.close()


def open_retrieval_query_service(
    index_dir: Path, provider: EmbeddingProvider
) -> RetrievalQueryService:
    from atlaslens_api.database import create_database_engine
    from atlaslens_api.retrieval.faiss_index import FaissImageIndex

    directory = index_dir.expanduser().resolve()
    index = FaissImageIndex.open(directory)
    database = directory / "metadata.sqlite3"
    if not database.is_file():
        raise IndexIntegrityError("retrieval metadata database is missing")
    metadata = SQLAlchemyImageMetadataRepository(
        create_database_engine(f"sqlite:///{database.as_posix()}")
    )
    try:
        return RetrievalQueryService(index, metadata, provider)
    except Exception:
        metadata.close()
        raise
