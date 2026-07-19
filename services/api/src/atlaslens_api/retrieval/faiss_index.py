from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Never
from uuid import uuid4

import faiss
import numpy as np

from atlaslens_api.retrieval.errors import (
    IndexIntegrityError,
    IndexNotFoundError,
    ProviderUnavailableError,
)
from atlaslens_api.retrieval.models import (
    Embedding,
    EmbeddingSpec,
    IndexMatch,
    ProviderDiagnostics,
)

_INDEX_NAME = "vectors.faiss"
_MANIFEST_NAME = "index.json"
_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FaissImageIndex:
    """Exact Phase 3 index; adapters can replace it without changing the protocols."""

    def __init__(self, index_dir: Path, spec: EmbeddingSpec, *, load: bool = False) -> None:
        self._directory = index_dir.expanduser().resolve()
        self._spec = spec
        self._index: Any = faiss.IndexIDMap2(faiss.IndexFlatIP(spec.dimension))
        self._build_time_ms = 0
        if load:
            self.reload()

    @classmethod
    def open(cls, index_dir: Path) -> FaissImageIndex:
        directory = index_dir.expanduser().resolve()
        manifest = cls._read_manifest(directory)
        try:
            spec = EmbeddingSpec.model_validate(manifest["embedding_spec"])
        except (KeyError, ValueError, TypeError) as exc:
            raise IndexIntegrityError("invalid embedding specification") from exc
        return cls(directory, spec, load=True)

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @property
    def directory(self) -> Path:
        return self._directory

    def ids(self) -> set[int]:
        values = faiss.vector_to_array(self._index.id_map)
        return {int(value) for value in values.tolist()}

    def append(self, ids: Sequence[int], embeddings: Sequence[Embedding]) -> None:
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings must have equal lengths")
        if not ids:
            return
        integer_ids = [int(value) for value in ids]
        if any(value < 0 or value > np.iinfo(np.int64).max for value in integer_ids):
            raise ValueError("FAISS ids must be non-negative int64 values")
        if len(set(integer_ids)) != len(integer_ids) or self.ids().intersection(integer_ids):
            raise ValueError("FAISS ids must be unique")
        for embedding in embeddings:
            if embedding.spec != self._spec:
                raise ValueError("embedding provider specification does not match the index")
        vectors = np.ascontiguousarray(
            np.stack([embedding.vector for embedding in embeddings]), dtype=np.float32
        )
        labels = np.ascontiguousarray(integer_ids, dtype=np.int64)
        self._index.add_with_ids(vectors, labels)

    def search(self, embedding: Embedding, top_k: int) -> list[IndexMatch]:
        return self.search_with_ties(embedding, top_k)[:top_k]

    def search_with_ties(self, embedding: Embedding, top_k: int) -> list[IndexMatch]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if embedding.spec != self._spec:
            raise ValueError("embedding provider specification does not match the index")
        if self.size == 0:
            return []
        requested = min(top_k, self.size)
        query = np.ascontiguousarray(embedding.vector.reshape(1, -1), dtype=np.float32)
        scores, labels = self._index.search(query, requested)
        valid = [
            (int(label), float(score))
            for score, label in zip(scores[0], labels[0], strict=True)
            if label >= 0
        ]
        if valid and requested < self.size:
            threshold = np.nextafter(np.float32(valid[-1][1]), np.float32(-math.inf))
            try:
                limits, range_scores, range_labels = self._index.range_search(
                    query, float(threshold)
                )
                start, end = int(limits[0]), int(limits[1])
                valid = [
                    (int(range_labels[offset]), float(range_scores[offset]))
                    for offset in range(start, end)
                    if range_labels[offset] >= 0
                ]
            except RuntimeError:
                all_scores, all_labels = self._index.search(query, self.size)
                valid = [
                    (int(label), float(score))
                    for score, label in zip(all_scores[0], all_labels[0], strict=True)
                    if label >= 0 and score >= threshold
                ]
        matches = [
            IndexMatch(index_id=label, distance=min(2.0, max(0.0, 1.0 - score)))
            for label, score in valid
        ]
        return sorted(matches, key=lambda item: (item.distance, item.index_id))

    def remove(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        labels = np.ascontiguousarray([int(value) for value in ids], dtype=np.int64)
        removed = int(self._index.remove_ids(labels))
        if removed != len(set(ids)):
            raise IndexIntegrityError("FAISS rollback id set mismatch")

    def persist(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        temporary_index = self._directory / f".{_INDEX_NAME}.{token}.tmp"
        temporary_manifest = self._directory / f".{_MANIFEST_NAME}.{token}.tmp"
        started = time.perf_counter()
        try:
            faiss.write_index(self._index, str(temporary_index))
            with temporary_index.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            self._build_time_ms = max(0, round((time.perf_counter() - started) * 1000))
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "backend": "faiss-idmap2-flatip",
                "embedding_spec": self._spec.model_dump(mode="json"),
                "count": self.size,
                "ids": sorted(self.ids()),
                "index_sha256": _sha256(temporary_index),
                "build_time_ms": self._build_time_ms,
            }
            temporary_manifest.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with temporary_manifest.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_index, self._directory / _INDEX_NAME)
            os.replace(temporary_manifest, self._directory / _MANIFEST_NAME)
        finally:
            temporary_index.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)

    def reload(self) -> None:
        manifest = self._read_manifest(self._directory)
        index_path = self._directory / _INDEX_NAME
        if not index_path.is_file():
            raise IndexNotFoundError("FAISS index artifact is missing")
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            raise IndexIntegrityError("unsupported index schema")
        if manifest.get("index_sha256") != _sha256(index_path):
            raise IndexIntegrityError("FAISS index checksum mismatch")
        try:
            persisted_spec = EmbeddingSpec.model_validate(manifest["embedding_spec"])
            loaded = faiss.read_index(str(index_path))
        except (KeyError, ValueError, RuntimeError, TypeError) as exc:
            raise IndexIntegrityError("FAISS index could not be loaded") from exc
        if persisted_spec != self._spec:
            raise IndexIntegrityError("embedding specification mismatch")
        self._index = loaded
        self._build_time_ms = int(manifest.get("build_time_ms", 0))
        self.verify()

    def verify(self) -> ProviderDiagnostics:
        manifest = self._read_manifest(self._directory)
        index_path = self._directory / _INDEX_NAME
        if not index_path.is_file() or manifest.get("index_sha256") != _sha256(index_path):
            raise IndexIntegrityError("FAISS index checksum mismatch")
        if int(manifest.get("count", -1)) != self.size:
            raise IndexIntegrityError("FAISS index count mismatch")
        manifest_ids = manifest.get("ids")
        if (
            not isinstance(manifest_ids, list)
            or {int(value) for value in manifest_ids} != self.ids()
        ):
            raise IndexIntegrityError("FAISS index id mismatch")
        if int(self._index.d) != self._spec.dimension:
            raise IndexIntegrityError("FAISS index dimension mismatch")
        return self._diagnostics("empty" if self.size == 0 else "ready")

    def diagnostics(self) -> ProviderDiagnostics:
        try:
            return self.verify()
        except (IndexIntegrityError, IndexNotFoundError, OSError, ValueError):
            return self._diagnostics("invalid", detail="index verification failed")

    def _diagnostics(
        self,
        status: Literal["ready", "empty", "unavailable", "invalid"],
        detail: str | None = None,
    ) -> ProviderDiagnostics:
        storage_size = sum(
            path.stat().st_size
            for path in (self._directory / _INDEX_NAME, self._directory / _MANIFEST_NAME)
            if path.is_file()
        )
        return ProviderDiagnostics(
            status=status,
            index_size=self.size,
            embedding_provider=self._spec.provider,
            embedding_version=self._spec.version,
            dimension=self._spec.dimension,
            build_time_ms=self._build_time_ms,
            storage_size_bytes=storage_size,
            detail=detail,
        )

    @staticmethod
    def _read_manifest(directory: Path) -> dict[str, Any]:
        path = directory / _MANIFEST_NAME
        if not path.is_file():
            raise IndexNotFoundError("index manifest is missing")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise IndexIntegrityError("index manifest is invalid") from exc
        if not isinstance(payload, dict):
            raise IndexIntegrityError("index manifest is invalid")
        return payload


class UnavailableRemoteImageIndex:
    def __init__(self, backend: str, spec: EmbeddingSpec) -> None:
        self._backend = backend
        self._spec = spec

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    @property
    def size(self) -> int:
        return 0

    def _raise(self) -> Never:
        raise ProviderUnavailableError(f"{self._backend} index adapter is not available")

    def append(self, ids: Sequence[int], embeddings: Sequence[Embedding]) -> None:
        del ids, embeddings
        self._raise()

    def persist(self) -> None:
        self._raise()

    def reload(self) -> None:
        self._raise()

    def remove(self, ids: Sequence[int]) -> None:
        del ids
        self._raise()

    def search(self, embedding: Embedding, top_k: int) -> list[IndexMatch]:
        del embedding, top_k
        self._raise()

    def ids(self) -> set[int]:
        self._raise()

    def verify(self) -> ProviderDiagnostics:
        self._raise()

    def diagnostics(self) -> ProviderDiagnostics:
        return ProviderDiagnostics(
            status="unavailable",
            index_size=0,
            embedding_provider=self._spec.provider,
            embedding_version=self._spec.version,
            dimension=self._spec.dimension,
            build_time_ms=0,
            storage_size_bytes=0,
            detail=f"{self._backend} index adapter is not available",
        )


class QdrantImageIndex(UnavailableRemoteImageIndex):
    def __init__(self, spec: EmbeddingSpec) -> None:
        super().__init__("Qdrant", spec)


class MilvusImageIndex(UnavailableRemoteImageIndex):
    def __init__(self, spec: EmbeddingSpec) -> None:
        super().__init__("Milvus", spec)
