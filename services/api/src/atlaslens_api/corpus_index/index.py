"""Versioned exact-cosine and FAISS corpus indexes with atomic publication."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

import faiss
import numpy as np

from atlaslens_api.corpus_index.artifacts import (
    atomic_write_json,
    read_json,
    sha256_json,
    sha256_path,
)
from atlaslens_api.corpus_index.descriptors import DescriptorDataset
from atlaslens_api.corpus_index.errors import ArtifactIntegrityError, IndexCompatibilityError
from atlaslens_api.corpus_index.models import (
    AssetProvenance,
    DescriptorSpec,
    FloatMatrix,
    FloatVector,
    SearchHit,
    normalize_vector,
)

IndexBackendName = Literal["exact", "faiss"]
_EXACT_BACKEND = "exact-numpy-v1"
_FAISS_BACKEND = "faiss-flat-ip-v1"
_INDEX_SCHEMA = "atlaslens-corpus-index-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _ShardBackend(Protocol):
    def search(self, query: FloatVector, top_k: int) -> list[tuple[int, float]]: ...


@dataclass(frozen=True, slots=True)
class _LoadedShard:
    shard_id: str
    assets: tuple[AssetProvenance, ...]
    backend: _ShardBackend
    assets_relative_path: str
    vectors_relative_path: str


@dataclass(frozen=True, slots=True)
class ArtifactInventoryItem:
    relative_path: str
    sha256: str
    size_bytes: int
    role: str
    shard_id: str | None = None

    def to_json(self) -> dict[str, str | int | None]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "role": self.role,
            "shard_id": self.shard_id,
        }


@dataclass(frozen=True, slots=True)
class RevocationImpact:
    asset_id: str
    found: bool
    shard_ids: tuple[str, ...]
    artifact_paths: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "found": self.found,
            "shard_ids": list(self.shard_ids),
            "artifact_paths": list(self.artifact_paths),
        }


@dataclass(frozen=True, slots=True)
class RebuildPlan:
    plan_id: str
    requested_asset_ids: tuple[str, ...]
    found_asset_ids: tuple[str, ...]
    missing_asset_ids: tuple[str, ...]
    affected_shards: tuple[str, ...]
    retained_asset_count: int
    removed_asset_count: int
    artifact_inventory: tuple[ArtifactInventoryItem, ...]
    steps: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-revocation-rebuild-plan-v1",
            "plan_id": self.plan_id,
            "requested_asset_ids": list(self.requested_asset_ids),
            "found_asset_ids": list(self.found_asset_ids),
            "missing_asset_ids": list(self.missing_asset_ids),
            "affected_shards": list(self.affected_shards),
            "retained_asset_count": self.retained_asset_count,
            "removed_asset_count": self.removed_asset_count,
            "artifact_inventory": [item.to_json() for item in self.artifact_inventory],
            "steps": list(self.steps),
        }


class _ExactCosineShard:
    def __init__(self, vectors: FloatMatrix, assets: Sequence[AssetProvenance]) -> None:
        self._vectors = vectors
        self._asset_ids = tuple(asset.asset_id for asset in assets)

    def search(self, query: FloatVector, top_k: int) -> list[tuple[int, float]]:
        if not len(self._asset_ids):
            return []
        scores = self._vectors @ query
        ordered = sorted(
            range(len(self._asset_ids)),
            key=lambda position: (float(1.0 - scores[position]), self._asset_ids[position]),
        )
        return [
            (position, min(2.0, max(0.0, float(1.0 - scores[position]))))
            for position in ordered[:top_k]
        ]


class _FaissCosineShard:
    def __init__(self, index: Any, assets: Sequence[AssetProvenance]) -> None:
        self._index = index
        self._asset_ids = tuple(asset.asset_id for asset in assets)

    def search(self, query: FloatVector, top_k: int) -> list[tuple[int, float]]:
        if not self._asset_ids:
            return []
        # IndexFlatIP is exact. Fetching the shard permits stable asset-ID tie ordering.
        scores, positions = self._index.search(
            np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32),
            len(self._asset_ids),
        )
        matches = [
            (int(position), min(2.0, max(0.0, float(1.0 - score))))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]
        return sorted(matches, key=lambda item: (item[1], self._asset_ids[item[0]]))[:top_k]


class PublishedCorpusIndex:
    """Checksum-verified sharded index and provenance lookup."""

    def __init__(
        self,
        directory: Path,
        metadata: dict[str, Any],
        spec: DescriptorSpec,
        shards: tuple[_LoadedShard, ...],
    ) -> None:
        self.directory = directory
        self.metadata = metadata
        self.spec = spec
        self._shards = shards

    @property
    def size(self) -> int:
        return sum(len(shard.assets) for shard in self._shards)

    @property
    def backend(self) -> str:
        return str(self.metadata["backend"])

    @property
    def source_policy_hash(self) -> str:
        return str(self.metadata["source_policy_hash"])

    @property
    def locked_holdout_hash(self) -> str | None:
        value = self.metadata.get("locked_holdout_hash")
        return str(value) if value is not None else None

    @property
    def split_lock_hash(self) -> str | None:
        value = self.metadata.get("split_lock_hash")
        return str(value) if value is not None else None

    @property
    def provenance(self) -> tuple[AssetProvenance, ...]:
        return tuple(asset for shard in self._shards for asset in shard.assets)

    def search(self, query: object, *, top_k: int) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        normalized = normalize_vector(query, dimension=self.spec.dimension)
        candidates: list[tuple[float, AssetProvenance]] = []
        for shard in self._shards:
            for position, distance in shard.backend.search(normalized, top_k):
                candidates.append((distance, shard.assets[position]))
        ordered = sorted(candidates, key=lambda item: (item[0], item[1].asset_id))[:top_k]
        return [
            SearchHit(
                rank=rank,
                cosine_distance=distance,
                reference_asset_id=asset.asset_id,
                source_id=asset.source_id,
                provenance_summary=asset.provenance_summary,
                province=asset.province,
                latitude=asset.latitude,
                longitude=asset.longitude,
            )
            for rank, (distance, asset) in enumerate(ordered, start=1)
        ]

    def revocation_impact(self, asset_id: str) -> RevocationImpact:
        shard_ids: list[str] = []
        artifacts: set[str] = {"metadata.json", "PUBLISHED.json"}
        for shard in self._shards:
            if any(asset.asset_id == asset_id for asset in shard.assets):
                shard_ids.append(shard.shard_id)
                artifacts.update((shard.assets_relative_path, shard.vectors_relative_path))
        return RevocationImpact(
            asset_id=asset_id,
            found=bool(shard_ids),
            shard_ids=tuple(shard_ids),
            artifact_paths=tuple(sorted(artifacts)) if shard_ids else (),
        )

    def artifact_inventory(self) -> tuple[ArtifactInventoryItem, ...]:
        items = [
            _inventory_item(self.directory, "metadata.json", "index_metadata"),
            _inventory_item(self.directory, "PUBLISHED.json", "publication_marker"),
        ]
        for shard in self._shards:
            items.extend(
                (
                    _inventory_item(
                        self.directory,
                        shard.assets_relative_path,
                        "asset_mapping",
                        shard.shard_id,
                    ),
                    _inventory_item(
                        self.directory,
                        shard.vectors_relative_path,
                        "vector_index",
                        shard.shard_id,
                    ),
                )
            )
        return tuple(sorted(items, key=lambda item: item.relative_path))

    def rebuild_plan(self, revoked_asset_ids: Sequence[str]) -> RebuildPlan:
        requested = tuple(sorted(set(revoked_asset_ids)))
        asset_to_shard = {
            asset.asset_id: shard.shard_id
            for shard in self._shards
            for asset in shard.assets
        }
        found = tuple(asset_id for asset_id in requested if asset_id in asset_to_shard)
        missing = tuple(asset_id for asset_id in requested if asset_id not in asset_to_shard)
        affected = tuple(sorted({asset_to_shard[asset_id] for asset_id in found}))
        inventory = self.artifact_inventory()
        basis = {
            "index_build_hash": self.metadata["build_hash"],
            "requested_asset_ids": list(requested),
            "found_asset_ids": list(found),
            "affected_shards": list(affected),
        }
        return RebuildPlan(
            plan_id=sha256_json(basis),
            requested_asset_ids=requested,
            found_asset_ids=found,
            missing_asset_ids=missing,
            affected_shards=affected,
            retained_asset_count=self.size - len(found),
            removed_asset_count=len(found),
            artifact_inventory=inventory,
            steps=(
                "mark requested assets revoked in the governed corpus manifest",
                "re-run leakage and rights validation with the unchanged locked policy",
                "rebuild affected shards from non-revoked descriptors",
                "verify every checksum and asset mapping in a staging directory",
                "atomically publish the replacement index and retain the rebuild receipt",
            ),
        )

    @classmethod
    def open(
        cls,
        directory: Path,
        *,
        expected_spec: DescriptorSpec | None = None,
        expected_source_policy_hash: str | None = None,
        expected_locked_holdout_hash: str | None = None,
        expected_split_lock_hash: str | None = None,
    ) -> PublishedCorpusIndex:
        resolved = directory.resolve()
        metadata_path = resolved / "metadata.json"
        marker_path = resolved / "PUBLISHED.json"
        try:
            metadata_value = read_json(metadata_path)
            marker_value = read_json(marker_path)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ArtifactIntegrityError("index publication metadata is invalid") from exc
        if not isinstance(metadata_value, dict) or not isinstance(marker_value, dict):
            raise ArtifactIntegrityError("index publication metadata is invalid")
        metadata = cast(dict[str, Any], metadata_value)
        if metadata.get("schema") != _INDEX_SCHEMA:
            raise ArtifactIntegrityError("unsupported corpus index schema")
        if marker_value.get("metadata_sha256") != sha256_path(metadata_path):
            raise ArtifactIntegrityError("index publication marker mismatch")
        try:
            spec = DescriptorSpec.from_json(metadata.get("descriptor_spec"))
        except ValueError as exc:
            raise ArtifactIntegrityError("index descriptor specification is invalid") from exc
        _verify_expected_compatibility(
            metadata,
            spec,
            expected_spec=expected_spec,
            expected_source_policy_hash=expected_source_policy_hash,
            expected_locked_holdout_hash=expected_locked_holdout_hash,
            expected_split_lock_hash=expected_split_lock_hash,
        )
        backend_name = metadata.get("backend")
        if backend_name not in {_EXACT_BACKEND, _FAISS_BACKEND}:
            raise ArtifactIntegrityError("index backend is unsupported")
        raw_shards = metadata.get("shards")
        if not isinstance(raw_shards, list):
            raise ArtifactIntegrityError("index shard metadata is invalid")
        loaded: list[_LoadedShard] = []
        seen_assets: set[str] = set()
        for raw_shard in raw_shards:
            loaded.append(
                _load_shard(
                    resolved,
                    raw_shard,
                    backend_name=backend_name,
                    spec=spec,
                    seen_assets=seen_assets,
                )
            )
        asset_count = metadata.get("asset_count")
        if not isinstance(asset_count, int) or asset_count != sum(
            len(shard.assets) for shard in loaded
        ):
            raise ArtifactIntegrityError("index asset count mismatch")
        return cls(resolved, metadata, spec, tuple(loaded))


def build_index(
    dataset: DescriptorDataset,
    *,
    output_dir: Path,
    backend: IndexBackendName,
    index_version: str,
    source_policy_hash: str,
    shard_size: int = 10_000,
    locked_holdout_hash: str | None = None,
    split_lock_hash: str | None = None,
) -> PublishedCorpusIndex:
    """Build and atomically publish an exact sharded corpus index."""

    if not index_version.strip():
        raise ValueError("index_version must not be blank")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    _require_sha256(source_policy_hash, "source_policy_hash")
    if source_policy_hash != dataset.source_policy_hash:
        raise IndexCompatibilityError("source policy does not match the descriptor dataset")
    if (locked_holdout_hash is None) != (split_lock_hash is None):
        raise ValueError("locked holdout and split lock hashes must be provided together")
    if locked_holdout_hash is not None and split_lock_hash is not None:
        _require_sha256(locked_holdout_hash, "locked_holdout_hash")
        _require_sha256(split_lock_hash, "split_lock_hash")
    backend_id = _EXACT_BACKEND if backend == "exact" else _FAISS_BACKEND
    build_basis = {
        "schema": _INDEX_SCHEMA,
        "index_version": index_version,
        "backend": backend_id,
        "descriptor_spec": dataset.spec.to_json(),
        "descriptor_input_hash": dataset.input_hash,
        "manifest_hash": dataset.manifest_hash,
        "source_policy_hash": source_policy_hash,
        "locked_holdout_hash": locked_holdout_hash,
        "split_lock_hash": split_lock_hash,
        "asset_count": dataset.count,
        "shard_size": shard_size,
    }
    build_hash = sha256_json(build_basis)
    resolved_output = output_dir.resolve()
    if resolved_output.exists():
        existing = PublishedCorpusIndex.open(resolved_output)
        if existing.metadata.get("build_hash") != build_hash:
            raise IndexCompatibilityError("published index configuration is incompatible")
        return existing
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    stage = resolved_output.with_name(f".{resolved_output.name}.{uuid4().hex}.tmp")
    try:
        stage.mkdir()
        shard_entries: list[dict[str, Any]] = []
        for number, offset in enumerate(range(0, dataset.count, shard_size)):
            shard_id = f"shard-{number:06d}"
            shard_dir = stage / "shards" / shard_id
            shard_dir.mkdir(parents=True)
            vectors = np.ascontiguousarray(
                dataset.vectors[offset : offset + shard_size], dtype=np.float32
            )
            assets = dataset.assets[offset : offset + len(vectors)]
            assets_path = shard_dir / "assets.json"
            atomic_write_json(assets_path, [asset.to_json() for asset in assets])
            if backend == "exact":
                vectors_path = shard_dir / "vectors.npy"
                with vectors_path.open("wb") as handle:
                    np.save(handle, vectors, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
            else:
                vectors_path = shard_dir / "vectors.faiss"
                index = faiss.IndexFlatIP(dataset.spec.dimension)
                index.add(vectors)
                faiss.write_index(index, str(vectors_path))
            shard_entries.append(
                {
                    "shard_id": shard_id,
                    "count": len(assets),
                    "assets_file": f"shards/{shard_id}/{assets_path.name}",
                    "assets_sha256": sha256_path(assets_path),
                    "vectors_file": f"shards/{shard_id}/{vectors_path.name}",
                    "vectors_sha256": sha256_path(vectors_path),
                    "asset_ids": [asset.asset_id for asset in assets],
                }
            )
        metadata = {
            **build_basis,
            "build_hash": build_hash,
            "shards": shard_entries,
        }
        metadata_path = stage / "metadata.json"
        atomic_write_json(metadata_path, metadata)
        atomic_write_json(
            stage / "PUBLISHED.json",
            {"schema": _INDEX_SCHEMA, "metadata_sha256": sha256_path(metadata_path)},
        )
        os.replace(stage, resolved_output)
    except OSError as exc:
        raise ArtifactIntegrityError("corpus index could not be published atomically") from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return PublishedCorpusIndex.open(
        resolved_output,
        expected_spec=dataset.spec,
        expected_source_policy_hash=source_policy_hash,
        expected_locked_holdout_hash=locked_holdout_hash,
        expected_split_lock_hash=split_lock_hash,
    )


def write_rebuild_plan(path: Path, plan: RebuildPlan) -> None:
    """Atomically persist a deterministic revocation rebuild plan."""

    atomic_write_json(path, plan.to_json())


def _load_shard(
    root: Path,
    raw_value: object,
    *,
    backend_name: object,
    spec: DescriptorSpec,
    seen_assets: set[str],
) -> _LoadedShard:
    if not isinstance(raw_value, dict):
        raise ArtifactIntegrityError("index shard metadata is invalid")
    value = cast(dict[str, Any], raw_value)
    shard_id = value.get("shard_id")
    count = value.get("count")
    assets_name = value.get("assets_file")
    vectors_name = value.get("vectors_file")
    asset_ids = value.get("asset_ids")
    if (
        not isinstance(shard_id, str)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(assets_name, str)
        or not isinstance(vectors_name, str)
        or not isinstance(asset_ids, list)
        or any(not isinstance(asset_id, str) for asset_id in asset_ids)
    ):
        raise ArtifactIntegrityError("index shard metadata is invalid")
    assets_path = _contained_artifact(root, assets_name)
    vectors_path = _contained_artifact(root, vectors_name)
    if value.get("assets_sha256") != sha256_path(assets_path):
        raise ArtifactIntegrityError("index asset mapping checksum mismatch")
    if value.get("vectors_sha256") != sha256_path(vectors_path):
        raise ArtifactIntegrityError("index vector checksum mismatch")
    try:
        asset_values = read_json(assets_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArtifactIntegrityError("index asset mapping is invalid") from exc
    if not isinstance(asset_values, list):
        raise ArtifactIntegrityError("index asset mapping is invalid")
    try:
        assets = tuple(AssetProvenance.from_json(item) for item in asset_values)
    except ValueError as exc:
        raise ArtifactIntegrityError("index asset mapping is invalid") from exc
    if (
        len(assets) != count
        or [asset.asset_id for asset in assets] != asset_ids
        or any(not asset.rights_validated or asset.revoked for asset in assets)
        or seen_assets.intersection(asset_ids)
    ):
        raise ArtifactIntegrityError("index asset mapping is inconsistent")
    seen_assets.update(asset_ids)
    if backend_name == _EXACT_BACKEND:
        try:
            vectors = cast(FloatMatrix, np.load(vectors_path, allow_pickle=False, mmap_mode="r"))
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError("exact index vectors are invalid") from exc
        if vectors.shape != (count, spec.dimension) or vectors.dtype != np.float32:
            raise ArtifactIntegrityError("exact index vector shape or dtype mismatch")
        if not np.isfinite(vectors).all() or not np.allclose(
            np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=1e-5
        ):
            raise ArtifactIntegrityError("exact index vectors are not finite unit vectors")
        backend: _ShardBackend = _ExactCosineShard(vectors, assets)
    else:
        try:
            faiss_index = faiss.read_index(str(vectors_path))
        except RuntimeError as exc:
            raise ArtifactIntegrityError("FAISS index cannot be loaded") from exc
        if int(faiss_index.d) != spec.dimension or int(faiss_index.ntotal) != count:
            raise ArtifactIntegrityError("FAISS index dimension or count mismatch")
        backend = _FaissCosineShard(faiss_index, assets)
    return _LoadedShard(shard_id, assets, backend, assets_name, vectors_name)


def _verify_expected_compatibility(
    metadata: dict[str, Any],
    spec: DescriptorSpec,
    *,
    expected_spec: DescriptorSpec | None,
    expected_source_policy_hash: str | None,
    expected_locked_holdout_hash: str | None,
    expected_split_lock_hash: str | None,
) -> None:
    if expected_spec is not None and spec != expected_spec:
        raise IndexCompatibilityError("index descriptor specification is incompatible")
    if (
        expected_source_policy_hash is not None
        and metadata.get("source_policy_hash") != expected_source_policy_hash
    ):
        raise IndexCompatibilityError("index source policy is incompatible")
    if (expected_locked_holdout_hash is None) != (expected_split_lock_hash is None):
        raise IndexCompatibilityError("expected holdout and split lock hashes must be paired")
    if expected_locked_holdout_hash is not None and expected_split_lock_hash is not None:
        if metadata.get("locked_holdout_hash") != expected_locked_holdout_hash:
            raise IndexCompatibilityError("index locked holdout is incompatible")
        if metadata.get("split_lock_hash") != expected_split_lock_hash:
            raise IndexCompatibilityError("index split lock is incompatible")
    for key in ("descriptor_input_hash", "manifest_hash", "source_policy_hash", "build_hash"):
        value = metadata.get(key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ArtifactIntegrityError(f"index {key} is invalid")
    holdout_hash = metadata.get("locked_holdout_hash")
    split_lock_hash = metadata.get("split_lock_hash")
    if (holdout_hash is None) != (split_lock_hash is None):
        raise ArtifactIntegrityError("index holdout and split lock hashes are not paired")
    if holdout_hash is not None and split_lock_hash is not None:
        if not isinstance(holdout_hash, str) or not _SHA256.fullmatch(holdout_hash):
            raise ArtifactIntegrityError("index locked_holdout_hash is invalid")
        if not isinstance(split_lock_hash, str) or not _SHA256.fullmatch(split_lock_hash):
            raise ArtifactIntegrityError("index split_lock_hash is invalid")
    index_version = metadata.get("index_version")
    if not isinstance(index_version, str) or not index_version.strip():
        raise ArtifactIntegrityError("index version is invalid")


def _contained_artifact(root: Path, relative_name: str) -> Path:
    candidate = (root / relative_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ArtifactIntegrityError("index artifact path escapes its publication") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise ArtifactIntegrityError("index artifact must be a regular non-symlink file")
    return candidate


def _inventory_item(
    root: Path, relative_path: str, role: str, shard_id: str | None = None
) -> ArtifactInventoryItem:
    path = _contained_artifact(root, relative_path)
    return ArtifactInventoryItem(
        relative_path=relative_path,
        sha256=sha256_path(path),
        size_bytes=path.stat().st_size,
        role=role,
        shard_id=shard_id,
    )


def _require_sha256(value: str, field: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
