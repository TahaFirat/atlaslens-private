"""Resumable, checksum-bound descriptor generation and publication."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np

from atlaslens_api.corpus_index.artifacts import (
    atomic_write_json,
    atomic_write_npy,
    read_json,
    sha256_json,
    sha256_path,
)
from atlaslens_api.corpus_index.errors import (
    ArtifactIntegrityError,
    CheckpointCompatibilityError,
    DescriptorValidationError,
)
from atlaslens_api.corpus_index.models import (
    AssetProvenance,
    DescriptorAsset,
    DescriptorSpec,
    FloatMatrix,
    normalize_matrix,
)
from atlaslens_api.corpus_index.providers import (
    DescriptorProvider,
    ProductionDescriptorRegistry,
)

_CHECKPOINT_SCHEMA = "atlaslens-descriptor-checkpoint-v1"
_DATASET_SCHEMA = "atlaslens-descriptor-dataset-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DescriptorBuildResult:
    dataset: DescriptorDataset
    resumed_assets: int
    generated_assets: int


class DescriptorDataset:
    """Verified, immutable descriptor publication."""

    def __init__(
        self,
        directory: Path,
        spec: DescriptorSpec,
        vectors: FloatMatrix,
        assets: tuple[AssetProvenance, ...],
        metadata: dict[str, Any],
    ) -> None:
        self.directory = directory
        self.spec = spec
        self.vectors = vectors
        self.assets = assets
        self.metadata = metadata

    @property
    def count(self) -> int:
        return len(self.assets)

    @property
    def config_hash(self) -> str:
        return str(self.metadata["config_hash"])

    @property
    def input_hash(self) -> str:
        return str(self.metadata["input_hash"])

    @property
    def source_policy_hash(self) -> str:
        return str(self.metadata["source_policy_hash"])

    @property
    def manifest_hash(self) -> str:
        return str(self.metadata["manifest_hash"])

    @classmethod
    def open(cls, directory: Path) -> DescriptorDataset:
        resolved = directory.resolve()
        metadata_path = resolved / "metadata.json"
        ready_path = resolved / "PUBLISHED.json"
        vectors_path = resolved / "vectors.npy"
        assets_path = resolved / "assets.json"
        try:
            metadata_value = read_json(metadata_path)
            ready_value = read_json(ready_path)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ArtifactIntegrityError("descriptor publication metadata is invalid") from exc
        if not isinstance(metadata_value, dict) or not isinstance(ready_value, dict):
            raise ArtifactIntegrityError("descriptor publication metadata is invalid")
        metadata = cast(dict[str, Any], metadata_value)
        if metadata.get("schema") != _DATASET_SCHEMA:
            raise ArtifactIntegrityError("unsupported descriptor dataset schema")
        if ready_value.get("metadata_sha256") != sha256_path(metadata_path):
            raise ArtifactIntegrityError("descriptor publication marker mismatch")
        if (
            not vectors_path.is_file()
            or metadata.get("vectors_sha256") != sha256_path(vectors_path)
        ):
            raise ArtifactIntegrityError("descriptor vector checksum mismatch")
        if not assets_path.is_file() or metadata.get("assets_sha256") != sha256_path(assets_path):
            raise ArtifactIntegrityError("descriptor asset mapping checksum mismatch")
        try:
            spec = DescriptorSpec.from_json(metadata.get("descriptor_spec"))
            asset_values = read_json(assets_path)
            vectors = cast(FloatMatrix, np.load(vectors_path, allow_pickle=False, mmap_mode="r"))
        except (OSError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("descriptor dataset cannot be loaded") from exc
        if not isinstance(asset_values, list):
            raise ArtifactIntegrityError("descriptor asset mapping is invalid")
        try:
            assets = tuple(AssetProvenance.from_json(value) for value in asset_values)
        except ValueError as exc:
            raise ArtifactIntegrityError("descriptor asset mapping is invalid") from exc
        count = metadata.get("count")
        if not isinstance(count, int) or count != len(assets):
            raise ArtifactIntegrityError("descriptor asset count mismatch")
        if vectors.shape != (count, spec.dimension) or vectors.dtype != np.float32:
            raise ArtifactIntegrityError("descriptor vector shape or dtype mismatch")
        if count and (
            not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=1e-5)
        ):
            raise ArtifactIntegrityError("published descriptors are not finite unit vectors")
        if len({asset.asset_id for asset in assets}) != len(assets):
            raise ArtifactIntegrityError("descriptor asset IDs are not unique")
        return cls(resolved, spec, vectors, assets, metadata)


def build_production_descriptors(
    assets: Sequence[DescriptorAsset],
    registry: ProductionDescriptorRegistry,
    *,
    work_dir: Path,
    output_dir: Path,
    manifest_hash: str,
    source_policy_hash: str,
    batch_size: int = 32,
) -> DescriptorBuildResult:
    """Build only through an explicitly activated rights-approved production adapter."""

    provider = registry.active()
    return build_descriptors(
        assets,
        provider,
        work_dir=work_dir,
        output_dir=output_dir,
        manifest_hash=manifest_hash,
        source_policy_hash=source_policy_hash,
        batch_size=batch_size,
    )


def build_descriptors(
    assets: Sequence[DescriptorAsset],
    provider: DescriptorProvider,
    *,
    work_dir: Path,
    output_dir: Path,
    manifest_hash: str,
    source_policy_hash: str,
    batch_size: int = 32,
) -> DescriptorBuildResult:
    """Generate normalized descriptors with validated atomic checkpoint resume."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _require_sha256(manifest_hash, "manifest_hash")
    _require_sha256(source_policy_hash, "source_policy_hash")
    ordered = tuple(sorted(assets, key=lambda asset: asset.provenance.asset_id))
    _validate_assets(ordered)
    spec = provider.spec
    config = {
        "schema": _CHECKPOINT_SCHEMA,
        "descriptor_spec": spec.to_json(),
        "batch_size": batch_size,
        "manifest_hash": manifest_hash,
        "source_policy_hash": source_policy_hash,
    }
    config_hash = sha256_json(config)
    input_hash = sha256_json([asset.provenance.to_json() for asset in ordered])
    resolved_output = output_dir.resolve()
    resolved_work = work_dir.resolve()
    if resolved_output == resolved_work:
        raise ValueError("descriptor work and output directories must differ")
    if resolved_output.exists():
        dataset = DescriptorDataset.open(resolved_output)
        _require_dataset_compatibility(dataset, config_hash, input_hash)
        return DescriptorBuildResult(dataset, resumed_assets=dataset.count, generated_assets=0)

    checkpoint_path = resolved_work / "descriptor-checkpoint.json"
    shard_dir = resolved_work / "descriptor-shards"
    resolved_work.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        checkpoint = _read_checkpoint(checkpoint_path)
        entries, completed = _verify_checkpoint(
            checkpoint,
            shard_dir,
            ordered,
            spec,
            config_hash=config_hash,
            input_hash=input_hash,
        )
    else:
        if any(shard_dir.iterdir()):
            raise CheckpointCompatibilityError("descriptor shards exist without a checkpoint")
        entries = []
        completed = 0
        checkpoint = _checkpoint_payload(config_hash, input_hash, entries, completed)
        atomic_write_json(checkpoint_path, checkpoint)

    resumed_assets = completed
    for offset in range(completed, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        matrix = normalize_matrix(
            provider.describe_batch([asset.locator for asset in batch]),
            rows=len(batch),
            dimension=spec.dimension,
        )
        shard_number = len(entries)
        vectors_name = f"shard-{shard_number:06d}.npy"
        assets_name = f"shard-{shard_number:06d}.json"
        vectors_path = shard_dir / vectors_name
        assets_path = shard_dir / assets_name
        atomic_write_npy(vectors_path, matrix)
        atomic_write_json(assets_path, [asset.provenance.to_json() for asset in batch])
        entry: dict[str, Any] = {
            "number": shard_number,
            "vectors_file": vectors_name,
            "vectors_sha256": sha256_path(vectors_path),
            "assets_file": assets_name,
            "assets_sha256": sha256_path(assets_path),
            "asset_ids": [asset.provenance.asset_id for asset in batch],
            "count": len(batch),
        }
        entries.append(entry)
        completed += len(batch)
        checkpoint = _checkpoint_payload(config_hash, input_hash, entries, completed)
        atomic_write_json(checkpoint_path, checkpoint)

    dataset = _publish_dataset(
        resolved_output,
        shard_dir,
        entries,
        ordered,
        spec,
        config_hash=config_hash,
        input_hash=input_hash,
        manifest_hash=manifest_hash,
        source_policy_hash=source_policy_hash,
    )
    return DescriptorBuildResult(
        dataset,
        resumed_assets=resumed_assets,
        generated_assets=len(ordered) - resumed_assets,
    )


def inspect_descriptor_checkpoint(path: Path) -> dict[str, Any]:
    """Return non-sensitive checkpoint state for CLI/operator inspection."""

    checkpoint = _read_checkpoint(path)
    entries = checkpoint.get("shards")
    return {
        "schema": checkpoint.get("schema"),
        "config_hash": checkpoint.get("config_hash"),
        "input_hash": checkpoint.get("input_hash"),
        "completed_assets": checkpoint.get("completed_assets"),
        "shard_count": len(entries) if isinstance(entries, list) else 0,
    }


def _validate_assets(assets: Sequence[DescriptorAsset]) -> None:
    ids: set[str] = set()
    for asset in assets:
        provenance = asset.provenance
        if provenance.asset_id in ids:
            raise DescriptorValidationError("descriptor asset IDs must be unique")
        ids.add(provenance.asset_id)
        if not provenance.rights_validated or provenance.revoked:
            raise DescriptorValidationError("descriptor asset is not rights-valid and active")
        if not asset.locator.is_file() or asset.locator.is_symlink():
            raise DescriptorValidationError("descriptor locator must be a regular non-symlink file")


def _checkpoint_payload(
    config_hash: str,
    input_hash: str,
    entries: Sequence[dict[str, Any]],
    completed: int,
) -> dict[str, Any]:
    return {
        "schema": _CHECKPOINT_SCHEMA,
        "config_hash": config_hash,
        "input_hash": input_hash,
        "completed_assets": completed,
        "shards": list(entries),
    }


def _read_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CheckpointCompatibilityError("descriptor checkpoint is invalid") from exc
    if not isinstance(value, dict):
        raise CheckpointCompatibilityError("descriptor checkpoint is invalid")
    return cast(dict[str, Any], value)


def _verify_checkpoint(
    checkpoint: dict[str, Any],
    shard_dir: Path,
    expected_assets: Sequence[DescriptorAsset],
    spec: DescriptorSpec,
    *,
    config_hash: str,
    input_hash: str,
) -> tuple[list[dict[str, Any]], int]:
    if checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
        raise CheckpointCompatibilityError("unsupported descriptor checkpoint schema")
    if checkpoint.get("config_hash") != config_hash or checkpoint.get("input_hash") != input_hash:
        raise CheckpointCompatibilityError("descriptor checkpoint configuration is incompatible")
    raw_entries = checkpoint.get("shards")
    completed = checkpoint.get("completed_assets")
    if not isinstance(raw_entries, list) or not isinstance(completed, int) or completed < 0:
        raise CheckpointCompatibilityError("descriptor checkpoint state is invalid")
    entries: list[dict[str, Any]] = []
    seen = 0
    for number, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise CheckpointCompatibilityError("descriptor checkpoint shard is invalid")
        entry = cast(dict[str, Any], raw_entry)
        vectors_name = entry.get("vectors_file")
        assets_name = entry.get("assets_file")
        count = entry.get("count")
        asset_ids = entry.get("asset_ids")
        if (
            entry.get("number") != number
            or not isinstance(vectors_name, str)
            or not isinstance(assets_name, str)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(asset_ids, list)
            or any(not isinstance(asset_id, str) for asset_id in asset_ids)
        ):
            raise CheckpointCompatibilityError("descriptor checkpoint shard is invalid")
        vectors_path = shard_dir / vectors_name
        assets_path = shard_dir / assets_name
        if (
            not vectors_path.is_file()
            or entry.get("vectors_sha256") != sha256_path(vectors_path)
            or not assets_path.is_file()
            or entry.get("assets_sha256") != sha256_path(assets_path)
        ):
            raise CheckpointCompatibilityError("descriptor checkpoint shard checksum mismatch")
        try:
            vectors = cast(FloatMatrix, np.load(vectors_path, allow_pickle=False))
            mapped_assets = read_json(assets_path)
        except (OSError, ValueError) as exc:
            raise CheckpointCompatibilityError("descriptor checkpoint shard is unreadable") from exc
        if vectors.shape != (count, spec.dimension) or not isinstance(mapped_assets, list):
            raise CheckpointCompatibilityError("descriptor checkpoint shard shape is invalid")
        try:
            normalized = normalize_matrix(vectors, rows=count, dimension=spec.dimension)
        except DescriptorValidationError as exc:
            raise CheckpointCompatibilityError("descriptor checkpoint vector is invalid") from exc
        if not np.allclose(vectors, normalized, atol=1e-5, rtol=1e-5):
            raise CheckpointCompatibilityError("descriptor checkpoint vector is not normalized")
        expected_ids = [
            asset.provenance.asset_id for asset in expected_assets[seen : seen + count]
        ]
        if asset_ids != expected_ids:
            raise CheckpointCompatibilityError("descriptor checkpoint asset sequence mismatch")
        try:
            mapped = [AssetProvenance.from_json(value) for value in mapped_assets]
        except ValueError as exc:
            raise CheckpointCompatibilityError("descriptor checkpoint mapping is invalid") from exc
        if [asset.asset_id for asset in mapped] != expected_ids:
            raise CheckpointCompatibilityError("descriptor checkpoint mapping mismatch")
        entries.append(entry)
        seen += count
    if completed != seen or completed > len(expected_assets):
        raise CheckpointCompatibilityError("descriptor checkpoint completion count mismatch")
    return entries, completed


def _publish_dataset(
    output_dir: Path,
    shard_dir: Path,
    entries: Sequence[dict[str, Any]],
    assets: Sequence[DescriptorAsset],
    spec: DescriptorSpec,
    *,
    config_hash: str,
    input_hash: str,
    manifest_hash: str,
    source_policy_hash: str,
) -> DescriptorDataset:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(f".{output_dir.name}.{uuid4().hex}.tmp")
    try:
        stage.mkdir()
        vectors_path = stage / "vectors.npy"
        if assets:
            target = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
                vectors_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(assets), spec.dimension),
            )
            offset = 0
            for entry in entries:
                shard = cast(
                    FloatMatrix,
                    np.load(shard_dir / str(entry["vectors_file"]), allow_pickle=False),
                )
                target[offset : offset + len(shard)] = shard
                offset += len(shard)
            target.flush()
            del target
        else:
            with vectors_path.open("wb") as handle:
                np.save(handle, np.empty((0, spec.dimension), dtype=np.float32), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
        assets_path = stage / "assets.json"
        atomic_write_json(assets_path, [asset.provenance.to_json() for asset in assets])
        metadata: dict[str, Any] = {
            "schema": _DATASET_SCHEMA,
            "descriptor_spec": spec.to_json(),
            "config_hash": config_hash,
            "input_hash": input_hash,
            "manifest_hash": manifest_hash,
            "source_policy_hash": source_policy_hash,
            "count": len(assets),
            "vectors_sha256": sha256_path(vectors_path),
            "assets_sha256": sha256_path(assets_path),
            "checkpoint_shards": len(entries),
        }
        metadata_path = stage / "metadata.json"
        atomic_write_json(metadata_path, metadata)
        atomic_write_json(
            stage / "PUBLISHED.json",
            {"schema": _DATASET_SCHEMA, "metadata_sha256": sha256_path(metadata_path)},
        )
        os.replace(stage, output_dir)
    except OSError as exc:
        raise ArtifactIntegrityError(
            "descriptor dataset could not be published atomically"
        ) from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return DescriptorDataset.open(output_dir)


def _require_dataset_compatibility(
    dataset: DescriptorDataset, config_hash: str, input_hash: str
) -> None:
    if dataset.config_hash != config_hash or dataset.input_hash != input_hash:
        raise CheckpointCompatibilityError("published descriptor dataset is incompatible")


def _require_sha256(value: str, field: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
