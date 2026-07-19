"""Mapillary split conversion into the governed Phase 3B1 corpus pipeline."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from atlaslens_api.corpus_index.artifacts import canonical_json_bytes, sha256_json, sha256_path
from atlaslens_api.corpus_pipeline.hashing import inspect_image
from atlaslens_api.corpus_pipeline.ingestion import CorpusIngestor, IngestionResult
from atlaslens_api.corpus_pipeline.models import IngestionConfig, ManifestAsset
from atlaslens_api.mapillary_demo.selection import (
    LockedMapillarySplit,
    SelectedMapillaryAsset,
)

MAPILLARY_SOURCE_ID = "mapillary_public_imagery_private_demo"
MAPILLARY_SOURCE_NAME = "Mapillary public imagery (private technical demo)"
MAPILLARY_POLICY_VERSION = "phase3b3-mapillary-private-demo-v1"
MAPILLARY_POLICY_EVIDENCE_DATE = "2026-07-17"
MAPILLARY_CORPUS_VERSION = "atlaslens-turkiye-corpus-phase3b3-mapillary-demo-v1"
MAPILLARY_INDEX_VERSION = "mapillary-private-demo-faiss-flatip-v1"
MAPILLARY_DESCRIPTOR_VERSION = "megaloc-phase3b2-mapillary-private-demo-v1"

type Phase3B1MimeType = Literal[
    "image/jpeg", "image/png", "image/webp", "image/tiff"
]


@dataclass(frozen=True, slots=True)
class MapillaryManifestBuildResult:
    corpus_root: Path
    manifest_path: Path
    manifest_sha256: str
    source_policy_sha256: str
    split_lock_sha256: str
    asset_count: int
    copied_assets: int
    reused_assets: int


def materialize_phase3b1_manifest(
    split: LockedMapillarySplit,
    *,
    acquisition_root: Path,
    corpus_root: Path,
    manifest_path: Path | None,
    source_policy_path: Path,
    coordinate_uncertainty_m: float,
) -> MapillaryManifestBuildResult:
    """Copy bounded normalized files and emit strict Phase 3B1 JSONL rows."""

    split.verify()
    if not 0.0 < coordinate_uncertainty_m <= 100_000.0:
        raise ValueError("coordinate uncertainty must be positive and bounded")
    policy_hash = sha256_path(_regular_file(source_policy_path))
    if policy_hash != split.source_policy_sha256:
        raise ValueError("source policy hash does not match the acquisition receipt")
    acquisition = acquisition_root.resolve()
    corpus = corpus_root.resolve()
    assets_root = corpus / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    copied = 0
    reused = 0
    for asset in sorted(split.all_assets, key=lambda item: item.asset_id):
        source = _contained_file(acquisition, asset.relative_path)
        identity = inspect_image(source, max_bytes=64 * 1024 * 1024, max_pixels=100_000_000)
        if (
            identity.sha256 != asset.normalized_sha256
            or identity.perceptual_hash != asset.perceptual_hash
            or (identity.width_px, identity.height_px) != (asset.width_px, asset.height_px)
        ):
            raise ValueError("selected Mapillary asset changed after split lock")
        mime_type = _validated_mime_type(identity.mime_type)
        extension = _extension(mime_type)
        destination = assets_root / f"{asset.asset_id}{extension}"
        if destination.exists():
            if destination.is_symlink() or sha256_path(destination) != asset.normalized_sha256:
                raise ValueError("existing corpus asset is incompatible with the locked split")
            reused += 1
        else:
            _atomic_copy(source, destination)
            if sha256_path(destination) != asset.normalized_sha256:
                destination.unlink(missing_ok=True)
                raise ValueError("corpus materialization checksum mismatch")
            copied += 1
        row = _phase3b1_row(
            asset,
            mime_type=mime_type,
            coordinate_uncertainty_m=coordinate_uncertainty_m,
            source_policy_sha256=policy_hash,
        )
        ManifestAsset.model_validate(row)
        rows.append(row)

    selected_manifest = manifest_path or corpus / "mapillary-phase3b1-manifest.jsonl"
    resolved_manifest = selected_manifest.resolve()
    try:
        resolved_manifest.relative_to(corpus)
    except ValueError as exc:
        raise ValueError("Phase 3B1 manifest must remain inside the corpus root") from exc
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    _atomic_write(resolved_manifest, payload)
    return MapillaryManifestBuildResult(
        corpus_root=corpus,
        manifest_path=resolved_manifest,
        manifest_sha256=sha256_path(resolved_manifest),
        source_policy_sha256=policy_hash,
        split_lock_sha256=split.lock_sha256,
        asset_count=len(rows),
        copied_assets=copied,
        reused_assets=reused,
    )


def ingest_with_phase3b1(
    manifest: MapillaryManifestBuildResult,
    *,
    work_root: Path,
    checkpoint_root: Path,
    manifest_schema_path: Path,
    source_policy_path: Path,
    checkpoint_path: Path | str = "phase3b3-mapillary-checkpoint.json",
    resume: bool = False,
) -> IngestionResult:
    """Run the existing rights, identity, leakage and split-lock implementation."""

    if sha256_path(_regular_file(source_policy_path)) != manifest.source_policy_sha256:
        raise ValueError("source policy changed after manifest materialization")
    ingestor = CorpusIngestor(
        corpus_root=manifest.corpus_root,
        work_root=work_root,
        checkpoint_root=checkpoint_root,
        manifest_schema_path=manifest_schema_path,
        source_policy_path=source_policy_path,
        config=IngestionConfig(
            descriptor_version=MAPILLARY_DESCRIPTOR_VERSION,
            index_version=MAPILLARY_INDEX_VERSION,
            near_duplicate_hamming_threshold=0,
            # Geographic positives are required for retrieval evaluation. Identity,
            # perceptual and sequence leakage are already locked; this radius only
            # rejects effectively identical coordinates instead of all nearby refs.
            spatial_leakage_radius_m=0.01,
            strict_rejections=True,
        ),
    )
    return ingestor.ingest(
        manifest.manifest_path,
        checkpoint_path=checkpoint_path,
        run_id=f"phase3b3-mapillary-{manifest.split_lock_sha256[:16]}",
        resume=resume,
    )


def attribution_records(split: LockedMapillarySplit) -> tuple[dict[str, object], ...]:
    """Return complete credential-free attribution metadata for index publication."""

    split.verify()
    return tuple(
        {
            "asset_id": asset.asset_id,
            "mapillary_image_id": asset.mapillary_image_id,
            "source_partition": MAPILLARY_SOURCE_ID,
            "source_page_url": asset.source_page_url,
            "attribution_text": asset.attribution_text,
            "license_identifier": asset.license_identifier,
            "license_url": asset.license_url,
            "creator_id": asset.creator_id,
            "captured_at": asset.captured_at.isoformat(),
            "acquired_at": asset.acquired_at.isoformat(),
            "latitude": asset.latitude,
            "longitude": asset.longitude,
            "compass_angle": asset.compass_angle,
            "sequence_id": asset.sequence_id,
            "city": asset.city,
            "province": asset.province,
            "province_code": asset.province_code,
            "raw_sha256": asset.raw_sha256,
            "normalized_sha256": asset.normalized_sha256,
            "perceptual_hash": asset.perceptual_hash,
            "split": asset.split,
            "reconciliation_state": "active",
        }
        for asset in sorted(split.all_assets, key=lambda item: item.asset_id)
    )


def attribution_manifest_hash(split: LockedMapillarySplit) -> str:
    return sha256_json(
        {
            "schema": "atlaslens-mapillary-attribution-v1",
            "source_policy_sha256": split.source_policy_sha256,
            "split_lock_sha256": split.lock_sha256,
            "records": list(attribution_records(split)),
        }
    )


def _phase3b1_row(
    asset: SelectedMapillaryAsset,
    *,
    mime_type: Phase3B1MimeType,
    coordinate_uncertainty_m: float,
    source_policy_sha256: str,
) -> dict[str, object]:
    contributor = asset.creator_id or "unknown-mapillary-contributor"
    holdout = asset.split == "holdout"
    return {
        "corpus_version": MAPILLARY_CORPUS_VERSION,
        "asset_id": asset.asset_id,
        "source_asset_id": asset.mapillary_image_id,
        "source_name": MAPILLARY_SOURCE_NAME,
        "source_url": asset.source_page_url,
        "contributor_or_owner": f"Mapillary contributor {contributor}",
        "capture_timestamp": asset.captured_at.isoformat(),
        "acquisition_timestamp": asset.acquired_at.isoformat(),
        "latitude": asset.latitude,
        "longitude": asset.longitude,
        "coordinate_accuracy_m": coordinate_uncertainty_m,
        "heading_degrees": asset.compass_angle,
        "sequence_id": asset.sequence_id,
        "capture_run_id": asset.sequence_id or f"mapillary-image-{asset.mapillary_image_id}",
        "image_sha256": asset.normalized_sha256,
        "perceptual_hash": asset.perceptual_hash,
        "perceptual_hash_algorithm": "dhash64-v1",
        "width_px": asset.width_px,
        "height_px": asset.height_px,
        "mime_type": mime_type,
        "asset_type": "street_level",
        "province_code": asset.province_code,
        "urbanicity": "unknown",
        "road_class": "unknown",
        "scene_type": "unknown",
        "road_context": "unknown",
        "terrain_class": "unknown",
        "vegetation_state": "unknown",
        "season": "unknown",
        "license_identifier": asset.license_identifier,
        "license_url": asset.license_url,
        "attribution_text": asset.attribution_text,
        "source_policy_decision": "GO_WITH_ATTRIBUTION",
        "commercial_use_decision": "GO_WITH_ATTRIBUTION",
        "derivative_index_decision": "GO_WITH_ATTRIBUTION",
        "personal_data_blur_state": "BLURRED_AT_SOURCE",
        "deletion_revocation_state": {
            "status": "ACTIVE",
            "last_checked_at": asset.acquired_at.isoformat(),
            "request_id": None,
        },
        "provenance_receipt": {
            "receipt_id": f"mapillary-receipt-{asset.mapillary_image_id}",
            "source_policy_version": MAPILLARY_POLICY_VERSION,
            "evidence_date": MAPILLARY_POLICY_EVIDENCE_DATE,
            "acquisition_method": "authorized_api",
            "receipt_sha256": sha256_json(
                {
                    "mapillary_image_id": asset.mapillary_image_id,
                    "raw_sha256": asset.raw_sha256,
                    "normalized_sha256": asset.normalized_sha256,
                    "source_policy_sha256": source_policy_sha256,
                    "split": asset.split,
                }
            ),
        },
        # Phase 3B1 requires one sequence to remain in one logical split group.
        # The geographic sampling cell is retained in the Phase 3B3 split lock.
        "spatial_split": (
            "mapillary-lineage-"
            + sha256_json({"sequence": asset.sequence_id or f"image:{asset.mapillary_image_id}"})[
                :24
            ]
        ),
        "role": "holdout" if holdout else "train",
        "tier": "tier_3_locked_holdout" if holdout else "tier_2_dense_corridor",
        "acquisition_ready": True,
        "source_tile_id": None,
        "parent_source_asset_id": None,
    }


def _contained_file(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Mapillary asset path escapes acquisition root") from exc
    return _regular_file(candidate)


def _regular_file(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError("required artifact must be a regular non-symlink file")
    return path


def _validated_mime_type(mime_type: str) -> Phase3B1MimeType:
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/tiff"}:
        raise ValueError("unsupported Phase 3B1 image MIME type")
    return cast(Phase3B1MimeType, mime_type)


def _extension(mime_type: Phase3B1MimeType) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/tiff": ".tiff",
    }
    return mapping[mime_type]


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with source.open("rb") as read_handle, temporary.open("xb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle, length=1024 * 1024)
            write_handle.flush()
            os.fsync(write_handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "MAPILLARY_CORPUS_VERSION",
    "MAPILLARY_DESCRIPTOR_VERSION",
    "MAPILLARY_INDEX_VERSION",
    "MAPILLARY_POLICY_EVIDENCE_DATE",
    "MAPILLARY_POLICY_VERSION",
    "MAPILLARY_SOURCE_ID",
    "MAPILLARY_SOURCE_NAME",
    "MapillaryManifestBuildResult",
    "attribution_manifest_hash",
    "attribution_records",
    "ingest_with_phase3b1",
    "materialize_phase3b1_manifest",
]
