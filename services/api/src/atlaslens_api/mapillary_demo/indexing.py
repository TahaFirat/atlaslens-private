"""Private-demo MegaLoc execution and atomic Phase 3B1 FAISS publication."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4

from atlaslens_api.corpus_index.artifacts import (
    atomic_write_json,
    read_json,
    sha256_path,
)
from atlaslens_api.corpus_index.descriptors import (
    DescriptorBuildResult,
    DescriptorDataset,
    build_descriptors,
)
from atlaslens_api.corpus_index.errors import ArtifactIntegrityError
from atlaslens_api.corpus_index.index import PublishedCorpusIndex, build_index
from atlaslens_api.corpus_index.models import (
    AssetProvenance,
    DescriptorAsset,
    DescriptorSpec,
    FloatMatrix,
    RuntimeKind,
)
from atlaslens_api.corpus_pipeline.ingestion import IngestionResult
from atlaslens_api.corpus_pipeline.safety import resolve_contained_file
from atlaslens_api.mapillary_demo.manifest import (
    MAPILLARY_INDEX_VERSION,
    MAPILLARY_SOURCE_ID,
    attribution_records,
)
from atlaslens_api.mapillary_demo.selection import LockedMapillarySplit
from atlaslens_api.megaloc_adapter.models import (
    MEGALOC_DESCRIPTOR_DIMENSION,
    MEGALOC_PREPROCESSING_ID,
    MEGALOC_PROVIDER_ID,
)

MEGALOC_PHASE3B2_ARTIFACT_SHA256 = (
    "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
)
PRIVATE_DEMO_AUTHORIZATION_ID = "phase3b3-mapillary-private-demo-v1"
_BUNDLE_SCHEMA = "atlaslens-mapillary-demo-index-v1"
_ATTRIBUTION_SCHEMA = "atlaslens-mapillary-attribution-v1"
_FAISS_BACKEND = "faiss-flat-ip-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PrivateDemoMegaLocExecutor(Protocol):
    @property
    def spec(self) -> DescriptorSpec: ...

    def describe_batch_for_private_demo(
        self,
        locators: Sequence[Path],
        *,
        authorization_id: str,
        source_policy_sha256: str,
    ) -> FloatMatrix: ...


class MapillaryPrivateDemoDescriptorProvider:
    """Non-production wrapper around the explicit Phase 3B3 MegaLoc seam."""

    def __init__(
        self,
        executor: PrivateDemoMegaLocExecutor,
        *,
        source_policy_sha256: str,
    ) -> None:
        _require_sha256(source_policy_sha256, "source_policy_sha256")
        spec = executor.spec
        if (
            spec.provider_id != MEGALOC_PROVIDER_ID
            or spec.dimension != MEGALOC_DESCRIPTOR_DIMENSION
            or spec.artifact_sha256 != MEGALOC_PHASE3B2_ARTIFACT_SHA256
            or spec.preprocessing_version != MEGALOC_PREPROCESSING_ID
        ):
            raise ValueError("private demo requires the exact Phase 3B2 MegaLoc artifact")
        self._executor = executor
        self._source_policy_sha256 = source_policy_sha256

    @property
    def spec(self) -> DescriptorSpec:
        return self._executor.spec

    @property
    def runtime_kind(self) -> RuntimeKind:
        # The wrapper is deliberately ineligible for ProductionDescriptorRegistry.
        return "test_only"

    def describe_batch(self, locators: Sequence[Path]) -> FloatMatrix:
        return self._executor.describe_batch_for_private_demo(
            locators,
            authorization_id=PRIVATE_DEMO_AUTHORIZATION_ID,
            source_policy_sha256=self._source_policy_sha256,
        )


@dataclass(frozen=True, slots=True)
class MapillaryAttribution:
    asset_id: str
    mapillary_image_id: str
    source_page_url: str
    attribution_text: str
    license_identifier: str
    license_url: str
    creator_id: str | None
    sequence_id: str | None
    captured_at: str
    latitude: float
    longitude: float
    city: str
    province: str
    province_code: str
    split: str
    raw_sha256: str
    normalized_sha256: str
    perceptual_hash: str

    @classmethod
    def from_json(cls, value: object) -> MapillaryAttribution:
        if not isinstance(value, dict):
            raise ArtifactIntegrityError("Mapillary attribution record is invalid")
        required = (
            "asset_id",
            "mapillary_image_id",
            "source_page_url",
            "attribution_text",
            "license_identifier",
            "license_url",
            "captured_at",
            "city",
            "province",
            "province_code",
            "split",
        )
        if any(not isinstance(value.get(field), str) for field in required):
            raise ArtifactIntegrityError("Mapillary attribution text is invalid")
        creator = value.get("creator_id")
        sequence = value.get("sequence_id")
        latitude = value.get("latitude")
        longitude = value.get("longitude")
        if creator is not None and not isinstance(creator, str):
            raise ArtifactIntegrityError("Mapillary attribution creator is invalid")
        if sequence is not None and not isinstance(sequence, str):
            raise ArtifactIntegrityError("Mapillary attribution sequence is invalid")
        if (
            isinstance(latitude, bool)
            or not isinstance(latitude, int | float)
            or isinstance(longitude, bool)
            or not isinstance(longitude, int | float)
        ):
            raise ArtifactIntegrityError("Mapillary attribution coordinates are invalid")
        source_page_url = str(value["source_page_url"])
        license_url = str(value["license_url"])
        _validate_source_url(source_page_url)
        if license_url != "https://creativecommons.org/licenses/by-sa/4.0/":
            raise ArtifactIntegrityError("Mapillary license URL is incompatible")
        if value.get("source_partition") != MAPILLARY_SOURCE_ID:
            raise ArtifactIntegrityError("Mapillary source partition is invalid")
        if value.get("license_identifier") != "CC-BY-SA-4.0":
            raise ArtifactIntegrityError("Mapillary license identifier is incompatible")
        image_id = str(value["mapillary_image_id"])
        asset_id = str(value["asset_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", image_id) or asset_id != (
            f"mapillary-{image_id}"
        ):
            raise ArtifactIntegrityError("Mapillary stable image identity is invalid")
        split = str(value["split"])
        if split not in {"reference", "holdout"}:
            raise ArtifactIntegrityError("Mapillary attribution split is invalid")
        if value.get("reconciliation_state") != "active":
            raise ArtifactIntegrityError("Mapillary attribution is not active")
        raw_sha256 = value.get("raw_sha256")
        normalized_sha256 = value.get("normalized_sha256")
        perceptual_hash = value.get("perceptual_hash")
        if (
            not isinstance(raw_sha256, str)
            or not _SHA256.fullmatch(raw_sha256)
            or not isinstance(normalized_sha256, str)
            or not _SHA256.fullmatch(normalized_sha256)
            or not isinstance(perceptual_hash, str)
            or re.fullmatch(r"[0-9a-f]{16,128}", perceptual_hash) is None
        ):
            raise ArtifactIntegrityError("Mapillary content hashes are invalid")
        if not -90.0 <= float(latitude) <= 90.0 or not -180.0 <= float(longitude) <= 180.0:
            raise ArtifactIntegrityError("Mapillary attribution coordinates are out of bounds")
        try:
            captured = datetime.fromisoformat(str(value["captured_at"]))
        except ValueError as exc:
            raise ArtifactIntegrityError("Mapillary capture timestamp is invalid") from exc
        if captured.tzinfo is None or captured.utcoffset() is None:
            raise ArtifactIntegrityError("Mapillary capture timestamp lacks a timezone")
        return cls(
            asset_id=asset_id,
            mapillary_image_id=image_id,
            source_page_url=source_page_url,
            attribution_text=str(value["attribution_text"]),
            license_identifier=str(value["license_identifier"]),
            license_url=license_url,
            creator_id=creator,
            sequence_id=sequence,
            captured_at=str(value["captured_at"]),
            latitude=float(latitude),
            longitude=float(longitude),
            city=str(value["city"]),
            province=str(value["province"]),
            province_code=str(value["province_code"]),
            split=split,
            raw_sha256=raw_sha256,
            normalized_sha256=normalized_sha256,
            perceptual_hash=perceptual_hash,
        )


@dataclass(frozen=True, slots=True)
class MapillaryDemoSearchHit:
    rank: int
    cosine_distance: float
    cosine_similarity: float
    reference_asset_id: str
    mapillary_image_id: str
    latitude: float
    longitude: float
    city: str
    province: str
    province_code: str
    creator_id: str | None
    captured_at: str
    attribution_text: str
    source_page_url: str
    license_identifier: str
    license_url: str
    experimental_demo: bool = True


class PublishedMapillaryDemoIndex:
    """Verified demo bundle around the existing PublishedCorpusIndex."""

    def __init__(
        self,
        directory: Path,
        metadata: dict[str, Any],
        index: PublishedCorpusIndex,
        attribution: dict[str, MapillaryAttribution],
    ) -> None:
        self.directory = directory
        self.metadata = metadata
        self.index = index
        self.attribution = attribution

    @property
    def size(self) -> int:
        return self.index.size

    @property
    def city(self) -> str:
        return str(self.metadata["city"])

    @property
    def province(self) -> str:
        return str(self.metadata["province"])

    @property
    def source_policy_sha256(self) -> str:
        return str(self.metadata["source_policy_sha256"])

    @property
    def split_lock_sha256(self) -> str:
        return str(self.metadata["selection_lock_sha256"])

    def search(self, query: object, *, top_k: int = 5) -> tuple[MapillaryDemoSearchHit, ...]:
        hits = self.index.search(query, top_k=top_k)
        enriched: list[MapillaryDemoSearchHit] = []
        for hit in hits:
            record = self.attribution.get(hit.reference_asset_id)
            if record is None or record.split != "reference":
                raise ArtifactIntegrityError(
                    "reference attribution is missing or not reference-only"
                )
            enriched.append(
                MapillaryDemoSearchHit(
                    rank=hit.rank,
                    cosine_distance=hit.cosine_distance,
                    cosine_similarity=1.0 - hit.cosine_distance,
                    reference_asset_id=hit.reference_asset_id,
                    mapillary_image_id=record.mapillary_image_id,
                    latitude=record.latitude,
                    longitude=record.longitude,
                    city=record.city,
                    province=record.province,
                    province_code=record.province_code,
                    creator_id=record.creator_id,
                    captured_at=record.captured_at,
                    attribution_text=record.attribution_text,
                    source_page_url=record.source_page_url,
                    license_identifier=record.license_identifier,
                    license_url=record.license_url,
                )
            )
        return tuple(enriched)

    @classmethod
    def open(
        cls,
        directory: Path,
        *,
        expected_source_policy_sha256: str | None = None,
        expected_selection_lock_sha256: str | None = None,
    ) -> PublishedMapillaryDemoIndex:
        root = directory.resolve()
        metadata_path = _contained_regular_file(root, "metadata.json")
        marker_path = _contained_regular_file(root, "PUBLISHED.json")
        try:
            metadata_value = read_json(metadata_path)
            marker_value = read_json(marker_path)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ArtifactIntegrityError("Mapillary demo publication is unreadable") from exc
        if not isinstance(metadata_value, dict) or not isinstance(marker_value, dict):
            raise ArtifactIntegrityError("Mapillary demo publication metadata is invalid")
        metadata = cast(dict[str, Any], metadata_value)
        if (
            metadata.get("schema") != _BUNDLE_SCHEMA
            or metadata.get("technical_demo_only") is not True
            or metadata.get("index_version") != MAPILLARY_INDEX_VERSION
            or metadata.get("model_artifact_sha256") != MEGALOC_PHASE3B2_ARTIFACT_SHA256
            or metadata.get("descriptor_dimension") != MEGALOC_DESCRIPTOR_DIMENSION
            or metadata.get("preprocessing_version") != MEGALOC_PREPROCESSING_ID
            or marker_value.get("schema") != _BUNDLE_SCHEMA
            or marker_value.get("metadata_sha256") != sha256_path(metadata_path)
        ):
            raise ArtifactIntegrityError("Mapillary demo publication compatibility failed")
        source_policy_hash = _metadata_hash(metadata, "source_policy_sha256")
        selection_lock_hash = _metadata_hash(metadata, "selection_lock_sha256")
        phase3b1_split_hash = _metadata_hash(metadata, "phase3b1_split_lock_sha256")
        if (
            expected_source_policy_sha256 is not None
            and source_policy_hash != expected_source_policy_sha256
        ):
            raise ArtifactIntegrityError("Mapillary source policy is incompatible")
        if (
            expected_selection_lock_sha256 is not None
            and selection_lock_hash != expected_selection_lock_sha256
        ):
            raise ArtifactIntegrityError("Mapillary selection lock is incompatible")

        attribution_name = metadata.get("attribution_file")
        index_name = metadata.get("base_index_directory")
        if not isinstance(attribution_name, str) or not isinstance(index_name, str):
            raise ArtifactIntegrityError("Mapillary bundle artifact names are invalid")
        attribution_path = _contained_regular_file(root, attribution_name)
        if sha256_path(attribution_path) != metadata.get("attribution_sha256"):
            raise ArtifactIntegrityError("Mapillary attribution checksum mismatch")
        base_metadata_path = _contained_regular_file(root, f"{index_name}/metadata.json")
        if sha256_path(base_metadata_path) != metadata.get("base_index_metadata_sha256"):
            raise ArtifactIntegrityError("base index metadata checksum mismatch")
        try:
            attribution_value = read_json(attribution_path)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ArtifactIntegrityError("Mapillary attribution is unreadable") from exc
        if (
            not isinstance(attribution_value, dict)
            or attribution_value.get("schema") != _ATTRIBUTION_SCHEMA
            or attribution_value.get("source_policy_sha256") != source_policy_hash
            or attribution_value.get("selection_lock_sha256") != selection_lock_hash
            or not isinstance(attribution_value.get("records"), list)
        ):
            raise ArtifactIntegrityError("Mapillary attribution manifest is invalid")
        records = tuple(
            MapillaryAttribution.from_json(value)
            for value in cast(list[object], attribution_value["records"])
        )
        by_id = {record.asset_id: record for record in records}
        if len(by_id) != len(records):
            raise ArtifactIntegrityError("Mapillary attribution asset IDs are not unique")

        index_root = (root / index_name).resolve()
        try:
            index_root.relative_to(root)
        except ValueError as exc:
            raise ArtifactIntegrityError("base index path escapes demo publication") from exc
        index = PublishedCorpusIndex.open(
            index_root,
            expected_source_policy_hash=source_policy_hash,
            expected_locked_holdout_hash=selection_lock_hash,
            expected_split_lock_hash=phase3b1_split_hash,
        )
        if (
            index.backend != _FAISS_BACKEND
            or index.metadata.get("index_version") != MAPILLARY_INDEX_VERSION
            or index.spec.artifact_sha256 != MEGALOC_PHASE3B2_ARTIFACT_SHA256
            or index.spec.dimension != MEGALOC_DESCRIPTOR_DIMENSION
            or index.spec.preprocessing_version != MEGALOC_PREPROCESSING_ID
            or index.metadata.get("manifest_hash") != metadata.get("manifest_sha256")
            or index.metadata.get("build_hash") != metadata.get("base_index_build_hash")
        ):
            raise ArtifactIntegrityError("base Mapillary FAISS index is incompatible")
        reference_ids = {asset.asset_id for asset in index.provenance}
        attributed_reference_ids = {
            record.asset_id for record in records if record.split == "reference"
        }
        if reference_ids != attributed_reference_ids:
            raise ArtifactIntegrityError("reference attribution inventory is incomplete")
        if index.size != metadata.get("reference_count"):
            raise ArtifactIntegrityError("Mapillary reference count is inconsistent")
        return cls(root, metadata, index, by_id)


@dataclass(frozen=True, slots=True)
class MapillaryDemoIndexBuildResult:
    publication: PublishedMapillaryDemoIndex
    reference_descriptors: DescriptorBuildResult
    holdout_descriptors: DescriptorBuildResult


def build_mapillary_demo_index(
    ingestion: IngestionResult,
    split: LockedMapillarySplit,
    executor: PrivateDemoMegaLocExecutor,
    *,
    corpus_root: Path,
    descriptor_work_root: Path,
    descriptor_output_root: Path,
    output_dir: Path,
    batch_size: int = 8,
) -> MapillaryDemoIndexBuildResult:
    """Reuse Phase 3B1 descriptor/checkpoint/FAISS publication end to end."""

    split.verify()
    if ingestion.corpus.source_policy_sha256 != split.source_policy_sha256:
        raise ValueError("ingested corpus source policy does not match the locked acquisition")
    if ingestion.corpus.manifest_sha256 == "" or not _SHA256.fullmatch(
        ingestion.corpus.manifest_sha256
    ):
        raise ValueError("ingested corpus manifest hash is invalid")
    selected_ids = {asset.asset_id for asset in split.all_assets}
    ingested_ids = {asset.asset_id for asset in ingestion.corpus.assets}
    if ingested_ids != selected_ids:
        raise ValueError("Phase 3B1 corpus does not exactly match the locked split")
    if ingestion.split_lock.lock_sha256 == split.lock_sha256:
        raise ValueError("logical selection lock and Phase 3B1 split lock must be independent")

    provider = MapillaryPrivateDemoDescriptorProvider(
        executor,
        source_policy_sha256=split.source_policy_sha256,
    )
    references, holdout = _descriptor_assets(ingestion, corpus_root=corpus_root)
    reference_result = build_descriptors(
        references,
        provider,
        work_dir=descriptor_work_root / "reference",
        output_dir=descriptor_output_root / "reference",
        manifest_hash=ingestion.corpus.manifest_sha256,
        source_policy_hash=split.source_policy_sha256,
        batch_size=batch_size,
    )
    holdout_result = build_descriptors(
        holdout,
        provider,
        work_dir=descriptor_work_root / "holdout",
        output_dir=descriptor_output_root / "holdout",
        manifest_hash=ingestion.corpus.manifest_sha256,
        source_policy_hash=split.source_policy_sha256,
        batch_size=batch_size,
    )
    publication = _publish_bundle(
        reference_result.dataset,
        holdout_result.dataset,
        split,
        phase3b1_split_lock_sha256=ingestion.split_lock.lock_sha256,
        output_dir=output_dir,
    )
    return MapillaryDemoIndexBuildResult(
        publication=publication,
        reference_descriptors=reference_result,
        holdout_descriptors=holdout_result,
    )


def _descriptor_assets(
    ingestion: IngestionResult,
    *,
    corpus_root: Path,
) -> tuple[tuple[DescriptorAsset, ...], tuple[DescriptorAsset, ...]]:
    reference: list[DescriptorAsset] = []
    holdout: list[DescriptorAsset] = []
    for asset in sorted(ingestion.corpus.assets, key=lambda item: item.asset_id):
        provenance = AssetProvenance(
            asset_id=asset.asset_id,
            source_id=asset.source_id,
            rights_decision=asset.rights_decision,
            license_record_id=asset.provenance_receipt_id,
            provenance_summary=(
                f"Mapillary {asset.source_asset_id}; {asset.license_identifier}; "
                f"receipt {asset.provenance_receipt_id}"
            ),
            content_sha256=asset.image_sha256,
            split="holdout" if asset.role == "holdout" else "reference",
            rights_validated=asset.permission_state == "validated",
            revoked=asset.revoked,
            province=asset.province_code,
            latitude=asset.latitude,
            longitude=asset.longitude,
            contributor_id=asset.contributor_or_owner,
            capture_run_id=asset.capture_run_id,
            sequence_id=asset.sequence_id,
            sampling_cell=asset.sampling_cell,
        )
        descriptor_asset = DescriptorAsset(
            locator=resolve_contained_file(corpus_root, asset.file_locator),
            provenance=provenance,
        )
        (holdout if asset.role == "holdout" else reference).append(descriptor_asset)
    if not reference or not holdout:
        raise ValueError("Mapillary demo requires both reference and holdout descriptors")
    return tuple(reference), tuple(holdout)


def _publish_bundle(
    references: DescriptorDataset,
    holdout: DescriptorDataset,
    split: LockedMapillarySplit,
    *,
    phase3b1_split_lock_sha256: str,
    output_dir: Path,
) -> PublishedMapillaryDemoIndex:
    _require_sha256(phase3b1_split_lock_sha256, "phase3b1_split_lock_sha256")
    output = output_dir.resolve()
    if output.exists():
        return PublishedMapillaryDemoIndex.open(
            output,
            expected_source_policy_sha256=split.source_policy_sha256,
            expected_selection_lock_sha256=split.lock_sha256,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        stage.mkdir()
        base_index = build_index(
            references,
            output_dir=stage / "index",
            backend="faiss",
            index_version=MAPILLARY_INDEX_VERSION,
            source_policy_hash=split.source_policy_sha256,
            shard_size=10_000,
            locked_holdout_hash=split.lock_sha256,
            split_lock_hash=phase3b1_split_lock_sha256,
        )
        attribution_payload = {
            "schema": _ATTRIBUTION_SCHEMA,
            "source_policy_sha256": split.source_policy_sha256,
            "selection_lock_sha256": split.lock_sha256,
            "records": list(attribution_records(split)),
        }
        attribution_path = stage / "mapillary-attribution.json"
        atomic_write_json(attribution_path, attribution_payload)
        metadata = {
            "schema": _BUNDLE_SCHEMA,
            "technical_demo_only": True,
            "production_approved": False,
            "index_version": MAPILLARY_INDEX_VERSION,
            "base_index_directory": "index",
            "base_index_metadata_sha256": sha256_path(stage / "index" / "metadata.json"),
            "base_index_build_hash": base_index.metadata["build_hash"],
            "base_index_artifacts": [item.to_json() for item in base_index.artifact_inventory()],
            "descriptor_spec": references.spec.to_json(),
            "descriptor_dimension": references.spec.dimension,
            "model_artifact_sha256": references.spec.artifact_sha256,
            "preprocessing_version": references.spec.preprocessing_version,
            "manifest_sha256": references.manifest_hash,
            "source_policy_sha256": split.source_policy_sha256,
            "selection_lock_sha256": split.lock_sha256,
            "phase3b1_split_lock_sha256": phase3b1_split_lock_sha256,
            "reference_descriptor_metadata_sha256": sha256_path(
                references.directory / "metadata.json"
            ),
            "reference_descriptor_vectors_sha256": references.metadata["vectors_sha256"],
            "holdout_descriptor_metadata_sha256": sha256_path(holdout.directory / "metadata.json"),
            "holdout_descriptor_vectors_sha256": holdout.metadata["vectors_sha256"],
            "attribution_file": attribution_path.name,
            "attribution_sha256": sha256_path(attribution_path),
            "aoi_id": split.decision.aoi_id,
            "aoi_version": split.decision.aoi_version,
            "city": split.decision.city,
            "province": split.decision.province,
            "reference_count": references.count,
            "holdout_count": holdout.count,
        }
        metadata_path = stage / "metadata.json"
        atomic_write_json(metadata_path, metadata)
        atomic_write_json(
            stage / "PUBLISHED.json",
            {"schema": _BUNDLE_SCHEMA, "metadata_sha256": sha256_path(metadata_path)},
        )
        os.replace(stage, output)
    except OSError as exc:
        raise ArtifactIntegrityError(
            "Mapillary demo index could not be published atomically"
        ) from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return PublishedMapillaryDemoIndex.open(
        output,
        expected_source_policy_sha256=split.source_policy_sha256,
        expected_selection_lock_sha256=split.lock_sha256,
    )


def _contained_regular_file(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ArtifactIntegrityError("Mapillary bundle path escapes publication") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise ArtifactIntegrityError("Mapillary bundle artifact is not a regular file")
    return candidate


def _metadata_hash(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ArtifactIntegrityError(f"Mapillary metadata {key} is invalid")
    return value


def _validate_source_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"mapillary.com", "www.mapillary.com"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ArtifactIntegrityError("Mapillary source URL is invalid")
    sensitive = {"access_token", "token", "authorization", "signature", "sig"}
    if any(
        key.casefold() in sensitive
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
    ):
        raise ArtifactIntegrityError("credential-shaped attribution URL is forbidden")


def _require_sha256(value: str, field: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


__all__ = [
    "MEGALOC_PHASE3B2_ARTIFACT_SHA256",
    "MapillaryAttribution",
    "MapillaryDemoIndexBuildResult",
    "MapillaryDemoSearchHit",
    "MapillaryPrivateDemoDescriptorProvider",
    "PRIVATE_DEMO_AUTHORIZATION_ID",
    "PrivateDemoMegaLocExecutor",
    "PublishedMapillaryDemoIndex",
    "build_mapillary_demo_index",
]
