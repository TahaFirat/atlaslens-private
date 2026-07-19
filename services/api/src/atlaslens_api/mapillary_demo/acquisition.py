"""Metadata-first coverage audit and bounded private Mapillary acquisition."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ValidationError

from .client import MapillaryClient, RemoteImage
from .errors import MapillaryDemoError, MapillaryLimitError, MapillarySafetyError
from .models import (
    MAPILLARY_PILOT_ROOT,
    AcquiredImage,
    AcquisitionCheckpoint,
    AcquisitionManifest,
    AcquisitionPlan,
    AoiCatalog,
    AreaOfInterest,
    CleanupReport,
    CoverageAudit,
    CoverageSummary,
    ImageMetadata,
)

_MIN_C_FREE_BYTES = 35 * 1024 * 1024 * 1024
_MIN_D_FREE_BYTES = 8 * 1024 * 1024 * 1024
_DESCRIPTOR_DIMENSION = 8_448
_MAX_IMAGE_PIXELS = 100_000_000
_MANIFEST_PATH = PurePosixPath("metadata/acquisition-manifest.json")
_CHECKPOINT_PATH = PurePosixPath("checkpoints/acquisition.json")


def canonical_json_bytes(value: object) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    if path.is_symlink() or not path.is_file():
        raise MapillarySafetyError("mapillary_file_invalid")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise MapillaryLimitError("mapillary_file_byte_cap_reached")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise MapillaryLimitError("mapillary_file_byte_cap_reached")
            digest.update(chunk)
    return digest.hexdigest()


def load_aoi_catalog(path: Path) -> AoiCatalog:
    return _load_model(path, AoiCatalog, max_bytes=1024 * 1024)


def load_coverage_audit(path: Path) -> CoverageAudit:
    return _load_model(path, CoverageAudit, max_bytes=4 * 1024 * 1024)


def load_acquisition_plan(path: Path) -> AcquisitionPlan:
    return _load_model(path, AcquisitionPlan, max_bytes=1024 * 1024)


def load_acquisition_manifest(path: Path) -> AcquisitionManifest:
    return _load_model(path, AcquisitionManifest, max_bytes=64 * 1024 * 1024)


def _load_model[ModelT: BaseModel](path: Path, model: type[ModelT], *, max_bytes: int) -> ModelT:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise MapillarySafetyError("mapillary_json_file_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return model.model_validate(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise MapillarySafetyError("mapillary_json_file_invalid") from exc


def _default_disk_free_bytes(drive: str) -> int:
    return shutil.disk_usage(drive).free


def _safe_root(root: Path, approved_root: Path) -> Path:
    if root.exists() and root.is_symlink():
        raise MapillarySafetyError("mapillary_pilot_root_invalid")
    selected = root.resolve(strict=False)
    approved = approved_root.resolve(strict=False)
    if selected != approved:
        raise MapillarySafetyError("mapillary_pilot_root_outside_approved_boundary")
    return selected


def _contained_path(root: Path, relative: PurePosixPath | str) -> Path:
    item = PurePosixPath(relative)
    if item.is_absolute() or not item.parts or ".." in item.parts:
        raise MapillarySafetyError("mapillary_relative_path_invalid")
    destination = root.joinpath(*item.parts)
    if not destination.resolve(strict=False).is_relative_to(root):
        raise MapillarySafetyError("mapillary_path_outside_pilot_root")
    current = root
    for part in item.parts[:-1]:
        current /= part
        if current.exists() and current.is_symlink():
            raise MapillarySafetyError("mapillary_path_symlink_refused")
    if destination.exists() and destination.is_symlink():
        raise MapillarySafetyError("mapillary_path_symlink_refused")
    return destination


def atomic_write_bytes(root: Path, relative: PurePosixPath | str, payload: bytes) -> Path:
    destination = _contained_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def atomic_write_model(
    root: Path, relative: PurePosixPath | str, value: BaseModel | Mapping[str, object]
) -> Path:
    return atomic_write_bytes(root, relative, canonical_json_bytes(value))


def _metadata_projection(item: ImageMetadata) -> dict[str, object]:
    return item.model_dump(mode="json")


def _spatial_cell(item: ImageMetadata, size: float) -> str:
    longitude, latitude = item.computed_geometry.coordinates
    return f"{math.floor(latitude / size)}:{math.floor(longitude / size)}"


def _compass_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    directions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return directions[int((value + 22.5) // 45.0) % 8]


def _coverage_summary(
    aoi: AreaOfInterest,
    rows: Sequence[ImageMetadata],
    *,
    max_image_bytes: int,
    rejected_item_count: int,
) -> CoverageSummary:
    ordered = sorted(rows, key=lambda item: item.mapillary_image_id)
    sequences = {item.sequence_id for item in ordered if item.sequence_id}
    contributors = {item.creator_id for item in ordered if item.creator_id}
    years = Counter(str(item.captured_at.year) for item in ordered)
    cells = {_spatial_cell(item, aoi.spatial_cell_size_degrees) for item in ordered}
    compass = Counter(_compass_bin(item.compass_angle) for item in ordered)
    image_count = len(ordered)
    selected_estimate = min(image_count, 1_600)
    descriptor_bytes = selected_estimate * _DESCRIPTOR_DIMENSION * 4
    metadata_hash = canonical_sha256([_metadata_projection(item) for item in ordered])
    sequence_rows: dict[str, list[ImageMetadata]] = defaultdict(list)
    for item in ordered:
        if item.sequence_id:
            sequence_rows[item.sequence_id].append(item)
    track_distance_m = 0.0
    for sequence in sequence_rows.values():
        points = sorted(
            sequence,
            key=lambda item: (item.captured_at, item.mapillary_image_id),
        )
        for prior, current in zip(points, points[1:], strict=False):
            distance_m = _distance_m(prior, current)
            if 1.0 <= distance_m <= 1_000.0:
                track_distance_m += distance_m
    approximate_road_km = track_distance_m / 1_000.0
    return CoverageSummary(
        aoi_id=aoi.aoi_id,
        aoi_version=aoi.version,
        display_name=aoi.display_name,
        region_kind=aoi.region_kind,
        province=aoi.province,
        city=aoi.city,
        image_count=image_count,
        rejected_item_count=rejected_item_count,
        sequence_count=len(sequences),
        contributor_count=len(contributors),
        capture_year_distribution=dict(sorted(years.items())),
        spatial_cell_count=len(cells),
        compass_direction_bins=dict(sorted(compass.items())),
        approximate_road_km=approximate_road_km,
        approximate_images_per_road_km=(
            image_count / approximate_road_km if approximate_road_km > 0.0 else 0.0
        ),
        approximate_sequence_density=(
            len(sequences) / approximate_road_km if approximate_road_km > 0.0 else 0.0
        ),
        estimated_selected_download_bytes=selected_estimate * max_image_bytes,
        expected_descriptor_bytes=descriptor_bytes,
        expected_index_bytes=descriptor_bytes,
        eligible_for_selection=(image_count > 0 and len(cells) >= 10 and len(sequences) >= 2),
        metadata_sha256=metadata_hash,
    )


def select_pilot_aoi(regions: Sequence[CoverageSummary]) -> tuple[str | None, str]:
    urban = [
        item
        for item in regions
        if item.region_kind == "urban"
        and item.eligible_for_selection
        and item.spatial_cell_count >= 10
        and item.sequence_count >= 2
    ]
    kayseri = next((item for item in urban if item.aoi_id == "kayseri-urban-v1"), None)
    if (
        kayseri is not None
        and kayseri.image_count >= 1_000
        and kayseri.spatial_cell_count >= 10
        and kayseri.sequence_count >= 2
    ):
        return kayseri.aoi_id, "kayseri_has_at_least_1000_distributed_eligible_images"
    if not urban:
        return None, "no_eligible_urban_aoi"
    selected = sorted(
        urban,
        key=lambda item: (
            -item.image_count,
            -item.spatial_cell_count,
            -item.sequence_count,
            -item.contributor_count,
            item.aoi_id,
        ),
    )[0]
    return selected.aoi_id, "best_covered_eligible_urban_aoi"


def _distance_m(a: ImageMetadata, b: ImageMetadata) -> float:
    longitude_a, latitude_a = a.computed_geometry.coordinates
    longitude_b, latitude_b = b.computed_geometry.coordinates
    phi_a = math.radians(latitude_a)
    phi_b = math.radians(latitude_b)
    delta_phi = math.radians(latitude_b - latitude_a)
    delta_lambda = math.radians(longitude_b - longitude_a)
    haversine = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
    )
    return (
        6_371_008.8
        * 2
        * math.atan2(
            math.sqrt(haversine),
            math.sqrt(max(0.0, 1.0 - haversine)),
        )
    )


def select_acquisition_candidates(
    rows: Sequence[RemoteImage],
    aoi: AreaOfInterest,
    *,
    cap: int,
    minimum_sequence_spacing_m: float = 50.0,
) -> tuple[RemoteImage, ...]:
    """Select a deterministic, spatially distributed bounded download pool."""

    if cap <= 0:
        raise MapillaryLimitError("mapillary_candidate_cap_invalid")
    unique: dict[str, RemoteImage] = {}
    for row in rows:
        image_id = row.metadata.mapillary_image_id
        prior = unique.get(image_id)
        if prior is not None and prior.metadata != row.metadata:
            raise MapillarySafetyError("mapillary_duplicate_metadata_conflict")
        unique.setdefault(image_id, row)

    groups: dict[tuple[str, str], deque[RemoteImage]] = defaultdict(deque)
    for row in unique.values():
        cell = _spatial_cell(row.metadata, aoi.spatial_cell_size_degrees)
        sequence = row.metadata.sequence_id or f"no-sequence:{row.metadata.mapillary_image_id}"
        groups[(cell, sequence)].append(row)
    for key, bucket in tuple(groups.items()):
        groups[key] = deque(
            sorted(
                bucket,
                key=lambda item: (
                    item.metadata.captured_at,
                    item.metadata.mapillary_image_id,
                ),
            )
        )

    selected: list[RemoteImage] = []
    sequence_points: dict[str, list[ImageMetadata]] = defaultdict(list)
    active = sorted(groups)
    while active and len(selected) < cap:
        next_active: list[tuple[str, str]] = []
        progressed = False
        for key in active:
            bucket = groups[key]
            accepted: RemoteImage | None = None
            while bucket:
                candidate = bucket.popleft()
                candidate_sequence = candidate.metadata.sequence_id
                if candidate_sequence and any(
                    _distance_m(candidate.metadata, prior) < minimum_sequence_spacing_m
                    for prior in sequence_points[candidate_sequence]
                ):
                    continue
                accepted = candidate
                break
            if accepted is not None:
                selected.append(accepted)
                progressed = True
                if accepted.metadata.sequence_id:
                    sequence_points[accepted.metadata.sequence_id].append(accepted.metadata)
            if bucket:
                next_active.append(key)
            if len(selected) >= cap:
                break
        if not progressed:
            break
        active = next_active
    return tuple(selected)


class MapillaryCoverageAuditor:
    def __init__(
        self,
        client: MapillaryClient,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self._now = now

    def audit(self, catalog: AoiCatalog) -> CoverageAudit:
        """Audit all five configured AOIs without requesting any media URL."""

        starting_requests = self._client.request_count
        starting_pages = self._client.page_count
        summaries: list[CoverageSummary] = []
        for aoi in catalog.aois:
            starting_rejected = self._client.rejected_item_count
            rows = [
                remote.metadata
                for remote in self._client.iter_images(
                    aoi.tiles,
                    include_thumbnail=False,
                    item_cap=self._client.limits.metadata_item_cap,
                )
            ]
            summaries.append(
                _coverage_summary(
                    aoi,
                    rows,
                    max_image_bytes=self._client.limits.max_image_bytes,
                    rejected_item_count=(
                        self._client.rejected_item_count - starting_rejected
                    ),
                )
            )
        selected, reason = select_pilot_aoi(summaries)
        return CoverageAudit(
            catalog_version=catalog.catalog_version,
            audited_at=self._now(),
            request_count=self._client.request_count - starting_requests,
            page_count=self._client.page_count - starting_pages,
            regions=tuple(summaries),
            selected_aoi_id=selected,
            selection_reason=reason,
        )


def build_acquisition_plan(
    audit: CoverageAudit,
    catalog: AoiCatalog,
    *,
    source_policy_path: Path,
    target_reference_images: int = 1_500,
    target_holdout_images: int = 100,
    hard_request_cap: int = 2_500,
    concurrency: int = 2,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AcquisitionPlan:
    selected, _ = select_pilot_aoi(audit.regions)
    if audit.selected_aoi_id is None or selected != audit.selected_aoi_id:
        raise MapillarySafetyError("mapillary_audit_selection_invalid")
    aoi = catalog.by_id(selected)
    source_policy_hash = sha256_file(source_policy_path, max_bytes=4 * 1024 * 1024)
    return AcquisitionPlan(
        aoi_id=aoi.aoi_id,
        aoi_version=aoi.version,
        coverage_audit_sha256=canonical_sha256(audit),
        source_policy_receipt_sha256=source_policy_hash,
        target_reference_images=target_reference_images,
        target_holdout_images=target_holdout_images,
        hard_request_cap=hard_request_cap,
        concurrency=concurrency,
        created_at=now(),
    )


def _normalize_image(payload: bytes) -> tuple[bytes, int, int]:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.width * opened.height > _MAX_IMAGE_PIXELS:
                raise MapillaryLimitError("mapillary_image_pixel_cap_reached")
            opened.load()
            normalized = ImageOps.exif_transpose(opened).convert("RGB")
            normalized.thumbnail((1_024, 1_024), Image.Resampling.LANCZOS)
            width, height = normalized.size
            output = io.BytesIO()
            normalized.save(output, format="JPEG", quality=95, optimize=False, progressive=False)
            return output.getvalue(), width, height
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise MapillarySafetyError("mapillary_image_decode_invalid") from exc


def _attribution(metadata: ImageMetadata) -> str:
    creator = metadata.creator_id or "contributor unavailable"
    return f"Mapillary image {metadata.mapillary_image_id} by {creator}, CC BY-SA 4.0"


class MapillaryAcquisition:
    """Checkpointed acquisition that can only write inside the approved pilot root."""

    def __init__(
        self,
        client: MapillaryClient,
        pilot_root: Path = Path(MAPILLARY_PILOT_ROOT),
        *,
        approved_root: Path = Path(MAPILLARY_PILOT_ROOT),
        disk_free_bytes: Callable[[str], int] = _default_disk_free_bytes,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self.root = _safe_root(pilot_root, approved_root)
        self._disk_free_bytes = disk_free_bytes
        self._now = now

    def _check_disk_gates(self) -> None:
        if self._disk_free_bytes("C:\\") < _MIN_C_FREE_BYTES:
            raise MapillaryLimitError("mapillary_c_disk_gate_failed")
        if self._disk_free_bytes("D:\\") < _MIN_D_FREE_BYTES:
            raise MapillaryLimitError("mapillary_d_disk_gate_failed")

    def _initial_manifest(self, plan: AcquisitionPlan) -> AcquisitionManifest:
        return AcquisitionManifest(
            aoi_id=plan.aoi_id,
            aoi_version=plan.aoi_version,
            source_policy_receipt_sha256=plan.source_policy_receipt_sha256,
            coverage_audit_sha256=plan.coverage_audit_sha256,
            created_at=self._now(),
            request_count=0,
            page_count=0,
            downloaded_bytes=0,
            assets=(),
        )

    def _state(
        self,
        plan: AcquisitionPlan,
        *,
        resume: bool,
    ) -> tuple[AcquisitionManifest, AcquisitionCheckpoint]:
        manifest_path = _contained_path(self.root, _MANIFEST_PATH)
        checkpoint_path = _contained_path(self.root, _CHECKPOINT_PATH)
        if manifest_path.exists() != checkpoint_path.exists():
            raise MapillarySafetyError("mapillary_acquisition_state_incomplete")
        plan_hash = canonical_sha256(plan)
        if not manifest_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = self._initial_manifest(plan)
            checkpoint = AcquisitionCheckpoint(
                plan_sha256=plan_hash,
                status="in_progress",
                completed_image_ids=(),
                downloaded_bytes=0,
                request_count=0,
                page_count=0,
                updated_at=self._now(),
            )
            atomic_write_model(self.root, _MANIFEST_PATH, manifest)
            atomic_write_model(self.root, _CHECKPOINT_PATH, checkpoint)
            return manifest, checkpoint
        if not resume:
            raise MapillarySafetyError("mapillary_acquisition_resume_required")
        manifest = load_acquisition_manifest(manifest_path)
        checkpoint = _load_model(checkpoint_path, AcquisitionCheckpoint, max_bytes=4 * 1024 * 1024)
        if (
            checkpoint.plan_sha256 != plan_hash
            or manifest.aoi_id != plan.aoi_id
            or manifest.aoi_version != plan.aoi_version
            or manifest.coverage_audit_sha256 != plan.coverage_audit_sha256
            or manifest.source_policy_receipt_sha256 != plan.source_policy_receipt_sha256
            or checkpoint.downloaded_bytes != manifest.downloaded_bytes
            or set(checkpoint.completed_image_ids)
            != {asset.mapillary_image_id for asset in manifest.assets}
        ):
            raise MapillarySafetyError("mapillary_acquisition_checkpoint_mismatch")
        return manifest, checkpoint

    def acquire(
        self,
        catalog: AoiCatalog,
        audit: CoverageAudit,
        plan: AcquisitionPlan,
        *,
        resume: bool = True,
    ) -> AcquisitionManifest:
        """Acquire the single audited AOI; no all-region acquisition path exists."""

        selected, _ = select_pilot_aoi(audit.regions)
        if (
            audit.imagery_downloaded is not False
            or canonical_sha256(audit) != plan.coverage_audit_sha256
            or selected is None
            or selected != audit.selected_aoi_id
            or plan.aoi_id != selected
        ):
            raise MapillarySafetyError("mapillary_metadata_audit_required")
        aoi = catalog.by_id(plan.aoi_id)
        if aoi.version != plan.aoi_version:
            raise MapillarySafetyError("mapillary_aoi_version_mismatch")
        if self._client.limits.request_cap > plan.hard_request_cap:
            raise MapillarySafetyError("mapillary_request_cap_policy_mismatch")
        if plan.concurrency > self._client.limits.concurrency:
            raise MapillarySafetyError("mapillary_concurrency_policy_mismatch")
        self._check_disk_gates()
        manifest, _ = self._state(plan, resume=resume)
        # Acquire the bounded candidate pool so leakage and distance filters can
        # still meet the 1,500-reference/100-holdout targets without unsafe fill.
        desired = plan.hard_image_cap
        if len(manifest.assets) >= desired:
            return manifest
        completed = {asset.mapillary_image_id for asset in manifest.assets}
        starting_requests = self._client.request_count
        starting_pages = self._client.page_count
        pending: list[RemoteImage] = []
        try:
            remote_rows: list[RemoteImage] = []
            try:
                remote_rows.extend(
                    self._client.iter_images(
                        aoi.tiles,
                        include_thumbnail=True,
                        item_cap=self._client.limits.metadata_item_cap,
                    )
                )
            except MapillaryLimitError as exc:
                if exc.code != "mapillary_item_cap_reached":
                    raise
            candidate_pool = select_acquisition_candidates(
                remote_rows,
                aoi,
                cap=plan.hard_image_cap,
            )
            for remote in candidate_pool:
                if remote.metadata.mapillary_image_id in completed:
                    continue
                pending.append(remote)
                if len(pending) >= plan.concurrency:
                    manifest = self._acquire_batch(plan, manifest, pending)
                    completed.update(item.metadata.mapillary_image_id for item in pending)
                    pending = []
                if len(manifest.assets) >= desired:
                    break
            if pending and len(manifest.assets) < desired:
                manifest = self._acquire_batch(plan, manifest, pending)
            status: Literal["in_progress", "completed", "cancelled", "failed"] = (
                "completed"
            )
            failure_code = None
        except MapillaryDemoError as exc:
            status = "cancelled" if exc.code == "mapillary_operation_cancelled" else "failed"
            failure_code = exc.code
            persisted_manifest = _contained_path(self.root, _MANIFEST_PATH)
            if persisted_manifest.exists():
                manifest = load_acquisition_manifest(persisted_manifest)
            self._write_checkpoint(
                plan,
                manifest,
                status=status,
                failure_code=failure_code,
                request_count=self._client.request_count - starting_requests,
                page_count=self._client.page_count - starting_pages,
            )
            raise
        manifest = manifest.model_copy(
            update={
                "request_count": self._client.request_count - starting_requests,
                "page_count": self._client.page_count - starting_pages,
            }
        )
        atomic_write_model(self.root, _MANIFEST_PATH, manifest)
        self._write_checkpoint(
            plan,
            manifest,
            status=status,
            failure_code=failure_code,
            request_count=manifest.request_count,
            page_count=manifest.page_count,
        )
        return manifest

    def _acquire_batch(
        self,
        plan: AcquisitionPlan,
        manifest: AcquisitionManifest,
        batch: Sequence[RemoteImage],
    ) -> AcquisitionManifest:
        self._check_disk_gates()
        remaining = plan.hard_raw_byte_cap - manifest.downloaded_bytes
        if remaining <= 0:
            raise MapillaryLimitError("mapillary_raw_byte_cap_reached")
        per_image_cap = min(self._client.limits.max_image_bytes, remaining // len(batch))
        if per_image_cap <= 0:
            raise MapillaryLimitError("mapillary_raw_byte_cap_reached")

        def download(remote: RemoteImage) -> tuple[RemoteImage, bytes]:
            if remote.thumbnail_url is None:
                raise MapillarySafetyError("mapillary_thumbnail_url_missing")
            payload, _ = self._client.download_thumbnail(
                remote.thumbnail_url.get_secret_value(), max_bytes=per_image_cap
            )
            return remote, payload

        with ThreadPoolExecutor(max_workers=plan.concurrency) as executor:
            downloaded = list(executor.map(download, batch))
        assets = list(manifest.assets)
        downloaded_bytes = manifest.downloaded_bytes
        for remote, raw_payload in downloaded:
            if len(assets) >= plan.hard_image_cap:
                raise MapillaryLimitError("mapillary_image_cap_reached")
            downloaded_bytes += len(raw_payload)
            if downloaded_bytes > plan.hard_raw_byte_cap:
                raise MapillaryLimitError("mapillary_raw_byte_cap_reached")
            normalized, width, height = _normalize_image(raw_payload)
            image_id = remote.metadata.mapillary_image_id
            relative = PurePosixPath("imagery") / f"{image_id}.jpg"
            destination = _contained_path(self.root, relative)
            normalized_hash = hashlib.sha256(normalized).hexdigest()
            if destination.exists():
                existing_hash = sha256_file(
                    destination,
                    max_bytes=self._client.limits.max_image_bytes,
                )
                if existing_hash != normalized_hash:
                    raise MapillarySafetyError("mapillary_existing_image_hash_mismatch")
            else:
                atomic_write_bytes(self.root, relative, normalized)
            acquired = AcquiredImage(
                mapillary_image_id=image_id,
                computed_geometry=remote.metadata.computed_geometry,
                captured_at=remote.metadata.captured_at,
                compass_angle=remote.metadata.compass_angle,
                sequence_id=remote.metadata.sequence_id,
                creator_id=remote.metadata.creator_id,
                width_px=width,
                height_px=height,
                source_page_url=f"https://www.mapillary.com/app/?pKey={image_id}",
                attribution_text=_attribution(remote.metadata),
                acquired_at=self._now(),
                raw_sha256=hashlib.sha256(raw_payload).hexdigest(),
                normalized_sha256=normalized_hash,
                relative_path=relative.as_posix(),
                byte_size=len(raw_payload),
                normalized_byte_size=len(normalized),
            )
            assets.append(acquired)
            manifest = manifest.model_copy(
                update={"assets": tuple(assets), "downloaded_bytes": downloaded_bytes}
            )
            atomic_write_model(self.root, _MANIFEST_PATH, manifest)
            self._write_checkpoint(plan, manifest, status="in_progress")
        return manifest

    def _write_checkpoint(
        self,
        plan: AcquisitionPlan,
        manifest: AcquisitionManifest,
        *,
        status: Literal["in_progress", "completed", "cancelled", "failed"],
        failure_code: str | None = None,
        request_count: int | None = None,
        page_count: int | None = None,
    ) -> None:
        checkpoint = AcquisitionCheckpoint(
            plan_sha256=canonical_sha256(plan),
            status=status,
            completed_image_ids=tuple(
                sorted(asset.mapillary_image_id for asset in manifest.assets)
            ),
            downloaded_bytes=manifest.downloaded_bytes,
            request_count=(self._client.request_count if request_count is None else request_count),
            page_count=self._client.page_count if page_count is None else page_count,
            updated_at=self._now(),
            failure_code=failure_code,
        )
        atomic_write_model(self.root, _CHECKPOINT_PATH, checkpoint)

    def status(self) -> dict[str, object]:
        manifest_path = _contained_path(self.root, _MANIFEST_PATH)
        checkpoint_path = _contained_path(self.root, _CHECKPOINT_PATH)
        if not manifest_path.exists() and not checkpoint_path.exists():
            return {"state": "not_started", "image_count": 0, "downloaded_bytes": 0}
        if manifest_path.exists() != checkpoint_path.exists():
            raise MapillarySafetyError("mapillary_acquisition_state_incomplete")
        manifest = load_acquisition_manifest(manifest_path)
        checkpoint = _load_model(checkpoint_path, AcquisitionCheckpoint, max_bytes=4 * 1024 * 1024)
        return {
            "state": checkpoint.status,
            "aoi_id": manifest.aoi_id,
            "image_count": len(manifest.assets),
            "downloaded_bytes": manifest.downloaded_bytes,
            "request_count": checkpoint.request_count,
            "page_count": checkpoint.page_count,
            "failure_code": checkpoint.failure_code,
        }

    def reconcile(self) -> AcquisitionManifest:
        """Recheck remote presence and local normalized hashes without media URLs."""

        self._check_disk_gates()
        manifest = load_acquisition_manifest(_contained_path(self.root, _MANIFEST_PATH))
        updated: list[AcquiredImage] = []
        for asset in manifest.assets:
            path = _contained_path(self.root, asset.relative_path)
            if (
                not path.exists()
                or sha256_file(path, max_bytes=self._client.limits.max_image_bytes)
                != asset.normalized_sha256
            ):
                state = "hash_mismatch"
            elif not self._client.image_exists(asset.mapillary_image_id):
                state = "missing_remote"
            else:
                state = "active"
            updated.append(asset.model_copy(update={"reconciliation_state": state}))
        result = manifest.model_copy(
            update={"assets": tuple(updated), "request_count": self._client.request_count}
        )
        atomic_write_model(self.root, _MANIFEST_PATH, result)
        return result

    def cleanup(
        self,
        *,
        retain_image_ids: Iterable[str] = (),
        execute: bool = False,
        validation_receipt: Mapping[str, object] | None = None,
    ) -> CleanupReport:
        """Delete only inventoried pilot images, after a mandatory dry-run/receipt gate."""

        manifest_path = _contained_path(self.root, _MANIFEST_PATH)
        manifest = load_acquisition_manifest(manifest_path)
        retained = frozenset(retain_image_ids)
        known = {asset.mapillary_image_id for asset in manifest.assets}
        if len(retained) > 20 or not retained.issubset(known):
            raise MapillarySafetyError("mapillary_cleanup_retention_invalid")
        if execute:
            if validation_receipt is None:
                raise MapillarySafetyError("mapillary_cleanup_validation_required")
            expected_manifest_hash = sha256_file(manifest_path, max_bytes=64 * 1024 * 1024)
            if (
                validation_receipt.get("index_validated") is not True
                or validation_receipt.get("attribution_validated") is not True
                or validation_receipt.get("manifest_sha256") != expected_manifest_hash
                or not _is_sha256(validation_receipt.get("index_sha256"))
            ):
                raise MapillarySafetyError("mapillary_cleanup_validation_invalid")
        candidates = [
            asset for asset in manifest.assets if asset.mapillary_image_id not in retained
        ]
        materialized: list[Path] = []
        candidate_bytes = 0
        materialized_candidate_bytes = 0
        for asset in manifest.assets:
            path = _contained_path(self.root, asset.relative_path)
            if path.exists():
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or sha256_file(path, max_bytes=self._client.limits.max_image_bytes)
                    != asset.normalized_sha256
                ):
                    raise MapillarySafetyError("mapillary_cleanup_hash_mismatch")
                if asset.mapillary_image_id not in retained:
                    candidate_bytes += path.stat().st_size
            materialized_path = _contained_path(
                self.root,
                PurePosixPath("phase3b1-corpus/assets")
                / f"mapillary-{asset.mapillary_image_id}.jpg",
            )
            if materialized_path.exists():
                if (
                    not materialized_path.is_file()
                    or materialized_path.is_symlink()
                    or sha256_file(
                        materialized_path,
                        max_bytes=self._client.limits.max_image_bytes,
                    )
                    != asset.normalized_sha256
                ):
                    raise MapillarySafetyError("mapillary_cleanup_path_invalid")
                materialized.append(materialized_path)
                materialized_candidate_bytes += materialized_path.stat().st_size
        if not execute:
            return CleanupReport(
                mode="dry_run",
                candidate_count=len(candidates),
                candidate_bytes=candidate_bytes,
                deleted_count=0,
                deleted_bytes=0,
                retained_count=len(retained),
                remaining_bytes=_inventory_bytes(self.root, manifest.assets),
                materialized_candidate_count=len(materialized),
                materialized_candidate_bytes=materialized_candidate_bytes,
            )
        deleted_count = 0
        deleted_bytes = 0
        for asset in candidates:
            path = _contained_path(self.root, asset.relative_path)
            if path.exists():
                size = path.stat().st_size
                path.unlink()
                deleted_count += 1
                deleted_bytes += size
        materialized_deleted_count = 0
        materialized_deleted_bytes = 0
        for path in materialized:
            size = path.stat().st_size
            path.unlink()
            materialized_deleted_count += 1
            materialized_deleted_bytes += size
        for asset in manifest.assets:
            if asset.mapillary_image_id in retained:
                atomic_write_model(
                    self.root,
                    PurePosixPath("retained-sidecars") / f"{asset.mapillary_image_id}.json",
                    asset,
                )
        return CleanupReport(
            mode="executed",
            candidate_count=len(candidates),
            candidate_bytes=candidate_bytes,
            deleted_count=deleted_count,
            deleted_bytes=deleted_bytes,
            retained_count=len(retained),
            remaining_bytes=_inventory_bytes(self.root, manifest.assets),
            materialized_candidate_count=len(materialized),
            materialized_candidate_bytes=materialized_candidate_bytes,
            materialized_deleted_count=materialized_deleted_count,
            materialized_deleted_bytes=materialized_deleted_bytes,
        )


def _inventory_bytes(root: Path, assets: Sequence[AcquiredImage]) -> int:
    total = 0
    for asset in assets:
        path = _contained_path(root, asset.relative_path)
        if path.exists() and path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "MapillaryAcquisition",
    "MapillaryCoverageAuditor",
    "atomic_write_bytes",
    "atomic_write_model",
    "build_acquisition_plan",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_acquisition_manifest",
    "load_acquisition_plan",
    "load_aoi_catalog",
    "load_coverage_audit",
    "select_acquisition_candidates",
    "select_pilot_aoi",
    "sha256_file",
]
