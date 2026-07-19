from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ValidationError

from atlaslens_api.corpus_pipeline import (
    DeletionRevocationState,
    ImageIdentity,
    ManifestAsset,
    ProvenanceReceipt,
    inspect_image,
)
from atlaslens_api.corpus_pipeline.errors import AssetSafetyError
from atlaslens_api.corpus_pipeline.safety import (
    atomic_write_bytes,
    atomic_write_model,
    ensure_safe_output_root,
    read_bounded_bytes,
    resolve_contained_file,
    resolve_output_file,
)

from .errors import CaptureImportError
from .geo import interpolate_track, sample_frames, validate_track
from .models import (
    CaptureAsset,
    CaptureCheckpoint,
    CaptureImportResult,
    CaptureInspection,
    CaptureInventory,
    CaptureManifestResult,
    CapturePlan,
    CaptureSampleResult,
    CaptureSyncResult,
    PrivacyDecision,
    PrivacyReview,
    RevocationUpdate,
    SyncedFrame,
    TrackPoint,
    canonical_sha256,
)
from .sources import (
    extract_video_frame,
    read_csv_track,
    read_gpx_track,
    read_image_sources,
    validate_ffmpeg,
    video_sha256,
)

SYNC_FILENAME = "capture-sync.json"
SAMPLE_FILENAME = "capture-sample.json"
INVENTORY_FILENAME = "capture-inventory.json"
CHECKPOINT_FILENAME = "capture-checkpoint.json"
MANIFEST_FILENAME = "manifest.json"

def _load_model[ModelT: BaseModel](
    root: Path,
    locator: str | Path,
    model: type[ModelT],
    *,
    code: str,
    max_bytes: int = 64 * 1024 * 1024,
) -> ModelT:
    try:
        path = resolve_contained_file(root, locator)
        payload = read_bounded_bytes(path, max_bytes=max_bytes, error_prefix=code)
        return model.model_validate_json(payload)
    except CaptureImportError:
        raise
    except (AssetSafetyError, ValidationError, ValueError) as exc:
        raise CaptureImportError(code) from exc


def load_capture_plan(input_root: Path, plan_path: str | Path) -> CapturePlan:
    """Load a bounded, contained plan without revealing its filesystem path."""

    return _load_model(input_root, plan_path, CapturePlan, code="capture_plan_invalid")


def load_capture_inventory(work_root: Path) -> CaptureInventory:
    return _load_model(
        work_root,
        INVENTORY_FILENAME,
        CaptureInventory,
        code="capture_inventory_invalid",
    )


def _source_fingerprint(
    plan_sha256: str,
    ordinal: int,
    content_sha256: str,
    timestamp: datetime,
) -> str:
    return canonical_sha256(
        {
            "plan_sha256": plan_sha256,
            "source_ordinal": ordinal,
            "content_sha256": content_sha256,
            "capture_timestamp": timestamp.isoformat(),
        }
    )


class CaptureImporter:
    """Offline first-party importer with private intermediate coordinates."""

    def __init__(self, input_root: Path, work_root: Path, plan: CapturePlan) -> None:
        self.input_root = input_root
        self.work_root = work_root
        self.plan = plan

    def _read_track(self) -> tuple[TrackPoint, ...]:
        locator = self.plan.track_locator
        if locator is None:
            return ()
        if self.plan.input_mode == "ordered_frames_csv" or locator.casefold().endswith(
            ".csv"
        ):
            points = read_csv_track(self.input_root, locator, self.plan)
        else:
            points = read_gpx_track(self.input_root, locator, self.plan)
        return validate_track(
            points,
            max_speed_kmh=self.plan.max_speed_kmh,
            max_route_distance_km=self.plan.max_route_distance_km,
        )

    def _image_frames(self) -> tuple[tuple[SyncedFrame, ...], int]:
        sources = read_image_sources(self.input_root, self.plan)
        track = self._read_track()
        frames: list[SyncedFrame] = []
        for source in sources:
            if self.plan.input_mode == "geotagged_images":
                point = source.embedded_point
                if point is None:
                    raise CaptureImportError("image_gps_missing")
            else:
                point = interpolate_track(
                    track,
                    source.timestamp,
                    max_gap_seconds=self.plan.max_interpolation_gap_seconds,
                )
            frames.append(
                SyncedFrame(
                    source_ordinal=source.ordinal,
                    source_fingerprint=_source_fingerprint(
                        self.plan.fingerprint,
                        source.ordinal,
                        source.identity.sha256,
                        source.timestamp,
                    ),
                    media_kind="image",
                    capture_timestamp=source.timestamp,
                    latitude=point.latitude,
                    longitude=point.longitude,
                    coordinate_accuracy_m=point.accuracy_m,
                    heading_degrees=point.heading_degrees,
                    expected_image_sha256=source.identity.sha256,
                )
            )
        return tuple(frames), len(track)

    def _video_frames(self) -> tuple[tuple[SyncedFrame, ...], int]:
        validate_ffmpeg(self.plan.ffmpeg_executable)
        _, digest = video_sha256(self.input_root, self.plan)
        assert self.plan.capture_started_at is not None
        assert self.plan.video_duration_seconds is not None
        start = self.plan.capture_started_at
        end = start + timedelta(seconds=self.plan.video_duration_seconds)
        track = self._read_track()
        if track:
            final_offset = self.plan.video_duration_seconds - min(
                0.001,
                self.plan.video_duration_seconds / 2.0,
            )
            final_timestamp = start + timedelta(seconds=final_offset)
            timestamps = {start, final_timestamp}
            timestamps.update(
                point.timestamp for point in track if start <= point.timestamp <= end
            )
            candidates = tuple(sorted(timestamps))
            points = tuple(
                interpolate_track(
                    track,
                    timestamp,
                    max_gap_seconds=self.plan.max_interpolation_gap_seconds,
                )
                for timestamp in candidates
            )
        else:
            assert self.plan.video_latitude is not None
            assert self.plan.video_longitude is not None
            assert self.plan.video_coordinate_accuracy_m is not None
            candidates = (start,)
            points = (
                TrackPoint(
                    timestamp=start,
                    latitude=self.plan.video_latitude,
                    longitude=self.plan.video_longitude,
                    accuracy_m=self.plan.video_coordinate_accuracy_m,
                ),
            )
        frames = tuple(
            SyncedFrame(
                source_ordinal=ordinal,
                source_fingerprint=_source_fingerprint(
                    self.plan.fingerprint,
                    ordinal,
                    digest,
                    timestamp,
                ),
                media_kind="video_frame",
                capture_timestamp=timestamp,
                latitude=point.latitude,
                longitude=point.longitude,
                coordinate_accuracy_m=point.accuracy_m,
                heading_degrees=point.heading_degrees,
                video_offset_seconds=(timestamp - start).total_seconds(),
            )
            for ordinal, (timestamp, point) in enumerate(zip(candidates, points, strict=True))
        )
        return frames, len(track)

    def _synchronized(self) -> CaptureSyncResult:
        if self.plan.input_mode == "video":
            frames, track_count = self._video_frames()
        else:
            frames, track_count = self._image_frames()
        if not frames:
            raise CaptureImportError("synchronized_frames_empty")
        return CaptureSyncResult(
            plan_sha256=self.plan.fingerprint,
            track_point_count=track_count,
            frames=frames,
        )

    def inspect(self) -> CaptureInspection:
        synchronized = self._synchronized()
        return CaptureInspection(
            plan_sha256=self.plan.fingerprint,
            input_mode=self.plan.input_mode,
            media_count=(
                1 if self.plan.input_mode == "video" else len(self.plan.media_locators)
            ),
            track_point_count=synchronized.track_point_count,
            synchronized_count=len(synchronized.frames),
            ffmpeg_required=self.plan.input_mode == "video",
        )

    def synchronize(self) -> CaptureSyncResult:
        result = self._synchronized()
        destination = resolve_output_file(self.work_root, SYNC_FILENAME)
        atomic_write_model(destination, result)
        return result

    def sample(self) -> CaptureSampleResult:
        synchronized = self.synchronize()
        result = sample_frames(
            synchronized.frames,
            plan_sha256=self.plan.fingerprint,
            distance_m=self.plan.sample_distance_m,
            stationary_radius_m=self.plan.stationary_radius_m,
        )
        destination = resolve_output_file(self.work_root, SAMPLE_FILENAME)
        atomic_write_model(destination, result)
        return result

    def preview(self) -> CaptureSampleResult:
        """Compute the full synchronization/sample preview without writing artifacts."""

        synchronized = self._synchronized()
        return sample_frames(
            synchronized.frames,
            plan_sha256=self.plan.fingerprint,
            distance_m=self.plan.sample_distance_m,
            stationary_radius_m=self.plan.stationary_radius_m,
        )

    def _materialize(self, frame: SyncedFrame) -> tuple[bytes, ImageIdentity]:
        if frame.media_kind == "image":
            locator = self.plan.media_locators[frame.source_ordinal]
            source = resolve_contained_file(self.input_root, locator)
            identity = inspect_image(
                source,
                max_bytes=self.plan.max_media_bytes,
                max_pixels=self.plan.max_image_pixels,
            )
            if identity.sha256 != frame.expected_image_sha256:
                raise CaptureImportError("capture_source_changed")
            payload = read_bounded_bytes(
                source,
                max_bytes=self.plan.max_media_bytes,
                error_prefix="capture_image",
            )
            return payload, identity
        executable = validate_ffmpeg(self.plan.ffmpeg_executable)
        video, _ = video_sha256(self.input_root, self.plan)
        temporary = resolve_output_file(
            self.work_root,
            f".capture-tmp/{frame.source_fingerprint}.png",
        )
        if temporary.exists():
            if temporary.is_symlink() or not temporary.is_file():
                raise CaptureImportError("capture_temporary_rejected")
            temporary.unlink()
        try:
            assert frame.video_offset_seconds is not None
            extract_video_frame(
                executable,
                video,
                temporary,
                offset_seconds=frame.video_offset_seconds,
                timeout_seconds=self.plan.ffmpeg_timeout_seconds,
            )
            identity = inspect_image(
                temporary,
                max_bytes=self.plan.max_media_bytes,
                max_pixels=self.plan.max_image_pixels,
            )
            payload = read_bounded_bytes(
                temporary,
                max_bytes=self.plan.max_media_bytes,
                error_prefix="capture_frame",
            )
            return payload, identity
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    def _verify_staged(self, asset: CaptureAsset) -> None:
        try:
            path = resolve_contained_file(self.work_root, asset.file_locator)
            identity = inspect_image(
                path,
                max_bytes=self.plan.max_media_bytes,
                max_pixels=self.plan.max_image_pixels,
            )
        except AssetSafetyError as exc:
            raise CaptureImportError("capture_staged_asset_invalid") from exc
        if identity.sha256 != asset.image_sha256:
            raise CaptureImportError("capture_staged_asset_changed")

    def _persist_state(
        self,
        inventory: CaptureInventory,
        completed_sources: dict[str, str],
        *,
        completed: bool,
    ) -> CaptureCheckpoint:
        inventory_path = resolve_output_file(self.work_root, INVENTORY_FILENAME)
        checkpoint_path = resolve_output_file(self.work_root, CHECKPOINT_FILENAME)
        checkpoint = CaptureCheckpoint(
            plan_sha256=self.plan.fingerprint,
            status="completed" if completed else "in_progress",
            completed_sources=dict(sorted(completed_sources.items())),
            updated_at=datetime.now(UTC),
        )
        atomic_write_model(inventory_path, inventory, require_idempotent=False)
        atomic_write_model(checkpoint_path, checkpoint, require_idempotent=False)
        return checkpoint

    def import_capture(
        self,
        *,
        resume: bool = True,
        max_new_assets: int | None = None,
    ) -> CaptureImportResult:
        if max_new_assets is not None and max_new_assets < 1:
            raise CaptureImportError("max_new_assets_invalid")
        ensure_safe_output_root(self.work_root)
        sampled = self.sample()
        inventory_path = self.work_root / INVENTORY_FILENAME
        checkpoint_path = self.work_root / CHECKPOINT_FILENAME
        state_exists = inventory_path.exists() or checkpoint_path.exists()
        if state_exists and not resume:
            raise CaptureImportError("capture_resume_required")
        if inventory_path.exists() != checkpoint_path.exists():
            raise CaptureImportError("capture_state_incomplete")
        if state_exists:
            inventory = load_capture_inventory(self.work_root)
            checkpoint = _load_model(
                self.work_root,
                CHECKPOINT_FILENAME,
                CaptureCheckpoint,
                code="capture_checkpoint_invalid",
            )
            if (
                inventory.plan_sha256 != self.plan.fingerprint
                or checkpoint.plan_sha256 != self.plan.fingerprint
            ):
                raise CaptureImportError("capture_plan_mismatch")
        else:
            inventory = CaptureInventory(plan_sha256=self.plan.fingerprint, assets=())
            checkpoint = CaptureCheckpoint(
                plan_sha256=self.plan.fingerprint,
                status="in_progress",
                completed_sources={},
                updated_at=datetime.now(UTC),
            )
            self._persist_state(inventory, {}, completed=False)

        current_fingerprints = {frame.source_fingerprint for frame in sampled.frames}
        if not set(checkpoint.completed_sources).issubset(current_fingerprints):
            raise CaptureImportError("capture_checkpoint_source_mismatch")
        assets = {asset.asset_id: asset for asset in inventory.assets}
        completed_sources = dict(checkpoint.completed_sources)
        imported_count = 0
        resumed_count = 0

        for frame in sampled.frames:
            completed_asset_id = completed_sources.get(frame.source_fingerprint)
            if completed_asset_id is not None:
                staged = assets.get(completed_asset_id)
                if staged is None:
                    raise CaptureImportError("capture_checkpoint_asset_missing")
                self._verify_staged(staged)
                resumed_count += 1
                continue
            if max_new_assets is not None and imported_count >= max_new_assets:
                break

            payload, identity = self._materialize(frame)
            asset_digest = canonical_sha256(
                {
                    "plan_sha256": self.plan.fingerprint,
                    "capture_run_id": self.plan.capture_run_id,
                    "image_sha256": identity.sha256,
                }
            )
            asset_id = f"capture-{asset_digest[:32]}"
            duplicate = assets.get(asset_id)
            if duplicate is not None:
                self._verify_staged(duplicate)
                completed_sources[frame.source_fingerprint] = asset_id
                self._persist_state(
                    CaptureInventory(
                        plan_sha256=self.plan.fingerprint,
                        assets=tuple(sorted(assets.values(), key=lambda item: item.asset_id)),
                    ),
                    completed_sources,
                    completed=False,
                )
                resumed_count += 1
                continue

            extension = ".jpg" if identity.mime_type == "image/jpeg" else ".png"
            file_locator = f"quarantine/{asset_id}{extension}"
            destination = resolve_output_file(self.work_root, file_locator)
            atomic_write_bytes(destination, payload)
            source_asset_id = f"first-party-{frame.source_fingerprint[:32]}"
            receipt_hash = canonical_sha256(
                {
                    "plan_sha256": self.plan.fingerprint,
                    "source_asset_id": source_asset_id,
                    "image_sha256": identity.sha256,
                    "capture_run_id": self.plan.capture_run_id,
                    "device_id": self.plan.device_id,
                    "rights": self.plan.rights.model_dump(mode="json"),
                }
            )
            asset = CaptureAsset(
                asset_id=asset_id,
                source_asset_id=source_asset_id,
                file_locator=file_locator,
                image_sha256=identity.sha256,
                perceptual_hash=identity.perceptual_hash,
                width_px=identity.width_px,
                height_px=identity.height_px,
                mime_type=cast(Literal["image/jpeg", "image/png"], identity.mime_type),
                capture_timestamp=frame.capture_timestamp,
                latitude=frame.latitude,
                longitude=frame.longitude,
                coordinate_accuracy_m=frame.coordinate_accuracy_m,
                heading_degrees=frame.heading_degrees,
                source_fingerprint=frame.source_fingerprint,
                source_name=self.plan.rights.source_name,
                contributor_or_owner=self.plan.contributor_or_owner,
                capture_run_id=self.plan.capture_run_id,
                sequence_id=self.plan.sequence_id,
                device_id=self.plan.device_id,
                corpus_version=self.plan.corpus_version,
                province_code=self.plan.province_code,
                spatial_split=self.plan.spatial_split,
                role=self.plan.role,
                tier=self.plan.tier,
                acquisition_timestamp=self.plan.acquisition_timestamp,
                urbanicity=self.plan.urbanicity,
                road_class=self.plan.road_class,
                scene_type=self.plan.scene_type,
                road_context=self.plan.road_context,
                terrain_class=self.plan.terrain_class,
                vegetation_state=self.plan.vegetation_state,
                season=self.plan.season,
                rights=self.plan.rights,
                provenance_receipt_id=f"receipt-{receipt_hash[:32]}",
                provenance_receipt_sha256=receipt_hash,
                revocation_checked_at=self.plan.acquisition_timestamp,
            )
            assets[asset_id] = asset
            completed_sources[frame.source_fingerprint] = asset_id
            inventory = CaptureInventory(
                plan_sha256=self.plan.fingerprint,
                assets=tuple(sorted(assets.values(), key=lambda item: item.asset_id)),
            )
            self._persist_state(inventory, completed_sources, completed=False)
            imported_count += 1

        completed = current_fingerprints.issubset(completed_sources)
        inventory = CaptureInventory(
            plan_sha256=self.plan.fingerprint,
            assets=tuple(sorted(assets.values(), key=lambda item: item.asset_id)),
        )
        self._persist_state(inventory, completed_sources, completed=completed)
        return CaptureImportResult(
            status="completed" if completed else "partial",
            plan_sha256=self.plan.fingerprint,
            input_count=sampled.input_count,
            sampled_count=sampled.sampled_count,
            imported_count=imported_count,
            resumed_count=resumed_count,
            pending_privacy_count=sum(
                asset.privacy.state == "pending" for asset in inventory.assets
            ),
            inventory=inventory,
        )


def _replace_inventory_asset(
    inventory: CaptureInventory,
    replacement: CaptureAsset,
) -> CaptureInventory:
    assets = tuple(
        sorted(
            (
                replacement if asset.asset_id == replacement.asset_id else asset
                for asset in inventory.assets
            ),
            key=lambda item: item.asset_id,
        )
    )
    return CaptureInventory(plan_sha256=inventory.plan_sha256, assets=assets)


def _find_asset(inventory: CaptureInventory, asset_id: str) -> CaptureAsset:
    for asset in inventory.assets:
        if asset.asset_id == asset_id:
            return asset
    raise CaptureImportError("capture_asset_not_found")


def _extension(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/png":
        return ".png"
    raise CaptureImportError("capture_image_format_unsupported")


def _privacy_source(
    root: Path,
    locator: str | Path,
    *,
    code: str,
) -> tuple[Path, bytes, ImageIdentity]:
    try:
        path = resolve_contained_file(root, locator)
        identity = inspect_image(path, max_bytes=40 * 1024 * 1024, max_pixels=100_000_000)
        if identity.mime_type not in {"image/jpeg", "image/png"}:
            raise CaptureImportError("capture_image_format_unsupported")
        payload = read_bounded_bytes(
            path,
            max_bytes=40 * 1024 * 1024,
            error_prefix=code,
        )
    except CaptureImportError:
        raise
    except AssetSafetyError as exc:
        raise CaptureImportError(code) from exc
    return path, payload, identity


def _reviewed_asset(
    work_root: Path,
    current: CaptureAsset,
    review: PrivacyReview,
    *,
    redacted_input_root: Path | None,
    redacted_copy: str | Path | None,
) -> tuple[CaptureAsset, tuple[Path, ...]]:
    current_path, current_payload, current_identity = _privacy_source(
        work_root,
        current.file_locator,
        code="capture_review_source_invalid",
    )
    if current_identity.sha256 != current.image_sha256:
        raise CaptureImportError("capture_review_source_changed")
    if review.redaction_applied:
        if redacted_input_root is None or redacted_copy is None:
            raise CaptureImportError("redacted_copy_required")
        _, redacted_payload, identity = _privacy_source(
            redacted_input_root,
            redacted_copy,
            code="redacted_copy_invalid",
        )
        original_locator = f"originals/{current.asset_id}{_extension(current.mime_type)}"
        original_destination = resolve_output_file(work_root, original_locator)
        atomic_write_bytes(original_destination, current_payload)
        file_locator = f"assets/{current.asset_id}{_extension(identity.mime_type)}"
        destination = resolve_output_file(work_root, file_locator)
        atomic_write_bytes(destination, redacted_payload, require_idempotent=False)
        assert review.redaction_receipt_sha256 is not None
        receipt_hash = canonical_sha256(
            {
                "prior_receipt_sha256": current.provenance_receipt_sha256,
                "redacted_image_sha256": identity.sha256,
                "redaction_receipt_sha256": review.redaction_receipt_sha256,
            }
        )
        updates: dict[str, object] = {
            "file_locator": file_locator,
            "image_sha256": identity.sha256,
            "perceptual_hash": identity.perceptual_hash,
            "width_px": identity.width_px,
            "height_px": identity.height_px,
            "mime_type": identity.mime_type,
            "provenance_receipt_id": f"receipt-{receipt_hash[:32]}",
            "provenance_receipt_sha256": receipt_hash,
            "privacy": review,
        }
        cleanup = (current_path,) if current_path != destination else ()
    else:
        if redacted_input_root is not None or redacted_copy is not None:
            raise CaptureImportError("unexpected_redacted_copy")
        if review.state == "approved":
            file_locator = f"assets/{current.asset_id}{_extension(current.mime_type)}"
        else:
            file_locator = f"quarantine/{current.asset_id}{_extension(current.mime_type)}"
        destination = resolve_output_file(work_root, file_locator)
        atomic_write_bytes(destination, current_payload, require_idempotent=False)
        updates = {"file_locator": file_locator, "privacy": review}
        cleanup = (current_path,) if current_path != destination else ()
    payload = current.model_dump(mode="python")
    payload.update(updates)
    return CaptureAsset.model_validate(payload), cleanup


def update_capture_privacy(
    work_root: Path,
    decision: PrivacyDecision,
    *,
    redacted_input_root: Path | None = None,
    redacted_copy: str | Path | None = None,
) -> CaptureInventory:
    """Record an explicit human decision; pending is never silently promoted."""

    inventory = load_capture_inventory(work_root)
    current = _find_asset(inventory, decision.asset_id)
    if decision.state == "pending":
        raise CaptureImportError("privacy_decision_required")
    review = PrivacyReview.model_validate(
        decision.model_dump(mode="python", exclude={"asset_id"})
    )
    if current.privacy == review:
        return inventory
    if current.privacy.state == "rejected":
        raise CaptureImportError("privacy_rejection_terminal")
    replacement, cleanup = _reviewed_asset(
        work_root,
        current,
        review,
        redacted_input_root=redacted_input_root,
        redacted_copy=redacted_copy,
    )
    updated = _replace_inventory_asset(inventory, replacement)
    destination = resolve_output_file(work_root, INVENTORY_FILENAME)
    atomic_write_model(destination, updated, require_idempotent=False)
    for path in cleanup:
        with suppress(OSError):
            path.unlink(missing_ok=True)
    return updated


def revoke_capture_asset(
    work_root: Path,
    update: RevocationUpdate,
) -> CaptureInventory:
    inventory = load_capture_inventory(work_root)
    current = _find_asset(inventory, update.asset_id)
    if current.revocation_status == "REVOKED":
        if (
            current.revocation_request_id == update.request_id
            and current.revocation_receipt_sha256 == update.receipt_sha256
        ):
            return inventory
        raise CaptureImportError("capture_asset_already_revoked")
    current_path, payload, identity = _privacy_source(
        work_root,
        current.file_locator,
        code="capture_revocation_source_invalid",
    )
    if identity.sha256 != current.image_sha256:
        raise CaptureImportError("capture_revocation_source_changed")
    quarantine_locator = (
        f"quarantine/{current.asset_id}{_extension(current.mime_type)}"
    )
    quarantine = resolve_output_file(work_root, quarantine_locator)
    atomic_write_bytes(quarantine, payload, require_idempotent=False)
    replacement_payload = current.model_dump(mode="python")
    replacement_payload.update(
        {
            "file_locator": quarantine_locator,
            "revocation_status": "REVOKED",
            "revocation_checked_at": update.revoked_at,
            "revocation_request_id": update.request_id,
            "revocation_receipt_sha256": update.receipt_sha256,
        }
    )
    replacement = CaptureAsset.model_validate(replacement_payload)
    updated = _replace_inventory_asset(inventory, replacement)
    destination = resolve_output_file(work_root, INVENTORY_FILENAME)
    atomic_write_model(destination, updated, require_idempotent=False)
    if current_path != quarantine:
        with suppress(OSError):
            current_path.unlink(missing_ok=True)
    return updated


def _manifest_asset(work_root: Path, asset: CaptureAsset) -> ManifestAsset:
    if asset.privacy.state != "approved" or asset.revocation_status != "ACTIVE":
        raise CaptureImportError("capture_asset_not_manifest_eligible")
    expected_locator = f"assets/{asset.asset_id}{_extension(asset.mime_type)}"
    if asset.file_locator != expected_locator:
        raise CaptureImportError("approved_asset_not_published")
    _, _, identity = _privacy_source(
        work_root,
        asset.file_locator,
        code="approved_asset_invalid",
    )
    if (
        identity.sha256 != asset.image_sha256
        or identity.perceptual_hash != asset.perceptual_hash
        or identity.width_px != asset.width_px
        or identity.height_px != asset.height_px
        or identity.mime_type != asset.mime_type
    ):
        raise CaptureImportError("approved_asset_identity_mismatch")
    personal_data_state: Literal["BLURRED_BY_ATLASLENS", "NOT_DETECTED"] = (
        "BLURRED_BY_ATLASLENS" if asset.privacy.redaction_applied else "NOT_DETECTED"
    )
    return ManifestAsset(
        corpus_version=asset.corpus_version,
        asset_id=asset.asset_id,
        source_asset_id=asset.source_asset_id,
        source_name=asset.source_name,
        source_url=(
            f"atlaslens-capture://{asset.capture_run_id}/{asset.source_asset_id}"
        ),
        contributor_or_owner=asset.contributor_or_owner,
        capture_timestamp=asset.capture_timestamp,
        acquisition_timestamp=asset.acquisition_timestamp,
        latitude=asset.latitude,
        longitude=asset.longitude,
        coordinate_accuracy_m=asset.coordinate_accuracy_m,
        heading_degrees=asset.heading_degrees,
        sequence_id=asset.sequence_id,
        capture_run_id=asset.capture_run_id,
        image_sha256=asset.image_sha256,
        perceptual_hash=asset.perceptual_hash,
        perceptual_hash_algorithm="dhash64-v1",
        width_px=asset.width_px,
        height_px=asset.height_px,
        mime_type=asset.mime_type,
        asset_type="street_level",
        province_code=asset.province_code,
        urbanicity=asset.urbanicity,
        road_class=asset.road_class,
        scene_type=asset.scene_type,
        road_context=asset.road_context,
        terrain_class=asset.terrain_class,
        vegetation_state=asset.vegetation_state,
        season=asset.season,
        license_identifier=asset.rights.license_identifier,
        license_url=asset.rights.license_url,
        attribution_text=asset.rights.attribution_text,
        source_policy_decision="FIRST_PARTY_ONLY",
        commercial_use_decision="FIRST_PARTY_ONLY",
        derivative_index_decision="FIRST_PARTY_ONLY",
        personal_data_blur_state=personal_data_state,
        deletion_revocation_state=DeletionRevocationState(
            status="ACTIVE",
            last_checked_at=asset.revocation_checked_at,
        ),
        provenance_receipt=ProvenanceReceipt(
            receipt_id=asset.provenance_receipt_id,
            source_policy_version=asset.rights.source_policy_version,
            evidence_date=asset.rights.source_policy_evidence_date,
            acquisition_method="first_party_capture",
            receipt_sha256=asset.provenance_receipt_sha256,
        ),
        spatial_split=asset.spatial_split,
        role=asset.role,
        tier=asset.tier,
        acquisition_ready=True,
    )


def build_capture_manifest(
    work_root: Path,
    output_path: str | Path = MANIFEST_FILENAME,
) -> CaptureManifestResult:
    inventory = load_capture_inventory(work_root)
    excluded_by_reason = {
        "privacy_pending": 0,
        "privacy_rejected": 0,
        "privacy_needs_redaction": 0,
        "revoked": 0,
    }
    eligible: list[ManifestAsset] = []
    for asset in inventory.assets:
        if asset.revocation_status == "REVOKED":
            excluded_by_reason["revoked"] += 1
        elif asset.privacy.state == "approved":
            eligible.append(_manifest_asset(work_root, asset))
        else:
            excluded_by_reason[f"privacy_{asset.privacy.state}"] += 1
    eligible.sort(key=lambda item: item.asset_id)
    rows = [asset.model_dump(mode="json") for asset in eligible]
    payload = (
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    destination = resolve_output_file(work_root, output_path)
    manifest_sha256 = atomic_write_bytes(
        destination,
        payload,
        require_idempotent=False,
    )
    excluded_count = sum(excluded_by_reason.values())
    return CaptureManifestResult(
        status="ready" if eligible else "empty",
        eligible_count=len(eligible),
        excluded_count=excluded_count,
        excluded_by_reason=excluded_by_reason,
        manifest_sha256=manifest_sha256,
    )


__all__ = [
    "CHECKPOINT_FILENAME",
    "INVENTORY_FILENAME",
    "MANIFEST_FILENAME",
    "SAMPLE_FILENAME",
    "SYNC_FILENAME",
    "CaptureImporter",
    "build_capture_manifest",
    "load_capture_inventory",
    "load_capture_plan",
    "revoke_capture_asset",
    "update_capture_privacy",
]
