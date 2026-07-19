from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from typing import Literal, Self
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.corpus_pipeline.models import (
    OPAQUE_ID_PATTERN,
    SHA256_PATTERN,
)

InputMode = Literal[
    "geotagged_images",
    "ordered_frames_gpx",
    "ordered_frames_csv",
    "video",
]
PrivacyState = Literal["pending", "approved", "rejected", "needs_redaction"]
SampleDistance = Literal[25, 50, 100]


class CaptureModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_aware(value: datetime | None) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC) if value is not None else None


def _require_safe_locator(value: str | None) -> str | None:
    if value is None:
        return None
    from pathlib import PurePath

    locator = PurePath(value)
    if locator.is_absolute() or not locator.parts or ".." in locator.parts:
        raise ValueError("locator must be a contained relative path")
    return value


class FirstPartyRights(CaptureModel):
    source_name: Literal["First-party AtlasLens-captured imagery"] = (
        "First-party AtlasLens-captured imagery"
    )
    source_policy_version: str = Field(min_length=1, max_length=160)
    source_policy_evidence_date: date
    license_identifier: str = Field(min_length=1, max_length=200)
    license_url: str = Field(min_length=1, max_length=2048)
    attribution_text: str = Field(min_length=1, max_length=2000)
    capture_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    contributor_assignment_sha256: str = Field(pattern=SHA256_PATTERN)
    privacy_notice_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("license_url")
    @classmethod
    def require_credential_free_uri(cls, value: str) -> str:
        parsed = urlsplit(value)
        if not parsed.scheme or parsed.username is not None or parsed.password is not None:
            raise ValueError("license URL must be an absolute credential-free URI")
        return value


class CapturePlan(CaptureModel):
    schema_version: Literal["atlaslens-first-party-capture-plan-v1"] = (
        "atlaslens-first-party-capture-plan-v1"
    )
    plan_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    input_mode: InputMode
    corpus_version: str = Field(
        pattern=r"^atlaslens-turkiye-corpus-[A-Za-z0-9._-]+$",
        max_length=160,
    )
    media_locators: tuple[str, ...] = Field(default=(), max_length=100_000)
    video_locator: str | None = Field(default=None, min_length=1, max_length=500)
    track_locator: str | None = Field(default=None, min_length=1, max_length=500)
    frame_timestamps: tuple[datetime, ...] | None = Field(default=None, max_length=100_000)
    capture_started_at: datetime | None = None
    frame_interval_seconds: float | None = Field(default=None, gt=0, le=86_400)
    video_duration_seconds: float | None = Field(default=None, gt=0, le=86_400)
    video_latitude: float | None = Field(default=None, ge=-90, le=90)
    video_longitude: float | None = Field(default=None, ge=-180, le=180)
    video_coordinate_accuracy_m: float | None = Field(default=None, gt=0, le=100_000)
    ffmpeg_executable: str | None = Field(default=None, min_length=1, max_length=1000)
    capture_timezone: str | None = Field(default=None, min_length=1, max_length=80)
    acquisition_timestamp: datetime
    capture_run_id: str = Field(min_length=1, max_length=256)
    sequence_id: str = Field(min_length=1, max_length=256)
    device_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    contributor_or_owner: str = Field(min_length=1, max_length=320)
    province_code: str = Field(pattern=r"^TR-(0[1-9]|[1-7][0-9]|8[01])$")
    spatial_split: str = Field(pattern=OPAQUE_ID_PATTERN)
    role: Literal["train", "development", "validation", "holdout"] = "train"
    tier: Literal[
        "tier_1_national_recall",
        "tier_2_dense_corridor",
        "tier_3_locked_holdout",
    ] = "tier_1_national_recall"
    sample_distance_m: SampleDistance = 50
    stationary_radius_m: float = Field(default=3.0, ge=0, le=100)
    default_coordinate_accuracy_m: float = Field(default=15.0, gt=0, le=100_000)
    max_interpolation_gap_seconds: float = Field(default=30.0, gt=0, le=3600)
    max_speed_kmh: float = Field(default=180.0, gt=0, le=1000)
    max_route_distance_km: float = Field(default=5000.0, gt=0, le=100_000)
    max_media_bytes: int = Field(
        default=40 * 1024 * 1024,
        gt=0,
        le=40 * 1024 * 1024,
    )
    max_video_bytes: int = Field(default=20 * 1024 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=100_000_000, gt=0, le=100_000_000)
    max_track_points: int = Field(default=1_000_000, gt=1)
    ffmpeg_timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    urbanicity: Literal[
        "urban_core", "suburban", "rural_settlement", "rural_road", "unknown"
    ] = "unknown"
    road_class: Literal[
        "motorway_or_expressway",
        "national_or_state_road",
        "provincial_or_district_road",
        "local_urban_street",
        "industrial_access",
        "unclassified_or_track",
        "not_applicable",
        "unknown",
    ] = "unknown"
    scene_type: Literal[
        "city_center",
        "suburban",
        "industrial",
        "rural_road",
        "terrain_dominant",
        "mixed",
        "unknown",
    ] = "unknown"
    road_context: Literal[
        "divided_highway",
        "ordinary_road",
        "rural_road",
        "junction",
        "service_area",
        "not_applicable",
        "unknown",
    ] = "unknown"
    terrain_class: Literal["mountainous", "flat", "rolling", "mixed", "unknown"] = (
        "unknown"
    )
    vegetation_state: Literal[
        "leaf_on",
        "leaf_off_or_dormant",
        "cropland_active",
        "sparse_or_arid",
        "snow_cover",
        "mixed",
        "not_applicable",
        "unknown",
    ] = "unknown"
    season: Literal["winter", "spring", "summer", "autumn", "unknown"] = "unknown"
    rights: FirstPartyRights

    @field_validator("media_locators")
    @classmethod
    def require_safe_unique_media(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("media locators must be unique")
        return tuple(_require_safe_locator(value) or "" for value in values)

    @field_validator("video_locator", "track_locator")
    @classmethod
    def require_safe_optional_locator(cls, value: str | None) -> str | None:
        return _require_safe_locator(value)

    @field_validator("frame_timestamps")
    @classmethod
    def require_aware_frame_times(
        cls, values: tuple[datetime, ...] | None
    ) -> tuple[datetime, ...] | None:
        if values is None:
            return None
        return tuple(_require_aware(value) for value in values)  # type: ignore[misc]

    @field_validator("capture_started_at", "acquisition_timestamp")
    @classmethod
    def require_aware_times(cls, value: datetime | None) -> datetime | None:
        return _require_aware(value)

    @model_validator(mode="after")
    def require_mode_contract(self) -> Self:
        if self.capture_timezone is not None:
            try:
                ZoneInfo(self.capture_timezone)
            except ZoneInfoNotFoundError as exc:
                raise ValueError("capture timezone is unavailable") from exc
        if (self.role == "holdout") != (self.tier == "tier_3_locked_holdout"):
            raise ValueError("holdout role and tier must agree")
        video_only_values = (
            self.video_locator,
            self.video_duration_seconds,
            self.video_latitude,
            self.video_longitude,
            self.video_coordinate_accuracy_m,
            self.ffmpeg_executable,
        )
        if self.input_mode != "video" and any(
            value is not None for value in video_only_values
        ):
            raise ValueError("image modes do not accept video-only fields")
        if self.input_mode == "geotagged_images":
            if not self.media_locators:
                raise ValueError("geotagged image mode requires explicit media locators")
            if (
                self.track_locator is not None
                or self.frame_timestamps is not None
                or self.capture_started_at is not None
                or self.frame_interval_seconds is not None
            ):
                raise ValueError("geotagged image mode uses embedded time and coordinates")
        elif self.input_mode in {"ordered_frames_gpx", "ordered_frames_csv"}:
            if not self.media_locators or self.track_locator is None:
                raise ValueError("ordered frame mode requires media and track locators")
            if self.frame_timestamps is not None and len(self.frame_timestamps) != len(
                self.media_locators
            ):
                raise ValueError("frame timestamp count must match media count")
            if self.frame_timestamps is None and (
                self.capture_started_at is None or self.frame_interval_seconds is None
            ):
                raise ValueError("ordered frames need timestamps or a bounded time cadence")
            if self.frame_timestamps is not None and (
                self.capture_started_at is not None
                or self.frame_interval_seconds is not None
            ):
                raise ValueError("ordered frame timing inputs are ambiguous")
        elif self.input_mode == "video":
            if self.video_locator is None or self.video_duration_seconds is None:
                raise ValueError("video mode requires a video locator and duration")
            if self.capture_started_at is None:
                raise ValueError("video mode requires a timezone-aware start time")
            if self.media_locators:
                raise ValueError("video mode does not accept image locators")
            if self.frame_timestamps is not None or self.frame_interval_seconds is not None:
                raise ValueError("video mode does not accept ordered-frame timing")
            coordinate_values = (
                self.video_latitude,
                self.video_longitude,
                self.video_coordinate_accuracy_m,
            )
            if any(value is not None for value in coordinate_values) and not all(
                value is not None for value in coordinate_values
            ):
                raise ValueError("video metadata coordinate must be complete")
            if self.track_locator is None and (
                self.video_latitude is None
                or self.video_longitude is None
                or self.video_coordinate_accuracy_m is None
            ):
                raise ValueError("video mode requires a route or explicit metadata coordinate")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class TrackPoint(CaptureModel):
    timestamp: datetime
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy_m: float = Field(gt=0, le=100_000)
    heading_degrees: float | None = Field(default=None, ge=0, lt=360)

    @field_validator("timestamp")
    @classmethod
    def require_time(cls, value: datetime) -> datetime:
        result = _require_aware(value)
        assert result is not None
        return result


class SyncedFrame(CaptureModel):
    source_ordinal: int = Field(ge=0)
    source_fingerprint: str = Field(pattern=SHA256_PATTERN)
    media_kind: Literal["image", "video_frame"]
    capture_timestamp: datetime
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    coordinate_accuracy_m: float = Field(gt=0, le=100_000)
    heading_degrees: float | None = Field(default=None, ge=0, lt=360)
    expected_image_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    video_offset_seconds: float | None = Field(default=None, ge=0, le=86_400)

    @field_validator("capture_timestamp")
    @classmethod
    def require_time(cls, value: datetime) -> datetime:
        result = _require_aware(value)
        assert result is not None
        return result


class CaptureInspection(CaptureModel):
    schema_version: Literal["atlaslens-capture-inspection-v1"] = (
        "atlaslens-capture-inspection-v1"
    )
    status: Literal["ready"] = "ready"
    plan_sha256: str = Field(pattern=SHA256_PATTERN)
    input_mode: InputMode
    media_count: int = Field(ge=0)
    track_point_count: int = Field(ge=0)
    synchronized_count: int = Field(ge=0)
    ffmpeg_required: bool
    privacy_default: Literal["pending"] = "pending"
    accuracy_claim: None = None


class CaptureSyncResult(CaptureModel):
    schema_version: Literal["atlaslens-capture-sync-v1"] = "atlaslens-capture-sync-v1"
    plan_sha256: str = Field(pattern=SHA256_PATTERN)
    track_point_count: int = Field(ge=0)
    frames: tuple[SyncedFrame, ...]


class CaptureSampleResult(CaptureModel):
    schema_version: Literal["atlaslens-capture-sample-v1"] = (
        "atlaslens-capture-sample-v1"
    )
    plan_sha256: str = Field(pattern=SHA256_PATTERN)
    distance_m: SampleDistance
    input_count: int = Field(ge=0)
    sampled_count: int = Field(ge=0)
    stationary_deduplicated_count: int = Field(ge=0)
    distance_excluded_count: int = Field(ge=0)
    frames: tuple[SyncedFrame, ...]


class PrivacyReview(CaptureModel):
    state: PrivacyState = "pending"
    reviewed_by: str | None = Field(default=None, min_length=1, max_length=160)
    reviewed_at: datetime | None = None
    decision_receipt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    redaction_applied: bool = False
    redaction_receipt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("reviewed_at")
    @classmethod
    def require_review_time(cls, value: datetime | None) -> datetime | None:
        return _require_aware(value)

    @model_validator(mode="after")
    def require_explicit_review(self) -> Self:
        if self.state == "pending":
            if any(
                value is not None
                for value in (
                    self.reviewed_by,
                    self.reviewed_at,
                    self.decision_receipt_sha256,
                    self.redaction_receipt_sha256,
                )
            ) or self.redaction_applied:
                raise ValueError("pending privacy state cannot contain a review decision")
            return self
        if (
            self.reviewed_by is None
            or self.reviewed_at is None
            or self.decision_receipt_sha256 is None
        ):
            raise ValueError("privacy state requires an explicit review receipt")
        if self.state != "approved" and self.redaction_applied:
            raise ValueError("only an approved explicit redaction may be recorded")
        if self.state == "approved" and self.redaction_applied:
            if self.redaction_receipt_sha256 is None:
                raise ValueError("approved redaction requires a receipt")
        elif self.redaction_receipt_sha256 is not None:
            raise ValueError("redaction receipt is only valid for approved redaction")
        return self


class PrivacyDecision(PrivacyReview):
    asset_id: str = Field(pattern=OPAQUE_ID_PATTERN)


class RevocationUpdate(CaptureModel):
    asset_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    request_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    revoked_at: datetime

    @field_validator("revoked_at")
    @classmethod
    def require_time(cls, value: datetime) -> datetime:
        result = _require_aware(value)
        assert result is not None
        return result


class CaptureAsset(CaptureModel):
    asset_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    source_asset_id: str = Field(min_length=1, max_length=256)
    file_locator: str = Field(min_length=1, max_length=500)
    image_sha256: str = Field(pattern=SHA256_PATTERN)
    perceptual_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    width_px: int = Field(gt=0, le=100_000)
    height_px: int = Field(gt=0, le=100_000)
    mime_type: Literal["image/jpeg", "image/png"]
    capture_timestamp: datetime
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    coordinate_accuracy_m: float = Field(gt=0, le=100_000)
    heading_degrees: float | None = Field(default=None, ge=0, lt=360)
    source_fingerprint: str = Field(pattern=SHA256_PATTERN)
    source_name: Literal["First-party AtlasLens-captured imagery"]
    contributor_or_owner: str = Field(min_length=1, max_length=320)
    capture_run_id: str = Field(min_length=1, max_length=256)
    sequence_id: str = Field(min_length=1, max_length=256)
    device_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    corpus_version: str = Field(max_length=160)
    province_code: str = Field(pattern=r"^TR-(0[1-9]|[1-7][0-9]|8[01])$")
    spatial_split: str = Field(pattern=OPAQUE_ID_PATTERN)
    role: Literal["train", "development", "validation", "holdout"]
    tier: Literal[
        "tier_1_national_recall",
        "tier_2_dense_corridor",
        "tier_3_locked_holdout",
    ]
    acquisition_timestamp: datetime
    urbanicity: Literal[
        "urban_core", "suburban", "rural_settlement", "rural_road", "unknown"
    ]
    road_class: Literal[
        "motorway_or_expressway",
        "national_or_state_road",
        "provincial_or_district_road",
        "local_urban_street",
        "industrial_access",
        "unclassified_or_track",
        "not_applicable",
        "unknown",
    ]
    scene_type: Literal[
        "city_center",
        "suburban",
        "industrial",
        "rural_road",
        "terrain_dominant",
        "mixed",
        "unknown",
    ]
    road_context: Literal[
        "divided_highway",
        "ordinary_road",
        "rural_road",
        "junction",
        "service_area",
        "not_applicable",
        "unknown",
    ]
    terrain_class: Literal["mountainous", "flat", "rolling", "mixed", "unknown"]
    vegetation_state: Literal[
        "leaf_on",
        "leaf_off_or_dormant",
        "cropland_active",
        "sparse_or_arid",
        "snow_cover",
        "mixed",
        "not_applicable",
        "unknown",
    ]
    season: Literal["winter", "spring", "summer", "autumn", "unknown"]
    rights: FirstPartyRights
    provenance_receipt_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    provenance_receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    privacy: PrivacyReview = Field(default_factory=PrivacyReview)
    revocation_status: Literal["ACTIVE", "REVOKED"] = "ACTIVE"
    revocation_checked_at: datetime
    revocation_request_id: str | None = Field(default=None, pattern=OPAQUE_ID_PATTERN)
    revocation_receipt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("file_locator")
    @classmethod
    def require_safe_file_locator(cls, value: str) -> str:
        result = _require_safe_locator(value)
        assert result is not None
        return result

    @field_validator(
        "capture_timestamp", "acquisition_timestamp", "revocation_checked_at"
    )
    @classmethod
    def require_times(cls, value: datetime) -> datetime:
        result = _require_aware(value)
        assert result is not None
        return result

    @model_validator(mode="after")
    def require_revocation_evidence(self) -> Self:
        evidence_present = (
            self.revocation_request_id is not None
            and self.revocation_receipt_sha256 is not None
        )
        if (self.revocation_status == "REVOKED") != evidence_present:
            raise ValueError("revocation state and evidence must agree")
        extension = ".jpg" if self.mime_type == "image/jpeg" else ".png"
        published = self.privacy.state == "approved" and self.revocation_status == "ACTIVE"
        directory = "assets" if published else "quarantine"
        if self.file_locator != f"{directory}/{self.asset_id}{extension}":
            raise ValueError("privacy state and capture file locator must agree")
        return self


class CaptureInventory(CaptureModel):
    schema_version: Literal["atlaslens-capture-inventory-v1"] = (
        "atlaslens-capture-inventory-v1"
    )
    plan_sha256: str = Field(pattern=SHA256_PATTERN)
    assets: tuple[CaptureAsset, ...]

    @model_validator(mode="after")
    def require_unique_assets(self) -> Self:
        identifiers = [asset.asset_id for asset in self.assets]
        fingerprints = [asset.source_fingerprint for asset in self.assets]
        if len(identifiers) != len(set(identifiers)) or len(fingerprints) != len(
            set(fingerprints)
        ):
            raise ValueError("capture inventory identities must be unique")
        return self


class CaptureCheckpoint(CaptureModel):
    schema_version: Literal["atlaslens-capture-checkpoint-v1"] = (
        "atlaslens-capture-checkpoint-v1"
    )
    plan_sha256: str = Field(pattern=SHA256_PATTERN)
    status: Literal["in_progress", "completed"]
    completed_sources: dict[str, str]
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def require_time(cls, value: datetime) -> datetime:
        result = _require_aware(value)
        assert result is not None
        return result


class CaptureImportResult(CaptureModel):
    status: Literal["partial", "completed"]
    plan_sha256: str = Field(pattern=SHA256_PATTERN)
    input_count: int = Field(ge=0)
    sampled_count: int = Field(ge=0)
    imported_count: int = Field(ge=0)
    resumed_count: int = Field(ge=0)
    pending_privacy_count: int = Field(ge=0)
    inventory: CaptureInventory


class CaptureManifestResult(CaptureModel):
    status: Literal["ready", "empty"]
    eligible_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    excluded_by_reason: dict[str, int]
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)


__all__ = [
    "CaptureAsset",
    "CaptureCheckpoint",
    "CaptureImportResult",
    "CaptureInspection",
    "CaptureInventory",
    "CaptureManifestResult",
    "CaptureModel",
    "CapturePlan",
    "CaptureSampleResult",
    "CaptureSyncResult",
    "FirstPartyRights",
    "InputMode",
    "PrivacyDecision",
    "PrivacyReview",
    "PrivacyState",
    "RevocationUpdate",
    "SampleDistance",
    "SyncedFrame",
    "TrackPoint",
    "canonical_sha256",
]
