"""Strict, credential-free models for the private Mapillary demo pilot."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAPILLARY_API_BASE_URL = "https://graph.mapillary.com"
MAPILLARY_API_FAMILY = "mapillary-graph-api-v4"
MAPILLARY_API_FIELDS = (
    "id",
    "computed_geometry",
    "captured_at",
    "compass_angle",
    "computed_compass_angle",
    "sequence",
    "creator",
    "width",
    "height",
    "thumb_1024_url",
)
MAPILLARY_ATTRIBUTION_URL = "https://www.mapillary.com/"
MAPILLARY_LICENSE_IDENTIFIER = "CC-BY-SA-4.0"
MAPILLARY_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
MAPILLARY_PILOT_ROOT = r"C:\AtlasLensPilot\mapillary-demo"
MAX_BBOX_AREA_SQUARE_DEGREES = 0.01
MAX_PILOT_IMAGES = 2_000
MAX_PILOT_RAW_BYTES = 2 * 1024 * 1024 * 1024

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BoundingBox(StrictModel):
    west: float = Field(ge=-180.0, le=180.0)
    south: float = Field(ge=-90.0, le=90.0)
    east: float = Field(ge=-180.0, le=180.0)
    north: float = Field(ge=-90.0, le=90.0)

    @model_validator(mode="after")
    def validate_order_and_area(self) -> BoundingBox:
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("bounding box must be southwest to northeast")
        if self.area_square_degrees >= MAX_BBOX_AREA_SQUARE_DEGREES:
            raise ValueError("bounding box must be smaller than the official area cap")
        return self

    @property
    def area_square_degrees(self) -> float:
        return (self.east - self.west) * (self.north - self.south)

    def as_query_value(self) -> str:
        return ",".join(
            format(value, ".7f") for value in (self.west, self.south, self.east, self.north)
        )


class AreaOfInterest(StrictModel):
    aoi_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    version: str = Field(pattern=r"^v[1-9][0-9]*$")
    display_name: str = Field(min_length=3, max_length=120)
    region_kind: Literal["urban", "corridor"]
    province: str = Field(min_length=2, max_length=80)
    city: str | None = Field(default=None, min_length=2, max_length=80)
    spatial_cell_size_degrees: float = Field(default=0.005, gt=0.0, le=0.05)
    tiles: tuple[BoundingBox, ...] = Field(min_length=1, max_length=64)

    @field_validator("tiles")
    @classmethod
    def require_unique_tiles(cls, value: tuple[BoundingBox, ...]) -> tuple[BoundingBox, ...]:
        identities = {tile.as_query_value() for tile in value}
        if len(identities) != len(value):
            raise ValueError("AOI tiles must be unique")
        return value


class AoiCatalog(StrictModel):
    schema_version: Literal["atlaslens-mapillary-aoi-v1"] = "atlaslens-mapillary-aoi-v1"
    catalog_version: Literal["phase3b3-aoi-v1"] = "phase3b3-aoi-v1"
    official_api_base_url: Literal["https://graph.mapillary.com"] = "https://graph.mapillary.com"
    official_bbox_area_limit_square_degrees: float = Field(default=0.01, ge=0.01, le=0.01)
    aois: tuple[AreaOfInterest, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def require_phase_regions(self) -> AoiCatalog:
        expected = {
            "kayseri-urban-v1",
            "ankara-urban-v1",
            "sivas-urban-v1",
            "kayseri-ankara-corridor-v1",
            "kayseri-sivas-corridor-v1",
        }
        if {aoi.aoi_id for aoi in self.aois} != expected:
            raise ValueError("catalog must contain exactly the five Phase 3B3 AOIs")
        return self

    def by_id(self, aoi_id: str) -> AreaOfInterest:
        for aoi in self.aois:
            if aoi.aoi_id == aoi_id:
                return aoi
        raise ValueError("AOI is not in the versioned catalog")


class ClientLimits(StrictModel):
    request_cap: int = Field(default=2_500, ge=1, le=10_000)
    page_cap: int = Field(default=300, ge=1, le=2_000)
    metadata_item_cap: int = Field(default=20_000, ge=1, le=100_000)
    page_size: int = Field(default=100, ge=1, le=1_000)
    response_byte_cap: int = Field(default=16 * 1024 * 1024, ge=1_024, le=64 * 1024 * 1024)
    raw_download_byte_cap: int = Field(
        default=MAX_PILOT_RAW_BYTES,
        ge=1,
        le=MAX_PILOT_RAW_BYTES,
    )
    image_cap: int = Field(default=MAX_PILOT_IMAGES, ge=1, le=MAX_PILOT_IMAGES)
    max_image_bytes: int = Field(default=8 * 1024 * 1024, ge=1_024, le=64 * 1024 * 1024)
    concurrency: int = Field(default=2, ge=1, le=4)
    timeout_seconds: float = Field(default=20.0, gt=0.0, le=120.0)
    retry_cap: int = Field(default=4, ge=0, le=8)
    backoff_base_seconds: float = Field(default=0.5, ge=0.0, le=10.0)
    backoff_cap_seconds: float = Field(default=30.0, gt=0.0, le=60.0)


class GeoPoint(StrictModel):
    type: Literal["Point"] = "Point"
    coordinates: tuple[float, float]

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, value: tuple[float, float]) -> tuple[float, float]:
        longitude, latitude = value
        if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
            raise ValueError("computed geometry is outside WGS84 bounds")
        return value


class ImageMetadata(StrictModel):
    mapillary_image_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    computed_geometry: GeoPoint
    captured_at: datetime
    compass_angle: float | None = Field(default=None, ge=0.0, lt=360.0)
    sequence_id: str | None = Field(default=None, max_length=256)
    creator_id: str | None = Field(default=None, max_length=256)
    width_px: int | None = Field(default=None, gt=0, le=200_000)
    height_px: int | None = Field(default=None, gt=0, le=200_000)

    @field_validator("captured_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("capture timestamp must include a timezone")
        return value


class CoverageSummary(StrictModel):
    aoi_id: str
    aoi_version: str
    display_name: str
    region_kind: Literal["urban", "corridor"]
    province: str
    city: str | None
    image_count: int = Field(ge=0)
    rejected_item_count: int = Field(ge=0)
    sequence_count: int = Field(ge=0)
    contributor_count: int = Field(ge=0)
    capture_year_distribution: dict[str, int]
    spatial_cell_count: int = Field(ge=0)
    compass_direction_bins: dict[str, int]
    approximate_road_km: float = Field(ge=0.0)
    road_length_semantics: Literal["bounded_sequence_track_sum_not_unique_road_length"] = (
        "bounded_sequence_track_sum_not_unique_road_length"
    )
    approximate_images_per_road_km: float = Field(ge=0.0)
    approximate_sequence_density: float = Field(ge=0.0)
    estimated_selected_download_bytes: int = Field(ge=0)
    selected_download_size_semantics: Literal["configured_per_image_hard_cap_upper_bound"] = (
        "configured_per_image_hard_cap_upper_bound"
    )
    expected_descriptor_bytes: int = Field(ge=0)
    expected_index_bytes: int = Field(ge=0)
    index_size_semantics: Literal[
        "faiss_float32_vector_payload_only_excludes_metadata_overhead"
    ] = "faiss_float32_vector_payload_only_excludes_metadata_overhead"
    eligible_for_selection: bool
    metadata_sha256: Sha256


class CoverageAudit(StrictModel):
    schema_version: Literal["atlaslens-mapillary-coverage-audit-v1"] = (
        "atlaslens-mapillary-coverage-audit-v1"
    )
    catalog_version: str
    api_family: Literal["mapillary-graph-api-v4"] = "mapillary-graph-api-v4"
    api_base_url: Literal["https://graph.mapillary.com"] = "https://graph.mapillary.com"
    api_fields: tuple[str, ...] = MAPILLARY_API_FIELDS[:-1]
    audited_at: datetime
    request_count: int = Field(ge=0)
    page_count: int = Field(ge=0)
    regions: tuple[CoverageSummary, ...] = Field(min_length=5, max_length=5)
    selected_aoi_id: str | None = None
    selection_reason: str
    imagery_downloaded: Literal[False] = False

    @field_validator("audited_at")
    @classmethod
    def require_aware_audit_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("audit timestamp must include a timezone")
        return value


class AcquisitionPlan(StrictModel):
    schema_version: Literal["atlaslens-mapillary-acquisition-plan-v1"] = (
        "atlaslens-mapillary-acquisition-plan-v1"
    )
    aoi_id: str
    aoi_version: str
    coverage_audit_sha256: Sha256
    source_policy_receipt_sha256: Sha256
    target_reference_images: int = Field(default=1_500, ge=1, le=1_900)
    target_holdout_images: int = Field(default=100, ge=1, le=500)
    hard_image_cap: int = Field(default=MAX_PILOT_IMAGES, ge=1, le=MAX_PILOT_IMAGES)
    hard_raw_byte_cap: int = Field(default=MAX_PILOT_RAW_BYTES, ge=1, le=MAX_PILOT_RAW_BYTES)
    hard_request_cap: int = Field(default=2_500, ge=1, le=10_000)
    concurrency: int = Field(default=2, ge=1, le=4)
    created_at: datetime

    @model_validator(mode="after")
    def require_bounded_total(self) -> AcquisitionPlan:
        if self.target_reference_images + self.target_holdout_images > self.hard_image_cap:
            raise ValueError("planned reference and holdout count exceeds the image cap")
        return self

    @field_validator("created_at")
    @classmethod
    def require_aware_plan_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("plan timestamp must include a timezone")
        return value


def _require_safe_persistent_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.hostname:
        raise ValueError("source URL must be absolute credential-free HTTPS")
    sensitive = {"access_token", "token", "authorization", "signature", "sig"}
    if any(key.lower() in sensitive for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
        raise ValueError("temporary or credential-shaped URL is not persistent metadata")
    return value


class AcquiredImage(StrictModel):
    mapillary_image_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    computed_geometry: GeoPoint
    captured_at: datetime
    compass_angle: float | None = Field(default=None, ge=0.0, lt=360.0)
    sequence_id: str | None = Field(default=None, max_length=256)
    creator_id: str | None = Field(default=None, max_length=256)
    width_px: int = Field(gt=0, le=200_000)
    height_px: int = Field(gt=0, le=200_000)
    source_page_url: str
    attribution_text: str = Field(min_length=3, max_length=512)
    license_identifier: Literal["CC-BY-SA-4.0"] = "CC-BY-SA-4.0"
    license_url: Literal["https://creativecommons.org/licenses/by-sa/4.0/"] = (
        "https://creativecommons.org/licenses/by-sa/4.0/"
    )
    acquired_at: datetime
    raw_sha256: Sha256
    normalized_sha256: Sha256
    relative_path: str
    byte_size: int = Field(gt=0, le=64 * 1024 * 1024)
    normalized_byte_size: int = Field(gt=0, le=64 * 1024 * 1024)
    reconciliation_state: Literal["active", "missing_remote", "hash_mismatch"] = "active"

    @field_validator("source_page_url")
    @classmethod
    def validate_source_page(cls, value: str) -> str:
        value = _require_safe_persistent_url(value)
        parsed = urlsplit(value)
        if parsed.hostname not in {"mapillary.com", "www.mapillary.com"}:
            raise ValueError("source page must be a Mapillary HTTPS page")
        return value

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not value or ".." in path.parts or "\\" in value:
            raise ValueError("asset path must be a contained POSIX relative path")
        return value

    @field_validator("captured_at", "acquired_at")
    @classmethod
    def require_aware_asset_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("asset timestamps must include a timezone")
        return value


class AcquisitionManifest(StrictModel):
    schema_version: Literal["atlaslens-mapillary-acquisition-v1"] = (
        "atlaslens-mapillary-acquisition-v1"
    )
    aoi_id: str
    aoi_version: str
    api_family: Literal["mapillary-graph-api-v4"] = "mapillary-graph-api-v4"
    api_base_url: Literal["https://graph.mapillary.com"] = "https://graph.mapillary.com"
    api_fields: tuple[str, ...] = MAPILLARY_API_FIELDS[:-1]
    license_identifier: Literal["CC-BY-SA-4.0"] = "CC-BY-SA-4.0"
    license_url: Literal["https://creativecommons.org/licenses/by-sa/4.0/"] = (
        "https://creativecommons.org/licenses/by-sa/4.0/"
    )
    attribution_url: Literal["https://www.mapillary.com/"] = "https://www.mapillary.com/"
    source_policy_receipt_sha256: Sha256
    coverage_audit_sha256: Sha256
    created_at: datetime
    request_count: int = Field(ge=0)
    page_count: int = Field(ge=0)
    downloaded_bytes: int = Field(ge=0, le=MAX_PILOT_RAW_BYTES)
    assets: tuple[AcquiredImage, ...] = Field(max_length=MAX_PILOT_IMAGES)

    @model_validator(mode="after")
    def require_unique_images_and_accounting(self) -> AcquisitionManifest:
        identifiers = {asset.mapillary_image_id for asset in self.assets}
        if len(identifiers) != len(self.assets):
            raise ValueError("acquisition manifest contains duplicate Mapillary IDs")
        if sum(asset.byte_size for asset in self.assets) != self.downloaded_bytes:
            raise ValueError("download byte accounting does not match the asset inventory")
        return self


class AcquisitionCheckpoint(StrictModel):
    schema_version: Literal["atlaslens-mapillary-acquisition-checkpoint-v1"] = (
        "atlaslens-mapillary-acquisition-checkpoint-v1"
    )
    plan_sha256: Sha256
    status: Literal["in_progress", "completed", "cancelled", "failed"]
    completed_image_ids: tuple[str, ...]
    downloaded_bytes: int = Field(ge=0, le=MAX_PILOT_RAW_BYTES)
    request_count: int = Field(ge=0)
    page_count: int = Field(ge=0)
    updated_at: datetime
    failure_code: str | None = None


class CleanupReport(StrictModel):
    schema_version: Literal["atlaslens-mapillary-cleanup-report-v1"] = (
        "atlaslens-mapillary-cleanup-report-v1"
    )
    mode: Literal["dry_run", "executed"]
    candidate_count: int = Field(ge=0)
    candidate_bytes: int = Field(ge=0)
    deleted_count: int = Field(ge=0)
    deleted_bytes: int = Field(ge=0)
    retained_count: int = Field(ge=0, le=20)
    remaining_bytes: int = Field(ge=0)
    materialized_candidate_count: int = Field(default=0, ge=0)
    materialized_candidate_bytes: int = Field(default=0, ge=0)
    materialized_deleted_count: int = Field(default=0, ge=0)
    materialized_deleted_bytes: int = Field(default=0, ge=0)


__all__ = [
    "AoiCatalog",
    "AcquiredImage",
    "AcquisitionCheckpoint",
    "AcquisitionManifest",
    "AcquisitionPlan",
    "AreaOfInterest",
    "BoundingBox",
    "CleanupReport",
    "ClientLimits",
    "CoverageAudit",
    "CoverageSummary",
    "GeoPoint",
    "ImageMetadata",
    "MAPILLARY_API_BASE_URL",
    "MAPILLARY_API_FAMILY",
    "MAPILLARY_API_FIELDS",
    "MAPILLARY_ATTRIBUTION_URL",
    "MAPILLARY_LICENSE_IDENTIFIER",
    "MAPILLARY_LICENSE_URL",
    "MAPILLARY_PILOT_ROOT",
    "MAX_PILOT_IMAGES",
    "MAX_PILOT_RAW_BYTES",
]
