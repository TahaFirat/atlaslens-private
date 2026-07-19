from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"
OPAQUE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$"
VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,159}$"

Decision = Literal[
    "GO",
    "GO_WITH_ATTRIBUTION",
    "FIRST_PARTY_ONLY",
    "WRITTEN_PERMISSION_REQUIRED",
    "LEGAL_REVIEW_REQUIRED",
    "RESEARCH_ONLY",
    "BLOCKED",
    "UNKNOWN",
]
Role = Literal["train", "development", "validation", "holdout"]
Tier = Literal[
    "tier_1_national_recall",
    "tier_2_dense_corridor",
    "tier_3_locked_holdout",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_uri(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or parsed.username is not None or parsed.password is not None:
        raise ValueError("URI must be absolute and credential-free")
    return value


class DeletionRevocationState(StrictModel):
    status: Literal["ACTIVE", "QUARANTINED", "DELETION_PENDING", "DELETED", "REVOKED"]
    last_checked_at: datetime
    request_id: str | None = Field(default=None, min_length=1, max_length=160)

    @field_validator("last_checked_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(UTC)


class ProvenanceReceipt(StrictModel):
    receipt_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    source_policy_version: str = Field(min_length=1, max_length=160)
    evidence_date: date
    acquisition_method: Literal[
        "first_party_capture",
        "partner_transfer",
        "authorized_api",
        "authorized_bulk_download",
        "manual_verified_import",
    ]
    receipt_sha256: str = Field(pattern=SHA256_PATTERN)


class ManifestAsset(StrictModel):
    """Executable mirror of one Phase 3A ``manifest-schema-v1`` asset row."""

    corpus_version: str = Field(
        pattern=r"^atlaslens-turkiye-corpus-[A-Za-z0-9._-]+$",
        max_length=160,
    )
    asset_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    source_asset_id: str = Field(min_length=1, max_length=256)
    source_name: str = Field(min_length=1, max_length=160)
    source_url: str = Field(min_length=1, max_length=2048)
    contributor_or_owner: str = Field(min_length=1, max_length=320)
    capture_timestamp: datetime | None
    acquisition_timestamp: datetime
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    coordinate_accuracy_m: float = Field(gt=0, le=100_000)
    heading_degrees: float | None = Field(ge=0, lt=360)
    sequence_id: str | None = Field(min_length=1, max_length=256)
    capture_run_id: str | None = Field(min_length=1, max_length=256)
    image_sha256: str = Field(pattern=SHA256_PATTERN)
    perceptual_hash: str = Field(pattern=r"^[0-9a-f]{16,128}$")
    perceptual_hash_algorithm: Literal["dhash64-v1", "phash64-v1", "phash256-v1"]
    width_px: int = Field(ge=1, le=100_000)
    height_px: int = Field(ge=1, le=100_000)
    mime_type: Literal["image/jpeg", "image/png", "image/webp", "image/tiff"]
    asset_type: Literal["street_level", "aerial", "satellite"]
    province_code: str = Field(pattern=r"^TR-(0[1-9]|[1-7][0-9]|8[01])$")
    urbanicity: Literal[
        "urban_core",
        "suburban",
        "rural_settlement",
        "rural_road",
        "unknown",
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
    license_identifier: str = Field(min_length=1, max_length=200)
    license_url: str = Field(min_length=1, max_length=2048)
    attribution_text: str = Field(min_length=1, max_length=2000)
    source_policy_decision: Decision
    commercial_use_decision: Decision
    derivative_index_decision: Decision
    personal_data_blur_state: Literal[
        "NOT_DETECTED",
        "BLURRED_AT_SOURCE",
        "BLURRED_BY_ATLASLENS",
        "REVIEW_REQUIRED",
        "UNKNOWN",
    ]
    deletion_revocation_state: DeletionRevocationState
    provenance_receipt: ProvenanceReceipt
    spatial_split: str = Field(pattern=OPAQUE_ID_PATTERN)
    role: Role
    tier: Tier
    acquisition_ready: bool
    source_tile_id: str | None = Field(default=None, min_length=1, max_length=256)
    parent_source_asset_id: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("source_url", "license_url")
    @classmethod
    def require_uri(cls, value: str) -> str:
        return _require_uri(value)

    @field_validator("capture_timestamp", "acquisition_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def require_schema_admission_invariants(self) -> Self:
        acquisition_decisions = {"GO", "GO_WITH_ATTRIBUTION", "FIRST_PARTY_ONLY"}
        if self.acquisition_ready and (
            self.source_policy_decision not in acquisition_decisions
            or self.commercial_use_decision not in acquisition_decisions
            or self.derivative_index_decision not in acquisition_decisions
            or self.personal_data_blur_state
            not in {"NOT_DETECTED", "BLURRED_AT_SOURCE", "BLURRED_BY_ATLASLENS"}
            or self.deletion_revocation_state.status != "ACTIVE"
        ):
            raise ValueError("acquisition-ready row violates the Phase 3A schema gate")
        if (self.role == "holdout") != (self.tier == "tier_3_locked_holdout"):
            raise ValueError("holdout role and tier must agree")
        return self


class SourcePolicyEntry(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, str_strip_whitespace=True)

    source_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    source_name: str = Field(min_length=1, max_length=320)
    current_decision: Decision
    evidence_date: date


class SourcePolicyDocument(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["atlaslens-source-policy-v1"]
    policy_version: str = Field(min_length=1, max_length=160)
    evidence_date: date
    decision_values: tuple[Decision, ...]
    acquisition_ready_decisions: tuple[Decision, ...]
    sources: tuple[SourcePolicyEntry, ...]

    @model_validator(mode="after")
    def require_unique_sources_and_fail_closed_decisions(self) -> Self:
        source_ids = [entry.source_id for entry in self.sources]
        source_names = [entry.source_name for entry in self.sources]
        if len(source_ids) != len(set(source_ids)) or len(source_names) != len(
            set(source_names)
        ):
            raise ValueError("source policy identifiers and names must be unique")
        expected: set[Decision] = {"GO", "GO_WITH_ATTRIBUTION", "FIRST_PARTY_ONLY"}
        if set(self.acquisition_ready_decisions) - expected:
            raise ValueError("source policy promotes a fail-closed decision")
        return self


class IngestionConfig(StrictModel):
    schema_version: Literal["atlaslens-corpus-ingestion-config-v1"] = (
        "atlaslens-corpus-ingestion-config-v1"
    )
    asset_subdirectory: str = Field(default="assets", min_length=1, max_length=80)
    descriptor_version: str = Field(default="unbuilt-v1", pattern=VERSION_PATTERN)
    index_version: str = Field(default="unbuilt-v1", pattern=VERSION_PATTERN)
    perceptual_hash_algorithm: Literal["dhash64-v1"] = "dhash64-v1"
    near_duplicate_hamming_threshold: int = Field(default=4, ge=0, le=64)
    spatial_leakage_radius_m: float = Field(default=100.0, gt=0, le=100_000)
    max_manifest_bytes: int = Field(default=32 * 1024 * 1024, gt=0)
    max_manifest_rows: int = Field(default=100_000, gt=0, le=1_000_000)
    max_asset_bytes: int = Field(default=40 * 1024 * 1024, gt=0)
    max_asset_pixels: int = Field(default=100_000_000, gt=0)
    strict_rejections: bool = True

    @field_validator("asset_subdirectory")
    @classmethod
    def require_single_safe_directory(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("asset_subdirectory must be one safe path component")
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_json_sha256(self.model_dump(mode="json"))


RejectionStage = Literal["manifest", "rights", "identity", "deduplication", "leakage"]


class AssetRejection(StrictModel):
    asset_id: str = Field(pattern=r"^(?:[A-Za-z0-9][A-Za-z0-9._:-]{0,159}|row-[0-9]+)$")
    stage: RejectionStage
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")


class RightsAdmission(StrictModel):
    asset_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    source_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    decision: Decision
    permission_state: Literal["validated"] = "validated"
    policy_version: str = Field(min_length=1, max_length=160)
    policy_evidence_date: date


class IngestedAsset(StrictModel):
    schema_version: Literal["atlaslens-ingested-asset-v1"] = "atlaslens-ingested-asset-v1"
    asset_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    source_asset_id: str = Field(min_length=1, max_length=256)
    source_tile_id: str | None = Field(default=None, min_length=1, max_length=256)
    parent_source_asset_id: str | None = Field(default=None, min_length=1, max_length=256)
    source_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    source_name: str = Field(min_length=1, max_length=320)
    file_locator: str = Field(min_length=1, max_length=500)
    image_sha256: str = Field(pattern=SHA256_PATTERN)
    perceptual_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    perceptual_hash_algorithm: Literal["dhash64-v1"] = "dhash64-v1"
    rights_decision: Decision
    permission_state: Literal["validated"] = "validated"
    license_identifier: str = Field(min_length=1, max_length=200)
    license_url: str = Field(min_length=1, max_length=2048)
    attribution_text: str = Field(min_length=1, max_length=2000)
    provenance_receipt_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    provenance_receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    privacy_review_state: Literal[
        "NOT_DETECTED", "BLURRED_AT_SOURCE", "BLURRED_BY_ATLASLENS"
    ]
    revoked: Literal[False] = False
    role: Role
    tier: Tier
    spatial_split: str = Field(pattern=OPAQUE_ID_PATTERN)
    sampling_cell: str = Field(pattern=OPAQUE_ID_PATTERN)
    province_code: str = Field(pattern=r"^TR-(0[1-9]|[1-7][0-9]|8[01])$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    coordinate_accuracy_m: float = Field(gt=0, le=100_000)
    contributor_or_owner: str = Field(min_length=1, max_length=320)
    capture_timestamp: datetime | None
    capture_run_id: str | None = Field(default=None, min_length=1, max_length=256)
    sequence_id: str | None = Field(default=None, min_length=1, max_length=256)
    descriptor_version: str = Field(pattern=VERSION_PATTERN)
    index_version: str = Field(pattern=VERSION_PATTERN)

    @field_validator("license_url")
    @classmethod
    def require_license_uri(cls, value: str) -> str:
        return _require_uri(value)

    @field_validator("capture_timestamp")
    @classmethod
    def require_capture_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("capture timestamp must include a timezone")
        return value


class IngestedCorpus(StrictModel):
    schema_version: Literal["atlaslens-ingested-corpus-v1"] = (
        "atlaslens-ingested-corpus-v1"
    )
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    manifest_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    source_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    assets: tuple[IngestedAsset, ...]
    rejections: tuple[AssetRejection, ...]

    @model_validator(mode="after")
    def require_unique_assets(self) -> Self:
        asset_ids = [asset.asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("ingested asset IDs must be unique")
        return self


LeakageKind = Literal[
    "exact_duplicate",
    "near_duplicate",
    "sequence_cross_split",
    "sequence_cross_spatial_split",
    "capture_run_cross_split",
    "source_parent_or_tile_cross_role",
    "spatial_cross_split",
]


class LeakageFinding(StrictModel):
    kind: LeakageKind
    retained_asset_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    excluded_asset_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    distance_m: float | None = Field(default=None, ge=0)
    hamming_distance: int | None = Field(default=None, ge=0, le=64)


class LeakageReport(StrictModel):
    schema_version: Literal["atlaslens-corpus-leakage-report-v1"] = (
        "atlaslens-corpus-leakage-report-v1"
    )
    status: Literal["passed", "excluded_conflicts"]
    input_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    findings: tuple[LeakageFinding, ...]

    @model_validator(mode="after")
    def require_consistent_counts(self) -> Self:
        if self.accepted_count + self.excluded_count != self.input_count:
            raise ValueError("leakage counts do not cover the input")
        if self.excluded_count != len(self.findings):
            raise ValueError("each excluded asset must have one primary finding")
        if (self.excluded_count == 0) != (self.status == "passed"):
            raise ValueError("leakage status does not match findings")
        return self


class SplitAssignment(StrictModel):
    asset_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    image_sha256: str = Field(pattern=SHA256_PATTERN)
    role: Role
    tier: Tier
    spatial_split: str = Field(pattern=OPAQUE_ID_PATTERN)


class SplitLock(StrictModel):
    schema_version: Literal["atlaslens-corpus-split-lock-v1"] = (
        "atlaslens-corpus-split-lock-v1"
    )
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    assignments: tuple[SplitAssignment, ...]
    lock_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def verify_lock(self) -> Self:
        ordered = tuple(sorted(self.assignments, key=lambda item: item.asset_id))
        if ordered != self.assignments:
            raise ValueError("split assignments must be ordered")
        asset_ids = [item.asset_id for item in ordered]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("split assignment IDs must be unique")
        expected = canonical_json_sha256(
            {
                "schema_version": self.schema_version,
                "manifest_sha256": self.manifest_sha256,
                "config_sha256": self.config_sha256,
                "assignments": [item.model_dump(mode="json") for item in ordered],
            }
        )
        if self.lock_sha256 != expected:
            raise ValueError("split lock hash is invalid")
        return self


class RevocationImpact(StrictModel):
    schema_version: Literal["atlaslens-revocation-impact-v1"] = (
        "atlaslens-revocation-impact-v1"
    )
    asset_ids: tuple[str, ...]
    impacted_artifacts: tuple[str, ...]
    rebuild_required: Literal[True] = True
    actions: tuple[
        Literal[
            "exclude_assets",
            "rebuild_descriptors",
            "rebuild_indexes",
            "verify_checksum_inventory",
        ],
        ...,
    ]
    plan_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_semantics: Literal["operational_plan_not_legal_or_cryptographic_certification"] = (
        "operational_plan_not_legal_or_cryptographic_certification"
    )


SafeArtifactName = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$"),
]


__all__ = [
    "AssetRejection",
    "Decision",
    "DeletionRevocationState",
    "IngestedAsset",
    "IngestedCorpus",
    "IngestionConfig",
    "LeakageFinding",
    "LeakageKind",
    "LeakageReport",
    "ManifestAsset",
    "ProvenanceReceipt",
    "RevocationImpact",
    "RightsAdmission",
    "Role",
    "SHA256_PATTERN",
    "SafeArtifactName",
    "SourcePolicyDocument",
    "SourcePolicyEntry",
    "SplitAssignment",
    "SplitLock",
    "StrictModel",
    "Tier",
    "VERSION_PATTERN",
    "canonical_json_sha256",
]
