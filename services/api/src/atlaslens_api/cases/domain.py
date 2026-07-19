from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Final
from urllib.parse import urlsplit
from uuid import UUID

DEFAULT_WORKSPACE_ID: Final = "local-default"
GENESIS_AUDIT_HASH: Final = "0" * 64

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class CasePurpose(StrEnum):
    JOURNALISM = "journalism"
    HUMANITARIAN = "humanitarian"
    DISASTER_RESPONSE = "disaster_response"
    INSURANCE = "insurance"
    AUTHORIZED_SECURITY_RESEARCH = "authorized_security_research"
    OTHER = "other"


class CaseStatus(StrEnum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


class CaseSensitivity(StrEnum):
    STANDARD = "standard"
    SENSITIVE = "sensitive"
    CONFLICT_RELATED = "conflict_related"


class MediaType(StrEnum):
    IMAGE = "image"


class MediaSourceType(StrEnum):
    UPLOAD = "upload"
    SOURCE_URL = "source_url"
    EXTERNAL_ARCHIVE = "external_archive"
    OTHER = "other"


class MediaStorageState(StrEnum):
    EPHEMERAL = "ephemeral"
    DELETED_AFTER_ANALYSIS = "deleted_after_analysis"
    UNAVAILABLE = "unavailable"
    EXTERNALLY_MANAGED = "externally_managed"


class EvidenceType(StrEnum):
    METADATA = "metadata"
    OCR = "ocr"
    VISUAL_CLUE = "visual_clue"
    MODEL_HYPOTHESIS = "model_hypothesis"
    RETRIEVAL_MATCH = "retrieval_match"
    MAP_EVIDENCE = "map_evidence"
    ANALYST_NOTE = "analyst_note"


class HypothesisOrigin(StrEnum):
    MODEL = "model"
    OPERATOR_CORRECTION = "operator_correction"
    IMPORTED = "imported"


class CalibrationState(StrEnum):
    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"
    NOT_APPLICABLE = "not_applicable"


class AdjudicationDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    WITHDRAWN = "withdrawn"


class ActorType(StrEnum):
    OPERATOR = "operator"
    SYSTEM = "system"
    MODEL = "model"


class CaseDomainError(ValueError):
    code = "case_invalid"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code or type(self).code


class CaseNotFoundError(CaseDomainError):
    code = "case_not_found"


class CaseConflictError(CaseDomainError):
    code = "case_conflict"


class CaseRelationshipError(CaseDomainError):
    code = "case_relationship_invalid"


class AnalysisNotFoundForCaseError(CaseDomainError):
    code = "analysis_not_found"


class AnalysisNotCompletedError(CaseDomainError):
    code = "analysis_not_completed"


class AuditIntegrityError(CaseDomainError):
    code = "audit_integrity_error"


class UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = UnsetType()


@dataclass(frozen=True, slots=True)
class CaseRecord:
    id: UUID
    workspace_id: str
    title: str
    description: str | None
    purpose: CasePurpose
    purpose_detail: str | None
    source_context: str
    authorization_attested: bool
    status: CaseStatus
    sensitivity: CaseSensitivity
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None
    created_by_actor_id: str
    retention_policy: str
    version: int
    media_count: int = 0
    evidence_count: int = 0
    hypothesis_count: int = 0
    adjudication_count: int = 0
    latest_adjudication_decision: AdjudicationDecision | None = None


@dataclass(frozen=True, slots=True)
class CaseUpdate:
    title: str | UnsetType = UNSET
    description: str | None | UnsetType = UNSET
    purpose: CasePurpose | UnsetType = UNSET
    purpose_detail: str | None | UnsetType = UNSET
    source_context: str | UnsetType = UNSET
    status: CaseStatus | UnsetType = UNSET
    sensitivity: CaseSensitivity | UnsetType = UNSET
    retention_policy: str | UnsetType = UNSET
    expected_version: int | None = None


@dataclass(frozen=True, slots=True)
class CaseMediaRecord:
    id: UUID
    case_id: UUID
    analysis_id: UUID | None
    media_type: MediaType
    source_type: MediaSourceType
    original_filename_display: str
    mime_type: str
    byte_size: int
    sha256: str
    captured_at: datetime | None
    received_at: datetime
    source_url: str | None
    archive_url: str | None
    source_description: str | None
    authorization_attested: bool
    storage_state: MediaStorageState
    analysis_status_at_link: str | None
    analysis_created_at: datetime | None
    analysis_expires_at: datetime | None
    materialized_at: datetime | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    id: UUID
    case_id: UUID
    media_id: UUID
    analysis_id: UUID
    evidence_type: EvidenceType
    provider: str
    provider_family: str
    summary: str
    structured_payload: dict[str, JsonValue]
    provenance: dict[str, JsonValue]
    observed_at: datetime
    created_at: datetime
    immutable_source_hash: str


@dataclass(frozen=True, slots=True)
class LocationHypothesisRecord:
    id: UUID
    case_id: UUID
    media_id: UUID
    analysis_id: UUID
    origin: HypothesisOrigin
    latitude: float
    longitude: float
    uncertainty_radius_m: float
    country_code: str | None
    region_name: str | None
    locality_name: str | None
    rank: int | None
    confidence_label: str | None
    calibration_state: CalibrationState
    supporting_evidence_ids: tuple[UUID, ...]
    model_family_groups: tuple[str, ...]
    created_at: datetime
    supersedes_hypothesis_id: UUID | None


@dataclass(frozen=True, slots=True)
class AdjudicationRecord:
    id: UUID
    case_id: UUID
    hypothesis_id: UUID
    actor_id: str
    decision: AdjudicationDecision
    rationale: str
    created_at: datetime
    supersedes_adjudication_id: UUID | None


@dataclass(frozen=True, slots=True)
class AuditEventRecord:
    id: UUID
    case_id: UUID
    sequence_number: int
    event_type: str
    actor_id: str
    actor_type: ActorType
    payload: dict[str, JsonValue]
    created_at: datetime
    previous_event_hash: str
    event_hash: str


@dataclass(frozen=True, slots=True)
class AuditIntegrityResult:
    valid: bool
    event_count: int
    verified_through_sequence: int
    failure_code: str | None = None
    failure_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    created: bool
    evidence: tuple[EvidenceRecord, ...]
    hypotheses: tuple[LocationHypothesisRecord, ...]


@dataclass(frozen=True, slots=True)
class OperatorCorrectionResult:
    hypothesis: LocationHypothesisRecord
    adjudication: AdjudicationRecord


_DISPLAY_FILENAME_MAX = 160
_METADATA_URL_MAX = 800
_HEX_64 = re.compile(r"^[a-f0-9]{64}$")
_COUNTRY_CODE = re.compile(r"^[A-Z]{2}$")
_MAPILLARY_ATTRIBUTION_QUERY = re.compile(r"^pKey=[A-Za-z0-9_-]{1,128}$")
_SENSITIVE_KEY_FRAGMENTS = (
    "apikey",
    "authorization",
    "credential",
    "filesystempath",
    "imagebytes",
    "originalfilename",
    "password",
    "prompt",
    "rawbytes",
    "rawimage",
    "rawocr",
    "secret",
    "storagepath",
    "token",
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def require_text(value: str, field: str, *, max_length: int) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise CaseDomainError(f"{field} is required", code=f"{field}_required")
    if len(normalized) > max_length:
        raise CaseDomainError(f"{field} is too long", code=f"{field}_too_long")
    return normalized


def optional_text(value: str | None, field: str, *, max_length: int) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise CaseDomainError(f"{field} is too long", code=f"{field}_too_long")
    return normalized


def validate_case_values(
    *,
    title: str,
    description: str | None,
    purpose: CasePurpose,
    purpose_detail: str | None,
    source_context: str,
    authorization_attested: bool,
    created_by_actor_id: str,
    retention_policy: str,
    workspace_id: str,
) -> tuple[str, str | None, str | None, str, str, str, str]:
    if not authorization_attested:
        raise CaseDomainError(
            "authorization attestation is required", code="authorization_attestation_required"
        )
    normalized_title = require_text(title, "title", max_length=200)
    normalized_description = optional_text(description, "description", max_length=4_000)
    normalized_detail = optional_text(purpose_detail, "purpose_detail", max_length=1_000)
    if purpose is CasePurpose.OTHER and normalized_detail is None:
        raise CaseDomainError(
            "purpose_detail is required for other", code="purpose_detail_required"
        )
    normalized_source = require_text(source_context, "source_context", max_length=2_000)
    normalized_actor = require_text(created_by_actor_id, "created_by_actor_id", max_length=160)
    normalized_retention = require_text(retention_policy, "retention_policy", max_length=160)
    normalized_workspace = require_text(workspace_id, "workspace_id", max_length=128)
    return (
        normalized_title,
        normalized_description,
        normalized_detail,
        normalized_source,
        normalized_actor,
        normalized_retention,
        normalized_workspace,
    )


def sanitize_display_filename(value: str) -> str:
    leaf = re.split(r"[\\/]", value)[-1]
    normalized = unicodedata.normalize("NFKC", leaf)
    cleaned = "".join(
        character if character.isalnum() or character in {" ", ".", "-", "_", "(", ")"} else "_"
        for character in normalized
        if not unicodedata.category(character).startswith("C")
    )
    cleaned = " ".join(cleaned.strip(" .").split())
    if not cleaned:
        cleaned = "image"
    return cleaned[:_DISPLAY_FILENAME_MAX].rstrip(" .") or "image"


def validate_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if _HEX_64.fullmatch(normalized) is None:
        raise CaseDomainError("sha256 must be lowercase hexadecimal", code="invalid_sha256")
    return normalized


def validate_country_code(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if _COUNTRY_CODE.fullmatch(normalized) is None:
        raise CaseDomainError("country_code must be ISO alpha-2", code="invalid_country_code")
    return normalized


def validate_wgs84(latitude: float, longitude: float, uncertainty_radius_m: float) -> None:
    if not all(math.isfinite(value) for value in (latitude, longitude, uncertainty_radius_m)):
        raise CaseDomainError("location values must be finite", code="invalid_location")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise CaseDomainError("location is outside WGS84 bounds", code="invalid_location")
    if uncertainty_radius_m <= 0:
        raise CaseDomainError(
            "uncertainty radius must be positive", code="invalid_uncertainty_radius"
        )


def validate_https_url(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if len(normalized) > _METADATA_URL_MAX:
        raise CaseDomainError(f"{field} is too long", code=f"{field}_too_long")
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise CaseDomainError(
            f"{field} must be a credential-free HTTPS URL", code=f"invalid_{field}"
        ) from error
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or "#" in normalized
    ):
        raise CaseDomainError(
            f"{field} must be a credential-free HTTPS URL", code=f"invalid_{field}"
        )
    if "?" in normalized:
        canonical_mapillary_attribution = (
            hostname == "www.mapillary.com"
            and parsed.netloc.casefold() == "www.mapillary.com"
            and port is None
            and parsed.path == "/app/"
            and _MAPILLARY_ATTRIBUTION_QUERY.fullmatch(parsed.query) is not None
        )
        if not canonical_mapillary_attribution:
            raise CaseDomainError(
                f"{field} contains a prohibited query string", code=f"invalid_{field}"
            )
    return normalized


def _safe_json_value(value: object, *, key_path: tuple[str, ...], depth: int) -> JsonValue:
    if depth > 8:
        raise CaseDomainError("structured payload is too deeply nested", code="unsafe_payload")
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CaseDomainError(
                "structured payload contains non-finite value", code="unsafe_payload"
            )
        return value
    if isinstance(value, list | tuple):
        if len(value) > 256:
            raise CaseDomainError("structured payload list is too large", code="unsafe_payload")
        return [_safe_json_value(item, key_path=key_path, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 128:
            raise CaseDomainError("structured payload object is too large", code="unsafe_payload")
        result: dict[str, JsonValue] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise CaseDomainError(
                    "structured payload keys must be strings", code="unsafe_payload"
                )
            key = require_text(raw_key, "payload_key", max_length=120)
            compact = re.sub(r"[^a-z0-9]", "", key.casefold())
            if any(fragment in compact for fragment in _SENSITIVE_KEY_FRAGMENTS):
                raise CaseDomainError(
                    f"sensitive payload key is not allowed: {'.'.join((*key_path, key))}",
                    code="unsafe_payload",
                )
            result[key] = _safe_json_value(item, key_path=(*key_path, key), depth=depth + 1)
        return result
    raise CaseDomainError("structured payload contains unsupported value", code="unsafe_payload")


def safe_json_object(value: object) -> dict[str, JsonValue]:
    safe = _safe_json_value(value, key_path=(), depth=0)
    if not isinstance(safe, dict):
        raise CaseDomainError("structured payload must be an object", code="unsafe_payload")
    return safe


def canonical_json_bytes(value: object) -> bytes:
    safe = _safe_json_value(value, key_path=(), depth=0)
    return json.dumps(
        safe,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def immutable_payload_hash(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def canonical_audit_hash(
    *,
    event_id: UUID,
    case_id: UUID,
    sequence_number: int,
    event_type: str,
    actor_id: str,
    actor_type: ActorType,
    payload: dict[str, JsonValue],
    created_at: datetime,
    previous_event_hash: str,
) -> str:
    timestamp = as_utc(created_at)
    if timestamp is None:
        raise AuditIntegrityError("audit timestamp is required")
    canonical = {
        "actor_id": actor_id,
        "actor_type": actor_type.value,
        "case_id": str(case_id),
        "created_at": timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "event_id": str(event_id),
        "event_type": event_type,
        "payload": payload,
        "previous_event_hash": previous_event_hash,
        "sequence_number": sequence_number,
    }
    return sha256(canonical_json_bytes(canonical)).hexdigest()
