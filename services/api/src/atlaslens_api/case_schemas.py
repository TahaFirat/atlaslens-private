from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from atlaslens_api.cases.domain import CaseDomainError, validate_https_url

type JsonValue = str | int | float | bool | None


class CaseStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, from_attributes=True)


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


class AuditActorType(StrEnum):
    OPERATOR = "operator"
    SYSTEM = "system"
    MODEL = "model"


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ActorId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]*$",
    ),
]
WorkspaceId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
RetentionPolicyId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
]
Sha256Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
CountryCode = Annotated[str, Field(pattern=r"^[A-Z]{2}$")]

_FORBIDDEN_PAYLOAD_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "file_path",
    "image_bytes",
    "local_path",
    "ocr_text",
    "original_filename",
    "password",
    "prompt",
    "raw_image",
    "raw_ocr",
    "secret",
    "token",
}
_SENSITIVE_KEY_FRAGMENTS = (
    "apikey",
    "authorization",
    "credential",
    "filepath",
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


def _validate_bounded_safe_json(value: dict[str, JsonValue], *, maximum_bytes: int) -> None:
    for key in value:
        normalized = key.strip().lower().replace("-", "_")
        compact = re.sub(r"[^a-z0-9]", "", normalized)
        if normalized in _FORBIDDEN_PAYLOAD_KEYS or any(
            fragment in compact for fragment in _SENSITIVE_KEY_FRAGMENTS
        ):
            raise ValueError("payload contains a prohibited sensitive field")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValueError("payload exceeds the allowed serialized size")


def sanitize_filename_display(value: str) -> str:
    leaf = value.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join("_" if unicodedata.category(char).startswith("C") else char for char in leaf)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    cleaned = re.sub(r"[^\w .()\[\]-]", "_", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip(" .")
    if not cleaned:
        cleaned = "unnamed-image"
    return cleaned[:160].rstrip(" .") or "unnamed-image"


def _validate_https_metadata_url(value: str | None) -> str | None:
    try:
        return validate_https_url(value, "metadata_url")
    except CaseDomainError as error:
        raise ValueError(str(error)) from error


class CaseCreateRequest(CaseStrictModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
    description: (
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=2_000)] | None
    ) = None
    purpose: CasePurpose
    purpose_detail: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
        | None
    ) = None
    source_context: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)
    ]
    sensitivity: CaseSensitivity
    authorization_attested: Literal[True]
    retention_policy: RetentionPolicyId
    created_by_actor_id: ActorId

    @model_validator(mode="after")
    def validate_purpose_detail(self) -> Self:
        if self.purpose == CasePurpose.OTHER and self.purpose_detail is None:
            raise ValueError("purpose_detail is required when purpose is other")
        return self


class CasePatchRequest(CaseStrictModel):
    actor_id: ActorId
    expected_version: int = Field(ge=1)
    title: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
        | None
    ) = None
    description: (
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=2_000)] | None
    ) = None
    purpose: CasePurpose | None = None
    purpose_detail: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
        | None
    ) = None
    source_context: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)]
        | None
    ) = None
    status: CaseStatus | None = None
    sensitivity: CaseSensitivity | None = None
    retention_policy: RetentionPolicyId | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> Self:
        changed = self.model_fields_set - {"actor_id", "expected_version"}
        if not changed:
            raise ValueError("at least one case field must be supplied")
        non_nullable = {
            "title",
            "purpose",
            "source_context",
            "status",
            "sensitivity",
            "retention_policy",
        }
        if any(
            field in self.model_fields_set and getattr(self, field) is None
            for field in non_nullable
        ):
            raise ValueError("non-nullable case fields cannot be cleared")
        if self.purpose == CasePurpose.OTHER and not self.purpose_detail:
            raise ValueError("purpose_detail is required when purpose is other")
        return self


class CaseView(CaseStrictModel):
    id: UUID
    workspace_id: WorkspaceId
    title: Annotated[str, Field(min_length=1, max_length=160)]
    description: Annotated[str, Field(max_length=2_000)] | None = None
    purpose: CasePurpose
    purpose_detail: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    source_context: Annotated[str, Field(min_length=1, max_length=1_000)]
    status: CaseStatus
    sensitivity: CaseSensitivity
    authorization_attested: bool
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    created_by_actor_id: ActorId
    retention_policy: RetentionPolicyId
    version: int = Field(ge=1)
    media_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    hypothesis_count: int = Field(ge=0)
    adjudication_count: int = Field(ge=0)
    latest_adjudication_decision: AdjudicationDecision | None = None


class CasePage(CaseStrictModel):
    items: list[CaseView] = Field(max_length=100)
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    ordering: Literal["updated_at_desc_id_desc"] = "updated_at_desc_id_desc"


class CaseMediaCreateRequest(CaseStrictModel):
    actor_id: ActorId
    media_type: Literal[MediaType.IMAGE]
    source_type: MediaSourceType
    original_filename_display: Annotated[str, Field(min_length=1, max_length=512)]
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    byte_size: int = Field(gt=0, le=104_857_600)
    sha256: Sha256Digest
    captured_at: datetime | None = None
    source_url: Annotated[str, Field(min_length=9, max_length=800)] | None = None
    archive_url: Annotated[str, Field(min_length=9, max_length=800)] | None = None
    source_description: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)]
        | None
    ) = None
    authorization_attested: Literal[True]
    storage_state: MediaStorageState

    @field_validator("original_filename_display")
    @classmethod
    def sanitize_filename(cls, value: str) -> str:
        return sanitize_filename_display(value)

    @field_validator("source_url", "archive_url")
    @classmethod
    def validate_metadata_url(cls, value: str | None) -> str | None:
        return _validate_https_metadata_url(value)

    @model_validator(mode="after")
    def validate_source_semantics(self) -> Self:
        if self.source_type == MediaSourceType.SOURCE_URL and self.source_url is None:
            raise ValueError("source_url is required for source_url media")
        if self.source_type == MediaSourceType.EXTERNAL_ARCHIVE and self.archive_url is None:
            raise ValueError("archive_url is required for external_archive media")
        if self.storage_state == MediaStorageState.EXTERNALLY_MANAGED and not (
            self.source_url or self.archive_url
        ):
            raise ValueError("externally managed media requires a source or archive URL")
        return self


class CaseMediaView(CaseStrictModel):
    id: UUID
    case_id: UUID
    analysis_id: UUID | None = None
    media_type: MediaType
    source_type: MediaSourceType
    original_filename_display: Annotated[str, Field(min_length=1, max_length=160)]
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    byte_size: int = Field(gt=0, le=104_857_600)
    sha256: Sha256Digest
    captured_at: datetime | None = None
    received_at: datetime
    source_url: Annotated[str, Field(min_length=9, max_length=800)] | None = None
    archive_url: Annotated[str, Field(min_length=9, max_length=800)] | None = None
    source_description: Annotated[str, Field(min_length=1, max_length=1_000)] | None = None
    authorization_attested: bool
    storage_state: MediaStorageState
    created_at: datetime
    analysis_status_at_link: Annotated[str, Field(min_length=1, max_length=40)] | None = None
    analysis_created_at: datetime | None = None
    analysis_expires_at: datetime | None = None
    materialized_at: datetime | None = None

    @field_validator("source_url", "archive_url")
    @classmethod
    def validate_metadata_url(cls, value: str | None) -> str | None:
        return _validate_https_metadata_url(value)


class CaseMediaPage(CaseStrictModel):
    items: list[CaseMediaView] = Field(max_length=100)
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    ordering: Literal["created_at_asc_id_asc"] = "created_at_asc_id_asc"


class AnalysisLinkRequest(CaseStrictModel):
    media_id: UUID
    actor_id: ActorId


class MaterializeEvidenceRequest(CaseStrictModel):
    actor_id: ActorId
    media_id: UUID


class MaterializationView(CaseStrictModel):
    case_id: UUID
    media_id: UUID
    analysis_id: UUID
    created: bool
    evidence_created: int = Field(ge=0, le=10_000)
    hypotheses_created: int = Field(ge=0, le=1_000)
    already_materialized: bool
    evidence_ids: list[UUID] = Field(max_length=10_000)
    hypothesis_ids: list[UUID] = Field(max_length=1_000)


class EvidenceView(CaseStrictModel):
    id: UUID
    case_id: UUID
    media_id: UUID
    analysis_id: UUID
    evidence_type: EvidenceType
    provider: Annotated[str, Field(min_length=1, max_length=120)]
    provider_family: Annotated[str, Field(min_length=1, max_length=120)]
    summary: Annotated[str, Field(min_length=1, max_length=1_000)]
    structured_payload: dict[str, JsonValue] = Field(max_length=64)
    provenance: dict[str, JsonValue] = Field(max_length=64)
    observed_at: datetime
    created_at: datetime
    immutable_source_hash: Sha256Digest

    @field_validator("structured_payload", "provenance")
    @classmethod
    def validate_safe_payload(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _validate_bounded_safe_json(value, maximum_bytes=32_768)
        return value


class EvidencePage(CaseStrictModel):
    items: list[EvidenceView] = Field(max_length=100)
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    ordering: Literal["created_at_asc_id_asc"] = "created_at_asc_id_asc"


class AdjudicationView(CaseStrictModel):
    id: UUID
    case_id: UUID
    hypothesis_id: UUID
    actor_id: ActorId
    decision: AdjudicationDecision
    rationale: Annotated[str, Field(min_length=1, max_length=2_000)]
    created_at: datetime
    supersedes_adjudication_id: UUID | None = None


class HypothesisView(CaseStrictModel):
    id: UUID
    case_id: UUID
    media_id: UUID
    analysis_id: UUID
    origin: HypothesisOrigin
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_m: float = Field(gt=0, le=20_050_000)
    country_code: CountryCode | None = None
    region_name: Annotated[str, Field(min_length=1, max_length=160)] | None = None
    locality_name: Annotated[str, Field(min_length=1, max_length=160)] | None = None
    rank: int | None = Field(default=None, ge=1, le=10_000)
    confidence_label: Annotated[str, Field(min_length=1, max_length=80)] | None = None
    calibration_state: CalibrationState
    supporting_evidence_ids: list[UUID] = Field(max_length=256)
    model_family_groups: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(
        max_length=32
    )
    created_at: datetime
    supersedes_hypothesis_id: UUID | None = None
    adjudications: list[AdjudicationView] = Field(default_factory=list, max_length=100)
    adjudication_count: int = Field(default=0, ge=0)
    latest_adjudication: AdjudicationView | None = None


class HypothesisPage(CaseStrictModel):
    items: list[HypothesisView] = Field(max_length=100)
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    ordering: Literal["created_at_asc_id_asc"] = "created_at_asc_id_asc"


class AdjudicationCreateRequest(CaseStrictModel):
    actor_id: ActorId
    decision: AdjudicationDecision
    rationale: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)
    ]
    supersedes_adjudication_id: UUID | None = None

    @model_validator(mode="after")
    def validate_withdrawal(self) -> Self:
        if (
            self.decision == AdjudicationDecision.WITHDRAWN
            and self.supersedes_adjudication_id is None
        ):
            raise ValueError("withdrawn decisions must supersede an adjudication")
        return self


class OperatorHypothesisCreateRequest(CaseStrictModel):
    actor_id: ActorId
    media_id: UUID
    analysis_id: UUID
    supersedes_hypothesis_id: UUID
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_m: float = Field(gt=0, le=20_050_000)
    country_code: CountryCode | None = None
    region_name: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
        | None
    ) = None
    locality_name: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
        | None
    ) = None
    supporting_evidence_ids: list[UUID] = Field(default_factory=list, max_length=256)
    decision: AdjudicationDecision
    rationale: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)
    ]

    @field_validator("country_code")
    @classmethod
    def normalize_country_code(cls, value: str | None) -> str | None:
        return value.upper() if value is not None else None

    @model_validator(mode="after")
    def validate_operator_correction(self) -> Self:
        if self.decision == AdjudicationDecision.WITHDRAWN:
            raise ValueError("a new operator correction cannot be withdrawn")
        if len(set(self.supporting_evidence_ids)) != len(self.supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids must be unique")
        return self


class OperatorCorrectionView(CaseStrictModel):
    hypothesis: HypothesisView
    adjudication: AdjudicationView


class AuditEventView(CaseStrictModel):
    id: UUID
    case_id: UUID
    sequence_number: int = Field(ge=1)
    event_type: Annotated[str, Field(min_length=1, max_length=120)]
    actor_id: ActorId
    actor_type: AuditActorType
    payload: dict[str, JsonValue] = Field(max_length=64)
    created_at: datetime
    previous_event_hash: Sha256Digest
    event_hash: Sha256Digest

    @field_validator("payload")
    @classmethod
    def validate_safe_payload(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _validate_bounded_safe_json(value, maximum_bytes=32_768)
        return value


class AuditEventPage(CaseStrictModel):
    items: list[AuditEventView] = Field(max_length=100)
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    ordering: Literal["sequence_number_asc"] = "sequence_number_asc"
    integrity_scope: Literal["tamper_evident_application_history"] = (
        "tamper_evident_application_history"
    )


class AuditIntegrityView(CaseStrictModel):
    case_id: UUID
    valid: bool
    checked_event_count: int = Field(ge=0)
    first_invalid_sequence: int | None = Field(default=None, ge=1)
    reason: Annotated[str, Field(min_length=1, max_length=160)] | None = None
    integrity_scope: Literal["tamper_evident_application_history"] = (
        "tamper_evident_application_history"
    )
    legally_certified_evidence: Literal[False] = False
