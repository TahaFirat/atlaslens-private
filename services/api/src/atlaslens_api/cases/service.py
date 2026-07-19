from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from atlaslens_api.repository import AnalysisRepository
from atlaslens_api.schemas import Analysis, AnalysisStatus, Candidate, Evidence

from .domain import (
    DEFAULT_WORKSPACE_ID,
    AdjudicationDecision,
    AdjudicationRecord,
    AnalysisNotCompletedError,
    AnalysisNotFoundForCaseError,
    AuditEventRecord,
    AuditIntegrityResult,
    CalibrationState,
    CaseDomainError,
    CaseMediaRecord,
    CasePurpose,
    CaseRecord,
    CaseRelationshipError,
    CaseSensitivity,
    CaseStatus,
    CaseUpdate,
    EvidenceRecord,
    EvidenceType,
    HypothesisOrigin,
    JsonValue,
    LocationHypothesisRecord,
    MaterializationResult,
    MediaSourceType,
    MediaStorageState,
    MediaType,
    OperatorCorrectionResult,
    UnsetType,
    as_utc,
    immutable_payload_hash,
    optional_text,
    require_text,
    safe_json_object,
    sanitize_display_filename,
    utc_now,
    validate_case_values,
    validate_country_code,
    validate_https_url,
    validate_sha256,
    validate_wgs84,
)
from .repository import EvidenceInsert, HypothesisInsert, SQLAlchemyCaseRepository


def _source_record_key(kind: str, analysis_id: UUID, source_id: str) -> str:
    digest = sha256(f"{kind}:{analysis_id}:{source_id}".encode()).hexdigest()
    return f"{kind}:{digest}"


def _record_uuid(kind: str, case_id: UUID, media_id: UUID, source_key: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"atlaslens:{kind}:{case_id}:{media_id}:{source_key}")


def _evidence_type(evidence: Evidence) -> EvidenceType:
    value = evidence.type.casefold()
    if "ocr" in value or "text" in value:
        return EvidenceType.OCR
    if "metadata" in value or "exif" in value:
        return EvidenceType.METADATA
    if "retrieval" in value or "match" in value:
        return EvidenceType.RETRIEVAL_MATCH
    if "map" in value:
        return EvidenceType.MAP_EVIDENCE
    if "model" in value or "prediction" in value:
        return EvidenceType.MODEL_HYPOTHESIS
    return EvidenceType.VISUAL_CLUE


def _normalize_evidence(
    analysis: Analysis,
    *,
    case_id: UUID,
    media_id: UUID,
    created_at: datetime,
) -> tuple[tuple[EvidenceInsert, ...], dict[str, UUID]]:
    inserts: list[EvidenceInsert] = []
    identifiers: dict[str, UUID] = {}
    for source in analysis.evidence:
        evidence_type = _evidence_type(source)
        source_key = _source_record_key("evidence", analysis.id, source.id)
        record_id = _record_uuid("evidence", case_id, media_id, source_key)
        identifiers[source.id] = record_id
        provider = require_text(source.provenance.provider_id, "provider", max_length=160)
        provider_family = require_text(
            source.provenance.provider_kind, "provider_family", max_length=160
        )
        summary = (
            "Redacted OCR evidence available"
            if evidence_type is EvidenceType.OCR
            else require_text(source.label, "evidence_summary", max_length=500)
        )
        structured_values: dict[str, JsonValue] = {
            "confidence": source.confidence,
            "confidence_basis": require_text(
                source.confidence_basis, "confidence_basis", max_length=500
            ),
            "sensitive": source.sensitive,
            "source_record_key": source_key,
        }
        if source.retrieval_context is not None:
            context = source.retrieval_context
            structured_values.update(
                {
                    "analysis_scope": context.analysis_scope,
                    "coverage_status": context.coverage_status,
                    "coverage_label": context.coverage_label,
                    "retrieval_provider": context.retrieval_provider,
                    "retrieval_scope": context.retrieval_scope,
                    "result_semantics": context.result_semantics,
                    "abstained": context.abstained,
                    "abstention_reason": context.abstention_reason,
                    "similarity_semantics": context.similarity_semantics,
                    "supported_region": context.supported_region,
                    "evidence_version": context.evidence_version,
                    "benchmark_version": context.benchmark_version,
                }
            )
        structured_payload = safe_json_object(structured_values)
        provenance = safe_json_object(
            {
                "execution_boundary": source.provenance.execution_boundary,
                "model_name": source.provenance.model_name,
                "output_schema_version": source.provenance.output_schema_version,
                "provider_id": provider,
                "provider_kind": provider_family,
                "provider_version": source.provenance.provider_version,
            }
        )
        immutable_hash = immutable_payload_hash(
            {
                "analysis_id": str(analysis.id),
                "case_id": str(case_id),
                "evidence_type": evidence_type.value,
                "media_id": str(media_id),
                "provenance": provenance,
                "provider": provider,
                "provider_family": provider_family,
                "structured_payload": structured_payload,
                "summary": summary,
            }
        )
        record = EvidenceRecord(
            id=record_id,
            case_id=case_id,
            media_id=media_id,
            analysis_id=analysis.id,
            evidence_type=evidence_type,
            provider=provider,
            provider_family=provider_family,
            summary=summary,
            structured_payload=structured_payload,
            provenance=provenance,
            observed_at=analysis.created_at,
            created_at=created_at,
            immutable_source_hash=immutable_hash,
        )
        inserts.append(EvidenceInsert(record=record, source_record_key=source_key))
    return tuple(inserts), identifiers


def _candidate_calibration(candidate: Candidate) -> tuple[CalibrationState, str | None]:
    if candidate.confidence_kind == "calibrated_probability":
        return CalibrationState.CALIBRATED, "calibrated_probability"
    if candidate.confidence_assessment is not None:
        return CalibrationState.UNCALIBRATED, candidate.confidence_assessment.label
    if candidate.confidence_kind == "uncalibrated_score":
        return CalibrationState.UNCALIBRATED, "uncalibrated"
    return CalibrationState.NOT_APPLICABLE, "source_reliability"


def _normalize_hypotheses(
    analysis: Analysis,
    *,
    case_id: UUID,
    media_id: UUID,
    evidence_identifiers: dict[str, UUID],
    created_at: datetime,
) -> tuple[HypothesisInsert, ...]:
    inserts: list[HypothesisInsert] = []
    for candidate in analysis.candidates:
        uncertainty_radius_m = candidate.radius_km * 1_000.0
        validate_wgs84(candidate.center.latitude, candidate.center.longitude, uncertainty_radius_m)
        source_key = _source_record_key("hypothesis", analysis.id, candidate.id)
        record_id = _record_uuid("hypothesis", case_id, media_id, source_key)
        calibration_state, confidence_label = _candidate_calibration(candidate)
        reverse = candidate.reverse_geocode
        country_code = validate_country_code(
            reverse.country_code if reverse is not None else candidate.country_code
        )
        supporting = tuple(
            evidence_identifiers[evidence_id] for evidence_id in candidate.evidence_ids
        )
        family_groups = tuple(
            sorted(
                {
                    require_text(provenance.provider_kind, "model_family_group", max_length=160)
                    for provenance in candidate.provenance
                }
            )
        )
        record = LocationHypothesisRecord(
            id=record_id,
            case_id=case_id,
            media_id=media_id,
            analysis_id=analysis.id,
            origin=HypothesisOrigin.MODEL,
            latitude=candidate.center.latitude,
            longitude=candidate.center.longitude,
            uncertainty_radius_m=uncertainty_radius_m,
            country_code=country_code,
            region_name=(None if reverse is None else reverse.region),
            locality_name=(None if reverse is None else reverse.city),
            rank=candidate.rank,
            confidence_label=confidence_label,
            calibration_state=calibration_state,
            supporting_evidence_ids=supporting,
            model_family_groups=family_groups,
            created_at=created_at,
            supersedes_hypothesis_id=None,
        )
        inserts.append(HypothesisInsert(record=record, source_record_key=source_key))
    return tuple(inserts)


class CaseInvestigationService:
    def __init__(
        self,
        repository: SQLAlchemyCaseRepository,
        analysis_repository: AnalysisRepository,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._repository = repository
        self._analysis_repository = analysis_repository
        self._clock = clock

    def _now(self) -> datetime:
        value = as_utc(self._clock())
        if value is None:
            raise RuntimeError("clock returned no timestamp")
        return value

    async def initialize(self) -> None:
        await self._repository.initialize()

    async def create_case(
        self,
        *,
        record_id: UUID | None = None,
        title: str,
        description: str | None,
        purpose: CasePurpose,
        purpose_detail: str | None,
        sensitivity: CaseSensitivity,
        source_context: str,
        authorization_attested: bool,
        created_by_actor_id: str,
        retention_policy: str,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> CaseRecord:
        (
            title,
            description,
            purpose_detail,
            source_context,
            created_by_actor_id,
            retention_policy,
            workspace_id,
        ) = validate_case_values(
            title=title,
            description=description,
            purpose=purpose,
            purpose_detail=purpose_detail,
            source_context=source_context,
            authorization_attested=authorization_attested,
            created_by_actor_id=created_by_actor_id,
            retention_policy=retention_policy,
            workspace_id=workspace_id,
        )
        timestamp = self._now()
        return await self._repository.create_case(
            CaseRecord(
                id=record_id or uuid4(),
                workspace_id=workspace_id,
                title=title,
                description=description,
                purpose=purpose,
                purpose_detail=purpose_detail,
                source_context=source_context,
                authorization_attested=True,
                status=CaseStatus.OPEN,
                sensitivity=sensitivity,
                created_at=timestamp,
                updated_at=timestamp,
                closed_at=None,
                created_by_actor_id=created_by_actor_id,
                retention_policy=retention_policy,
                version=1,
            )
        )

    async def get_case(
        self, case_id: UUID, *, workspace_id: str = DEFAULT_WORKSPACE_ID
    ) -> CaseRecord:
        return await self._repository.get_case(case_id, workspace_id=workspace_id)

    async def list_cases(
        self,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[CaseRecord], int]:
        return await self._repository.list_cases(
            workspace_id=workspace_id, limit=limit, offset=offset
        )

    async def update_case(
        self,
        case_id: UUID,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        actor_id: str,
        update: CaseUpdate,
    ) -> CaseRecord:
        mutable_values = (
            update.title,
            update.description,
            update.purpose,
            update.purpose_detail,
            update.source_context,
            update.status,
            update.sensitivity,
            update.retention_policy,
        )
        if all(isinstance(value, UnsetType) for value in mutable_values):
            raise CaseDomainError("case update has no changes", code="case_update_empty")
        current = await self.get_case(case_id, workspace_id=workspace_id)
        title = current.title if isinstance(update.title, UnsetType) else update.title
        description = (
            current.description if isinstance(update.description, UnsetType) else update.description
        )
        purpose = current.purpose if isinstance(update.purpose, UnsetType) else update.purpose
        purpose_detail = (
            current.purpose_detail
            if isinstance(update.purpose_detail, UnsetType)
            else update.purpose_detail
        )
        source_context = (
            current.source_context
            if isinstance(update.source_context, UnsetType)
            else update.source_context
        )
        status = current.status if isinstance(update.status, UnsetType) else update.status
        sensitivity = (
            current.sensitivity if isinstance(update.sensitivity, UnsetType) else update.sensitivity
        )
        retention_policy = (
            current.retention_policy
            if isinstance(update.retention_policy, UnsetType)
            else update.retention_policy
        )
        (
            title,
            description,
            purpose_detail,
            source_context,
            _,
            retention_policy,
            workspace_id,
        ) = validate_case_values(
            title=title,
            description=description,
            purpose=purpose,
            purpose_detail=purpose_detail,
            source_context=source_context,
            authorization_attested=current.authorization_attested,
            created_by_actor_id=current.created_by_actor_id,
            retention_policy=retention_policy,
            workspace_id=workspace_id,
        )
        actor_id = require_text(actor_id, "actor_id", max_length=160)
        expected_version = (
            current.version if update.expected_version is None else update.expected_version
        )
        if expected_version < 1:
            raise CaseDomainError("expected_version must be positive", code="invalid_case_version")
        timestamp = self._now()
        closed_at = (
            current.closed_at or timestamp
            if status in {CaseStatus.RESOLVED, CaseStatus.ARCHIVED}
            else None
        )
        replacement = replace(
            current,
            workspace_id=workspace_id,
            title=title,
            description=description,
            purpose=purpose,
            purpose_detail=purpose_detail,
            source_context=source_context,
            status=status,
            sensitivity=sensitivity,
            retention_policy=retention_policy,
            updated_at=timestamp,
            closed_at=closed_at,
        )
        return await self._repository.update_case(
            replacement, expected_version=expected_version, actor_id=actor_id
        )

    async def create_media(
        self,
        case_id: UUID,
        *,
        record_id: UUID | None = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        actor_id: str,
        media_type: MediaType,
        source_type: MediaSourceType,
        original_filename_display: str,
        mime_type: str,
        byte_size: int,
        sha256: str,
        captured_at: datetime | None,
        received_at: datetime,
        source_url: str | None,
        archive_url: str | None,
        source_description: str | None,
        authorization_attested: bool,
        storage_state: MediaStorageState,
        analysis_id: UUID | None = None,
    ) -> CaseMediaRecord:
        await self.get_case(case_id, workspace_id=workspace_id)
        if not authorization_attested:
            raise CaseDomainError(
                "media authorization attestation is required",
                code="authorization_attestation_required",
            )
        if media_type is not MediaType.IMAGE:
            raise CaseDomainError("Phase 2 supports images only", code="unsupported_media_type")
        normalized_mime = mime_type.strip().lower()
        if normalized_mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise CaseDomainError("unsupported image MIME type", code="unsupported_media_type")
        if byte_size <= 0:
            raise CaseDomainError("byte_size must be positive", code="invalid_byte_size")
        normalized_sha256 = validate_sha256(sha256)
        source_url = validate_https_url(source_url, "source_url")
        archive_url = validate_https_url(archive_url, "archive_url")
        if source_type is MediaSourceType.SOURCE_URL and source_url is None:
            raise CaseDomainError("source_url is required", code="source_url_required")
        if source_type is MediaSourceType.EXTERNAL_ARCHIVE and archive_url is None:
            raise CaseDomainError("archive_url is required", code="archive_url_required")
        if (
            storage_state is MediaStorageState.EXTERNALLY_MANAGED
            and source_url is None
            and archive_url is None
        ):
            raise CaseDomainError(
                "externally managed media requires a URL", code="external_location_required"
            )
        actor_id = require_text(actor_id, "actor_id", max_length=160)
        source_description = optional_text(
            source_description, "source_description", max_length=2_000
        )
        created_at = self._now()
        normalized_received = as_utc(received_at)
        if normalized_received is None:
            raise CaseDomainError("received_at is required", code="received_at_required")
        linked_status: str | None = None
        linked_created: datetime | None = None
        linked_expires: datetime | None = None
        if analysis_id is not None:
            stored = await self._analysis_repository.get(analysis_id)
            if stored is None:
                raise AnalysisNotFoundForCaseError("analysis not found")
            if (
                stored.analysis.image is not None
                and stored.analysis.image.sha256 != normalized_sha256
            ):
                raise CaseRelationshipError(
                    "analysis image hash does not match case media",
                    code="analysis_media_hash_mismatch",
                )
            linked_status = stored.analysis.status.value
            linked_created = stored.analysis.created_at
            linked_expires = stored.analysis.expires_at
        record = CaseMediaRecord(
            id=record_id or uuid4(),
            case_id=case_id,
            analysis_id=analysis_id,
            media_type=media_type,
            source_type=source_type,
            original_filename_display=sanitize_display_filename(original_filename_display),
            mime_type=normalized_mime,
            byte_size=byte_size,
            sha256=normalized_sha256,
            captured_at=as_utc(captured_at),
            received_at=normalized_received,
            source_url=source_url,
            archive_url=archive_url,
            source_description=source_description,
            authorization_attested=True,
            storage_state=storage_state,
            analysis_status_at_link=linked_status,
            analysis_created_at=as_utc(linked_created),
            analysis_expires_at=as_utc(linked_expires),
            materialized_at=None,
            created_at=created_at,
        )
        return await self._repository.create_media(
            record, workspace_id=workspace_id, actor_id=actor_id
        )

    async def list_media(
        self,
        case_id: UUID,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[CaseMediaRecord], int]:
        return await self._repository.list_media(
            case_id, workspace_id=workspace_id, limit=limit, offset=offset
        )

    async def link_analysis(
        self,
        case_id: UUID,
        analysis_id: UUID,
        *,
        media_id: UUID,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        actor_id: str,
    ) -> CaseMediaRecord:
        actor_id = require_text(actor_id, "actor_id", max_length=160)
        stored = await self._analysis_repository.get(analysis_id)
        if stored is None:
            raise AnalysisNotFoundForCaseError("analysis not found")
        media = await self._repository.get_media(case_id, media_id, workspace_id=workspace_id)
        if stored.analysis.image is not None and stored.analysis.image.sha256 != media.sha256:
            raise CaseRelationshipError(
                "analysis image hash does not match case media",
                code="analysis_media_hash_mismatch",
            )
        return await self._repository.link_analysis(
            case_id,
            media_id,
            analysis_id,
            workspace_id=workspace_id,
            actor_id=actor_id,
            analysis_status=stored.analysis.status.value,
            analysis_created_at=stored.analysis.created_at,
            analysis_expires_at=stored.analysis.expires_at,
            linked_at=self._now(),
        )

    async def materialize_analysis(
        self,
        case_id: UUID,
        analysis_id: UUID,
        *,
        media_id: UUID | None = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        actor_id: str,
    ) -> MaterializationResult:
        actor_id = require_text(actor_id, "actor_id", max_length=160)
        media = (
            await self._repository.media_for_analysis(
                case_id, analysis_id, workspace_id=workspace_id
            )
            if media_id is None
            else await self._repository.get_media(case_id, media_id, workspace_id=workspace_id)
        )
        if media.analysis_id != analysis_id:
            raise CaseDomainError(
                "analysis is not linked to media", code="case_relationship_invalid"
            )
        timestamp = self._now()
        if media.materialized_at is not None:
            return await self._repository.materialize_analysis(
                case_id,
                media.id,
                analysis_id,
                workspace_id=workspace_id,
                actor_id=actor_id,
                evidence=(),
                hypotheses=(),
                analysis_storage_present=False,
                materialized_at=timestamp,
            )
        stored = await self._analysis_repository.get(analysis_id)
        if stored is None:
            raise AnalysisNotFoundForCaseError("analysis not found or retention-deleted")
        if stored.analysis.status is not AnalysisStatus.COMPLETED:
            raise AnalysisNotCompletedError(
                "only completed analyses can be materialized",
                code=f"analysis_{stored.analysis.status.value}",
            )
        if stored.analysis.image is not None and stored.analysis.image.sha256 != media.sha256:
            raise CaseRelationshipError(
                "analysis image hash does not match case media",
                code="analysis_media_hash_mismatch",
            )
        evidence, identifiers = _normalize_evidence(
            stored.analysis,
            case_id=case_id,
            media_id=media.id,
            created_at=timestamp,
        )
        hypotheses = _normalize_hypotheses(
            stored.analysis,
            case_id=case_id,
            media_id=media.id,
            evidence_identifiers=identifiers,
            created_at=timestamp,
        )
        return await self._repository.materialize_analysis(
            case_id,
            media.id,
            analysis_id,
            workspace_id=workspace_id,
            actor_id=actor_id,
            evidence=evidence,
            hypotheses=hypotheses,
            analysis_storage_present=stored.storage_key is not None,
            materialized_at=timestamp,
        )

    async def list_evidence(
        self,
        case_id: UUID,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        media_id: UUID | None = None,
        analysis_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[EvidenceRecord], int]:
        return await self._repository.list_evidence(
            case_id,
            workspace_id=workspace_id,
            media_id=media_id,
            analysis_id=analysis_id,
            limit=limit,
            offset=offset,
        )

    async def list_hypotheses(
        self,
        case_id: UUID,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        media_id: UUID | None = None,
        analysis_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[LocationHypothesisRecord], int]:
        return await self._repository.list_hypotheses(
            case_id,
            workspace_id=workspace_id,
            media_id=media_id,
            analysis_id=analysis_id,
            limit=limit,
            offset=offset,
        )

    async def add_adjudication(
        self,
        case_id: UUID,
        hypothesis_id: UUID,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        actor_id: str,
        decision: AdjudicationDecision,
        rationale: str,
        supersedes_adjudication_id: UUID | None = None,
    ) -> AdjudicationRecord:
        await self._repository.get_hypothesis(case_id, hypothesis_id, workspace_id=workspace_id)
        if decision is AdjudicationDecision.WITHDRAWN and supersedes_adjudication_id is None:
            raise CaseDomainError(
                "withdrawn decision must supersede an adjudication",
                code="superseded_adjudication_required",
            )
        record = AdjudicationRecord(
            id=uuid4(),
            case_id=case_id,
            hypothesis_id=hypothesis_id,
            actor_id=require_text(actor_id, "actor_id", max_length=160),
            decision=decision,
            rationale=require_text(rationale, "rationale", max_length=2_000),
            created_at=self._now(),
            supersedes_adjudication_id=supersedes_adjudication_id,
        )
        return await self._repository.add_adjudication(record, workspace_id=workspace_id)

    async def list_adjudications(
        self,
        case_id: UUID,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        hypothesis_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AdjudicationRecord], int]:
        return await self._repository.list_adjudications(
            case_id,
            workspace_id=workspace_id,
            hypothesis_id=hypothesis_id,
            limit=limit,
            offset=offset,
        )

    async def add_operator_correction(
        self,
        case_id: UUID,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        actor_id: str,
        media_id: UUID,
        analysis_id: UUID,
        supersedes_hypothesis_id: UUID,
        latitude: float,
        longitude: float,
        uncertainty_radius_m: float,
        country_code: str | None,
        region_name: str | None,
        locality_name: str | None,
        confidence_label: str | None = None,
        calibration_state: CalibrationState = CalibrationState.NOT_APPLICABLE,
        supporting_evidence_ids: Sequence[UUID] = (),
        model_family_groups: Sequence[str] = (),
        decision: AdjudicationDecision,
        rationale: str,
    ) -> OperatorCorrectionResult:
        actor_id = require_text(actor_id, "actor_id", max_length=160)
        validate_wgs84(latitude, longitude, uncertainty_radius_m)
        superseded = await self._repository.get_hypothesis(
            case_id, supersedes_hypothesis_id, workspace_id=workspace_id
        )
        if superseded.media_id != media_id or superseded.analysis_id != analysis_id:
            raise CaseDomainError(
                "operator correction relationship is invalid",
                code="case_relationship_invalid",
            )
        if calibration_state is not CalibrationState.NOT_APPLICABLE:
            raise CaseDomainError(
                "operator corrections cannot claim model calibration",
                code="invalid_operator_calibration",
            )
        confidence_label = optional_text(confidence_label, "confidence_label", max_length=120)
        if confidence_label is not None and any(
            character.isdigit() or character == "%" for character in confidence_label
        ):
            raise CaseDomainError(
                "operator confidence label cannot be a probability",
                code="invalid_confidence_label",
            )
        if len(set(supporting_evidence_ids)) != len(supporting_evidence_ids):
            raise CaseDomainError(
                "supporting_evidence_ids must be unique", code="duplicate_evidence_id"
            )
        if decision is AdjudicationDecision.WITHDRAWN:
            raise CaseDomainError(
                "new operator correction cannot be withdrawn", code="invalid_decision"
            )
        groups = tuple(
            sorted(
                {
                    require_text(value, "model_family_group", max_length=160)
                    for value in model_family_groups
                }
            )
        )
        timestamp = self._now()
        hypothesis_id = uuid4()
        hypothesis = LocationHypothesisRecord(
            id=hypothesis_id,
            case_id=case_id,
            media_id=media_id,
            analysis_id=analysis_id,
            origin=HypothesisOrigin.OPERATOR_CORRECTION,
            latitude=latitude,
            longitude=longitude,
            uncertainty_radius_m=uncertainty_radius_m,
            country_code=validate_country_code(country_code),
            region_name=optional_text(region_name, "region_name", max_length=160),
            locality_name=optional_text(locality_name, "locality_name", max_length=160),
            rank=None,
            confidence_label=confidence_label,
            calibration_state=calibration_state,
            supporting_evidence_ids=tuple(supporting_evidence_ids),
            model_family_groups=groups,
            created_at=timestamp,
            supersedes_hypothesis_id=supersedes_hypothesis_id,
        )
        adjudication = AdjudicationRecord(
            id=uuid4(),
            case_id=case_id,
            hypothesis_id=hypothesis_id,
            actor_id=actor_id,
            decision=decision,
            rationale=require_text(rationale, "rationale", max_length=2_000),
            created_at=timestamp,
            supersedes_adjudication_id=None,
        )
        return await self._repository.add_operator_correction(
            hypothesis,
            adjudication,
            workspace_id=workspace_id,
            source_record_key=f"operator:{uuid4()}",
        )

    async def list_audit_events(
        self,
        case_id: UUID,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AuditEventRecord], int]:
        return await self._repository.list_audit_events(
            case_id, workspace_id=workspace_id, limit=limit, offset=offset
        )

    async def verify_audit_integrity(
        self, case_id: UUID, *, workspace_id: str = DEFAULT_WORKSPACE_ID
    ) -> AuditIntegrityResult:
        return await self._repository.verify_audit_integrity(case_id, workspace_id=workspace_id)
