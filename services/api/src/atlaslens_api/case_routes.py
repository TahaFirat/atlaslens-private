from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Query, Request, status

from atlaslens_api.case_schemas import (
    AdjudicationCreateRequest,
    AdjudicationView,
    AnalysisLinkRequest,
    AuditEventPage,
    AuditEventView,
    AuditIntegrityView,
    CaseCreateRequest,
    CaseMediaCreateRequest,
    CaseMediaPage,
    CaseMediaView,
    CasePage,
    CasePatchRequest,
    CaseView,
    EvidencePage,
    EvidenceView,
    HypothesisPage,
    HypothesisView,
    MaterializationView,
    MaterializeEvidenceRequest,
    OperatorCorrectionView,
    OperatorHypothesisCreateRequest,
)
from atlaslens_api.cases.domain import (
    DEFAULT_WORKSPACE_ID,
    UNSET,
    AnalysisNotCompletedError,
    AnalysisNotFoundForCaseError,
    AuditIntegrityError,
    CaseConflictError,
    CaseDomainError,
    CaseNotFoundError,
    CaseRelationshipError,
    CaseUpdate,
    LocationHypothesisRecord,
)
from atlaslens_api.cases.domain import (
    AdjudicationDecision as DomainAdjudicationDecision,
)
from atlaslens_api.cases.domain import (
    CalibrationState as DomainCalibrationState,
)
from atlaslens_api.cases.domain import (
    CasePurpose as DomainCasePurpose,
)
from atlaslens_api.cases.domain import (
    CaseSensitivity as DomainCaseSensitivity,
)
from atlaslens_api.cases.domain import (
    CaseStatus as DomainCaseStatus,
)
from atlaslens_api.cases.domain import (
    MediaSourceType as DomainMediaSourceType,
)
from atlaslens_api.cases.domain import (
    MediaStorageState as DomainMediaStorageState,
)
from atlaslens_api.cases.domain import (
    MediaType as DomainMediaType,
)
from atlaslens_api.errors import AppError

if TYPE_CHECKING:
    from atlaslens_api.cases import CaseInvestigationService


router = APIRouter(prefix="/api/v1/cases", tags=["cases"])

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"description": "The case-scoped resource was not found."},
    409: {"description": "The requested operation conflicts with current case state."},
    422: {"description": "The request failed bounded validation."},
    503: {"description": "The local case service is unavailable."},
}


def _service(request: Request) -> CaseInvestigationService:
    services = getattr(request.app.state, "services", None)
    service = getattr(services, "case_service", None)
    if service is None:
        raise AppError(
            503,
            "case_service_unavailable",
            "error.case_service_unavailable",
            "Case service unavailable",
        )
    return cast("CaseInvestigationService", service)


def _as_app_error(error: CaseDomainError) -> AppError:
    if isinstance(error, CaseNotFoundError):
        return AppError(404, error.code, "error.case_not_found", "Not found")
    if isinstance(error, AnalysisNotFoundForCaseError):
        return AppError(404, error.code, "error.analysis_not_found", "Not found")
    if isinstance(error, CaseRelationshipError) or error.code == "case_relationship_invalid":
        return AppError(404, error.code, "error.case_resource_not_found", "Not found")
    if isinstance(error, AnalysisNotCompletedError):
        return AppError(409, error.code, "error.analysis_not_completed", "Analysis not completed")
    if isinstance(error, CaseConflictError | AuditIntegrityError):
        return AppError(409, error.code, f"error.{error.code}", "Case conflict")
    return AppError(422, error.code, f"error.{error.code}", "Invalid case request")


async def _hypothesis_view(
    service: CaseInvestigationService,
    case_id: UUID,
    hypothesis: LocationHypothesisRecord,
) -> HypothesisView:
    adjudications, total = await service.list_adjudications(
        case_id,
        workspace_id=DEFAULT_WORKSPACE_ID,
        hypothesis_id=hypothesis.id,
        limit=100,
        offset=0,
    )
    if total > 100:
        adjudications, _ = await service.list_adjudications(
            case_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            hypothesis_id=hypothesis.id,
            limit=100,
            offset=total - 100,
        )
    views = [AdjudicationView.model_validate(item) for item in adjudications]
    view = HypothesisView.model_validate(hypothesis)
    return view.model_copy(
        update={
            "adjudications": views,
            "adjudication_count": total,
            "latest_adjudication": views[-1] if views else None,
        }
    )


@router.post(
    "",
    operation_id="createCase",
    response_model=CaseView,
    status_code=status.HTTP_201_CREATED,
    responses=_ERROR_RESPONSES,
)
async def create_case(request: Request, payload: CaseCreateRequest) -> CaseView:
    try:
        record = await _service(request).create_case(
            title=payload.title,
            description=payload.description,
            purpose=DomainCasePurpose(payload.purpose.value),
            purpose_detail=payload.purpose_detail,
            sensitivity=DomainCaseSensitivity(payload.sensitivity.value),
            source_context=payload.source_context,
            authorization_attested=payload.authorization_attested,
            created_by_actor_id=payload.created_by_actor_id,
            retention_policy=payload.retention_policy,
            workspace_id=DEFAULT_WORKSPACE_ID,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return CaseView.model_validate(record)


@router.get("", operation_id="listCases", response_model=CasePage, responses=_ERROR_RESPONSES)
async def list_cases(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CasePage:
    try:
        records, total = await _service(request).list_cases(
            workspace_id=DEFAULT_WORKSPACE_ID,
            limit=limit,
            offset=offset,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return CasePage(
        items=[CaseView.model_validate(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{case_id}", operation_id="getCase", response_model=CaseView, responses=_ERROR_RESPONSES
)
async def get_case(request: Request, case_id: UUID) -> CaseView:
    try:
        record = await _service(request).get_case(case_id, workspace_id=DEFAULT_WORKSPACE_ID)
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return CaseView.model_validate(record)


@router.patch(
    "/{case_id}", operation_id="patchCase", response_model=CaseView, responses=_ERROR_RESPONSES
)
async def patch_case(
    request: Request,
    case_id: UUID,
    payload: CasePatchRequest,
) -> CaseView:
    fields = payload.model_fields_set
    update = CaseUpdate(
        title=payload.title if "title" in fields and payload.title is not None else UNSET,
        description=payload.description if "description" in fields else UNSET,
        purpose=(
            DomainCasePurpose(payload.purpose.value)
            if "purpose" in fields and payload.purpose is not None
            else UNSET
        ),
        purpose_detail=payload.purpose_detail if "purpose_detail" in fields else UNSET,
        source_context=(
            payload.source_context
            if "source_context" in fields and payload.source_context is not None
            else UNSET
        ),
        status=(
            DomainCaseStatus(payload.status.value)
            if "status" in fields and payload.status is not None
            else UNSET
        ),
        sensitivity=(
            DomainCaseSensitivity(payload.sensitivity.value)
            if "sensitivity" in fields and payload.sensitivity is not None
            else UNSET
        ),
        retention_policy=(
            payload.retention_policy
            if "retention_policy" in fields and payload.retention_policy is not None
            else UNSET
        ),
        expected_version=payload.expected_version,
    )
    try:
        record = await _service(request).update_case(
            case_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            actor_id=payload.actor_id,
            update=update,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return CaseView.model_validate(record)


@router.post(
    "/{case_id}/media",
    operation_id="createCaseMedia",
    response_model=CaseMediaView,
    status_code=status.HTTP_201_CREATED,
    responses=_ERROR_RESPONSES,
)
async def create_case_media(
    request: Request,
    case_id: UUID,
    payload: CaseMediaCreateRequest,
) -> CaseMediaView:
    try:
        record = await _service(request).create_media(
            case_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            actor_id=payload.actor_id,
            media_type=DomainMediaType(payload.media_type.value),
            source_type=DomainMediaSourceType(payload.source_type.value),
            original_filename_display=payload.original_filename_display,
            mime_type=payload.mime_type,
            byte_size=payload.byte_size,
            sha256=payload.sha256,
            captured_at=payload.captured_at,
            received_at=datetime.now(UTC),
            source_url=payload.source_url,
            archive_url=payload.archive_url,
            source_description=payload.source_description,
            authorization_attested=payload.authorization_attested,
            storage_state=DomainMediaStorageState(payload.storage_state.value),
            analysis_id=None,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return CaseMediaView.model_validate(record)


@router.get(
    "/{case_id}/media",
    operation_id="listCaseMedia",
    response_model=CaseMediaPage,
    responses=_ERROR_RESPONSES,
)
async def list_case_media(
    request: Request,
    case_id: UUID,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CaseMediaPage:
    try:
        records, total = await _service(request).list_media(
            case_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            limit=limit,
            offset=offset,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return CaseMediaPage(
        items=[CaseMediaView.model_validate(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/{case_id}/analyses/{analysis_id}/link",
    operation_id="linkCaseAnalysis",
    response_model=CaseMediaView,
    responses=_ERROR_RESPONSES,
)
async def link_analysis(
    request: Request,
    case_id: UUID,
    analysis_id: UUID,
    payload: AnalysisLinkRequest,
) -> CaseMediaView:
    try:
        record = await _service(request).link_analysis(
            case_id,
            analysis_id,
            media_id=payload.media_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            actor_id=payload.actor_id,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return CaseMediaView.model_validate(record)


@router.post(
    "/{case_id}/analyses/{analysis_id}/materialize-evidence",
    operation_id="materializeCaseEvidence",
    response_model=MaterializationView,
    responses=_ERROR_RESPONSES,
)
async def materialize_evidence(
    request: Request,
    case_id: UUID,
    analysis_id: UUID,
    payload: MaterializeEvidenceRequest,
) -> MaterializationView:
    try:
        result = await _service(request).materialize_analysis(
            case_id,
            analysis_id,
            media_id=payload.media_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            actor_id=payload.actor_id,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    evidence_ids = [record.id for record in result.evidence]
    hypothesis_ids = [record.id for record in result.hypotheses]
    return MaterializationView(
        case_id=case_id,
        media_id=payload.media_id,
        analysis_id=analysis_id,
        created=result.created,
        evidence_created=len(evidence_ids) if result.created else 0,
        hypotheses_created=len(hypothesis_ids) if result.created else 0,
        already_materialized=not result.created,
        evidence_ids=evidence_ids,
        hypothesis_ids=hypothesis_ids,
    )


@router.get(
    "/{case_id}/evidence",
    operation_id="listCaseEvidence",
    response_model=EvidencePage,
    responses=_ERROR_RESPONSES,
)
async def list_evidence(
    request: Request,
    case_id: UUID,
    media_id: Annotated[UUID | None, Query()] = None,
    analysis_id: Annotated[UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> EvidencePage:
    try:
        records, total = await _service(request).list_evidence(
            case_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            media_id=media_id,
            analysis_id=analysis_id,
            limit=limit,
            offset=offset,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return EvidencePage(
        items=[EvidenceView.model_validate(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{case_id}/hypotheses",
    operation_id="listCaseHypotheses",
    response_model=HypothesisPage,
    responses=_ERROR_RESPONSES,
)
async def list_hypotheses(
    request: Request,
    case_id: UUID,
    media_id: Annotated[UUID | None, Query()] = None,
    analysis_id: Annotated[UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> HypothesisPage:
    service = _service(request)
    try:
        records, total = await service.list_hypotheses(
            case_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            media_id=media_id,
            analysis_id=analysis_id,
            limit=limit,
            offset=offset,
        )
        views = [await _hypothesis_view(service, case_id, record) for record in records]
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return HypothesisPage(items=views, total=total, limit=limit, offset=offset)


@router.post(
    "/{case_id}/hypotheses/{hypothesis_id}/adjudications",
    operation_id="createHypothesisAdjudication",
    response_model=AdjudicationView,
    status_code=status.HTTP_201_CREATED,
    responses=_ERROR_RESPONSES,
)
async def add_adjudication(
    request: Request,
    case_id: UUID,
    hypothesis_id: UUID,
    payload: AdjudicationCreateRequest,
) -> AdjudicationView:
    try:
        record = await _service(request).add_adjudication(
            case_id,
            hypothesis_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            actor_id=payload.actor_id,
            decision=DomainAdjudicationDecision(payload.decision.value),
            rationale=payload.rationale,
            supersedes_adjudication_id=payload.supersedes_adjudication_id,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return AdjudicationView.model_validate(record)


@router.post(
    "/{case_id}/operator-hypotheses",
    operation_id="createOperatorHypothesis",
    response_model=OperatorCorrectionView,
    status_code=status.HTTP_201_CREATED,
    responses=_ERROR_RESPONSES,
)
async def add_operator_hypothesis(
    request: Request,
    case_id: UUID,
    payload: OperatorHypothesisCreateRequest,
) -> OperatorCorrectionView:
    try:
        result = await _service(request).add_operator_correction(
            case_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            actor_id=payload.actor_id,
            media_id=payload.media_id,
            analysis_id=payload.analysis_id,
            supersedes_hypothesis_id=payload.supersedes_hypothesis_id,
            latitude=payload.latitude,
            longitude=payload.longitude,
            uncertainty_radius_m=payload.uncertainty_radius_m,
            country_code=payload.country_code,
            region_name=payload.region_name,
            locality_name=payload.locality_name,
            confidence_label=None,
            calibration_state=DomainCalibrationState.NOT_APPLICABLE,
            supporting_evidence_ids=tuple(payload.supporting_evidence_ids),
            model_family_groups=(),
            decision=DomainAdjudicationDecision(payload.decision.value),
            rationale=payload.rationale,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    adjudication = AdjudicationView.model_validate(result.adjudication)
    hypothesis = HypothesisView.model_validate(result.hypothesis).model_copy(
        update={
            "adjudications": [adjudication],
            "adjudication_count": 1,
            "latest_adjudication": adjudication,
        }
    )
    return OperatorCorrectionView(hypothesis=hypothesis, adjudication=adjudication)


@router.get(
    "/{case_id}/audit-events",
    operation_id="listCaseAuditEvents",
    response_model=AuditEventPage,
    responses=_ERROR_RESPONSES,
)
async def list_audit_events(
    request: Request,
    case_id: UUID,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AuditEventPage:
    try:
        records, total = await _service(request).list_audit_events(
            case_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            limit=limit,
            offset=offset,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return AuditEventPage(
        items=[AuditEventView.model_validate(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{case_id}/audit-integrity",
    operation_id="getCaseAuditIntegrity",
    response_model=AuditIntegrityView,
    responses=_ERROR_RESPONSES,
)
async def get_audit_integrity(request: Request, case_id: UUID) -> AuditIntegrityView:
    try:
        result = await _service(request).verify_audit_integrity(
            case_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
        )
    except CaseDomainError as error:
        raise _as_app_error(error) from error
    return AuditIntegrityView(
        case_id=case_id,
        valid=result.valid,
        checked_event_count=result.event_count,
        first_invalid_sequence=result.failure_sequence,
        reason=result.failure_code,
    )
