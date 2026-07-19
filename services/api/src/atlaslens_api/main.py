from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.base import RequestResponseEndpoint

from atlaslens_api.case_routes import router as case_router
from atlaslens_api.cases.repository import SQLAlchemyCaseRepository
from atlaslens_api.cases.service import CaseInvestigationService
from atlaslens_api.cleanup import DefaultRetentionCleanupService
from atlaslens_api.config import Settings
from atlaslens_api.constraints.https_transport import PinnedHttpsMapTransport
from atlaslens_api.constraints.overpass import OverpassMapConstraintProvider
from atlaslens_api.constraints.phase5b import BoundedMapEvidenceEvaluator
from atlaslens_api.database import create_database_engine
from atlaslens_api.dataset_qa import DatasetQAError, DatasetQAReportCatalog
from atlaslens_api.errors import AppError, bad_request
from atlaslens_api.evaluation.catalog import EvaluationCatalogError, EvaluationReportCatalog
from atlaslens_api.events import TERMINAL_EVENT_TYPES, AnalysisEventBroker
from atlaslens_api.fusion import FUSION_POLICY_VERSION, DeterministicCandidateFusionService
from atlaslens_api.gazetteer import (
    ForwardGazetteerManager,
    GazetteerManager,
    GazetteerMetadata,
    SQLiteForwardGazetteerResolver,
    SQLiteGazetteerResolver,
)
from atlaslens_api.global_prediction.clustering import GeoClipCandidateClusterer
from atlaslens_api.global_prediction.reverse_geocoding import (
    CachedClusterReverseGeocoder,
    SQLiteReverseGeocodeCache,
)
from atlaslens_api.history import history_item
from atlaslens_api.image_processing import PreparedImage, SafeImageProcessor
from atlaslens_api.inference import (
    DevelopmentMockProvider,
    GeoCLIPInferenceAdapter,
    GeolocationInferenceProvider,
    ProviderEnsembleCoordinator,
)
from atlaslens_api.job_queue import InProcessJobQueue, JobQueueFullError
from atlaslens_api.mapillary_demo.api import (
    MapillaryDemoRuntime,
    PublishedMapillaryDemoSearchIndex,
    build_mapillary_demo_router,
)
from atlaslens_api.megaloc_adapter import inspect_megaloc_approval
from atlaslens_api.model_management.manifest import default_manifest_path, load_manifest
from atlaslens_api.model_management.service import ModelManagementService
from atlaslens_api.phase5b import (
    Phase5BEvidenceReranker,
    Phase5BRerankConfig,
    default_phase5b_config_path,
)
from atlaslens_api.phase6a import (
    Phase6AHybridConfig,
    Phase6AHybridEvidenceEngine,
    default_phase6a_config_path,
)
from atlaslens_api.phase6b.ensemble import Phase6BLocalModelEnsemble
from atlaslens_api.phase6b.fusion import (
    Phase6BFusionConfig,
    Phase6BGeographicFusionEngine,
    load_phase6b_fusion_config,
)
from atlaslens_api.phase6b.integration import Phase6BPipelineExtension
from atlaslens_api.phase6b.ocr import PaddleOCRProvider, PreferredOCRProvider
from atlaslens_api.phase6b.openai_assist import (
    OpenAIGeoReviewConfig,
    OpenAIGeoReviewProvider,
    build_openai_geo_review_provider,
    default_openai_geo_config_path,
)
from atlaslens_api.phase6b.osv5m import OSV5MProvider
from atlaslens_api.phase6b.plonk import PlonkModelRouter, PlonkProvider
from atlaslens_api.phase6b.runtime_verification import (
    validate_rapidocr_runtime_verification,
)
from atlaslens_api.phase6b.scheduler import HeavyModelScheduler
from atlaslens_api.phase6b.worker_client import (
    LocalWorkerHTTPClient,
    OSV5MHTTPWorkerClient,
    PaddleOCRHTTPWorkerClient,
    PlonkHTTPWorkerClient,
    WorkerClientError,
    WorkerHealth,
)
from atlaslens_api.phase6c.catalogue import load_coordinate_catalogue
from atlaslens_api.phase6c.fusion import (
    Phase6CFusionConfig,
    Phase6CGeographicFusionEngine,
    load_phase6c_fusion_config,
)
from atlaslens_api.phase6c.hierarchical import (
    GeoCLIPHierarchicalConfig,
    GeoCLIPHierarchicalSearchProvider,
)
from atlaslens_api.phase6c.integration import (
    Phase6CCacheVersions,
    Phase6CPipelineExtension,
)
from atlaslens_api.phase6c.megaloc import (
    MEGALOC_DESCRIPTOR_VERSION,
    MEGALOC_MODEL_REVISION,
    MEGALOC_SOURCE_REVISION,
    MegaLocRetrievalConfig,
    MegaLocRetrievalProvider,
    create_megaloc_http_worker_client,
)
from atlaslens_api.phase6c.ocr import Phase6COCRConfig, load_phase6c_ocr_config
from atlaslens_api.phase6c.reference_index import (
    ReferenceIndexDiagnostics,
    open_reference_index,
)
from atlaslens_api.pipeline import AnalysisJob, AnalysisPipeline
from atlaslens_api.place_evidence.service import PlaceEvidenceService
from atlaslens_api.providers.base import (
    ExifProvider,
    GlobalGeolocationProvider,
    ImageQualityProvider,
    OCRProvider,
    RetrievalProvider,
    VisionClueProvider,
)
from atlaslens_api.providers.exif import PillowExifProvider
from atlaslens_api.providers.geoclip import GeoCLIPGlobalGeolocationProvider, select_device
from atlaslens_api.providers.nvidia_vision import NvidiaVisionClueProvider
from atlaslens_api.providers.ocr import TesseractOCRProvider
from atlaslens_api.providers.openai_vision import OpenAIVisionClueProvider
from atlaslens_api.providers.quality import OpenCVImageQualityProvider
from atlaslens_api.providers.rapidocr import (
    RapidOCRProvider,
    load_verified_rapidocr_runtime,
)
from atlaslens_api.providers.retrieval import FaissRetrievalProvider
from atlaslens_api.rate_limit import ConnectionLimiter, SlidingWindowRateLimiter
from atlaslens_api.repository import (
    AnalysisHistoryFilters,
    AnalysisRepository,
    DuplicateIdempotencyError,
    SQLAlchemyAnalysisRepository,
)
from atlaslens_api.retrieval.errors import RetrievalError
from atlaslens_api.retrieval.providers import production_embedding_provider
from atlaslens_api.retrieval.service import open_retrieval_query_service
from atlaslens_api.safe_logging import configure_logging
from atlaslens_api.schemas import (
    Analysis,
    AnalysisAccepted,
    AnalysisEvent,
    AnalysisHistoryPage,
    AnalysisMode,
    AnalysisStatus,
    CapabilitiesResponse,
    DatasetQAReport,
    DatasetQAReportList,
    DeleteResponse,
    EvaluationReportList,
    EvaluationReportSummary,
    HealthResponse,
    ModelStatusItem,
    ModelStatusResponse,
    Phase6CLeakageAuditSummary,
    ProblemDetails,
    Progress,
    ProviderCapability,
    ProviderStatusItem,
    ProviderStatusResponse,
    ReadinessResponse,
    RetentionPolicy,
    SystemIntelligenceLeakageAudit,
    SystemIntelligenceModelCard,
    SystemIntelligenceReferenceIndexCard,
    SystemIntelligenceResponse,
)
from atlaslens_api.segmentation import SceneSegmentationProvider, SegFormerSceneProvider
from atlaslens_api.storage import LocalTemporaryStorage
from atlaslens_api.trained_artifacts.integration import build_custom_provider
from atlaslens_api.trained_artifacts.manager import TrainedArtifactManager
from atlaslens_api.turkiye_reference import TurkiyeReferenceIndexProvider

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_IDEMPOTENCY_KEY = re.compile(r"^[\x21-\x7e]{8,128}$")
_logger = logging.getLogger("atlaslens_api.request")


def _phase3b2_megaloc_artifact_approved(config: Settings) -> bool:
    raw_path = config.megaloc_phase3b2_config_path.strip()
    project_root = Path(__file__).resolve().parents[4]
    selected_path: Path | None = None
    if raw_path:
        candidate = Path(raw_path)
        selected_path = candidate if candidate.is_absolute() else project_root / candidate
    readiness = inspect_megaloc_approval(selected_path, project_root=project_root)
    return bool(
        readiness.ready
        and readiness.artifact_sha256
        == config.turkiye_reference_descriptor_artifact_sha256
        and readiness.descriptor_version == config.turkiye_reference_descriptor_version
        and readiness.descriptor_dimension == config.turkiye_reference_descriptor_dimension
        and readiness.preprocessing_id
        == config.turkiye_reference_descriptor_preprocessing_version
    )


def _rapidocr_requested(config: Settings) -> bool:
    """Keep the Phase 6B fallback active even when PaddleOCR is disabled."""

    return bool(
        config.ocr_provider == "rapidocr"
        and config.rapidocr_enabled
        and (
            config.ocr_enabled
            or (config.phase6b_enabled and config.paddleocr_fallback_to_rapidocr)
        )
    )


def _ocr_requested(config: Settings) -> bool:
    return bool(
        config.ocr_enabled
        or (
            config.ocr_provider == "rapidocr"
            and config.phase6b_enabled
            and (
                config.paddleocr_enabled
                or (config.rapidocr_enabled and config.paddleocr_fallback_to_rapidocr)
            )
        )
    )


type Phase6BWorkerState = Literal[
    "not_installed",
    "dependencies_installed",
    "weights_prepared",
    "worker_unreachable",
    "model_load_failed",
    "inference_not_verified",
    "ready",
    "disabled",
]

type PublicProviderState = Literal[
    "ready",
    "not_installed",
    "dependencies_installed",
    "weights_prepared",
    "worker_unreachable",
    "model_load_failed",
    "inference_not_verified",
    "incomplete",
    "loading",
    "failed",
    "disabled",
    "unavailable",
]

type SystemParticipation = Literal[
    "primary",
    "candidate",
    "fallback",
    "descriptive",
    "verifier",
    "optional_review",
    "shadow",
    "disabled",
    "not_integrated",
]

type SystemExecutionMode = Literal[
    "in_process",
    "isolated_worker",
    "isolated_process",
    "cloud",
    "not_integrated",
]


def _public_megaloc_state(state: str | None, *, enabled: bool) -> PublicProviderState:
    if not enabled:
        return "disabled"
    states: dict[str, PublicProviderState] = {
        "ready": "ready",
        "not_installed": "not_installed",
        "worker_unreachable": "worker_unreachable",
        "model_load_not_verified": "model_load_failed",
        "model_load_failed": "model_load_failed",
        "inference_not_verified": "inference_not_verified",
        "inference_failed": "failed",
        "reference_index_unavailable": "unavailable",
        "descriptor_version_mismatch": "unavailable",
    }
    return states.get(state or "", "unavailable")


@dataclass(slots=True)
class Phase6BWorkerBinding:
    client: LocalWorkerHTTPClient | None
    state: Phase6BWorkerState
    health: WorkerHealth | None = None
    last_error: str | None = None
    transport_bound: bool = False

    @property
    def ready(self) -> bool:
        return self.state == "ready" and self.health is not None and self.health.ready

    async def refresh(self) -> Phase6BWorkerState:
        if self.client is None:
            return self.state
        try:
            self.health = await self.client.health(timeout_seconds=0.25)
        except (WorkerClientError, OSError, ValueError) as exc:
            self.state = "worker_unreachable"
            self.last_error = getattr(exc, "code", "worker_unreachable")
        else:
            self.state, self.last_error = _worker_state(self.health)
            if self.state == "ready" and not self.transport_bound:
                self.state = "worker_unreachable"
                self.last_error = "isolated_worker_transport_not_bound"
        return self.state


def _worker_state(health: WorkerHealth) -> tuple[Phase6BWorkerState, str | None]:
    if not health.import_ok:
        return "not_installed", health.last_error or "provider_import_missing"
    if not health.weights_available:
        return "dependencies_installed", health.last_error or "pretrained_weights_missing"
    if not health.load_verified:
        if health.last_error == "model_load_failed":
            return "model_load_failed", health.last_error
        return "weights_prepared", health.last_error or "model_load_not_verified"
    if not health.real_inference_verified:
        return "inference_not_verified", health.last_error or "real_inference_not_verified"
    return "ready", None


def _connect_worker(
    *,
    enabled: bool,
    provider: str,
    host: str,
    port: int,
    provider_revision: str,
    model_revisions: tuple[str, ...],
    timeout_seconds: float,
    max_image_bytes: int,
) -> Phase6BWorkerBinding:
    if not enabled:
        return Phase6BWorkerBinding(client=None, state="disabled", last_error="disabled")
    client = LocalWorkerHTTPClient(
        provider=provider,
        host=host,
        port=port,
        provider_revision=provider_revision,
        model_revisions=model_revisions,
        timeout_seconds=timeout_seconds,
        max_image_bytes=max_image_bytes,
    )
    try:
        health = client.health_sync(timeout_seconds=min(1.0, timeout_seconds))
    except (WorkerClientError, OSError, ValueError) as exc:
        return Phase6BWorkerBinding(
            client=client,
            state="worker_unreachable",
            last_error=getattr(exc, "code", "worker_unreachable"),
        )
    state, last_error = _worker_state(health)
    return Phase6BWorkerBinding(
        client=client,
        state=state,
        health=health,
        last_error=last_error,
        transport_bound=state == "ready",
    )


@dataclass(slots=True)
class AppServices:
    settings: Settings
    repository: AnalysisRepository
    case_service: CaseInvestigationService
    storage: LocalTemporaryStorage
    image_processor: SafeImageProcessor
    queue: InProcessJobQueue
    broker: AnalysisEventBroker
    pipeline: AnalysisPipeline
    cleanup: DefaultRetentionCleanupService
    rate_limiter: SlidingWindowRateLimiter
    connection_limiter: ConnectionLimiter
    exif_provider: ExifProvider
    quality_provider: ImageQualityProvider
    ocr_provider: OCRProvider
    vision_provider: VisionClueProvider
    global_provider: GlobalGeolocationProvider | None
    inference_coordinator: ProviderEnsembleCoordinator | None
    trained_artifact_manager: TrainedArtifactManager
    retrieval_provider: RetrievalProvider
    segmentation_provider: SceneSegmentationProvider
    reverse_geocoder: CachedClusterReverseGeocoder
    phase6b_extension: Phase6BPipelineExtension | None
    phase6c_extension: Phase6CPipelineExtension | None
    megaloc_provider: MegaLocRetrievalProvider
    phase6c_reference_index: ReferenceIndexDiagnostics
    turkiye_reference_provider: TurkiyeReferenceIndexProvider
    mapillary_demo_provider: MapillaryDemoRuntime
    osv5m_provider: OSV5MProvider
    plonk_provider: PlonkProvider
    paddleocr_provider: PaddleOCRProvider
    rapidocr_provider: RapidOCRProvider | None
    phase6b_worker_bindings: dict[str, Phase6BWorkerBinding]
    heavy_model_scheduler: HeavyModelScheduler
    rapidocr_runtime_verified: bool
    rapidocr_runtime_reason: str | None
    rapidocr_last_success_at: datetime | None
    openai_geo_provider: OpenAIGeoReviewProvider | None
    openai_geo_config: OpenAIGeoReviewConfig
    engine: Any
    active_artifacts: dict[UUID, tuple[str, str]]


def _client_key(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "request-unknown"))


def _safe_route_path(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    return str(route_path) if route_path else "unmatched_route"


def _problem(error: AppError, request: Request) -> JSONResponse:
    body = ProblemDetails(
        type=f"https://atlaslens.invalid/problems/{error.code}",
        title=error.title,
        status=error.status_code,
        code=error.code,
        message_key=error.message_key,
        request_id=_request_id(request),
        retry_after_seconds=error.retry_after_seconds,
    )
    headers = {"Cache-Control": "no-store"}
    if error.retry_after_seconds is not None:
        headers["Retry-After"] = str(error.retry_after_seconds)
    return JSONResponse(
        status_code=error.status_code,
        content=body.model_dump(mode="json"),
        media_type="application/problem+json",
        headers=headers,
    )


def _accepted(analysis_id: UUID) -> AnalysisAccepted:
    base = f"/api/v1/analyses/{analysis_id}"
    return AnalysisAccepted(
        id=analysis_id,
        status_url=base,
        events_url=f"{base}/events",
        delete_url=base,
    )


def _phase6b_geographic_capability(
    status: Any,
    *,
    enabled: bool,
    runtime: Phase6BWorkerBinding,
) -> ProviderCapability:
    operational: Phase6BWorkerState = runtime.state if enabled else "disabled"
    available = operational == "ready" and status.available
    health = runtime.health
    reason = None if available else runtime.last_error or status.reason_code or operational
    return ProviderCapability(
        provider_id=status.provider,
        enabled=enabled,
        available=available,
        execution_boundary="local",
        reason_code=reason,
        installed=bool(health is not None and health.import_ok),
        verified=available,
        operational_status=operational,
        model_name=status.model_id,
        model_revision=status.model_revision,
        device=status.device,
        calibration_state="uncalibrated",
        provider_type="global_geolocation",
        offline=True,
        license="MIT",
        limitation="Pretrained model output is uncalibrated and may be geographically broad.",
        weights_available=health.weights_available if health is not None else False,
        usable=available,
        execution_mode="isolated_worker",
        source_revision=status.source_revision,
        load_error=reason,
    )


def _worker_intelligence_card(
    *,
    binding: Phase6BWorkerBinding,
    enabled: bool,
    provider_available: bool,
    model_id: str,
    display_name: str,
    runtime_model_id: str,
    repository_url: str,
    purpose: str,
    model_revision: str,
    source_revision: str,
    license_name: str,
    requested_device: str,
    participation: SystemParticipation,
    provider_reason: str | None = None,
) -> SystemIntelligenceModelCard:
    available = bool(enabled and binding.ready and provider_available)
    state: PublicProviderState = binding.state if enabled else "disabled"
    health = binding.health
    worker_reachable = (
        None
        if not enabled
        else False
        if binding.state == "worker_unreachable"
        else health.process_running
        if health is not None
        else None
    )
    error_code = (
        None
        if available
        else "disabled"
        if not enabled
        else binding.last_error or provider_reason or "provider_unavailable"
    )
    return SystemIntelligenceModelCard(
        model_id=model_id,
        display_name=display_name,
        runtime_model_id=runtime_model_id,
        repository_url=repository_url,
        purpose=purpose,
        enabled=enabled,
        available=available,
        status=state,
        installed=health.import_ok if health is not None else None,
        weights_available=health.weights_available if health is not None else None,
        worker_reachable=worker_reachable,
        model_loaded=(
            health.model_loaded if health is not None and worker_reachable else None
        ),
        load_verified=health.load_verified if health is not None else None,
        real_inference_verified=(
            health.real_inference_verified if health is not None else None
        ),
        device=health.device if health is not None else requested_device,
        execution_mode="isolated_worker",
        source_revision=source_revision,
        model_revision=model_revision,
        license=license_name,
        error_code=error_code,
        current_participation=participation if enabled else "disabled",
    )


async def _system_intelligence_snapshot(
    services: AppServices,
) -> SystemIntelligenceResponse:
    config = services.settings
    await asyncio.gather(
        *(binding.refresh() for binding in services.phase6b_worker_bindings.values())
    )
    megaloc_status = await services.megaloc_provider.status()

    global_status = (
        services.global_provider.status() if services.global_provider is not None else None
    )
    geoclip_enabled = bool(config.global_model_enabled and global_status is not None)
    geoclip_available = bool(
        geoclip_enabled and global_status is not None and global_status.status == "ready"
    )
    geoclip_participation: SystemParticipation = "primary"
    if services.inference_coordinator is not None:
        for status in services.inference_coordinator.statuses():
            if status.provider_id == "geoclip-global-v1" and status.mode in {
                "primary",
                "candidate",
                "shadow",
            }:
                geoclip_participation = status.mode
                break
    geoclip_error = (
        None
        if geoclip_available
        else "disabled"
        if not geoclip_enabled
        else global_status.reason_code or global_status.status
        if global_status is not None
        else "provider_unavailable"
    )
    geoclip = SystemIntelligenceModelCard(
        model_id="geoclip_original",
        display_name="GeoCLIP original gallery",
        runtime_model_id=(
            global_status.model_name if global_status is not None else "GeoCLIP"
        ),
        repository_url="https://github.com/VicenteVivan/geo-clip",
        purpose="Original fixed-gallery global geolocation candidate recall.",
        enabled=geoclip_enabled,
        available=geoclip_available,
        status=(global_status.status if geoclip_enabled and global_status else "disabled"),
        installed=global_status.installed if global_status is not None else None,
        weights_available=global_status.installed if global_status is not None else None,
        worker_reachable=None,
        model_loaded=global_status.verified if global_status is not None else None,
        load_verified=global_status.verified if global_status is not None else None,
        real_inference_verified=None,
        device=global_status.device if global_status is not None else None,
        execution_mode="in_process",
        source_revision="7a1a23b49648a5872a771cfda28490a17ab17d15",
        model_revision=(
            global_status.model_revision if global_status is not None else "1.2.0"
        ),
        license="MIT",
        error_code=geoclip_error,
        current_participation=(
            geoclip_participation if geoclip_enabled else "disabled"
        ),
    )

    hierarchy_enabled = bool(
        config.phase6c_enabled
        and config.geoclip_hierarchical_enabled
        and config.geoclip_global_grid_enabled
    )
    hierarchy_available = bool(
        hierarchy_enabled
        and geoclip_available
        and services.phase6c_extension is not None
        and services.phase6c_extension.hierarchical_available
    )
    hierarchy = SystemIntelligenceModelCard(
        model_id="geoclip_hierarchical",
        display_name="GeoCLIP hierarchical search",
        runtime_model_id="GeoCLIP-1.2.0",
        repository_url="https://github.com/VicenteVivan/geo-clip",
        purpose="Bounded global grid, catalogue and geodesic coordinate refinement.",
        enabled=hierarchy_enabled,
        available=hierarchy_available,
        status=(
            "ready"
            if hierarchy_available
            else "disabled"
            if not hierarchy_enabled
            else "unavailable"
        ),
        installed=global_status.installed if global_status is not None else None,
        weights_available=global_status.installed if global_status is not None else None,
        worker_reachable=None,
        model_loaded=global_status.verified if global_status is not None else None,
        load_verified=global_status.verified if global_status is not None else None,
        real_inference_verified=None,
        device=global_status.device if global_status is not None else None,
        execution_mode="in_process",
        source_revision="7a1a23b49648a5872a771cfda28490a17ab17d15",
        model_revision=(
            global_status.model_revision if global_status is not None else "1.2.0"
        ),
        license="MIT",
        error_code=(
            None
            if hierarchy_available
            else "disabled"
            if not hierarchy_enabled
            else "geoclip_hierarchical_unavailable"
        ),
        current_participation="candidate" if hierarchy_enabled else "disabled",
    )

    osv_status = services.osv5m_provider.status()
    osv_enabled = config.phase6b_enabled and config.osv5m_enabled
    osv5m = _worker_intelligence_card(
        binding=services.phase6b_worker_bindings["osv5m"],
        enabled=osv_enabled,
        provider_available=osv_status.available,
        model_id="osv5m",
        display_name="OSV-5M baseline",
        runtime_model_id=osv_status.model_id,
        repository_url="https://github.com/gastruc/osv5m",
        purpose="Independent direct global coordinate regression candidate source.",
        model_revision=osv_status.model_revision,
        source_revision=osv_status.source_revision,
        license_name="MIT",
        requested_device=osv_status.device,
        participation="candidate",
        provider_reason=osv_status.reason_code,
    )

    plonk_status = services.plonk_provider.status()
    plonk_enabled = config.phase6b_enabled and config.plonk_enabled
    plonk = _worker_intelligence_card(
        binding=services.phase6b_worker_bindings["plonk"],
        enabled=plonk_enabled,
        provider_available=plonk_status.available,
        model_id="plonk",
        display_name="PLONK selected runtime",
        runtime_model_id=plonk_status.model_id,
        repository_url="https://github.com/nicolas-dufour/plonk",
        purpose="Scene-routed probabilistic global geolocation samples.",
        model_revision=plonk_status.model_revision,
        source_revision=plonk_status.source_revision,
        license_name="MIT",
        requested_device=plonk_status.device,
        participation="candidate",
        provider_reason=plonk_status.reason_code,
    )

    segmentation_status = services.segmentation_provider.status()
    segmentation_enabled = bool(config.phase6a_enabled and segmentation_status.enabled)
    segmentation_available = bool(
        segmentation_enabled and segmentation_status.usable
    )
    segmentation = SystemIntelligenceModelCard(
        model_id="segformer",
        display_name="AtlasLens SegFormer-B2 v4",
        runtime_model_id=services.segmentation_provider.descriptor.model_name,
        repository_url=None,
        purpose="Descriptive scene segmentation; it has zero direct geographic weight.",
        enabled=segmentation_enabled,
        available=segmentation_available,
        status=(segmentation_status.status if segmentation_enabled else "disabled"),
        installed=segmentation_status.installed,
        weights_available=segmentation_status.prepared,
        worker_reachable=None,
        model_loaded=segmentation_status.loaded,
        load_verified=segmentation_status.loaded,
        real_inference_verified=None,
        device=segmentation_status.device,
        execution_mode="in_process",
        source_revision=None,
        model_revision=services.segmentation_provider.descriptor.version,
        license=None,
        error_code=(
            None
            if segmentation_available
            else "disabled"
            if not segmentation_enabled
            else segmentation_status.load_error or segmentation_status.status
        ),
        current_participation=(
            "descriptive" if segmentation_enabled else "disabled"
        ),
    )

    paddle_status = services.paddleocr_provider.descriptor
    paddle_enabled = config.phase6b_enabled and config.paddleocr_enabled
    paddle_health = services.phase6b_worker_bindings["paddleocr"].health
    paddleocr = _worker_intelligence_card(
        binding=services.phase6b_worker_bindings["paddleocr"],
        enabled=paddle_enabled,
        provider_available=paddle_status.available,
        model_id="paddleocr",
        display_name="PaddleOCR",
        runtime_model_id=paddle_status.model_name or "PaddleOCR",
        repository_url="https://github.com/PaddlePaddle/PaddleOCR",
        purpose="Primary local Turkish and Latin text recognition for place evidence.",
        model_revision=(
            paddle_health.model_revision if paddle_health is not None else paddle_status.version
        ),
        source_revision="211989f046cc1878460f9e65574690c00a127a1a",
        license_name="Apache-2.0",
        requested_device=config.paddleocr_device,
        participation="primary",
        provider_reason=paddle_status.unavailable_reason_code,
    )

    rapid_enabled = _rapidocr_requested(config)
    rapid_installed = bool(
        services.rapidocr_provider is not None
        and services.rapidocr_provider.descriptor.available
    )
    rapid_available = bool(
        rapid_enabled and rapid_installed and services.rapidocr_runtime_verified
    )
    rapid_status: PublicProviderState = (
        "ready"
        if rapid_available
        else "disabled"
        if not rapid_enabled
        else "not_installed"
        if not rapid_installed
        else "inference_not_verified"
    )
    rapidocr = SystemIntelligenceModelCard(
        model_id="rapidocr",
        display_name="RapidOCR",
        runtime_model_id=(
            services.rapidocr_provider.descriptor.model_name
            if services.rapidocr_provider is not None
            else "PP-OCRv6-small-onnx"
        ),
        repository_url="https://github.com/RapidAI/RapidOCR",
        purpose="Bounded local OCR fallback when primary OCR cannot complete.",
        enabled=rapid_enabled,
        available=rapid_available,
        status=rapid_status,
        installed=rapid_installed,
        weights_available=rapid_installed,
        worker_reachable=None,
        model_loaded=None,
        load_verified=services.rapidocr_runtime_verified,
        real_inference_verified=services.rapidocr_runtime_verified,
        device=config.rapidocr_device,
        execution_mode="isolated_process",
        source_revision="7fe716f8e38bb9a43f2680159f38deb14d8b1930",
        model_revision="rapidocr-3.9.1",
        license="Apache-2.0",
        last_success_at=services.rapidocr_last_success_at,
        error_code=(
            None
            if rapid_available
            else "disabled"
            if not rapid_enabled
            else services.rapidocr_runtime_reason or rapid_status
        ),
        current_participation=(
            "fallback" if rapid_enabled and paddle_enabled else "primary"
            if rapid_enabled
            else "disabled"
        ),
    )

    megaloc_enabled = config.phase6c_enabled and config.megaloc_enabled
    megaloc_public_status = _public_megaloc_state(
        megaloc_status.state,
        enabled=megaloc_enabled,
    )
    megaloc = SystemIntelligenceModelCard(
        model_id="megaloc",
        display_name="MegaLoc retrieval",
        runtime_model_id=megaloc_status.model_id,
        repository_url="https://github.com/gmberton/MegaLoc",
        purpose="Visual-place descriptor retrieval over the licensed Türkiye index.",
        enabled=megaloc_enabled,
        available=megaloc_status.available,
        status=megaloc_public_status,
        installed=megaloc_status.installed,
        weights_available=megaloc_status.weights_available,
        worker_reachable=megaloc_status.worker_reachable,
        model_loaded=megaloc_status.model_loaded,
        load_verified=megaloc_status.load_verified,
        real_inference_verified=megaloc_status.real_inference_verified,
        device=megaloc_status.device,
        execution_mode="isolated_worker",
        source_revision=megaloc_status.source_revision,
        model_revision=megaloc_status.model_revision,
        license="MIT",
        error_code=(None if megaloc_status.available else megaloc_status.reason_code),
        current_participation="candidate" if megaloc_enabled else "disabled",
    )

    g3_enabled = bool(config.phase6c_enabled and config.g3_enabled)
    g3 = SystemIntelligenceModelCard(
        model_id="g3",
        display_name="G3 verifier",
        runtime_model_id="Jia-py/G3-checkpoint",
        repository_url="https://github.com/Jia-py/G3",
        purpose="Optional correlated candidate verifier; no runtime is integrated.",
        enabled=g3_enabled,
        available=False,
        status="unavailable" if g3_enabled else "disabled",
        installed=None,
        weights_available=None,
        worker_reachable=None,
        model_loaded=None,
        load_verified=False,
        real_inference_verified=False,
        device=None,
        execution_mode="not_integrated",
        source_revision="b4e3acf7c0ac51221f21b7877fefb4826715c9e2",
        model_revision="12d886fc2a1e59b3b52821acee193084420409cc",
        license="Apache-2.0",
        error_code="not_integrated" if g3_enabled else "disabled",
        current_participation="not_integrated" if g3_enabled else "disabled",
    )

    openai_capability = (
        services.openai_geo_provider.capability()
        if services.openai_geo_provider is not None
        else None
    )
    openai_enabled = bool(config.phase6b_enabled and config.openai_geo_enabled)
    openai_available = bool(
        openai_enabled
        and openai_capability is not None
        and openai_capability.key_configured
        and openai_capability.budget_available
    )
    openai_review = SystemIntelligenceModelCard(
        model_id="openai_geo_review",
        display_name="OpenAI hard-case review",
        runtime_model_id=config.openai_geo_model,
        repository_url=None,
        purpose="Consent-gated optional review of already supplied candidate IDs.",
        enabled=openai_enabled,
        available=openai_available,
        status=(
            "ready"
            if openai_available
            else "disabled"
            if not openai_enabled
            else "unavailable"
        ),
        installed=None,
        weights_available=None,
        worker_reachable=None,
        model_loaded=None,
        load_verified=None,
        real_inference_verified=None,
        device=None,
        execution_mode="cloud",
        source_revision=None,
        model_revision=config.openai_geo_model,
        license=None,
        error_code=(
            None
            if openai_available
            else "disabled"
            if not openai_enabled
            else "missing_api_key"
            if openai_capability is None or not openai_capability.key_configured
            else "budget_unavailable"
        ),
        current_participation=(
            "optional_review" if openai_enabled else "disabled"
        ),
    )

    index = services.phase6c_reference_index
    index_enabled = bool(config.phase6c_enabled and config.reference_index_enabled)
    index_usable = bool(
        index_enabled
        and index.status == "ready"
        and index.leakage_status == "passed"
    )
    index_status: Literal["ready", "empty", "unavailable", "invalid", "disabled"] = (
        "disabled"
        if not index_enabled
        else "unavailable"
        if index.status == "ready" and index.leakage_status != "passed"
        else index.status
    )
    index_health: Literal[
        "healthy", "degraded", "unavailable", "invalid", "disabled"
    ] = (
        "disabled"
        if not index_enabled
        else "healthy"
        if index_usable
        else "invalid"
        if index.status == "invalid" or index.leakage_status == "failed"
        else "unavailable"
        if index.status == "unavailable" or index.leakage_status == "unavailable"
        else "degraded"
    )
    leakage_audit = (
        SystemIntelligenceLeakageAudit(
            status=index.leakage_audit.status,
            audit_version=index.leakage_audit.audit_version,
            audit_fingerprint=index.leakage_audit.audit_fingerprint,
            source_report_sha256=index.leakage_audit.source_report_sha256,
            checked_reference_count=index.leakage_audit.checked_reference_count,
            descriptor_checked_count=index.leakage_audit.descriptor_checked_count,
            excluded_reference_count=index.leakage_audit.excluded_reference_count,
            pre_index_excluded_reference_count=(
                index.leakage_audit.pre_index_excluded_reference_count
            ),
        )
        if index.leakage_audit is not None
        else None
    )
    index_card = SystemIntelligenceReferenceIndexCard(
        enabled=index_enabled,
        status=index_status,
        reason_code=(
            index.reason_code
            if not index_enabled or index.status != "ready" or index_usable
            else "reference_index_leakage_not_passed"
        ),
        index_version=index.index_version,
        descriptor_version=index.descriptor_version,
        count=index.count,
        sequences=index.independent_sequences,
        countries=index.country_count,
        provinces=index.province_count,
        images_per_province=index.images_per_province,
        source_distribution=index.source_distribution,
        built_at=index.built_at,
        disk_usage_bytes=index.disk_usage_bytes,
        leakage_status=index.leakage_status,
        leakage_audit=leakage_audit,
        duplicates=index.duplicate_count,
        excluded=index.excluded_count,
        attributions=list(index.attributions or ()),
        health=index_health,
    )
    return SystemIntelligenceResponse(
        active_pipeline_version=(
            config.phase6c_pipeline_version if config.phase6c_enabled else "legacy-v1"
        ),
        models=[
            geoclip,
            hierarchy,
            osv5m,
            plonk,
            segmentation,
            paddleocr,
            rapidocr,
            megaloc,
            g3,
            openai_review,
        ],
        reference_index=index_card,
    )


def _parse_uuid(raw: str, *, delete: bool = False) -> UUID:
    try:
        return UUID(raw)
    except ValueError as exc:
        if delete:
            raise bad_request("invalid_analysis_id", "error.invalid_analysis_id") from exc
        raise AppError(404, "analysis_not_found", "error.analysis_not_found", "Not found") from exc


def create_app(
    settings: Settings | None = None,
    *,
    repository: AnalysisRepository | None = None,
    exif_provider: ExifProvider | None = None,
    quality_provider: ImageQualityProvider | None = None,
    ocr_provider: OCRProvider | None = None,
    vision_provider: VisionClueProvider | None = None,
    global_provider: GlobalGeolocationProvider | None = None,
    inference_coordinator: ProviderEnsembleCoordinator | None = None,
    retrieval_provider: RetrievalProvider | None = None,
    map_evaluator: BoundedMapEvidenceEvaluator | None = None,
    segmentation_provider: SceneSegmentationProvider | None = None,
    phase6a_clusterer: GeoClipCandidateClusterer | None = None,
    phase6a_reverse_geocoder: CachedClusterReverseGeocoder | None = None,
    phase6a_reranker: Phase6AHybridEvidenceEngine | None = None,
    turkiye_reference_provider: TurkiyeReferenceIndexProvider | None = None,
    mapillary_demo_provider: MapillaryDemoRuntime | None = None,
) -> FastAPI:
    config = settings or Settings()
    configure_logging(config.log_level)
    engine = create_database_engine(config.database_url)
    analysis_repository = repository or SQLAlchemyAnalysisRepository(engine)
    case_repository = SQLAlchemyCaseRepository(engine)
    case_service = CaseInvestigationService(case_repository, analysis_repository)
    storage = LocalTemporaryStorage(config.temp_storage_dir)
    image_processor = SafeImageProcessor(config, storage)
    turkiye_reference = turkiye_reference_provider or TurkiyeReferenceIndexProvider(
        enabled=config.turkiye_reference_index_enabled,
        index_path=config.turkiye_reference_index_path,
        source_policy_path=config.turkiye_reference_source_policy_path,
        descriptor_provider=config.turkiye_reference_descriptor_id,
        descriptor_version=config.turkiye_reference_descriptor_version,
        descriptor_dimension=config.turkiye_reference_descriptor_dimension,
        descriptor_artifact_sha256=(
            config.turkiye_reference_descriptor_artifact_sha256
        ),
        descriptor_preprocessing_version=(
            config.turkiye_reference_descriptor_preprocessing_version
        ),
        descriptor_artifact_approved=_phase3b2_megaloc_artifact_approved(config),
    )
    broker = AnalysisEventBroker()
    queue = InProcessJobQueue(config.max_queued_jobs)
    exif = exif_provider or PillowExifProvider()
    quality = quality_provider or OpenCVImageQualityProvider()
    if vision_provider is not None:
        vision = vision_provider
    elif config.cloud_vision_provider == "nvidia":
        vision = NvidiaVisionClueProvider(
            enabled=config.nvidia_vision_enabled,
            api_key=config.nvidia_key_value,
            model=config.nvidia_vision_model,
            timeout_seconds=config.nvidia_vision_timeout_seconds,
            maximum_edge=config.nvidia_vision_maximum_image_edge,
            jpeg_quality=config.nvidia_vision_jpeg_quality,
        )
    else:
        vision = OpenAIVisionClueProvider(
            api_key=config.openai_key_value,
            model=config.openai_vision_model,
            timeout_seconds=config.openai_vision_timeout_seconds,
        )
    global_model = global_provider
    if global_model is None and config.global_model_enabled:
        model_management = ModelManagementService(
            config.model_cache_dir, load_manifest(default_manifest_path())
        )
        global_model = GeoCLIPGlobalGeolocationProvider(
            model_management,
            device_selector=lambda: select_device(config.global_model_device),
            timeout_seconds=config.global_model_timeout_seconds,
            max_concurrency=config.global_model_max_concurrency,
            internal_top_k=config.geoclip_internal_top_k,
        )
    trained_artifact_manager = TrainedArtifactManager(config.model_cache_dir)
    selected_inference_coordinator = inference_coordinator
    if selected_inference_coordinator is None:
        inference_providers: list[GeolocationInferenceProvider] = []
        if config.enable_mock_inference:
            inference_providers.append(
                DevelopmentMockProvider(
                    config.mock_fixture_name,
                    environment=config.app_env,
                    mode="primary",
                )
            )
        else:
            custom_provider = build_custom_provider(
                trained_artifact_manager,
                config.custom_model_id,
                enabled=config.custom_model_enabled,
                device=config.custom_model_device,
                max_input_bytes=config.max_upload_bytes,
            )
            if global_model is not None:
                geoclip_mode: Literal["candidate", "primary"] = (
                    "candidate" if custom_provider.mode == "primary" else "primary"
                )
                inference_providers.append(GeoCLIPInferenceAdapter(global_model, mode=geoclip_mode))
            inference_providers.append(custom_provider)
        if inference_providers:
            selected_inference_coordinator = ProviderEnsembleCoordinator(tuple(inference_providers))
    gazetteer = None
    gazetteer_manager = GazetteerManager(config.gazetteer_cache_dir)
    gazetteer_info = gazetteer_manager.info()
    if (
        gazetteer_info.status == "ready"
        and gazetteer_info.schema_version is not None
        and gazetteer_info.license is not None
    ):
        gazetteer = SQLiteGazetteerResolver(
            gazetteer_manager.database_path,
            GazetteerMetadata(
                dataset=gazetteer_info.dataset,
                version=gazetteer_info.schema_version,
                source="GeoNames",
                license=gazetteer_info.license,
            ),
        )

    place_service = None
    forward_manager = ForwardGazetteerManager(config.gazetteer_cache_dir)
    forward_info = forward_manager.info()
    if (
        forward_info.status == "ready"
        and forward_info.schema_version is not None
        and forward_info.license is not None
    ):
        forward_metadata = GazetteerMetadata(
            dataset=forward_info.dataset,
            version=forward_info.schema_version,
            source="GeoNames",
            license=forward_info.license,
        )
        place_service = PlaceEvidenceService(
            SQLiteForwardGazetteerResolver(forward_manager.database_path, forward_metadata),
            forward_metadata,
        )

    phase6b_ocr_enabled = config.phase6b_enabled and config.paddleocr_enabled
    paddle_binding = _connect_worker(
        enabled=phase6b_ocr_enabled and config.paddleocr_worker_enabled,
        provider="paddleocr",
        host=config.paddleocr_worker_host,
        port=config.paddleocr_worker_port,
        provider_revision="3.7.0",
        model_revisions=("PP-OCRv5_server_det+latin_PP-OCRv5_mobile_rec",),
        timeout_seconds=config.paddleocr_timeout_seconds,
        max_image_bytes=config.max_upload_bytes,
    )
    paddle_client = paddle_binding.client
    paddle_worker = (
        PaddleOCRHTTPWorkerClient(paddle_client, device=config.paddleocr_device)
        if paddle_binding.ready and paddle_client is not None
        else None
    )
    paddleocr = PaddleOCRProvider(
        enabled=phase6b_ocr_enabled,
        worker=paddle_worker,
        artifact_verified=paddle_binding.ready,
        place_service=place_service,
        provider_version="3.7.0",
        model_name="PP-OCRv5_server_det + latin_PP-OCRv5_mobile_rec",
        device=config.paddleocr_device,
        timeout_seconds=config.paddleocr_timeout_seconds,
        max_input_bytes=config.max_upload_bytes,
        max_decoded_pixels=config.max_decoded_pixels,
        max_side=config.max_image_dimension,
    )
    rapidocr: RapidOCRProvider | None = None
    if ocr_provider is not None:
        ocr = ocr_provider
    elif config.ocr_provider == "rapidocr":
        rapidocr_enabled = _rapidocr_requested(config)
        rapidocr_runtime = load_verified_rapidocr_runtime(
            config.rapidocr_model_root, device=config.rapidocr_device
        )
        rapidocr = RapidOCRProvider(
            enabled=rapidocr_enabled,
            runtime=rapidocr_runtime,
            place_service=place_service,
            timeout_seconds=config.rapidocr_timeout_seconds,
            max_input_bytes=config.max_upload_bytes,
            max_decoded_pixels=config.max_decoded_pixels,
            max_side=config.max_image_dimension,
        )
        ocr = PreferredOCRProvider(paddleocr, rapidocr) if phase6b_ocr_enabled else rapidocr
    else:
        ocr = TesseractOCRProvider(enabled=config.ocr_enabled, command=config.tesseract_cmd)

    rapid_verification = (
        validate_rapidocr_runtime_verification(
            Path(__file__).resolve().parents[4],
            config.rapidocr_model_root,
            expected_source_revision="7fe716f8e38bb9a43f2680159f38deb14d8b1930",
            expected_model_revision="rapidocr-3.9.1",
        )
        if rapidocr is not None and rapidocr.descriptor.available
        else None
    )
    rapid_runtime_verified = bool(
        rapid_verification is not None and rapid_verification.verification is not None
    )
    rapid_runtime_reason = (
        None
        if rapid_runtime_verified
        else rapid_verification.reason_code
        if rapid_verification is not None
        else rapidocr.descriptor.unavailable_reason_code
        if rapidocr is not None
        else "disabled"
    )

    retrieval = retrieval_provider
    if retrieval is None:
        retrieval_service = None
        retrieval_reason: str | None = None
        if not config.phase5b_enabled or not config.retrieval_enabled:
            retrieval_reason = "disabled"
        else:
            try:
                embedding_provider = production_embedding_provider(
                    config.retrieval_embedding_provider,
                    cache_root=config.model_cache_dir,
                    requested_device=config.retrieval_device,
                )
                if not embedding_provider.available:
                    retrieval_reason = (
                        embedding_provider.unavailable_reason or "model_not_installed"
                    )
                else:
                    retrieval_service = open_retrieval_query_service(
                        config.retrieval_index_dir, embedding_provider
                    )
            except (ImportError, OSError, RetrievalError, ValueError):
                retrieval_reason = "index_unavailable"
        retrieval = FaissRetrievalProvider(
            retrieval_service,
            enabled=config.phase5b_enabled and config.retrieval_enabled,
            top_k=config.retrieval_top_k,
            timeout_seconds=config.retrieval_timeout_seconds,
            max_input_bytes=config.max_upload_bytes,
            unavailable_reason=retrieval_reason,
        )

    selected_map_evaluator = map_evaluator
    if selected_map_evaluator is None and config.map_evidence_enabled:
        try:
            map_provider = OverpassMapConstraintProvider(
                PinnedHttpsMapTransport(),
                endpoint=config.overpass_endpoint,
                user_agent=config.overpass_user_agent,
                timeout_seconds=config.overpass_timeout_seconds,
            )
        except (OSError, ValueError):
            selected_map_evaluator = None
        else:
            selected_map_evaluator = BoundedMapEvidenceEvaluator(
                map_provider, timeout_seconds=config.overpass_timeout_seconds
            )

    segmentation = segmentation_provider or SegFormerSceneProvider(
        enabled=config.phase6a_enabled and config.segmentation_enabled,
        model_directory=config.segmentation_model_dir,
        requested_device=config.segmentation_device,
        minimum_class_ratio=config.segmentation_min_class_ratio,
        maximum_dominant_classes=config.segmentation_max_dominant_classes,
        timeout_seconds=config.segmentation_timeout_seconds,
        max_input_bytes=config.max_upload_bytes,
        max_decoded_pixels=config.max_decoded_pixels,
        max_image_dimension=config.max_image_dimension,
    )
    selected_phase6a_clusterer = phase6a_clusterer
    if selected_phase6a_clusterer is None and config.phase6a_enabled:
        selected_phase6a_clusterer = GeoClipCandidateClusterer(
            radius_km=config.geoclip_cluster_radius_km,
            max_clusters=config.geoclip_internal_top_k,
        )
    selected_reverse_geocoder = phase6a_reverse_geocoder or CachedClusterReverseGeocoder(
        gazetteer,
        SQLiteReverseGeocodeCache(config.gazetteer_cache_dir / "reverse-geocode-cache.sqlite3"),
        top_k=config.reverse_geocode_top_k,
        timeout_seconds=config.reverse_geocode_timeout_seconds,
    )
    selected_phase6a_reranker = phase6a_reranker
    if selected_phase6a_reranker is None and config.phase6a_enabled:
        selected_phase6a_reranker = Phase6AHybridEvidenceEngine(
            Phase6AHybridConfig.from_path(default_phase6a_config_path())
        )

    scheduler = HeavyModelScheduler(
        max_heavy_concurrency=config.gpu_max_heavy_concurrency,
        max_resident_models=config.gpu_max_resident_models,
    )
    cuda_available = lambda: select_device("auto") == "cuda"  # noqa: E731
    osv5m_binding = _connect_worker(
        enabled=(
            config.phase6b_enabled
            and config.osv5m_enabled
            and config.osv5m_worker_enabled
        ),
        provider="osv5m",
        host=config.osv5m_worker_host,
        port=config.osv5m_worker_port,
        provider_revision="4e6075387ecde4255410785ffb83830c9aa099f6",
        model_revisions=(config.osv5m_model_revision,),
        timeout_seconds=config.osv5m_timeout_seconds,
        max_image_bytes=config.max_upload_bytes,
    )
    osv5m_client = osv5m_binding.client
    osv5m_worker = (
        OSV5MHTTPWorkerClient(osv5m_client, model_id=config.osv5m_model_id)
        if osv5m_binding.ready and osv5m_client is not None
        else None
    )
    osv5m = OSV5MProvider(
        enabled=config.phase6b_enabled and config.osv5m_enabled,
        worker=osv5m_worker,
        scheduler=scheduler,
        model_id=config.osv5m_model_id,
        model_revision=config.osv5m_model_revision,
        source_revision="4e6075387ecde4255410785ffb83830c9aa099f6",
        device=config.osv5m_device,
        cuda_available=cuda_available,
        timeout_seconds=config.osv5m_timeout_seconds,
        max_input_bytes=config.max_upload_bytes,
    )
    plonk_router = PlonkModelRouter(
        osv_model_id=config.plonk_osv_model_id,
        yfcc_model_id=config.plonk_yfcc_model_id,
        inat_model_id=config.plonk_inat_model_id,
    )
    plonk_revisions = {
        config.plonk_osv_model_id: "e23229f4dd91d52560e8827f5bb2c68257fa162f",
        config.plonk_yfcc_model_id: "4f358d09938a89ed239a847777729e95c5d187bc",
        config.plonk_inat_model_id: "8da6edcbdd01ff04a61f9d06e2de23ea300d1a35",
    }
    plonk_binding = _connect_worker(
        enabled=(
            config.phase6b_enabled
            and config.plonk_enabled
            and config.plonk_worker_enabled
        ),
        provider="plonk",
        host=config.plonk_worker_host,
        port=config.plonk_worker_port,
        provider_revision="76d46410910c9dfec9e19ed371450ebc7051cdf3",
        model_revisions=("scene-routed", *plonk_revisions.values()),
        timeout_seconds=config.plonk_timeout_seconds,
        max_image_bytes=config.max_upload_bytes,
    )
    plonk_client = plonk_binding.client
    plonk_worker = (
        PlonkHTTPWorkerClient(plonk_client)
        if plonk_binding.ready and plonk_client is not None
        else None
    )
    plonk = PlonkProvider(
        enabled=config.phase6b_enabled and config.plonk_enabled,
        worker=plonk_worker,
        scheduler=scheduler,
        router=plonk_router,
        model_revisions=plonk_revisions,
        source_revision="76d46410910c9dfec9e19ed371450ebc7051cdf3",
        device=config.plonk_device,
        cuda_available=cuda_available,
        sample_count=config.plonk_sample_count,
        timeout_seconds=config.plonk_timeout_seconds,
        max_input_bytes=config.max_upload_bytes,
    )
    try:
        phase6b_fusion_config = load_phase6b_fusion_config(config.phase6b_fusion_config)
    except (OSError, ValueError):
        phase6b_fusion_config = Phase6BFusionConfig()
    phase6b_fusion = Phase6BGeographicFusionEngine(phase6b_fusion_config)

    openai_geo_payload = OpenAIGeoReviewConfig.from_path(
        default_openai_geo_config_path()
    ).model_dump(mode="python")
    openai_geo_payload.update(
        {
            "enabled": config.openai_geo_enabled,
            "model": config.openai_geo_model,
            "reasoning_effort": config.openai_geo_reasoning_effort,
            "image_detail": config.openai_geo_image_detail,
            "max_output_tokens": config.openai_geo_max_output_tokens,
            "timeout_seconds": config.openai_geo_timeout_seconds,
            "maximum_image_edge": config.openai_geo_maximum_image_edge,
            "monthly_budget_usd": Decimal(str(config.openai_geo_monthly_budget_usd)),
            "daily_call_limit": config.openai_geo_daily_call_limit,
            "per_analysis_call_limit": config.openai_geo_per_analysis_call_limit,
            "cache_ttl_days": config.openai_geo_cache_ttl_days,
            "allow_high_detail_retry": config.openai_geo_allow_high_detail_retry,
        }
    )
    openai_geo_config = OpenAIGeoReviewConfig.model_validate(openai_geo_payload)
    openai_geo_provider: OpenAIGeoReviewProvider | None = None
    if config.phase6b_enabled and openai_geo_config.enabled:
        try:
            openai_geo_provider = build_openai_geo_review_provider(
                openai_geo_config,
                ledger_path=config.openai_geo_ledger_path,
                api_key=config.openai_key_value,
            )
        except (OSError, ValueError):
            openai_geo_provider = None
    phase6b_extension = (
        Phase6BPipelineExtension(
            local_ensemble=Phase6BLocalModelEnsemble(
                osv5m=osv5m,
                plonk=plonk,
                fusion=phase6b_fusion,
            ),
            openai_provider=openai_geo_provider,
            openai_config=openai_geo_config,
        )
        if config.phase6b_enabled
        else None
    )

    phase6c_hierarchical: GeoCLIPHierarchicalSearchProvider | None = None
    if (
        config.phase6c_enabled
        and config.geoclip_hierarchical_enabled
        and config.geoclip_global_grid_enabled
        and isinstance(global_model, GeoCLIPGlobalGeolocationProvider)
    ):
        try:
            refinement_levels = tuple(
                float(item.strip())
                for item in config.geoclip_grid_refinement_levels_km.split(",")
                if item.strip()
            )
            phase6c_hierarchical = GeoCLIPHierarchicalSearchProvider(
                global_model,
                load_coordinate_catalogue(),
                GeoCLIPHierarchicalConfig(
                    global_grid_points=config.geoclip_grid_points,
                    refinement_levels_km=refinement_levels,
                    output_candidates=config.geoclip_grid_max_candidates,
                    timeout_seconds=min(180.0, config.global_model_timeout_seconds),
                ),
            )
        except (OSError, ValueError):
            phase6c_hierarchical = None

    if config.phase6c_enabled and config.reference_index_enabled:
        phase6c_index_result = open_reference_index(config.reference_index_path)
    else:
        phase6c_index_result = None
    phase6c_index_diagnostics = (
        phase6c_index_result.diagnostics
        if phase6c_index_result is not None
        else ReferenceIndexDiagnostics(
            status="unavailable",
            reason_code="index_disabled",
            count=0,
            independent_sequences=0,
            disk_usage_bytes=0,
        )
    )
    leakage_attestation = phase6c_index_diagnostics.leakage_audit
    if (
        phase6c_index_diagnostics.leakage_status == "passed"
        and leakage_attestation is not None
    ):
        phase6c_leakage_summary = Phase6CLeakageAuditSummary(
            status="passed",
            audit_version=leakage_attestation.audit_version,
            report_fingerprint=leakage_attestation.audit_fingerprint,
            references_checked=leakage_attestation.checked_reference_count,
            references_excluded=leakage_attestation.excluded_reference_count,
        )
    elif phase6c_index_diagnostics.leakage_status in {"failed", "unavailable"}:
        phase6c_leakage_summary = Phase6CLeakageAuditSummary(
            status="failed",
            audit_version="atlaslens-leakage-audit-v1",
            references_checked=0,
            references_excluded=0,
            reason_code=(
                "reference_index_leakage_attestation_invalid"
                if phase6c_index_diagnostics.leakage_status == "failed"
                else "reference_index_leakage_attestation_unavailable"
            ),
        )
    else:
        phase6c_leakage_summary = Phase6CLeakageAuditSummary(
            status="not_run",
            audit_version="atlaslens-leakage-audit-v1",
            references_checked=0,
            references_excluded=0,
            reason_code="reference_index_leakage_not_attested",
        )
    megaloc_worker = None
    if (
        (
            (config.phase6c_enabled and config.megaloc_enabled)
            or config.atlaslens_turkiye_demo_enabled
        )
        and config.megaloc_worker_enabled
    ):
        try:
            megaloc_worker = create_megaloc_http_worker_client(
                host=config.megaloc_worker_host,
                port=config.megaloc_worker_port,
                timeout_seconds=config.megaloc_timeout_seconds,
                max_image_bytes=config.max_upload_bytes,
            )
        except ValueError:
            megaloc_worker = None
    try:
        megaloc_cuda_available = (
            select_device("auto") == "cuda"
            if config.phase6c_enabled or config.atlaslens_turkiye_demo_enabled
            else False
        )
    except (ImportError, RuntimeError):
        megaloc_cuda_available = False
    megaloc = MegaLocRetrievalProvider(
        enabled=config.phase6c_enabled and config.megaloc_enabled,
        worker=megaloc_worker,
        reference_index=(
            phase6c_index_result.index
            if phase6c_index_result is not None
            and config.reference_index_enabled
            and phase6c_index_diagnostics.leakage_status == "passed"
            else None
        ),
        scheduler=scheduler,
        device=config.megaloc_device,
        cuda_available=megaloc_cuda_available,
        allow_cpu_fallback=config.megaloc_device == "auto",
        config=MegaLocRetrievalConfig(
            timeout_seconds=config.megaloc_timeout_seconds,
            max_input_bytes=config.max_upload_bytes,
            max_matches=config.megaloc_top_k,
        ),
    )
    demo_device: Literal["cuda", "cpu"] = (
        "cuda"
        if config.megaloc_device == "auto" and megaloc_cuda_available
        else "cpu"
        if config.megaloc_device == "auto"
        else config.megaloc_device
    )
    mapillary_demo = mapillary_demo_provider or MapillaryDemoRuntime(
        index=PublishedMapillaryDemoSearchIndex(
            enabled=config.atlaslens_turkiye_demo_enabled,
            bundle_path=config.atlaslens_turkiye_demo_bundle_path,
            expected_publication_sha256=(
                config.atlaslens_turkiye_demo_expected_publication_sha256
            ),
            expected_source_policy_sha256=(
                config.atlaslens_turkiye_demo_expected_source_policy_sha256
            ),
            expected_selection_lock_sha256=(
                config.atlaslens_turkiye_demo_expected_selection_lock_sha256
            ),
            uncertainty_radius_m=(
                config.atlaslens_turkiye_demo_uncertainty_radius_m
            ),
        ),
        worker=megaloc_worker,
        device=demo_device,
        timeout_seconds=config.megaloc_timeout_seconds,
        top_k=config.atlaslens_turkiye_demo_top_k,
    )
    if config.phase6c_enabled:
        try:
            phase6c_ocr_config = load_phase6c_ocr_config(config.phase6c_ocr_config)
        except (OSError, ValueError):
            raise ValueError("phase6c_ocr_config_invalid") from None
    else:
        phase6c_ocr_config = Phase6COCRConfig()
    phase6c_ocr_config_digest = hashlib.sha256(
        phase6c_ocr_config.model_dump_json().encode()
    ).hexdigest()
    if config.phase6c_enabled:
        try:
            phase6c_fusion_config = load_phase6c_fusion_config(
                config.phase6c_fusion_config
            )
        except (OSError, ValueError):
            raise ValueError("phase6c_fusion_config_invalid") from None
    else:
        phase6c_fusion_config = Phase6CFusionConfig()
    phase6c_fusion = Phase6CGeographicFusionEngine(phase6c_fusion_config)
    phase6c_config_digest = hashlib.sha256(
        phase6c_fusion_config.model_dump_json().encode()
    ).hexdigest()
    global_revision = (
        global_model.status().model_revision if global_model is not None else None
    )
    phase6c_extension = (
        Phase6CPipelineExtension(
            hierarchical=phase6c_hierarchical,
            megaloc=megaloc,
            fusion=phase6c_fusion,
            cache_versions=Phase6CCacheVersions(
                pipeline_version=config.phase6c_pipeline_version,
                provider_versions=(
                    "geoclip-global-v1",
                    "geoclip-hierarchical-v1",
                    "osv5m:4e6075387ecde4255410785ffb83830c9aa099f6",
                    "plonk:76d46410910c9dfec9e19ed371450ebc7051cdf3",
                    f"megaloc:{MEGALOC_SOURCE_REVISION}",
                    f"ocr:{ocr.descriptor.id}:{ocr.descriptor.version}",
                ),
                model_revisions=tuple(
                    item
                    for item in (
                        global_revision,
                        config.osv5m_model_revision,
                        *plonk_revisions.values(),
                        MEGALOC_MODEL_REVISION,
                        MEGALOC_DESCRIPTOR_VERSION,
                    )
                    if item is not None
                ),
                fusion_config_version=phase6c_fusion_config.version,
                fusion_config_sha256=phase6c_config_digest,
                reference_index_version=phase6c_index_diagnostics.index_version,
                ocr_version=(
                    f"{ocr.descriptor.id}:{ocr.descriptor.version}:"
                    f"{phase6c_ocr_config.version}:{phase6c_ocr_config_digest}"
                ),
                openai_prompt_version="openai-geo-review-v1",
                reference_index_leakage_attestation=(
                    f"{phase6c_index_diagnostics.leakage_status}:"
                    f"{leakage_attestation.audit_fingerprint if leakage_attestation else 'none'}"
                ),
            ),
            ocr_descriptor=ocr.descriptor,
            ocr_config=phase6c_ocr_config,
            leakage_audit=phase6c_leakage_summary,
            turkiye_refinement_enabled=config.geoclip_turkiye_refinement_enabled,
        )
        if config.phase6c_enabled
        else None
    )

    phase5b_reranker = (
        Phase5BEvidenceReranker(Phase5BRerankConfig.from_path(default_phase5b_config_path()))
        if config.phase5b_enabled
        else None
    )
    fusion = DeterministicCandidateFusionService()
    pipeline = AnalysisPipeline(
        repository=analysis_repository,
        storage=storage,
        image_processor=image_processor,
        broker=broker,
        exif_provider=exif,
        quality_provider=quality,
        ocr_provider=ocr,
        vision_provider=vision,
        fusion=fusion,
        keep_uploads=config.keep_uploads,
        global_provider=global_model,
        inference_coordinator=selected_inference_coordinator,
        max_inference_input_bytes=config.max_upload_bytes,
        inference_top_k=config.geoclip_internal_top_k,
        gazetteer=gazetteer,
        global_prediction_timeout_seconds=config.global_model_timeout_seconds,
        ocr_timeout_seconds=max(
            config.paddleocr_timeout_seconds,
            config.rapidocr_timeout_seconds,
        ),
        vision_timeout_seconds=max(
            40.0,
            (
                config.nvidia_vision_timeout_seconds
                if config.cloud_vision_provider == "nvidia"
                else config.openai_vision_timeout_seconds
            ),
        ),
        retrieval_provider=retrieval,
        retrieval_timeout_seconds=config.retrieval_timeout_seconds,
        phase5b_reranker=phase5b_reranker,
        map_evaluator=selected_map_evaluator,
        phase6a_reranker=selected_phase6a_reranker,
        geoclip_clusterer=selected_phase6a_clusterer,
        reverse_geocoder=selected_reverse_geocoder,
        segmentation_provider=(segmentation if config.phase6a_enabled else None),
        segmentation_timeout_seconds=config.segmentation_timeout_seconds,
        phase6b_extension=phase6b_extension,
        phase6c_extension=phase6c_extension,
    )
    cleanup = DefaultRetentionCleanupService(
        analysis_repository,
        storage,
        ttl_seconds=config.retention_ttl_seconds,
    )
    services = AppServices(
        settings=config,
        repository=analysis_repository,
        case_service=case_service,
        storage=storage,
        image_processor=image_processor,
        queue=queue,
        broker=broker,
        pipeline=pipeline,
        cleanup=cleanup,
        rate_limiter=SlidingWindowRateLimiter(),
        connection_limiter=ConnectionLimiter(),
        exif_provider=exif,
        quality_provider=quality,
        ocr_provider=ocr,
        vision_provider=vision,
        global_provider=global_model,
        inference_coordinator=selected_inference_coordinator,
        trained_artifact_manager=trained_artifact_manager,
        retrieval_provider=retrieval,
        segmentation_provider=segmentation,
        reverse_geocoder=selected_reverse_geocoder,
        phase6b_extension=phase6b_extension,
        phase6c_extension=phase6c_extension,
        megaloc_provider=megaloc,
        phase6c_reference_index=phase6c_index_diagnostics,
        turkiye_reference_provider=turkiye_reference,
        mapillary_demo_provider=mapillary_demo,
        osv5m_provider=osv5m,
        plonk_provider=plonk,
        paddleocr_provider=paddleocr,
        rapidocr_provider=rapidocr,
        phase6b_worker_bindings={
            "osv5m": osv5m_binding,
            "plonk": plonk_binding,
            "paddleocr": paddle_binding,
        },
        heavy_model_scheduler=scheduler,
        rapidocr_runtime_verified=rapid_runtime_verified,
        rapidocr_runtime_reason=rapid_runtime_reason,
        rapidocr_last_success_at=(
            rapid_verification.verification.verified_at
            if rapid_verification is not None
            and rapid_verification.verification is not None
            else None
        ),
        openai_geo_provider=openai_geo_provider,
        openai_geo_config=openai_geo_config,
        engine=engine,
        active_artifacts={},
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await storage.initialize()
        await analysis_repository.initialize()
        await case_service.initialize()
        await cleanup.startup_cleanup()
        await queue.start()
        await cleanup.start()
        try:
            yield
        finally:
            await cleanup.stop()
            await queue.shutdown()
            if phase6c_extension is not None:
                await phase6c_extension.close()
            elif config.atlaslens_turkiye_demo_enabled:
                await services.mapillary_demo_provider.close()
            await scheduler.close()
            for provider in (ocr, retrieval, segmentation, vision):
                close = getattr(provider, "close", None)
                if callable(close):
                    result = close()
                    if inspect.isawaitable(result):
                        await result
            for binding in services.phase6b_worker_bindings.values():
                if binding.client is not None:
                    await binding.client.close()
            dispose = getattr(engine, "dispose", None)
            if dispose is not None:
                await asyncio.to_thread(dispose)

    app = FastAPI(
        title="AtlasLens API",
        version=config.app_version,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.services = services
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Idempotency-Key", "X-Request-ID", "Accept"],
        expose_headers=["Location", "X-Request-ID", "Retry-After"],
        max_age=600,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.monotonic()
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = supplied if _REQUEST_ID.fullmatch(supplied) else uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        _logger.info(
            "http_request_completed",
            extra={
                "event": "http_request_completed",
                "request_id": request.state.request_id,
                "method": request.method,
                "path": _safe_route_path(request),
                "status_code": response.status_code,
                "duration_ms": int((time.monotonic() - started) * 1000),
            },
        )
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
        return _problem(error, request)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, _: RequestValidationError) -> JSONResponse:
        return _problem(
            AppError(422, "validation_error", "error.validation", "Validation failed"),
            request,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, _: Exception) -> JSONResponse:
        _logger.error(
            "unhandled_request_error",
            extra={
                "event": "unhandled_request_error",
                "request_id": _request_id(request),
                "path": _safe_route_path(request),
                "error_code": "internal_server_error",
            },
        )
        return _problem(
            AppError(500, "internal_server_error", "error.internal", "Internal server error"),
            request,
        )

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(version=config.app_version)

    @app.get("/api/v1/ready", response_model=ReadinessResponse)
    async def ready(response: Response) -> ReadinessResponse:
        database_ready = await analysis_repository.is_ready()
        storage_ready = storage.root.exists() and storage.root.is_dir()
        turkiye_reference_status = services.turkiye_reference_provider.status()
        mapillary_demo_status = await services.mapillary_demo_provider.status()
        checks: dict[str, Literal["ok", "error"]] = {
            "database": "ok" if database_ready else "error",
            "temporary_storage": "ok" if storage_ready else "error",
            "turkiye_reference_index": (
                "error" if turkiye_reference_status.state == "not_ready" else "ok"
            ),
            "turkiye_mapillary_demo": (
                "error" if mapillary_demo_status.state == "not_ready" else "ok"
            ),
        }
        is_ready = all(value == "ok" for value in checks.values())
        if not is_ready:
            response.status_code = 503
        return ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            checks=checks,
        )

    @app.get("/api/v1/capabilities", response_model=CapabilitiesResponse)
    async def capabilities() -> CapabilitiesResponse:
        await asyncio.gather(
            *(binding.refresh() for binding in services.phase6b_worker_bindings.values())
        )
        megaloc_status = (
            await services.phase6c_extension.megaloc_status()
            if services.phase6c_extension is not None
            else None
        )
        turkiye_reference_status = services.turkiye_reference_provider.status()
        mapillary_demo_status = await services.mapillary_demo_provider.status()
        legacy_cloud_available = services.vision_provider.descriptor.available
        openai_geo_status = (
            services.openai_geo_provider.capability()
            if services.openai_geo_provider is not None
            else None
        )
        openai_geo_ready = bool(
            openai_geo_status is not None
            and openai_geo_status.enabled
            and openai_geo_status.key_configured
        )
        cloud_available = legacy_cloud_available or openai_geo_ready
        enabled_modes = [AnalysisMode.LOCAL_ONLY]
        if cloud_available:
            enabled_modes.append(AnalysisMode.CLOUD_ASSISTED)

        def capability(provider: Any, enabled: bool) -> ProviderCapability:
            descriptor = provider.descriptor
            return ProviderCapability(
                provider_id=descriptor.id,
                enabled=enabled,
                available=descriptor.available,
                execution_boundary=descriptor.execution_boundary,
                reason_code=descriptor.unavailable_reason_code,
                provider_type=descriptor.kind,
                offline=descriptor.execution_boundary == "local",
            )

        global_capability = None
        if services.global_provider is not None:
            global_status = services.global_provider.status()
            descriptor = services.global_provider.descriptor
            global_capability = ProviderCapability(
                provider_id=descriptor.id,
                enabled=config.global_model_enabled,
                available=global_status.status == "ready",
                execution_boundary=descriptor.execution_boundary,
                reason_code=global_status.reason_code
                or (None if global_status.status == "ready" else global_status.status),
                installed=global_status.installed,
                verified=global_status.verified,
                operational_status=global_status.status,
                model_name=global_status.model_name,
                model_revision=global_status.model_revision,
                device=global_status.device,
                calibration_state=global_status.calibration_state,
                weights_available=global_status.verified,
                usable=global_status.status == "ready",
                execution_mode="in_process",
                source_revision="7a1a23b49648a5872a771cfda28490a17ab17d15",
            )
        hierarchy_enabled = bool(
            config.phase6c_enabled
            and config.geoclip_hierarchical_enabled
            and config.geoclip_global_grid_enabled
        )
        hierarchy_available = bool(
            hierarchy_enabled
            and services.phase6c_extension is not None
            and services.phase6c_extension.hierarchical_available
            and global_capability is not None
            and global_capability.available
        )
        hierarchy_capability = ProviderCapability(
            provider_id="geoclip_hierarchical_search",
            enabled=hierarchy_enabled,
            available=hierarchy_available,
            execution_boundary="local",
            reason_code=(
                None
                if hierarchy_available
                else "disabled"
                if not hierarchy_enabled
                else "geoclip_hierarchical_unavailable"
            ),
            installed=bool(global_capability and global_capability.installed),
            verified=bool(global_capability and global_capability.verified),
            operational_status=(
                "ready"
                if hierarchy_available
                else "disabled"
                if not hierarchy_enabled
                else "unavailable"
            ),
            model_name="GeoCLIP hierarchical coordinate search",
            model_revision="geoclip-hierarchical-v1",
            device=global_capability.device if global_capability is not None else None,
            calibration_state="uncalibrated",
            provider_type="global_geolocation",
            offline=True,
            limitation="Raw coordinate cosine scores are not calibrated confidence.",
            weights_available=bool(global_capability and global_capability.weights_available),
            usable=hierarchy_available,
            execution_mode="in_process",
        )
        index_diagnostics = services.phase6c_reference_index
        index_enabled = config.phase6c_enabled and config.reference_index_enabled
        index_available = bool(
            index_enabled
            and index_diagnostics.status == "ready"
            and index_diagnostics.leakage_status == "passed"
        )
        index_reason = (
            None
            if index_available
            else "reference_index_leakage_not_passed"
            if index_enabled
            and index_diagnostics.status == "ready"
            and index_diagnostics.leakage_status != "passed"
            else index_diagnostics.reason_code
        )
        index_capability = ProviderCapability(
            provider_id="turkiye_megaloc_reference_index",
            enabled=index_enabled,
            available=index_available,
            execution_boundary="local",
            reason_code=index_reason,
            installed=index_diagnostics.status in {"ready", "empty"},
            verified=index_available,
            operational_status=(
                "ready" if index_available else "disabled" if not index_enabled else "unavailable"
            ),
            model_name="Licensed Türkiye MegaLoc reference index",
            model_revision=index_diagnostics.index_version,
            provider_type="reference_index",
            offline=True,
            limitation="Reference coverage is bounded and absence is neutral evidence.",
            usable=index_available,
            execution_mode="in_process",
        )
        megaloc_enabled = config.phase6c_enabled and config.megaloc_enabled
        megaloc_available = bool(megaloc_enabled and megaloc_status and megaloc_status.available)
        megaloc_operational = _public_megaloc_state(
            megaloc_status.state if megaloc_status is not None else None,
            enabled=megaloc_enabled,
        )
        megaloc_capability = ProviderCapability(
            provider_id="megaloc_retrieval",
            enabled=megaloc_enabled,
            available=megaloc_available,
            execution_boundary="local",
            reason_code=(
                None
                if megaloc_available
                else "disabled"
                if not megaloc_enabled
                else megaloc_status.reason_code
                if megaloc_status is not None
                else "provider_unavailable"
            ),
            installed=bool(megaloc_status and megaloc_status.load_verified),
            verified=bool(megaloc_status and megaloc_status.real_inference_verified),
            operational_status=megaloc_operational,
            model_name=megaloc_status.model_id if megaloc_status is not None else "MegaLoc",
            model_revision=(
                megaloc_status.model_revision
                if megaloc_status is not None
                else MEGALOC_MODEL_REVISION
            ),
            device=megaloc_status.device if megaloc_status is not None else config.megaloc_device,
            calibration_state="uncalibrated",
            provider_type="visual_retrieval",
            offline=True,
            license="MIT",
            limitation="Descriptor similarity is not calibrated confidence or geographic proof.",
            weights_available=bool(megaloc_status and megaloc_status.load_verified),
            usable=megaloc_available,
            execution_mode="isolated_worker",
            source_revision=MEGALOC_SOURCE_REVISION,
            load_error=(
                None
                if megaloc_available
                else megaloc_status.reason_code
                if megaloc_status is not None
                else "provider_unavailable"
            ),
        )

        segmentation_status = services.segmentation_provider.status()
        segmentation_enabled = config.phase6a_enabled and segmentation_status.enabled
        segmentation_capability = ProviderCapability(
            provider_id=segmentation_status.provider_id,
            enabled=segmentation_enabled,
            available=segmentation_enabled and segmentation_status.usable,
            execution_boundary="local",
            reason_code=(
                segmentation_status.load_error
                or (
                    None
                    if segmentation_enabled and segmentation_status.usable
                    else "disabled"
                    if not segmentation_enabled
                    else segmentation_status.status
                )
            ),
            installed=segmentation_status.installed,
            verified=segmentation_status.prepared,
            operational_status=(segmentation_status.status if segmentation_enabled else "disabled"),
            model_name=services.segmentation_provider.descriptor.model_name,
            model_revision=services.segmentation_provider.descriptor.version,
            device=segmentation_status.device,
            provider_type="scene_segmentation",
            offline=True,
            limitation="Scene pixel proportions only; this provider does not geolocate.",
            weights_available=segmentation_status.prepared,
            usable=segmentation_enabled and segmentation_status.usable,
            execution_mode="in_process",
            load_error=segmentation_status.load_error,
        )
        reverse_enabled = config.phase6a_enabled
        reverse_available = reverse_enabled and services.reverse_geocoder.available
        reverse_capability = ProviderCapability(
            provider_id=services.reverse_geocoder.provider_id,
            enabled=reverse_enabled,
            available=reverse_available,
            execution_boundary="local",
            reason_code=(
                None if reverse_available else "disabled" if not reverse_enabled else "unavailable"
            ),
            installed=gazetteer_info.status == "ready",
            verified=(
                gazetteer_info.status == "ready"
                and gazetteer_info.schema_version is not None
                and gazetteer_info.license is not None
            ),
            operational_status=(
                "ready"
                if reverse_available
                else "disabled"
                if not reverse_enabled
                else "unavailable"
            ),
            provider_type="reverse_geocoding",
            offline=True,
            license=gazetteer_info.license,
            limitation="Names candidate coordinates only; it never changes ranking.",
        )
        rapid_enabled = bool(
            services.rapidocr_provider is not None and _rapidocr_requested(config)
        )
        rapid_installed = bool(
            services.rapidocr_provider is not None
            and services.rapidocr_provider.descriptor.available
        )
        rapid_available = rapid_enabled and rapid_installed and services.rapidocr_runtime_verified
        rapid_capability = ProviderCapability(
            provider_id=(
                services.rapidocr_provider.descriptor.id
                if services.rapidocr_provider is not None
                else "rapidocr-ppocrv6-local"
            ),
            enabled=rapid_enabled,
            available=rapid_available,
            execution_boundary="local",
            reason_code=(
                None
                if rapid_available
                else "disabled"
                if not rapid_enabled
                else services.rapidocr_runtime_reason or "inference_not_verified"
            ),
            installed=rapid_installed,
            verified=services.rapidocr_runtime_verified,
            operational_status=(
                "ready"
                if rapid_available
                else "disabled"
                if not rapid_enabled
                else "not_installed"
                if not rapid_installed
                else "inference_not_verified"
            ),
            model_name=(
                services.rapidocr_provider.descriptor.model_name
                if services.rapidocr_provider is not None
                else "PP-OCRv6-small-onnx"
            ),
            model_revision="rapidocr-3.9.1",
            device=config.rapidocr_device,
            provider_type="ocr",
            offline=True,
            license="Apache-2.0",
            limitation="Fallback OCR evidence only; OCR text is not itself a location.",
            weights_available=rapid_installed,
            usable=rapid_available,
            execution_mode="isolated_worker",
            source_revision="7fe716f8e38bb9a43f2680159f38deb14d8b1930",
            load_error=None if rapid_available else services.rapidocr_runtime_reason,
        )

        return CapabilitiesResponse(
            supported_formats=["jpeg", "png", "webp"],
            max_upload_bytes=config.max_upload_bytes,
            max_decoded_pixels=config.max_decoded_pixels,
            enabled_analysis_modes=enabled_modes,
            providers={
                "exif": capability(services.exif_provider, True),
                "quality": capability(services.quality_provider, True),
                "ocr": capability(
                    services.ocr_provider,
                    _ocr_requested(config),
                ),
                "retrieval": capability(
                    services.retrieval_provider,
                    config.phase5b_enabled and config.retrieval_enabled,
                ),
                "visual_clues": segmentation_capability,
                "place_research": reverse_capability,
                "map_research": ProviderCapability(
                    provider_id="osm-overpass-map-evidence",
                    enabled=config.map_evidence_enabled,
                    available=selected_map_evaluator is not None,
                    execution_boundary="cloud",
                    reason_code=(
                        None
                        if selected_map_evaluator is not None
                        else "disabled"
                        if not config.map_evidence_enabled
                        else "unavailable"
                    ),
                    provider_type="map_research",
                    offline=False,
                    license="ODbL 1.0 / OpenStreetMap contributor attribution",
                    limitation="OSM coverage varies; absence is neutral evidence.",
                ),
                "cloud_vision": capability(services.vision_provider, legacy_cloud_available),
                "osv5m": _phase6b_geographic_capability(
                    services.osv5m_provider.status(),
                    enabled=config.phase6b_enabled and config.osv5m_enabled,
                    runtime=services.phase6b_worker_bindings["osv5m"],
                ),
                "plonk": _phase6b_geographic_capability(
                    services.plonk_provider.status(),
                    enabled=config.phase6b_enabled and config.plonk_enabled,
                    runtime=services.phase6b_worker_bindings["plonk"],
                ),
                "geoclip_hierarchical": hierarchy_capability,
                "megaloc": megaloc_capability,
                "reference_index": index_capability,
                "turkiye_reference_index": ProviderCapability(
                    provider_id=services.turkiye_reference_provider.provider_id,
                    enabled=turkiye_reference_status.enabled,
                    available=turkiye_reference_status.available,
                    execution_boundary="local",
                    reason_code=turkiye_reference_status.reason_code,
                    operational_status=turkiye_reference_status.state,
                    model_name=turkiye_reference_status.descriptor_provider,
                    model_revision=turkiye_reference_status.descriptor_version,
                    provider_type="reference_retrieval",
                    offline=True,
                    calibration_state="uncalibrated",
                    limitation=(
                        "Vector distance and rank only; never a confidence probability. "
                        "This seam is isolated from analysis fusion and ranking."
                    ),
                    usable=turkiye_reference_status.available,
                    execution_mode="in_process",
                ),
                "turkiye_mapillary_demo": ProviderCapability(
                    provider_id="mapillary-private-demo-v1",
                    enabled=mapillary_demo_status.enabled,
                    available=mapillary_demo_status.available,
                    execution_boundary="local",
                    reason_code=mapillary_demo_status.reason_code,
                    installed=mapillary_demo_status.index_checksum is not None,
                    verified=mapillary_demo_status.available,
                    operational_status=(
                        "ready"
                        if mapillary_demo_status.state == "active"
                        else mapillary_demo_status.state
                    ),
                    model_name="MegaLoc private Mapillary demo",
                    model_revision=mapillary_demo_status.model_version,
                    provider_type="reference_retrieval_demo",
                    offline=True,
                    license="Mapillary imagery / CC-BY-SA-4.0 attribution required",
                    calibration_state="uncalibrated",
                    limitation=(
                        "Private pilot-city technical demo only. Raw cosine similarity "
                        "is not confidence, probability, or geographic proof."
                    ),
                    usable=mapillary_demo_status.available,
                    execution_mode="isolated_worker",
                ),
                "paddleocr": ProviderCapability(
                    provider_id=services.paddleocr_provider.descriptor.id,
                    enabled=config.phase6b_enabled and config.paddleocr_enabled,
                    available=(
                        services.paddleocr_provider.descriptor.available
                        and services.phase6b_worker_bindings["paddleocr"].state == "ready"
                    ),
                    execution_boundary="local",
                    reason_code=services.phase6b_worker_bindings["paddleocr"].last_error,
                    installed=bool(
                        services.phase6b_worker_bindings["paddleocr"].health
                        and services.phase6b_worker_bindings["paddleocr"].health.import_ok
                    ),
                    verified=services.phase6b_worker_bindings["paddleocr"].state == "ready",
                    operational_status=services.phase6b_worker_bindings["paddleocr"].state,
                    model_name=services.paddleocr_provider.descriptor.model_name,
                    model_revision=services.paddleocr_provider.descriptor.version,
                    device="cpu",
                    provider_type="ocr",
                    offline=True,
                    weights_available=bool(
                        services.phase6b_worker_bindings["paddleocr"].health
                        and services.phase6b_worker_bindings["paddleocr"].health.weights_available
                    ),
                    usable=(
                        services.paddleocr_provider.descriptor.available
                        and services.phase6b_worker_bindings["paddleocr"].state == "ready"
                    ),
                    execution_mode="isolated_worker",
                    source_revision="211989f046cc1878460f9e65574690c00a127a1a",
                    load_error=services.phase6b_worker_bindings["paddleocr"].last_error,
                ),
                "rapidocr": rapid_capability,
                "openai_geo_review": ProviderCapability(
                    provider_id="openai-geo-review",
                    enabled=config.phase6b_enabled and config.openai_geo_enabled,
                    available=openai_geo_ready,
                    execution_boundary="cloud",
                    reason_code=(
                        None
                        if openai_geo_ready
                        else "provider_disabled"
                        if not config.openai_geo_enabled
                        else "missing_api_key"
                        if config.openai_key_value is None
                        else "usage_ledger_unavailable"
                    ),
                    installed=True,
                    verified=True,
                    operational_status=(
                        "ready"
                        if openai_geo_ready
                        else "disabled"
                        if not config.openai_geo_enabled
                        else "unavailable"
                    ),
                    model_name=config.openai_geo_model,
                    provider_type="cloud_reasoning",
                    offline=False,
                    usable=openai_geo_ready,
                    key_configured=(
                        openai_geo_status.key_configured
                        if openai_geo_status is not None
                        else config.openai_key_value is not None
                    ),
                    budget_available=(
                        openai_geo_status.budget_available
                        if openai_geo_status is not None
                        else False
                    ),
                ),
                **(
                    {"global_geolocation": global_capability}
                    if global_capability is not None
                    else {}
                ),
            },
            retention=RetentionPolicy(
                keep_uploads=config.keep_uploads,
                ttl_seconds=config.retention_ttl_seconds,
                originals_deleted_after_analysis=not config.keep_uploads,
            ),
            version=config.app_version,
        )

    @app.get("/api/v1/providers", response_model=ProviderStatusResponse)
    async def list_providers() -> ProviderStatusResponse:
        await asyncio.gather(
            *(binding.refresh() for binding in services.phase6b_worker_bindings.values())
        )
        megaloc_status = (
            await services.phase6c_extension.megaloc_status()
            if services.phase6c_extension is not None
            else None
        )
        providers: list[ProviderStatusItem] = []
        turkiye_reference_status = services.turkiye_reference_provider.status()
        providers.append(
            ProviderStatusItem(
                provider_id=services.turkiye_reference_provider.provider_id,
                provider_type="reference_retrieval",
                mode="shadow" if turkiye_reference_status.enabled else "disabled",
                available=turkiye_reference_status.available,
                status=turkiye_reference_status.state,
                classification="real",
                model_name=turkiye_reference_status.descriptor_provider,
                model_revision=turkiye_reference_status.descriptor_version,
                calibration_state="uncalibrated",
                reason_code=turkiye_reference_status.reason_code,
                usable=turkiye_reference_status.available,
                execution_mode="in_process",
            )
        )
        inference_ids: set[str] = set()
        if services.inference_coordinator is not None:
            for status in services.inference_coordinator.statuses():
                inference_ids.add(status.provider_id)
                providers.append(
                    ProviderStatusItem(
                        provider_id=status.provider_id,
                        provider_type=status.provider_type,
                        mode=status.mode,
                        available=status.available,
                        status=status.status,
                        classification=status.classification,
                        model_name=status.model_name,
                        model_revision=status.model_revision,
                        device=status.device,
                        calibration_state=status.calibration_state,
                        reason_code=status.reason_code,
                    )
                )

        def append_descriptor(provider: Any, *, enabled: bool) -> None:
            descriptor = provider.descriptor
            if descriptor.id in inference_ids:
                return
            providers.append(
                ProviderStatusItem(
                    provider_id=descriptor.id,
                    provider_type=descriptor.kind,
                    mode="primary" if enabled else "disabled",
                    available=descriptor.available and enabled,
                    status=(
                        "ready"
                        if descriptor.available and enabled
                        else "disabled"
                        if not enabled
                        else "unavailable"
                    ),
                    classification="real",
                    model_name=descriptor.model_name,
                    reason_code=(descriptor.unavailable_reason_code if enabled else "disabled"),
                )
            )

        append_descriptor(services.exif_provider, enabled=True)
        append_descriptor(services.quality_provider, enabled=True)
        append_descriptor(
            services.ocr_provider,
            enabled=_ocr_requested(config),
        )
        append_descriptor(
            services.retrieval_provider,
            enabled=config.phase5b_enabled and config.retrieval_enabled,
        )
        append_descriptor(
            services.vision_provider,
            enabled=services.vision_provider.descriptor.available,
        )
        segmentation_status = services.segmentation_provider.status()
        segmentation_enabled = config.phase6a_enabled and segmentation_status.enabled
        providers.append(
            ProviderStatusItem(
                provider_id=segmentation_status.provider_id,
                provider_type="scene_segmentation",
                mode="candidate" if segmentation_enabled else "disabled",
                available=segmentation_enabled and segmentation_status.usable,
                status=(segmentation_status.status if segmentation_enabled else "disabled"),
                classification="real",
                model_name=services.segmentation_provider.descriptor.model_name,
                model_revision=services.segmentation_provider.descriptor.version,
                device=segmentation_status.device,
                reason_code=(
                    segmentation_status.load_error
                    or (
                        None
                        if segmentation_enabled and segmentation_status.usable
                        else "disabled"
                        if not segmentation_enabled
                        else segmentation_status.status
                    )
                ),
            )
        )
        reverse_enabled = config.phase6a_enabled
        reverse_available = reverse_enabled and services.reverse_geocoder.available
        providers.append(
            ProviderStatusItem(
                provider_id=services.reverse_geocoder.provider_id,
                provider_type="reverse_geocoding",
                mode="candidate" if reverse_enabled else "disabled",
                available=reverse_available,
                status=(
                    "ready"
                    if reverse_available
                    else "disabled"
                    if not reverse_enabled
                    else "unavailable"
                ),
                classification="real",
                model_name="Local GeoNames coordinate naming",
                reason_code=(
                    None
                    if reverse_available
                    else "disabled"
                    if not reverse_enabled
                    else "unavailable"
                ),
            )
        )
        hierarchy_enabled = bool(
            config.phase6c_enabled
            and config.geoclip_hierarchical_enabled
            and config.geoclip_global_grid_enabled
        )
        global_status = (
            services.global_provider.status() if services.global_provider is not None else None
        )
        hierarchy_available = bool(
            hierarchy_enabled
            and services.phase6c_extension is not None
            and services.phase6c_extension.hierarchical_available
            and global_status is not None
            and global_status.status == "ready"
        )
        providers.append(
            ProviderStatusItem(
                provider_id="geoclip_hierarchical_search",
                provider_type="global_geolocation",
                mode="candidate" if hierarchy_enabled else "disabled",
                available=hierarchy_available,
                status=(
                    "ready"
                    if hierarchy_available
                    else "disabled"
                    if not hierarchy_enabled
                    else "unavailable"
                ),
                classification="real",
                model_name="GeoCLIP hierarchical coordinate search",
                model_revision="geoclip-hierarchical-v1",
                device=global_status.device if global_status is not None else None,
                calibration_state="uncalibrated",
                reason_code=(
                    None
                    if hierarchy_available
                    else "disabled"
                    if not hierarchy_enabled
                    else "geoclip_hierarchical_unavailable"
                ),
                weights_available=bool(global_status and global_status.installed),
                usable=hierarchy_available,
                execution_mode="in_process",
            )
        )
        index_status = services.phase6c_reference_index
        index_enabled = config.phase6c_enabled and config.reference_index_enabled
        index_available = bool(
            index_enabled
            and index_status.status == "ready"
            and index_status.leakage_status == "passed"
        )
        index_reason = (
            None
            if index_available
            else "reference_index_leakage_not_passed"
            if index_enabled
            and index_status.status == "ready"
            and index_status.leakage_status != "passed"
            else index_status.reason_code
        )
        providers.append(
            ProviderStatusItem(
                provider_id="turkiye_megaloc_reference_index",
                provider_type="reference_index",
                mode="candidate" if index_enabled else "disabled",
                available=index_available,
                status=(
                    "ready"
                    if index_available
                    else "disabled"
                    if not index_enabled
                    else "unavailable"
                ),
                classification="real",
                model_name=(
                    f"Türkiye MegaLoc reference index ({index_status.count} references)"
                )[:120],
                model_revision=(
                    index_status.index_version[:120]
                    if index_status.index_version is not None
                    else None
                ),
                reason_code=index_reason,
                usable=index_available,
                execution_mode="in_process",
            )
        )
        megaloc_enabled = config.phase6c_enabled and config.megaloc_enabled
        megaloc_available = bool(megaloc_enabled and megaloc_status and megaloc_status.available)
        providers.append(
            ProviderStatusItem(
                provider_id="megaloc_retrieval",
                provider_type="visual_retrieval",
                mode="candidate" if megaloc_enabled else "disabled",
                available=megaloc_available,
                status=_public_megaloc_state(
                    megaloc_status.state if megaloc_status is not None else None,
                    enabled=megaloc_enabled,
                ),
                classification="real",
                model_name=megaloc_status.model_id if megaloc_status is not None else "MegaLoc",
                model_revision=(
                    megaloc_status.model_revision
                    if megaloc_status is not None
                    else MEGALOC_MODEL_REVISION
                ),
                device=(
                    megaloc_status.device if megaloc_status is not None else config.megaloc_device
                ),
                calibration_state="uncalibrated",
                reason_code=(
                    None
                    if megaloc_available
                    else "disabled"
                    if not megaloc_enabled
                    else megaloc_status.reason_code
                    if megaloc_status is not None
                    else "provider_unavailable"
                ),
                weights_available=bool(megaloc_status and megaloc_status.load_verified),
                usable=megaloc_available,
                execution_mode="isolated_worker",
                source_revision=MEGALOC_SOURCE_REVISION,
                load_error=(
                    None
                    if megaloc_available
                    else megaloc_status.reason_code
                    if megaloc_status is not None
                    else "provider_unavailable"
                ),
            )
        )
        for phase6b_status, enabled, binding_name in (
            (
                services.osv5m_provider.status(),
                config.phase6b_enabled and config.osv5m_enabled,
                "osv5m",
            ),
            (
                services.plonk_provider.status(),
                config.phase6b_enabled and config.plonk_enabled,
                "plonk",
            ),
        ):
            runtime = services.phase6b_worker_bindings[binding_name]
            available = enabled and runtime.state == "ready" and phase6b_status.available
            providers.append(
                ProviderStatusItem(
                    provider_id=phase6b_status.provider,
                    provider_type="global_geolocation",
                    mode="candidate" if enabled else "disabled",
                    available=available,
                    status=runtime.state if enabled else "disabled",
                    classification="real",
                    model_name=phase6b_status.model_id,
                    model_revision=phase6b_status.model_revision,
                    device=phase6b_status.device,
                    calibration_state="uncalibrated",
                    reason_code=None if available else runtime.last_error,
                    weights_available=bool(runtime.health and runtime.health.weights_available),
                    usable=available,
                    execution_mode="isolated_worker",
                    source_revision=phase6b_status.source_revision,
                    load_error=None if available else runtime.last_error,
                )
            )
        paddle_enabled = config.phase6b_enabled and config.paddleocr_enabled
        paddle_runtime = services.phase6b_worker_bindings["paddleocr"]
        paddle_available = (
            paddle_enabled
            and paddle_runtime.state == "ready"
            and services.paddleocr_provider.descriptor.available
        )
        providers.append(
            ProviderStatusItem(
                provider_id=services.paddleocr_provider.descriptor.id,
                provider_type="ocr",
                mode="candidate" if paddle_enabled else "disabled",
                available=paddle_available,
                status=paddle_runtime.state if paddle_enabled else "disabled",
                classification="real",
                model_name=services.paddleocr_provider.descriptor.model_name,
                model_revision=services.paddleocr_provider.descriptor.version,
                device="cpu",
                reason_code=None if paddle_available else paddle_runtime.last_error,
                weights_available=bool(
                    paddle_runtime.health and paddle_runtime.health.weights_available
                ),
                usable=paddle_available,
                execution_mode="isolated_worker",
                source_revision="211989f046cc1878460f9e65574690c00a127a1a",
                load_error=None if paddle_available else paddle_runtime.last_error,
            )
        )
        rapid_enabled = bool(
            services.rapidocr_provider is not None and _rapidocr_requested(config)
        )
        rapid_installed = bool(
            services.rapidocr_provider is not None
            and services.rapidocr_provider.descriptor.available
        )
        rapid_available = rapid_enabled and rapid_installed and services.rapidocr_runtime_verified
        providers.append(
            ProviderStatusItem(
                provider_id=(
                    services.rapidocr_provider.descriptor.id
                    if services.rapidocr_provider is not None
                    else "rapidocr-ppocrv6-local"
                ),
                provider_type="ocr",
                mode="candidate" if rapid_enabled else "disabled",
                available=rapid_available,
                status=(
                    "ready"
                    if rapid_available
                    else "disabled"
                    if not rapid_enabled
                    else "not_installed"
                    if not rapid_installed
                    else "inference_not_verified"
                ),
                classification="real",
                model_name=(
                    services.rapidocr_provider.descriptor.model_name
                    if services.rapidocr_provider is not None
                    else "PP-OCRv6-small-onnx"
                ),
                model_revision="rapidocr-3.9.1",
                device=config.rapidocr_device,
                reason_code=(
                    None
                    if rapid_available
                    else "disabled"
                    if not rapid_enabled
                    else services.rapidocr_runtime_reason or "inference_not_verified"
                ),
                weights_available=rapid_installed,
                usable=rapid_available,
                execution_mode="isolated_worker",
                source_revision="7fe716f8e38bb9a43f2680159f38deb14d8b1930",
                load_error=None if rapid_available else services.rapidocr_runtime_reason,
            )
        )
        openai_status = (
            services.openai_geo_provider.capability()
            if services.openai_geo_provider is not None
            else None
        )
        openai_enabled = config.phase6b_enabled and config.openai_geo_enabled
        openai_ready = bool(
            openai_status is not None and openai_status.key_configured and openai_status.enabled
        )
        providers.append(
            ProviderStatusItem(
                provider_id="openai-geo-review",
                provider_type="cloud_reasoning",
                mode="candidate" if openai_enabled else "disabled",
                available=openai_ready,
                status=(
                    "ready" if openai_ready else "disabled" if not openai_enabled else "unavailable"
                ),
                classification="real",
                model_name=config.openai_geo_model,
                reason_code=(
                    None
                    if openai_ready
                    else "disabled"
                    if not openai_enabled
                    else "missing_api_key"
                    if config.openai_key_value is None
                    else "usage_ledger_unavailable"
                ),
                key_configured=(
                    openai_status.key_configured
                    if openai_status is not None
                    else config.openai_key_value is not None
                ),
                budget_available=(
                    openai_status.budget_available if openai_status is not None else False
                ),
                usable=openai_ready,
            )
        )
        return ProviderStatusResponse(
            providers=sorted(providers, key=lambda item: item.provider_id)
        )

    def require_operator_api() -> None:
        if not config.operator_api_enabled or config.app_env == "production":
            raise AppError(
                404,
                "operator_api_unavailable",
                "error.operator_api_unavailable",
                "Not found",
            )

    @app.get(
        "/api/v1/system-intelligence",
        response_model=SystemIntelligenceResponse,
    )
    async def system_intelligence() -> SystemIntelligenceResponse:
        require_operator_api()
        return await _system_intelligence_snapshot(services)

    @app.get("/api/v1/models", response_model=ModelStatusResponse)
    async def list_models() -> ModelStatusResponse:
        require_operator_api()
        models: list[ModelStatusItem] = []
        if services.global_provider is not None:
            status = services.global_provider.status()
            models.append(
                ModelStatusItem(
                    model_id="geoclip",
                    model_version=status.model_revision,
                    mode="primary",
                    verified=status.verified,
                    status=("verified" if status.verified else "disabled"),
                    reason_code=(
                        None if status.status == "ready" else status.reason_code or status.status
                    ),
                )
            )
        registered = {item.model_id: item for item in services.trained_artifact_manager.list()}
        registered.setdefault(
            config.custom_model_id,
            services.trained_artifact_manager.info(config.custom_model_id),
        )
        for info in sorted(registered.values(), key=lambda item: item.model_id):
            models.append(
                ModelStatusItem(
                    model_id=info.model_id,
                    model_version=info.model_version,
                    artifact_digest=info.artifact_identity,
                    mode=info.mode.value,
                    verified=info.verified,
                    status=info.status,
                    reason_code=info.reason_code,
                )
            )
        return ModelStatusResponse(models=models)

    @app.get("/api/v1/analyses", response_model=AnalysisHistoryPage)
    async def list_analyses(
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
        status: AnalysisStatus | None = None,
        classification: Literal["real", "simulated"] | None = None,
        provider: Annotated[str | None, Query(max_length=80)] = None,
        search: Annotated[str | None, Query(max_length=120)] = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> AnalysisHistoryPage:
        if created_from is not None and created_to is not None and created_from > created_to:
            raise bad_request("invalid_history_range", "error.invalid_history_range")
        page, total = await analysis_repository.list_history(
            filters=AnalysisHistoryFilters(
                status=status.value if status is not None else None,
                classification=classification,
                provider=provider.strip() if provider else None,
                search=search.strip() if search else None,
                created_from=created_from,
                created_to=created_to,
            ),
            limit=limit,
            offset=offset,
        )
        return AnalysisHistoryPage(
            items=[history_item(item) for item in page],
            total=total,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/evaluations", response_model=EvaluationReportList)
    async def list_evaluations() -> EvaluationReportList:
        require_operator_api()
        if not config.evaluation_report_dir.is_dir():
            return EvaluationReportList(reports=[])
        try:
            return EvaluationReportCatalog(config.evaluation_report_dir).list_reports()
        except EvaluationCatalogError as exc:
            raise AppError(
                404,
                "evaluation_catalog_unavailable",
                "error.evaluation_catalog_unavailable",
                "Not found",
            ) from exc

    @app.get("/api/v1/evaluations/{report_id}", response_model=EvaluationReportSummary)
    async def get_evaluation(report_id: str) -> EvaluationReportSummary:
        require_operator_api()
        try:
            return EvaluationReportCatalog(config.evaluation_report_dir).get_report(report_id)
        except EvaluationCatalogError as exc:
            raise AppError(
                404,
                "evaluation_report_unavailable",
                "error.evaluation_report_unavailable",
                "Not found",
            ) from exc

    @app.get("/api/v1/datasets/qa", response_model=DatasetQAReportList)
    async def list_dataset_qa_reports() -> DatasetQAReportList:
        require_operator_api()
        if not config.dataset_qa_report_dir.is_dir():
            return DatasetQAReportList(reports=[])
        try:
            return DatasetQAReportCatalog(config.dataset_qa_report_dir).list_reports()
        except DatasetQAError as exc:
            raise AppError(
                404,
                "dataset_qa_catalog_unavailable",
                "error.dataset_qa_catalog_unavailable",
                "Not found",
            ) from exc

    @app.get("/api/v1/datasets/qa/{report_id}", response_model=DatasetQAReport)
    async def get_dataset_qa_report(report_id: str) -> DatasetQAReport:
        require_operator_api()
        try:
            return DatasetQAReportCatalog(config.dataset_qa_report_dir).get_report(report_id)
        except DatasetQAError as exc:
            raise AppError(
                404,
                "dataset_qa_report_unavailable",
                "error.dataset_qa_report_unavailable",
                "Not found",
            ) from exc

    @app.post("/api/v1/analyses", response_model=AnalysisAccepted, status_code=202)
    async def create_analysis(
        request: Request,
        response: Response,
        image: Annotated[UploadFile, File()],
        analysis_mode: Annotated[AnalysisMode, Form()],
        cloud_processing_consent: Annotated[bool, Form()],
        authorization_acknowledged: Annotated[bool, Form()],
        allow_cloud_assist: Annotated[bool, Form()] = False,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> AnalysisAccepted:
        client = _client_key(request)
        allowed, retry_after = await services.rate_limiter.check(
            f"analysis:{client}", config.analysis_rate_limit_per_minute
        )
        if not allowed:
            raise AppError(
                429,
                "analysis_rate_limited",
                "error.rate_limited",
                "Too many analyses",
                retry_after_seconds=retry_after,
            )
        if not authorization_acknowledged:
            raise bad_request("authorization_required", "error.authorization_required")
        if allow_cloud_assist and analysis_mode != AnalysisMode.CLOUD_ASSISTED:
            raise bad_request("cloud_mode_required", "error.cloud_mode_required")
        if analysis_mode == AnalysisMode.CLOUD_ASSISTED:
            if not cloud_processing_consent:
                raise bad_request("cloud_consent_required", "error.cloud_consent_required")
            if not allow_cloud_assist and not services.vision_provider.descriptor.available:
                raise AppError(
                    422,
                    "cloud_provider_unavailable",
                    "error.cloud_provider_unavailable",
                    "Cloud provider unavailable",
                )
            cloud_allowed, cloud_retry = await services.rate_limiter.check(
                f"cloud:{client}", config.cloud_rate_limit_per_minute
            )
            if not cloud_allowed:
                raise AppError(
                    429,
                    "cloud_rate_limited",
                    "error.cloud_rate_limited",
                    "Too many cloud analyses",
                    retry_after_seconds=cloud_retry,
                )

        idempotency_hash: str | None = None
        if idempotency_key is not None:
            if not _IDEMPOTENCY_KEY.fullmatch(idempotency_key):
                raise bad_request("invalid_idempotency_key", "error.invalid_idempotency_key")
            if services.phase6c_extension is None:
                idempotency_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
                existing = await analysis_repository.get_by_idempotency(idempotency_hash)
                if existing is not None:
                    accepted = _accepted(existing.analysis.id)
                    response.headers["Location"] = accepted.status_url
                    await image.close()
                    return accepted

        stored_upload = None
        prepared: PreparedImage | None = None
        try:
            stored_upload = await storage.save_upload(image, config.max_upload_bytes)
            prepared = await image_processor.prepare(
                stored_upload,
                content_type=image.content_type,
                original_filename=image.filename,
            )
        except BaseException:
            if stored_upload is not None:
                await storage.delete(stored_upload.handle.key)
            raise
        finally:
            await image.close()

        if idempotency_key is not None and services.phase6c_extension is not None:
            cache_fingerprint = services.phase6c_extension.cache_fingerprint(
                prepared.summary.sha256
            )
            idempotency_hash = hashlib.sha256(
                (
                    f"{idempotency_key}:{cache_fingerprint}:{analysis_mode.value}:"
                    f"{cloud_processing_consent}:{allow_cloud_assist}"
                ).encode()
            ).hexdigest()
            existing = await analysis_repository.get_by_idempotency(idempotency_hash)
            if existing is not None:
                await storage.delete(prepared.normalized.key)
                await storage.delete(prepared.original.key)
                accepted = _accepted(existing.analysis.id)
                response.headers["Location"] = accepted.status_url
                return accepted

        analysis_id = uuid4()
        now = datetime.now(UTC)
        queued = Analysis(
            id=analysis_id,
            status=AnalysisStatus.QUEUED,
            analysis_mode=analysis_mode,
            created_at=now,
            expires_at=now + timedelta(seconds=config.retention_ttl_seconds),
            progress=Progress(stage="queued", percent=0, message_key="progress.queued"),
            image=prepared.summary,
            quality=None,
            evidence=[],
            candidates=[],
            abstention=None,
            warnings=[],
            timings_ms={},
            fusion_policy_version=FUSION_POLICY_VERSION,
            pipeline_version=(
                config.phase6c_pipeline_version
                if services.phase6c_extension is not None
                else "legacy-v1"
            ),
            failure=None,
        )
        try:
            await analysis_repository.create(
                queued,
                storage_key=prepared.original.key,
                idempotency_hash=idempotency_hash,
            )
        except DuplicateIdempotencyError as exc:
            await storage.delete(prepared.normalized.key)
            await storage.delete(prepared.original.key)
            if idempotency_hash is None:
                raise
            existing = await analysis_repository.get_by_idempotency(idempotency_hash)
            if existing is None:
                raise AppError(
                    500, "idempotency_conflict", "error.internal", "Internal error"
                ) from exc
            accepted = _accepted(existing.analysis.id)
            response.headers["Location"] = accepted.status_url
            return accepted

        await broker.publish(analysis_id, "progress", AnalysisStatus.QUEUED, queued.progress)
        services.active_artifacts[analysis_id] = (
            prepared.original.key,
            prepared.normalized.key,
        )
        job = AnalysisJob(
            id=analysis_id,
            request_id=_request_id(request),
            cloud_consent=cloud_processing_consent,
            prepared=prepared,
            allow_cloud_assist=allow_cloud_assist,
        )

        async def run_job(cancellation: asyncio.Event) -> None:
            try:
                await pipeline.run(job, cancellation)
            finally:
                services.active_artifacts.pop(analysis_id, None)

        try:
            await queue.submit(analysis_id, run_job)
        except JobQueueFullError as exc:
            services.active_artifacts.pop(analysis_id, None)
            await analysis_repository.delete(analysis_id)
            await storage.delete(prepared.normalized.key)
            await storage.delete(prepared.original.key)
            raise AppError(
                429,
                "job_queue_full",
                "error.job_queue_full",
                "Analysis queue is full",
                retry_after_seconds=5,
            ) from exc

        accepted = _accepted(analysis_id)
        response.headers["Location"] = accepted.status_url
        return accepted

    @app.get("/api/v1/analyses/{analysis_id}", response_model=Analysis)
    async def get_analysis(analysis_id: str) -> Analysis:
        parsed = _parse_uuid(analysis_id)
        stored = await analysis_repository.get(parsed)
        if stored is None:
            raise AppError(404, "analysis_not_found", "error.analysis_not_found", "Not found")
        return stored.analysis

    @app.post(
        "/api/v1/analyses/{analysis_id}/rerun",
        response_model=AnalysisAccepted,
        status_code=202,
    )
    async def rerun_analysis(
        analysis_id: str, request: Request, response: Response
    ) -> AnalysisAccepted:
        parsed = _parse_uuid(analysis_id)
        stored = await analysis_repository.get(parsed)
        if stored is None:
            raise AppError(404, "analysis_not_found", "error.analysis_not_found", "Not found")
        if stored.analysis.analysis_mode != AnalysisMode.LOCAL_ONLY:
            raise AppError(
                409,
                "rerun_requires_reupload_and_consent",
                "error.rerun_requires_reupload_and_consent",
                "Rerun requires re-upload",
            )
        if stored.storage_key is None or stored.analysis.image is None:
            raise AppError(
                409,
                "rerun_source_not_retained",
                "error.rerun_source_not_retained",
                "Rerun source unavailable",
            )
        image_format = stored.analysis.image.format
        extension = "jpg" if image_format == "jpeg" else image_format
        content_type = "image/jpeg" if image_format == "jpeg" else f"image/{image_format}"
        cloned = None
        prepared = None
        try:
            cloned = await storage.clone_upload(stored.storage_key, config.max_upload_bytes)
            prepared = await image_processor.prepare(
                cloned,
                content_type=content_type,
                original_filename=f"retained.{extension}",
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            if cloned is not None:
                await storage.delete(cloned.handle.key)
            raise AppError(
                409,
                "rerun_source_not_retained",
                "error.rerun_source_not_retained",
                "Rerun source unavailable",
            ) from exc
        new_id = uuid4()
        now = datetime.now(UTC)
        queued = Analysis(
            id=new_id,
            status=AnalysisStatus.QUEUED,
            analysis_mode=AnalysisMode.LOCAL_ONLY,
            created_at=now,
            expires_at=now + timedelta(seconds=config.retention_ttl_seconds),
            progress=Progress(stage="queued", percent=0, message_key="progress.queued"),
            image=prepared.summary,
            quality=None,
            evidence=[],
            candidates=[],
            warnings=[],
            timings_ms={},
            fusion_policy_version=FUSION_POLICY_VERSION,
            pipeline_version=(
                config.phase6c_pipeline_version
                if services.phase6c_extension is not None
                else "legacy-v1"
            ),
        )
        await analysis_repository.create(
            queued,
            storage_key=prepared.original.key,
            idempotency_hash=None,
        )
        await broker.publish(new_id, "progress", AnalysisStatus.QUEUED, queued.progress)
        services.active_artifacts[new_id] = (
            prepared.original.key,
            prepared.normalized.key,
        )
        job = AnalysisJob(
            id=new_id,
            request_id=_request_id(request),
            cloud_consent=False,
            prepared=prepared,
        )

        async def run_rerun(cancellation: asyncio.Event) -> None:
            try:
                await pipeline.run(job, cancellation)
            finally:
                services.active_artifacts.pop(new_id, None)

        try:
            await queue.submit(new_id, run_rerun)
        except JobQueueFullError as exc:
            services.active_artifacts.pop(new_id, None)
            await analysis_repository.delete(new_id)
            await storage.delete(prepared.normalized.key)
            await storage.delete(prepared.original.key)
            raise AppError(
                429,
                "job_queue_full",
                "error.job_queue_full",
                "Analysis queue is full",
                retry_after_seconds=5,
            ) from exc
        accepted = _accepted(new_id)
        response.headers["Location"] = accepted.status_url
        return accepted

    @app.delete("/api/v1/analyses/{analysis_id}", response_model=DeleteResponse)
    async def delete_analysis(analysis_id: str) -> DeleteResponse:
        parsed = _parse_uuid(analysis_id, delete=True)
        await queue.cancel(parsed)
        stored = await analysis_repository.delete(parsed)
        artifacts = services.active_artifacts.pop(parsed, ())
        for key in artifacts:
            await storage.delete(key)
        if stored is not None:
            await storage.delete(stored.storage_key)
            progress = Progress(stage="deleted", percent=100, message_key="progress.deleted")
            await broker.publish(parsed, "deleted", AnalysisStatus.DELETED, progress)
        return DeleteResponse(id=parsed)

    @app.get("/api/v1/analyses/{analysis_id}/events")
    async def analysis_events(request: Request, analysis_id: str) -> StreamingResponse:
        parsed = _parse_uuid(analysis_id)
        stored = await analysis_repository.get(parsed)
        if stored is None:
            raise AppError(404, "analysis_not_found", "error.analysis_not_found", "Not found")
        client = _client_key(request)
        acquired = await services.connection_limiter.acquire(
            client, config.max_sse_connections_per_client
        )
        if not acquired:
            raise AppError(
                429,
                "sse_connection_limit",
                "error.sse_connection_limit",
                "Too many event streams",
                retry_after_seconds=5,
            )
        history, event_queue = await broker.subscribe(parsed)
        terminal_status = stored.analysis.status
        if terminal_status in {
            AnalysisStatus.COMPLETED,
            AnalysisStatus.FAILED,
            AnalysisStatus.DELETED,
        } and not any(event.event_type in TERMINAL_EVENT_TYPES for event in history):
            event_name = terminal_status.value
            await broker.publish(parsed, event_name, terminal_status, stored.analysis.progress)
            history, _ = await broker.subscribe(parsed)
            await broker.unsubscribe(parsed, _)

        async def stream() -> AsyncIterator[str]:
            started = time.monotonic()
            try:
                for event in history:
                    yield _encode_sse(event)
                    if event.event_type in TERMINAL_EVENT_TYPES:
                        return
                while time.monotonic() - started < config.max_sse_lifetime_seconds:
                    if await request.is_disconnected():
                        return
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(), timeout=config.sse_heartbeat_seconds
                        )
                    except TimeoutError:
                        current = await analysis_repository.get(parsed)
                        if current is None:
                            deleted_event = AnalysisEvent(
                                event_id=f"deleted-{uuid4().hex[:8]}",
                                event_type="deleted",
                                analysis_id=parsed,
                                occurred_at=datetime.now(UTC),
                                status=AnalysisStatus.DELETED,
                                progress=Progress(
                                    stage="deleted",
                                    percent=100,
                                    message_key="progress.deleted",
                                ),
                            )
                            yield _encode_sse(deleted_event)
                            return
                        yield _encode_sse(broker.heartbeat(parsed, current.analysis.status))
                        continue
                    yield _encode_sse(event)
                    if event.event_type in TERMINAL_EVENT_TYPES:
                        return
            finally:
                await broker.unsubscribe(parsed, event_queue)
                await services.connection_limiter.release(client)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store, no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    app.include_router(
        build_mapillary_demo_router(
            runtime=services.mapillary_demo_provider,
            storage=storage,
            image_processor=image_processor,
            max_upload_bytes=config.max_upload_bytes,
        )
    )
    app.include_router(case_router)
    return app


def _encode_sse(event: AnalysisEvent) -> str:
    return f"event: {event.event_type}\nid: {event.event_id}\ndata: {event.model_dump_json()}\n\n"


app = create_app()
