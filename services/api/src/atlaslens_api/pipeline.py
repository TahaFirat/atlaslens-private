from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from atlaslens_api.constraints.models import MapClue
from atlaslens_api.constraints.phase5b import BoundedMapEvidenceEvaluator
from atlaslens_api.events import AnalysisEventBroker
from atlaslens_api.fusion import (
    CandidateBatch,
    DeterministicCandidateFusionService,
    ExifCandidateProvider,
    VisionCandidateProvider,
    provenance_for,
)
from atlaslens_api.gazetteer import GazetteerResolver
from atlaslens_api.global_prediction import (
    GlobalPredictionCandidateProvider,
    GlobalPredictionSynthesis,
)
from atlaslens_api.global_prediction.clustering import GeoClipCandidateClusterer
from atlaslens_api.global_prediction.reverse_geocoding import (
    CachedClusterReverseGeocoder,
    ReverseGeocodedCluster,
)
from atlaslens_api.image_processing import PreparedImage, SafeImageProcessor
from atlaslens_api.inference import (
    CoordinatedInference,
    InferenceCancelledError,
    InferenceRequest,
    InferenceResult,
    ProviderEnsembleCoordinator,
    descriptor_for_inference,
    legacy_outcome_for_inference,
)
from atlaslens_api.phase5b import Phase5BEvidenceReranker
from atlaslens_api.phase5b.integration import Phase5BSourceAdapter
from atlaslens_api.phase5b.models import EvidenceHypothesis
from atlaslens_api.phase5b.retrieval_integration import Phase5BRetrievalAdapter
from atlaslens_api.phase6a import Phase6AHybridEvidenceEngine
from atlaslens_api.phase6b.integration import Phase6BPipelineExtension, Phase6BPipelineResult
from atlaslens_api.phase6c.integration import Phase6CPipelineExtension
from atlaslens_api.providers.base import (
    ExifProvider,
    GlobalGeolocationProvider,
    GlobalPredictionResult,
    ImageQualityProvider,
    InvocationContext,
    OCRProvider,
    OCRResult,
    OutcomeStatus,
    ProviderDescriptor,
    ProviderOutcome,
    RetrievalProvider,
    VisionClueProvider,
)
from atlaslens_api.providers.retrieval import validated_retrieval_hits
from atlaslens_api.repository import AnalysisNotFoundError, AnalysisRepository
from atlaslens_api.retrieval.models import RetrievalHit
from atlaslens_api.schemas import (
    Abstention,
    Analysis,
    AnalysisMode,
    AnalysisStatus,
    Evidence,
    FailureSummary,
    Phase5BDiagnostics,
    Phase6BCloudAssistSummary,
    Phase6BFusionSummary,
    Phase6BOCRSummary,
    Phase6BProviderPredictionSummary,
    Phase6CAnalysisSummary,
    Progress,
    ProviderComparisonSummary,
    ProviderRunDiagnostic,
    SceneClassSummary,
    SceneGroupSummary,
    SceneSegmentationSummary,
    SceneTagSummary,
    SimulationSummary,
)
from atlaslens_api.segmentation import SceneSegmentationProvider, SegmentationResult
from atlaslens_api.storage import StorageBackend


class JobCancelledError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AnalysisJob:
    id: UUID
    request_id: str
    cloud_consent: bool
    prepared: PreparedImage
    allow_cloud_assist: bool = False

    def __repr__(self) -> str:
        return "AnalysisJob(id=<redacted>, mode_handle=<redacted>)"


@dataclass(frozen=True, slots=True)
class _TimedInvocation[T]:
    value: T | None
    duration_ms: int
    timed_out: bool = False


class AnalysisPipeline:
    def __init__(
        self,
        *,
        repository: AnalysisRepository,
        storage: StorageBackend,
        image_processor: SafeImageProcessor,
        broker: AnalysisEventBroker,
        exif_provider: ExifProvider,
        quality_provider: ImageQualityProvider,
        ocr_provider: OCRProvider,
        vision_provider: VisionClueProvider,
        fusion: DeterministicCandidateFusionService,
        keep_uploads: bool,
        global_provider: GlobalGeolocationProvider | None = None,
        inference_coordinator: ProviderEnsembleCoordinator | None = None,
        max_inference_input_bytes: int = 20 * 1024 * 1024,
        inference_top_k: int = 5,
        gazetteer: GazetteerResolver | None = None,
        global_prediction_timeout_seconds: float = 30.0,
        ocr_timeout_seconds: float = 12.0,
        vision_timeout_seconds: float = 40.0,
        retrieval_provider: RetrievalProvider | None = None,
        retrieval_timeout_seconds: float = 20.0,
        phase5b_reranker: Phase5BEvidenceReranker | None = None,
        map_evaluator: BoundedMapEvidenceEvaluator | None = None,
        phase6a_reranker: Phase6AHybridEvidenceEngine | None = None,
        geoclip_clusterer: GeoClipCandidateClusterer | None = None,
        reverse_geocoder: CachedClusterReverseGeocoder | None = None,
        segmentation_provider: SceneSegmentationProvider | None = None,
        segmentation_timeout_seconds: float = 90.0,
        phase6b_extension: Phase6BPipelineExtension | None = None,
        phase6c_extension: Phase6CPipelineExtension | None = None,
    ) -> None:
        if (
            global_prediction_timeout_seconds <= 0
            or ocr_timeout_seconds <= 0
            or vision_timeout_seconds <= 0
            or retrieval_timeout_seconds <= 0
            or segmentation_timeout_seconds <= 0
        ):
            raise ValueError("provider timeouts must be positive")
        if not 1 <= inference_top_k <= 100:
            raise ValueError("inference Top-K must be between 1 and 100")
        if max_inference_input_bytes <= 0:
            raise ValueError("max inference input bytes must be positive")
        self._repository = repository
        self._storage = storage
        self._image_processor = image_processor
        self._broker = broker
        self._exif = exif_provider
        self._quality = quality_provider
        self._ocr = ocr_provider
        self._vision = vision_provider
        self._fusion = fusion
        self._exif_candidates = ExifCandidateProvider()
        self._vision_candidates = VisionCandidateProvider()
        self._global = global_provider
        self._inference = inference_coordinator
        self._max_inference_input_bytes = max_inference_input_bytes
        self._inference_top_k = inference_top_k
        self._global_timeout_seconds = global_prediction_timeout_seconds
        self._ocr_timeout_seconds = ocr_timeout_seconds
        self._vision_timeout_seconds = vision_timeout_seconds
        self._global_candidates = GlobalPredictionCandidateProvider(gazetteer=gazetteer)
        self._retrieval = retrieval_provider
        self._retrieval_timeout_seconds = retrieval_timeout_seconds
        self._phase5b = phase5b_reranker
        self._phase5b_sources = Phase5BSourceAdapter()
        self._phase5b_retrieval = Phase5BRetrievalAdapter()
        self._map_evaluator = map_evaluator
        self._phase6a = phase6a_reranker
        self._geoclip_clusterer = geoclip_clusterer
        self._reverse_geocoder = reverse_geocoder
        self._segmentation = segmentation_provider
        self._segmentation_timeout_seconds = segmentation_timeout_seconds
        self._phase6b = phase6b_extension
        self._phase6c = phase6c_extension
        self._keep_uploads = keep_uploads

    async def run(self, job: AnalysisJob, cancellation: asyncio.Event) -> None:
        completed = False
        artifacts_cleaned = False
        fanout_tasks: list[asyncio.Task[Any]] = []
        analysis: Analysis | None = None
        started = time.monotonic()
        try:
            stored = await self._repository.get(job.id)
            if stored is None:
                return
            analysis = stored.analysis.model_copy(
                update={
                    "status": AnalysisStatus.PROCESSING,
                    "image": job.prepared.summary,
                }
            )
            analysis = await self._advance(analysis, "preprocessing", 15)
            self._check_cancelled(cancellation)

            context = InvocationContext(
                analysis_id=job.id,
                request_id=job.request_id,
                mode=analysis.analysis_mode,
                cloud_consent=job.cloud_consent,
                deadline=datetime.now(UTC)
                + timedelta(
                    seconds=max(
                        45.0,
                        self._ocr_timeout_seconds + 10.0,
                        self._vision_timeout_seconds + 10.0,
                        self._global_timeout_seconds
                        + self._retrieval_timeout_seconds
                        + self._segmentation_timeout_seconds
                        + 20.0,
                    )
                ),
                cancellation=cancellation,
            )
            evidence: list[Evidence] = []
            batches: list[CandidateBatch] = []
            warnings: list[str] = []
            timings: dict[str, int] = {}
            ocr_outcome: ProviderOutcome[OCRResult] | None = None
            global_outcome: ProviderOutcome[GlobalPredictionResult] | None = None
            global_synthesis: GlobalPredictionSynthesis | None = None
            retrieval_outcome: ProviderOutcome[object] | None = None
            phase5b_diagnostics: Phase5BDiagnostics | None = None
            scene_analysis: SceneSegmentationSummary | None = None
            segmentation_outcome: ProviderOutcome[SegmentationResult] | None = None
            segmentation_diagnostic: ProviderRunDiagnostic | None = None
            global_descriptor: ProviderDescriptor | None = None
            result_classification = "real"
            phase6a_eligible = False
            simulation: SimulationSummary | None = None
            provider_comparisons: list[ProviderComparisonSummary] = []
            phase6b_geoclip_result: InferenceResult | None = None
            model_predictions: dict[str, Phase6BProviderPredictionSummary] | None = None
            phase6b_fusion: Phase6BFusionSummary | None = None
            phase6b_ocr: Phase6BOCRSummary | None = None
            cloud_assist: Phase6BCloudAssistSummary | None = None
            phase6b_result: Phase6BPipelineResult | None = None
            phase6c_summary: Phase6CAnalysisSummary | None = None

            analysis = await self._advance(analysis, "reading_metadata", 28)
            self._check_cancelled(cancellation)

            analysis = await self._advance(analysis, "exif", 40)
            provider_started = time.monotonic()
            try:
                exif_outcome = await self._invoke(
                    self._exif.extract(job.prepared.original, context),
                    cancellation,
                    deadline_seconds=6.0,
                )
            except TimeoutError:
                exif_outcome = None
                warnings.append("provider.exif.timeout")
            timings["exif"] = self._elapsed_ms(provider_started)
            if exif_outcome is not None and exif_outcome.status == OutcomeStatus.SUCCEEDED:
                if exif_outcome.value is not None:
                    exif_provenance = provenance_for(self._exif.descriptor)
                    evidence_id = "evidence-exif-gps"
                    evidence.append(
                        Evidence(
                            id=evidence_id,
                            type="exif",
                            label="evidence.embedded_gps",
                            display_value="evidence.embedded_gps_present",
                            confidence=0.95,
                            confidence_basis=(
                                "phase1.embedded_metadata_source_support_not_authenticity"
                            ),
                            source=self._exif.descriptor.id,
                            sensitive=True,
                            provenance=exif_provenance,
                        )
                    )
                    batches.append(
                        self._exif_candidates.propose(
                            exif_outcome.value, evidence_id, exif_provenance
                        )
                    )
                    warnings.append("warning.exif_may_be_stale_or_altered")
            elif exif_outcome is not None and exif_outcome.status == OutcomeStatus.FAILED:
                warnings.append("provider.exif.invalid_or_unavailable")
            self._check_cancelled(cancellation)

            analysis = await self._advance(analysis, "quality", 55)
            provider_started = time.monotonic()
            try:
                quality_outcome = await self._invoke(
                    self._quality.analyze(job.prepared.normalized, context),
                    cancellation,
                    deadline_seconds=10.0,
                )
            except TimeoutError:
                quality_outcome = None
                warnings.append("provider.quality.timeout")
            timings["quality"] = self._elapsed_ms(provider_started)
            quality = None
            if quality_outcome is not None and quality_outcome.status == OutcomeStatus.SUCCEEDED:
                quality = quality_outcome.value
                if quality is not None:
                    evidence.append(
                        Evidence(
                            id="evidence-image-quality",
                            type="quality",
                            label="evidence.image_quality",
                            display_value="evidence.image_quality_measured",
                            confidence=1.0,
                            confidence_basis="phase1.deterministic_image_measurement",
                            source=self._quality.descriptor.id,
                            sensitive=False,
                            provenance=provenance_for(self._quality.descriptor),
                        )
                    )
                    warnings.extend(quality.warnings)
            else:
                warnings.append("provider.quality.unavailable")
            analysis = analysis.model_copy(update={"quality": quality})
            self._check_cancelled(cancellation)

            analysis = await self._advance(analysis, "extracting_text", 67)
            self._check_cancelled(cancellation)

            # OCR, global inference and real retrieval depend only on the immutable
            # prepared image and the already-completed EXIF/quality prerequisites.
            # Start them together, then consume them in the preserved stage order.
            ocr_task = asyncio.create_task(
                self._timed_invoke(
                    self._ocr.extract(job.prepared.normalized, context),
                    cancellation,
                    deadline_seconds=self._ocr_timeout_seconds + 1.0,
                )
            )
            fanout_tasks.append(ocr_task)

            coordinated_task: asyncio.Task[_TimedInvocation[CoordinatedInference]] | None = None
            legacy_global_task: (
                asyncio.Task[_TimedInvocation[ProviderOutcome[GlobalPredictionResult]]] | None
            ) = None
            if self._inference is not None:
                request = InferenceRequest(
                    image_handle=job.prepared.normalized,
                    image_bytes=None,
                    width=job.prepared.summary.width,
                    height=job.prepared.summary.height,
                    exif=(
                        exif_outcome.value
                        if exif_outcome is not None
                        and exif_outcome.status == OutcomeStatus.SUCCEEDED
                        else None
                    ),
                    analysis_mode=analysis.analysis_mode,
                    top_k=self._inference_top_k,
                    cancellation=cancellation,
                    deadline=min(
                        context.deadline,
                        datetime.now(UTC) + timedelta(seconds=self._global_timeout_seconds),
                    ),
                    trace_id=job.request_id,
                    max_input_bytes=self._max_inference_input_bytes,
                )
                coordinated_task = asyncio.create_task(
                    self._timed_await(self._inference.infer(request))
                )
                fanout_tasks.append(coordinated_task)
            elif self._global is not None:
                remaining_seconds = max(0.1, (context.deadline - datetime.now(UTC)).total_seconds())
                legacy_global_task = asyncio.create_task(
                    self._timed_invoke(
                        self._global.predict(job.prepared.normalized, context),
                        cancellation,
                        deadline_seconds=min(self._global_timeout_seconds, remaining_seconds),
                    )
                )
                fanout_tasks.append(legacy_global_task)

            retrieval_task: asyncio.Task[_TimedInvocation[ProviderOutcome[object]]] | None = None
            if (
                self._retrieval is not None
                and self._retrieval_fanout_is_real_only()
                and (self._phase6a is None or self._geoclip_clusterer is None)
            ):
                retrieval_task = asyncio.create_task(
                    self._timed_invoke(
                        self._retrieval.search(job.prepared.normalized, context),
                        cancellation,
                        deadline_seconds=self._retrieval_timeout_seconds + 1.0,
                    )
                )
                fanout_tasks.append(retrieval_task)

            timed_ocr = await ocr_task
            ocr_outcome = timed_ocr.value
            timings["ocr"] = timed_ocr.duration_ms
            if timed_ocr.timed_out:
                warnings.append("provider.ocr.timeout")
            if ocr_outcome is not None and ocr_outcome.status == OutcomeStatus.SUCCEEDED:
                if ocr_outcome.value is not None and ocr_outcome.value.redacted_snippets:
                    evidence.append(
                        Evidence(
                            id="evidence-ocr-redacted",
                            type="ocr",
                            label="evidence.ocr_redacted_text",
                            display_value=" · ".join(ocr_outcome.value.redacted_snippets)[:500],
                            confidence=0.60,
                            confidence_basis="phase1.local_ocr_detection_unverified_text",
                            source=self._ocr.descriptor.id,
                            sensitive=False,
                            provenance=provenance_for(self._ocr.descriptor),
                        )
                    )
            elif ocr_outcome is not None and ocr_outcome.status == OutcomeStatus.FAILED:
                warnings.append("provider.ocr.unavailable")
            self._check_cancelled(cancellation)

            global_abstention_reason: str | None = None
            if self._inference is not None or self._global is not None:
                analysis = await self._advance(analysis, "global_prediction", 74)
                if self._inference is not None:
                    if coordinated_task is None:
                        raise ValueError("coordinated inference task is unavailable")
                    try:
                        timed_global = await coordinated_task
                    except InferenceCancelledError as exc:
                        raise JobCancelledError from exc
                    coordinated = timed_global.value
                    if coordinated is None:
                        raise ValueError("coordinated inference returned no result")
                    timings["global_prediction"] = timed_global.duration_ms
                    warnings.extend(coordinated.warnings)
                    provider_comparisons = list(coordinated.comparisons)
                    if any(
                        execution.status.mode == "shadow" and execution.result.status != "skipped"
                        for execution in coordinated.executions
                    ):
                        analysis = await self._advance(analysis, "custom_model_shadow", 75)
                    selected_inference = coordinated.selected
                    decision_inference = selected_inference or coordinated.decision_result
                    if decision_inference is not None:
                        global_descriptor = descriptor_for_inference(decision_inference)
                        global_outcome = legacy_outcome_for_inference(decision_inference)
                    phase6b_geoclip_result = next(
                        (
                            execution.result
                            for execution in coordinated.executions
                            if "geoclip" in execution.result.provider_id.casefold()
                            and execution.result.classification == "real"
                        ),
                        None,
                    )
                    if selected_inference is not None:
                        result_classification = selected_inference.classification
                        if selected_inference.classification == "simulated":
                            if selected_inference.scenario_id is None:
                                raise ValueError("simulated inference scenario is missing")
                            simulation = SimulationSummary(
                                scenario_id=selected_inference.scenario_id
                            )
                else:
                    if legacy_global_task is None or self._global is None:
                        raise ValueError("global provider task is unavailable")
                    timed_global_outcome = await legacy_global_task
                    timings["global_prediction"] = timed_global_outcome.duration_ms
                    global_outcome = timed_global_outcome.value
                    global_descriptor = self._global.descriptor
                    if timed_global_outcome.timed_out:
                        global_outcome = None
                        warnings.append("provider.global_prediction.inference_timeout")
                        global_abstention_reason = "global_prediction_timeout"
                if (
                    global_outcome is not None
                    and global_outcome.status == OutcomeStatus.SUCCEEDED
                    and global_outcome.value is not None
                    and global_descriptor is not None
                ):
                    try:
                        # Keep the legacy/public synthesis bounded while retaining
                        # the full private Top-K result for Phase 6A clustering.
                        public_global_result = global_outcome.value.model_copy(
                            update={"hypotheses": global_outcome.value.hypotheses[:5]}
                        )
                        synthesized = self._global_candidates.synthesize(
                            public_global_result, global_descriptor
                        )
                    except ValueError:
                        warnings.append("provider.global_prediction.invalid_model_output")
                        global_abstention_reason = "global_prediction_invalid_output"
                    else:
                        global_synthesis = synthesized
                        if result_classification == "simulated":
                            batches.clear()
                            evidence = [item for item in evidence if item.type == "quality"]
                            evidence.extend(synthesized.evidence)
                            batches.append(synthesized.batch)
                        elif self._phase5b is None and not self._is_phase6a_eligible(
                            result_classification=result_classification,
                            descriptor=global_descriptor,
                            outcome=global_outcome,
                        ):
                            evidence.extend(synthesized.evidence)
                            batches.append(synthesized.batch)
                        warnings.extend(synthesized.warnings)
                elif global_outcome is not None:
                    failure = global_outcome.failure
                    failure_code = (
                        failure.code if failure is not None else global_outcome.status.value
                    )
                    warnings.append(f"provider.global_prediction.{failure_code}")
                    if failure is not None and failure.subreason_code is not None:
                        warnings.append(
                            f"provider.global_prediction.{failure_code}.{failure.subreason_code}"
                        )
                    if global_outcome.status == OutcomeStatus.SKIPPED:
                        global_abstention_reason = "global_prediction_unavailable"
                    elif global_outcome.status == OutcomeStatus.FAILED:
                        global_abstention_reason = {
                            "model_not_installed": "model_not_installed",
                            "inference_timeout": "global_prediction_timeout",
                            "invalid_model_output": "global_prediction_invalid_output",
                        }.get(failure_code, "global_prediction_failed")
                    else:
                        global_abstention_reason = "global_prediction_abstained"
                self._check_cancelled(cancellation)

            phase6a_eligible = self._is_phase6a_eligible(
                result_classification=result_classification,
                descriptor=global_descriptor,
                outcome=global_outcome,
            )

            retrieval_hits: tuple[RetrievalHit, ...] = ()
            retrieval_diagnostics: dict[str, object] | None = None
            if (
                self._retrieval is not None
                and result_classification == "real"
                and not phase6a_eligible
            ):
                analysis = await self._advance(analysis, "retrieving_references", 76)
                if retrieval_task is None:
                    retrieval_task = asyncio.create_task(
                        self._timed_invoke(
                            self._retrieval.search(job.prepared.normalized, context),
                            cancellation,
                            deadline_seconds=self._retrieval_timeout_seconds + 1.0,
                        )
                    )
                    fanout_tasks.append(retrieval_task)
                timed_retrieval = await retrieval_task
                retrieval_outcome = timed_retrieval.value
                timings["visual_retrieval"] = timed_retrieval.duration_ms
                if timed_retrieval.timed_out:
                    warnings.append("provider.visual_retrieval.timeout")
                if (
                    retrieval_outcome is not None
                    and retrieval_outcome.status == OutcomeStatus.SUCCEEDED
                    and retrieval_outcome.value is not None
                ):
                    retrieval_hits = validated_retrieval_hits(retrieval_outcome.value)
                    if not retrieval_hits and retrieval_outcome.value != []:
                        warnings.append("provider.visual_retrieval.invalid_output")
                elif retrieval_outcome is not None and retrieval_outcome.failure is not None:
                    warnings.append(f"provider.visual_retrieval.{retrieval_outcome.failure.code}")
                diagnostics = getattr(self._retrieval, "diagnostics", None)
                if callable(diagnostics):
                    try:
                        raw_diagnostics = diagnostics()
                    except (OSError, RuntimeError, ValueError):
                        raw_diagnostics = None
                    if isinstance(raw_diagnostics, dict):
                        retrieval_diagnostics = raw_diagnostics
                self._check_cancelled(cancellation)

            if (
                self._segmentation is not None
                and result_classification == "real"
                and self._segmentation.status().enabled
            ):
                analysis = await self._advance(analysis, "scene_segmentation", 77)
                timed_segmentation = await self._timed_invoke(
                    self._segmentation.analyze(job.prepared.normalized, context),
                    cancellation,
                    deadline_seconds=self._segmentation_timeout_seconds + 1.0,
                )
                segmentation_outcome = timed_segmentation.value
                timings["scene_segmentation"] = timed_segmentation.duration_ms
                segmentation_diagnostic = self._segmentation_run_diagnostic(
                    segmentation_outcome,
                    duration_ms=timed_segmentation.duration_ms,
                    timed_out=timed_segmentation.timed_out,
                )
                if (
                    segmentation_outcome is not None
                    and segmentation_outcome.status == OutcomeStatus.SUCCEEDED
                    and segmentation_outcome.value is not None
                ):
                    scene_analysis = self._scene_segmentation_summary(segmentation_outcome.value)
                elif timed_segmentation.timed_out or (
                    segmentation_outcome is not None
                    and segmentation_outcome.failure is not None
                    and segmentation_outcome.failure.code in {"timeout", "inference_timeout"}
                ):
                    warnings.append("provider.segmentation.timeout")
                elif segmentation_diagnostic.reason_code in {
                    "invalid_model_output",
                    "invalid_output",
                }:
                    warnings.append("provider.segmentation.invalid_output")
                else:
                    warnings.append("provider.segmentation.unavailable")
                self._check_cancelled(cancellation)

            analysis = await self._advance(analysis, "resolving_places", 78)
            if phase6a_eligible:
                if (
                    self._phase6a is None
                    or self._geoclip_clusterer is None
                    or global_outcome is None
                    or global_outcome.value is None
                    or global_descriptor is None
                ):
                    raise ValueError("Phase 6A eligible state is incomplete")
                analysis = await self._advance(analysis, "clustering_candidates", 79)
                cluster_started = time.monotonic()
                clusters = self._geoclip_clusterer.cluster(global_outcome.value.hypotheses)
                timings["geoclip_clustering"] = self._elapsed_ms(cluster_started)
                analysis = await self._advance(analysis, "naming_candidate_clusters", 81)
                reverse_started = time.monotonic()
                if self._reverse_geocoder is not None:
                    try:
                        enriched_clusters = await self._invoke(
                            self._reverse_geocoder.enrich(clusters),
                            cancellation,
                            deadline_seconds=(
                                self._reverse_geocoder.maximum_duration_seconds + 1.0
                            ),
                        )
                    except TimeoutError:
                        enriched_clusters = tuple(
                            ReverseGeocodedCluster(
                                cluster=cluster,
                                place=None,
                                warning="provider.reverse_geocoding.timeout",
                            )
                            for cluster in clusters
                        )
                else:
                    enriched_clusters = tuple(
                        ReverseGeocodedCluster(
                            cluster=cluster,
                            place=None,
                            warning="provider.reverse_geocoding.unavailable",
                        )
                        for cluster in clusters
                    )
                timings["reverse_geocoding"] = self._elapsed_ms(reverse_started)
                reverse_diagnostic = self._reverse_geocoding_diagnostic(
                    enriched_clusters,
                    duration_ms=timings["reverse_geocoding"],
                )
                source_adaptation = self._phase5b_sources.adapt(
                    global_descriptor=global_descriptor,
                    ocr_descriptor=self._ocr.descriptor,
                    global_synthesis=global_synthesis,
                    ocr_result=(
                        ocr_outcome.value
                        if ocr_outcome is not None and ocr_outcome.status == OutcomeStatus.SUCCEEDED
                        else None
                    ),
                    global_outcome=global_outcome,
                    ocr_outcome=ocr_outcome,
                    global_duration_ms=timings.get("global_prediction"),
                    ocr_duration_ms=timings.get("ocr"),
                    ocr_device=self._provider_device(self._ocr),
                )
                providers = [*source_adaptation.providers]
                if segmentation_diagnostic is not None:
                    providers.append(segmentation_diagnostic)
                providers.append(reverse_diagnostic)
                partial_failures = [*source_adaptation.partial_failures]
                if (
                    segmentation_diagnostic is not None
                    and segmentation_diagnostic.status == "failed"
                    and segmentation_diagnostic.reason_code is not None
                ):
                    partial_failures.append(
                        f"phase6a.segmentation.{segmentation_diagnostic.reason_code}"
                    )
                partial_failures.extend(
                    item.warning for item in enriched_clusters if item.warning is not None
                )
                phase6a_result = self._phase6a.rank(
                    enriched_clusters,
                    geoclip_descriptor=global_descriptor,
                    ocr_descriptor=self._ocr.descriptor,
                    ocr=(
                        ocr_outcome.value
                        if ocr_outcome is not None and ocr_outcome.status == OutcomeStatus.SUCCEEDED
                        else None
                    ),
                    quality=quality,
                    segmentation=scene_analysis,
                    providers=providers,
                    partial_failures=tuple(dict.fromkeys(partial_failures)),
                )
                evidence.extend(phase6a_result.evidence)
                if phase6a_result.batch.candidates:
                    batches.append(phase6a_result.batch)
                phase5b_diagnostics = phase6a_result.diagnostics
                warnings.extend(phase6a_result.warnings)
            elif self._phase5b is not None and result_classification == "real":
                effective_global_descriptor = global_descriptor or (
                    self._global.descriptor
                    if self._global is not None
                    else self._unavailable_descriptor(
                        "global-geolocation-unavailable", "global_geolocation"
                    )
                )
                source_adaptation = self._phase5b_sources.adapt(
                    global_descriptor=effective_global_descriptor,
                    ocr_descriptor=self._ocr.descriptor,
                    global_synthesis=global_synthesis,
                    ocr_result=(
                        ocr_outcome.value
                        if ocr_outcome is not None and ocr_outcome.status == OutcomeStatus.SUCCEEDED
                        else None
                    ),
                    global_outcome=global_outcome,
                    ocr_outcome=ocr_outcome,
                    global_duration_ms=timings.get("global_prediction"),
                    ocr_duration_ms=timings.get("ocr"),
                    ocr_device=self._provider_device(self._ocr),
                )
                retrieval_descriptor = self._provider_descriptor(
                    self._retrieval,
                    fallback_id="visual-retrieval-unavailable",
                    fallback_kind="image_retrieval",
                )
                retrieval_adaptation = self._phase5b_retrieval.adapt(
                    descriptor=retrieval_descriptor,
                    outcome=retrieval_outcome,
                    hits=retrieval_hits,
                    duration_ms=timings.get("visual_retrieval", 0),
                    diagnostics=retrieval_diagnostics,
                )
                phase5b_hypotheses = [
                    *source_adaptation.hypotheses,
                    *retrieval_adaptation.hypotheses,
                ]
                map_provider: ProviderRunDiagnostic | None = None
                map_failure: str | None = None
                if self._map_evaluator is not None:
                    analysis = await self._advance(analysis, "checking_map_evidence", 80)
                    map_started = time.monotonic()
                    phase5b_hypotheses, map_provider, map_failure = await self._apply_map_evidence(
                        phase5b_hypotheses
                    )
                    timings["map_evidence"] = self._elapsed_ms(map_started)
                evidence.extend(source_adaptation.evidence)
                evidence.extend(retrieval_adaptation.evidence)
                providers = [
                    *source_adaptation.providers,
                    retrieval_adaptation.provider,
                ]
                if map_provider is not None:
                    providers.append(map_provider)
                partial_failures = [
                    *source_adaptation.partial_failures,
                    *retrieval_adaptation.partial_failures,
                ]
                if map_failure is not None:
                    partial_failures.append(map_failure)
                phase5b_result = self._phase5b.rank_candidates(
                    phase5b_hypotheses,
                    providers=providers,
                    reference_index=retrieval_adaptation.reference_index,
                    partial_failures=partial_failures,
                )
                phase5b_diagnostics = phase5b_result.diagnostics
                if phase5b_result.batch.candidates:
                    batches.append(phase5b_result.batch)
                elif global_synthesis is not None and not phase5b_hypotheses:
                    evidence.extend(global_synthesis.evidence)
                    batches.append(global_synthesis.batch)
                    warnings.append("provider.phase5b.safe_global_fallback")
            if (
                self._phase6b is not None
                and phase6b_geoclip_result is not None
                and result_classification == "real"
            ):
                analysis = await self._advance(analysis, "multi_model_geolocation", 83)
                phase6b_result = await self._phase6b.run(
                    job.prepared.normalized.path,
                    analysis_id=job.id,
                    geoclip=phase6b_geoclip_result,
                    scene=scene_analysis,
                    ocr_outcome=ocr_outcome,
                    ocr_provider_id=self._ocr.descriptor.id,
                    quality=quality,
                    allow_cloud_assist=job.allow_cloud_assist,
                    cloud_consent=job.cloud_consent,
                    cancellation=cancellation,
                )
                model_predictions = phase6b_result.model_predictions
                phase6b_fusion = phase6b_result.fusion
                phase6b_ocr = phase6b_result.ocr
                cloud_assist = phase6b_result.cloud_assist
                evidence.extend(phase6b_result.evidence)
                timings.update(phase6b_result.timings_ms)
                warnings.extend(phase6b_result.warnings)
                top_cluster = (
                    phase6b_fusion.candidate_clusters[0]
                    if phase6b_fusion.candidate_clusters
                    else None
                )
                if (
                    phase6b_result.candidate_batch is not None
                    and top_cluster is not None
                    and top_cluster.independent_family_count >= 2
                ):
                    batches = [
                        batch
                        for batch in batches
                        if batch.provider_id != "phase6a-hybrid-evidence-engine"
                    ]
                    batches.append(phase6b_result.candidate_batch)
            if self._phase6c is not None and result_classification == "real":
                analysis = await self._advance(analysis, "candidate_recall", 84)
                try:
                    phase6c_result = await self._phase6c.run(
                        job.prepared.normalized.path,
                        image_sha256=job.prepared.summary.sha256,
                        geoclip=phase6b_geoclip_result,
                        phase6b=phase6b_result,
                        ocr_outcome=ocr_outcome,
                        cancellation=cancellation,
                    )
                except asyncio.CancelledError:
                    raise
                except (OSError, RuntimeError, TimeoutError, ValueError):
                    warnings.append("provider.phase6c.safe_degradation")
                else:
                    phase6c_summary = phase6c_result.summary
                    evidence.extend(phase6c_result.evidence)
                    timings.update(phase6c_result.timings_ms)
                    warnings.extend(phase6c_result.warnings)
                    if phase6c_result.replacement_eligible:
                        batches = [
                            batch
                            for batch in batches
                            if batch.provider_id
                            not in {
                                "phase5b-evidence-engine",
                                "phase6a-hybrid-evidence-engine",
                                "phase6b-multimodel-fusion",
                            }
                        ]
                        if phase6c_result.candidate_batch is not None:
                            batches.append(phase6c_result.candidate_batch)
            if analysis.analysis_mode == AnalysisMode.CLOUD_ASSISTED:
                analysis = await self._advance(analysis, "extracting_visual_clues", 85)
            if (
                result_classification == "simulated"
                and analysis.analysis_mode == AnalysisMode.CLOUD_ASSISTED
            ):
                warnings.append("provider.inference.simulation_isolated_from_real_providers")
            elif (
                analysis.analysis_mode == AnalysisMode.CLOUD_ASSISTED and not job.allow_cloud_assist
            ):
                if not job.cloud_consent:
                    warnings.append("provider.cloud_vision.privacy_denied")
                elif not self._vision.descriptor.available:
                    warnings.append("provider.cloud_vision.unavailable")
                else:
                    provider_started = time.monotonic()
                    derivative = await self._image_processor.cloud_derivative(
                        job.prepared.normalized
                    )
                    try:
                        vision_outcome = await self._invoke(
                            self._vision.extract(derivative, context),
                            cancellation,
                            deadline_seconds=self._vision_timeout_seconds,
                        )
                    except TimeoutError:
                        vision_outcome = None
                        warnings.append("provider.cloud_vision.timeout")
                    finally:
                        del derivative
                    timings["cloud_vision"] = self._elapsed_ms(provider_started)
                    if (
                        vision_outcome is not None
                        and vision_outcome.status == OutcomeStatus.SUCCEEDED
                        and vision_outcome.value is not None
                    ):
                        vision_provenance = provenance_for(self._vision.descriptor)
                        vision_evidence_ids: list[str] = []
                        for index, hypothesis in enumerate(vision_outcome.value.hypotheses[:5]):
                            digest = hashlib.sha256(
                                (
                                    f"{hypothesis.latitude:.5f},{hypothesis.longitude:.5f},"
                                    f"{hypothesis.label.casefold()}"
                                ).encode()
                            ).hexdigest()[:12]
                            evidence_id = f"evidence-visual-{index + 1}-{digest}"
                            vision_evidence_ids.append(evidence_id)
                            evidence.append(
                                Evidence(
                                    id=evidence_id,
                                    type="visual_clue",
                                    label="evidence.visible_geographic_clues",
                                    display_value=" · ".join(hypothesis.visible_clues)[:500],
                                    confidence=min(hypothesis.confidence, 0.40),
                                    confidence_basis="phase1.capped_unverified_visual_source_support",
                                    source=self._vision.descriptor.id,
                                    sensitive=False,
                                    provenance=vision_provenance,
                                )
                            )
                        batches.append(
                            self._vision_candidates.propose(
                                vision_outcome.value,
                                vision_evidence_ids,
                                vision_provenance,
                            )
                        )
                    elif (
                        vision_outcome is not None and vision_outcome.status == OutcomeStatus.FAILED
                    ):
                        warnings.append("provider.cloud_vision.safe_degradation")
            self._check_cancelled(cancellation)

            analysis = await self._advance(analysis, "reranking", 90)
            fusion_started = time.monotonic()
            fused = self._fusion.fuse(evidence, batches)
            if (
                not fused.candidates
                and global_abstention_reason is not None
                and fused.abstention is not None
            ):
                fused = type(fused)(
                    candidates=[],
                    abstention=Abstention(
                        reason_code=global_abstention_reason,
                        message_key=f"abstention.{global_abstention_reason}",
                    ),
                )
            timings["fusion"] = self._elapsed_ms(fusion_started)
            analysis = await self._advance(analysis, "finalizing", 96)
            timings["total"] = self._elapsed_ms(started)
            final_progress = Progress(
                stage="completed", percent=100, message_key="progress.completed"
            )
            analysis = analysis.model_copy(
                update={
                    "status": AnalysisStatus.COMPLETED,
                    "progress": final_progress,
                    "evidence": evidence,
                    "candidates": fused.candidates,
                    "abstention": fused.abstention,
                    "warnings": list(dict.fromkeys(warnings)),
                    "timings_ms": timings,
                    "fusion_policy_version": (
                        phase6c_summary.fusion_version
                        if phase6c_summary is not None
                        else phase6b_fusion.version
                        if phase6b_fusion is not None
                        else phase5b_diagnostics.reranker_version
                        if phase5b_diagnostics is not None
                        else self._fusion.policy_version
                    ),
                    "pipeline_version": (
                        "phase6c-v1"
                        if self._phase6c is not None
                        else analysis.pipeline_version
                    ),
                    "phase5b_diagnostics": phase5b_diagnostics,
                    "scene_analysis": scene_analysis,
                    "model_predictions": model_predictions,
                    "fusion": phase6b_fusion,
                    "ocr": phase6b_ocr,
                    "cloud_assist": cloud_assist,
                    "phase6c": phase6c_summary,
                    "result_classification": result_classification,
                    "simulation": simulation,
                    "provider_comparisons": provider_comparisons,
                    "failure": None,
                }
            )
            analysis = Analysis.model_validate(analysis.model_dump())
            await self._cancel_fanout_tasks(fanout_tasks)
            await self._cleanup_artifacts(job, preserve_original=self._keep_uploads)
            artifacts_cleaned = True
            await self._repository.save(analysis)
            await self._broker.publish(
                analysis.id, "completed", AnalysisStatus.COMPLETED, final_progress
            )
            completed = True
        except (JobCancelledError, AnalysisNotFoundError):
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._cancel_fanout_tasks(fanout_tasks)
            await self._cleanup_artifacts(job, preserve_original=False)
            artifacts_cleaned = True
            await self._persist_safe_failure(job.id, analysis)
        finally:
            await self._cancel_fanout_tasks(fanout_tasks)
            if not artifacts_cleaned:
                await self._cleanup_artifacts(
                    job, preserve_original=completed and self._keep_uploads
                )

    def _is_phase6a_eligible(
        self,
        *,
        result_classification: str,
        descriptor: ProviderDescriptor | None,
        outcome: ProviderOutcome[GlobalPredictionResult] | None,
    ) -> bool:
        return bool(
            self._phase6a is not None
            and self._geoclip_clusterer is not None
            and result_classification == "real"
            and descriptor is not None
            and descriptor.id == "geoclip-global-v1"
            and outcome is not None
            and outcome.status == OutcomeStatus.SUCCEEDED
            and outcome.value is not None
            and outcome.value.provider_id == "geoclip-global-v1"
        )

    @staticmethod
    def _scene_segmentation_summary(
        result: SegmentationResult,
    ) -> SceneSegmentationSummary:
        return SceneSegmentationSummary(
            provider=result.provider,
            device=result.device,
            inference_ms=result.inference_ms,
            image_width=result.image_width,
            image_height=result.image_height,
            semantic_label_names_available=result.semantic_label_names_available,
            dominant_classes=[
                SceneClassSummary(
                    class_id=item.class_id,
                    class_name=item.class_name,
                    pixel_ratio=item.pixel_ratio,
                    percentage=item.percentage,
                )
                for item in result.dominant_classes
            ],
            scene_groups=[
                SceneGroupSummary(
                    name=name,
                    pixel_ratio=ratio,
                    percentage=round(ratio * 100.0, 2),
                )
                for name, ratio in sorted(
                    result.scene_groups.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            scene_tags=[
                SceneTagSummary(
                    name=item.name,
                    strength=item.strength,
                    strength_semantics=item.strength_semantics,
                    reason=item.reason,
                )
                for item in result.scene_tags
            ],
            warnings=list(result.warnings),
        )

    def _segmentation_run_diagnostic(
        self,
        outcome: ProviderOutcome[SegmentationResult] | None,
        *,
        duration_ms: int,
        timed_out: bool,
    ) -> ProviderRunDiagnostic:
        if self._segmentation is None:
            raise ValueError("segmentation provider is unavailable")
        value = outcome.value if outcome is not None else None
        if timed_out:
            status = "failed"
            reason = "timeout"
        elif outcome is None:
            status = "failed"
            reason = "unavailable"
        elif outcome.status == OutcomeStatus.SUCCEEDED and value is not None:
            status = "succeeded"
            reason = None
        elif outcome.status == OutcomeStatus.SUCCEEDED:
            status = "failed"
            reason = "invalid_output"
        else:
            status = outcome.status.value
            reason = outcome.failure.code if outcome.failure is not None else outcome.status.value
        return ProviderRunDiagnostic(
            provider_id=self._segmentation.descriptor.id,
            provider_type="scene_segmentation",
            status=status,
            duration_ms=max(0, duration_ms),
            reason_code=reason,
            device=(value.device if value is not None else self._segmentation.status().device),
            offline=True,
        )

    def _reverse_geocoding_diagnostic(
        self,
        enriched: tuple[ReverseGeocodedCluster, ...],
        *,
        duration_ms: int,
    ) -> ProviderRunDiagnostic:
        warnings = {item.warning for item in enriched if item.warning is not None}
        provider_id = (
            self._reverse_geocoder.provider_id
            if self._reverse_geocoder is not None
            else "reverse-geocoding-unavailable"
        )
        if self._reverse_geocoder is None or not self._reverse_geocoder.available:
            status = "skipped"
            reason = "unavailable"
        elif "provider.reverse_geocoding.timeout" in warnings:
            status = "failed"
            reason = "timeout"
        elif "provider.reverse_geocoding.failed" in warnings:
            status = "failed"
            reason = "failed"
        elif not enriched:
            status = "abstained"
            reason = "empty_candidate_set"
        else:
            status = "succeeded"
            reason = None
        return ProviderRunDiagnostic(
            provider_id=provider_id,
            provider_type="reverse_geocoding",
            status=status,
            duration_ms=max(0, duration_ms),
            reason_code=reason,
            device="cpu",
            offline=True,
        )

    async def _advance(self, analysis: Analysis, stage: str, percent: int) -> Analysis:
        progress = Progress(stage=stage, percent=percent, message_key=f"progress.{stage}")
        updated = analysis.model_copy(
            update={"status": AnalysisStatus.PROCESSING, "progress": progress}
        )
        await self._repository.save(updated)
        await self._broker.publish(updated.id, "progress", updated.status, progress)
        return updated

    async def _apply_map_evidence(
        self, hypotheses: list[EvidenceHypothesis]
    ) -> tuple[list[EvidenceHypothesis], ProviderRunDiagnostic, str | None]:
        if self._map_evaluator is None:
            raise ValueError("map evaluator is unavailable")
        started = time.monotonic()
        feature_by_place_type = {
            "road": "roads",
            "station": "railway",
            "airport": "airport",
            "public_landmark": "buildings",
            "public_institution": "buildings",
        }
        updated: list[EvidenceHypothesis] = []
        attempted = 0
        known = 0
        for hypothesis in hypotheses:
            place = hypothesis.place_matches[0] if hypothesis.place_matches else None
            map_feature = feature_by_place_type.get(place.match_type) if place is not None else None
            if place is None or map_feature is None or attempted >= 5:
                updated.append(hypothesis)
                continue
            attempted += 1
            observations = await self._map_evaluator.evaluate(
                (
                    MapClue(
                        clue=f"public_place_type_{place.match_type}",
                        map_feature=map_feature,
                    ),
                ),
                latitude=hypothesis.latitude,
                longitude=hypothesis.longitude,
                radius_km=hypothesis.uncertainty_radius_km,
            )
            if any(item.status != "unknown" for item in observations):
                known += 1
            updated.append(hypothesis.model_copy(update={"map_observations": observations}))
        status: Literal["succeeded", "abstained"] = "succeeded" if known else "abstained"
        reason = None if known else "no_structured_map_result"
        diagnostic = ProviderRunDiagnostic(
            provider_id="osm-overpass-map-evidence",
            provider_type="map_research",
            status=status,
            duration_ms=self._elapsed_ms(started),
            reason_code=reason,
            device=None,
            offline=False,
        )
        failure = "phase5b.map.unavailable" if attempted > 0 and known == 0 else None
        return updated, diagnostic, failure

    def _retrieval_fanout_is_real_only(self) -> bool:
        """Keep simulated runs isolated from real retrieval side effects."""
        if self._inference is None:
            return True
        try:
            statuses = self._inference.statuses()
        except (OSError, RuntimeError, ValueError):
            return False
        return all(status.classification == "real" for status in statuses)

    @staticmethod
    async def _timed_await[T](operation: Awaitable[T]) -> _TimedInvocation[T]:
        started = time.monotonic()
        value = await operation
        return _TimedInvocation(
            value=value,
            duration_ms=AnalysisPipeline._elapsed_ms(started),
        )

    async def _timed_invoke[T](
        self,
        operation: Awaitable[T],
        cancellation: asyncio.Event,
        *,
        deadline_seconds: float,
    ) -> _TimedInvocation[T]:
        started = time.monotonic()
        try:
            value = await self._invoke(
                operation,
                cancellation,
                deadline_seconds=deadline_seconds,
            )
        except TimeoutError:
            return _TimedInvocation(
                value=None,
                duration_ms=self._elapsed_ms(started),
                timed_out=True,
            )
        return _TimedInvocation(value=value, duration_ms=self._elapsed_ms(started))

    @staticmethod
    async def _cancel_fanout_tasks(tasks: list[asyncio.Task[Any]]) -> None:
        if not tasks:
            return
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        tasks.clear()

    @staticmethod
    def _provider_descriptor(
        provider: object | None, *, fallback_id: str, fallback_kind: str
    ) -> ProviderDescriptor:
        descriptor = getattr(provider, "descriptor", None)
        if isinstance(descriptor, ProviderDescriptor):
            return descriptor
        return AnalysisPipeline._unavailable_descriptor(fallback_id, fallback_kind)

    @staticmethod
    def _unavailable_descriptor(provider_id: str, kind: str) -> ProviderDescriptor:
        return ProviderDescriptor(
            id=provider_id,
            kind=kind,
            version="unavailable",
            execution_boundary="local",
            criticality="optional",
            available=False,
            unavailable_reason_code="unavailable",
        )

    @staticmethod
    def _provider_device(provider: object) -> str | None:
        status = getattr(provider, "safe_status", None)
        if not callable(status):
            return None
        try:
            value = status()
        except (OSError, RuntimeError, ValueError):
            return None
        if not isinstance(value, dict):
            return None
        device = value.get("device")
        return device if isinstance(device, str) else None

    async def _persist_safe_failure(self, analysis_id: UUID, current: Analysis | None) -> None:
        stored = await self._repository.get(analysis_id)
        if stored is None:
            return
        analysis = current or stored.analysis
        progress = Progress(stage="failed", percent=100, message_key="progress.failed")
        failed = analysis.model_copy(
            update={
                "status": AnalysisStatus.FAILED,
                "progress": progress,
                "failure": FailureSummary(
                    code="internal_analysis_error",
                    message_key="error.analysis_failed",
                    retryable=False,
                ),
            }
        )
        try:
            await self._repository.save(failed)
            await self._broker.publish(analysis_id, "failed", AnalysisStatus.FAILED, progress)
        except AnalysisNotFoundError:
            return

    async def _cleanup_artifacts(self, job: AnalysisJob, *, preserve_original: bool) -> None:
        await self._storage.delete(job.prepared.normalized.key)
        if not preserve_original:
            await self._storage.delete(job.prepared.original.key)
            await self._repository.clear_storage_key(job.id)

    @staticmethod
    def _check_cancelled(cancellation: asyncio.Event) -> None:
        if cancellation.is_set():
            raise JobCancelledError

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    @staticmethod
    async def _invoke[T](
        operation: Awaitable[T],
        cancellation: asyncio.Event,
        *,
        deadline_seconds: float,
    ) -> T:
        provider_task = asyncio.ensure_future(operation)
        cancellation_task = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {provider_task, cancellation_task},
                timeout=deadline_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_task in done and cancellation.is_set():
                provider_task.cancel()
                await asyncio.gather(provider_task, return_exceptions=True)
                raise JobCancelledError
            if provider_task not in done:
                provider_task.cancel()
                await asyncio.gather(provider_task, return_exceptions=True)
                raise TimeoutError
            return await provider_task
        finally:
            if not provider_task.done():
                provider_task.cancel()
            cancellation_task.cancel()
            await asyncio.gather(
                provider_task,
                cancellation_task,
                return_exceptions=True,
            )
