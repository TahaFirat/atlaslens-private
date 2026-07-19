from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from atlaslens_api.config import Settings
from atlaslens_api.events import AnalysisEventBroker
from atlaslens_api.fusion import DeterministicCandidateFusionService
from atlaslens_api.global_prediction.clustering import (
    GeoClipCandidateCluster,
    GeoClipCandidateClusterer,
)
from atlaslens_api.global_prediction.reverse_geocoding import ReverseGeocodedCluster
from atlaslens_api.image_processing import PreparedImage, SafeImageProcessor
from atlaslens_api.inference import DevelopmentMockProvider, ProviderEnsembleCoordinator
from atlaslens_api.phase6a import (
    Phase6AHybridConfig,
    Phase6AHybridEvidenceEngine,
    default_phase6a_config_path,
)
from atlaslens_api.pipeline import AnalysisJob, AnalysisPipeline
from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    GlobalProviderStatus,
    InvocationContext,
    OCRProvider,
    OCRResult,
    ProviderDescriptor,
    ProviderOutcome,
    RetrievalProvider,
)
from atlaslens_api.providers.exif import PillowExifProvider
from atlaslens_api.providers.ocr import TesseractOCRProvider
from atlaslens_api.providers.openai_vision import OpenAIVisionClueProvider
from atlaslens_api.providers.quality import OpenCVImageQualityProvider
from atlaslens_api.repository import StoredAnalysis
from atlaslens_api.schemas import Analysis, AnalysisMode, AnalysisStatus, ImageSummary, Progress
from atlaslens_api.segmentation import (
    DominantClass,
    SceneSegmentationProvider,
    SegmentationProviderStatus,
    SegmentationResult,
)
from atlaslens_api.storage import LocalImageHandle, LocalTemporaryStorage
from conftest import gps_jpeg, image_bytes


class MemoryRepository:
    def __init__(self, analysis: Analysis, storage_key: str) -> None:
        self.analysis = analysis
        self.storage_key: str | None = storage_key

    async def get(self, analysis_id: UUID) -> StoredAnalysis | None:
        if analysis_id != self.analysis.id:
            return None
        return StoredAnalysis(self.analysis, self.storage_key, None)

    async def save(self, analysis: Analysis) -> None:
        self.analysis = analysis

    async def clear_storage_key(self, analysis_id: UUID) -> None:
        if analysis_id == self.analysis.id:
            self.storage_key = None


class FakeGlobalProvider:
    descriptor = ProviderDescriptor(
        id="test-global-provider",
        kind="global_geolocation",
        version="test",
        execution_boundary="local",
        criticality="optional",
        available=True,
        model_name="test-only-model",
    )

    def __init__(self, outcome: ProviderOutcome[GlobalPredictionResult]) -> None:
        self.outcome = outcome
        self.calls = 0

    def status(self) -> GlobalProviderStatus:
        return GlobalProviderStatus(
            status="ready",
            installed=True,
            verified=True,
            model_name="test-only-model",
            model_revision="test",
            device="cpu",
            calibration_state="uncalibrated",
        )

    async def predict(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[GlobalPredictionResult]:
        assert handle.path.is_file()
        assert context.mode == AnalysisMode.LOCAL_ONLY
        self.calls += 1
        return self.outcome


class GeoClipGlobalProvider(FakeGlobalProvider):
    descriptor = FakeGlobalProvider.descriptor.model_copy(
        update={"id": "geoclip-global-v1", "model_name": "geoclip"}
    )


class CountingSegmentationProvider:
    descriptor = ProviderDescriptor(
        id="segformer-scene-v1",
        kind="scene_segmentation",
        version="test",
        execution_boundary="local",
        criticality="optional",
        available=True,
        model_name="test-segformer",
    )

    def __init__(self) -> None:
        self.calls = 0

    def status(self) -> SegmentationProviderStatus:
        return SegmentationProviderStatus(
            provider_id=self.descriptor.id,
            status="ready",
            enabled=True,
            installed=True,
            prepared=True,
            loaded=True,
            usable=True,
            device="cpu",
            weight_source="ema",
            num_labels=2,
            semantic_label_names_available=True,
            checkpoint_sha256="a" * 64,
        )

    async def analyze(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[SegmentationResult]:
        assert handle.path.is_file()
        assert context.mode == AnalysisMode.LOCAL_ONLY
        self.calls += 1
        return ProviderOutcome.succeeded(
            SegmentationResult(
                provider=self.descriptor.id,
                device="cpu",
                inference_ms=2,
                image_width=64,
                image_height=48,
                semantic_label_names_available=True,
                dominant_classes=(
                    DominantClass(
                        class_id=1,
                        class_name="building",
                        pixel_ratio=1.0,
                        percentage=100.0,
                    ),
                ),
                scene_groups={"built_environment": 1.0},
            )
        )


class CountingGeoClipClusterer(GeoClipCandidateClusterer):
    def __init__(self) -> None:
        super().__init__(radius_km=40)
        self.calls = 0

    def cluster(
        self, hypotheses: Sequence[GlobalPredictionHypothesis]
    ) -> tuple[GeoClipCandidateCluster, ...]:
        self.calls += 1
        return super().cluster(hypotheses)


class CountingReverseGeocoder:
    provider_id = "test-reverse-geocoder"
    available = True
    maximum_duration_seconds = 1.0

    def __init__(self) -> None:
        self.calls = 0

    async def enrich(
        self, clusters: Sequence[GeoClipCandidateCluster]
    ) -> tuple[ReverseGeocodedCluster, ...]:
        self.calls += 1
        return tuple(ReverseGeocodedCluster(cluster=cluster, place=None) for cluster in clusters)


class ProviderFanoutGate:
    def __init__(self) -> None:
        self.started: set[str] = set()
        self.cancelled: set[str] = set()
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self, provider: str) -> None:
        self.started.add(provider)
        if self.started == {"ocr", "global", "retrieval"}:
            self.all_started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.add(provider)
            raise


class GatedGlobalProvider(FakeGlobalProvider):
    def __init__(
        self,
        outcome: ProviderOutcome[GlobalPredictionResult],
        gate: ProviderFanoutGate,
    ) -> None:
        super().__init__(outcome)
        self._gate = gate

    async def predict(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[GlobalPredictionResult]:
        await self._gate.wait("global")
        return await super().predict(handle, context)


class GatedOCRProvider:
    descriptor = ProviderDescriptor(
        id="test-gated-ocr",
        kind="ocr",
        version="test",
        execution_boundary="local",
        criticality="optional",
        available=True,
    )

    def __init__(self, gate: ProviderFanoutGate) -> None:
        self._gate = gate

    async def extract(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[OCRResult]:
        del context
        assert handle.path.is_file()
        await self._gate.wait("ocr")
        return ProviderOutcome.abstained()


class GatedRetrievalProvider:
    descriptor = ProviderDescriptor(
        id="test-gated-retrieval",
        kind="image_retrieval",
        version="test",
        execution_boundary="local",
        criticality="optional",
        available=True,
    )

    def __init__(self, gate: ProviderFanoutGate) -> None:
        self._gate = gate

    async def search(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[object]:
        del context
        assert handle.path.is_file()
        await self._gate.wait("retrieval")
        return ProviderOutcome.succeeded([])


class CountingRetrievalProvider:
    descriptor = GatedRetrievalProvider.descriptor

    def __init__(self) -> None:
        self.calls = 0

    async def search(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[object]:
        del handle, context
        self.calls += 1
        return ProviderOutcome.succeeded([])


def prediction_result(provider_id: str = "test-global-provider") -> GlobalPredictionResult:
    hypotheses = [
        GlobalPredictionHypothesis(
            rank=index,
            original_rank=index,
            latitude=latitude,
            longitude=longitude,
            raw_score=score,
            score_type="uncalibrated_gallery_softmax",
            normalization_method="softmax_over_fixed_gallery",
            calibration_state="uncalibrated",
            limitations=["test.fixture_only"],
        )
        for index, (latitude, longitude, score) in enumerate(
            [(40.0, 30.0, 0.5), (10.0, -80.0, 0.3), (-20.0, 120.0, 0.2)],
            1,
        )
    ]
    return GlobalPredictionResult(
        provider_id=provider_id,
        model_name="test-only-model",
        model_revision="test",
        implementation_revision="test",
        device="cpu",
        dtype="float32",
        inference_ms=1,
        hypotheses=hypotheses,
    )


async def run_pipeline(
    tmp_path: Path,
    provider: FakeGlobalProvider,
    payload: bytes,
    *,
    ocr_provider: OCRProvider | None = None,
    retrieval_provider: RetrievalProvider | None = None,
    inference_coordinator: ProviderEnsembleCoordinator | None = None,
    cancellation: asyncio.Event | None = None,
    phase6a_reranker: Phase6AHybridEvidenceEngine | None = None,
    geoclip_clusterer: GeoClipCandidateClusterer | None = None,
    reverse_geocoder: CountingReverseGeocoder | None = None,
    segmentation_provider: SceneSegmentationProvider | None = None,
) -> Analysis:
    storage = LocalTemporaryStorage(tmp_path / "uploads")
    await storage.initialize()
    original = await storage.create_temporary(".upload")
    normalized = await storage.create_temporary(".jpg")
    original.path.write_bytes(payload)
    normalized.path.write_bytes(payload)
    analysis_id = uuid4()
    now = datetime.now(UTC)
    queued = Analysis(
        id=analysis_id,
        status=AnalysisStatus.QUEUED,
        analysis_mode=AnalysisMode.LOCAL_ONLY,
        created_at=now,
        expires_at=now + timedelta(hours=1),
        progress=Progress(stage="queued", percent=0, message_key="progress.queued"),
        image=None,
        quality=None,
        evidence=[],
        candidates=[],
        abstention=None,
        warnings=[],
        timings_ms={},
        fusion_policy_version="phase1-fusion-v1",
        failure=None,
    )
    repository = MemoryRepository(queued, original.key)
    settings = Settings(_env_file=None, temp_storage_dir=tmp_path / "uploads")
    prepared = PreparedImage(
        original=original,
        normalized=normalized,
        summary=ImageSummary(
            format="jpeg",
            width=64,
            height=48,
            megapixels=0.0031,
            sha256="0" * 64,
            orientation_normalized=False,
            exif_present=False,
        ),
    )
    pipeline = AnalysisPipeline(
        repository=repository,
        storage=storage,
        image_processor=SafeImageProcessor(settings, storage),
        broker=AnalysisEventBroker(),
        exif_provider=PillowExifProvider(),
        quality_provider=OpenCVImageQualityProvider(),
        ocr_provider=ocr_provider or TesseractOCRProvider(enabled=False, command=None),
        vision_provider=OpenAIVisionClueProvider(api_key=None, model="unused", timeout_seconds=1),
        fusion=DeterministicCandidateFusionService(),
        keep_uploads=False,
        global_provider=provider,
        inference_coordinator=inference_coordinator,
        retrieval_provider=retrieval_provider,
        phase6a_reranker=phase6a_reranker,
        geoclip_clusterer=geoclip_clusterer,
        reverse_geocoder=reverse_geocoder,
        segmentation_provider=segmentation_provider,
    )
    await pipeline.run(
        AnalysisJob(
            id=analysis_id,
            request_id="request-test-global",
            cloud_consent=False,
            prepared=prepared,
        ),
        cancellation or asyncio.Event(),
    )
    return repository.analysis


@pytest.mark.asyncio
async def test_local_only_non_exif_runs_global_provider_and_returns_model_candidates(
    tmp_path: Path,
) -> None:
    provider = FakeGlobalProvider(ProviderOutcome.succeeded(prediction_result()))
    analysis = await run_pipeline(tmp_path, provider, image_bytes())
    assert provider.calls == 1
    assert analysis.status == AnalysisStatus.COMPLETED
    assert len(analysis.candidates) == 3
    assert analysis.abstention is None
    assert all(candidate.model_prediction is not None for candidate in analysis.candidates)
    assert all(candidate.geoclip_cluster is None for candidate in analysis.candidates)
    assert all(candidate.confidence is None for candidate in analysis.candidates)
    assert all(candidate.radius_km >= 750 for candidate in analysis.candidates)
    assert analysis.scene_analysis is None


@pytest.mark.asyncio
async def test_real_geoclip_runs_phase6a_scene_cluster_and_reverse_pipeline(
    tmp_path: Path,
) -> None:
    segmentation = CountingSegmentationProvider()
    clusterer = CountingGeoClipClusterer()
    reverse_geocoder = CountingReverseGeocoder()
    reranker = Phase6AHybridEvidenceEngine(
        Phase6AHybridConfig.from_path(default_phase6a_config_path())
    )
    provider = GeoClipGlobalProvider(
        ProviderOutcome.succeeded(prediction_result("geoclip-global-v1"))
    )

    analysis = await run_pipeline(
        tmp_path,
        provider,
        image_bytes(),
        phase6a_reranker=reranker,
        geoclip_clusterer=clusterer,
        reverse_geocoder=reverse_geocoder,
        segmentation_provider=segmentation,
    )

    assert analysis.status == AnalysisStatus.COMPLETED
    assert segmentation.calls == 1
    assert clusterer.calls == 1
    assert reverse_geocoder.calls == 1
    assert analysis.scene_analysis is not None
    assert analysis.scene_analysis.provider == "segformer-scene-v1"
    assert analysis.scene_analysis.dominant_classes[0].class_name == "building"
    assert analysis.fusion_policy_version == "phase6a-v1"
    assert analysis.phase5b_diagnostics is not None
    assert analysis.phase5b_diagnostics.reranker_version == "phase6a-v1"
    assert analysis.candidates
    assert all(candidate.geoclip_cluster is not None for candidate in analysis.candidates)
    assert all(candidate.model_prediction is None for candidate in analysis.candidates)
    assert all(candidate.confidence is None for candidate in analysis.candidates)
    assert all(
        candidate.confidence_assessment is not None
        and candidate.confidence_assessment.score is None
        and not candidate.confidence_assessment.calibrated
        for candidate in analysis.candidates
    )


@pytest.mark.asyncio
async def test_unavailable_global_provider_has_specific_abstention(tmp_path: Path) -> None:
    provider = FakeGlobalProvider(ProviderOutcome.skipped("model_not_installed"))
    analysis = await run_pipeline(tmp_path, provider, image_bytes())
    assert analysis.candidates == []
    assert analysis.abstention is not None
    assert analysis.abstention.reason_code == "global_prediction_unavailable"
    assert "provider.global_prediction.model_not_installed" in analysis.warnings


@pytest.mark.asyncio
async def test_global_failure_does_not_erase_exif_candidate(tmp_path: Path) -> None:
    provider = FakeGlobalProvider(
        ProviderOutcome.failed(
            "internal_provider_error", retryable=False, attempts=1, duration_ms=1
        )
    )
    analysis = await run_pipeline(tmp_path, provider, gps_jpeg(41.0, 29.0))
    assert analysis.abstention is None
    assert analysis.candidates
    assert analysis.candidates[0].verification_status == "metadata_only"
    assert "provider.global_prediction.internal_provider_error" in analysis.warnings


@pytest.mark.asyncio
async def test_invalid_model_output_exposes_only_the_safe_subreason(tmp_path: Path) -> None:
    provider = FakeGlobalProvider(
        ProviderOutcome.failed(
            "invalid_model_output",
            retryable=False,
            attempts=1,
            duration_ms=1,
            subreason_code="gps_shape_invalid",
        )
    )
    analysis = await run_pipeline(tmp_path, provider, image_bytes())

    assert analysis.candidates == []
    assert analysis.abstention is not None
    assert analysis.abstention.reason_code == "global_prediction_invalid_output"
    assert "provider.global_prediction.invalid_model_output" in analysis.warnings
    assert "provider.global_prediction.invalid_model_output.gps_shape_invalid" in analysis.warnings
    assert str(tmp_path) not in " ".join(analysis.warnings)


@pytest.mark.asyncio
async def test_pipeline_starts_ocr_global_and_retrieval_concurrently(
    tmp_path: Path,
) -> None:
    gate = ProviderFanoutGate()
    provider = GatedGlobalProvider(ProviderOutcome.succeeded(prediction_result()), gate)
    pipeline_task = asyncio.create_task(
        run_pipeline(
            tmp_path,
            provider,
            image_bytes(),
            ocr_provider=GatedOCRProvider(gate),
            retrieval_provider=GatedRetrievalProvider(gate),
        )
    )
    try:
        await asyncio.wait_for(gate.all_started.wait(), timeout=1.0)
    except BaseException:
        gate.release.set()
        await pipeline_task
        raise

    assert gate.started == {"ocr", "global", "retrieval"}
    gate.release.set()
    analysis = await asyncio.wait_for(pipeline_task, timeout=3.0)

    assert analysis.status == AnalysisStatus.COMPLETED
    assert {"ocr", "global_prediction", "visual_retrieval"} <= set(analysis.timings_ms)


@pytest.mark.asyncio
async def test_simulated_inference_does_not_prestart_real_retrieval(
    tmp_path: Path,
) -> None:
    retrieval = CountingRetrievalProvider()
    segmentation = CountingSegmentationProvider()
    clusterer = CountingGeoClipClusterer()
    reverse_geocoder = CountingReverseGeocoder()
    mock = DevelopmentMockProvider(
        "kayseri-development.json",
        environment="test",
        mode="primary",
    )
    analysis = await run_pipeline(
        tmp_path,
        FakeGlobalProvider(ProviderOutcome.succeeded(prediction_result())),
        image_bytes(),
        retrieval_provider=retrieval,
        inference_coordinator=ProviderEnsembleCoordinator((mock,)),
        phase6a_reranker=Phase6AHybridEvidenceEngine(
            Phase6AHybridConfig.from_path(default_phase6a_config_path())
        ),
        geoclip_clusterer=clusterer,
        reverse_geocoder=reverse_geocoder,
        segmentation_provider=segmentation,
    )

    assert analysis.status == AnalysisStatus.COMPLETED
    assert analysis.result_classification == "simulated"
    assert retrieval.calls == 0
    assert segmentation.calls == 0
    assert clusterer.calls == 0
    assert reverse_geocoder.calls == 0
    assert analysis.scene_analysis is None
    assert analysis.candidates
    assert all(candidate.geoclip_cluster is None for candidate in analysis.candidates)
    assert all(candidate.model_prediction is not None for candidate in analysis.candidates)


@pytest.mark.asyncio
async def test_pipeline_cancels_all_fanout_tasks_before_artifact_cleanup(
    tmp_path: Path,
) -> None:
    gate = ProviderFanoutGate()
    cancellation = asyncio.Event()
    provider = GatedGlobalProvider(ProviderOutcome.succeeded(prediction_result()), gate)
    pipeline_task = asyncio.create_task(
        run_pipeline(
            tmp_path,
            provider,
            image_bytes(),
            ocr_provider=GatedOCRProvider(gate),
            retrieval_provider=GatedRetrievalProvider(gate),
            cancellation=cancellation,
        )
    )
    await asyncio.wait_for(gate.all_started.wait(), timeout=1.0)

    cancellation.set()
    await asyncio.wait_for(pipeline_task, timeout=3.0)

    assert gate.cancelled == {"ocr", "global", "retrieval"}
    assert list((tmp_path / "uploads").iterdir()) == []
