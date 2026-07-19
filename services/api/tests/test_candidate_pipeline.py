from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from uuid import UUID, uuid5

import pytest

from atlaslens_api.candidate_pipeline import (
    CandidateVerificationPipeline,
    ConservativeVerificationPolicy,
    GeometrySignal,
    MapConstraintSignal,
)
from atlaslens_api.candidate_pipeline.models import (
    FinalClassification,
    GeometryClassification,
    PipelineProgress,
    SignalClassification,
)
from atlaslens_api.reranking import (
    CandidateHypothesisBuilder,
    CandidateScoreFusionService,
    RerankConfig,
    RerankFeatureExtractor,
    RetrievalReranker,
)
from atlaslens_api.retrieval.models import EmbeddingSpec, ImageMetadata, RetrievalHit

NAMESPACE = UUID("00000000-0000-4000-8000-000000000044")
CONFIG_PATH = Path(__file__).parents[3] / "config" / "reranking" / "phase4-v1.json"


def hit(name: str, latitude: float, longitude: float, *, source: str) -> RetrievalHit:
    image_id = uuid5(NAMESPACE, name)
    return RetrievalHit(
        distance=0.15,
        provider=EmbeddingSpec(provider="test", version="1", dimension=2),
        metadata=ImageMetadata(
            image_id=image_id,
            index_id=image_id.int % 100_000,
            latitude=latitude,
            longitude=longitude,
            source=source,
            license="CC-BY-4.0",
            capture_type="street",
            hash=hashlib.sha256(name.encode()).hexdigest(),
            embedding_provider="test",
            embedding_version="1",
        ),
    )


def build_services() -> tuple[RerankConfig, list, RetrievalReranker]:
    settings = RerankConfig.from_path(CONFIG_PATH)
    builder = CandidateHypothesisBuilder(settings.clustering)
    hypotheses = builder.build(
        [
            hit("a", 41.0, 29.0, source="source-a"),
            hit("b", 41.01, 29.01, source="source-b"),
            hit("c", 41.02, 29.02, source="source-c"),
        ]
    )
    extractor = RerankFeatureExtractor(settings)
    fusion = CandidateScoreFusionService(settings, extractor)
    return settings, hypotheses, RetrievalReranker(settings, fusion)


class RecordingTelemetry:
    def __init__(self) -> None:
        self.updates: list[PipelineProgress] = []

    def progress(self, update: PipelineProgress) -> None:
        self.updates.append(update)


class SupportingMap:
    async def evaluate(self, hypothesis, cancellation):
        del hypothesis, cancellation
        return MapConstraintSignal(
            classification=SignalClassification.SUPPORTS,
            strength=0.7,
            provider_id="fixture-map",
            provider_version="1",
            observation_code="road_context_supports",
            provenance=("offline-fixture",),
        )


class ContradictingMap:
    async def evaluate(self, hypothesis, cancellation):
        del hypothesis, cancellation
        return MapConstraintSignal(
            classification=SignalClassification.CONTRADICTS,
            strength=0.9,
            provider_id="fixture-map",
            provider_version="1",
            observation_code="water_land_conflict",
            provenance=("offline-fixture",),
        )


class FailingMap:
    async def evaluate(self, hypothesis, cancellation):
        del hypothesis, cancellation
        raise RuntimeError("private provider failure")


class SlowMap:
    async def evaluate(self, hypothesis, cancellation):
        del hypothesis, cancellation
        await asyncio.sleep(1)
        raise AssertionError("deadline was not enforced")


class RecordingGeometry:
    def __init__(self, classification=GeometryClassification.INCONCLUSIVE) -> None:
        self.classification = classification
        self.calls: list[UUID] = []

    async def verify(self, hypothesis, reference_hit_id, cancellation):
        del hypothesis, cancellation
        self.calls.append(reference_hit_id)
        return GeometrySignal(
            reference_hit_id=reference_hit_id,
            classification=self.classification,
            strength=0.8 if self.classification == GeometryClassification.SUPPORTED else 0.2,
            provider_id="fixture-geometry",
            provider_version="1",
            reason_code="generated_fixture_result",
            provenance=("generated-fixture",),
        )


def pipeline(settings, reranker, **kwargs):
    return CandidateVerificationPipeline(
        config=settings.pipeline,
        reranker=reranker,
        policy=ConservativeVerificationPolicy(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_pipeline_progress_provenance_and_reference_bounds() -> None:
    settings, hypotheses, reranker = build_services()
    telemetry = RecordingTelemetry()
    geometry = RecordingGeometry(GeometryClassification.SUPPORTED)
    outcome = await pipeline(
        settings,
        reranker,
        map_collaborator=SupportingMap(),
        geometry_collaborator=geometry,
        telemetry=telemetry,
    ).run(hypotheses, asyncio.Event())
    assessment = outcome.assessments[0]
    assert assessment.classification == FinalClassification.MULTI_SOURCE_SUPPORTED
    assert assessment.score_semantics == "uncalibrated_relative_rank"
    assert assessment.provenance
    assert assessment.reference_attributions
    assert len(geometry.calls) <= settings.pipeline.max_references_per_candidate
    assert telemetry.updates[0].stage == "reranking"
    assert telemetry.updates[-1].stage == "completed"


@pytest.mark.asyncio
async def test_optional_provider_failure_is_partial_and_candidate_survives() -> None:
    settings, hypotheses, reranker = build_services()
    outcome = await pipeline(
        settings,
        reranker,
        map_collaborator=FailingMap(),
        geometry_collaborator=RecordingGeometry(),
    ).run(hypotheses, asyncio.Event())
    assert outcome.assessments
    assert outcome.partial_failures[0].code == "map_constraint_failed"
    assert "private provider failure" not in outcome.model_dump_json()


@pytest.mark.asyncio
async def test_strong_contradiction_penalizes_and_abstains() -> None:
    settings, hypotheses, reranker = build_services()
    baseline = reranker.rerank(hypotheses).results[0].relative_rank_score
    outcome = await pipeline(
        settings, reranker, map_collaborator=ContradictingMap()
    ).run(hypotheses, asyncio.Event())
    assessment = outcome.assessments[0]
    assert assessment.classification == FinalClassification.CONTRADICTED
    assert assessment.relative_rank_score < baseline
    assert assessment.contradictions
    assert outcome.abstention_reason == "all_candidates_contradicted_or_unavailable"


@pytest.mark.asyncio
async def test_empty_or_weak_candidates_abstain() -> None:
    settings, _, reranker = build_services()
    outcome = await pipeline(settings, reranker).run([], asyncio.Event())
    assert outcome.assessments == ()
    assert outcome.abstention_reason == "insufficient_reranking_support"


@pytest.mark.asyncio
async def test_cancellation_propagates_without_partial_result() -> None:
    settings, hypotheses, reranker = build_services()
    cancellation = asyncio.Event()
    cancellation.set()
    with pytest.raises(asyncio.CancelledError):
        await pipeline(settings, reranker).run(hypotheses, cancellation)


@pytest.mark.asyncio
async def test_pipeline_enforces_total_deadline_as_partial_failure() -> None:
    settings, hypotheses, reranker = build_services()
    short_budget = settings.pipeline.model_copy(update={"timeout_seconds": 0.01})
    bounded = CandidateVerificationPipeline(
        config=short_budget,
        reranker=reranker,
        policy=ConservativeVerificationPolicy(),
        map_collaborator=SlowMap(),
    )
    outcome = await bounded.run(hypotheses, asyncio.Event())
    assert outcome.elapsed_ms < 500
    assert [failure.code for failure in outcome.partial_failures] == [
        "map_constraint_timeout"
    ]
