from __future__ import annotations

from collections.abc import Sequence

import pytest

from atlaslens_api.constraints import BoundedMapEvidenceEvaluator, MapClue
from atlaslens_api.constraints.models import MapConstraintObservation
from atlaslens_api.phase5b import (
    EvidenceContradiction,
    EvidenceHypothesis,
    Phase5BEvidenceReranker,
    Phase5BRerankConfig,
    default_phase5b_config_path,
)
from atlaslens_api.schemas import (
    GeoPoint,
    MapConstraintSummary,
    PlaceEvidenceSummary,
    Provenance,
    ProviderRunDiagnostic,
    ReferenceIndexDiagnostic,
    RetrievalMatchSummary,
)


def config() -> Phase5BRerankConfig:
    return Phase5BRerankConfig.from_path(default_phase5b_config_path())


def provenance(provider: str) -> Provenance:
    return Provenance(
        provider_id=provider,
        provider_kind="test_evidence",
        provider_version="1",
        execution_boundary="local",
        output_schema_version="test-v1",
    )


def place(
    name: str = "Public Place",
    *,
    latitude: float = 10.0,
    longitude: float = 20.0,
    ambiguity: int = 1,
) -> PlaceEvidenceSummary:
    return PlaceEvidenceSummary(
        matched_entity="redacted-public-place",
        normalized_name=name,
        country_code="AA",
        region="Public Region",
        center=GeoPoint(latitude=latitude, longitude=longitude),
        match_type="city",
        text_similarity=0.9,
        ambiguity_count=ambiguity,
        evidence_strength=0.8,
        source="installed-gazetteer",
        dataset_version="v1",
        license="CC-BY-4.0",
    )


def retrieval(
    reference_id: str,
    *,
    latitude: float = 10.01,
    longitude: float = 20.01,
    similarity: float = 0.85,
) -> RetrievalMatchSummary:
    return RetrievalMatchSummary(
        reference_id=reference_id,
        provider="embedding-v1",
        source=f"licensed-source-{reference_id}",
        distance=1 - similarity,
        relative_similarity=similarity,
        center=GeoPoint(latitude=latitude, longitude=longitude),
        geographic_cluster="cluster-a",
        license="CC-BY-4.0",
        attribution="Public test attribution",
        display_allowed=False,
    )


def map_observation(status: str = "supported", reliability: float = 0.8) -> MapConstraintSummary:
    return MapConstraintSummary(
        clue="visible structured road clue",
        map_feature="roads",
        status=status,
        reliability=reliability,
        query_radius_km=5,
        provider="local-map",
        limitation="coverage varies",
    )


def hypothesis(
    identifier: str,
    *,
    provider: str = "geoclip",
    source: str = "geoclip-gallery",
    kind: str = "global_model",
    latitude: float = 10.0,
    longitude: float = 20.0,
    score: float = 0.1,
    radius: float = 750.0,
    rank: int = 1,
    family: str | None = None,
    content_hash: str | None = None,
    places: tuple[PlaceEvidenceSummary, ...] = (),
    retrievals: tuple[RetrievalMatchSummary, ...] = (),
    maps: tuple[MapConstraintSummary, ...] = (),
    contradictions: tuple[EvidenceContradiction, ...] = (),
) -> EvidenceHypothesis:
    return EvidenceHypothesis(
        id=identifier,
        provider_id=provider,
        source_id=source,
        source_kind=kind,  # type: ignore[arg-type]
        latitude=latitude,
        longitude=longitude,
        raw_score=score,
        score_semantics="uncalibrated_relative_score",
        uncertainty_radius_km=radius,
        original_rank=rank,
        capture_family_id=family,
        content_hash=content_hash,
        evidence_ids=(f"evidence-{identifier}",),
        provenance=(provenance(provider),),
        place_matches=places,
        retrieval_matches=retrievals,
        map_observations=maps,
        contradictions=contradictions,
    )


def test_model_variants_are_suppressed_and_keep_broad_uncertainty() -> None:
    inputs = [
        hypothesis(
            f"model-{index}",
            latitude=10 + index * 0.001,
            longitude=20 + index * 0.001,
            score=0.2 - index * 0.01,
            rank=index + 1,
            family="one-gallery-family",
        )
        for index in range(5)
    ]
    result = Phase5BEvidenceReranker(config()).rerank(inputs)
    assert not result.abstained
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.member_ids == ("model-0",)
    assert cluster.suppressed_member_count == 4
    assert cluster.assessment.classification == "model_only"
    assert cluster.assessment.provider_diversity == 1
    assert cluster.uncertainty_radius_km == 750
    assert "phase5b.repeated_source_members_suppressed" in cluster.assessment.limitations


def test_independent_place_and_retrieval_support_narrow_without_probability() -> None:
    inputs = [
        hypothesis("model"),
        hypothesis(
            "place",
            provider="local-ocr-place",
            source="installed-gazetteer",
            kind="ocr_place",
            score=0.8,
            radius=25,
            places=(place(),),
        ),
        hypothesis(
            "retrieval-a",
            provider="local-embedding",
            source="licensed-a",
            kind="visual_retrieval",
            latitude=10.01,
            longitude=20.01,
            score=0.85,
            radius=5,
            retrievals=(retrieval("a"),),
            family="capture-a",
        ),
        hypothesis(
            "retrieval-b",
            provider="local-embedding",
            source="licensed-b",
            kind="visual_retrieval",
            latitude=10.012,
            longitude=20.012,
            score=0.82,
            radius=5,
            retrievals=(retrieval("b", latitude=10.012, longitude=20.012),),
            family="capture-b",
        ),
    ]
    result = Phase5BEvidenceReranker(config()).rerank(inputs)
    cluster = result.clusters[0]
    assert cluster.assessment.classification == "multi_source_supported"
    assert cluster.assessment.provider_diversity == 3
    assert cluster.assessment.source_diversity == 4
    assert 5 <= cluster.uncertainty_radius_km < 750
    assert cluster.assessment.score_semantics == "uncalibrated_relative_rank"
    assert any(item.feature == "place_support" for item in cluster.assessment.score_breakdown)
    assert any(
        item.feature == "retrieval_similarity" for item in cluster.assessment.score_breakdown
    )


def test_capture_family_and_content_hash_cannot_multiply_support() -> None:
    digest = "a" * 64
    inputs = [
        hypothesis(
            "retrieval-1",
            provider="embed",
            source="source-a",
            kind="visual_retrieval",
            score=0.9,
            radius=5,
            family="sequence-a",
            content_hash=digest,
            retrievals=(retrieval("one"),),
        ),
        hypothesis(
            "retrieval-2",
            provider="embed",
            source="source-b",
            kind="visual_retrieval",
            score=0.8,
            radius=5,
            family="sequence-a",
            content_hash=digest,
            retrievals=(retrieval("two"),),
        ),
    ]
    cluster = Phase5BEvidenceReranker(config()).rerank(inputs).clusters[0]
    assert len(cluster.member_ids) == 1
    assert cluster.assessment.source_diversity == 1
    assert len(cluster.assessment.retrieval_matches) == 1


def test_strong_contradiction_causes_honest_abstention() -> None:
    contradicted = hypothesis(
        "contradicted",
        maps=(map_observation("contradicted", 0.9),),
        contradictions=(
            EvidenceContradiction(reason_code="phase5b.test.contradiction", strength=0.9),
        ),
    )
    result = Phase5BEvidenceReranker(config()).rerank([contradicted])
    assert result.abstained
    assert result.abstention_reason == "phase5b.all_candidates_contradicted"
    assert result.clusters[0].assessment.classification == "contradicted"
    penalty = next(
        item
        for item in result.clusters[0].assessment.score_breakdown
        if item.feature == "contradiction_penalty"
    )
    assert penalty.contribution < 0


def test_map_support_changes_stable_relative_order() -> None:
    unsupported = hypothesis("a", latitude=0, longitude=0, score=0.1, rank=1)
    supported = hypothesis(
        "b",
        latitude=30,
        longitude=30,
        score=0.1,
        rank=2,
        maps=(map_observation(),),
    )
    engine = Phase5BEvidenceReranker(config())
    first = engine.rerank([unsupported, supported])
    second = engine.rerank([supported, unsupported])
    assert [item.id for item in first.clusters] == [item.id for item in second.clusters]
    assert first.clusters[0].member_ids == ("b",)
    assert first.clusters[0].assessment.classification == "map_supported"
    assert (
        "phase5b.rank_increased.evidence_support"
        in first.clusters[0].assessment.movement_reasons
    )


def test_empty_input_abstains_and_candidate_adapter_returns_diagnostics() -> None:
    engine = Phase5BEvidenceReranker(config())
    assert engine.rerank([]).abstention_reason == "phase5b.no_hypotheses"
    input_hypothesis = hypothesis("model")
    result = engine.rank_candidates(
        [input_hypothesis],
        providers=[
            ProviderRunDiagnostic(
                provider_id="geoclip",
                provider_type="global_model",
                status="succeeded",
                duration_ms=10,
                device="cpu",
                offline=True,
            )
        ],
        reference_index=ReferenceIndexDiagnostic(status="disabled", image_count=0),
        partial_failures=["phase5b.optional_provider_unavailable"],
    )
    assert len(result.batch.candidates) == 1
    draft = result.batch.candidates[0]
    assert draft.confidence is None
    assert draft.phase5b_assessment is not None
    assert draft.evidence_ids == ["evidence-model"]
    assert result.diagnostics.providers[0].offline
    assert result.diagnostics.reference_index is not None


class InjectedMapProvider:
    provider_id = "injected-real-seam"
    available = True

    def __init__(self, observations: Sequence[MapConstraintObservation]) -> None:
        self.observations = list(observations)
        self.clue_count = 0
        self.radius = 0.0

    async def evaluate(
        self,
        clues: Sequence[MapClue],
        *,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> list[MapConstraintObservation]:
        del latitude, longitude
        self.clue_count = len(clues)
        self.radius = radius_km
        return self.observations


@pytest.mark.asyncio
async def test_bounded_map_evaluator_deduplicates_and_caps_structured_clues() -> None:
    observation = MapConstraintObservation(
        clue="road clue",
        map_feature="roads",
        status="supported",
        reliability=0.7,
        query_radius_km=10,
        provider="injected-real-seam",
    )
    provider = InjectedMapProvider([observation])
    evaluator = BoundedMapEvidenceEvaluator(provider, max_clues=2, max_query_radius_km=10)
    observed = await evaluator.evaluate(
        [
            MapClue(clue="road clue", map_feature="roads"),
            MapClue(clue="road clue", map_feature="roads"),
            MapClue(clue="water clue", map_feature="water"),
            MapClue(clue="rail clue", map_feature="railway"),
        ],
        latitude=1,
        longitude=2,
        radius_km=50,
    )
    assert provider.clue_count == 2
    assert provider.radius == 10
    assert len(observed) == 1
    assert observed[0].status == "supported"


@pytest.mark.asyncio
async def test_bounded_map_evaluator_fails_closed_without_network_fallback() -> None:
    class FailingProvider(InjectedMapProvider):
        async def evaluate(
            self,
            clues: Sequence[MapClue],
            *,
            latitude: float,
            longitude: float,
            radius_km: float,
        ) -> list[MapConstraintObservation]:
            del clues, latitude, longitude, radius_km
            raise OSError("safe test failure")

    evaluator = BoundedMapEvidenceEvaluator(FailingProvider([]))
    result = await evaluator.evaluate(
        [MapClue(clue="road clue", map_feature="roads")],
        latitude=1,
        longitude=2,
        radius_km=5,
    )
    assert result[0].status == "unknown"
    assert result[0].provider == "phase5b-map-evidence"
    assert "failed safely" in (result[0].limitation or "")
