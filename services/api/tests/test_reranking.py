from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid5

import pytest

from atlaslens_api.reranking import (
    CandidateHypothesisBuilder,
    CandidateScoreFusionService,
    RerankConfig,
    RerankFeatureExtractor,
    RetrievalReranker,
)
from atlaslens_api.reranking.models import (
    HypothesisClassification,
    RerankClassification,
)
from atlaslens_api.retrieval.models import EmbeddingSpec, ImageMetadata, RetrievalHit

NAMESPACE = UUID("00000000-0000-4000-8000-000000000004")
CONFIG_PATH = Path(__file__).parents[3] / "config" / "reranking" / "phase4-v1.json"


def config() -> RerankConfig:
    return RerankConfig.from_path(CONFIG_PATH)


def hit(
    name: str,
    latitude: float,
    longitude: float,
    *,
    distance: float = 0.2,
    source: str = "source-a",
    content: str | None = None,
) -> RetrievalHit:
    image_id = uuid5(NAMESPACE, name)
    return RetrievalHit(
        distance=distance,
        provider=EmbeddingSpec(provider="test-embedding", version="1", dimension=4),
        metadata=ImageMetadata(
            image_id=image_id,
            index_id=image_id.int % 1_000_000,
            latitude=latitude,
            longitude=longitude,
            source=source,
            license="CC-BY-4.0",
            capture_type="street",
            hash=hashlib.sha256((content or name).encode()).hexdigest(),
            embedding_provider="test-embedding",
            embedding_version="1",
        ),
    )


def services() -> tuple[CandidateHypothesisBuilder, RetrievalReranker, CandidateScoreFusionService]:
    settings = config()
    builder = CandidateHypothesisBuilder(settings.clustering)
    extractor = RerankFeatureExtractor(settings)
    fusion = CandidateScoreFusionService(settings, extractor)
    return builder, RetrievalReranker(settings, fusion), fusion


def test_config_is_versioned_and_rejects_unknown_fields(tmp_path: Path) -> None:
    loaded = config()
    assert loaded.version == "phase4-v1"
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["unknown"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        RerankConfig.from_path(invalid)


def test_nearby_hits_cluster_and_distant_hits_separate() -> None:
    builder, _, _ = services()
    hypotheses = builder.build(
        [
            hit("near-a", 41.0, 29.0),
            hit("near-b", 41.01, 29.01, source="source-b"),
            hit("distant-a", 48.85, 2.35),
            hit("distant-b", 48.86, 2.36, source="source-b"),
        ]
    )
    assert len(hypotheses) == 2
    assert sorted(len(item.hit_ids) for item in hypotheses) == [2, 2]
    assert all(item.uncertainty_radius_km > 0 for item in hypotheses)


def test_antimeridian_and_pole_centers_are_spherical_and_finite() -> None:
    builder, _, _ = services()
    antimeridian = builder.build(
        [hit("east", 10.0, 179.9), hit("west", 10.0, -179.9, source="source-b")]
    )[0]
    assert abs(antimeridian.longitude) > 179.8
    assert antimeridian.uncertainty_radius_km > 0

    pole = builder.build(
        [hit("pole-a", 89.9, 0.0), hit("pole-b", 89.9, 120.0, source="source-b")]
    )[0]
    assert pole.latitude > 89.8
    assert -180 <= pole.longitude <= 180


def test_source_hash_domination_and_outlier_are_retained_but_not_scored() -> None:
    builder, _, _ = services()
    hypothesis = builder.build(
        [
            hit("a", 0.0, 0.0, content="same"),
            hit("duplicate", 0.005, 0.005, content="same"),
            hit("b", 0.01, 0.01),
            hit("source-limit", 0.015, 0.015),
            hit("outlier", 0.0, 0.4, source="source-b"),
        ]
    )[0]
    assert len(hypothesis.hit_ids) == 5
    assert len(hypothesis.supporting_hit_ids) == 2
    assert len(hypothesis.suppressed_hit_ids) == 2
    assert len(hypothesis.outlier_hit_ids) == 1
    assert set(hypothesis.hit_ids) == (
        set(hypothesis.supporting_hit_ids)
        | set(hypothesis.suppressed_hit_ids)
        | set(hypothesis.outlier_hit_ids)
    )


def test_multi_source_classification_and_stable_ties() -> None:
    builder, reranker, _ = services()
    hits = [
        hit("alpha-a", 10.0, 10.0),
        hit("alpha-b", 10.01, 10.01, source="source-b"),
        hit("beta-a", -10.0, -10.0),
        hit("beta-b", -10.01, -10.01, source="source-b"),
    ]
    first_hypotheses = builder.build(hits)
    second_hypotheses = builder.build(list(reversed(hits)))
    assert all(
        item.classification == HypothesisClassification.MULTI_SOURCE
        for item in first_hypotheses
    )
    first = reranker.rerank(first_hypotheses)
    second = reranker.rerank(second_hypotheses)
    assert [item.hypothesis_id for item in first.results] == [
        item.hypothesis_id for item in second.results
    ]
    assert [item.rank for item in first.results] == list(range(1, len(first.results) + 1))


def test_raw_weighted_breakdown_and_contradiction_penalty() -> None:
    builder, _, fusion = services()
    hypothesis = builder.build(
        [hit("one", 20.0, 20.0), hit("two", 20.01, 20.01, source="source-b")]
    )[0]
    baseline = fusion.score(hypothesis)
    contradicted = fusion.score(hypothesis, contradiction_strength=0.8)
    without_compactness = fusion.score(
        hypothesis, available_overrides={"compactness": None}
    )
    assert baseline.relative_rank_score > contradicted.relative_rank_score
    assert contradicted.contradiction_penalty > 0
    assert contradicted.classification == RerankClassification.CONTRADICTED
    assert all(
        item.raw * item.weight == pytest.approx(item.contribution)
        for item in baseline.score_breakdown
    )
    assert "compactness" not in {item.feature for item in without_compactness.score_breakdown}
    assert baseline.score_semantics == "uncalibrated_relative_rank"


def test_singleton_and_empty_input_abstain_without_fake_score() -> None:
    builder, reranker, _ = services()
    singleton = builder.build([hit("single", 0.0, 0.0)])
    outcome = reranker.rerank(singleton)
    assert outcome.results == ()
    assert outcome.abstention_reason == "insufficient_reranking_support"
    assert builder.build([]) == []


def test_reranker_deduplicates_and_bounds_candidates() -> None:
    builder, reranker, _ = services()
    hits = []
    for index in range(10):
        latitude = -70.0 + index * 14.0
        hits.extend(
            [
                hit(f"cluster-{index}-a", latitude, 0.0),
                hit(
                    f"cluster-{index}-b",
                    latitude + 0.01,
                    0.01,
                    source="source-b",
                ),
            ]
        )
    hypotheses = builder.build(hits)
    outcome = reranker.rerank([*hypotheses, *hypotheses])
    assert len(outcome.results) == config().scoring.max_candidates
    assert len({result.hypothesis_id for result in outcome.results}) == len(outcome.results)
