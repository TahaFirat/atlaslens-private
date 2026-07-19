from __future__ import annotations

from collections.abc import Mapping, Sequence

from atlaslens_api.reranking.config import RerankConfig
from atlaslens_api.reranking.models import (
    CandidateHypothesis,
    RerankClassification,
    RerankOutcome,
    RerankResult,
    ScoreBreakdown,
)


class RerankFeatureExtractor:
    def __init__(self, config: RerankConfig) -> None:
        self._config = config

    def extract(self, hypothesis: CandidateHypothesis) -> dict[str, float]:
        supporting = [member for member in hypothesis.members if member.role == "supporting"]
        similarity = sum(max(0.0, 1.0 - member.distance / 2.0) for member in supporting) / len(
            supporting
        )
        return {
            "retrieval_similarity": similarity,
            "support_count": min(
                1.0, len(supporting) / self._config.scoring.support_normalizer
            ),
            "source_diversity": hypothesis.contributing_source_count / len(supporting),
            "hash_diversity": hypothesis.hash_count / len(hypothesis.members),
            "compactness": 1.0
            / (1.0 + hypothesis.uncertainty_radius_km / self._config.clustering.radius_km),
        }


class CandidateScoreFusionService:
    _REASONS = {
        "retrieval_similarity": "phase4.cosine_distance_relative_similarity",
        "support_count": "phase4.bounded_independent_support_count",
        "source_diversity": "phase4.contributing_source_diversity",
        "hash_diversity": "phase4.content_hash_diversity",
        "compactness": "phase4.robust_geodesic_compactness",
    }

    def __init__(self, config: RerankConfig, extractor: RerankFeatureExtractor) -> None:
        self._config = config
        self._extractor = extractor

    def score(
        self,
        hypothesis: CandidateHypothesis,
        *,
        contradiction_strength: float = 0.0,
        available_overrides: Mapping[str, float | None] | None = None,
        rank: int = 1,
    ) -> RerankResult:
        if not 0.0 <= contradiction_strength <= 1.0:
            raise ValueError("contradiction strength must be between zero and one")
        features = self._extractor.extract(hypothesis)
        if available_overrides:
            for name, value in available_overrides.items():
                if name not in self._config.scoring.weights:
                    raise ValueError(f"unsupported rerank feature: {name}")
                if value is None:
                    features.pop(name, None)
                elif not 0.0 <= value <= 1.0:
                    raise ValueError("feature values must be between zero and one")
                else:
                    features[name] = value
        active = {
            name: value
            for name, value in features.items()
            if name in self._config.scoring.weights and self._config.scoring.weights[name] > 0
        }
        total_weight = sum(self._config.scoring.weights[name] for name in active)
        if total_weight <= 0:
            raise ValueError("no available weighted reranking features")
        breakdown = tuple(
            ScoreBreakdown(
                feature=name,
                raw=active[name],
                weight=self._config.scoring.weights[name] / total_weight,
                contribution=active[name] * self._config.scoring.weights[name] / total_weight,
                reason=self._REASONS[name],
            )
            for name in sorted(active)
        )
        raw_score = sum(item.contribution for item in breakdown)
        penalty = contradiction_strength * self._config.scoring.contradiction_penalty
        relative_score = max(0.0, min(1.0, raw_score - penalty))
        if len(hypothesis.supporting_hit_ids) < self._config.scoring.minimum_supporting_hits:
            classification = RerankClassification.INSUFFICIENT_SUPPORT
        elif contradiction_strength >= self._config.scoring.contradiction_threshold:
            classification = RerankClassification.CONTRADICTED
        elif relative_score >= self._config.scoring.minimum_relative_score:
            classification = RerankClassification.COMPETITIVE
        else:
            classification = RerankClassification.WEAK
        return RerankResult(
            hypothesis_id=hypothesis.id,
            rank=rank,
            relative_rank_score=relative_score,
            classification=classification,
            score_breakdown=breakdown,
            contradiction_strength=contradiction_strength,
            contradiction_penalty=penalty,
            source_diversity=hypothesis.contributing_source_count,
            hash_diversity=hypothesis.contributing_hash_count,
            contributing_hit_ids=hypothesis.supporting_hit_ids,
        )


class RetrievalReranker:
    def __init__(self, config: RerankConfig, fusion: CandidateScoreFusionService) -> None:
        self._config = config
        self._fusion = fusion

    def score_hypothesis(
        self, hypothesis: CandidateHypothesis, *, contradiction_strength: float = 0.0
    ) -> RerankResult:
        return self._fusion.score(hypothesis, contradiction_strength=contradiction_strength)

    def rerank(self, hypotheses: Sequence[CandidateHypothesis]) -> RerankOutcome:
        deduplicated = {hypothesis.id: hypothesis for hypothesis in hypotheses}
        scored = [self._fusion.score(hypothesis) for hypothesis in deduplicated.values()]
        eligible = [
            result
            for result in scored
            if result.classification == RerankClassification.COMPETITIVE
        ]
        eligible.sort(key=lambda item: (-item.relative_rank_score, item.hypothesis_id))
        kept = eligible[: self._config.scoring.max_candidates]
        ranked = tuple(
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(kept, 1)
        )
        kept_ids = {result.hypothesis_id for result in ranked}
        dropped = tuple(
            sorted(
                result.hypothesis_id
                for result in scored
                if result.hypothesis_id not in kept_ids
            )
        )
        if not ranked:
            return RerankOutcome(
                results=(),
                abstention_reason="insufficient_reranking_support",
                dropped_hypothesis_ids=dropped,
            )
        return RerankOutcome(results=ranked, dropped_hypothesis_ids=dropped)
