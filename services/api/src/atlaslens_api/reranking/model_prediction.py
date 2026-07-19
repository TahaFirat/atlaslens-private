from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from atlaslens_api.providers.base import GlobalPredictionHypothesis


@dataclass(frozen=True, slots=True)
class ModelRankedHypothesis:
    hypothesis: GlobalPredictionHypothesis
    rank: int
    relative_rank_score: float


class ModelPredictionReranker:
    """Phase 4-compatible deterministic ranking for non-retrieval model hypotheses.

    It deliberately consumes no retrieval IDs and does not reinterpret the raw
    bounded model score as confidence or probability.
    """

    version: ClassVar[str] = "phase4-model-only-v1"

    def rerank(
        self, hypotheses: list[GlobalPredictionHypothesis]
    ) -> tuple[ModelRankedHypothesis, ...]:
        ordered = sorted(
            hypotheses,
            key=lambda item: (
                item.rank,
                item.original_rank,
                -item.raw_score,
                item.latitude,
                item.longitude,
            ),
        )[:5]
        if any(not isfinite(item.raw_score) or not 0 <= item.raw_score <= 1 for item in ordered):
            if any(item.score_type == "uncalibrated_gallery_softmax" for item in ordered):
                raise ValueError("softmax model ranking score is invalid")
            raise ValueError("bounded model ranking score is invalid")
        return tuple(
            ModelRankedHypothesis(
                hypothesis=hypothesis,
                rank=rank,
                relative_rank_score=hypothesis.raw_score,
            )
            for rank, hypothesis in enumerate(ordered, start=1)
        )
