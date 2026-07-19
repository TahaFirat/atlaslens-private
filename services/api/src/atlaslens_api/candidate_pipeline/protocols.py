from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import UUID

from atlaslens_api.candidate_pipeline.models import (
    FinalCandidateAssessment,
    GeometrySignal,
    MapConstraintSignal,
    PipelineProgress,
)
from atlaslens_api.reranking.models import CandidateHypothesis, RerankResult


class MapConstraintCollaborator(Protocol):
    async def evaluate(
        self, hypothesis: CandidateHypothesis, cancellation: asyncio.Event
    ) -> MapConstraintSignal: ...


class GeometryCollaborator(Protocol):
    async def verify(
        self,
        hypothesis: CandidateHypothesis,
        reference_hit_id: UUID,
        cancellation: asyncio.Event,
    ) -> GeometrySignal: ...


class VerificationPolicy(Protocol):
    def assess(
        self,
        hypothesis: CandidateHypothesis,
        rerank: RerankResult,
        map_observations: tuple[MapConstraintSignal, ...],
        geometry_results: tuple[GeometrySignal, ...],
    ) -> FinalCandidateAssessment: ...


class VerificationTelemetry(Protocol):
    def progress(self, update: PipelineProgress) -> None: ...


Telemetry = VerificationTelemetry


class NullTelemetry:
    def progress(self, update: PipelineProgress) -> None:
        del update
