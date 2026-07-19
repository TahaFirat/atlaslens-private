from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Sequence
from typing import TypeVar

from atlaslens_api.candidate_pipeline.models import (
    FinalClassification,
    GeometryClassification,
    GeometrySignal,
    MapConstraintSignal,
    PartialFailure,
    PipelineOutcome,
    PipelineProgress,
    SignalClassification,
)
from atlaslens_api.candidate_pipeline.protocols import (
    GeometryCollaborator,
    MapConstraintCollaborator,
    NullTelemetry,
    Telemetry,
    VerificationPolicy,
)
from atlaslens_api.reranking.config import PipelineConfig
from atlaslens_api.reranking.models import CandidateHypothesis
from atlaslens_api.reranking.service import RetrievalReranker

T = TypeVar("T")


class CandidateVerificationPipeline:
    def __init__(
        self,
        *,
        config: PipelineConfig,
        reranker: RetrievalReranker,
        policy: VerificationPolicy,
        map_collaborator: MapConstraintCollaborator | None = None,
        geometry_collaborator: GeometryCollaborator | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._config = config
        self._reranker = reranker
        self._policy = policy
        self._map = map_collaborator
        self._geometry = geometry_collaborator
        self._telemetry = telemetry or NullTelemetry()

    async def run(
        self, hypotheses: Sequence[CandidateHypothesis], cancellation: asyncio.Event
    ) -> PipelineOutcome:
        started = time.monotonic()
        deadline = started + self._config.timeout_seconds
        self._check_cancelled(cancellation)
        self._telemetry.progress(PipelineProgress(stage="reranking", completed=0, total=1))
        reranked = self._reranker.rerank(hypotheses)
        self._telemetry.progress(PipelineProgress(stage="reranking", completed=1, total=1))
        if not reranked.results:
            return PipelineOutcome(
                assessments=(),
                abstention_reason=reranked.abstention_reason,
                partial_failures=(),
                processed_candidates=0,
                elapsed_ms=self._elapsed_ms(started),
            )

        by_id = {hypothesis.id: hypothesis for hypothesis in hypotheses}
        results = reranked.results[: self._config.max_candidates]
        assessments = []
        failures: list[PartialFailure] = []
        total = len(results)
        for index, initial_rerank in enumerate(results, 1):
            self._check_cancelled(cancellation)
            if time.monotonic() >= deadline:
                failures.append(PartialFailure(code="pipeline_timeout"))
                break
            hypothesis = by_id[initial_rerank.hypothesis_id]
            self._telemetry.progress(
                PipelineProgress(
                    stage="map_constraints",
                    completed=index - 1,
                    total=total,
                    hypothesis_id=hypothesis.id,
                )
            )
            map_signals: tuple[MapConstraintSignal, ...] = ()
            if self._map is not None:
                try:
                    map_signals = (
                        await self._bounded(
                            self._map.evaluate(hypothesis, cancellation), deadline, cancellation
                        ),
                    )
                except TimeoutError:
                    failures.append(
                        PartialFailure(code="map_constraint_timeout", hypothesis_id=hypothesis.id)
                    )
                except Exception:
                    failures.append(
                        PartialFailure(code="map_constraint_failed", hypothesis_id=hypothesis.id)
                    )
            self._check_cancelled(cancellation)

            geometry_signals: list[GeometrySignal] = []
            reference_ids = hypothesis.supporting_hit_ids[
                : self._config.max_references_per_candidate
            ]
            self._telemetry.progress(
                PipelineProgress(
                    stage="geometry",
                    completed=0,
                    total=len(reference_ids),
                    hypothesis_id=hypothesis.id,
                )
            )
            if self._geometry is not None:
                for reference_index, hit_id in enumerate(reference_ids, 1):
                    self._check_cancelled(cancellation)
                    try:
                        geometry_signals.append(
                            await self._bounded(
                                self._geometry.verify(hypothesis, hit_id, cancellation),
                                deadline,
                                cancellation,
                            )
                        )
                    except TimeoutError:
                        failures.append(
                            PartialFailure(
                                code="geometry_timeout",
                                hypothesis_id=hypothesis.id,
                                reference_hit_id=hit_id,
                            )
                        )
                        break
                    except Exception:
                        failures.append(
                            PartialFailure(
                                code="geometry_failed",
                                hypothesis_id=hypothesis.id,
                                reference_hit_id=hit_id,
                            )
                        )
                    self._telemetry.progress(
                        PipelineProgress(
                            stage="geometry",
                            completed=reference_index,
                            total=len(reference_ids),
                            hypothesis_id=hypothesis.id,
                        )
                    )

            contradiction_strength = max(
                [
                    signal.strength
                    for signal in map_signals
                    if signal.classification == SignalClassification.CONTRADICTS
                ]
                + [
                    signal.strength
                    for signal in geometry_signals
                    if signal.classification == GeometryClassification.REJECTED
                ]
                + [0.0]
            )
            rescored = self._reranker.score_hypothesis(
                hypothesis, contradiction_strength=contradiction_strength
            )
            self._telemetry.progress(
                PipelineProgress(
                    stage="assessment",
                    completed=index - 1,
                    total=total,
                    hypothesis_id=hypothesis.id,
                )
            )
            assessments.append(
                self._policy.assess(
                    hypothesis,
                    rescored,
                    map_signals,
                    tuple(geometry_signals),
                )
            )

        assessments.sort(key=lambda item: (-item.relative_rank_score, item.hypothesis_id))
        viable = [
            item
            for item in assessments
            if item.classification
            not in {FinalClassification.CONTRADICTED, FinalClassification.ABSTAINED}
        ]
        self._telemetry.progress(
            PipelineProgress(stage="completed", completed=len(assessments), total=total)
        )
        return PipelineOutcome(
            assessments=tuple(assessments),
            abstention_reason=None if viable else "all_candidates_contradicted_or_unavailable",
            partial_failures=tuple(failures),
            processed_candidates=len(assessments),
            elapsed_ms=self._elapsed_ms(started),
        )

    @staticmethod
    def _check_cancelled(cancellation: asyncio.Event) -> None:
        if cancellation.is_set():
            raise asyncio.CancelledError

    @classmethod
    async def _bounded(
        cls, operation: Awaitable[T], deadline: float, cancellation: asyncio.Event
    ) -> T:
        cls._check_cancelled(cancellation)
        operation_task = asyncio.ensure_future(operation)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise TimeoutError
        cancellation_task = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {operation_task, cancellation_task},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_task in done and cancellation.is_set():
                operation_task.cancel()
                await asyncio.gather(operation_task, return_exceptions=True)
                raise asyncio.CancelledError
            if operation_task not in done:
                operation_task.cancel()
                await asyncio.gather(operation_task, return_exceptions=True)
                raise TimeoutError
            return await operation_task
        finally:
            cancellation_task.cancel()
            await asyncio.gather(cancellation_task, return_exceptions=True)

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))
