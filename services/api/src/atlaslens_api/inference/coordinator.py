from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from atlaslens_api.inference.models import (
    GeolocationInferenceProvider,
    InferenceCandidate,
    InferenceProviderStatus,
    InferenceRequest,
    InferenceResult,
    failure_result,
)
from atlaslens_api.schemas import ProviderComparisonSummary


class InferenceCancelledError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProviderExecution:
    status: InferenceProviderStatus
    result: InferenceResult

    def __repr__(self) -> str:
        return (
            f"ProviderExecution(provider_id={self.status.provider_id!r}, "
            f"mode={self.status.mode!r}, status={self.result.status!r})"
        )


@dataclass(frozen=True, slots=True)
class CoordinatedInference:
    selected: InferenceResult | None
    decision_result: InferenceResult | None
    executions: tuple[ProviderExecution, ...]
    comparisons: tuple[ProviderComparisonSummary, ...]
    warnings: tuple[str, ...]

    def __repr__(self) -> str:
        selected_id = self.selected.provider_id if self.selected is not None else None
        return (
            f"CoordinatedInference(selected={selected_id!r}, "
            f"provider_count={len(self.executions)})"
        )


class ProviderEnsembleCoordinator:
    """Runs independent providers concurrently and applies a deterministic mode policy."""

    def __init__(self, providers: tuple[GeolocationInferenceProvider, ...]) -> None:
        if len(providers) > 9:
            raise ValueError("at most nine inference providers may be registered")
        provider_ids = [provider.provider_id for provider in providers]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("inference provider identifiers must be unique")
        self._providers = providers

    def statuses(self) -> tuple[InferenceProviderStatus, ...]:
        return tuple(self._safe_status(provider) for provider in self._providers)

    async def infer(self, request: InferenceRequest) -> CoordinatedInference:
        if request.cancellation.is_set():
            raise InferenceCancelledError
        statuses = self.statuses()
        executions: list[ProviderExecution | None] = [None] * len(self._providers)
        tasks: list[tuple[int, asyncio.Task[InferenceResult]]] = []
        for index, (provider, status) in enumerate(zip(self._providers, statuses, strict=True)):
            if status.mode == "disabled":
                executions[index] = ProviderExecution(
                    status=status,
                    result=failure_result(
                        status,
                        outcome_status="skipped",
                        code="disabled",
                        retryable=False,
                    ),
                )
                continue
            if not status.available:
                executions[index] = ProviderExecution(
                    status=status,
                    result=failure_result(
                        status,
                        outcome_status="skipped",
                        code=status.reason_code or "unavailable",
                        retryable=False,
                    ),
                )
                continue
            task = asyncio.create_task(self._invoke(provider, status, request))
            tasks.append((index, task))
        try:
            if tasks:
                values = await asyncio.gather(*(task for _, task in tasks))
                for (index, _), value in zip(tasks, values, strict=True):
                    executions[index] = ProviderExecution(status=statuses[index], result=value)
        except (InferenceCancelledError, asyncio.CancelledError):
            for _, task in tasks:
                task.cancel()
            await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
            raise
        completed = tuple(execution for execution in executions if execution is not None)
        selected = self._select(completed)
        decision_result = selected or self._decision_result(completed)
        comparisons = self._comparisons(completed, selected)
        warnings = self._warnings(completed, selected)
        return CoordinatedInference(
            selected=selected,
            decision_result=decision_result,
            executions=completed,
            comparisons=comparisons,
            warnings=warnings,
        )

    @staticmethod
    def _safe_status(
        provider: GeolocationInferenceProvider,
    ) -> InferenceProviderStatus:
        try:
            status = provider.status()
        except Exception:
            return ProviderEnsembleCoordinator._unavailable_status(
                provider, "status_unavailable"
            )
        if not isinstance(status, InferenceProviderStatus):
            return ProviderEnsembleCoordinator._unavailable_status(
                provider, "status_contract_mismatch"
            )
        if (
            status.provider_id != provider.provider_id
            or status.mode != provider.mode
            or status.classification != provider.classification
        ):
            return ProviderEnsembleCoordinator._unavailable_status(
                provider, "status_contract_mismatch"
            )
        return status

    @staticmethod
    def _unavailable_status(
        provider: GeolocationInferenceProvider, reason_code: str
    ) -> InferenceProviderStatus:
        return InferenceProviderStatus(
            provider_id=provider.provider_id,
            provider_type="global_geolocation",
            provider_revision="unavailable",
            mode=provider.mode,
            available=False,
            status="failed",
            classification=provider.classification,
            calibration_state="uncalibrated",
            reason_code=reason_code,
        )

    @staticmethod
    async def _invoke(
        provider: GeolocationInferenceProvider,
        status: InferenceProviderStatus,
        request: InferenceRequest,
    ) -> InferenceResult:
        started = time.monotonic()
        remaining = (request.deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            return failure_result(
                status,
                outcome_status="failed",
                code="timeout",
                retryable=True,
            )
        provider_task = asyncio.create_task(provider.infer(request))
        cancellation_task = asyncio.create_task(request.cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {provider_task, cancellation_task},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_task in done and request.cancellation.is_set():
                provider_task.cancel()
                await asyncio.gather(provider_task, return_exceptions=True)
                raise InferenceCancelledError
            if provider_task not in done:
                provider_task.cancel()
                await asyncio.gather(provider_task, return_exceptions=True)
                return failure_result(
                    status,
                    outcome_status="failed",
                    code="timeout",
                    retryable=True,
                    runtime_ms=_elapsed_ms(started),
                )
            try:
                result = await provider_task
            except asyncio.CancelledError:
                if request.cancellation.is_set():
                    raise InferenceCancelledError from None
                return failure_result(
                    status,
                    outcome_status="failed",
                    code="internal_provider_error",
                    retryable=False,
                    runtime_ms=_elapsed_ms(started),
                )
            except Exception:
                return failure_result(
                    status,
                    outcome_status="failed",
                    code="internal_provider_error",
                    retryable=False,
                    runtime_ms=_elapsed_ms(started),
                )
            if not isinstance(result, InferenceResult):
                return failure_result(
                    status,
                    outcome_status="failed",
                    code="invalid_model_output",
                    retryable=False,
                    runtime_ms=_elapsed_ms(started),
                    subreason_code="invalid_result_type",
                )
            if (
                result.provider_id != status.provider_id
                or result.classification != status.classification
                or result.provider_revision != status.provider_revision
                or len(result.candidates) > request.top_k
            ):
                return failure_result(
                    status,
                    outcome_status="failed",
                    code="invalid_model_output",
                    retryable=False,
                    runtime_ms=_elapsed_ms(started),
                    subreason_code="provider_contract_mismatch",
                )
            return result
        finally:
            cancellation_task.cancel()
            await asyncio.gather(cancellation_task, return_exceptions=True)

    @staticmethod
    def _select(executions: tuple[ProviderExecution, ...]) -> InferenceResult | None:
        for classification in ("real", "simulated"):
            for mode in ("primary", "candidate"):
                for execution in executions:
                    if (
                        execution.status.mode == mode
                        and execution.result.classification == classification
                        and execution.result.status == "succeeded"
                    ):
                        return execution.result
        return None

    @staticmethod
    def _decision_result(
        executions: tuple[ProviderExecution, ...],
    ) -> InferenceResult | None:
        for mode in ("primary", "candidate"):
            for execution in executions:
                if execution.status.mode == mode:
                    return execution.result
        return None

    @staticmethod
    def _comparisons(
        executions: tuple[ProviderExecution, ...], selected: InferenceResult | None
    ) -> tuple[ProviderComparisonSummary, ...]:
        comparisons: list[ProviderComparisonSummary] = []
        for execution in executions:
            mode = execution.status.mode
            result = execution.result
            if mode not in {"shadow", "candidate"} or result is selected:
                continue
            if (
                selected is not None
                and selected.classification == "real"
                and result.classification == "simulated"
            ):
                comparisons.append(
                    ProviderComparisonSummary(
                        provider_id=result.provider_id,
                        mode=mode,
                        status="skipped",
                        runtime_ms=result.runtime_ms,
                        ranking_impact="none",
                        failure_code="simulated_excluded_from_real_result",
                    )
                )
                continue
            distance: float | None = None
            overlap: int | None = None
            if (
                selected is not None
                and selected.status == "succeeded"
                and result.status == "succeeded"
                and selected.classification == result.classification
            ):
                distance = _distance_km(selected.candidates[0], result.candidates[0])
                overlap = sum(
                    any(_distance_km(candidate, other) <= 1.0 for other in selected.candidates)
                    for candidate in result.candidates
                )
            comparisons.append(
                ProviderComparisonSummary(
                    provider_id=result.provider_id,
                    mode=mode,
                    status=result.status,
                    runtime_ms=result.runtime_ms,
                    distance_to_primary_km=distance,
                    candidate_overlap=overlap,
                    ranking_impact=(
                        "eligible_not_applied"
                        if mode == "candidate" and result.status == "succeeded"
                        else "none"
                    ),
                    failure_code=(result.failure.code if result.failure is not None else None),
                )
            )
        return tuple(comparisons[:8])

    @staticmethod
    def _warnings(
        executions: tuple[ProviderExecution, ...], selected: InferenceResult | None
    ) -> tuple[str, ...]:
        warnings: list[str] = list(selected.warnings if selected is not None else ())
        for execution in executions:
            if execution.result.failure is not None and execution.status.mode != "disabled":
                warnings.append(
                    f"provider.inference.{execution.status.provider_id}."
                    f"{execution.result.failure.code}"
                )
            if (
                selected is not None
                and selected.classification == "real"
                and execution.result.classification == "simulated"
                and execution.result.status == "succeeded"
            ):
                warnings.append("provider.inference.simulated_excluded_from_real_result")
        return tuple(dict.fromkeys(warnings))


def _distance_km(first: InferenceCandidate, second: InferenceCandidate) -> float:
    first_latitude = math.radians(float(first.latitude))
    second_latitude = math.radians(float(second.latitude))
    delta_latitude = second_latitude - first_latitude
    delta_longitude = math.radians(
        float(second.longitude) - float(first.longitude)
    )
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(first_latitude)
        * math.cos(second_latitude)
        * math.sin(delta_longitude / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(haversine)))


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
