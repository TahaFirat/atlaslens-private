from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from openai import AsyncOpenAI
from pydantic import ValidationError

from atlaslens_api.phase6b.openai_assist.cache_key import (
    build_openai_geo_cache_key,
    evidence_fingerprint,
)
from atlaslens_api.phase6b.openai_assist.config import OpenAIGeoReviewConfig
from atlaslens_api.phase6b.openai_assist.image import create_reduced_image_derivative
from atlaslens_api.phase6b.openai_assist.ledger import (
    CallFinalStatus,
    OpenAIBudgetSummary,
    OpenAIUsageLedger,
    SQLiteOpenAIUsageLedger,
)
from atlaslens_api.phase6b.openai_assist.models import (
    OpenAIGeoReviewCapability,
    OpenAIGeoReviewRequest,
    OpenAIGeoReviewResult,
    OpenAIReviewStructuredOutput,
    OpenAIReviewUsage,
)
from atlaslens_api.phase6b.openai_assist.policy import (
    OpenAIHardCaseTriggerPolicy,
    TriggerDecision,
)
from atlaslens_api.phase6b.openai_assist.prompt import (
    OPENAI_GEO_PROMPT_VERSION,
    OPENAI_GEO_SYSTEM_PROMPT,
    build_openai_geo_user_text,
)


class _ResponsesAPI(Protocol):
    async def parse(self, **kwargs: Any) -> Any: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesAPI


class _ReviewCancelled(Exception):
    pass


class OpenAIGeoReviewProvider:
    """Single-call, fail-closed cloud adjudicator for hard cases only."""

    def __init__(
        self,
        config: OpenAIGeoReviewConfig,
        ledger: OpenAIUsageLedger,
        *,
        api_key: str | None,
        client: _OpenAIClient | None = None,
    ) -> None:
        self._config = config
        self._ledger = ledger
        self._key_configured = bool(api_key and api_key.strip())
        if client is not None:
            self._client: _OpenAIClient | None = client
        elif self._key_configured:
            self._client = cast(
                _OpenAIClient,
                AsyncOpenAI(api_key=api_key, max_retries=0),
            )
        else:
            self._client = None
        self._policy = OpenAIHardCaseTriggerPolicy(config)

    def __repr__(self) -> str:
        return (
            "OpenAIGeoReviewProvider(model="
            f"{self._config.model!r}, enabled={self._config.enabled!r}, "
            f"key_configured={self._key_configured!r})"
        )

    def capability(self) -> OpenAIGeoReviewCapability:
        budget_available = False
        if self._config.enabled and self._key_configured and self._client is not None:
            try:
                summary = self.usage_summary()
                budget_available = (
                    summary.cloud_calls_today < self._config.daily_call_limit
                    and summary.remaining_configured_budget_usd
                    >= self._config.budget_reservation_usd
                )
            except (OSError, sqlite3.Error, ValueError):
                budget_available = False
        return OpenAIGeoReviewCapability(
            enabled=self._config.enabled,
            key_configured=self._key_configured and self._client is not None,
            model=self._config.model,
            budget_available=budget_available,
        )

    def usage_summary(self) -> OpenAIBudgetSummary:
        return self._ledger.budget_summary(monthly_budget_usd=self._config.monthly_budget_usd)

    async def review(
        self,
        request: OpenAIGeoReviewRequest,
        *,
        cancellation: asyncio.Event | None = None,
    ) -> OpenAIGeoReviewResult:
        cancelled = bool(cancellation and cancellation.is_set())
        initial = self._policy.evaluate(
            request.signals,
            key_configured=self._key_configured and self._client is not None,
            cloud_consent=request.cloud_consent,
            image_valid=request.image_valid,
            cancelled=cancelled,
        )
        if initial.action != "call":
            return self._not_called(initial)

        try:
            derivative = create_reduced_image_derivative(
                request.image_bytes,
                maximum_edge=self._config.maximum_image_edge,
                jpeg_quality=self._config.jpeg_quality,
            )
        except ValueError:
            return self._not_called(TriggerDecision(action="skip", reasons=("invalid_image",)))

        fingerprint = evidence_fingerprint(request.evidence)
        cache_key = build_openai_geo_cache_key(
            image_sha256=derivative.source_sha256,
            model=self._config.model,
            prompt_version=OPENAI_GEO_PROMPT_VERSION,
            candidate_evidence_fingerprint=fingerprint,
            image_detail=self._config.image_detail,
        )
        try:
            cached = self._ledger.get_cached(cache_key)
        except (OSError, sqlite3.Error, ValueError):
            return self._not_called(
                TriggerDecision(action="skip", reasons=("usage_ledger_unavailable",)),
                cache_key=cache_key,
            )
        if cached is not None:
            try:
                cached.validate_candidate_scope(
                    {item.candidate_id for item in request.evidence.candidates},
                    maximum_adjustment=self._config.maximum_candidate_adjustment,
                )
            except ValueError:
                cached = None
        if cached is not None:
            usage_recorded = True
            try:
                self._ledger.record_cache_hit(
                    analysis_id=str(request.analysis_id),
                    model=self._config.model,
                    prompt_version=OPENAI_GEO_PROMPT_VERSION,
                    image_detail=self._config.image_detail,
                    cost_estimate_version=self._config.pricing.version,
                )
            except (OSError, sqlite3.Error, ValueError):
                usage_recorded = False
            return OpenAIGeoReviewResult(
                status="completed",
                called=False,
                cache_hit=True,
                cache_key=cache_key,
                model=self._config.model,
                trigger_reasons=("cache_hit",),
                review=cached,
                estimated_cost_usd=Decimal("0"),
                cost_estimate_version=self._config.pricing.version,
                cost_basis="cache_hit",
                usage_recorded=usage_recorded,
                warnings=() if usage_recorded else ("usage_ledger_write_failed",),
            )

        try:
            user_text = build_openai_geo_user_text(request.evidence)
        except ValueError:
            return self._not_called(
                TriggerDecision(action="skip", reasons=("evidence_payload_too_large",)),
                cache_key=cache_key,
            )

        try:
            reservation = self._ledger.reserve_call(
                analysis_id=str(request.analysis_id),
                model=self._config.model,
                prompt_version=OPENAI_GEO_PROMPT_VERSION,
                image_detail=self._config.image_detail,
                reservation_cost_usd=self._config.budget_reservation_usd,
                cost_estimate_version=self._config.pricing.version,
                monthly_budget_usd=self._config.monthly_budget_usd,
                daily_call_limit=self._config.daily_call_limit,
                per_analysis_call_limit=self._config.per_analysis_call_limit,
            )
        except (OSError, sqlite3.Error, ValueError):
            return self._not_called(
                TriggerDecision(action="skip", reasons=("usage_ledger_unavailable",)),
                cache_key=cache_key,
            )
        if not reservation.allowed or reservation.reservation_id is None:
            return self._not_called(
                TriggerDecision(action="skip", reasons=(reservation.reason_code,)),
                cache_key=cache_key,
            )

        try:
            response = await self._call_once(user_text, derivative.data_url, cancellation)
        except _ReviewCancelled:
            return self._failed_after_call(
                reservation.reservation_id,
                initial,
                status="cancelled",
                reason_code="analysis_cancelled",
                cache_key=cache_key,
            )
        except TimeoutError:
            return self._failed_after_call(
                reservation.reservation_id,
                initial,
                status="timeout",
                reason_code="upstream_timeout",
                cache_key=cache_key,
            )
        except Exception as exc:  # SDK details are intentionally discarded.
            status, reason = self._classify_error(exc)
            return self._failed_after_call(
                reservation.reservation_id,
                initial,
                status=status,
                reason_code=reason,
                cache_key=cache_key,
            )

        usage = self._extract_usage(response)
        estimated_cost = self._config.pricing.estimate(usage)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None and self._contains_refusal(response):
            return self._failed_after_call(
                reservation.reservation_id,
                initial,
                status="refused",
                reason_code="model_refusal",
                usage=usage,
                estimated_cost=estimated_cost,
                cache_key=cache_key,
            )
        try:
            review = OpenAIReviewStructuredOutput.model_validate(parsed)
            review.validate_candidate_scope(
                {item.candidate_id for item in request.evidence.candidates},
                maximum_adjustment=self._config.maximum_candidate_adjustment,
            )
        except (ValidationError, TypeError, ValueError):
            return self._failed_after_call(
                reservation.reservation_id,
                initial,
                status="invalid_output",
                reason_code="invalid_structured_output",
                usage=usage,
                estimated_cost=estimated_cost,
                cache_key=cache_key,
            )

        usage_recorded = self._finalize_safely(
            reservation.reservation_id,
            status="success",
            usage=usage,
            estimated_cost=estimated_cost,
        )
        warnings: list[str] = []
        if not usage_recorded:
            warnings.append("usage_ledger_write_failed")
        try:
            self._ledger.put_cached(
                cache_key,
                review,
                model=self._config.model,
                prompt_version=OPENAI_GEO_PROMPT_VERSION,
                image_detail=self._config.image_detail,
                ttl_days=self._config.cache_ttl_days,
            )
        except (OSError, sqlite3.Error, ValueError):
            warnings.append("review_cache_write_failed")
        cost = estimated_cost or self._config.budget_reservation_usd
        basis: Literal["usage_based", "reservation_upper_bound"] = (
            "usage_based" if estimated_cost is not None else "reservation_upper_bound"
        )
        return OpenAIGeoReviewResult(
            status="completed",
            called=True,
            cache_hit=False,
            cache_key=cache_key,
            model=self._config.model,
            trigger_reasons=initial.reasons,
            review=review,
            usage=usage,
            estimated_cost_usd=cost,
            cost_estimate_version=self._config.pricing.version,
            cost_basis=basis,
            usage_recorded=usage_recorded,
            warnings=tuple(warnings),
        )

    async def _call_once(
        self,
        user_text: str,
        image_data_url: str,
        cancellation: asyncio.Event | None,
    ) -> object:
        if self._client is None:
            raise RuntimeError("provider client unavailable")
        call = self._client.responses.parse(
            model=self._config.model,
            input=[
                {"role": "system", "content": OPENAI_GEO_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_text,
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                            "detail": self._config.image_detail,
                        },
                    ],
                },
            ],
            text_format=OpenAIReviewStructuredOutput,
            reasoning={"effort": self._config.reasoning_effort},
            max_output_tokens=self._config.max_output_tokens,
            tools=[],
            tool_choice="none",
            store=False,
        )
        call_task = asyncio.create_task(call)
        cancel_task: asyncio.Task[bool] | None = None
        wait_for: set[asyncio.Task[Any]] = {call_task}
        if cancellation is not None:
            cancel_task = asyncio.create_task(cancellation.wait())
            wait_for.add(cancel_task)
        done, pending = await asyncio.wait(
            wait_for,
            timeout=self._config.timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if (
            cancel_task is not None
            and cancel_task in done
            and cancellation is not None
            and cancellation.is_set()
        ):
            call_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await call_task
            raise _ReviewCancelled
        if call_task not in done:
            call_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await call_task
            raise TimeoutError
        if cancel_task is not None:
            cancel_task.cancel()
        return await call_task

    def _not_called(
        self, decision: TriggerDecision, *, cache_key: str | None = None
    ) -> OpenAIGeoReviewResult:
        return OpenAIGeoReviewResult(
            status="skipped",
            reason_code=decision.reasons[0],
            called=False,
            cache_hit=False,
            cache_key=cache_key,
            model=self._config.model,
            trigger_reasons=decision.reasons,
        )

    def _failed_after_call(
        self,
        reservation_id: int,
        decision: TriggerDecision,
        *,
        status: CallFinalStatus,
        reason_code: str,
        usage: OpenAIReviewUsage | None = None,
        estimated_cost: Decimal | None = None,
        cache_key: str | None = None,
    ) -> OpenAIGeoReviewResult:
        usage_recorded = self._finalize_safely(
            reservation_id,
            status=status,
            usage=usage,
            estimated_cost=estimated_cost,
            failure_code=reason_code,
        )
        result_status: Literal["refused", "failed"] = "refused" if status == "refused" else "failed"
        cost = estimated_cost or self._config.budget_reservation_usd
        basis: Literal["usage_based", "reservation_upper_bound"] = (
            "usage_based" if estimated_cost is not None else "reservation_upper_bound"
        )
        return OpenAIGeoReviewResult(
            status=result_status,
            reason_code=reason_code,
            called=True,
            cache_hit=False,
            cache_key=cache_key,
            model=self._config.model,
            trigger_reasons=decision.reasons,
            usage=usage,
            estimated_cost_usd=cost,
            cost_estimate_version=self._config.pricing.version,
            cost_basis=basis,
            usage_recorded=usage_recorded,
            warnings=() if usage_recorded else ("usage_ledger_write_failed",),
        )

    def _finalize_safely(
        self,
        reservation_id: int,
        *,
        status: CallFinalStatus,
        usage: OpenAIReviewUsage | None,
        estimated_cost: Decimal | None,
        failure_code: str | None = None,
    ) -> bool:
        try:
            self._ledger.finalize_call(
                reservation_id,
                status=status,
                usage=usage,
                estimated_cost=estimated_cost,
                cost_estimate_version=self._config.pricing.version,
                failure_code=failure_code,
            )
        except (OSError, sqlite3.Error, ValueError):
            return False
        return True

    @staticmethod
    def _extract_usage(response: object) -> OpenAIReviewUsage | None:
        raw = getattr(response, "usage", None)
        if raw is None:
            return None

        def value(source: object, name: str) -> object:
            return source.get(name) if isinstance(source, dict) else getattr(source, name, None)

        input_details = value(raw, "input_tokens_details")
        cached_tokens = value(input_details, "cached_tokens") if input_details is not None else None
        try:
            return OpenAIReviewUsage.model_validate(
                {
                    "input_tokens": value(raw, "input_tokens"),
                    "cached_input_tokens": cached_tokens,
                    "output_tokens": value(raw, "output_tokens"),
                    "total_tokens": value(raw, "total_tokens"),
                }
            )
        except (ValidationError, TypeError, ValueError):
            return None

    @staticmethod
    def _contains_refusal(response: object) -> bool:
        output = getattr(response, "output", ())
        for item in output or ():
            content = (
                item.get("content", ()) if isinstance(item, dict) else getattr(item, "content", ())
            )
            for part in content or ():
                part_type = (
                    part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                )
                if part_type == "refusal":
                    return True
        return False

    @staticmethod
    def _classify_error(exc: Exception) -> tuple[CallFinalStatus, str]:
        name = type(exc).__name__.lower()
        status = getattr(exc, "status_code", None)
        if status == 429 or "ratelimit" in name or "quota" in name:
            return "quota_error", "quota_or_rate_limit"
        if status in {401, 403} or "authentication" in name or "permission" in name:
            return "failure", "upstream_auth"
        if "timeout" in name:
            return "timeout", "upstream_timeout"
        return "failure", "upstream_failure"


def build_openai_geo_review_provider(
    config: OpenAIGeoReviewConfig,
    *,
    ledger_path: Path,
    api_key: str | None,
) -> OpenAIGeoReviewProvider:
    """Production constructor; loading it never makes a network request."""

    return OpenAIGeoReviewProvider(
        config,
        SQLiteOpenAIUsageLedger(ledger_path),
        api_key=api_key,
    )
