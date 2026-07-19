from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from atlaslens_api.phase6b.openai_assist.config import OpenAIGeoReviewConfig
from atlaslens_api.phase6b.openai_assist.models import HardCaseSignals, ShortCode


class TriggerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["call", "skip", "use_cache"]
    reasons: tuple[ShortCode, ...] = Field(min_length=1, max_length=8)


class OpenAIHardCaseTriggerPolicy:
    def __init__(self, config: OpenAIGeoReviewConfig) -> None:
        self._config = config

    def evaluate(
        self,
        signals: HardCaseSignals,
        *,
        key_configured: bool,
        cloud_consent: bool,
        cache_hit: bool = False,
        budget_available: bool = True,
        daily_limit_available: bool = True,
        image_valid: bool = True,
        cancelled: bool = False,
    ) -> TriggerDecision:
        if not self._config.enabled:
            return TriggerDecision(action="skip", reasons=("provider_disabled",))
        if not key_configured:
            return TriggerDecision(action="skip", reasons=("missing_api_key",))
        if not cloud_consent:
            return TriggerDecision(action="skip", reasons=("cloud_consent_denied",))
        if not image_valid:
            return TriggerDecision(action="skip", reasons=("invalid_image",))
        if cancelled:
            return TriggerDecision(action="skip", reasons=("analysis_cancelled",))
        if cache_hit:
            return TriggerDecision(action="use_cache", reasons=("cache_hit",))
        if signals.local_confidence_label in {"high", "very_high"}:
            return TriggerDecision(action="skip", reasons=("local_evidence_sufficient",))
        if signals.strong_ocr_and_independent_consensus:
            return TriggerDecision(action="skip", reasons=("local_consensus_with_ocr",))

        triggers: list[str] = []
        if signals.local_confidence_label == "low":
            triggers.append("low_local_confidence")
        if (
            signals.top_candidate_margin is not None
            and signals.top_candidate_margin < self._config.low_margin_threshold
        ):
            triggers.append("small_candidate_margin")
        if signals.strong_model_disagreement:
            triggers.append("strong_model_disagreement")
        if signals.maximum_model_separation_km >= self._config.large_separation_km:
            triggers.append("large_model_separation")
        if signals.strong_ocr_conflict:
            triggers.append("strong_ocr_conflict")
        if signals.no_usable_candidate:
            triggers.append("no_usable_candidate")
        if signals.candidate_dispersion_km >= self._config.extreme_dispersion_km:
            triggers.append("extreme_candidate_dispersion")
        if signals.user_explicit_review:
            triggers.append("user_explicit_review")
        if not triggers:
            return TriggerDecision(action="skip", reasons=("no_hard_case_trigger",))
        if not budget_available:
            return TriggerDecision(action="skip", reasons=("monthly_budget_exhausted",))
        if not daily_limit_available:
            return TriggerDecision(action="skip", reasons=("daily_call_limit_reached",))
        return TriggerDecision(action="call", reasons=tuple(triggers))
