from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SAFE_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,159}$")
_ALLOWED_CONFIGURATION_KEYS = frozenset(
    {
        "candidate_count_limit",
        "geographic_diversity_km",
        "global_grid_points",
        "grid_refinement_levels_km",
        "fusion_weights",
        "provider_trigger_thresholds",
        "reference_sampling_policy",
        "ocr_crop_thresholds",
        "model_routing_thresholds",
    }
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AggregateMetrics(_FrozenModel):
    split: Literal["development", "validation", "worldwide"]
    sample_count: int = Field(ge=0, le=1_000_000)
    country_accuracy: float | None = Field(default=None, ge=0, le=1)
    province_accuracy: float | None = Field(default=None, ge=0, le=1)
    city_accuracy: float | None = Field(default=None, ge=0, le=1)
    median_error_km: float | None = Field(default=None, ge=0)
    mean_error_km: float | None = Field(default=None, ge=0)
    recall_within_km: dict[Literal[25, 100, 250, 750], float] = Field(
        default_factory=dict, max_length=4
    )
    city_recall_at_k: dict[Literal[1, 3, 5], float] = Field(
        default_factory=dict, max_length=3
    )
    candidate_recall: float | None = Field(default=None, ge=0, le=1)
    provider_failure_rate: float = Field(default=0, ge=0, le=1)
    provider_latency_ms: dict[str, float] = Field(default_factory=dict, max_length=32)
    accuracy_claim_allowed: bool = False

    @field_validator("recall_within_km", "city_recall_at_k")
    @classmethod
    def bounded_ratios(cls, value: dict[int, float]) -> dict[int, float]:
        if any(not math.isfinite(item) or not 0 <= item <= 1 for item in value.values()):
            raise ValueError("recall metrics must be finite ratios")
        return value

    @field_validator("provider_latency_ms")
    @classmethod
    def bounded_latencies(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not key or len(key) > 80 for key in value):
            raise ValueError("provider latency keys are invalid")
        if any(not math.isfinite(item) or item < 0 for item in value.values()):
            raise ValueError("provider latencies must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def claims_need_enough_samples(self) -> AggregateMetrics:
        if self.accuracy_claim_allowed and self.sample_count < 100:
            raise ValueError("accuracy claims require at least 100 independent samples")
        return self


class ImprovementCandidate(_FrozenModel):
    iteration: int = Field(ge=1, le=20)
    code_revision: str = Field(pattern=_SAFE_REVISION.pattern)
    configuration_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    changes: dict[str, object] = Field(min_length=1, max_length=16)
    development: AggregateMetrics
    validation: AggregateMetrics
    worldwide: AggregateMetrics | None = None

    @model_validator(mode="after")
    def validate_splits_and_changes(self) -> ImprovementCandidate:
        if self.development.split != "development" or self.validation.split != "validation":
            raise ValueError("aggregate metrics are assigned to the wrong split")
        if self.worldwide is not None and self.worldwide.split != "worldwide":
            raise ValueError("worldwide metrics are assigned to the wrong split")
        if not set(self.changes) <= _ALLOWED_CONFIGURATION_KEYS:
            raise ValueError("iteration contains a forbidden configuration change")
        serialized = repr(self.changes).casefold()
        forbidden = ("target", "holdout", "ground_truth", "image_sha", "filename")
        if any(marker in serialized for marker in forbidden):
            raise ValueError("iteration contains target-specific configuration")
        return self


class ImprovementPolicy(_FrozenModel):
    version: Literal["phase6c-improvement-v1"] = "phase6c-improvement-v1"
    max_iterations: int = Field(default=8, ge=1, le=20)
    max_wall_clock_seconds: int = Field(default=14_400, ge=60, le=86_400)
    minimum_claim_samples: int = Field(default=100, ge=100, le=100_000)
    maximum_accuracy_regression: float = Field(default=0.02, ge=0, le=0.2)
    maximum_candidate_recall_regression: float = Field(default=0.01, ge=0, le=0.2)
    maximum_error_increase_ratio: float = Field(default=0.05, ge=0, le=1)
    maximum_failure_rate_increase: float = Field(default=0.02, ge=0, le=0.2)


class ImprovementDecision(_FrozenModel):
    iteration: int
    accepted: bool
    reason_codes: tuple[str, ...] = Field(min_length=1, max_length=16)
    decision_uses_holdout: Literal[False] = False
    holdout_may_run_after_decision: bool
    accuracy_claim_allowed: bool


class ControlledImprovementEvaluator:
    """Choose generic configurations from aggregate splits before holdout evaluation."""

    def __init__(self, policy: ImprovementPolicy | None = None) -> None:
        self.policy = policy or ImprovementPolicy()

    def evaluate(
        self,
        baseline: ImprovementCandidate,
        candidate: ImprovementCandidate,
    ) -> ImprovementDecision:
        if candidate.iteration > self.policy.max_iterations:
            return self._reject(candidate, "maximum_iterations_exceeded")
        if candidate.iteration <= baseline.iteration:
            return self._reject(candidate, "iteration_order_invalid")
        if (
            baseline.validation.sample_count != candidate.validation.sample_count
            or baseline.development.sample_count != candidate.development.sample_count
        ):
            return self._reject(candidate, "evaluation_denominator_changed")
        reasons = self._regressions(baseline.validation, candidate.validation)
        if baseline.worldwide is not None or candidate.worldwide is not None:
            if baseline.worldwide is None or candidate.worldwide is None:
                reasons.append("worldwide_evaluation_missing")
            elif baseline.worldwide.sample_count != candidate.worldwide.sample_count:
                reasons.append("worldwide_denominator_changed")
            else:
                reasons.extend(self._regressions(baseline.worldwide, candidate.worldwide))
        if reasons:
            return ImprovementDecision(
                iteration=candidate.iteration,
                accepted=False,
                reason_codes=tuple(dict.fromkeys(reasons)),
                holdout_may_run_after_decision=False,
                accuracy_claim_allowed=False,
            )
        claim_allowed = (
            candidate.validation.sample_count >= self.policy.minimum_claim_samples
            and candidate.validation.accuracy_claim_allowed
        )
        return ImprovementDecision(
            iteration=candidate.iteration,
            accepted=True,
            reason_codes=("aggregate_validation_passed",),
            holdout_may_run_after_decision=True,
            accuracy_claim_allowed=claim_allowed,
        )

    def evaluate_sequence(
        self, candidates: Sequence[ImprovementCandidate]
    ) -> tuple[ImprovementDecision, ...]:
        if len(candidates) < 2:
            raise ValueError("an improvement sequence requires a baseline and candidate")
        decisions: list[ImprovementDecision] = []
        accepted = candidates[0]
        for candidate in candidates[1 : self.policy.max_iterations + 1]:
            decision = self.evaluate(accepted, candidate)
            decisions.append(decision)
            if decision.accepted:
                accepted = candidate
        return tuple(decisions)

    def _regressions(
        self, baseline: AggregateMetrics, candidate: AggregateMetrics
    ) -> list[str]:
        reasons: list[str] = []
        for name in ("country_accuracy", "province_accuracy", "city_accuracy"):
            left = getattr(baseline, name)
            right = getattr(candidate, name)
            if (
                left is not None
                and right is not None
                and right < left - self.policy.maximum_accuracy_regression
            ):
                reasons.append(f"{candidate.split}_{name}_regressed")
        if (
            baseline.candidate_recall is not None
            and candidate.candidate_recall is not None
            and candidate.candidate_recall
            < baseline.candidate_recall - self.policy.maximum_candidate_recall_regression
        ):
            reasons.append(f"{candidate.split}_candidate_recall_regressed")
        for name in ("median_error_km", "mean_error_km"):
            left = getattr(baseline, name)
            right = getattr(candidate, name)
            if left is not None and right is not None and right > left * (
                1 + self.policy.maximum_error_increase_ratio
            ):
                reasons.append(f"{candidate.split}_{name}_regressed")
        if candidate.provider_failure_rate > (
            baseline.provider_failure_rate + self.policy.maximum_failure_rate_increase
        ):
            reasons.append(f"{candidate.split}_provider_failure_regressed")
        return reasons

    @staticmethod
    def _reject(candidate: ImprovementCandidate, reason: str) -> ImprovementDecision:
        return ImprovementDecision(
            iteration=candidate.iteration,
            accepted=False,
            reason_codes=(reason,),
            holdout_may_run_after_decision=False,
            accuracy_claim_allowed=False,
        )


def allowed_improvement_keys() -> frozenset[str]:
    return _ALLOWED_CONFIGURATION_KEYS


def candidate_from_mapping(value: Mapping[str, object]) -> ImprovementCandidate:
    return ImprovementCandidate.model_validate(value)
