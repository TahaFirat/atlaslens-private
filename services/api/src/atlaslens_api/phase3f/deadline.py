"""Authoritative Phase 3F attempt and training deadline model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

SECONDS_PER_MINUTE: Final = 60
PROVISIONING_BUDGET_SECONDS: Final = 60 * SECONDS_PER_MINUTE
BOOTSTRAP_BUDGET_SECONDS: Final = 10 * SECONDS_PER_MINUTE
SALVAGE_RESERVE_SECONDS: Final = 5 * SECONDS_PER_MINUTE
MIN_TRAINING_SECONDS: Final = 30 * SECONDS_PER_MINUTE
MAX_TRAINING_SECONDS: Final = 4 * 60 * 60 + 30 * SECONDS_PER_MINUTE
MIN_SUPPORTED_TOTAL_MINUTES: Final = (
    PROVISIONING_BUDGET_SECONDS
    + BOOTSTRAP_BUDGET_SECONDS
    + SALVAGE_RESERVE_SECONDS
    + MIN_TRAINING_SECONDS
) // SECONDS_PER_MINUTE
MAX_SUPPORTED_TOTAL_MINUTES: Final = (
    PROVISIONING_BUDGET_SECONDS
    + BOOTSTRAP_BUDGET_SECONDS
    + SALVAGE_RESERVE_SECONDS
    + MAX_TRAINING_SECONDS
) // SECONDS_PER_MINUTE
RECOMMENDED_TOTAL_MINUTES: Final = MAX_SUPPORTED_TOTAL_MINUTES
CHILD_CLOCK_SKEW_TOLERANCE_SECONDS: Final = 60
_MAX_CHECKED_MINUTES: Final = (2**31 - 1) // SECONDS_PER_MINUTE


class TrainingDeadlineError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def checked_minutes_to_seconds(minutes: int) -> int:
    if (
        not isinstance(minutes, int)
        or isinstance(minutes, bool)
        or not 0 <= minutes <= _MAX_CHECKED_MINUTES
    ):
        raise TrainingDeadlineError("TRAINING_WALL_TIME_UNIT_INVALID")
    seconds = minutes * SECONDS_PER_MINUTE
    if seconds // SECONDS_PER_MINUTE != minutes:
        raise TrainingDeadlineError("TRAINING_WALL_TIME_UNIT_INVALID")
    return seconds


@dataclass(frozen=True, slots=True)
class TrainingDeadlinePlan:
    requested_total_minutes: int
    requested_total_seconds: int
    provisioning_budget_seconds: int
    bootstrap_budget_seconds: int
    salvage_reserve_seconds: int
    training_budget_seconds: int
    min_supported_minutes: int
    max_supported_minutes: int
    recommended_total_minutes: int
    computed_child_deadline_class: str
    local_blockers: tuple[str, ...]

    @property
    def ready_for_cloud(self) -> bool:
        return not self.local_blockers

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-phase3f-training-deadline-plan-v1",
            "action": "training-deadline-plan",
            "requested_total_minutes": self.requested_total_minutes,
            "requested_total_seconds": self.requested_total_seconds,
            "provisioning_budget_seconds": self.provisioning_budget_seconds,
            "bootstrap_budget_seconds": self.bootstrap_budget_seconds,
            "salvage_reserve_seconds": self.salvage_reserve_seconds,
            "training_budget_seconds": self.training_budget_seconds,
            "min_supported_minutes": self.min_supported_minutes,
            "max_supported_minutes": self.max_supported_minutes,
            "recommended_total_minutes": self.recommended_total_minutes,
            "computed_child_deadline_class": self.computed_child_deadline_class,
            "local_blockers": list(self.local_blockers),
            "ready_for_cloud": self.ready_for_cloud,
            "runpod_api_calls": 0,
            "create_attempts": 0,
            "cloud_mutations": 0,
            "network_calls": 0,
            "dataset_writes": 0,
            "secrets_included": False,
        }


def build_training_deadline_plan(requested_total_minutes: int) -> TrainingDeadlinePlan:
    blockers: list[str] = []
    try:
        total_seconds = checked_minutes_to_seconds(requested_total_minutes)
    except TrainingDeadlineError as exc:
        blockers.append(exc.code)
        total_seconds = 0
    if not (
        MIN_SUPPORTED_TOTAL_MINUTES
        <= requested_total_minutes
        <= MAX_SUPPORTED_TOTAL_MINUTES
    ):
        blockers.append("TRAINING_TOTAL_WALL_MINUTES_INVALID")
    reserved = (
        PROVISIONING_BUDGET_SECONDS
        + BOOTSTRAP_BUDGET_SECONDS
        + SALVAGE_RESERVE_SECONDS
    )
    training_seconds = min(MAX_TRAINING_SECONDS, max(0, total_seconds - reserved))
    if training_seconds < MIN_TRAINING_SECONDS:
        blockers.append("TRAINING_TIME_BUDGET_INSUFFICIENT")
    return TrainingDeadlinePlan(
        requested_total_minutes=requested_total_minutes,
        requested_total_seconds=total_seconds,
        provisioning_budget_seconds=PROVISIONING_BUDGET_SECONDS,
        bootstrap_budget_seconds=BOOTSTRAP_BUDGET_SECONDS,
        salvage_reserve_seconds=SALVAGE_RESERVE_SECONDS,
        training_budget_seconds=training_seconds,
        min_supported_minutes=MIN_SUPPORTED_TOTAL_MINUTES,
        max_supported_minutes=MAX_SUPPORTED_TOTAL_MINUTES,
        recommended_total_minutes=RECOMMENDED_TOTAL_MINUTES,
        computed_child_deadline_class="future_epoch_seconds",
        local_blockers=tuple(dict.fromkeys(blockers)),
    )


def require_training_deadline_plan(requested_total_minutes: int) -> TrainingDeadlinePlan:
    plan = build_training_deadline_plan(requested_total_minutes)
    if plan.local_blockers:
        raise TrainingDeadlineError(plan.local_blockers[0])
    return plan


@dataclass(frozen=True, slots=True)
class ChildDeadline:
    total_runtime_seconds: int
    elapsed_monotonic_seconds: int
    remaining_total_seconds: int
    training_budget_seconds: int
    deadline_epoch: int


def compute_child_deadline(
    *,
    total_runtime_seconds: int,
    started_monotonic: float,
    now_monotonic: float,
    now_epoch: float,
) -> ChildDeadline:
    if (
        not isinstance(total_runtime_seconds, int)
        or isinstance(total_runtime_seconds, bool)
        or total_runtime_seconds % SECONDS_PER_MINUTE != 0
    ):
        raise TrainingDeadlineError("TRAINING_WALL_TIME_UNIT_INVALID")
    requested_minutes = total_runtime_seconds // SECONDS_PER_MINUTE
    require_training_deadline_plan(requested_minutes)
    if not all(
        math.isfinite(value) for value in (started_monotonic, now_monotonic, now_epoch)
    ):
        raise TrainingDeadlineError("TRAINING_CLOCK_INVALID")
    if now_monotonic < started_monotonic:
        raise TrainingDeadlineError("TRAINING_MONOTONIC_CLOCK_INVALID")
    elapsed = math.ceil(now_monotonic - started_monotonic)
    remaining = total_runtime_seconds - elapsed
    training_seconds = min(
        MAX_TRAINING_SECONDS,
        remaining - SALVAGE_RESERVE_SECONDS,
    )
    if training_seconds < MIN_TRAINING_SECONDS:
        raise TrainingDeadlineError("REMOTE_TRAINING_TIME_REMAINING_INSUFFICIENT")
    epoch = math.floor(now_epoch) + training_seconds
    if epoch <= now_epoch:
        raise TrainingDeadlineError("TRAINING_DEADLINE_INVALID")
    return ChildDeadline(
        total_runtime_seconds=total_runtime_seconds,
        elapsed_monotonic_seconds=elapsed,
        remaining_total_seconds=remaining,
        training_budget_seconds=training_seconds,
        deadline_epoch=epoch,
    )


def epoch_deadline_to_monotonic(
    deadline_epoch: float,
    *,
    now_epoch: float,
    now_monotonic: float,
) -> float:
    if not all(math.isfinite(value) for value in (deadline_epoch, now_epoch, now_monotonic)):
        raise TrainingDeadlineError("TRAINING_DEADLINE_INVALID")
    remaining = deadline_epoch - now_epoch
    if not 0 < remaining <= MAX_TRAINING_SECONDS + CHILD_CLOCK_SKEW_TOLERANCE_SECONDS:
        raise TrainingDeadlineError("TRAINING_DEADLINE_INVALID")
    return now_monotonic + remaining


__all__ = [
    "BOOTSTRAP_BUDGET_SECONDS",
    "ChildDeadline",
    "MAX_SUPPORTED_TOTAL_MINUTES",
    "MAX_TRAINING_SECONDS",
    "MIN_SUPPORTED_TOTAL_MINUTES",
    "MIN_TRAINING_SECONDS",
    "PROVISIONING_BUDGET_SECONDS",
    "RECOMMENDED_TOTAL_MINUTES",
    "SALVAGE_RESERVE_SECONDS",
    "TrainingDeadlineError",
    "TrainingDeadlinePlan",
    "build_training_deadline_plan",
    "checked_minutes_to_seconds",
    "compute_child_deadline",
    "epoch_deadline_to_monotonic",
    "require_training_deadline_plan",
]
