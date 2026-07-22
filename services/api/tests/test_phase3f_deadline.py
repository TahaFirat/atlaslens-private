from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from atlaslens_api.phase3f.deadline import (
    MAX_SUPPORTED_TOTAL_MINUTES,
    MAX_TRAINING_SECONDS,
    MIN_SUPPORTED_TOTAL_MINUTES,
    MIN_TRAINING_SECONDS,
    TrainingDeadlineError,
    build_training_deadline_plan,
    compute_child_deadline,
    epoch_deadline_to_monotonic,
)

ROOT = Path(__file__).resolve().parents[3]


def _load_supervisor() -> ModuleType:
    path = ROOT / "scripts" / "phase3f" / "supervisor.py"
    spec = importlib.util.spec_from_file_location("phase3f_deadline_supervisor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_345_minute_parameter_reserves_overhead_and_caps_training() -> None:
    plan = build_training_deadline_plan(345)

    assert plan.ready_for_cloud is True
    assert plan.requested_total_seconds == 20_700
    assert plan.provisioning_budget_seconds == 3_600
    assert plan.bootstrap_budget_seconds == 600
    assert plan.salvage_reserve_seconds == 300
    assert plan.training_budget_seconds == 16_200
    assert plan.min_supported_minutes == 105
    assert plan.max_supported_minutes == 345


@pytest.mark.parametrize(
    ("minutes", "ready", "blocker"),
    (
        (MIN_SUPPORTED_TOTAL_MINUTES, True, None),
        (MAX_SUPPORTED_TOTAL_MINUTES, True, None),
        (
            MAX_SUPPORTED_TOTAL_MINUTES + 1,
            False,
            "TRAINING_TOTAL_WALL_MINUTES_INVALID",
        ),
    ),
)
def test_supported_total_minute_boundaries(
    minutes: int, ready: bool, blocker: str | None
) -> None:
    plan = build_training_deadline_plan(minutes)

    assert plan.ready_for_cloud is ready
    assert (plan.local_blockers[0] if plan.local_blockers else None) == blocker


def test_legacy_live_deadline_exceeded_training_predicate_after_bootstrap() -> None:
    legacy_child_seconds = 330 * 60
    observed_bootstrap_seconds = 210.9765151552856
    remaining_at_child = legacy_child_seconds - observed_bootstrap_seconds

    assert remaining_at_child > MAX_TRAINING_SECONDS + 60
    assert remaining_at_child - (MAX_TRAINING_SECONDS + 60) == pytest.approx(
        3329.0234848447144
    )


def test_past_or_excessive_epoch_deadline_is_rejected() -> None:
    with pytest.raises(TrainingDeadlineError, match="TRAINING_DEADLINE_INVALID"):
        epoch_deadline_to_monotonic(999.0, now_epoch=1_000.0, now_monotonic=50.0)
    with pytest.raises(TrainingDeadlineError, match="TRAINING_DEADLINE_INVALID"):
        epoch_deadline_to_monotonic(
            1_000.0 + MAX_TRAINING_SECONDS + 61,
            now_epoch=1_000.0,
            now_monotonic=50.0,
        )


def test_remaining_time_below_training_minimum_fails_before_child() -> None:
    total_seconds = MAX_SUPPORTED_TOTAL_MINUTES * 60
    elapsed = total_seconds - 300 - MIN_TRAINING_SECONDS + 1

    with pytest.raises(
        TrainingDeadlineError,
        match="REMOTE_TRAINING_TIME_REMAINING_INSUFFICIENT",
    ):
        compute_child_deadline(
            total_runtime_seconds=total_seconds,
            started_monotonic=10.0,
            now_monotonic=10.0 + elapsed,
            now_epoch=1_700_000_000.0,
        )


def test_provisioning_and_bootstrap_elapsed_are_deducted_monotonically() -> None:
    total_seconds = MAX_SUPPORTED_TOTAL_MINUTES * 60
    child = compute_child_deadline(
        total_runtime_seconds=total_seconds,
        started_monotonic=100.0,
        now_monotonic=100.0 + 70 * 60,
        now_epoch=1_700_000_000.25,
    )

    assert child.elapsed_monotonic_seconds == 4_200
    assert child.remaining_total_seconds == 16_500
    assert child.training_budget_seconds == MAX_TRAINING_SECONDS
    assert child.deadline_epoch == 1_700_016_200


def test_minutes_cannot_be_passed_as_seconds() -> None:
    with pytest.raises(TrainingDeadlineError, match="TRAINING_WALL_TIME_UNIT_INVALID"):
        compute_child_deadline(
            total_runtime_seconds=345,
            started_monotonic=0.0,
            now_monotonic=1.0,
            now_epoch=1_700_000_000.0,
        )


def test_monotonic_clock_drives_budget_despite_wall_clock_shift() -> None:
    total_seconds = MAX_SUPPORTED_TOTAL_MINUTES * 60
    earlier_wall = compute_child_deadline(
        total_runtime_seconds=total_seconds,
        started_monotonic=100.0,
        now_monotonic=700.0,
        now_epoch=900.0,
    )
    later_wall = compute_child_deadline(
        total_runtime_seconds=total_seconds,
        started_monotonic=100.0,
        now_monotonic=700.0,
        now_epoch=9_000.0,
    )
    monotonic_forward = compute_child_deadline(
        total_runtime_seconds=total_seconds,
        started_monotonic=100.0,
        now_monotonic=100.0 + 80 * 60,
        now_epoch=9_000.0,
    )

    assert earlier_wall.training_budget_seconds == later_wall.training_budget_seconds
    assert later_wall.deadline_epoch - earlier_wall.deadline_epoch == 8_100
    assert monotonic_forward.training_budget_seconds == 15_600
    with pytest.raises(
        TrainingDeadlineError, match="TRAINING_MONOTONIC_CLOCK_INVALID"
    ):
        compute_child_deadline(
            total_runtime_seconds=total_seconds,
            started_monotonic=100.0,
            now_monotonic=99.0,
            now_epoch=1_000.0,
        )


def test_epoch_is_converted_once_to_internal_monotonic_deadline() -> None:
    converted = epoch_deadline_to_monotonic(
        1_600.0,
        now_epoch=1_000.0,
        now_monotonic=50.0,
    )

    assert converted == 650.0


def test_invalid_deadline_plan_stops_before_token_or_create(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_supervisor()
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(
        module,
        "RunPodV1Client",
        lambda *_args, **_kwargs: pytest.fail("RunPod client must not be created"),
    )

    exit_code = module.main(
        [
            "--deadline-plan",
            "--max-wall-minutes",
            str(MAX_SUPPORTED_TOTAL_MINUTES + 1),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["local_blockers"] == ["TRAINING_TOTAL_WALL_MINUTES_INVALID"]
    assert output["create_attempts"] == 0
    assert output["runpod_api_calls"] == 0
    assert output["cloud_mutations"] == 0
