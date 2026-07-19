from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlaslens_api.phase6c.improvement import (
    AggregateMetrics,
    ControlledImprovementEvaluator,
    ImprovementCandidate,
    ImprovementPolicy,
)


def metrics(split: str, *, city: float = 0.5, recall: float = 0.6) -> AggregateMetrics:
    return AggregateMetrics.model_validate(
        {
            "split": split,
            "sample_count": 120,
            "country_accuracy": 0.8,
            "province_accuracy": 0.6,
            "city_accuracy": city,
            "median_error_km": 80,
            "mean_error_km": 140,
            "candidate_recall": recall,
            "provider_failure_rate": 0.02,
            "accuracy_claim_allowed": True,
        }
    )


def candidate(iteration: int, *, city: float = 0.5, recall: float = 0.6) -> ImprovementCandidate:
    return ImprovementCandidate(
        iteration=iteration,
        code_revision=f"revision-{iteration}",
        configuration_fingerprint=f"{iteration:064x}",
        changes={"global_grid_points": 2048 + iteration},
        development=metrics("development", city=city, recall=recall),
        validation=metrics("validation", city=city, recall=recall),
        worldwide=metrics("worldwide", city=city, recall=recall),
    )


def test_aggregate_improvement_is_chosen_before_holdout_can_run() -> None:
    decision = ControlledImprovementEvaluator().evaluate(
        candidate(1), candidate(2, city=0.52, recall=0.7)
    )

    assert decision.accepted is True
    assert decision.decision_uses_holdout is False
    assert decision.holdout_may_run_after_decision is True


def test_target_only_or_aggregate_regressing_configuration_is_rejected() -> None:
    decision = ControlledImprovementEvaluator().evaluate(
        candidate(1), candidate(2, city=0.4, recall=0.5)
    )

    assert decision.accepted is False
    assert decision.holdout_may_run_after_decision is False
    assert "validation_city_accuracy_regressed" in decision.reason_codes


@pytest.mark.parametrize(
    "changes",
    (
        {"target_filename": "image.jpg"},
        {"fusion_weights": {"city": "target_city_label"}},
        {"image_sha": "0" * 64},
    ),
)
def test_target_specific_configuration_is_forbidden(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ImprovementCandidate(
            iteration=1,
            code_revision="revision",
            configuration_fingerprint="0" * 64,
            changes=changes,
            development=metrics("development"),
            validation=metrics("validation"),
        )


def test_maximum_iteration_count_is_enforced() -> None:
    decision = ControlledImprovementEvaluator(ImprovementPolicy(max_iterations=2)).evaluate(
        candidate(1), candidate(3, city=0.6, recall=0.7)
    )

    assert decision.accepted is False
    assert decision.reason_codes == ("maximum_iterations_exceeded",)


def test_small_evaluation_cannot_make_an_accuracy_claim() -> None:
    with pytest.raises(ValidationError):
        AggregateMetrics(split="validation", sample_count=1, accuracy_claim_allowed=True)
