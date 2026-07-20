from __future__ import annotations

from decimal import Decimal

import pytest

from atlaslens_api.phase3f.runpod import RunPodAPIError
from atlaslens_api.phase3f.safety import (
    MAX_RUNTIME_SECONDS,
    BudgetPolicy,
    Phase3FSafetyError,
    PodRecord,
    PodRequest,
    RunPodLease,
    SinglePodSession,
    redact_sensitive,
)


class _RunPodStub:
    def __init__(self) -> None:
        self.pods = [PodRecord("pre-existing", "another-run")]
        self.create_calls = 0
        self.terminate_calls: list[str] = []
        self.raise_after_create = False
        self.capacity_race_without_pod = False
        self.leave_after_terminate = False

    def list_pods(self) -> tuple[PodRecord, ...]:
        return tuple(self.pods)

    def create_pod(self, request: PodRequest) -> PodRecord:
        self.create_calls += 1
        if self.capacity_race_without_pod:
            raise RunPodAPIError("GPU_CAPACITY_ALLOCATION_REJECTED")
        pod = PodRecord("phase3f-pod", request.run_marker)
        self.pods.append(pod)
        if self.raise_after_create:
            raise RuntimeError("provider failed with https://secret.invalid/?token=unsafe")
        return pod

    def terminate_pod(self, pod_id: str) -> None:
        self.terminate_calls.append(pod_id)
        if not self.leave_after_terminate:
            self.pods = [item for item in self.pods if item.pod_id != pod_id]


def _request() -> PodRequest:
    return PodRequest(
        run_marker="phase3f-run-001",
        idempotency_key="phase3f-create-001",
        # The requested ceiling stays below the target even at the full wall time.
        hourly_cost_usd=Decimal("0.50"),
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
    )


def test_one_create_is_terminated_and_inventory_is_restored() -> None:
    client = _RunPodStub()
    session = SinglePodSession(client)

    def operation(lease: RunPodLease) -> str:
        assert lease.assert_within_limits(elapsed_seconds=1) < Decimal("0.01")
        return "complete"

    result = session.execute(_request(), operation)

    assert result.value == "complete"
    assert client.create_calls == 1
    assert client.terminate_calls == ["phase3f-pod"]
    assert client.pods == [PodRecord("pre-existing", "another-run")]
    assert result.audit.create_attempts == 1
    assert result.audit.termination_attempts == 1
    assert result.audit.termination_verified is True
    assert result.audit.before_inventory_sha256 == result.audit.after_inventory_sha256


def test_operation_failure_still_terminates_and_session_cannot_retry() -> None:
    client = _RunPodStub()
    session = SinglePodSession(client)

    with pytest.raises(RuntimeError, match="work failed"):
        session.execute(
            _request(),
            lambda _lease: (_ for _ in ()).throw(RuntimeError("work failed")),
        )

    assert client.create_calls == 1
    assert client.terminate_calls == ["phase3f-pod"]
    assert session.last_audit is not None
    assert session.last_audit.termination_verified is True
    with pytest.raises(Phase3FSafetyError, match="session_already_used"):
        session.execute(_request(), lambda _lease: None)
    assert client.create_calls == 1


def test_ambiguous_create_without_exact_id_is_not_marker_terminated() -> None:
    client = _RunPodStub()
    client.raise_after_create = True
    session = SinglePodSession(client)

    with pytest.raises(Phase3FSafetyError, match="termination_not_verified"):
        session.execute(_request(), lambda _lease: None)

    assert client.create_calls == 1
    assert client.terminate_calls == []
    assert client.pods == [
        PodRecord("pre-existing", "another-run"),
        PodRecord("phase3f-pod", "phase3f-run-001"),
    ]
    assert session.last_audit is not None
    assert session.last_audit.termination_verified is False


def test_capacity_race_has_zero_retry_and_restores_empty_inventory() -> None:
    client = _RunPodStub()
    client.pods = []
    client.capacity_race_without_pod = True
    session = SinglePodSession(client)

    with pytest.raises(Phase3FSafetyError, match="GPU_CAPACITY_RACE_NO_POD"):
        session.execute(_request(), lambda _lease: None)

    assert client.create_calls == 1
    assert client.terminate_calls == []
    assert client.pods == []
    assert session.last_audit is not None
    assert session.last_audit.create_attempts == 1
    assert session.last_audit.termination_attempts == 0
    assert session.last_audit.termination_verified is True
    assert session.last_audit.before_count == session.last_audit.after_count == 0
    assert session.last_audit.before_inventory_sha256 == session.last_audit.after_inventory_sha256


def test_unremoved_pod_is_reported_even_when_work_succeeds() -> None:
    client = _RunPodStub()
    client.leave_after_terminate = True
    session = SinglePodSession(client)

    with pytest.raises(Phase3FSafetyError, match="termination_not_verified"):
        session.execute(_request(), lambda _lease: "not-complete")

    assert client.create_calls == 1
    assert client.terminate_calls == ["phase3f-pod"]
    assert session.last_audit is not None
    assert session.last_audit.termination_verified is False


def test_budget_is_validated_before_any_provider_call() -> None:
    client = _RunPodStub()
    session = SinglePodSession(client)
    request = PodRequest(
        run_marker="phase3f-run-over-budget",
        idempotency_key="phase3f-create-over-budget",
        hourly_cost_usd=Decimal("0.51"),
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
    )

    with pytest.raises(Phase3FSafetyError, match="hourly_cost_exceeds_phase3f"):
        session.execute(request, lambda _lease: None)

    assert client.create_calls == 0
    assert session.last_audit is None


@pytest.mark.parametrize(
    ("elapsed_seconds", "provider_cost", "code"),
    [
        (MAX_RUNTIME_SECONDS, None, "runtime_limit_reached"),
        (1, "7.50", "soft_stop_budget_reached"),
        (1, "9", "termination_budget_reached"),
        (1, "10", "absolute_budget_reached"),
    ],
)
def test_runtime_budget_stops_are_independent(
    elapsed_seconds: int, provider_cost: str | None, code: str
) -> None:
    policy = BudgetPolicy()
    with pytest.raises(Phase3FSafetyError, match=code):
        policy.assert_within_run_limits(
            elapsed_seconds=elapsed_seconds,
            hourly_cost_usd=Decimal("0.50"),
            provider_cost_usd=provider_cost,
        )


def test_redaction_removes_headers_tokens_urls_and_unknown_values() -> None:
    redacted = redact_sensitive(
        {
            "Authorization": "Bearer abc123",
            "detail": "failed at https://api.runpod.io/?api_key=abc123",
            "nested": [{"cookie": "session=abc123"}],
            "error": RuntimeError("unsafe"),
        }
    )
    rendered = repr(redacted)
    assert "abc123" not in rendered
    assert "api.runpod.io" not in rendered
    assert "unsafe" not in rendered
    assert "<redacted>" in rendered
    assert "<redacted-url>" in rendered
