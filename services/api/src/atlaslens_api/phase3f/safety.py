"""Budget, inventory, and teardown guardrails for the bounded Phase 3F run."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol, TypeVar, cast

TARGET_BUDGET_USD = Decimal("6")
SOFT_STOP_BUDGET_USD = Decimal("7.50")
TERMINATE_BUDGET_USD = Decimal("9")
ABSOLUTE_BUDGET_USD = Decimal("10")
MAX_HOURLY_COST_USD = Decimal("0.50")
MAX_RUNTIME_SECONDS = 5 * 60 * 60 + 45 * 60

_SAFE_RESOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,190}$")
_SAFE_MARKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_GPU_TYPE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._():+-]{0,190}$")
_SENSITIVE_KEY = re.compile(
    r"(?:authorization|cookie|credential|password|secret|token|api[-_]?key)", re.I
)
_URL = re.compile(r"https?://[^\s]+", re.I)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


class Phase3FSafetyError(RuntimeError):
    """A stable error code that never includes provider responses or credentials."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Phase3FSafetyError(code)


def _decimal(value: Decimal | int | str, code: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise Phase3FSafetyError(code) from exc
    _require(parsed.is_finite(), code)
    return parsed


def redact_sensitive(value: object) -> object:
    """Recursively redact common credentials and URLs before audit serialization."""

    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if _SENSITIVE_KEY.search(str(key))
                else redact_sensitive(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        return _URL.sub("<redacted-url>", _BEARER.sub("Bearer <redacted>", value))
    if isinstance(value, Decimal):
        return str(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    return f"<{type(value).__name__}>"


@dataclass(frozen=True, slots=True)
class BudgetProjection:
    hourly_cost_usd: Decimal
    requested_seconds: int
    committed_cost_usd: Decimal
    projected_total_usd: Decimal


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """The pre-declared Phase 3F ceiling; callers may only make it stricter."""

    target_usd: Decimal = TARGET_BUDGET_USD
    soft_stop_usd: Decimal = SOFT_STOP_BUDGET_USD
    terminate_usd: Decimal = TERMINATE_BUDGET_USD
    absolute_usd: Decimal = ABSOLUTE_BUDGET_USD
    max_hourly_cost_usd: Decimal = MAX_HOURLY_COST_USD
    max_runtime_seconds: int = MAX_RUNTIME_SECONDS

    def __post_init__(self) -> None:
        target = _decimal(self.target_usd, "budget_target_invalid")
        soft_stop = _decimal(self.soft_stop_usd, "budget_soft_stop_invalid")
        terminate = _decimal(self.terminate_usd, "budget_terminate_invalid")
        absolute = _decimal(self.absolute_usd, "budget_absolute_invalid")
        hourly = _decimal(self.max_hourly_cost_usd, "hourly_cost_invalid")
        _require(Decimal("0") < target <= TARGET_BUDGET_USD, "budget_target_exceeds_phase3f")
        _require(
            target < soft_stop <= SOFT_STOP_BUDGET_USD,
            "budget_soft_stop_exceeds_phase3f",
        )
        _require(
            soft_stop < terminate <= TERMINATE_BUDGET_USD,
            "budget_terminate_exceeds_phase3f",
        )
        _require(
            terminate < absolute <= ABSOLUTE_BUDGET_USD,
            "budget_absolute_exceeds_phase3f",
        )
        _require(
            Decimal("0") < hourly <= MAX_HOURLY_COST_USD,
            "hourly_cost_exceeds_phase3f",
        )
        _require(
            0 < self.max_runtime_seconds <= MAX_RUNTIME_SECONDS,
            "runtime_exceeds_phase3f",
        )
        object.__setattr__(self, "target_usd", target)
        object.__setattr__(self, "soft_stop_usd", soft_stop)
        object.__setattr__(self, "terminate_usd", terminate)
        object.__setattr__(self, "absolute_usd", absolute)
        object.__setattr__(self, "max_hourly_cost_usd", hourly)

    def project(
        self,
        *,
        hourly_cost_usd: Decimal | int | str,
        requested_seconds: int,
        committed_cost_usd: Decimal | int | str = Decimal("0"),
    ) -> BudgetProjection:
        hourly = _decimal(hourly_cost_usd, "hourly_cost_invalid")
        committed = _decimal(committed_cost_usd, "committed_cost_invalid")
        _require(Decimal("0") < hourly <= self.max_hourly_cost_usd, "hourly_cost_exceeds_phase3f")
        _require(
            0 < requested_seconds <= self.max_runtime_seconds,
            "runtime_exceeds_phase3f",
        )
        _require(Decimal("0") <= committed < self.terminate_usd, "budget_already_exhausted")
        projected = committed + hourly * Decimal(requested_seconds) / Decimal(3600)
        _require(projected <= self.target_usd, "projected_cost_exceeds_target")
        return BudgetProjection(hourly, requested_seconds, committed, projected)

    def assert_within_run_limits(
        self,
        *,
        elapsed_seconds: int,
        hourly_cost_usd: Decimal | int | str,
        provider_cost_usd: Decimal | int | str | None = None,
    ) -> Decimal:
        _require(elapsed_seconds >= 0, "elapsed_time_invalid")
        hourly = _decimal(hourly_cost_usd, "hourly_cost_invalid")
        _require(Decimal("0") < hourly <= self.max_hourly_cost_usd, "hourly_cost_exceeds_phase3f")
        estimated = hourly * Decimal(elapsed_seconds) / Decimal(3600)
        observed = (
            Decimal("0")
            if provider_cost_usd is None
            else _decimal(provider_cost_usd, "provider_cost_invalid")
        )
        _require(observed >= 0, "provider_cost_invalid")
        effective = max(estimated, observed)
        _require(effective < self.absolute_usd, "absolute_budget_reached")
        _require(effective < self.terminate_usd, "termination_budget_reached")
        _require(effective < self.soft_stop_usd, "soft_stop_budget_reached")
        _require(elapsed_seconds < self.max_runtime_seconds, "runtime_limit_reached")
        return effective


@dataclass(frozen=True, slots=True)
class PodRecord:
    pod_id: str
    run_marker: str | None = None

    def __post_init__(self) -> None:
        _require(bool(_SAFE_RESOURCE_ID.fullmatch(self.pod_id)), "pod_id_invalid")
        if self.run_marker is not None:
            _require(bool(_SAFE_MARKER.fullmatch(self.run_marker)), "run_marker_invalid")


@dataclass(frozen=True, slots=True)
class PodInventory:
    records: tuple[PodRecord, ...]

    @classmethod
    def capture(cls, records: Sequence[PodRecord]) -> PodInventory:
        ordered = tuple(sorted(records, key=lambda item: item.pod_id))
        _require(
            len({item.pod_id for item in ordered}) == len(ordered),
            "duplicate_pod_inventory",
        )
        return cls(ordered)

    def for_marker(self, marker: str) -> tuple[PodRecord, ...]:
        return tuple(item for item in self.records if item.run_marker == marker)

    @property
    def sha256(self) -> str:
        payload = [
            {"pod_id": item.pod_id, "run_marker": item.run_marker}
            for item in self.records
        ]
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PodRequest:
    run_marker: str
    idempotency_key: str
    hourly_cost_usd: Decimal
    max_runtime_seconds: int
    gpu_type_id: str | None = None
    gpu_count: int = 1
    interruptible: bool = False
    network_volume_id: str | None = None
    public_ports: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(_SAFE_MARKER.fullmatch(self.run_marker)), "run_marker_invalid")
        _require(bool(_SAFE_MARKER.fullmatch(self.idempotency_key)), "idempotency_key_invalid")
        hourly = _decimal(self.hourly_cost_usd, "hourly_cost_invalid")
        if self.gpu_type_id is not None:
            _require(
                bool(_SAFE_GPU_TYPE_ID.fullmatch(self.gpu_type_id)),
                "gpu_type_id_invalid",
            )
        _require(self.gpu_count == 1, "gpu_count_must_be_one")
        _require(not self.interruptible, "interruptible_gpu_forbidden")
        _require(self.network_volume_id is None, "network_volume_forbidden")
        _require(
            self.public_ports in {(), (22,)},
            "non_ssh_public_ports_forbidden",
        )
        object.__setattr__(self, "hourly_cost_usd", hourly)


class RunPodClient(Protocol):
    """Minimal provider seam. Implementations own authentication and HTTP details."""

    def list_pods(self) -> Sequence[PodRecord]: ...

    def create_pod(self, request: PodRequest) -> PodRecord: ...

    def terminate_pod(self, pod_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class RunPodLease:
    pod: PodRecord
    request: PodRequest
    policy: BudgetPolicy

    def assert_within_limits(
        self,
        *,
        elapsed_seconds: int,
        provider_cost_usd: Decimal | int | str | None = None,
    ) -> Decimal:
        return self.policy.assert_within_run_limits(
            elapsed_seconds=elapsed_seconds,
            hourly_cost_usd=self.request.hourly_cost_usd,
            provider_cost_usd=provider_cost_usd,
        )


@dataclass(frozen=True, slots=True)
class RunPodAudit:
    before_count: int
    after_count: int | None
    before_inventory_sha256: str
    after_inventory_sha256: str | None
    create_attempts: int
    termination_attempts: int
    termination_verified: bool

    def to_dict(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            redact_sensitive(
                {
                    "schema_version": "atlaslens-phase3f-runpod-audit-v1",
                    "before_count": self.before_count,
                    "after_count": self.after_count,
                    "before_inventory_sha256": self.before_inventory_sha256,
                    "after_inventory_sha256": self.after_inventory_sha256,
                    "create_attempts": self.create_attempts,
                    "termination_attempts": self.termination_attempts,
                    "termination_verified": self.termination_verified,
                }
            ),
        )


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RunPodExecution[T]:
    value: T
    audit: RunPodAudit


class SinglePodSession:
    """Create at most once and accept success only after verified teardown."""

    def __init__(
        self,
        client: RunPodClient,
        *,
        policy: BudgetPolicy | None = None,
        cleanup_poll_attempts: int = 1,
        cleanup_poll_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        _require(1 <= cleanup_poll_attempts <= 60, "cleanup_poll_attempts_invalid")
        _require(0.0 <= cleanup_poll_seconds <= 10.0, "cleanup_poll_seconds_invalid")
        self._client = client
        self._policy = policy or BudgetPolicy()
        self._cleanup_poll_attempts = cleanup_poll_attempts
        self._cleanup_poll_seconds = cleanup_poll_seconds
        self._sleep = sleep
        self._create_attempted = False
        self.last_audit: RunPodAudit | None = None

    def execute(
        self,
        request: PodRequest,
        operation: Callable[[RunPodLease], T],
        *,
        committed_cost_usd: Decimal | int | str = Decimal("0"),
    ) -> RunPodExecution[T]:
        _require(not self._create_attempted, "session_already_used")
        self._policy.project(
            hourly_cost_usd=request.hourly_cost_usd,
            requested_seconds=request.max_runtime_seconds,
            committed_cost_usd=committed_cost_usd,
        )
        before = PodInventory.capture(self._safe_list("initial_inventory_unavailable"))
        _require(not before.for_marker(request.run_marker), "run_marker_already_present")
        self._create_attempted = True
        created: PodRecord | None = None
        value: T | None = None
        operation_completed = False
        operation_error: BaseException | None = None
        try:
            created = self._client.create_pod(request)
            _require(created.run_marker == request.run_marker, "created_pod_marker_mismatch")
            current = PodInventory.capture(self._safe_list("created_inventory_unavailable"))
            expected_ids = {item.pod_id for item in before.records} | {created.pod_id}
            _require(
                {item.pod_id for item in current.records} == expected_ids,
                "unexpected_inventory_after_create",
            )
            _require(current.for_marker(request.run_marker) == (created,), "created_pod_not_unique")
            value = operation(RunPodLease(created, request, self._policy))
            operation_completed = True
        except BaseException as exc:
            failed_pod = getattr(exc, "pod_record", None)
            if isinstance(failed_pod, PodRecord):
                created = failed_pod
            operation_error = exc

        cleanup_error = self._cleanup(request.run_marker, before, created)
        if cleanup_error is not None:
            if operation_error is not None:
                raise cleanup_error from operation_error
            raise cleanup_error
        if operation_error is not None:
            if getattr(operation_error, "code", None) == "GPU_CAPACITY_ALLOCATION_REJECTED":
                audit = self.last_audit
                if audit is None:  # pragma: no cover - cleanup assigns it or raises first
                    raise Phase3FSafetyError("capacity_race_cleanup_unverified")
                if audit.termination_attempts == 0 and audit.termination_verified:
                    raise Phase3FSafetyError("GPU_CAPACITY_RACE_NO_POD") from operation_error
                raise Phase3FSafetyError(
                    "GPU_CAPACITY_RACE_AMBIGUOUS_POD_CLEANED"
                ) from operation_error
            raise operation_error
        _require(operation_completed and self.last_audit is not None, "operation_not_completed")
        audit = self.last_audit
        if audit is None:  # pragma: no cover - narrowed from the invariant above
            raise Phase3FSafetyError("operation_not_completed")
        return RunPodExecution(cast(T, value), audit)

    def _safe_list(self, code: str) -> Sequence[PodRecord]:
        try:
            return self._client.list_pods()
        except Exception as exc:
            raise Phase3FSafetyError(code) from exc

    def _cleanup(
        self,
        _marker: str,
        before: PodInventory,
        created: PodRecord | None,
    ) -> Phase3FSafetyError | None:
        termination_ids: set[str] = set()
        discovery_failed = False
        try:
            PodInventory.capture(self._safe_list("cleanup_inventory_unavailable"))
        except Phase3FSafetyError:
            discovery_failed = True
        if created is not None:
            termination_ids.add(created.pod_id)

        termination_call_failed = False
        for pod_id in sorted(termination_ids):
            try:
                self._client.terminate_pod(pod_id)
            except Exception:
                termination_call_failed = True

        after: PodInventory | None = None
        for attempt in range(self._cleanup_poll_attempts):
            with suppress(Phase3FSafetyError):
                after = PodInventory.capture(self._safe_list("final_inventory_unavailable"))
            if after == before:
                break
            if attempt + 1 < self._cleanup_poll_attempts:
                self._sleep(self._cleanup_poll_seconds)
        verified = after == before
        self.last_audit = RunPodAudit(
            before_count=len(before.records),
            after_count=None if after is None else len(after.records),
            before_inventory_sha256=before.sha256,
            after_inventory_sha256=None if after is None else after.sha256,
            create_attempts=1,
            termination_attempts=len(termination_ids),
            termination_verified=verified,
        )
        if after is None:
            return Phase3FSafetyError("final_inventory_unavailable")
        if not verified:
            return Phase3FSafetyError("termination_not_verified")
        if discovery_failed and created is None:
            return Phase3FSafetyError("cleanup_inventory_unavailable")
        if termination_call_failed and termination_ids:
            # A timed-out terminate call is harmless only when inventory proves removal.
            return None
        return None


__all__ = [
    "ABSOLUTE_BUDGET_USD",
    "BudgetPolicy",
    "BudgetProjection",
    "MAX_HOURLY_COST_USD",
    "MAX_RUNTIME_SECONDS",
    "Phase3FSafetyError",
    "PodInventory",
    "PodRecord",
    "PodRequest",
    "RunPodAudit",
    "RunPodClient",
    "RunPodExecution",
    "RunPodLease",
    "SinglePodSession",
    "SOFT_STOP_BUDGET_USD",
    "TARGET_BUDGET_USD",
    "TERMINATE_BUDGET_USD",
    "redact_sensitive",
]
