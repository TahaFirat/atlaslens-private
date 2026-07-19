"""Receipt-bound local operator controls for the Phase 3F RunPod pilot."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from atlaslens_api.phase3f.runpod import RunPodInventory

OPERATOR_RECEIPT_SCHEMA = "atlaslens-phase3f-operator-receipt-v1"
_MAX_RECEIPT_BYTES = 64 * 1024
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_RESOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,190}$")
_STAGES = frozenset({"preflight", "running", "failed", "terminated"})


class Phase3FOperatorError(RuntimeError):
    """Stable operator failure code without provider or credential detail."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Phase3FOperatorError(code)


def _decimal(value: object, code: str) -> Decimal:
    _require(not isinstance(value, bool), code)
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise Phase3FOperatorError(code) from exc
    _require(parsed.is_finite(), code)
    return parsed


def _string(value: object, code: str) -> str:
    _require(isinstance(value, str), code)
    return cast(str, value)


@dataclass(frozen=True, slots=True)
class OperatorReceipt:
    run_id: str
    run_marker: str
    pod_id: str | None
    supervisor_pid: int
    stage: str
    started_at: str
    finished_at: str | None
    max_spend_usd: Decimal
    soft_stop_usd: Decimal
    hard_stop_usd: Decimal
    max_gpu_hourly_usd: Decimal
    max_wall_minutes: int
    cleanup_verified: bool

    def __post_init__(self) -> None:
        _require(bool(_RUN_ID.fullmatch(self.run_id)), "OPERATOR_RECEIPT_RUN_ID_INVALID")
        _require(
            self.run_marker == f"atlaslens-phase3f-{self.run_id}",
            "OPERATOR_RECEIPT_MARKER_INVALID",
        )
        if self.pod_id is not None:
            _require(
                bool(_RESOURCE_ID.fullmatch(self.pod_id)),
                "OPERATOR_RECEIPT_POD_ID_INVALID",
            )
        _require(
            isinstance(self.supervisor_pid, int)
            and not isinstance(self.supervisor_pid, bool)
            and self.supervisor_pid > 0,
            "OPERATOR_RECEIPT_PID_INVALID",
        )
        _require(self.stage in _STAGES, "OPERATOR_RECEIPT_STAGE_INVALID")
        _require(
            bool(self.started_at)
            and "\r" not in self.started_at
            and "\n" not in self.started_at,
            "OPERATOR_RECEIPT_TIME_INVALID",
        )
        if self.finished_at is not None:
            _require(
                bool(self.finished_at)
                and "\r" not in self.finished_at
                and "\n" not in self.finished_at,
                "OPERATOR_RECEIPT_TIME_INVALID",
            )
        maximum = _decimal(self.max_spend_usd, "OPERATOR_RECEIPT_BUDGET_INVALID")
        soft = _decimal(self.soft_stop_usd, "OPERATOR_RECEIPT_BUDGET_INVALID")
        hard = _decimal(self.hard_stop_usd, "OPERATOR_RECEIPT_BUDGET_INVALID")
        hourly = _decimal(self.max_gpu_hourly_usd, "OPERATOR_RECEIPT_BUDGET_INVALID")
        _require(
            Decimal("0") < soft < hard < maximum <= Decimal("10"),
            "OPERATOR_RECEIPT_BUDGET_INVALID",
        )
        _require(
            Decimal("0") < hourly <= Decimal("0.50"),
            "OPERATOR_RECEIPT_BUDGET_INVALID",
        )
        _require(
            isinstance(self.max_wall_minutes, int)
            and not isinstance(self.max_wall_minutes, bool)
            and 1 <= self.max_wall_minutes <= 345,
            "OPERATOR_RECEIPT_WALL_TIME_INVALID",
        )
        _require(
            not (self.stage == "preflight" and self.pod_id is not None),
            "OPERATOR_RECEIPT_STAGE_INVALID",
        )
        _require(
            not (self.stage == "running" and self.pod_id is None),
            "OPERATOR_RECEIPT_STAGE_INVALID",
        )
        _require(
            not self.cleanup_verified or self.stage in {"failed", "terminated"},
            "OPERATOR_RECEIPT_STAGE_INVALID",
        )
        object.__setattr__(self, "max_spend_usd", maximum)
        object.__setattr__(self, "soft_stop_usd", soft)
        object.__setattr__(self, "hard_stop_usd", hard)
        object.__setattr__(self, "max_gpu_hourly_usd", hourly)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": OPERATOR_RECEIPT_SCHEMA,
            "run_id": self.run_id,
            "run_marker": self.run_marker,
            "pod_id": self.pod_id,
            "pod_id_sha256": (
                None
                if self.pod_id is None
                else hashlib.sha256(self.pod_id.encode("utf-8")).hexdigest()
            ),
            "supervisor_pid": self.supervisor_pid,
            "stage": self.stage,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "max_spend_usd": str(self.max_spend_usd),
            "soft_stop_usd": str(self.soft_stop_usd),
            "hard_stop_usd": str(self.hard_stop_usd),
            "max_gpu_hourly_usd": str(self.max_gpu_hourly_usd),
            "max_wall_minutes": self.max_wall_minutes,
            "cleanup_verified": self.cleanup_verified,
            "secret_values_included": False,
        }

    @classmethod
    def from_dict(cls, value: object) -> OperatorReceipt:
        _require(isinstance(value, Mapping), "OPERATOR_RECEIPT_INVALID")
        row = cast(Mapping[str, object], value)
        expected = {
            "schema",
            "run_id",
            "run_marker",
            "pod_id",
            "pod_id_sha256",
            "supervisor_pid",
            "stage",
            "started_at",
            "finished_at",
            "max_spend_usd",
            "soft_stop_usd",
            "hard_stop_usd",
            "max_gpu_hourly_usd",
            "max_wall_minutes",
            "cleanup_verified",
            "secret_values_included",
        }
        _require(set(row) == expected, "OPERATOR_RECEIPT_INVALID")
        _require(row.get("schema") == OPERATOR_RECEIPT_SCHEMA, "OPERATOR_RECEIPT_INVALID")
        pod_value = row.get("pod_id")
        _require(pod_value is None or isinstance(pod_value, str), "OPERATOR_RECEIPT_POD_ID_INVALID")
        pod_id = cast(str | None, pod_value)
        expected_hash = (
            None
            if pod_id is None
            else hashlib.sha256(pod_id.encode("utf-8")).hexdigest()
        )
        _require(row.get("pod_id_sha256") == expected_hash, "OPERATOR_RECEIPT_POD_HASH_INVALID")
        pid = row.get("supervisor_pid")
        wall = row.get("max_wall_minutes")
        cleanup = row.get("cleanup_verified")
        _require(isinstance(pid, int) and not isinstance(pid, bool), "OPERATOR_RECEIPT_PID_INVALID")
        _require(
            isinstance(wall, int) and not isinstance(wall, bool),
            "OPERATOR_RECEIPT_WALL_TIME_INVALID",
        )
        _require(isinstance(cleanup, bool), "OPERATOR_RECEIPT_INVALID")
        _require(row.get("secret_values_included") is False, "OPERATOR_RECEIPT_SECRET_FLAG_INVALID")
        finished_value = row.get("finished_at")
        _require(
            finished_value is None or isinstance(finished_value, str),
            "OPERATOR_RECEIPT_TIME_INVALID",
        )
        return cls(
            run_id=_string(row.get("run_id"), "OPERATOR_RECEIPT_RUN_ID_INVALID"),
            run_marker=_string(row.get("run_marker"), "OPERATOR_RECEIPT_MARKER_INVALID"),
            pod_id=pod_id,
            supervisor_pid=cast(int, pid),
            stage=_string(row.get("stage"), "OPERATOR_RECEIPT_STAGE_INVALID"),
            started_at=_string(row.get("started_at"), "OPERATOR_RECEIPT_TIME_INVALID"),
            finished_at=cast(str | None, finished_value),
            max_spend_usd=_decimal(row.get("max_spend_usd"), "OPERATOR_RECEIPT_BUDGET_INVALID"),
            soft_stop_usd=_decimal(row.get("soft_stop_usd"), "OPERATOR_RECEIPT_BUDGET_INVALID"),
            hard_stop_usd=_decimal(row.get("hard_stop_usd"), "OPERATOR_RECEIPT_BUDGET_INVALID"),
            max_gpu_hourly_usd=_decimal(
                row.get("max_gpu_hourly_usd"),
                "OPERATOR_RECEIPT_BUDGET_INVALID",
            ),
            max_wall_minutes=cast(int, wall),
            cleanup_verified=cast(bool, cleanup),
        )

    def update_lifecycle(
        self,
        *,
        stage: str,
        pod_id: str | None = None,
        finished_at: str | None = None,
        cleanup_verified: bool = False,
    ) -> OperatorReceipt:
        return replace(
            self,
            stage=stage,
            pod_id=self.pod_id if pod_id is None else pod_id,
            finished_at=finished_at,
            cleanup_verified=cleanup_verified,
        )


def read_operator_receipt(path: Path) -> OperatorReceipt:
    _require(path.is_file() and not path.is_symlink(), "OPERATOR_RECEIPT_MISSING")
    try:
        size = path.stat().st_size
        _require(0 < size <= _MAX_RECEIPT_BYTES, "OPERATOR_RECEIPT_INVALID")
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase3FOperatorError("OPERATOR_RECEIPT_INVALID") from exc
    return OperatorReceipt.from_dict(payload)


def write_operator_receipt(path: Path, receipt: OperatorReceipt) -> None:
    _require(not path.is_symlink(), "OPERATOR_RECEIPT_PATH_INVALID")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(receipt.to_dict(), separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        temporary.write_bytes(payload)
        try:
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    except OSError as exc:
        raise Phase3FOperatorError("OPERATOR_RECEIPT_WRITE_FAILED") from exc


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def require_startable_receipt(
    path: Path,
    *,
    is_process_running: Callable[[int], bool] = process_is_running,
) -> None:
    if not path.exists():
        return
    receipt = read_operator_receipt(path)
    if receipt.cleanup_verified and receipt.stage in {"failed", "terminated"}:
        return
    if is_process_running(receipt.supervisor_pid):
        raise Phase3FOperatorError("OPERATOR_PROCESS_ALREADY_RUNNING")
    raise Phase3FOperatorError("STALE_OPERATOR_RECEIPT_REQUIRES_TERMINATE")


@dataclass(frozen=True, slots=True)
class OperatorInventoryStatus:
    receipt_stage: str
    receipt_bound_pod_present: bool
    pods: int
    endpoints: int
    network_volumes: int
    templates: int
    cleanup_verified: bool

    def to_public_dict(self, receipt: OperatorReceipt, *, action: str) -> dict[str, object]:
        return {
            "action": action,
            "run_id": receipt.run_id,
            "pod_id_sha256": (
                None
                if receipt.pod_id is None
                else hashlib.sha256(receipt.pod_id.encode("utf-8")).hexdigest()
            ),
            "receipt_stage": receipt.stage,
            "receipt_bound_pod_present": self.receipt_bound_pod_present,
            "pods": self.pods,
            "endpoints": self.endpoints,
            "network_volumes": self.network_volumes,
            "templates": self.templates,
            "cleanup_verified": self.cleanup_verified,
            "secret_values_included": False,
        }


class OperatorInventoryClient(Protocol):
    def inventory(self) -> RunPodInventory: ...

    def terminate_pod(self, pod_id: str) -> None: ...


def inspect_operator_inventory(
    receipt: OperatorReceipt,
    inventory: RunPodInventory,
) -> OperatorInventoryStatus:
    _require(not inventory.endpoint_ids, "ACTIVE_ENDPOINT_INVENTORY_NOT_ZERO")
    _require(not inventory.network_volume_ids, "NETWORK_VOLUME_INVENTORY_NOT_ZERO")
    _require(not inventory.template_ids, "TEMPLATE_INVENTORY_NOT_ZERO")
    if receipt.pod_id is None:
        _require(not inventory.pods, "OPERATOR_RECEIPT_POD_ID_MISSING")
        present = False
    else:
        matching = tuple(item for item in inventory.pods if item.pod_id == receipt.pod_id)
        _require(
            len(inventory.pods) == len(matching),
            "UNEXPECTED_POD_INVENTORY",
        )
        _require(len(matching) <= 1, "UNEXPECTED_POD_INVENTORY")
        if matching:
            _require(
                matching[0].run_marker == receipt.run_marker,
                "OPERATOR_RECEIPT_POD_MARKER_MISMATCH",
            )
        present = bool(matching)
    empty = (
        not inventory.pods
        and not inventory.endpoint_ids
        and not inventory.network_volume_ids
        and not inventory.template_ids
    )
    return OperatorInventoryStatus(
        receipt_stage=receipt.stage,
        receipt_bound_pod_present=present,
        pods=len(inventory.pods),
        endpoints=len(inventory.endpoint_ids),
        network_volumes=len(inventory.network_volume_ids),
        templates=len(inventory.template_ids),
        cleanup_verified=empty,
    )


def terminate_receipt_bound_pod(
    client: OperatorInventoryClient,
    receipt: OperatorReceipt,
    *,
    poll_attempts: int = 20,
    poll_seconds: float = 3.0,
    sleep: Callable[[float], None] = time.sleep,
) -> OperatorInventoryStatus:
    _require(1 <= poll_attempts <= 60, "TERMINATE_POLL_ATTEMPTS_INVALID")
    _require(0 <= poll_seconds <= 10, "TERMINATE_POLL_SECONDS_INVALID")
    status = inspect_operator_inventory(receipt, client.inventory())
    if not status.receipt_bound_pod_present:
        return status
    _require(receipt.pod_id is not None, "OPERATOR_RECEIPT_POD_ID_MISSING")
    client.terminate_pod(cast(str, receipt.pod_id))
    for attempt in range(poll_attempts):
        status = inspect_operator_inventory(receipt, client.inventory())
        if status.cleanup_verified:
            return status
        if attempt + 1 < poll_attempts:
            sleep(poll_seconds)
    raise Phase3FOperatorError("PHASE3F_TERMINATION_UNVERIFIED")


__all__ = [
    "OPERATOR_RECEIPT_SCHEMA",
    "OperatorInventoryClient",
    "OperatorInventoryStatus",
    "OperatorReceipt",
    "Phase3FOperatorError",
    "inspect_operator_inventory",
    "process_is_running",
    "read_operator_receipt",
    "require_startable_receipt",
    "terminate_receipt_bound_pod",
    "write_operator_receipt",
]
