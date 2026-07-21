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

from atlaslens_api.phase3f.runpod import (
    ON_DEMAND_PRICE_TOLERANCE_USD,
    PodConnectivityProgressDiagnostic,
    PodGPUAttestationProgressDiagnostic,
    PodRentalAttestationDiagnostic,
    RunPodInventory,
)

OPERATOR_RECEIPT_SCHEMA = "atlaslens-phase3f-operator-receipt-v2"
_LEGACY_OPERATOR_RECEIPT_SCHEMA = "atlaslens-phase3f-operator-receipt-v1"
_MAX_RECEIPT_BYTES = 64 * 1024
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_RESOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,190}$")
_GPU_TYPE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._():+-]{0,190}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGES = frozenset({"preflight", "running", "failed", "terminated"})
_RENTAL_EVIDENCE = frozenset(
    {"explicit_interruptible_false", "request_and_on_demand_price_attested"}
)
_INTERRUPTIBLE_JSON_TYPES = frozenset({"missing", "null", "string", "boolean"})
_GPU_ATTESTATION_OUTCOMES = frozenset({"pending", "attested", "failed"})
_GPU_ATTESTATION_PATHS = frozenset(
    {"gpu.id", "machine.gpuTypeId", "machine.gpuType.id"}
)
_GPU_ATTESTATION_STATUSES = frozenset({"RUNNING", "EXITED", "TERMINATED"})
_GPU_ATTESTATION_COST = "graphql_uninterruptable_price_match"
_CONNECTIVITY_OUTCOMES = frozenset({"pending", "ready", "failed"})
_FAILURE_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


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
    pod_bound_at: str | None = None
    rental_evidence: str | None = None
    request_interruptible: bool | None = None
    selected_gpu_id: str | None = None
    selected_uninterruptable_price: Decimal | None = None
    create_http_status: int | None = None
    create_cost_per_hr: Decimal | None = None
    price_delta_usd: Decimal | None = None
    desired_status: str | None = None
    cloud_type: str | None = None
    create_interruptible_present: bool | None = None
    create_interruptible_json_type: str | None = None
    get_verification_http_status: int | None = None
    get_interruptible_present: bool | None = None
    get_interruptible_json_type: str | None = None
    pod_inventory_count: int | None = None
    explicit_false_source: str | None = None
    gpu_attestation_outcome: str | None = None
    gpu_attestation_failure_code: str | None = None
    normalized_gpu_path: str | None = None
    gpu_poll_count: int | None = None
    gpu_poll_elapsed_seconds: float | None = None
    final_desired_status: str | None = None
    expected_gpu_id: str | None = None
    observed_gpu_id: str | None = None
    observed_gpu_id_sha256: str | None = None
    gpu_count: int | None = None
    cost_attestation: str | None = None
    create_http_class: str | None = None
    allocation_attested_at: str | None = None
    connectivity_outcome: str | None = None
    connectivity_failure_code: str | None = None
    public_ip_present: bool | None = None
    tcp_port_present: bool | None = None
    connectivity_poll_count: int | None = None
    connectivity_elapsed_seconds: float | None = None
    ssh_ready: bool | None = None

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
        if self.pod_bound_at is not None:
            _require(
                bool(self.pod_bound_at)
                and "\r" not in self.pod_bound_at
                and "\n" not in self.pod_bound_at,
                "OPERATOR_RECEIPT_TIME_INVALID",
            )
        if self.allocation_attested_at is not None:
            _require(
                bool(self.allocation_attested_at)
                and "\r" not in self.allocation_attested_at
                and "\n" not in self.allocation_attested_at,
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
        if self.rental_evidence is not None:
            _require(
                self.pod_id is not None and self.pod_bound_at is not None,
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
            _require(self.rental_evidence in _RENTAL_EVIDENCE, "OPERATOR_RECEIPT_EVIDENCE_INVALID")
            _require(self.request_interruptible is False, "OPERATOR_RECEIPT_EVIDENCE_INVALID")
            _require(
                isinstance(self.selected_gpu_id, str)
                and bool(_GPU_TYPE_ID.fullmatch(self.selected_gpu_id)),
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
            selected_price = _decimal(
                self.selected_uninterruptable_price,
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
            create_price = _decimal(
                self.create_cost_per_hr,
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
            price_delta = _decimal(
                self.price_delta_usd,
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
            _require(
                Decimal("0") < selected_price <= hourly
                and Decimal("0") < create_price <= hourly
                and Decimal("0") <= price_delta <= ON_DEMAND_PRICE_TOLERANCE_USD
                and price_delta == abs(selected_price - create_price),
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
            _require(self.create_http_status == 201, "OPERATOR_RECEIPT_EVIDENCE_INVALID")
            _require(self.desired_status == "RUNNING", "OPERATOR_RECEIPT_EVIDENCE_INVALID")
            _require(
                self.cloud_type in {"SECURE", "COMMUNITY"},
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
            _require(
                isinstance(self.create_interruptible_present, bool)
                and self.create_interruptible_json_type in _INTERRUPTIBLE_JSON_TYPES,
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
            if self.rental_evidence == "request_and_on_demand_price_attested":
                _require(
                    self.create_interruptible_json_type in {"missing", "null", "string"}
                    and self.get_verification_http_status == 200
                    and isinstance(self.get_interruptible_present, bool)
                    and self.get_interruptible_json_type in {"missing", "null", "string"}
                    and self.pod_inventory_count == 1
                    and self.explicit_false_source is None,
                    "OPERATOR_RECEIPT_EVIDENCE_INVALID",
                )
            else:
                _require(
                    self.explicit_false_source in {"create_response", "authenticated_get"},
                    "OPERATOR_RECEIPT_EVIDENCE_INVALID",
                )
                if self.explicit_false_source == "create_response":
                    _require(
                        self.create_interruptible_present is True
                        and self.create_interruptible_json_type == "boolean"
                        and self.get_verification_http_status in {None, 200},
                        "OPERATOR_RECEIPT_EVIDENCE_INVALID",
                    )
                else:
                    _require(
                        self.get_verification_http_status == 200
                        and self.get_interruptible_present is True
                        and self.get_interruptible_json_type == "boolean",
                        "OPERATOR_RECEIPT_EVIDENCE_INVALID",
                    )
            object.__setattr__(self, "selected_uninterruptable_price", selected_price)
            object.__setattr__(self, "create_cost_per_hr", create_price)
            object.__setattr__(self, "price_delta_usd", price_delta)
        else:
            _require(
                all(
                    value is None
                    for value in (
                        self.request_interruptible,
                        self.selected_gpu_id,
                        self.selected_uninterruptable_price,
                        self.create_http_status,
                        self.create_cost_per_hr,
                        self.price_delta_usd,
                        self.desired_status,
                        self.cloud_type,
                        self.create_interruptible_present,
                        self.create_interruptible_json_type,
                        self.get_verification_http_status,
                        self.get_interruptible_present,
                        self.get_interruptible_json_type,
                        self.pod_inventory_count,
                        self.explicit_false_source,
                    )
                ),
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
        if self.gpu_attestation_outcome is not None:
            _require(
                self.pod_id is not None
                and self.pod_bound_at is not None
                and self.gpu_attestation_outcome in _GPU_ATTESTATION_OUTCOMES
                and self.create_http_class == "success_201",
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            _require(
                self.gpu_attestation_failure_code is None
                or bool(_FAILURE_CODE.fullmatch(self.gpu_attestation_failure_code)),
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            _require(
                (self.gpu_attestation_outcome == "failed")
                == (self.gpu_attestation_failure_code is not None),
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            _require(
                self.normalized_gpu_path is None
                or self.normalized_gpu_path in _GPU_ATTESTATION_PATHS,
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            _require(
                isinstance(self.gpu_poll_count, int)
                and not isinstance(self.gpu_poll_count, bool)
                and 0 <= self.gpu_poll_count <= 10_000,
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            _require(
                isinstance(self.gpu_poll_elapsed_seconds, int | float)
                and not isinstance(self.gpu_poll_elapsed_seconds, bool)
                and 0 <= self.gpu_poll_elapsed_seconds <= 180,
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            _require(
                self.final_desired_status is None
                or self.final_desired_status in _GPU_ATTESTATION_STATUSES,
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            for gpu_id in (self.expected_gpu_id, self.observed_gpu_id):
                _require(
                    gpu_id is None
                    or bool(_GPU_TYPE_ID.fullmatch(gpu_id)),
                    "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
                )
            _require(
                self.observed_gpu_id_sha256 is None
                or bool(_SHA256.fullmatch(self.observed_gpu_id_sha256)),
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            _require(
                self.observed_gpu_id is None
                or self.observed_gpu_id_sha256 is None,
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            _require(
                self.expected_gpu_id is not None,
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            _require(
                self.gpu_count is None
                or (
                    isinstance(self.gpu_count, int)
                    and not isinstance(self.gpu_count, bool)
                    and 0 <= self.gpu_count <= 16
                ),
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            _require(
                self.cost_attestation is None
                or self.cost_attestation == _GPU_ATTESTATION_COST,
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
            if self.gpu_attestation_outcome == "attested":
                _require(
                    self.rental_evidence is not None
                    and self.normalized_gpu_path is not None
                    and self.observed_gpu_id == self.expected_gpu_id
                    and self.gpu_count == 1
                    and self.final_desired_status == "RUNNING"
                    and self.cost_attestation == _GPU_ATTESTATION_COST,
                    "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
                )
        else:
            _require(
                all(
                    value is None
                    for value in (
                        self.gpu_attestation_failure_code,
                        self.normalized_gpu_path,
                        self.gpu_poll_count,
                        self.gpu_poll_elapsed_seconds,
                        self.final_desired_status,
                        self.expected_gpu_id,
                        self.observed_gpu_id,
                        self.observed_gpu_id_sha256,
                        self.gpu_count,
                        self.cost_attestation,
                        self.create_http_class,
                    )
                ),
                "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
            )
        if self.connectivity_outcome is not None:
            _require(
                self.pod_id is not None
                and self.pod_bound_at is not None
                and self.rental_evidence is not None
                and self.connectivity_outcome in _CONNECTIVITY_OUTCOMES,
                "OPERATOR_RECEIPT_CONNECTIVITY_INVALID",
            )
            _require(
                self.connectivity_failure_code is None
                or bool(_FAILURE_CODE.fullmatch(self.connectivity_failure_code)),
                "OPERATOR_RECEIPT_CONNECTIVITY_INVALID",
            )
            _require(
                (self.connectivity_outcome == "failed")
                == (self.connectivity_failure_code is not None),
                "OPERATOR_RECEIPT_CONNECTIVITY_INVALID",
            )
            _require(
                isinstance(self.public_ip_present, bool)
                and isinstance(self.tcp_port_present, bool)
                and isinstance(self.ssh_ready, bool),
                "OPERATOR_RECEIPT_CONNECTIVITY_INVALID",
            )
            _require(
                isinstance(self.connectivity_poll_count, int)
                and not isinstance(self.connectivity_poll_count, bool)
                and 0 <= self.connectivity_poll_count <= 10_000,
                "OPERATOR_RECEIPT_CONNECTIVITY_INVALID",
            )
            _require(
                isinstance(self.connectivity_elapsed_seconds, int | float)
                and not isinstance(self.connectivity_elapsed_seconds, bool)
                and 0 <= self.connectivity_elapsed_seconds <= 180,
                "OPERATOR_RECEIPT_CONNECTIVITY_INVALID",
            )
            if self.connectivity_outcome == "ready":
                _require(
                    self.public_ip_present is True
                    and self.tcp_port_present is True
                    and self.ssh_ready is True,
                    "OPERATOR_RECEIPT_CONNECTIVITY_INVALID",
                )
        else:
            _require(
                all(
                    value is None
                    for value in (
                        self.connectivity_failure_code,
                        self.public_ip_present,
                        self.tcp_port_present,
                        self.connectivity_poll_count,
                        self.connectivity_elapsed_seconds,
                        self.ssh_ready,
                    )
                ),
                "OPERATOR_RECEIPT_CONNECTIVITY_INVALID",
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
            "pod_bound_at": self.pod_bound_at,
            "max_spend_usd": str(self.max_spend_usd),
            "soft_stop_usd": str(self.soft_stop_usd),
            "hard_stop_usd": str(self.hard_stop_usd),
            "max_gpu_hourly_usd": str(self.max_gpu_hourly_usd),
            "max_wall_minutes": self.max_wall_minutes,
            "cleanup_verified": self.cleanup_verified,
            "rental_evidence": self.rental_evidence,
            "request_interruptible": self.request_interruptible,
            "selected_gpu_id": self.selected_gpu_id,
            "selected_uninterruptable_price": (
                None
                if self.selected_uninterruptable_price is None
                else str(self.selected_uninterruptable_price)
            ),
            "create_http_status": self.create_http_status,
            "create_cost_per_hr": (
                None if self.create_cost_per_hr is None else str(self.create_cost_per_hr)
            ),
            "price_delta_usd": (
                None if self.price_delta_usd is None else str(self.price_delta_usd)
            ),
            "desired_status": self.desired_status,
            "cloud_type": self.cloud_type,
            "create_interruptible_present": self.create_interruptible_present,
            "create_interruptible_json_type": self.create_interruptible_json_type,
            "get_verification_http_status": self.get_verification_http_status,
            "get_interruptible_present": self.get_interruptible_present,
            "get_interruptible_json_type": self.get_interruptible_json_type,
            "pod_inventory_count": self.pod_inventory_count,
            "explicit_false_source": self.explicit_false_source,
            "gpu_attestation_outcome": self.gpu_attestation_outcome,
            "gpu_attestation_failure_code": self.gpu_attestation_failure_code,
            "normalized_gpu_path": self.normalized_gpu_path,
            "gpu_poll_count": self.gpu_poll_count,
            "gpu_poll_elapsed_seconds": self.gpu_poll_elapsed_seconds,
            "final_desired_status": self.final_desired_status,
            "expected_gpu_id": self.expected_gpu_id,
            "observed_gpu_id": self.observed_gpu_id,
            "observed_gpu_id_sha256": self.observed_gpu_id_sha256,
            "gpu_count": self.gpu_count,
            "cost_attestation": self.cost_attestation,
            "create_http_class": self.create_http_class,
            "allocation_attested_at": self.allocation_attested_at,
            "connectivity_outcome": self.connectivity_outcome,
            "connectivity_failure_code": self.connectivity_failure_code,
            "public_ip_present": self.public_ip_present,
            "tcp_port_present": self.tcp_port_present,
            "connectivity_poll_count": self.connectivity_poll_count,
            "connectivity_elapsed_seconds": self.connectivity_elapsed_seconds,
            "ssh_ready": self.ssh_ready,
            "secret_values_included": False,
        }

    @classmethod
    def from_dict(cls, value: object) -> OperatorReceipt:
        _require(isinstance(value, Mapping), "OPERATOR_RECEIPT_INVALID")
        row = cast(Mapping[str, object], value)
        legacy_expected = {
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
        evidence_fields = {
            "pod_bound_at",
            "rental_evidence",
            "request_interruptible",
            "selected_gpu_id",
            "selected_uninterruptable_price",
            "create_http_status",
            "create_cost_per_hr",
            "price_delta_usd",
            "desired_status",
            "cloud_type",
            "create_interruptible_present",
            "create_interruptible_json_type",
            "get_verification_http_status",
            "get_interruptible_present",
            "get_interruptible_json_type",
            "pod_inventory_count",
            "explicit_false_source",
        }
        gpu_attestation_fields = {
            "gpu_attestation_outcome",
            "gpu_attestation_failure_code",
            "normalized_gpu_path",
            "gpu_poll_count",
            "gpu_poll_elapsed_seconds",
            "final_desired_status",
            "expected_gpu_id",
            "observed_gpu_id",
            "observed_gpu_id_sha256",
            "gpu_count",
            "cost_attestation",
            "create_http_class",
        }
        connectivity_fields = {
            "allocation_attested_at",
            "connectivity_outcome",
            "connectivity_failure_code",
            "public_ip_present",
            "tcp_port_present",
            "connectivity_poll_count",
            "connectivity_elapsed_seconds",
            "ssh_ready",
        }
        schema = row.get("schema")
        is_legacy = schema == _LEGACY_OPERATOR_RECEIPT_SCHEMA
        _require(
            (is_legacy and set(row) == legacy_expected)
            or (
                schema == OPERATOR_RECEIPT_SCHEMA
                and frozenset(row)
                in {
                    frozenset(legacy_expected | evidence_fields),
                    frozenset(
                        legacy_expected | evidence_fields | gpu_attestation_fields
                    ),
                    frozenset(
                        legacy_expected
                        | evidence_fields
                        | gpu_attestation_fields
                        | connectivity_fields
                    ),
                }
            ),
            "OPERATOR_RECEIPT_INVALID",
        )
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
        pod_bound_at = None if is_legacy else row.get("pod_bound_at")
        rental_evidence = None if is_legacy else row.get("rental_evidence")
        request_interruptible = None if is_legacy else row.get("request_interruptible")
        selected_gpu_id = None if is_legacy else row.get("selected_gpu_id")
        selected_price = None if is_legacy else row.get("selected_uninterruptable_price")
        create_status = None if is_legacy else row.get("create_http_status")
        create_price = None if is_legacy else row.get("create_cost_per_hr")
        price_delta = None if is_legacy else row.get("price_delta_usd")
        desired_status = None if is_legacy else row.get("desired_status")
        cloud_type = None if is_legacy else row.get("cloud_type")
        create_present = None if is_legacy else row.get("create_interruptible_present")
        create_type = None if is_legacy else row.get("create_interruptible_json_type")
        get_status = None if is_legacy else row.get("get_verification_http_status")
        get_present = None if is_legacy else row.get("get_interruptible_present")
        get_type = None if is_legacy else row.get("get_interruptible_json_type")
        inventory_count = None if is_legacy else row.get("pod_inventory_count")
        false_source = None if is_legacy else row.get("explicit_false_source")
        has_gpu_attestation = "gpu_attestation_outcome" in row
        gpu_attestation_outcome = (
            row.get("gpu_attestation_outcome") if has_gpu_attestation else None
        )
        gpu_attestation_failure_code = (
            row.get("gpu_attestation_failure_code") if has_gpu_attestation else None
        )
        normalized_gpu_path = row.get("normalized_gpu_path") if has_gpu_attestation else None
        gpu_poll_count = row.get("gpu_poll_count") if has_gpu_attestation else None
        gpu_poll_elapsed_seconds = (
            row.get("gpu_poll_elapsed_seconds") if has_gpu_attestation else None
        )
        final_desired_status = (
            row.get("final_desired_status") if has_gpu_attestation else None
        )
        expected_gpu_id = row.get("expected_gpu_id") if has_gpu_attestation else None
        observed_gpu_id = row.get("observed_gpu_id") if has_gpu_attestation else None
        observed_gpu_id_sha256 = (
            row.get("observed_gpu_id_sha256") if has_gpu_attestation else None
        )
        gpu_count = row.get("gpu_count") if has_gpu_attestation else None
        cost_attestation = row.get("cost_attestation") if has_gpu_attestation else None
        create_http_class = row.get("create_http_class") if has_gpu_attestation else None
        has_connectivity = "connectivity_outcome" in row
        allocation_attested_at = (
            row.get("allocation_attested_at") if has_connectivity else None
        )
        connectivity_outcome = row.get("connectivity_outcome") if has_connectivity else None
        connectivity_failure_code = (
            row.get("connectivity_failure_code") if has_connectivity else None
        )
        public_ip_present = row.get("public_ip_present") if has_connectivity else None
        tcp_port_present = row.get("tcp_port_present") if has_connectivity else None
        connectivity_poll_count = (
            row.get("connectivity_poll_count") if has_connectivity else None
        )
        connectivity_elapsed_seconds = (
            row.get("connectivity_elapsed_seconds") if has_connectivity else None
        )
        ssh_ready = row.get("ssh_ready") if has_connectivity else None
        for optional_string in (
            pod_bound_at,
            rental_evidence,
            selected_gpu_id,
            desired_status,
            cloud_type,
            create_type,
            get_type,
            false_source,
            gpu_attestation_outcome,
            gpu_attestation_failure_code,
            normalized_gpu_path,
            final_desired_status,
            expected_gpu_id,
            observed_gpu_id,
            observed_gpu_id_sha256,
            cost_attestation,
            create_http_class,
            allocation_attested_at,
            connectivity_outcome,
            connectivity_failure_code,
        ):
            _require(
                optional_string is None or isinstance(optional_string, str),
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
        for optional_boolean in (
            request_interruptible,
            create_present,
            get_present,
            public_ip_present,
            tcp_port_present,
            ssh_ready,
        ):
            _require(
                optional_boolean is None or isinstance(optional_boolean, bool),
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
        for optional_integer in (create_status, get_status, inventory_count):
            _require(
                optional_integer is None
                or (isinstance(optional_integer, int) and not isinstance(optional_integer, bool)),
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
        _require(
            gpu_poll_count is None
            or (isinstance(gpu_poll_count, int) and not isinstance(gpu_poll_count, bool)),
            "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
        )
        _require(
            gpu_poll_elapsed_seconds is None
            or (
                isinstance(gpu_poll_elapsed_seconds, int | float)
                and not isinstance(gpu_poll_elapsed_seconds, bool)
            ),
            "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
        )
        _require(
            gpu_count is None
            or (isinstance(gpu_count, int) and not isinstance(gpu_count, bool)),
            "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
        )
        _require(
            connectivity_poll_count is None
            or (
                isinstance(connectivity_poll_count, int)
                and not isinstance(connectivity_poll_count, bool)
            ),
            "OPERATOR_RECEIPT_CONNECTIVITY_INVALID",
        )
        _require(
            connectivity_elapsed_seconds is None
            or (
                isinstance(connectivity_elapsed_seconds, int | float)
                and not isinstance(connectivity_elapsed_seconds, bool)
            ),
            "OPERATOR_RECEIPT_CONNECTIVITY_INVALID",
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
            pod_bound_at=cast(str | None, pod_bound_at),
            rental_evidence=cast(str | None, rental_evidence),
            request_interruptible=cast(bool | None, request_interruptible),
            selected_gpu_id=cast(str | None, selected_gpu_id),
            selected_uninterruptable_price=(
                None
                if selected_price is None
                else _decimal(selected_price, "OPERATOR_RECEIPT_EVIDENCE_INVALID")
            ),
            create_http_status=cast(int | None, create_status),
            create_cost_per_hr=(
                None
                if create_price is None
                else _decimal(create_price, "OPERATOR_RECEIPT_EVIDENCE_INVALID")
            ),
            price_delta_usd=(
                None
                if price_delta is None
                else _decimal(price_delta, "OPERATOR_RECEIPT_EVIDENCE_INVALID")
            ),
            desired_status=cast(str | None, desired_status),
            cloud_type=cast(str | None, cloud_type),
            create_interruptible_present=cast(bool | None, create_present),
            create_interruptible_json_type=cast(str | None, create_type),
            get_verification_http_status=cast(int | None, get_status),
            get_interruptible_present=cast(bool | None, get_present),
            get_interruptible_json_type=cast(str | None, get_type),
            pod_inventory_count=cast(int | None, inventory_count),
            explicit_false_source=cast(str | None, false_source),
            gpu_attestation_outcome=cast(str | None, gpu_attestation_outcome),
            gpu_attestation_failure_code=cast(
                str | None, gpu_attestation_failure_code
            ),
            normalized_gpu_path=cast(str | None, normalized_gpu_path),
            gpu_poll_count=cast(int | None, gpu_poll_count),
            gpu_poll_elapsed_seconds=cast(
                float | None, gpu_poll_elapsed_seconds
            ),
            final_desired_status=cast(str | None, final_desired_status),
            expected_gpu_id=cast(str | None, expected_gpu_id),
            observed_gpu_id=cast(str | None, observed_gpu_id),
            observed_gpu_id_sha256=cast(str | None, observed_gpu_id_sha256),
            gpu_count=cast(int | None, gpu_count),
            cost_attestation=cast(str | None, cost_attestation),
            create_http_class=cast(str | None, create_http_class),
            allocation_attested_at=cast(str | None, allocation_attested_at),
            connectivity_outcome=cast(str | None, connectivity_outcome),
            connectivity_failure_code=cast(str | None, connectivity_failure_code),
            public_ip_present=cast(bool | None, public_ip_present),
            tcp_port_present=cast(bool | None, tcp_port_present),
            connectivity_poll_count=cast(int | None, connectivity_poll_count),
            connectivity_elapsed_seconds=cast(
                float | None, connectivity_elapsed_seconds
            ),
            ssh_ready=cast(bool | None, ssh_ready),
        )

    def update_lifecycle(
        self,
        *,
        stage: str,
        pod_id: str | None = None,
        pod_bound_at: str | None = None,
        finished_at: str | None = None,
        cleanup_verified: bool = False,
    ) -> OperatorReceipt:
        return replace(
            self,
            stage=stage,
            pod_id=self.pod_id if pod_id is None else pod_id,
            pod_bound_at=(
                self.pod_bound_at if pod_bound_at is None else pod_bound_at
            ),
            finished_at=finished_at,
            cleanup_verified=cleanup_verified,
        )

    def record_rental_attestation(
        self,
        attestation: PodRentalAttestationDiagnostic,
        *,
        allocation_attested_at: str | None = None,
    ) -> OperatorReceipt:
        _require(
            self.stage == "running"
            and self.pod_id is not None
            and self.pod_bound_at is not None,
            "OPERATOR_RECEIPT_STAGE_INVALID",
        )
        return replace(
            self,
            rental_evidence=attestation.evidence,
            request_interruptible=attestation.request_interruptible,
            selected_gpu_id=attestation.selected_gpu_id,
            selected_uninterruptable_price=attestation.selected_uninterruptable_price,
            create_http_status=attestation.create_http_status,
            create_cost_per_hr=attestation.create_cost_per_hr,
            price_delta_usd=attestation.price_delta_usd,
            desired_status=attestation.desired_status,
            cloud_type=attestation.cloud_type,
            create_interruptible_present=attestation.create_interruptible_present,
            create_interruptible_json_type=attestation.create_interruptible_json_type,
            get_verification_http_status=attestation.get_verification_http_status,
            get_interruptible_present=attestation.get_interruptible_present,
            get_interruptible_json_type=attestation.get_interruptible_json_type,
            pod_inventory_count=attestation.pod_inventory_count,
            explicit_false_source=attestation.explicit_false_source,
            allocation_attested_at=allocation_attested_at,
        )

    def record_gpu_attestation_progress(
        self,
        diagnostic: PodGPUAttestationProgressDiagnostic,
    ) -> OperatorReceipt:
        _require(
            self.stage == "running"
            and self.pod_id is not None
            and self.pod_bound_at is not None,
            "OPERATOR_RECEIPT_STAGE_INVALID",
        )
        return replace(
            self,
            gpu_attestation_outcome=diagnostic.outcome,
            gpu_attestation_failure_code=diagnostic.failure_code,
            normalized_gpu_path=diagnostic.normalized_gpu_path,
            gpu_poll_count=diagnostic.poll_count,
            gpu_poll_elapsed_seconds=diagnostic.poll_elapsed_seconds,
            final_desired_status=diagnostic.final_desired_status,
            expected_gpu_id=diagnostic.expected_gpu_id,
            observed_gpu_id=diagnostic.observed_gpu_id,
            observed_gpu_id_sha256=diagnostic.observed_gpu_id_sha256,
            gpu_count=diagnostic.gpu_count,
            cost_attestation=diagnostic.cost_attestation,
            create_http_class=diagnostic.create_http_class,
        )

    def record_connectivity_progress(
        self,
        diagnostic: PodConnectivityProgressDiagnostic,
    ) -> OperatorReceipt:
        _require(
            self.stage == "running"
            and self.pod_id is not None
            and self.pod_bound_at is not None
            and self.rental_evidence is not None,
            "OPERATOR_RECEIPT_STAGE_INVALID",
        )
        return replace(
            self,
            connectivity_outcome=diagnostic.outcome,
            connectivity_failure_code=diagnostic.failure_code,
            public_ip_present=diagnostic.public_ip_present,
            tcp_port_present=diagnostic.tcp_port_present,
            connectivity_poll_count=diagnostic.poll_count,
            connectivity_elapsed_seconds=diagnostic.elapsed_seconds,
            ssh_ready=diagnostic.ssh_ready,
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


def archive_completed_operator_receipt(path: Path) -> Path | None:
    if not path.exists():
        return None
    receipt = read_operator_receipt(path)
    _require(
        receipt.cleanup_verified and receipt.stage in {"failed", "terminated"},
        "ACTIVE_OPERATOR_RECEIPT_NOT_ARCHIVABLE",
    )
    archive_path = path.parent / "archive" / f"{receipt.run_id}.json"
    if archive_path.exists():
        _require(
            read_operator_receipt(archive_path) == receipt,
            "OPERATOR_RECEIPT_ARCHIVE_CONFLICT",
        )
    else:
        write_operator_receipt(archive_path, receipt)
    try:
        path.unlink()
    except OSError as exc:
        raise Phase3FOperatorError("OPERATOR_RECEIPT_ARCHIVE_FAILED") from exc
    return archive_path


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
    "archive_completed_operator_receipt",
    "inspect_operator_inventory",
    "process_is_running",
    "read_operator_receipt",
    "require_startable_receipt",
    "terminate_receipt_bound_pod",
    "write_operator_receipt",
]
