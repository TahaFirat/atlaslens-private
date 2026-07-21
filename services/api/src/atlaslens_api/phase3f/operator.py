"""Receipt-bound local operator controls for the Phase 3F RunPod pilot."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
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
_GPU_ATTESTATION_OUTCOMES = frozenset({"pending", "attested", "passed", "failed"})
_GPU_ATTESTATION_PATHS = frozenset(
    {"gpu.id", "machine.gpuTypeId", "machine.gpuType.id"}
)
_GPU_COUNT_ATTESTATION_PATHS = frozenset(
    {"gpuCount", "gpu.count", "machine.gpuType.count"}
)
_GPU_ATTESTATION_STATUSES = frozenset({"RUNNING", "EXITED", "TERMINATED"})
_GPU_ATTESTATION_COST = "graphql_uninterruptable_price_match"
_CONNECTIVITY_OUTCOMES = frozenset({"pending", "ready", "failed"})
_FAILURE_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_ARCHIVE_INDEX_SCHEMA = "atlaslens-phase3f-operator-archive-index-v2"
_ARCHIVE_SHA256_PREFIX_LENGTH = 16
_CANONICAL_ARCHIVE_NAME = re.compile(
    r"^v2--(?P<run_id>[0-9a-f]{32})--(?P<attempt_id>[0-9a-f]{32})--"
    r"(?P<stage>failed|terminated)--(?P<sha256>[0-9a-f]{16})\.json$"
)
_LEGACY_COMPOUND_ARCHIVE_NAME = re.compile(
    r"^(?P<run_id>[0-9a-f]{32})-(?P<attempt_id>[0-9a-f]{32})-"
    r"(?P<stage>failed|terminated)-(?P<sha256>[0-9a-f]{16,64})\.json$"
)
_LEGACY_ARCHIVE_NAME = re.compile(r"^[0-9a-f]{32}\.json$")
_PARTIAL_NAME = re.compile(r"^\.?[A-Za-z0-9_.-]{1,190}\.(?:partial|tmp)$")
_ARCHIVE_NAME_VERSIONS = frozenset({"v2", "legacy_v1_compound", "legacy_run"})


class Phase3FOperatorError(RuntimeError):
    """Stable operator failure code without provider or credential detail."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ArchiveNameIdentity:
    """Typed, path-free identity decoded from one archive filename."""

    version: str
    run_id: str
    attempt_id: str | None
    terminal_stage: str | None
    receipt_content_sha256_prefix: str | None

    def __post_init__(self) -> None:
        _require(self.version in _ARCHIVE_NAME_VERSIONS, "OPERATOR_ARCHIVE_NAME_INVALID")
        _require(bool(_RUN_ID.fullmatch(self.run_id)), "OPERATOR_ARCHIVE_NAME_INVALID")
        if self.version == "legacy_run":
            _require(
                self.attempt_id is None
                and self.terminal_stage is None
                and self.receipt_content_sha256_prefix is None,
                "OPERATOR_ARCHIVE_NAME_INVALID",
            )
            return
        _require(
            isinstance(self.attempt_id, str)
            and bool(_RUN_ID.fullmatch(self.attempt_id))
            and self.terminal_stage in {"failed", "terminated"}
            and isinstance(self.receipt_content_sha256_prefix, str)
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{16,64}",
                    self.receipt_content_sha256_prefix,
                )
            ),
            "OPERATOR_ARCHIVE_NAME_INVALID",
        )
        if self.version == "v2":
            _require(
                len(cast(str, self.receipt_content_sha256_prefix))
                == _ARCHIVE_SHA256_PREFIX_LENGTH,
                "OPERATOR_ARCHIVE_NAME_INVALID",
            )


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
    attempt_id: str | None = None
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
    normalized_gpu_count_path: str | None = None
    gpu_poll_count: int | None = None
    gpu_poll_elapsed_seconds: float | None = None
    final_desired_status: str | None = None
    expected_gpu_id: str | None = None
    observed_gpu_id: str | None = None
    observed_gpu_id_sha256: str | None = None
    gpu_count: int | None = None
    cost_attestation: str | None = None
    create_http_class: str | None = None
    receipt_bound_pod_count: int | None = None
    unexpected_pod_count: int | None = None
    endpoint_count: int | None = None
    network_volume_count: int | None = None
    template_count: int | None = None
    receipt_bound_match: bool | None = None
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
        if self.attempt_id is not None:
            _require(
                bool(_RUN_ID.fullmatch(self.attempt_id)),
                "OPERATOR_RECEIPT_ATTEMPT_ID_INVALID",
            )
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
                self.normalized_gpu_count_path is None
                or self.normalized_gpu_count_path in _GPU_COUNT_ATTESTATION_PATHS,
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
            post_create_inventory = (
                self.receipt_bound_pod_count,
                self.unexpected_pod_count,
                self.endpoint_count,
                self.network_volume_count,
                self.template_count,
            )
            if any(value is not None for value in post_create_inventory):
                _require(
                    all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and 0 <= value <= 10_000
                        for value in post_create_inventory
                    )
                    and isinstance(self.receipt_bound_match, bool),
                    "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
                )
            else:
                _require(
                    self.receipt_bound_match is None,
                    "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
                )
            if self.gpu_attestation_outcome in {"attested", "passed"}:
                legacy_inventory = all(
                    value is None
                    for value in (*post_create_inventory, self.receipt_bound_match)
                )
                safe_inventory = (
                    self.receipt_bound_pod_count == 1
                    and self.unexpected_pod_count == 0
                    and self.endpoint_count == 0
                    and self.network_volume_count == 0
                    and self.template_count == 0
                    and self.receipt_bound_match is True
                )
                _require(
                    self.rental_evidence is not None
                    and self.normalized_gpu_path is not None
                    and self.observed_gpu_id == self.expected_gpu_id
                    and self.gpu_count == 1
                    and (
                        self.gpu_attestation_outcome == "attested"
                        or self.normalized_gpu_count_path is not None
                    )
                    and self.final_desired_status == "RUNNING"
                    and self.cost_attestation == _GPU_ATTESTATION_COST
                    and (legacy_inventory or safe_inventory),
                    "OPERATOR_RECEIPT_GPU_ATTESTATION_INVALID",
                )
        else:
            _require(
                all(
                    value is None
                    for value in (
                        self.gpu_attestation_failure_code,
                        self.normalized_gpu_path,
                        self.normalized_gpu_count_path,
                        self.gpu_poll_count,
                        self.gpu_poll_elapsed_seconds,
                        self.final_desired_status,
                        self.expected_gpu_id,
                        self.observed_gpu_id,
                        self.observed_gpu_id_sha256,
                        self.gpu_count,
                        self.cost_attestation,
                        self.create_http_class,
                        self.receipt_bound_pod_count,
                        self.unexpected_pod_count,
                        self.endpoint_count,
                        self.network_volume_count,
                        self.template_count,
                        self.receipt_bound_match,
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
            "attempt_id": self.attempt_id,
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
            "normalized_gpu_count_path": self.normalized_gpu_count_path,
            "gpu_poll_count": self.gpu_poll_count,
            "gpu_poll_elapsed_seconds": self.gpu_poll_elapsed_seconds,
            "final_desired_status": self.final_desired_status,
            "expected_gpu_id": self.expected_gpu_id,
            "observed_gpu_id": self.observed_gpu_id,
            "observed_gpu_id_sha256": self.observed_gpu_id_sha256,
            "gpu_count": self.gpu_count,
            "cost_attestation": self.cost_attestation,
            "create_http_class": self.create_http_class,
            "receipt_bound_pod_count": self.receipt_bound_pod_count,
            "unexpected_pod_count": self.unexpected_pod_count,
            "endpoint_count": self.endpoint_count,
            "network_volume_count": self.network_volume_count,
            "template_count": self.template_count,
            "receipt_bound_match": self.receipt_bound_match,
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
            "normalized_gpu_count_path",
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
        post_create_inventory_fields = {
            "receipt_bound_pod_count",
            "unexpected_pod_count",
            "endpoint_count",
            "network_volume_count",
            "template_count",
            "receipt_bound_match",
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
        legacy_gpu_attestation_fields = gpu_attestation_fields - {
            "normalized_gpu_count_path"
        }
        schema = row.get("schema")
        is_legacy = schema == _LEGACY_OPERATOR_RECEIPT_SCHEMA
        normalized_fields = set(row)
        normalized_fields.discard("attempt_id")
        _require(
            (is_legacy and set(row) == legacy_expected)
            or (
                schema == OPERATOR_RECEIPT_SCHEMA
                and frozenset(normalized_fields)
                in {
                    frozenset(legacy_expected | evidence_fields),
                    *(
                        frozenset(legacy_expected | evidence_fields | gpu_fields | suffix)
                        for gpu_fields in (
                            legacy_gpu_attestation_fields,
                            gpu_attestation_fields,
                        )
                        for suffix in (
                            set(),
                            connectivity_fields,
                            post_create_inventory_fields | connectivity_fields,
                        )
                    ),
                }
            ),
            "OPERATOR_RECEIPT_INVALID",
        )
        attempt_id = row.get("attempt_id") if not is_legacy else None
        _require(
            attempt_id is None
            or (isinstance(attempt_id, str) and bool(_RUN_ID.fullmatch(attempt_id))),
            "OPERATOR_RECEIPT_ATTEMPT_ID_INVALID",
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
        normalized_gpu_count_path = (
            row.get("normalized_gpu_count_path") if has_gpu_attestation else None
        )
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
        has_post_create_inventory = "receipt_bound_pod_count" in row
        receipt_bound_pod_count = (
            row.get("receipt_bound_pod_count") if has_post_create_inventory else None
        )
        unexpected_pod_count = (
            row.get("unexpected_pod_count") if has_post_create_inventory else None
        )
        endpoint_count = row.get("endpoint_count") if has_post_create_inventory else None
        network_volume_count = (
            row.get("network_volume_count") if has_post_create_inventory else None
        )
        template_count = row.get("template_count") if has_post_create_inventory else None
        receipt_bound_match = (
            row.get("receipt_bound_match") if has_post_create_inventory else None
        )
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
            normalized_gpu_count_path,
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
            receipt_bound_match,
        ):
            _require(
                optional_boolean is None or isinstance(optional_boolean, bool),
                "OPERATOR_RECEIPT_EVIDENCE_INVALID",
            )
        for optional_integer in (
            create_status,
            get_status,
            inventory_count,
            receipt_bound_pod_count,
            unexpected_pod_count,
            endpoint_count,
            network_volume_count,
            template_count,
        ):
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
            attempt_id=cast(str | None, attempt_id),
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
            normalized_gpu_count_path=cast(str | None, normalized_gpu_count_path),
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
            receipt_bound_pod_count=cast(int | None, receipt_bound_pod_count),
            unexpected_pod_count=cast(int | None, unexpected_pod_count),
            endpoint_count=cast(int | None, endpoint_count),
            network_volume_count=cast(int | None, network_volume_count),
            template_count=cast(int | None, template_count),
            receipt_bound_match=cast(bool | None, receipt_bound_match),
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
            normalized_gpu_count_path=diagnostic.normalized_gpu_count_path,
            gpu_poll_count=diagnostic.poll_count,
            gpu_poll_elapsed_seconds=diagnostic.poll_elapsed_seconds,
            final_desired_status=diagnostic.final_desired_status,
            expected_gpu_id=diagnostic.expected_gpu_id,
            observed_gpu_id=diagnostic.observed_gpu_id,
            observed_gpu_id_sha256=diagnostic.observed_gpu_id_sha256,
            gpu_count=diagnostic.gpu_count,
            cost_attestation=diagnostic.cost_attestation,
            create_http_class=diagnostic.create_http_class,
            receipt_bound_pod_count=diagnostic.receipt_bound_pod_count,
            unexpected_pod_count=diagnostic.unexpected_pod_count,
            endpoint_count=diagnostic.endpoint_count,
            network_volume_count=diagnostic.network_volume_count,
            template_count=diagnostic.template_count,
            receipt_bound_match=diagnostic.receipt_bound_match,
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
    if pid == os.getpid():
        return True
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
    if receipt.stage in {"failed", "terminated"}:
        raise Phase3FOperatorError("UNCLEAN_OPERATOR_RECEIPT_REQUIRES_TERMINATE")
    if is_process_running(receipt.supervisor_pid):
        raise Phase3FOperatorError("OPERATOR_PROCESS_ALREADY_RUNNING")
    raise Phase3FOperatorError("STALE_OPERATOR_RECEIPT_REQUIRES_TERMINATE")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _receipt_bytes(path: Path) -> bytes:
    _require(path.is_file() and not path.is_symlink(), "OPERATOR_RECEIPT_MISSING")
    try:
        size = path.stat().st_size
        _require(0 < size <= _MAX_RECEIPT_BYTES, "OPERATOR_RECEIPT_INVALID")
        return path.read_bytes()
    except OSError as exc:
        raise Phase3FOperatorError("OPERATOR_RECEIPT_INVALID") from exc


def _receipt_semantic_sha256(receipt: OperatorReceipt) -> str:
    # Optional null fields are deliberately omitted so adding a backward-compatible
    # receipt field cannot change a legacy receipt's semantic identity.
    semantic = {
        key: value for key, value in receipt.to_dict().items() if value is not None
    }
    return hashlib.sha256(_canonical_json_bytes(semantic)).hexdigest()


def operator_budget_linkage(receipt: OperatorReceipt) -> str:
    if receipt.attempt_id is not None:
        return f"attempt:{receipt.run_id}:{receipt.attempt_id}"
    if receipt.pod_id is not None:
        pod_hash = hashlib.sha256(receipt.pod_id.encode("utf-8")).hexdigest()
        return f"pod:{pod_hash}"
    return f"legacy:{receipt.run_id}:{_receipt_semantic_sha256(receipt)}"


def _resolved_attempt_id(
    receipt: OperatorReceipt,
    semantic_sha256: str,
) -> tuple[str, str]:
    if receipt.attempt_id is not None:
        return receipt.attempt_id, "explicit"
    digest = hashlib.sha256(
        f"atlaslens-phase3f-legacy-attempt-v1:{semantic_sha256}".encode("ascii")
    ).hexdigest()
    return digest[:32], "legacy_semantic_sha256"


def format_archive_name(
    receipt: OperatorReceipt,
    *,
    content_sha256: str | None = None,
) -> str:
    """Return the sole canonical v2 archive name for a terminal receipt."""

    _require(
        receipt.stage in {"failed", "terminated"},
        "ACTIVE_OPERATOR_RECEIPT_NOT_ARCHIVABLE",
    )
    digest = (
        hashlib.sha256(_canonical_json_bytes(receipt.to_dict())).hexdigest()
        if content_sha256 is None
        else content_sha256
    )
    _require(bool(_SHA256.fullmatch(digest)), "OPERATOR_ARCHIVE_CONTENT_HASH_INVALID")
    semantic_sha256 = _receipt_semantic_sha256(receipt)
    attempt_id, _source = _resolved_attempt_id(receipt, semantic_sha256)
    name = (
        f"v2--{receipt.run_id}--{attempt_id}--{receipt.stage}--"
        f"{digest[:_ARCHIVE_SHA256_PREFIX_LENGTH]}.json"
    )
    identity = parse_archive_name(name)
    _require(
        identity.version == "v2"
        and identity.run_id == receipt.run_id
        and identity.attempt_id == attempt_id
        and identity.terminal_stage == receipt.stage
        and identity.receipt_content_sha256_prefix
        == digest[:_ARCHIVE_SHA256_PREFIX_LENGTH],
        "OPERATOR_ARCHIVE_NAME_ROUNDTRIP_FAILED",
    )
    return name


def parse_archive_name(name: str) -> ArchiveNameIdentity:
    """Parse canonical or immutable legacy archive names without filesystem access."""

    _require(
        isinstance(name, str)
        and bool(name)
        and len(name) <= 255
        and "/" not in name
        and "\\" not in name,
        "OPERATOR_ARCHIVE_NAME_INVALID",
    )
    try:
        name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise Phase3FOperatorError("OPERATOR_ARCHIVE_NAME_INVALID") from exc
    canonical_match = _CANONICAL_ARCHIVE_NAME.fullmatch(name)
    if canonical_match is not None:
        return ArchiveNameIdentity(
            version="v2",
            run_id=canonical_match.group("run_id"),
            attempt_id=canonical_match.group("attempt_id"),
            terminal_stage=canonical_match.group("stage"),
            receipt_content_sha256_prefix=canonical_match.group("sha256"),
        )
    legacy_compound_match = _LEGACY_COMPOUND_ARCHIVE_NAME.fullmatch(name)
    if legacy_compound_match is not None:
        return ArchiveNameIdentity(
            version="legacy_v1_compound",
            run_id=legacy_compound_match.group("run_id"),
            attempt_id=legacy_compound_match.group("attempt_id"),
            terminal_stage=legacy_compound_match.group("stage"),
            receipt_content_sha256_prefix=legacy_compound_match.group("sha256"),
        )
    if _LEGACY_ARCHIVE_NAME.fullmatch(name) is not None:
        return ArchiveNameIdentity(
            version="legacy_run",
            run_id=name[:-5],
            attempt_id=None,
            terminal_stage=None,
            receipt_content_sha256_prefix=None,
        )
    raise Phase3FOperatorError("OPERATOR_ARCHIVE_NAME_INVALID")


def _archive_entry(path: Path) -> dict[str, object]:
    raw = _receipt_bytes(path)
    receipt = read_operator_receipt(path)
    content_sha256 = hashlib.sha256(raw).hexdigest()
    semantic_sha256 = _receipt_semantic_sha256(receipt)
    resolved_attempt_id, identity_source = _resolved_attempt_id(receipt, semantic_sha256)
    archive_identity = parse_archive_name(path.name)
    _require(
        archive_identity.run_id == receipt.run_id,
        "OPERATOR_ARCHIVE_NAME_MISMATCH",
    )
    if archive_identity.version == "legacy_run":
        attempt_id = resolved_attempt_id
    else:
        _require(
            archive_identity.terminal_stage == receipt.stage
            and isinstance(archive_identity.receipt_content_sha256_prefix, str)
            and content_sha256.startswith(
                archive_identity.receipt_content_sha256_prefix
            ),
            "OPERATOR_ARCHIVE_NAME_MISMATCH",
        )
        attempt_id = cast(str, archive_identity.attempt_id)
        if receipt.attempt_id is not None or archive_identity.version == "v2":
            _require(
                attempt_id == resolved_attempt_id,
                "OPERATOR_ARCHIVE_NAME_MISMATCH",
            )
        else:
            # The historical unversioned compound codec derived this segment
            # from an evolving to_dict() schema. Its filename is the immutable
            # attempt identity; run/stage/content hash remain fully validated.
            identity_source = "legacy_v1_filename"
    classification = (
        "terminal"
        if receipt.stage in {"failed", "terminated"} and receipt.cleanup_verified
        else "active"
        if receipt.stage in {"preflight", "running"}
        else "unclean"
    )
    return {
        "archive_name": path.name,
        "archive_name_version": archive_identity.version,
        "run_id": receipt.run_id,
        "attempt_id": attempt_id,
        "attempt_identity_source": identity_source,
        "terminal_stage": receipt.stage,
        "classification": classification,
        "cleanup_verified": receipt.cleanup_verified,
        "receipt_content_sha256": content_sha256,
        "receipt_semantic_sha256": semantic_sha256,
        "duplicate_group_sha256": semantic_sha256,
        "budget_linkage": operator_budget_linkage(receipt),
    }


def _partial_receipts(operator_root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for parent in (operator_root, operator_root / "archive"):
        if not parent.exists():
            continue
        _require(parent.is_dir() and not parent.is_symlink(), "OPERATOR_ARCHIVE_INVALID")
        for path in parent.iterdir():
            if not path.name.endswith((".partial", ".tmp")):
                continue
            _require(
                bool(_PARTIAL_NAME.fullmatch(path.name))
                and path.is_file()
                and not path.is_symlink()
                and path.stat().st_size <= _MAX_RECEIPT_BYTES,
                "OPERATOR_ARCHIVE_PARTIAL_INVALID",
            )
            candidates.append(path)
    return tuple(sorted(candidates, key=lambda item: str(item).casefold()))


def reconcile_operator_archive_index(
    operator_root: Path,
    *,
    write: bool = False,
) -> dict[str, object]:
    _require(not operator_root.is_symlink(), "OPERATOR_ARCHIVE_INVALID")
    archive_root = operator_root / "archive"
    if archive_root.exists():
        _require(
            archive_root.is_dir() and not archive_root.is_symlink(),
            "OPERATOR_ARCHIVE_INVALID",
        )
        paths = tuple(sorted(archive_root.glob("*.json"), key=lambda item: item.name.casefold()))
    else:
        paths = ()
        if write:
            archive_root.mkdir(parents=True, exist_ok=True)
    _require(len(paths) <= 10_000, "OPERATOR_ARCHIVE_CAP_EXCEEDED")
    folded_names: set[str] = set()
    entries: list[dict[str, object]] = []
    for path in paths:
        folded = path.name.casefold()
        _require(folded not in folded_names, "OPERATOR_ARCHIVE_CASE_CONFLICT")
        folded_names.add(folded)
        entry = _archive_entry(path)
        entries.append(entry)

    entries.sort(
        key=lambda item: (
            cast(str, item["run_id"]),
            cast(str, item["attempt_id"]),
            cast(str, item["receipt_content_sha256"]),
            cast(str, item["archive_name"]).casefold(),
        )
    )
    duplicate_sizes: dict[str, int] = {}
    budget_linkage_sizes: dict[str, int] = {}
    for entry in entries:
        group = cast(str, entry["duplicate_group_sha256"])
        duplicate_sizes[group] = duplicate_sizes.get(group, 0) + 1
        linkage = cast(str, entry["budget_linkage"])
        budget_linkage_sizes[linkage] = budget_linkage_sizes.get(linkage, 0) + 1
    previous = "0" * 64
    chained_entries: list[dict[str, object]] = []
    for entry in entries:
        chained = {**entry, "previous_entry_sha256": previous}
        entry_sha256 = hashlib.sha256(_canonical_json_bytes(chained)).hexdigest()
        chained["entry_sha256"] = entry_sha256
        chained_entries.append(chained)
        previous = entry_sha256
    partials = _partial_receipts(operator_root)
    document_base: dict[str, object] = {
        "schema": _ARCHIVE_INDEX_SCHEMA,
        "entries": chained_entries,
        "chain_head_sha256": previous,
        "archive_receipt_count": len(entries),
        "legacy_attempt_count": sum(
            entry["attempt_identity_source"]
            in {"legacy_semantic_sha256", "legacy_v1_filename"}
            for entry in entries
        ),
        "duplicate_group_count": sum(size > 1 for size in duplicate_sizes.values()),
        "duplicate_receipt_count": sum(max(0, size - 1) for size in duplicate_sizes.values()),
        "duplicate_budget_linkage_count": sum(
            max(0, size - 1) for size in budget_linkage_sizes.values()
        ),
        "active_receipt_count": sum(entry["classification"] == "active" for entry in entries),
        "unclean_receipt_count": sum(entry["classification"] == "unclean" for entry in entries),
        "partial_file_count": len(partials),
        "archive_conflicts": 0,
        "secret_values_included": False,
    }
    document = {
        **document_base,
        "index_sha256": hashlib.sha256(_canonical_json_bytes(document_base)).hexdigest(),
    }
    if write:
        index_path = operator_root / "archive-index.json"
        payload = _canonical_json_bytes(document)
        try:
            operator_root.mkdir(parents=True, exist_ok=True)
            if index_path.exists():
                _require(not index_path.is_symlink(), "OPERATOR_ARCHIVE_INDEX_INVALID")
                if index_path.read_bytes() == payload:
                    return document
            temporary = index_path.with_name(
                f".{index_path.name}.{uuid4().hex}.partial"
            )
            temporary.write_bytes(payload)
            try:
                os.chmod(temporary, 0o600)
                os.replace(temporary, index_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        except OSError as exc:
            raise Phase3FOperatorError("OPERATOR_ARCHIVE_INDEX_WRITE_FAILED") from exc
    return document


def _lock_state(
    lock_path: Path,
    *,
    is_process_running: Callable[[int], bool],
) -> str:
    try:
        lock_stat = lock_path.lstat()
    except FileNotFoundError:
        return "absent"
    except OSError as exc:
        raise Phase3FOperatorError("OPERATOR_ARCHIVE_LOCK_INVALID") from exc
    _require(stat.S_ISREG(lock_stat.st_mode), "OPERATOR_ARCHIVE_LOCK_INVALID")
    try:
        raw = lock_path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        try:
            age_seconds = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return "absent"
        return "initializing" if age_seconds < 5 else "invalid"
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "atlaslens-phase3f-operator-archive-lock-v1"
        or not isinstance(value.get("pid"), int)
        or isinstance(value.get("pid"), bool)
        or not isinstance(value.get("nonce"), str)
        or not _RUN_ID.fullmatch(cast(str, value.get("nonce")))
    ):
        return "invalid"
    return "active" if is_process_running(cast(int, value["pid"])) else "stale"


@contextmanager
def operator_lifecycle_lock(
    operator_root: Path,
    *,
    is_process_running: Callable[[int], bool] = process_is_running,
) -> Iterator[None]:
    _require(not operator_root.is_symlink(), "OPERATOR_ARCHIVE_INVALID")
    operator_root.mkdir(parents=True, exist_ok=True)
    lock_path = operator_root / ".archive.lock"
    nonce = uuid4().hex
    deadline = time.monotonic() + 5
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except (FileExistsError, PermissionError) as exc:
            state = _lock_state(lock_path, is_process_running=is_process_running)
            if state == "absent":
                if isinstance(exc, PermissionError) and time.monotonic() >= deadline:
                    raise Phase3FOperatorError("OPERATOR_ARCHIVE_LOCK_FAILED") from exc
                time.sleep(0.01)
                continue
            if state == "stale":
                with suppress(FileNotFoundError):
                    lock_path.unlink()
                continue
            if state == "invalid":
                raise Phase3FOperatorError("OPERATOR_ARCHIVE_LOCK_INVALID") from None
            if time.monotonic() >= deadline:
                raise Phase3FOperatorError("OPERATOR_ARCHIVE_LOCK_ACTIVE") from None
            time.sleep(0.01)
            continue
        except OSError as exc:
            raise Phase3FOperatorError("OPERATOR_ARCHIVE_LOCK_FAILED") from exc
        try:
            payload = _canonical_json_bytes(
                {
                    "schema": "atlaslens-phase3f-operator-archive-lock-v1",
                    "pid": os.getpid(),
                    "nonce": nonce,
                }
            )
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        break
    try:
        yield
    finally:
        release_deadline = time.monotonic() + 1
        while True:
            try:
                row = json.loads(lock_path.read_bytes())
                if not isinstance(row, Mapping) or not hmac.compare_digest(
                    str(row.get("nonce", "")), nonce
                ):
                    break
                lock_path.unlink()
                break
            except FileNotFoundError:
                break
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                if time.monotonic() >= release_deadline:
                    raise Phase3FOperatorError(
                        "OPERATOR_ARCHIVE_LOCK_RELEASE_FAILED"
                    ) from exc
                time.sleep(0.01)


def _archive_completed_locked(
    path: Path,
    *,
    reconcile_index: bool = True,
) -> Path | None:
    if not path.exists():
        if reconcile_index:
            reconcile_operator_archive_index(path.parent, write=True)
        return None
    raw = _receipt_bytes(path)
    receipt = read_operator_receipt(path)
    _require(
        receipt.cleanup_verified and receipt.stage in {"failed", "terminated"},
        "ACTIVE_OPERATOR_RECEIPT_NOT_ARCHIVABLE",
    )
    content_sha256 = hashlib.sha256(raw).hexdigest()
    archive_root = path.parent / "archive"
    _require(not archive_root.is_symlink(), "OPERATOR_ARCHIVE_INVALID")
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / format_archive_name(
        receipt,
        content_sha256=content_sha256,
    )
    if target.exists():
        _require(
            target.is_file() and not target.is_symlink(),
            "OPERATOR_ARCHIVE_INVALID",
        )
        _require(
            target.read_bytes() == raw,
            "OPERATOR_RECEIPT_CONTENT_HASH_COLLISION",
        )
    try:
        if target.exists():
            path.unlink()
        else:
            os.replace(path, target)
    except OSError as exc:
        raise Phase3FOperatorError("OPERATOR_RECEIPT_ARCHIVE_FAILED") from exc
    if reconcile_index:
        reconcile_operator_archive_index(path.parent, write=True)
    return target


def operator_archive_index_matches(
    operator_root: Path,
    index: Mapping[str, object] | None = None,
) -> bool:
    """Return whether the stored index is byte-exact for the current archive."""

    expected = (
        reconcile_operator_archive_index(operator_root, write=False)
        if index is None
        else dict(index)
    )
    index_path = operator_root / "archive-index.json"
    if not index_path.exists():
        return False
    _require(
        index_path.is_file() and not index_path.is_symlink(),
        "OPERATOR_ARCHIVE_INDEX_INVALID",
    )
    try:
        return hmac.compare_digest(
            index_path.read_bytes(),
            _canonical_json_bytes(expected),
        )
    except OSError as exc:
        raise Phase3FOperatorError("OPERATOR_ARCHIVE_INDEX_INVALID") from exc


def _operator_state_sha256(operator_root: Path) -> str:
    rows: list[dict[str, object]] = []
    if not operator_root.exists():
        return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()
    _require(
        operator_root.is_dir() and not operator_root.is_symlink(),
        "OPERATOR_ARCHIVE_INVALID",
    )
    for path in sorted(
        (candidate for candidate in operator_root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(operator_root).as_posix().casefold(),
    ):
        relative = path.relative_to(operator_root).as_posix()
        if relative == ".archive.lock":
            continue
        _require(not path.is_symlink(), "OPERATOR_ARCHIVE_INVALID")
        raw = path.read_bytes()
        rows.append(
            {
                "path": relative,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def _quarantine_partial_locked(path: Path, operator_root: Path) -> None:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Phase3FOperatorError("OPERATOR_ARCHIVE_PARTIAL_INVALID") from exc
    digest = hashlib.sha256(raw).hexdigest()
    quarantine_root = operator_root / "quarantine"
    _require(not quarantine_root.is_symlink(), "OPERATOR_ARCHIVE_INVALID")
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / f"v1--{digest}.quarantined"
    try:
        if target.exists():
            _require(
                target.is_file()
                and not target.is_symlink()
                and target.read_bytes() == raw,
                "OPERATOR_ARCHIVE_QUARANTINE_CONFLICT",
            )
            path.unlink()
        else:
            os.replace(path, target)
    except OSError as exc:
        raise Phase3FOperatorError("OPERATOR_ARCHIVE_QUARANTINE_FAILED") from exc


def _recover_partial_receipts_locked(
    operator_root: Path,
) -> tuple[int, int]:
    recovered = 0
    quarantined = 0
    for path in _partial_receipts(operator_root):
        try:
            receipt = read_operator_receipt(path)
        except Phase3FOperatorError:
            _quarantine_partial_locked(path, operator_root)
            quarantined += 1
            continue
        if not receipt.cleanup_verified or receipt.stage not in {"failed", "terminated"}:
            _quarantine_partial_locked(path, operator_root)
            quarantined += 1
            continue
        raw = _receipt_bytes(path)
        digest = hashlib.sha256(raw).hexdigest()
        archive_root = operator_root / "archive"
        _require(not archive_root.is_symlink(), "OPERATOR_ARCHIVE_INVALID")
        archive_root.mkdir(parents=True, exist_ok=True)
        target = archive_root / format_archive_name(receipt, content_sha256=digest)
        try:
            if target.exists():
                _require(
                    target.is_file()
                    and not target.is_symlink()
                    and target.read_bytes() == raw,
                    "OPERATOR_RECEIPT_CONTENT_HASH_COLLISION",
                )
                path.unlink()
            else:
                os.replace(path, target)
        except OSError as exc:
            raise Phase3FOperatorError("OPERATOR_RECEIPT_ARCHIVE_FAILED") from exc
        recovered += 1
    return recovered, quarantined


def reconcile_local_operator_receipts(operator_root: Path) -> dict[str, object]:
    """Crash-safely reconcile only local receipt/archive/index state."""

    with operator_lifecycle_lock(operator_root):
        current_path = operator_root / "phase3f-current.json"
        current_before = int(current_path.exists())
        if current_before:
            current = read_operator_receipt(current_path)
            _require(
                current.cleanup_verified
                and current.stage in {"failed", "terminated"},
                "ACTIVE_OPERATOR_RECEIPT_NOT_ARCHIVABLE",
            )
        before_index = reconcile_operator_archive_index(operator_root, write=False)
        before_state_sha256 = _operator_state_sha256(operator_root)
        stored_index_matched_before = operator_archive_index_matches(
            operator_root,
            before_index,
        )
        recovered_partials, quarantined_partials = _recover_partial_receipts_locked(
            operator_root
        )
        archived = _archive_completed_locked(
            current_path,
            reconcile_index=False,
        )
        final_index = reconcile_operator_archive_index(operator_root, write=True)
        _require(
            operator_archive_index_matches(operator_root, final_index),
            "OPERATOR_ARCHIVE_INDEX_MISMATCH",
        )
        after_state_sha256 = _operator_state_sha256(operator_root)
        return {
            "before_current_receipt_count": current_before,
            "after_current_receipt_count": int(current_path.exists()),
            "before_archive_receipt_count": before_index["archive_receipt_count"],
            "after_archive_receipt_count": final_index["archive_receipt_count"],
            "before_partial_file_count": before_index["partial_file_count"],
            "after_partial_file_count": final_index["partial_file_count"],
            "recovered_partial_receipt_count": recovered_partials,
            "quarantined_partial_file_count": quarantined_partials,
            "terminal_receipt_archived": archived is not None,
            "stored_index_matched_before": stored_index_matched_before,
            "stored_index_matches_after": True,
            "archive_conflicts": final_index["archive_conflicts"],
            "active_local_receipts": final_index["active_receipt_count"],
            "unclean_local_receipts": final_index["unclean_receipt_count"],
            "duplicate_budget_linkage_count": final_index[
                "duplicate_budget_linkage_count"
            ],
            "filename_parse_failure_count": 0,
            "filename_content_identity_mismatch_count": 0,
            "index_hash_chain_mismatch_count": 0,
            "before_operator_state_sha256": before_state_sha256,
            "after_operator_state_sha256": after_state_sha256,
            "changed": not hmac.compare_digest(
                before_state_sha256,
                after_state_sha256,
            ),
            "secret_values_included": False,
        }


def archive_completed_operator_receipt(path: Path) -> Path | None:
    with operator_lifecycle_lock(path.parent):
        return _archive_completed_locked(path)


def prepare_operator_attempt(
    path: Path,
    receipt: OperatorReceipt,
    *,
    is_process_running: Callable[[int], bool] = process_is_running,
) -> Path | None:
    _require(
        receipt.attempt_id is not None
        and receipt.stage == "preflight"
        and receipt.pod_id is None
        and receipt.cleanup_verified is False,
        "OPERATOR_ATTEMPT_RECEIPT_INVALID",
    )
    with operator_lifecycle_lock(path.parent, is_process_running=is_process_running):
        require_startable_receipt(path, is_process_running=is_process_running)
        archived = _archive_completed_locked(path)
        _require(not path.exists(), "OPERATOR_CURRENT_RECEIPT_CONFLICT")
        index = reconcile_operator_archive_index(path.parent, write=True)
        _require(
            index["active_receipt_count"] == 0
            and index["unclean_receipt_count"] == 0
            and index["archive_conflicts"] == 0,
            "OPERATOR_ARCHIVE_NOT_STARTABLE",
        )
        write_operator_receipt(path, receipt)
        confirmed = read_operator_receipt(path)
        _require(
            confirmed.attempt_id is not None
            and hmac.compare_digest(confirmed.attempt_id, cast(str, receipt.attempt_id))
            and hmac.compare_digest(confirmed.run_id, receipt.run_id)
            and confirmed.stage == "preflight"
            and confirmed.pod_id is None,
            "OPERATOR_ATTEMPT_RECEIPT_NOT_CONFIRMED",
        )
        return archived


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
    "ArchiveNameIdentity",
    "OperatorInventoryClient",
    "OperatorInventoryStatus",
    "OperatorReceipt",
    "Phase3FOperatorError",
    "archive_completed_operator_receipt",
    "format_archive_name",
    "inspect_operator_inventory",
    "operator_budget_linkage",
    "operator_archive_index_matches",
    "operator_lifecycle_lock",
    "prepare_operator_attempt",
    "process_is_running",
    "read_operator_receipt",
    "reconcile_local_operator_receipts",
    "reconcile_operator_archive_index",
    "parse_archive_name",
    "require_startable_receipt",
    "terminate_receipt_bound_pod",
    "write_operator_receipt",
]
