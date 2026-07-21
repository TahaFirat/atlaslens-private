from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

import atlaslens_api.phase3f.operator as operator_module
from atlaslens_api.phase3f.operator import (
    OperatorReceipt,
    Phase3FOperatorError,
    archive_completed_operator_receipt,
    format_archive_name,
    inspect_operator_inventory,
    operator_archive_index_matches,
    parse_archive_name,
    prepare_operator_attempt,
    read_operator_receipt,
    reconcile_local_operator_receipts,
    reconcile_operator_archive_index,
    require_startable_receipt,
    terminate_receipt_bound_pod,
    write_operator_receipt,
)
from atlaslens_api.phase3f.runpod import (
    GPUAvailabilityReport,
    GPUOffer,
    PodConnection,
    PodConnectivityProgressDiagnostic,
    PodGPUAttestationProgressDiagnostic,
    PodRentalAttestationDiagnostic,
    RunPodInventory,
)
from atlaslens_api.phase3f.safety import (
    BudgetPolicy,
    PodRecord,
    PodRequest,
    RunPodLease,
)

ROOT = Path(__file__).resolve().parents[3]
START_PATH = ROOT / "scripts" / "start-phase3f-runpod.ps1"
STATUS_PATH = ROOT / "scripts" / "status-phase3f-runpod.ps1"
STOP_PATH = ROOT / "scripts" / "stop-phase3f-runpod.ps1"
SUPERVISOR_PATH = ROOT / "scripts" / "phase3f" / "supervisor.py"


def _receipt(*, pod_id: str | None = "phase3f-pod", stage: str = "running") -> OperatorReceipt:
    return OperatorReceipt(
        run_id="a" * 32,
        attempt_id="b" * 32,
        run_marker=f"atlaslens-phase3f-{'a' * 32}",
        pod_id=pod_id,
        supervisor_pid=12345,
        stage=stage,
        started_at="2026-07-19T18:00:00+00:00",
        finished_at=None,
        max_spend_usd=Decimal("10"),
        soft_stop_usd=Decimal("7.5"),
        hard_stop_usd=Decimal("9"),
        max_gpu_hourly_usd=Decimal("0.50"),
        max_wall_minutes=345,
        cleanup_verified=False,
    )


def _terminal_attempt(
    attempt: int,
    *,
    pod_id: str | None = None,
    cleanup_verified: bool = True,
) -> OperatorReceipt:
    receipt = replace(
        _receipt(pod_id=pod_id, stage="running" if pod_id else "preflight"),
        attempt_id=f"{attempt:032x}",
        supervisor_pid=12_000 + attempt,
        started_at=f"2026-07-19T18:{attempt:02d}:00+00:00",
    )
    return receipt.update_lifecycle(
        stage="terminated",
        finished_at=f"2026-07-19T18:{attempt:02d}:30+00:00",
        cleanup_verified=cleanup_verified,
    )


class _OperatorClient:
    def __init__(self, inventory: RunPodInventory) -> None:
        self.current = inventory
        self.terminate_calls: list[str] = []

    def inventory(self) -> RunPodInventory:
        return self.current

    def terminate_pod(self, pod_id: str) -> None:
        self.terminate_calls.append(pod_id)
        self.current = RunPodInventory(
            tuple(item for item in self.current.pods if item.pod_id != pod_id),
            self.current.endpoint_ids,
            self.current.network_volume_ids,
            self.current.template_ids,
        )


def _load_supervisor() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase3f_operator_supervisor", SUPERVISOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operator_receipt_round_trip_is_atomic_and_pod_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "operator state with spaces" / "phase3f-current.json"
    receipt = _receipt()

    write_operator_receipt(path, receipt)

    assert read_operator_receipt(path) == receipt
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["secret_values_included"] is False
    assert payload["pod_id"] == "phase3f-pod"
    assert payload["pod_id_sha256"]
    assert not tuple(path.parent.glob("*.partial"))

    payload["pod_id"] = "different-pod"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Phase3FOperatorError, match="OPERATOR_RECEIPT_POD_HASH_INVALID"):
        read_operator_receipt(path)


def test_create_id_is_atomically_bound_before_response_field_validation(
    tmp_path: Path,
) -> None:
    module = _load_supervisor()
    path = tmp_path / "operator" / "phase3f-current.json"
    receipt = _receipt(pod_id=None, stage="preflight")
    write_operator_receipt(path, receipt)
    pod = PodRecord("created-pod", receipt.run_marker)

    module._bind_operator_pod(
        path,
        expected_run_id=receipt.run_id,
        expected_attempt_id=cast(str, receipt.attempt_id),
        pod=pod,
    )

    bound = read_operator_receipt(path)
    assert bound.pod_id == "created-pod"
    assert bound.pod_bound_at is not None
    assert bound.stage == "running"
    assert bound.cleanup_verified is False
    assert not tuple(path.parent.glob("*.partial"))


def test_attestation_is_atomically_recorded_after_pod_binding(tmp_path: Path) -> None:
    module = _load_supervisor()
    path = tmp_path / "operator" / "phase3f-current.json"
    receipt = _receipt(pod_id=None, stage="preflight")
    write_operator_receipt(path, receipt)
    module._bind_operator_pod(
        path,
        expected_run_id=receipt.run_id,
        expected_attempt_id=cast(str, receipt.attempt_id),
        pod=PodRecord("created-pod", receipt.run_marker),
    )
    attestation = PodRentalAttestationDiagnostic(
        evidence="request_and_on_demand_price_attested",
        request_interruptible=False,
        create_http_status=201,
        selected_gpu_id="NVIDIA RTX A5000",
        selected_uninterruptable_price=Decimal("0.16"),
        create_cost_per_hr=Decimal("0.160"),
        price_delta_usd=Decimal("0.000"),
        desired_status="RUNNING",
        cloud_type="SECURE",
        create_interruptible_present=False,
        create_interruptible_json_type="missing",
        get_verification_http_status=200,
        get_interruptible_present=False,
        get_interruptible_json_type="missing",
        pod_inventory_count=1,
        explicit_false_source=None,
        create_http_class="success_201",
        normalized_gpu_path="machine.gpuTypeId",
        gpu_poll_count=2,
        gpu_poll_elapsed_seconds=1.0,
        observed_gpu_id="NVIDIA RTX A5000",
        gpu_count=1,
        normalized_gpu_count_path="gpuCount",
        cost_attestation="graphql_uninterruptable_price_match",
    )
    pending = PodGPUAttestationProgressDiagnostic(
        outcome="pending",
        failure_code=None,
        normalized_gpu_path="machine.gpuTypeId",
        poll_count=2,
        poll_elapsed_seconds=1.0,
        final_desired_status="RUNNING",
        expected_gpu_id="NVIDIA RTX A5000",
        observed_gpu_id="NVIDIA RTX A5000",
        gpu_count=1,
        cost_attestation="graphql_uninterruptable_price_match",
        receipt_bound_pod_count=1,
        unexpected_pod_count=0,
        endpoint_count=0,
        network_volume_count=0,
        template_count=0,
        receipt_bound_match=True,
    )

    module._record_operator_gpu_attestation_progress(
        path,
        expected_run_id=receipt.run_id,
        expected_attempt_id=cast(str, receipt.attempt_id),
        diagnostic=pending,
    )

    module._record_operator_rental_attestation(
        path,
        expected_run_id=receipt.run_id,
        expected_attempt_id=cast(str, receipt.attempt_id),
        attestation=attestation,
    )
    module._record_operator_gpu_attestation_progress(
        path,
        expected_run_id=receipt.run_id,
        expected_attempt_id=cast(str, receipt.attempt_id),
        diagnostic=PodGPUAttestationProgressDiagnostic(
            outcome="passed",
            failure_code=None,
            normalized_gpu_path="machine.gpuTypeId",
            normalized_gpu_count_path="gpuCount",
            poll_count=2,
            poll_elapsed_seconds=1.0,
            final_desired_status="RUNNING",
            expected_gpu_id="NVIDIA RTX A5000",
            observed_gpu_id="NVIDIA RTX A5000",
            gpu_count=1,
            cost_attestation="graphql_uninterruptable_price_match",
            receipt_bound_pod_count=1,
            unexpected_pod_count=0,
            endpoint_count=0,
            network_volume_count=0,
            template_count=0,
            receipt_bound_match=True,
        ),
    )
    module._record_operator_connectivity_progress(
        path,
        expected_run_id=receipt.run_id,
        expected_attempt_id=cast(str, receipt.attempt_id),
        diagnostic=PodConnectivityProgressDiagnostic(
            outcome="ready",
            failure_code=None,
            public_ip_present=True,
            tcp_port_present=True,
            poll_count=4,
            elapsed_seconds=6.0,
            ssh_ready=True,
        ),
    )

    recorded = read_operator_receipt(path)
    assert recorded.pod_id == "created-pod"
    assert recorded.pod_bound_at is not None
    assert recorded.rental_evidence == "request_and_on_demand_price_attested"
    assert recorded.request_interruptible is False
    assert recorded.selected_gpu_id == "NVIDIA RTX A5000"
    assert recorded.selected_uninterruptable_price == Decimal("0.16")
    assert recorded.create_cost_per_hr == Decimal("0.160")
    assert recorded.get_interruptible_json_type == "missing"
    assert recorded.pod_inventory_count == 1
    assert recorded.gpu_attestation_outcome == "passed"
    assert recorded.normalized_gpu_path == "machine.gpuTypeId"
    assert recorded.normalized_gpu_count_path == "gpuCount"
    assert recorded.gpu_poll_count == 2
    assert recorded.observed_gpu_id == "NVIDIA RTX A5000"
    assert recorded.gpu_count == 1
    assert recorded.cost_attestation == "graphql_uninterruptable_price_match"
    assert recorded.receipt_bound_pod_count == 1
    assert recorded.unexpected_pod_count == 0
    assert recorded.endpoint_count == 0
    assert recorded.network_volume_count == 0
    assert recorded.template_count == 0
    assert recorded.receipt_bound_match is True
    assert recorded.allocation_attested_at is not None
    assert recorded.connectivity_outcome == "ready"
    assert recorded.public_ip_present is True
    assert recorded.tcp_port_present is True
    assert recorded.connectivity_poll_count == 4
    assert recorded.connectivity_elapsed_seconds == 6.0
    assert recorded.ssh_ready is True
    assert recorded.to_dict()["secret_values_included"] is False
    assert "interruptible_field_verified" not in recorded.to_dict()
    assert "192.0.2.10" not in repr(recorded.to_dict())
    assert "10341" not in repr(recorded.to_dict())
    assert not tuple(path.parent.glob("*.partial"))


def test_ready_allocation_advances_to_transfer_and_training_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_supervisor()
    receipt = _receipt()
    receipt_path = tmp_path / "operator" / "phase3f-current.json"
    write_operator_receipt(receipt_path, receipt)
    request = PodRequest(
        run_marker=receipt.run_marker,
        idempotency_key="phase3f-connectivity-test",
        hourly_cost_usd=Decimal("0.50"),
        max_runtime_seconds=60,
        gpu_type_id="NVIDIA L4",
        public_ports=(22,),
    )
    lease = RunPodLease(
        PodRecord("phase3f-pod", receipt.run_marker),
        request,
        BudgetPolicy(),
    )
    transitions: list[str] = []
    watched: list[list[str]] = []

    class _ReadyClient:
        def await_pod_connectivity(
            self,
            pod: PodRecord,
            observed_request: PodRequest,
            *,
            ssh_probe: object,
        ) -> PodConnection:
            assert pod == lease.pod
            assert observed_request == request
            assert callable(ssh_probe)
            transitions.append("connectivity_ready")
            return PodConnection(
                pod_id=pod.pod_id,
                public_ip="192.0.2.10",
                public_ssh_port=10341,
                gpu_display_name="NVIDIA L4",
                hourly_price=Decimal("0.39"),
            )

    def fake_run_command(arguments: list[str], *, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        transitions.append("transfer_command")
        assert arguments

    def fake_run_watched(
        arguments: list[str],
        *,
        lease: RunPodLease,
        started: float,
        status_event: str = "PHASE3F_CLOUD_JOB_RUNNING",
        **_kwargs: object,
    ) -> None:
        assert lease.pod.pod_id == "phase3f-pod"
        assert started > 0
        transitions.append(
            "bootstrap_started"
            if status_event == "PHASE3F_REMOTE_BOOTSTRAP_RUNNING"
            else "training_started"
        )
        watched.append(arguments)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module, "_run_watched", fake_run_watched)
    started = time.monotonic()

    result = module._operation(
        _ReadyClient(),
        lease,
        bundle_root=tmp_path / "bundle",
        key=tmp_path / "key",
        known_hosts=tmp_path / "known-hosts",
        download_path=tmp_path / "output.tar",
        run_id=receipt.run_id,
        started=started,
        operator_receipt=receipt,
        operator_receipt_path=receipt_path,
        remote_job_seconds=60,
        training_dataset=tmp_path / "sealed-acquisition.tar",
        environment_contract_sha256="c" * 64,
        dependency_lock_sha256="d" * 64,
    )

    assert transitions.index("connectivity_ready") < transitions.index("transfer_command")
    assert transitions.index("transfer_command") < transitions.index("training_started")
    assert len(watched) == 2
    assert "remote_bootstrap.py" in " ".join(watched[0])
    assert "training_job.py" in " ".join(watched[1])
    assert "phase3f-venv/bin/python" in " ".join(watched[1])
    assert result["gpu"] == "NVIDIA L4"
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    assert events.index("PHASE3F_SSH_READY") < events.index("PHASE3F_TRANSFER_STARTED")
    assert events.index("PHASE3F_TRANSFER_VERIFIED") < events.index(
        "PHASE3F_REMOTE_BOOTSTRAP_STARTED"
    )
    assert events.index("PHASE3F_REMOTE_DEPENDENCIES_READY") < events.index(
        "PHASE3F_CLOUD_JOB_STARTED"
    )
    assert events.count("PHASE3F_REMOTE_BOOTSTRAP_STARTED") == 1
    assert events.count("PHASE3F_REMOTE_DEPENDENCIES_READY") == 1


def test_legacy_operator_receipt_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "phase3f-current.json"
    current = _receipt()
    payload = current.to_dict()
    payload["schema"] = "atlaslens-phase3f-operator-receipt-v1"
    for field in (
        "pod_bound_at",
        "attempt_id",
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
        "allocation_attested_at",
        "receipt_bound_pod_count",
        "unexpected_pod_count",
        "endpoint_count",
        "network_volume_count",
        "template_count",
        "receipt_bound_match",
        "connectivity_outcome",
        "connectivity_failure_code",
        "public_ip_present",
        "tcp_port_present",
        "connectivity_poll_count",
        "connectivity_elapsed_seconds",
        "ssh_ready",
    ):
        del payload[field]
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = read_operator_receipt(path)

    assert restored.pod_id == current.pod_id
    assert restored.pod_bound_at is None
    assert restored.rental_evidence is None


def test_previous_v2_operator_receipt_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "phase3f-current.json"
    payload = _receipt().to_dict()
    for field in (
        "attempt_id",
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
        "allocation_attested_at",
        "receipt_bound_pod_count",
        "unexpected_pod_count",
        "endpoint_count",
        "network_volume_count",
        "template_count",
        "receipt_bound_match",
        "connectivity_outcome",
        "connectivity_failure_code",
        "public_ip_present",
        "tcp_port_present",
        "connectivity_poll_count",
        "connectivity_elapsed_seconds",
        "ssh_ready",
    ):
        del payload[field]
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = read_operator_receipt(path)

    assert restored.pod_id == "phase3f-pod"
    assert restored.gpu_attestation_outcome is None
    assert restored.create_http_class is None
    assert restored.attempt_id is None


def test_stale_or_live_operator_receipt_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "phase3f-current.json"
    write_operator_receipt(path, _receipt())

    with pytest.raises(Phase3FOperatorError, match="OPERATOR_PROCESS_ALREADY_RUNNING"):
        require_startable_receipt(path, is_process_running=lambda _pid: True)
    with pytest.raises(
        Phase3FOperatorError,
        match="STALE_OPERATOR_RECEIPT_REQUIRES_TERMINATE",
    ):
        require_startable_receipt(path, is_process_running=lambda _pid: False)

    terminated = _receipt().update_lifecycle(
        stage="terminated",
        finished_at="2026-07-19T19:00:00+00:00",
        cleanup_verified=True,
    )
    write_operator_receipt(path, terminated)
    require_startable_receipt(path, is_process_running=lambda _pid: True)


def test_completed_receipt_is_archived_without_fake_termination(tmp_path: Path) -> None:
    path = tmp_path / "_operator" / "phase3f-current.json"
    completed = _receipt(pod_id=None, stage="preflight").update_lifecycle(
        stage="terminated",
        finished_at="2026-07-19T19:00:00+00:00",
        cleanup_verified=True,
    )
    write_operator_receipt(path, completed)

    archived = archive_completed_operator_receipt(path)

    assert archived is not None
    assert archived.name.startswith(
        f"v2--{completed.run_id}--{completed.attempt_id}--terminated--"
    )
    assert read_operator_receipt(archived) == completed
    assert not path.exists()
    assert completed.pod_id is None


def test_same_run_archives_eight_terminal_attempts_without_conflict(
    tmp_path: Path,
) -> None:
    current = tmp_path / "operator state with spaces" / "phase3f-current.json"
    archived: set[str] = set()

    for attempt in range(1, 9):
        write_operator_receipt(current, _terminal_attempt(attempt))
        destination = archive_completed_operator_receipt(current)
        assert destination is not None
        archived.add(destination.name)

    index = reconcile_operator_archive_index(current.parent, write=False)
    assert len(archived) == 8
    assert index["archive_receipt_count"] == 8
    assert index["legacy_attempt_count"] == 0
    assert index["duplicate_receipt_count"] == 0
    assert index["active_receipt_count"] == 0
    assert index["unclean_receipt_count"] == 0
    assert index["archive_conflicts"] == 0
    assert not current.exists()


def test_byte_identical_archive_retry_is_idempotent(tmp_path: Path) -> None:
    current = tmp_path / "_operator" / "phase3f-current.json"
    receipt = _terminal_attempt(1)
    write_operator_receipt(current, receipt)
    first = archive_completed_operator_receipt(current)
    index_path = current.parent / "archive-index.json"
    first_index = index_path.read_bytes()

    write_operator_receipt(current, receipt)
    second = archive_completed_operator_receipt(current)

    assert first == second
    assert first is not None
    assert len(tuple((current.parent / "archive").glob("*.json"))) == 1
    assert index_path.read_bytes() == first_index
    assert not current.exists()


def test_semantically_identical_legacy_receipts_form_duplicate_group(
    tmp_path: Path,
) -> None:
    current = tmp_path / "_operator" / "phase3f-current.json"
    current.parent.mkdir(parents=True)
    payload = _terminal_attempt(1).to_dict()
    del payload["attempt_id"]
    current.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    first = archive_completed_operator_receipt(current)
    current.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    second = archive_completed_operator_receipt(current)

    index = reconcile_operator_archive_index(current.parent, write=False)
    assert first is not None and second is not None and first != second
    assert index["archive_receipt_count"] == 2
    assert index["legacy_attempt_count"] == 2
    assert index["duplicate_group_count"] == 1
    assert index["duplicate_receipt_count"] == 1


def test_content_prefix_collision_never_overwrites_existing_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_sha256 = hashlib.sha256

    class _SharedPrefixHash:
        def __init__(self, payload: bytes = b"") -> None:
            self.inner = real_sha256(payload)

        def update(self, payload: bytes) -> None:
            self.inner.update(payload)

        def hexdigest(self) -> str:
            return "0" * 16 + self.inner.hexdigest()[16:]

    monkeypatch.setattr(operator_module.hashlib, "sha256", _SharedPrefixHash)
    current = tmp_path / "_operator" / "phase3f-current.json"
    first_receipt = _terminal_attempt(1)
    second_receipt = replace(
        _terminal_attempt(1),
        supervisor_pid=99_999,
        started_at="2026-07-19T19:01:00+00:00",
        finished_at="2026-07-19T19:01:30+00:00",
    )
    write_operator_receipt(current, first_receipt)
    first = archive_completed_operator_receipt(current)
    assert first is not None
    first_bytes = first.read_bytes()
    write_operator_receipt(current, second_receipt)
    with pytest.raises(
        Phase3FOperatorError,
        match="OPERATOR_RECEIPT_CONTENT_HASH_COLLISION",
    ):
        archive_completed_operator_receipt(current)

    assert first.read_bytes() == first_bytes
    assert read_operator_receipt(first) == first_receipt
    assert read_operator_receipt(current) == second_receipt


def test_archive_index_recovers_from_crash_partial_without_receipt_mutation(
    tmp_path: Path,
) -> None:
    current = tmp_path / "_operator" / "phase3f-current.json"
    receipt = _terminal_attempt(1)
    write_operator_receipt(current, receipt)
    archived = archive_completed_operator_receipt(current)
    assert archived is not None
    before = archived.read_bytes()
    index_path = current.parent / "archive-index.json"
    index_path.write_bytes(b"crash-truncated")
    partial = current.parent / ".archive-index.json.deadbeef.tmp"
    partial.write_bytes(b"partial")

    recovered = reconcile_operator_archive_index(current.parent, write=True)

    assert archived.read_bytes() == before
    assert recovered["archive_receipt_count"] == 1
    assert recovered["partial_file_count"] == 1
    assert json.loads(index_path.read_bytes())["index_sha256"] == recovered["index_sha256"]


def test_stale_archive_lock_is_recovered_before_new_attempt(tmp_path: Path) -> None:
    current = tmp_path / "_operator" / "phase3f-current.json"
    current.parent.mkdir(parents=True)
    lock = current.parent / ".archive.lock"
    lock.write_text(
        json.dumps(
            {
                "schema": "atlaslens-phase3f-operator-archive-lock-v1",
                "pid": 999_999,
                "nonce": "c" * 32,
            }
        ),
        encoding="utf-8",
    )
    receipt = replace(_receipt(pod_id=None, stage="preflight"), attempt_id="d" * 32)

    prepare_operator_attempt(
        current,
        receipt,
        is_process_running=lambda _pid: False,
    )

    assert read_operator_receipt(current) == receipt
    assert not lock.exists()


def test_active_and_unclean_receipts_block_new_attempt(tmp_path: Path) -> None:
    current = tmp_path / "_operator" / "phase3f-current.json"
    next_receipt = replace(
        _receipt(pod_id=None, stage="preflight"),
        attempt_id="d" * 32,
    )
    write_operator_receipt(current, _receipt(pod_id=None, stage="preflight"))
    with pytest.raises(Phase3FOperatorError, match="OPERATOR_PROCESS_ALREADY_RUNNING"):
        prepare_operator_attempt(
            current,
            next_receipt,
            is_process_running=lambda _pid: True,
        )

    unclean = _terminal_attempt(2, cleanup_verified=False)
    write_operator_receipt(current, unclean)
    with pytest.raises(
        Phase3FOperatorError,
        match="UNCLEAN_OPERATOR_RECEIPT_REQUIRES_TERMINATE",
    ):
        prepare_operator_attempt(
            current,
            next_receipt,
            is_process_running=lambda _pid: False,
        )


def test_cleanup_verified_terminal_receipt_allows_new_attempt(tmp_path: Path) -> None:
    current = tmp_path / "_operator" / "phase3f-current.json"
    write_operator_receipt(current, _terminal_attempt(1))
    next_receipt = replace(
        _receipt(pod_id=None, stage="preflight"),
        attempt_id="d" * 32,
    )

    archived = prepare_operator_attempt(
        current,
        next_receipt,
        is_process_running=lambda _pid: False,
    )

    assert archived is not None
    assert read_operator_receipt(current) == next_receipt
    assert reconcile_operator_archive_index(current.parent)["archive_receipt_count"] == 1


def test_parallel_archive_calls_create_one_immutable_receipt(tmp_path: Path) -> None:
    current = tmp_path / "_operator" / "phase3f-current.json"
    for attempt in range(1, 9):
        receipt = _terminal_attempt(attempt)
        write_operator_receipt(current, receipt)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(
                executor.map(
                    lambda _index: archive_completed_operator_receipt(current),
                    range(2),
                )
            )

        paths = tuple(path for path in results if path is not None)
        assert len(paths) == 1
        assert read_operator_receipt(paths[0]) == receipt
    assert len(tuple((current.parent / "archive").glob("*.json"))) == 8
    assert reconcile_operator_archive_index(current.parent)["archive_receipt_count"] == 8


def test_archive_names_are_lowercase_and_refuse_case_ambiguous_legacy_name(
    tmp_path: Path,
) -> None:
    current = tmp_path / "Operator State With Spaces" / "phase3f-current.json"
    write_operator_receipt(current, _terminal_attempt(1))
    archived = archive_completed_operator_receipt(current)
    assert archived is not None
    assert archived.name == archived.name.lower()

    ambiguous = current.parent / "archive" / f"{'A' * 32}.json"
    write_operator_receipt(ambiguous, _terminal_attempt(2))
    with pytest.raises(Phase3FOperatorError, match="OPERATOR_ARCHIVE_NAME_INVALID"):
        reconcile_operator_archive_index(current.parent)


def test_archive_codec_round_trip_uses_versioned_fixed_grammar() -> None:
    receipt = _terminal_attempt(1)

    name = format_archive_name(receipt)
    identity = parse_archive_name(name)

    assert name == name.lower()
    assert identity.version == "v2"
    assert identity.run_id == receipt.run_id
    assert identity.attempt_id == receipt.attempt_id
    assert identity.terminal_stage == "terminated"
    assert identity.receipt_content_sha256_prefix is not None
    assert len(identity.receipt_content_sha256_prefix) == 16


def test_real_failing_archive_name_is_typed_as_immutable_legacy_compound() -> None:
    name = (
        "ce23d58c42bf76e6de0b0117725e3ea7-"
        "05253cd0446d09b1cf2cae8a94e8494b-terminated-1690a5c8cf564239.json"
    )

    identity = parse_archive_name(name)

    assert identity.version == "legacy_v1_compound"
    assert identity.attempt_id == "05253cd0446d09b1cf2cae8a94e8494b"
    assert identity.receipt_content_sha256_prefix == "1690a5c8cf564239"


@pytest.mark.parametrize(
    "name",
    (
        "v2--" + "a" * 32 + "--" + "b" * 32 + "--running--" + "c" * 16 + ".json",
        "v2--" + "a" * 32 + "--" + "b" * 32 + "--terminated--" + "c" * 15 + ".json",
        "v2--" + "A" * 32 + "--" + "b" * 32 + "--terminated--" + "c" * 16 + ".json",
        "../receipt.json",
        "archive\\receipt.json",
        "v2--receipt-ı.json",
    ),
)
def test_archive_codec_rejects_stage_case_separator_hash_and_unicode_drift(
    name: str,
) -> None:
    with pytest.raises(Phase3FOperatorError, match="OPERATOR_ARCHIVE_NAME_INVALID"):
        parse_archive_name(name)


def test_legacy_run_and_compound_names_share_the_canonical_parser(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_operator"
    archive = root / "archive"
    archive.mkdir(parents=True)
    receipt = _terminal_attempt(1)
    payload = receipt.to_dict()
    del payload["attempt_id"]
    raw = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    digest = hashlib.sha256(raw).hexdigest()
    legacy_run = archive / f"{receipt.run_id}.json"
    legacy_run.write_bytes(raw)

    first_index = reconcile_operator_archive_index(root, write=True)
    legacy_run.rename(
        archive
        / (
            f"{receipt.run_id}-{'c' * 32}-terminated-{digest[:16]}.json"
        )
    )
    second_index = reconcile_operator_archive_index(root, write=True)

    assert first_index["legacy_attempt_count"] == 1
    assert second_index["legacy_attempt_count"] == 1
    assert second_index["entries"][0]["attempt_identity_source"] == "legacy_v1_filename"
    assert operator_archive_index_matches(root, second_index)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("run", "d" * 32),
        ("attempt", "e" * 32),
        ("stage", "failed"),
        ("sha", "0" * 16),
    ),
)
def test_canonical_filename_content_identity_mismatch_fails_typed(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    root = tmp_path / "_operator"
    archive = root / "archive"
    archive.mkdir(parents=True)
    receipt = _terminal_attempt(1)
    path = archive / format_archive_name(receipt)
    write_operator_receipt(path, receipt)
    identity = parse_archive_name(path.name)
    parts = {
        "run": identity.run_id,
        "attempt": cast(str, identity.attempt_id),
        "stage": cast(str, identity.terminal_stage),
        "sha": cast(str, identity.receipt_content_sha256_prefix),
    }
    parts[field] = replacement
    mismatched = archive / (
        f"v2--{parts['run']}--{parts['attempt']}--{parts['stage']}--"
        f"{parts['sha']}.json"
    )
    path.rename(mismatched)

    with pytest.raises(Phase3FOperatorError, match="OPERATOR_ARCHIVE_NAME_MISMATCH"):
        reconcile_operator_archive_index(root)


def test_local_receipt_reconciliation_recovers_partial_and_quarantines_index_temp(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_operator"
    root.mkdir(parents=True)
    partial_receipt = root / ".terminal-receipt.tmp"
    write_operator_receipt(partial_receipt, _terminal_attempt(1))
    (root / ".archive-index.json.deadbeef.tmp").write_bytes(b"truncated")

    result = reconcile_local_operator_receipts(root)
    index = reconcile_operator_archive_index(root)

    assert result["recovered_partial_receipt_count"] == 1
    assert result["quarantined_partial_file_count"] == 1
    assert result["after_partial_file_count"] == 0
    assert index["archive_receipt_count"] == 1
    assert len(tuple((root / "quarantine").glob("*.quarantined"))) == 1
    assert operator_archive_index_matches(root, index)


def test_local_receipt_reconciliation_is_noop_on_second_run(tmp_path: Path) -> None:
    root = tmp_path / "_operator"
    current = root / "phase3f-current.json"
    receipt = _terminal_attempt(1)
    write_operator_receipt(current, receipt)

    first = reconcile_local_operator_receipts(root)
    first_files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    second = reconcile_local_operator_receipts(root)
    second_files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }

    assert first["before_current_receipt_count"] == 1
    assert first["after_current_receipt_count"] == 0
    assert first["terminal_receipt_archived"] is True
    assert first["changed"] is True
    assert second["terminal_receipt_archived"] is False
    assert second["changed"] is False
    assert second["before_operator_state_sha256"] == second["after_operator_state_sha256"]
    assert first_files == second_files
    assert operator_archive_index_matches(root)


def test_status_is_read_only_and_rejects_a_second_pod() -> None:
    receipt = _receipt()
    expected = RunPodInventory(
        (PodRecord("phase3f-pod", receipt.run_marker),),
        (),
        (),
        (),
    )
    client = _OperatorClient(expected)

    status = inspect_operator_inventory(receipt, client.inventory())

    assert status.receipt_bound_pod_present is True
    assert status.cleanup_verified is False
    assert client.terminate_calls == []
    assert client.current == expected

    with pytest.raises(Phase3FOperatorError, match="UNEXPECTED_POD_INVENTORY"):
        inspect_operator_inventory(
            receipt,
            RunPodInventory(
                (
                    PodRecord("phase3f-pod", receipt.run_marker),
                    PodRecord("another-pod", "another-run"),
                ),
                (),
                (),
                (),
            ),
        )


def test_emergency_terminate_targets_only_receipt_pod_and_verifies_all_inventory() -> None:
    receipt = _receipt()
    client = _OperatorClient(
        RunPodInventory((PodRecord("phase3f-pod", receipt.run_marker),), (), (), ())
    )

    status = terminate_receipt_bound_pod(
        client,
        receipt,
        poll_attempts=2,
        poll_seconds=0,
    )

    assert client.terminate_calls == ["phase3f-pod"]
    assert status.cleanup_verified is True
    assert status.pods == status.endpoints == status.network_volumes == status.templates == 0


def test_emergency_terminate_never_touches_unexpected_pod_or_endpoint() -> None:
    receipt = _receipt()
    unexpected = _OperatorClient(
        RunPodInventory((PodRecord("another-pod", "another-run"),), (), (), ())
    )
    with pytest.raises(Phase3FOperatorError, match="UNEXPECTED_POD_INVENTORY"):
        terminate_receipt_bound_pod(unexpected, receipt)
    assert unexpected.terminate_calls == []

    endpoint = _OperatorClient(
        RunPodInventory((PodRecord("phase3f-pod", receipt.run_marker),), ("endpoint-1",), (), ())
    )
    with pytest.raises(Phase3FOperatorError, match="ACTIVE_ENDPOINT_INVENTORY_NOT_ZERO"):
        terminate_receipt_bound_pod(endpoint, receipt)
    assert endpoint.terminate_calls == []


def test_default_supervisor_action_is_local_dry_run_without_state_or_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_supervisor()
    monkeypatch.setenv("RUNPOD_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(module, "_disk_gate", lambda: None)
    monkeypatch.setattr(
        module,
        "_preflight_repository",
        lambda: ("f" * 40, ("tracked.py",)),
    )
    monkeypatch.setattr(module, "_verify_local_readiness", lambda: None)

    class _ForbiddenClient:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("dry-run constructed an API client")

    monkeypatch.setattr(module, "RunPodV1Client", _ForbiddenClient)
    runtime = ROOT / ".local" / "operator test paths" / tmp_path.name

    result = module.main(["--runtime-root", str(runtime)])

    assert result == 0
    assert not runtime.exists()
    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "dry-run"
    assert output["runpod_api_calls"] == 0
    assert output["cloud_mutations"] == 0
    assert output["create_attempts"] == 0
    assert output["payload_contract_valid"] is True
    assert output["secret_values_included"] is False
    fields = {item["name"]: item["json_type"] for item in output["payload_fields"]}
    assert fields["interruptible"] == "boolean"
    assert fields["gpuCount"] == "number"
    assert fields["volumeInGb"] == "number"
    assert "networkVolumeId" not in fields
    assert "templateId" not in fields
    assert "RUNPOD_SECRET" not in repr(output)


def test_live_readiness_is_authenticated_read_only_and_does_not_write_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_supervisor()
    monkeypatch.setenv("RUNPOD_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(module, "_disk_gate", lambda: None)
    monkeypatch.setattr(
        module,
        "_preflight_repository",
        lambda: ("f" * 40, ("tracked.py",)),
    )
    monkeypatch.setattr(module, "_verify_local_readiness", lambda: None)
    offer = GPUOffer(
        gpu_type_id="NVIDIA L4",
        display_name="L4",
        memory_gb=24,
        hourly_price=Decimal("0.39"),
        stock_status="Low",
        cloud_type="SECURE",
        secure_cloud=True,
        community_cloud=False,
        available_gpu_counts=None,
    )
    report = GPUAvailabilityReport(
        selected_offer=offer,
        candidates=(),
        graphql_request_count=2,
        http_statuses=(200, 200),
        top_level_keys=(("data",), ("data",)),
        response_schema_classification="official_gpuTypes_list_and_detail_lists",
    )

    class _ReadOnlyClient:
        api_request_count = 6
        cloud_mutation_count = 0
        last_graphql_errors: tuple[object, ...] = ()

        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> _ReadOnlyClient:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def inventory(self) -> RunPodInventory:
            return RunPodInventory((), (), (), ())

        def check_gpu_availability(self, *, max_hourly_price: Decimal) -> GPUAvailabilityReport:
            assert max_hourly_price == Decimal("0.50")
            return report

    monkeypatch.setattr(module, "RunPodV1Client", _ReadOnlyClient)
    runtime = ROOT / ".local" / "live readiness test paths" / tmp_path.name

    result = module.main(["--live-readiness", "--runtime-root", str(runtime)])

    assert result == 0
    assert not runtime.exists()
    output = json.loads(capsys.readouterr().out)
    assert output["ready_for_execute"] is True
    assert output["selected_gpu_id"] == "NVIDIA L4"
    assert output["selected_gpu_display_name"] == "L4"
    assert output["selected_hourly_price"] == "0.39"
    assert output["selected_stock_status"] == "Low"
    assert output["capacity_confirmed"] is False
    assert output["capacity_evidence"] == "advertised_stock_status"
    assert output["create_attempt_limit"] == 1
    assert output["runpod_api_calls"] == 6
    assert output["cloud_mutations"] == 0
    assert output["create_attempts"] == 0
    assert output["payload_contract_valid"] is True
    assert "RUNPOD_SECRET" not in repr(output)


def test_powershell_wrappers_are_explicit_receipt_bound_and_secret_safe() -> None:
    start = START_PATH.read_text(encoding="utf-8")
    status = STATUS_PATH.read_text(encoding="utf-8")
    stop = STOP_PATH.read_text(encoding="utf-8")
    combined = start + status + stop

    assert "[switch]$Execute" in start
    assert "[switch]$LiveReadiness" in start
    assert '$arguments += "--execute"' in start
    assert '$arguments += "--live-readiness"' in start
    assert "$MaxSpendUsd = 10" in start
    assert "$SoftStopUsd = 7.5" in start
    assert "$HardStopUsd = 9" in start
    assert "$MaxGpuHourlyUsd = 0.50" in start
    assert "$MaxWallMinutes = 345" in start
    assert "finally" in start
    assert "stop-phase3f-runpod.ps1" in start
    assert 'Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")' in combined
    assert "phase3f-current.json" in status
    assert "phase3f-current.json" in stop
    assert "--status" in status
    assert "--terminate" in stop
    assert "--execute" not in status + stop
    assert "RUNPOD_API_KEY" in combined
    assert "Authorization" not in combined
    assert "ApiKey" not in combined
    assert "Token" not in combined
    assert ".env" not in combined
    assert "asda.html" not in combined


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="Windows PowerShell unavailable")
@pytest.mark.parametrize("path", [START_PATH, STATUS_PATH, STOP_PATH])
def test_powershell_wrappers_parse_without_errors(path: Path) -> None:
    command = (
        "$errors=$null;$tokens=$null;"
        f"[Management.Automation.Language.Parser]::ParseFile('{path}',[ref]$tokens,[ref]$errors)"
        "|Out-Null;if($errors.Count -ne 0){$errors|ForEach-Object{$_.Message};exit 1}"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=Path(os.environ.get("TEMP", r"C:\tmp")),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_operator_module_does_not_need_subprocess_or_shell() -> None:
    source_path = (
        ROOT / "services" / "api" / "src" / "atlaslens_api" / "phase3f" / "operator.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "shell=True" not in source
    assert sys.version_info >= (3, 12)
