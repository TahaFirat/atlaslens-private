from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

from atlaslens_api.phase3f.operator import (
    OperatorReceipt,
    Phase3FOperatorError,
    inspect_operator_inventory,
    read_operator_receipt,
    require_startable_receipt,
    terminate_receipt_bound_pod,
    write_operator_receipt,
)
from atlaslens_api.phase3f.runpod import RunPodInventory
from atlaslens_api.phase3f.safety import PodRecord

ROOT = Path(__file__).resolve().parents[3]
START_PATH = ROOT / "scripts" / "start-phase3f-runpod.ps1"
STATUS_PATH = ROOT / "scripts" / "status-phase3f-runpod.ps1"
STOP_PATH = ROOT / "scripts" / "stop-phase3f-runpod.ps1"
SUPERVISOR_PATH = ROOT / "scripts" / "phase3f" / "supervisor.py"


def _receipt(*, pod_id: str | None = "phase3f-pod", stage: str = "running") -> OperatorReceipt:
    return OperatorReceipt(
        run_id="a" * 32,
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


def test_powershell_wrappers_are_explicit_receipt_bound_and_secret_safe() -> None:
    start = START_PATH.read_text(encoding="utf-8")
    status = STATUS_PATH.read_text(encoding="utf-8")
    stop = STOP_PATH.read_text(encoding="utf-8")
    combined = start + status + stop

    assert "[switch]$Execute" in start
    assert '$arguments += "--execute"' in start
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
