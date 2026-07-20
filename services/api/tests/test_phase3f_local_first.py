from __future__ import annotations

import hashlib
import inspect
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from atlaslens_api.phase3f import cloud_job as worker
from atlaslens_api.phase3f import local_first
from atlaslens_api.phase3f.cloud_job import (
    Phase3FCloudJobError,
    _verify_canonical_vendor_text,
)

ROOT = Path(__file__).parents[3]
LAUNCHER = ROOT / "scripts" / "phase3f-local.ps1"


def _write_current(
    runtime_root: Path,
    run_id: str,
    stage: str = "ACQUISITION_SEALED",
    *,
    error_code: str | None = None,
) -> None:
    run_root = runtime_root / run_id
    run_root.mkdir(parents=True)
    (runtime_root / "current.json").write_text(
        json.dumps({"schema": local_first.LOCAL_CURRENT_SCHEMA, "run_id": run_id}),
        encoding="utf-8",
    )
    (run_root / "state.json").write_text(
        json.dumps(
            {
                "schema": local_first.LOCAL_STATE_SCHEMA,
                "run_id": run_id,
                "stage": stage,
                "error_code": error_code,
                "secrets_included": False,
            }
        ),
        encoding="utf-8",
    )


def test_mapillary_secret_is_process_environment_only_and_never_described(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAPILLARY_ACCESS_TOKEN", raising=False)
    with pytest.raises(local_first.LocalFirstError) as missing:
        local_first._mapillary_token()
    assert missing.value.code == "MAPILLARY_ACCESS_TOKEN_MISSING"

    monkeypatch.setenv(
        "MAPILLARY_ACCESS_TOKEN",
        "{{ RUNPOD_SECRET_atlaslens_mapillary_access_token }}",
    )
    with pytest.raises(local_first.LocalFirstError) as unresolved:
        local_first._mapillary_token()
    assert unresolved.value.code == "UNRESOLVED_RUNPOD_SECRET_REFERENCE"

    token = "MLY_fixture-value-must-not-escape"
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", token)
    assert local_first._mapillary_token() == token
    assert token not in str(missing.value)
    assert token not in str(unresolved.value)


def test_mapillary_auth_failures_are_terminal_acquisition_states() -> None:
    assert (
        local_first._acquisition_failure_stage("MAPILLARY_TOKEN_REJECTED")
        == "ACQUISITION_FAILED_TERMINAL"
    )
    assert (
        local_first._acquisition_failure_stage("MAPILLARY_PERMISSION_DENIED")
        == "ACQUISITION_FAILED_TERMINAL"
    )


def test_checksum_inventory_seals_and_detects_tampering(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    payload = sealed / "assets" / "aa" / "image.jpg"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"fixture-image")
    inventory = local_first._inventory(sealed, max_bytes=1024)
    (sealed / "checksum-inventory.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )

    assert local_first._validate_inventory(sealed)["total_size_bytes"] == 13
    payload.write_bytes(b"tampered")
    with pytest.raises(local_first.LocalFirstError) as error:
        local_first._validate_inventory(sealed)
    assert error.value.code == "SEALED_INVENTORY_MISMATCH"


def test_vendor_trust_accepts_only_exact_lf_or_crlf_representation(tmp_path: Path) -> None:
    lf = b"first\nsecond\n"
    crlf = lf.replace(b"\n", b"\r\n")
    path = tmp_path / "vendor.py"
    expected_lf = hashlib.sha256(lf).hexdigest()
    expected_crlf = hashlib.sha256(crlf).hexdigest()

    for payload in (lf, crlf):
        path.write_bytes(payload)
        _verify_canonical_vendor_text(
            path,
            canonical_lf_sha256=expected_lf,
            windows_crlf_sha256=expected_crlf,
            max_bytes=128,
        )

    path.write_bytes(b"first\r\nsecond\n")
    with pytest.raises(Phase3FCloudJobError) as error:
        _verify_canonical_vendor_text(
            path,
            canonical_lf_sha256=expected_lf,
            windows_crlf_sha256=expected_crlf,
            max_bytes=128,
        )
    assert error.value.code == "MEGALOC_VENDOR_SHA256_MISMATCH"


def _diagnostic_result(name: str, response_class: str) -> dict[str, object]:
    return {
        "name": name,
        "status_code": 200 if response_class == "success" else 503,
        "response_class": response_class,
    }


def _diagnostic_matrix() -> dict[str, dict[str, object]]:
    names = (
        "baseline_auth",
        "exact_failing_request",
        "exact_limit_1",
        "exact_minimal_fields",
        "reduced_cell_first",
        "reduced_cell_last",
        "exact_limit_25",
        "exact_limit_50",
        "exact_repeat",
        "adaptive_confirmation",
    )
    results = {name: _diagnostic_result(name, "server_error") for name in names}
    results["baseline_auth"] = _diagnostic_result("baseline_auth", "success")
    return results


def test_diagnostic_requires_repeated_shape_evidence_before_change() -> None:
    bbox = _diagnostic_matrix()
    bbox["reduced_cell_first"] = _diagnostic_result("reduced_cell_first", "success")
    bbox["reduced_cell_last"] = _diagnostic_result("reduced_cell_last", "success")
    assert local_first._diagnose_probe_results(bbox) == {
        "code": "BBOX_PARTITION_REQUIRED",
        "deterministic_cause": True,
        "request_change_authorized": True,
        "change": "deterministic_quarter_cell_partition",
    }

    page_limit = _diagnostic_matrix()
    page_limit["exact_limit_50"] = _diagnostic_result("exact_limit_50", "success")
    page_limit["adaptive_confirmation"] = {
        **_diagnostic_result("adaptive_confirmation", "success"),
        "confirmed_limit": 50,
    }
    diagnosis = local_first._diagnose_probe_results(page_limit)
    assert diagnosis["code"] == "PAGE_LIMIT_REDUCTION_REQUIRED"
    assert diagnosis["safe_page_size"] == 50
    assert diagnosis["deterministic_cause"] is True

    unresolved = _diagnostic_matrix()
    assert local_first._diagnose_probe_results(unresolved) == {
        "code": "PROVIDER_GENERAL_5XX_NOT_REQUEST_SHAPE_ISOLATED",
        "deterministic_cause": False,
        "request_change_authorized": False,
    }


def test_diagnose_acquisition_is_ten_retryless_metadata_only_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_id = "e" * 32
    _write_current(
        runtime_root,
        run_id,
        "ACQUISITION_FAILED_RESUMABLE",
        error_code="MAPILLARY_API_SERVER_RETRY_EXHAUSTED",
    )
    aoi_config = ROOT / "config" / "phase3f" / "city-coverage-aoi-v1.json"
    areas = worker.load_city_areas(aoi_config)
    checkpoint = worker._new_metadata_page_checkpoint(areas, run_id)
    work_root = runtime_root / run_id / "acquisition-work"
    worker._write_metadata_page_checkpoint(work_root / "metadata-pages.json", checkpoint)
    token = "MLY_fixture-diagnostic-secret"
    unsafe_body = f"unsafe-response-{token}"
    unsafe_request_id = "provider-request-private-value"
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", token)
    observed: list[httpx.Request] = []
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        status = 200 if len(observed) == 1 else 503
        return httpx.Response(
            status,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-FB-Request-ID": unsafe_request_id,
            },
            content=unsafe_body.encode(),
        )

    def client_factory(**kwargs: object) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(local_first.httpx, "Client", client_factory)
    result = local_first.diagnose_acquisition(
        local_first.DiagnoseAcquisitionConfig(
            runtime_root=runtime_root,
            aoi_config_path=aoi_config,
        )
    )

    assert len(observed) == 10
    assert all(request.method == "GET" and request.url.path == "/images" for request in observed)
    assert result["probe_request_count"] == 10
    assert result["image_download_requests"] == 0
    assert result["retry_attempts_per_request"] == 0
    assert result["diagnosis"] == {
        "code": "PROVIDER_GENERAL_5XX_NOT_REQUEST_SHAPE_ISOLATED",
        "deterministic_cause": False,
        "request_change_authorized": False,
    }
    receipt_path = (
        runtime_root / run_id / "diagnostics" / "acquisition-diagnostic.json"
    )
    persisted = receipt_path.read_text(encoding="utf-8")
    assert token not in persisted
    assert unsafe_body not in persisted
    assert unsafe_request_id not in persisted
    receipt = json.loads(persisted)
    assert receipt["exact_failing_request"] == {
        "endpoint": "/images",
        "parameter_names": ["bbox", "fields", "limit"],
        "checkpoint_city_index": 0,
        "checkpoint_box_index": 0,
        "next_url_present": False,
    }
    assert all(probe["response_body_retained"] is False for probe in receipt["probes"])
    assert receipt["secrets_included"] is False


def test_compute_validates_sealed_acquisition_before_model_or_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_id = "a" * 32
    _write_current(runtime_root, run_id)
    (runtime_root / run_id / "sealed-acquisition").mkdir()
    model_loaded = False

    def refuse_unsealed(_root: Path) -> local_first.VerifiedAcquisition:
        raise local_first.LocalFirstError("SEALED_INVENTORY_MISMATCH")

    class ForbiddenRuntime:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            nonlocal model_loaded
            model_loaded = True

    monkeypatch.setattr(local_first, "verify_sealed_acquisition", refuse_unsealed)
    monkeypatch.setattr(local_first.worker, "MegaLocRuntime", ForbiddenRuntime)

    with pytest.raises(local_first.LocalFirstError) as error:
        local_first.run_compute(
            local_first.ComputeConfig(
                runtime_root=runtime_root,
                model_path=tmp_path / "model.safetensors",
                vendor_root=tmp_path / "vendor",
            )
        )
    assert error.value.code == "SEALED_INVENTORY_MISMATCH"
    assert model_loaded is False
    assert not (runtime_root / run_id / "compute-output").exists()


def test_stage_boundaries_exclude_gpu_from_acquisition_and_mapillary_from_compute() -> None:
    acquisition_source = inspect.getsource(local_first.run_acquisition)
    compute_source = inspect.getsource(local_first.run_compute)

    assert "MegaLocRuntime" not in acquisition_source
    assert "torch" not in acquisition_source
    assert compute_source.index("verify_sealed_acquisition") < compute_source.index(
        "MegaLocRuntime"
    )
    assert "MapillaryClient" not in compute_source
    assert 'os.environ.pop("MAPILLARY_ACCESS_TOKEN", None)' in compute_source
    assert "_network_disabled" in compute_source


def test_compute_network_guard_refuses_inet_connections() -> None:
    original_socket = socket.socket
    with local_first._network_disabled():
        with pytest.raises(local_first.LocalFirstError) as error:
            socket.create_connection(("127.0.0.1", 9))
        assert error.value.code == "COMPUTE_NETWORK_DISABLED"
    assert socket.socket is original_socket


def test_cleanup_is_preview_first_and_exact_run_confirmed(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_id = "b" * 32
    _write_current(runtime_root, run_id)
    payload = runtime_root / run_id / "work.bin"
    payload.write_bytes(b"temporary")
    temp_payload = runtime_root / "_tmp" / "scratch.bin"
    temp_payload.parent.mkdir()
    temp_payload.write_bytes(b"scratch")

    preview = local_first.cleanup(runtime_root, execute=False, confirm_run_id=None)
    assert preview["execute"] is False
    assert payload.exists()

    with pytest.raises(local_first.LocalFirstError) as error:
        local_first.cleanup(runtime_root, execute=True, confirm_run_id="c" * 32)
    assert error.value.code == "LOCAL_CLEANUP_CONFIRMATION_MISMATCH"
    assert payload.exists()

    result = local_first.cleanup(runtime_root, execute=True, confirm_run_id=run_id)
    assert result["removed"] is True
    assert not (runtime_root / run_id).exists()
    assert not (runtime_root / "_tmp").exists()
    assert not (runtime_root / "current.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell launcher is Windows-only")
def test_launcher_status_runs_outside_repo_with_space_safe_paths(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime with spaces"
    run_id = "d" * 32
    _write_current(runtime_root, run_id)
    environment = os.environ.copy()
    environment.pop("MAPILLARY_ACCESS_TOKEN", None)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-Action",
            "Status",
            "-RuntimeRoot",
            str(runtime_root),
            "-PythonPath",
            sys.executable,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["run_id"] == run_id
    assert result["stage"] == "ACQUISITION_SEALED"
    assert result["secrets_included"] is False


@pytest.mark.skipif(os.name != "nt", reason="PowerShell launcher is Windows-only")
def test_launcher_missing_token_stops_before_runtime_or_network(tmp_path: Path) -> None:
    runtime_root = tmp_path / "must not be created"
    environment = os.environ.copy()
    environment.pop("MAPILLARY_ACCESS_TOKEN", None)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-Action",
            "AcquireOnly",
            "-RuntimeRoot",
            str(runtime_root),
            "-PythonPath",
            sys.executable,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert "MAPILLARY_ACCESS_TOKEN_MISSING" in completed.stderr
    assert not runtime_root.exists()


def test_launcher_contract_is_local_only_and_secret_safe() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    for action in (
        "AcquireOnly",
        "DiagnoseAcquisition",
        "Status",
        "Resume",
        "ComputeOnly",
        "Cleanup",
    ):
        assert action in script
    assert "$env:MAPILLARY_ACCESS_TOKEN" in script
    assert "Remove-Item Env:MAPILLARY_ACCESS_TOKEN" in script
    assert "LOCAL_RUNTIME_C_DRIVE_REFUSED" in script
    assert "8GB" in script
    assert ".env" not in script
    assert "asda.html" not in script
    assert "RUNPOD_API_KEY" not in script
    assert "Invoke-WebRequest" not in script
    assert "Invoke-RestMethod" not in script
