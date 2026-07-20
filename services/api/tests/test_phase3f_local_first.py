from __future__ import annotations

import hashlib
import inspect
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from atlaslens_api.phase3f import local_first
from atlaslens_api.phase3f.cloud_job import (
    Phase3FCloudJobError,
    _verify_canonical_vendor_text,
)

ROOT = Path(__file__).parents[3]
LAUNCHER = ROOT / "scripts" / "phase3f-local.ps1"


def _write_current(runtime_root: Path, run_id: str, stage: str = "ACQUISITION_SEALED") -> None:
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

    for action in ("AcquireOnly", "Status", "Resume", "ComputeOnly", "Cleanup"):
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
