from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
START = (ROOT / "scripts" / "start-phase6b-workers.ps1").read_text(encoding="utf-8")
STOP = (ROOT / "scripts" / "stop-phase6b-workers.ps1").read_text(encoding="utf-8")


def test_start_script_uses_exact_isolated_workers_and_loopback_ports() -> None:
    expected = {
        "osv5m": ("8791", ".local\\workers\\osv5m\\.venv\\Scripts\\python.exe"),
        "plonk": ("8792", ".local\\workers\\plonk\\.venv\\Scripts\\python.exe"),
        "paddleocr": (
            "8793",
            ".local\\workers\\paddleocr\\.venv\\Scripts\\python.exe",
        ),
    }
    for provider, (port, python_path) in expected.items():
        assert f'services\\model-workers\\{provider}\\worker.py' in START
        assert python_path in START
        assert f"Port = {port}" in START
    assert 'PADDLEOCR_WORKER_HOST = "127.0.0.1"' in START
    assert 'PADDLEOCR_WORKER_PORT = "8793"' in START


def test_start_script_fails_closed_on_foreign_ports_and_forces_offline_mode() -> None:
    assert "Test-HealthIdentity $health $spec" in START
    assert "Test-CommandIdentity $command $spec" in START
    assert "foreign or identity-mismatched" in START
    assert 'HF_HUB_OFFLINE = "1"' in START
    assert 'TRANSFORMERS_OFFLINE = "1"' in START
    assert 'WindowStyle Hidden' in START


def test_pid_metadata_binds_process_identity_and_start_time() -> None:
    for script in (START, STOP):
        assert 'atlaslens-worker-process-v1' in script
        assert 'process_started_at_utc' in script
        assert 'Get-CimInstance -ClassName Win32_Process' in script
        assert 'ExecutablePath' in script
        assert 'CommandLine' in script
    assert '.local\\run\\phase6b-workers' in START


def test_stop_script_unloads_then_stops_only_revalidated_owned_process() -> None:
    unload_position = STOP.index('/v1/unload')
    revalidate_position = STOP.index('Revalidate immediately before termination')
    stop_position = STOP.index('Stop-Process -Id $processId', revalidate_position)
    assert unload_position < revalidate_position < stop_position
    assert 'Test-RecordIdentity $record $process $spec' in STOP
    assert 'Test-CommandIdentity (Get-CommandInfo $processId) $spec' in STOP
    assert 'Stop-Process -Id $processId -Force' in STOP
    assert 'did not stop within the bounded timeout' in STOP
