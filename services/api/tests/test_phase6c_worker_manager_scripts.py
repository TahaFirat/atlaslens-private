from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
START = (ROOT / "scripts" / "start-phase6c-workers.ps1").read_text()
STOP = (ROOT / "scripts" / "stop-phase6c-workers.ps1").read_text()


def test_phase6c_worker_manager_pins_loopback_identity_and_hidden_process() -> None:
    assert '"127.0.0.1"' in START
    assert "$port = 8794" in START
    assert "1af071c68fc3ab6c6018c5c868391763516e50f7" in START
    assert "7cb9f7970d366fdf059963d04d372e503e8e9df9" in START
    assert "-WindowStyle Hidden" in START
    assert "verify_phase6c_workers.py" in START
    assert "atlaslens-worker-process-v1" in START
    assert "pid = $listenerOwner" in START
    assert "launcher_pid = $process.Id" in START
    assert "$record.pid -ne $owner" in START
    assert "$listenerProcess.StartTime" in START
    assert "$oldPythonPath = $env:PYTHONPATH" in START
    assert r'"services\api\src"' in START
    assert "$env:PYTHONPATH = $oldPythonPath" in START


def test_phase6c_stop_refuses_untrusted_listener_and_revalidates_pid() -> None:
    assert "untrusted listener" in STOP
    assert "PID was recycled" in STOP
    assert "listener ownership changed" in STOP
    assert "/v1/unload" in STOP
    assert "Stop-Process" in STOP
