from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEV = (ROOT / "scripts" / "dev.ps1").read_text(encoding="utf-8")


def test_dev_smoke_mode_reuses_the_normal_cleanup_path() -> None:
    assert "[switch]$SmokeTest" in DEV
    ready = DEV.index("AtlasLens core services are ready")
    smoke = DEV.index("if ($SmokeTest)", ready)
    loop = DEV.index("while ((", smoke)
    cleanup = DEV.index("finally {", loop)
    assert "return" in DEV[smoke:loop]
    assert "Stop-AtlasLensProcessTree -RootProcess $process" in DEV[cleanup:]


def test_dev_cleanup_is_scoped_to_processes_started_by_the_command() -> None:
    assert "function Stop-AtlasLensProcessTree" in DEV
    assert "ParentProcessId -in $parents" in DEV
    assert "CreationDate -ge $rootStartedAt" in DEV
    assert "$RootProcess.StartTime.AddSeconds(-2)" in DEV
    assert "Stop-Process -Id $processId" in DEV
    assert "Stop-Process -Id $rootId" in DEV


def test_dev_propagates_external_runtime_and_env_isolation() -> None:
    assert '"-RuntimeRoot"' in DEV
    assert '"-IgnoreProjectEnv"' in DEV
    assert "Assert-AtlasLensApiRuntime $runtime" in DEV
