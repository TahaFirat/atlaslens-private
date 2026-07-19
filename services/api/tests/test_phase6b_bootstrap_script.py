from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_phase6b_bootstrap_materializes_source_only_checkout_and_uses_module_cli() -> None:
    script = (ROOT / "scripts" / "bootstrap_phase6b_models.ps1").read_text(
        encoding="utf-8"
    )

    assert '"checkout", "--detach", "--force"' in script
    assert '"-r", $requirements' in script
    assert "from huggingface_hub import snapshot_download" in script
    assert "huggingface_hub.commands" not in script
    assert "hf.exe" not in script
    assert '"pip", "check", "--python"' in script


def test_phase6b_worker_hub_client_is_pinned_in_external_lock() -> None:
    lock = json.loads((ROOT / "config" / "external-models.lock.json").read_text())

    assert lock["tooling"]["huggingface_hub"] == "1.23.0"


def test_phase6b_diagnostics_probe_official_osv_import_and_binary_weight() -> None:
    diagnostics = (ROOT / "scripts" / "verify_phase6b_models.py").read_text(
        encoding="utf-8"
    )

    assert '"models.huggingface"' in diagnostics
    assert '"pytorch_model.bin"' in diagnostics
    assert "LocalWorkerHTTPClient" in diagnostics
    assert '"worker_unreachable"' in diagnostics
    assert '"inference_not_verified"' in diagnostics
    assert 'health = "prepared"' not in diagnostics
