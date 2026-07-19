from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_phase6c_lock_pins_required_megaloc_and_keeps_g3_conditional() -> None:
    lock = json.loads((ROOT / "config" / "external-models.lock.json").read_text())
    by_name = {item["name"]: item for item in lock["models"]}

    assert by_name["megaloc"]["source_revision"] == (
        "1af071c68fc3ab6c6018c5c868391763516e50f7"
    )
    assert by_name["megaloc"]["model_revision"] == (
        "7cb9f7970d366fdf059963d04d372e503e8e9df9"
    )
    assert by_name["megaloc"]["weight_sha256"] == (
        "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
    )
    assert by_name["g3"]["execution_mode"] == "not_integrated"
    assert "g3.index" in by_name["g3"]["excluded_artifacts"]


def test_phase6c_bootstrap_downloads_only_the_pinned_weight_and_writes_receipt() -> None:
    script = (ROOT / "scripts" / "bootstrap_phase6c_models.ps1").read_text()

    assert (
        '"https://huggingface.co/$repository/resolve/$revision/$filename?download=true"'
        in script
    )
    assert '"--output", $weight, $downloadUrl' in script
    assert "datasets_downloaded = $false" in script
    assert "Get-FileHash" in script
    assert "snapshot_download" not in script
    assert "hf_hub_download" not in script
    assert "g3.index" not in script


def test_phase6c_verifier_requires_real_load_and_inference_by_default() -> None:
    verifier = (ROOT / "scripts" / "verify_phase6c_models.py").read_text()

    assert "adapter.load" in verifier
    assert "adapter.infer" in verifier
    assert "real_inference_verified" in verifier
    assert "descriptor_norm" in verifier
