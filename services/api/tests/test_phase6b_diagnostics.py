from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from atlaslens_api.phase6b.runtime_verification import (
    validate_rapidocr_runtime_verification,
    write_rapidocr_runtime_verification,
)

ROOT = Path(__file__).resolve().parents[3]


def _diagnostics_module() -> ModuleType:
    path = ROOT / "scripts" / "verify_phase6b_models.py"
    spec = importlib.util.spec_from_file_location("phase6b_diagnostics_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSTICS = _diagnostics_module()


def _health(**overrides: Any) -> Any:
    values = {
        "process_running": True,
        "import_ok": True,
        "weights_available": True,
        "model_loaded": True,
        "load_verified": True,
        "inference_verified": True,
        "provider_revision": "a" * 40,
        "model_revision": "b" * 40,
        "device": "cpu",
        "last_error": None,
    }
    values.update(overrides)
    return DIAGNOSTICS.WorkerSnapshot(**values)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"enabled": False}, "disabled"),
        ({"worker_python_exists": False}, "not_installed"),
        ({"import_ok": False}, "not_installed"),
        ({"weights_available": False}, "dependencies_installed"),
        ({"source_revision_matches": False}, "weights_prepared"),
        ({"health": None, "health_error": "worker_unreachable"}, "worker_unreachable"),
        ({"health": _health(load_verified=False, model_loaded=False)}, "weights_prepared"),
        (
            {
                "health": _health(
                    load_verified=False,
                    model_loaded=False,
                    last_error="model_load_failed",
                )
            },
            "model_load_failed",
        ),
        ({"health": _health(inference_verified=False)}, "inference_not_verified"),
        ({"health": _health()}, "ready"),
    ],
)
def test_exact_isolated_provider_states(arguments: dict[str, Any], expected: str) -> None:
    values: dict[str, Any] = {
        "enabled": True,
        "worker_python_exists": True,
        "import_ok": True,
        "weights_available": True,
        "source_revision_matches": True,
        "health": _health(),
        "health_error": None,
    }
    values.update(arguments)

    state, _ = DIAGNOSTICS.classify_isolated_state(**values)

    assert state == expected
    assert state in DIAGNOSTICS.PROVIDER_STATES


def test_imports_and_weights_alone_never_report_ready() -> None:
    state, reason = DIAGNOSTICS.classify_isolated_state(
        enabled=True,
        worker_python_exists=True,
        import_ok=True,
        weights_available=True,
        source_revision_matches=True,
        health=None,
        health_error="worker_unreachable",
    )

    assert state == "worker_unreachable"
    assert reason == "worker_unreachable"


def test_rapidocr_runtime_proof_is_bound_and_contains_no_payload(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    model_root = tmp_path / "rapidocr-model"
    provider_source = project / "services/api/src/atlaslens_api/providers/rapidocr.py"
    worker_source = project / "services/api/src/atlaslens_api/providers/rapidocr_worker.py"
    provider_source.parent.mkdir(parents=True)
    provider_source.write_text("provider source", encoding="utf-8")
    worker_source.write_text("worker source", encoding="utf-8")
    model_root.mkdir()
    (model_root / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-rapidocr-v1",
                "provider_version": "3.9.1",
                "runtime_family": "onnxruntime-cpu",
                "runtime_version": "1.27.0",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="real RapidOCR"):
        write_rapidocr_runtime_verification(
            project,
            model_root,
            source_revision="a" * 40,
            model_revision="rapidocr-3.9.1",
            model_load_succeeded=True,
            real_inference_succeeded=False,
        )

    receipt = write_rapidocr_runtime_verification(
        project,
        model_root,
        source_revision="a" * 40,
        model_revision="rapidocr-3.9.1",
        model_load_succeeded=True,
        real_inference_succeeded=True,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)

    assert payload["load_verified"] is True
    assert payload["inference_verified"] is True
    assert "image" not in serialized
    assert "text" not in serialized
    assert str(project) not in serialized
    assert str(model_root) not in serialized
    assert validate_rapidocr_runtime_verification(
        project,
        model_root,
        expected_source_revision="a" * 40,
        expected_model_revision="rapidocr-3.9.1",
    ).verification is not None

    worker_source.write_text("changed worker source", encoding="utf-8")
    stale = validate_rapidocr_runtime_verification(
        project,
        model_root,
        expected_source_revision="a" * 40,
        expected_model_revision="rapidocr-3.9.1",
    )
    assert stale.verification is None
    assert stale.reason_code == "runtime_verification_stale"


def test_rapidocr_artifacts_without_runtime_proof_are_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    model_root = tmp_path / "rapidocr"
    model_root.mkdir()
    monkeypatch.setenv("RAPIDOCR_MODEL_ROOT", str(model_root))
    monkeypatch.setattr(DIAGNOSTICS, "_worker_import", lambda *args: True)
    monkeypatch.setattr(
        DIAGNOSTICS, "load_verified_rapidocr_runtime", lambda *args, **kwargs: object()
    )
    model = {
        "name": "rapidocr",
        "source_revision": "a" * 40,
        "model_id": "PP-OCRv6-small-onnx",
        "model_revision": "rapidocr-3.9.1",
        "execution_mode": "isolated_process",
    }

    provider = DIAGNOSTICS._collect_rapidocr(model, project, tmp_path)

    assert provider["state"] == "inference_not_verified"
    assert provider["load_verified"] is False
    assert provider["inference_verified"] is False

