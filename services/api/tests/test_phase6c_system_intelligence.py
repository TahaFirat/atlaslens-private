from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from atlaslens_api.config import Settings
from atlaslens_api.main import create_app
from atlaslens_api.phase6c.reference_index import (
    LEAKAGE_ATTESTATION_FILENAME,
    MegaLocDescriptorSpec,
    ReferenceBuildInput,
    ReferenceIndexBuildError,
    ReferenceIndexBuildPolicy,
    ReferenceIndexLeakageAttestation,
    build_reference_index,
    open_reference_index,
    write_reference_leakage_attestation,
)

ROOT = Path(__file__).resolve().parents[3]


def _build_index(root: Path, *, name: str = "index") -> Path:
    inputs = root / f"{name}-inputs"
    inputs.mkdir()
    image_path = inputs / "reference.png"
    Image.new("RGB", (32, 24), (40, 80, 120)).save(image_path, "PNG")
    manifest = ReferenceBuildInput.model_validate(
        {
            "schema_version": "atlaslens-megaloc-reference-input-v1",
            "records": [
                {
                    "reference_id": "reference-1",
                    "source": "manual",
                    "source_family": "licensed-fixture",
                    "source_image_id": "source-image-1",
                    "source_sequence_id": "sequence-1",
                    "source_url": "https://example.org/reference/1",
                    "latitude": 39.0,
                    "longitude": 35.0,
                    "coordinate_uncertainty_m": 25.0,
                    "heading_degrees": None,
                    "captured_at": "2024-01-01T00:00:00Z",
                    "country": "Türkiye",
                    "province": "Ankara",
                    "city": "Ankara",
                    "license": "CC-BY-4.0",
                    "license_url": "https://example.org/license",
                    "attribution": "Public fixture contributor",
                    "asset_key": "licensed/reference-1.png",
                    "image_path": image_path.name,
                    "descriptor_path": None,
                }
            ],
        }
    )
    manifest_path = inputs / "reference-input.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    output = root / name
    build_reference_index(
        input_manifest=manifest_path,
        input_root=inputs,
        output_directory=output,
        index_version="turkiye-test-index-v1",
        descriptor_spec=MegaLocDescriptorSpec(
            model_id="test-megaloc",
            model_revision="test-model-v1",
            source_revision="test-source-v1",
            descriptor_version="test-descriptor-v1",
            dimension=3,
        ),
        descriptor_matrix=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        policy=ReferenceIndexBuildPolicy(perceptual_hash_hamming_threshold=0),
        built_at=datetime(2026, 7, 14, tzinfo=UTC),
    )
    return output


def _attestation(*, index_version: str = "turkiye-test-index-v1") -> Any:
    return ReferenceIndexLeakageAttestation(
        audit_fingerprint="a" * 64,
        source_report_sha256="b" * 64,
        index_version=index_version,
        descriptor_version="test-descriptor-v1",
        checked_reference_count=1,
        descriptor_checked_count=1,
        pre_index_excluded_reference_count=1,
    )


def _report(*, passed: bool = True, descriptor_checked: int = 1) -> dict[str, object]:
    exclusion = {
        "reference_key": "excluded-reference",
        "reasons": ["target_sha256_match"],
        "sha256_equal": True,
        "perceptual_distance": 0,
        "crop_transform_distance": 0,
        "geometric_supported": False,
        "descriptor_similarity": 1.0,
    }
    return {
        "audit_version": "atlaslens-leakage-audit-v1",
        "status": "passed" if passed else "failed",
        "holdout_id": "synthetic-test-holdout",
        "holdout_sha256": "c" * 64,
        "holdout_perceptual_hash": "d" * 16,
        "checked_reference_count": 1 if passed else 2,
        "excluded_reference_count": 0 if passed else 1,
        "checks": {"descriptor_similarity": "completed"},
        "descriptor_provider": "megaloc",
        "descriptor_version": "test-descriptor-v1",
        "descriptor_checked_count": descriptor_checked,
        "audit_fingerprint": "e" * 64,
        "exclusions": [] if passed else [exclusion],
        "limitations": [],
    }


def _run_attestation(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_root = str(ROOT / "services" / "api" / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, environment.get("PYTHONPATH")) if item
    )
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "attest_reference_index_leakage.py"),
            *arguments,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_leakage_attestation_is_index_bound_safe_and_tamper_evident(
    tmp_path: Path,
) -> None:
    index = _build_index(tmp_path)
    before = open_reference_index(index).diagnostics
    assert before.status == "ready"
    assert before.leakage_status == "not_run"
    assert before.leakage_audit is None

    with pytest.raises(ReferenceIndexBuildError, match="mismatch"):
        write_reference_leakage_attestation(
            index,
            _attestation(index_version="different-index-v1"),
        )
    destination = write_reference_leakage_attestation(index, _attestation())
    after = open_reference_index(index).diagnostics

    assert destination.name == LEAKAGE_ATTESTATION_FILENAME
    assert after.leakage_status == "passed"
    assert after.leakage_audit is not None
    assert after.leakage_audit.pre_index_excluded_reference_count == 1
    serialized = destination.read_text(encoding="utf-8")
    assert "holdout_id" not in serialized
    assert "holdout_sha256" not in serialized
    assert str(tmp_path) not in serialized

    payload = json.loads(serialized)
    payload["checked_reference_count"] = 2
    destination.write_text(json.dumps(payload), encoding="utf-8")
    tampered = open_reference_index(index).diagnostics
    assert tampered.status == "ready"
    assert tampered.leakage_status == "failed"
    assert tampered.leakage_audit is None


def test_attestation_cli_accepts_only_complete_clean_real_descriptor_audit(
    tmp_path: Path,
) -> None:
    index = _build_index(tmp_path, name="cli-index")
    final_report = tmp_path / "final-report.json"
    prior_report = tmp_path / "prior-report.json"
    final_report.write_text(json.dumps(_report()), encoding="utf-8")
    prior_report.write_text(json.dumps(_report(passed=False)), encoding="utf-8")

    completed = _run_attestation(
        "--index",
        str(index),
        "--report",
        str(final_report),
        "--prior-report",
        str(prior_report),
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["pre_index_excluded_reference_count"] == 1
    diagnostics = open_reference_index(index).diagnostics
    assert diagnostics.leakage_status == "passed"
    assert diagnostics.leakage_audit is not None
    assert diagnostics.leakage_audit.source_report_sha256 == hashlib.sha256(
        final_report.read_bytes()
    ).hexdigest()

    rejected_index = _build_index(tmp_path, name="rejected-index")
    incomplete = tmp_path / "incomplete-report.json"
    incomplete.write_text(
        json.dumps(_report(descriptor_checked=0)),
        encoding="utf-8",
    )
    rejected = _run_attestation(
        "--index",
        str(rejected_index),
        "--report",
        str(incomplete),
    )
    assert rejected.returncode == 2
    assert json.loads(rejected.stdout) == {
        "status": "error",
        "code": "leakage_attestation_failed",
    }
    assert open_reference_index(rejected_index).diagnostics.leakage_status == "not_run"


def test_system_intelligence_is_operator_gated_and_privacy_safe(
    client_factory: Any,
    tmp_path: Path,
) -> None:
    assert client_factory().get("/api/v1/system-intelligence").status_code == 404

    index = _build_index(tmp_path, name="observability-index")
    unavailable_client = client_factory(
        app_env="development",
        operator_api_enabled=True,
        phase6c_enabled=True,
        phase6b_enabled=False,
        global_model_enabled=False,
        megaloc_worker_enabled=False,
        reference_index_path=index,
    )
    unavailable = unavailable_client.get("/api/v1/system-intelligence")
    assert unavailable.status_code == 200
    unavailable_body = unavailable.json()
    assert unavailable_body["reference_index"]["count"] == 1
    assert unavailable_body["reference_index"]["status"] == "unavailable"
    assert unavailable_body["reference_index"]["health"] == "degraded"
    assert unavailable_body["reference_index"]["leakage_status"] == "not_run"
    assert unavailable_body["reference_index"]["attributions"] == [
        "manual: Public fixture contributor"
    ]
    capability = unavailable_client.get("/api/v1/capabilities").json()["providers"]
    assert capability["reference_index"]["available"] is False
    assert (
        capability["reference_index"]["reason_code"]
        == "reference_index_leakage_not_passed"
    )

    write_reference_leakage_attestation(index, _attestation())
    ready_client = client_factory(
        app_env="development",
        operator_api_enabled=True,
        phase6c_enabled=True,
        phase6b_enabled=False,
        global_model_enabled=False,
        megaloc_worker_enabled=False,
        reference_index_path=index,
        mapillary_access_token="private-test-token",
    )
    response = ready_client.get("/api/v1/system-intelligence")
    assert response.status_code == 200
    body = response.json()
    model_ids = [item["model_id"] for item in body["models"]]
    assert model_ids == [
        "geoclip_original",
        "geoclip_hierarchical",
        "osv5m",
        "plonk",
        "segformer",
        "paddleocr",
        "rapidocr",
        "megaloc",
        "g3",
        "openai_geo_review",
    ]
    assert body["reference_index"]["status"] == "ready"
    assert body["reference_index"]["health"] == "healthy"
    assert body["reference_index"]["leakage_status"] == "passed"
    assert body["reference_index"]["leakage_audit"]["checked_reference_count"] == 1
    g3 = next(item for item in body["models"] if item["model_id"] == "g3")
    assert (g3["enabled"], g3["available"], g3["status"]) == (
        False,
        False,
        "disabled",
    )
    serialized = response.text
    for forbidden in (
        str(index),
        str(tmp_path),
        "private-test-token",
        "mapillary_access_token",
        "authorization",
        "raw_ocr",
        "latitude",
        "longitude",
    ):
        assert forbidden not in serialized


def test_phase6c_checked_configs_fail_closed_only_when_enabled(tmp_path: Path) -> None:
    common = {
        "database_url": f"sqlite:///{(tmp_path / 'atlaslens.db').as_posix()}",
        "temp_storage_dir": tmp_path / "tmp",
        "phase6b_enabled": False,
        "global_model_enabled": False,
        "megaloc_worker_enabled": False,
        "reference_index_path": tmp_path / "missing-index",
    }
    with pytest.raises(ValueError, match="phase6c_fusion_config_invalid"):
        create_app(
            Settings(
                _env_file=None,
                phase6c_enabled=True,
                phase6c_fusion_config=tmp_path / "missing-fusion.json",
                **common,
            )
        )
    with pytest.raises(ValueError, match="phase6c_ocr_config_invalid"):
        create_app(
            Settings(
                _env_file=None,
                phase6c_enabled=True,
                phase6c_ocr_config=tmp_path / "missing-ocr.json",
                **common,
            )
        )

    disabled = Settings(
        _env_file=None,
        phase6c_enabled=False,
        phase6c_fusion_config=tmp_path / "missing-fusion.json",
        phase6c_ocr_config=tmp_path / "missing-ocr.json",
        operator_api_enabled=True,
        app_env="development",
        **common,
    )
    with TestClient(create_app(disabled), raise_server_exceptions=False) as client:
        response = client.get("/api/v1/system-intelligence")
    assert response.status_code == 200
    assert response.json()["active_pipeline_version"] == "legacy-v1"
