from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from atlaslens_api.inference.providers import CustomModelArtifact
from atlaslens_api.model_management.cli import run_models_command
from atlaslens_api.trained_artifacts import integration as trained_integration
from atlaslens_api.trained_artifacts.errors import TrainedArtifactError
from atlaslens_api.trained_artifacts.manager import TrainedArtifactManager
from atlaslens_api.trained_artifacts.models import DeploymentMode
from atlaslens_api.trained_artifacts.onnx_runtime import OnnxCoordinateRuntime


def _manifest_payload(artifact: bytes, **changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "atlaslens-trained-artifact-v1",
        "model_id": "atlaslens-custom-geolocation",
        "provider_id": "atlaslens-custom-v1",
        "model_version": "1.0.0",
        "implementation_revision": "custom-adapter-v1",
        "task": "global_geolocation",
        "artifact_format": "onnx",
        "runtime_adapter": "onnx-coordinate-v1",
        "artifact_file": "model.onnx",
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "artifact_size_bytes": len(artifact),
        "input": {
            "tensor_name": "image",
            "width": 384,
            "height": 384,
            "color_space": "RGB",
            "layout": "NCHW",
            "dtype": "float32",
            "resize_method": "bicubic",
            "normalization": "mean_std",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "preprocessing_version": "custom-rgb-384-v1",
        },
        "output": {
            "type": "top_k_coordinates",
            "coordinate_order": "lat_lon",
            "top_k": 5,
            "score_type": "uncalibrated_model_logit",
            "output_schema_version": "coordinate-topk-v1",
            "coordinates_output": "coordinates",
            "scores_output": "scores",
        },
        "training": {
            "dataset_fingerprint": "a" * 64,
            "code_revision": "1234567",
            "completed_at": datetime(2026, 7, 12, tzinfo=UTC).isoformat(),
        },
        "evaluation": {
            "report_required_before_promotion": True,
            "minimum_sample_count": 30,
            "primary_minimum_sample_count": 60,
            "maximum_country_top1_regression": 0.05,
            "maximum_recall_200km_regression": 0.05,
            "maximum_median_error_increase": 25.0,
            "maximum_latency_p95_ms": 5000.0,
        },
        "license": {
            "name": "operator-owned",
            "status": "approved",
            "commercial_use": "allowed",
            "review_reference": "license-review-2026-07-12",
        },
    }
    payload.update(changes)
    return payload


def _write_package(root: Path, artifact: bytes = b"reviewed-onnx-test-bytes") -> Path:
    (root / "model.onnx").write_bytes(artifact)
    manifest = root / "model-manifest.yaml"
    manifest.write_text(
        json.dumps(_manifest_payload(artifact), indent=2), encoding="utf-8"
    )
    return manifest


def _promotion_payload(identity: str, **changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "model_id": "atlaslens-custom-geolocation",
        "model_version": "1.0.0",
        "artifact_identity": identity,
        "evaluation_fingerprint": "b" * 64,
        "paired_sample_set_sha256": "c" * 64,
        "sample_count": 60,
        "baseline_provider_id": "geoclip-global-v1",
        "candidate_provider_id": "atlaslens-custom-v1",
        "country_top1_delta": 0.01,
        "recall_200km_delta": 0.02,
        "median_error_increase_km": -5.0,
        "latency_p95_ms": 1200.0,
        "safety_checks_passed": True,
        "output_schema_compatible": True,
        "lineage_verified": True,
        "license_approved": True,
        "subgroup_regressions_passed": True,
        "runtime_isolation_verified": True,
        "operator_approved": True,
        "limitations": ["held_out_report_not_a_location_guarantee"],
    }
    payload.update(changes)
    return payload


def test_register_verify_and_list_enter_shadow_mode(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manager = TrainedArtifactManager(tmp_path / "cache")

    registered = manager.register_local(_write_package(package))
    assert registered.mode == DeploymentMode.DISABLED
    assert registered.verified is False

    verified = manager.verify("atlaslens-custom-geolocation")
    assert verified.mode == DeploymentMode.SHADOW
    assert verified.verified is True
    assert manager.list()[0].artifact_identity == verified.artifact_identity
    assert "redacted" in repr(manager.verified_artifact(verified.model_id))


def test_registration_rejects_checksum_mismatch(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    artifact = b"artifact"
    manifest = _write_package(package, artifact)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifact_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainedArtifactError, match="artifact_identity_mismatch"):
        TrainedArtifactManager(tmp_path / "cache").register_local(manifest)


def test_registration_rejects_unsafe_or_pickle_format(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    artifact = b"artifact"
    (package / "model.onnx").write_bytes(artifact)
    payload = _manifest_payload(artifact, artifact_format="torchscript")
    manifest = package / "model-manifest.yaml"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainedArtifactError, match="invalid_manifest"):
        TrainedArtifactManager(tmp_path / "cache").register_local(manifest)


def test_registration_rejects_incomplete_preprocessing_contract(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    artifact = b"artifact"
    (package / "model.onnx").write_bytes(artifact)
    payload = _manifest_payload(artifact)
    input_contract = dict(payload["input"])  # type: ignore[arg-type]
    input_contract.pop("std")
    payload["input"] = input_contract
    manifest = package / "model-manifest.yaml"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainedArtifactError, match="invalid_manifest"):
        TrainedArtifactManager(tmp_path / "cache").register_local(manifest)


def test_registration_rejects_incomplete_output_contract(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    artifact = b"artifact"
    (package / "model.onnx").write_bytes(artifact)
    payload = _manifest_payload(artifact)
    output_contract = dict(payload["output"])  # type: ignore[arg-type]
    output_contract.pop("scores_output")
    payload["output"] = output_contract
    manifest = package / "model-manifest.yaml"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainedArtifactError, match="invalid_manifest"):
        TrainedArtifactManager(tmp_path / "cache").register_local(manifest)


def test_registration_rejects_probability_claim_for_uncalibrated_model(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    artifact = b"artifact"
    (package / "model.onnx").write_bytes(artifact)
    payload = _manifest_payload(artifact)
    output_contract = dict(payload["output"])  # type: ignore[arg-type]
    output_contract["score_type"] = "probability"
    payload["output"] = output_contract
    manifest = package / "model-manifest.yaml"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainedArtifactError, match="invalid_manifest"):
        TrainedArtifactManager(tmp_path / "cache").register_local(manifest)


def test_verification_detects_post_registration_tamper(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manager = TrainedArtifactManager(tmp_path / "cache")
    manager.register_local(_write_package(package))
    registered_artifact = (
        tmp_path
        / "cache"
        / "trained-artifacts"
        / "atlaslens-custom-geolocation"
        / "model.onnx"
    )
    registered_artifact.write_bytes(b"tampered")

    with pytest.raises(TrainedArtifactError, match="artifact_identity_mismatch"):
        manager.verify("atlaslens-custom-geolocation")


def test_runtime_payload_rejects_same_size_post_verification_tamper(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manager = TrainedArtifactManager(tmp_path / "cache")
    manager.register_local(_write_package(package))
    manager.verify("atlaslens-custom-geolocation")
    registered_artifact = (
        tmp_path
        / "cache"
        / "trained-artifacts"
        / "atlaslens-custom-geolocation"
        / "model.onnx"
    )
    registered_artifact.write_bytes(b"x" * registered_artifact.stat().st_size)

    with pytest.raises(TrainedArtifactError, match="artifact_identity_mismatch"):
        manager.verified_artifact_payload(
            "atlaslens-custom-geolocation", maximum_bytes=1024
        )


def test_promotion_fails_closed_then_writes_receipts(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manager = TrainedArtifactManager(tmp_path / "cache")
    manager.register_local(_write_package(package))
    verified = manager.verify("atlaslens-custom-geolocation")
    report = tmp_path / "promotion.json"
    report.write_text(
        json.dumps(_promotion_payload(verified.artifact_identity, operator_approved=False)),
        encoding="utf-8",
    )

    with pytest.raises(TrainedArtifactError, match="promotion_gate_failed"):
        manager.promote(
            verified.model_id,
            from_mode=DeploymentMode.SHADOW,
            to_mode=DeploymentMode.CANDIDATE,
            report_path=report,
        )

    report.write_text(
        json.dumps(_promotion_payload(verified.artifact_identity)), encoding="utf-8"
    )
    candidate = manager.promote(
        verified.model_id,
        from_mode=DeploymentMode.SHADOW,
        to_mode=DeploymentMode.CANDIDATE,
        report_path=report,
    )
    assert candidate.to_mode == DeploymentMode.CANDIDATE
    assert manager.info(verified.model_id).mode == DeploymentMode.CANDIDATE
    primary = manager.promote(
        verified.model_id,
        from_mode=DeploymentMode.CANDIDATE,
        to_mode=DeploymentMode.PRIMARY,
        report_path=report,
    )
    assert primary.to_mode == DeploymentMode.PRIMARY
    assert manager.verified_artifact(verified.model_id).receipt.mode == DeploymentMode.PRIMARY


def test_primary_promotion_requires_runtime_isolation(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manager = TrainedArtifactManager(tmp_path / "cache")
    manager.register_local(_write_package(package))
    verified = manager.verify("atlaslens-custom-geolocation")
    report = tmp_path / "promotion.json"
    report.write_text(
        json.dumps(_promotion_payload(verified.artifact_identity)), encoding="utf-8"
    )
    manager.promote(
        verified.model_id,
        from_mode=DeploymentMode.SHADOW,
        to_mode=DeploymentMode.CANDIDATE,
        report_path=report,
    )
    report.write_text(
        json.dumps(
            _promotion_payload(
                verified.artifact_identity, runtime_isolation_verified=False
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrainedArtifactError, match="promotion_gate_failed"):
        manager.promote(
            verified.model_id,
            from_mode=DeploymentMode.CANDIDATE,
            to_mode=DeploymentMode.PRIMARY,
            report_path=report,
        )


def test_unregister_requires_confirmation(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manager = TrainedArtifactManager(tmp_path / "cache")
    manager.register_local(_write_package(package))

    with pytest.raises(TrainedArtifactError, match="confirmation_required"):
        manager.unregister("atlaslens-custom-geolocation", confirmed=False)
    manager.unregister("atlaslens-custom-geolocation", confirmed=True)
    assert manager.info("atlaslens-custom-geolocation").status == "not_registered"


def test_operator_cli_register_verify_promote_and_unregister(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manifest = _write_package(package)
    cache = tmp_path / "cache"
    prefix = ["--cache-root", str(cache)]

    registered = run_models_command([*prefix, "register-local", "--manifest", str(manifest)])
    assert registered["status"] == "registered"
    verified = run_models_command(
        [*prefix, "verify", "atlaslens-custom-geolocation"]
    )
    assert verified["mode"] == "shadow"
    smoke_image = package / "smoke.jpg"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(smoke_image)
    with monkeypatch.context() as missing_runtime:
        missing_runtime.setattr(
            trained_integration.importlib.util, "find_spec", lambda _: None
        )
        with pytest.raises(TrainedArtifactError, match="runtime_unavailable"):
            run_models_command(
                [
                    *prefix,
                    "test",
                    "atlaslens-custom-geolocation",
                    "--image",
                    str(smoke_image),
                    "--device",
                    "cpu",
                ]
            )
    identity = str(verified["artifact_identity"])
    report = tmp_path / "promotion.json"
    report.write_text(json.dumps(_promotion_payload(identity)), encoding="utf-8")
    promoted = run_models_command(
        [
            *prefix,
            "promote",
            "atlaslens-custom-geolocation",
            "--from",
            "shadow",
            "--to",
            "candidate",
            "--report",
            str(report),
        ]
    )
    assert promoted["to"] == "candidate"
    removed = run_models_command(
        [*prefix, "unregister", "atlaslens-custom-geolocation", "--yes"]
    )
    assert removed["status"] == "unregistered"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reviewed_onnx_runtime_contract_uses_model_bytes_and_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manager = TrainedArtifactManager(tmp_path / "cache")
    manager.register_local(_write_package(package))
    receipt = manager.verify("atlaslens-custom-geolocation")
    selected_providers: list[list[str]] = []

    class Session:
        def __init__(self, payload: bytes, *, sess_options: object, providers: list[str]):
            assert isinstance(payload, bytes)
            assert sess_options is not None
            selected_providers.append(providers)

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="image")]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="coordinates"), SimpleNamespace(name="scores")]

        def run(
            self, outputs: list[str], inputs: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            assert outputs == ["coordinates", "scores"]
            assert inputs["image"].shape == (1, 3, 384, 384)
            return [
                np.asarray([[[38.7, 35.4], [41.0, 29.0]]], dtype=np.float32),
                np.asarray([[0.7, 0.3]], dtype=np.float32),
            ]

    class Options:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    fake_runtime = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        SessionOptions=Options,
        InferenceSession=Session,
    )
    monkeypatch.setattr(
        "atlaslens_api.trained_artifacts.onnx_runtime.importlib.import_module",
        lambda name: fake_runtime if name == "onnxruntime" else None,
    )
    runtime = OnnxCoordinateRuntime(
        manager,
        "atlaslens-custom-geolocation",
        device=device,  # type: ignore[arg-type]
    )
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (20, 40, 60)).save(buffer, format="PNG")
    candidates = runtime.predict(
        artifact=CustomModelArtifact(
            provider_id="atlaslens-custom-v1",
            artifact_id="atlaslens-custom-geolocation",
            artifact_digest=receipt.artifact_identity,
            model_name="AtlasLens custom",
            model_revision="1.0.0",
            runtime_revision="custom-adapter-v1",
            adapter_id="onnx-coordinate-v1",
            verified=True,
            adapter_supported=True,
            supported_devices=("cpu", "cuda"),
            max_input_bytes=1024 * 1024,
            output_dtype="float32",
            score_semantics="uncalibrated_model_score",
            normalization_method="custom-rgb-384-v1",
            calibration_state="uncalibrated",
        ),
        image_bytes=buffer.getvalue(),
        width=64,
        height=48,
        top_k=1,
        device=device,  # type: ignore[arg-type]
    )
    assert candidates[0]["raw_score"] == pytest.approx(0.7)  # type: ignore[index]
    expected = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    assert selected_providers[0][0] == expected
