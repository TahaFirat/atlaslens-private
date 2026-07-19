from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from atlaslens_api.cli import main as atlas_cli_main
from atlaslens_api.model_management.cli import build_models_parser
from atlaslens_api.model_management.cli import main as model_cli_main
from atlaslens_api.model_management.errors import ModelManagementError
from atlaslens_api.model_management.manifest import default_manifest_path, load_manifest
from atlaslens_api.model_management.models import ModelManifest
from atlaslens_api.model_management.service import ModelManagementService

ASSETS = (
    "geoclip/model/weights/image_encoder_mlp_weights.pth",
    "geoclip/model/weights/location_encoder_weights.pth",
    "geoclip/model/weights/logit_scale_weights.pth",
    "geoclip/model/gps_gallery/coordinates_100K.csv",
)


def test_top_level_cli_help_lists_phase5_groups(capsys) -> None:
    assert atlas_cli_main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "models" in output
    assert "gazetteer" in output
    assert "benchmark" in output
    assert "acceptance" in output


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_installed_management(root: Path) -> ModelManagementService:
    wheel = root / "source.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in ASSETS:
            archive.writestr(name, b"LAT,LON\n1,2\n" if name.endswith(".csv") else name.encode())
    clip_weights = b"safe-test-safetensors"
    base = load_manifest(default_manifest_path())
    manifest = base.model_copy(
        update={
            "wheel_sha256": _digest(wheel.read_bytes()),
            "clip_weights_sha256": _digest(clip_weights),
        }
    )

    def wheel_fetcher(_url: str, destination: Path) -> None:
        shutil.copyfile(wheel, destination)

    def snapshot_fetcher(spec: ModelManifest, destination: Path) -> str:
        for filename in spec.required_clip_files:
            target = destination / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(clip_weights if filename == spec.clip_weights_filename else b"{}")
        return spec.clip_revision

    service = ModelManagementService(
        root / "cache",
        manifest,
        wheel_fetcher=wheel_fetcher,
        snapshot_fetcher=snapshot_fetcher,
    )
    service.install()
    return service


def test_atomic_install_verify_and_info(tmp_path: Path) -> None:
    service = build_installed_management(tmp_path)
    receipt = service.verify()
    assert receipt.resolved_clip_revision == service.manifest.clip_revision
    assert set(receipt.geoclip_asset_sha256) == {
        "image_encoder_mlp_weights.pth",
        "location_encoder_weights.pth",
        "logit_scale_weights.pth",
        "coordinates_100K.csv",
    }
    info = service.info(verify=True)
    assert info.status == "verified"
    assert info.storage_size_bytes > 0
    assert str(tmp_path) not in info.model_dump_json()
    assert not any(
        path.name.startswith(".geoclip.installing") for path in service.cache_root.iterdir()
    )


def test_tamper_is_detected(tmp_path: Path) -> None:
    service = build_installed_management(tmp_path)
    (service.geoclip_assets_directory() / "location_encoder_weights.pth").write_bytes(b"tampered")
    with pytest.raises(ModelManagementError, match="checksum_mismatch"):
        service.verify()
    assert service.info(verify=True).status == "invalid"


def test_non_weight_snapshot_tamper_is_detected(tmp_path: Path) -> None:
    service = build_installed_management(tmp_path)
    relative = next(
        item
        for item in service.manifest.required_clip_files
        if item != service.manifest.clip_weights_filename
    )
    (service.snapshot_directory() / relative).write_bytes(b"tampered-config")
    with pytest.raises(ModelManagementError, match="checksum_mismatch"):
        service.verify()


def test_receipt_cannot_reauthorize_modified_official_wheel_assets(
    tmp_path: Path,
) -> None:
    service = build_installed_management(tmp_path)
    receipt_path = service.model_directory / "installation.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    filename = next(iter(receipt["geoclip_asset_sha256"]))
    (service.geoclip_assets_directory() / filename).write_bytes(b"tampered")
    receipt["geoclip_asset_sha256"][filename] = hashlib.sha256(b"tampered").hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ModelManagementError, match="checksum_mismatch"):
        service.verify()


def test_install_rejects_a_different_resolved_revision(tmp_path: Path) -> None:
    wheel = tmp_path / "source.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in ASSETS:
            archive.writestr(name, b"LAT,LON\n1,2\n")
    base = load_manifest(default_manifest_path())
    manifest = base.model_copy(update={"wheel_sha256": _digest(wheel.read_bytes())})

    def wheel_fetcher(_url: str, destination: Path) -> None:
        shutil.copyfile(wheel, destination)

    def wrong_revision(spec: ModelManifest, destination: Path) -> str:
        for filename in spec.required_clip_files:
            target = destination / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"not-used")
        return "different-revision"

    service = ModelManagementService(
        tmp_path / "cache",
        manifest,
        wheel_fetcher=wheel_fetcher,
        snapshot_fetcher=wrong_revision,
    )
    with pytest.raises(ModelManagementError, match="revision_mismatch"):
        service.install()
    assert not service.model_directory.exists()


def test_bad_download_leaves_no_installation(tmp_path: Path) -> None:
    manifest = load_manifest(default_manifest_path())

    def wrong_wheel(_url: str, destination: Path) -> None:
        destination.write_bytes(b"wrong")

    service = ModelManagementService(tmp_path, manifest, wheel_fetcher=wrong_wheel)
    with pytest.raises(ModelManagementError, match="checksum_mismatch"):
        service.install()
    assert not service.model_directory.exists()


def test_interrupted_install_removes_partial_staging(tmp_path: Path) -> None:
    manifest = load_manifest(default_manifest_path())

    def interrupted(_url: str, _destination: Path) -> None:
        raise KeyboardInterrupt

    service = ModelManagementService(
        tmp_path / "cache", manifest, wheel_fetcher=interrupted
    )
    with pytest.raises(KeyboardInterrupt):
        service.install()
    assert not service.model_directory.exists()
    assert not any(service.cache_root.glob(".geoclip.installing-*"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wheel_filename", "../outside.whl"),
        ("clip_snapshot_directory", "../outside"),
        ("geoclip_assets_directory", "../outside"),
    ],
)
def test_receipt_path_traversal_is_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    service = build_installed_management(tmp_path)
    receipt_path = service.model_directory / "installation.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ModelManagementError, match="unsafe_installation_path"):
        service.verify()


def test_remove_requires_confirmation_and_rejects_symlink(tmp_path: Path) -> None:
    service = build_installed_management(tmp_path)
    with pytest.raises(ModelManagementError, match="confirmation_required"):
        service.remove(confirmed=False)
    service.remove(confirmed=True)
    assert not service.model_directory.exists()


def test_test_and_benchmark_are_diagnostics_not_accuracy_claims(tmp_path: Path) -> None:
    service = build_installed_management(tmp_path)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    assert service.test(image, lambda _path: 5).status == "passed"
    failed = service.test(image, lambda _path: 0)
    assert (failed.status, failed.reason_code) == ("failed", "insufficient_hypotheses")
    benchmark = service.benchmark([image, image], lambda _path: 5)
    assert benchmark.status == "completed"
    assert benchmark.note == "diagnostic_timings_not_a_performance_claim"


def test_model_smoke_preserves_a_safe_provider_subreason(tmp_path: Path) -> None:
    service = build_installed_management(tmp_path)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")

    def fail(_path: Path) -> int:
        raise ModelManagementError(
            "invalid_model_output", subreason_code="gps_shape_invalid"
        )

    result = service.test(image, fail)
    assert result.status == "failed"
    assert result.reason_code == "invalid_model_output"
    assert result.subreason_code == "gps_shape_invalid"
    assert str(tmp_path) not in result.model_dump_json()


def test_benchmark_cli_accepts_runs_warmup_and_device() -> None:
    args = build_models_parser().parse_args(
        [
            "benchmark",
            "geoclip",
            "--image",
            "fixture.jpg",
            "--runs",
            "5",
            "--warmup-runs",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert (args.runs, args.warmup_runs, args.device) == (5, 2, "cpu")


def test_failed_provider_smoke_returns_nonzero(tmp_path: Path, capsys) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"not-decoded-before-installation-check")
    exit_code = model_cli_main(
        [
            "--cache-root",
            str(tmp_path / "empty-cache"),
            "test",
            "geoclip",
            "--image",
            str(image),
        ]
    )
    assert exit_code == 1
    output = capsys.readouterr().out
    assert '"status": "failed"' in output
    assert '"reason_code": "model_not_installed"' in output
