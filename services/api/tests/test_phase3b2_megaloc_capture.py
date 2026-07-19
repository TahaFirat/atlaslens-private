from __future__ import annotations

import argparse
import ast
import hashlib
import json
import socket
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from atlaslens_api.capture_import import CAPTURE_COMMANDS, FFMPEG_NOT_AVAILABLE_MESSAGE
from atlaslens_api.capture_import.errors import CaptureImportError
from atlaslens_api.capture_import.geo import interpolate_track, sample_frames, validate_track
from atlaslens_api.capture_import.models import (
    CapturePlan,
    PrivacyReview,
    RevocationUpdate,
    SyncedFrame,
    TrackPoint,
)
from atlaslens_api.capture_import.sources import read_gpx_track
from atlaslens_api.corpus_cli import build_parser
from atlaslens_api.corpus_cli import main as corpus_cli_main
from atlaslens_api.corpus_index.models import ProviderApproval
from atlaslens_api.corpus_index.providers import ProductionDescriptorRegistry
from atlaslens_api.corpus_pipeline import AssetSafetyError
from atlaslens_api.megaloc_adapter import (
    MegaLocAdapterConfig,
    MegaLocAdapterError,
    MegaLocApproval,
    MegaLocDescriptorProvider,
    load_megaloc_config,
)
from atlaslens_api.megaloc_adapter import provider as megaloc_provider_module

EXPECTED_CAPTURE_COMMANDS = (
    "inspect-capture",
    "import-capture",
    "sync-gpx",
    "sample-route",
    "privacy-review",
    "build-capture-manifest",
)


def _capture_plan_payload() -> dict[str, object]:
    return {
        "schema_version": "atlaslens-first-party-capture-plan-v1",
        "plan_id": "phase3b2-synthetic-plan",
        "input_mode": "ordered_frames_gpx",
        "corpus_version": "atlaslens-turkiye-corpus-phase3b2-synthetic-v1",
        "media_locators": ["frames/0001.png", "frames/0002.png"],
        "track_locator": "tracks/route.gpx",
        "frame_timestamps": [
            "2026-07-17T08:00:00Z",
            "2026-07-17T08:00:10Z",
        ],
        "acquisition_timestamp": "2026-07-17T09:00:00Z",
        "capture_run_id": "synthetic-run-1",
        "sequence_id": "synthetic-sequence-1",
        "device_id": "synthetic-device-1",
        "contributor_or_owner": "phase3b2-synthetic-owner",
        "province_code": "TR-38",
        "spatial_split": "phase3b2-synthetic-cell",
        "sample_distance_m": 50,
        "rights": {
            "source_policy_version": "2026-07-16",
            "source_policy_evidence_date": "2026-07-17",
            "license_identifier": "AtlasLens-first-party-synthetic-fixture",
            "license_url": "https://fixture.invalid/license",
            "attribution_text": "Generated synthetic fixture; never production imagery",
            "capture_policy_sha256": "a" * 64,
            "contributor_assignment_sha256": "b" * 64,
            "privacy_notice_sha256": "c" * 64,
        },
    }


def _megaloc_config_payload() -> dict[str, object]:
    return {
        "schema": "atlaslens-megaloc-phase3b2-v1",
        "paths": {
            "source_dir": "source",
            "source_code_path": "source/model.py",
            "source_license_path": "source/LICENSE",
            "artifact_path": "artifacts/model.safetensors",
            "receipt_path": "receipts/artifact.json",
            "smoke_receipt_path": "receipts/smoke.json",
        },
        "source": {"code_sha256": "1" * 64, "revision": "synthetic-revision"},
        "artifact": {
            "sha256": "2" * 64,
            "size": 128,
            "format": "safetensors",
            "receipt_sha256": "3" * 64,
            "model_id": "megaloc",
            "model_revision": "synthetic-model-revision",
        },
        "licenses": {
            "source_license_sha256": "4" * 64,
            "source_license_spdx": "Apache-2.0",
            "weight_license_spdx": "Apache-2.0",
        },
        "descriptor": {
            "version": "megaloc-7cb9f797-max560-imagenet-v1",
            "dimension": 8448,
            "preprocessing_id": "rgb-imagenet-max560-multiple14-v1",
            "maximum_edge": 560,
        },
        "execution": {
            "device": "cpu",
            "max_batch_size": 4,
            "max_input_bytes": 1048576,
            "max_image_pixels": 1000000,
            "deterministic_algorithms": True,
            "reduce_batch_on_cuda_oom": True,
        },
        "approval": {
            "artifact_hash_approved": True,
            "source_code_approved": True,
            "source_license_approved": True,
            "weight_license_approved": True,
            "descriptor_contract_approved": True,
            "preprocessing_approved": True,
            "production_use_approved": True,
            "approval_record_id": "approval-synthetic",
            "weight_license_record_id": "license-synthetic",
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approved_megaloc_config(
    root: Path,
    *,
    device: str = "cpu",
    max_batch_size: int = 4,
) -> MegaLocAdapterConfig:
    source_dir = root / "source"
    artifact_dir = root / "artifacts"
    receipt_dir = root / "receipts"
    source_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    receipt_dir.mkdir()
    source_code = source_dir / "model.py"
    source_license = source_dir / "LICENSE"
    artifact = artifact_dir / "model.safetensors"
    receipt = receipt_dir / "artifact.json"
    source_code.write_text("# generated test-only source marker\n", encoding="utf-8")
    source_license.write_text("Apache-2.0 generated fixture\n", encoding="utf-8")
    artifact.write_bytes(b"phase3b2-synthetic-safetensors-placeholder")
    artifact_sha256 = _sha256(artifact)
    model_revision = "7cb9f797-synthetic"
    descriptor_version = (
        f"megaloc-{model_revision[:8]}-{artifact_sha256[:12]}-max560-imagenet-rgb-v1"
    )
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-model-receipt-v1",
                "provider": "megaloc",
                "source_revision": "synthetic-source-revision",
                "model_id": "megaloc",
                "model_revision": model_revision,
                "weight_file": artifact.name,
                "weight_size": artifact.stat().st_size,
                "weight_sha256": artifact_sha256,
                "license": "Apache-2.0",
                "datasets_downloaded": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return MegaLocAdapterConfig(
        project_root=root,
        source_dir=source_dir,
        source_code_path=source_code,
        source_code_sha256=_sha256(source_code),
        source_revision="synthetic-source-revision",
        source_license_path=source_license,
        source_license_sha256=_sha256(source_license),
        source_license_spdx="Apache-2.0",
        artifact_path=artifact,
        artifact_sha256=artifact_sha256,
        artifact_size=artifact.stat().st_size,
        artifact_format="safetensors",
        receipt_path=receipt,
        receipt_sha256=_sha256(receipt),
        smoke_receipt_path=receipt_dir / "smoke.json",
        model_id="megaloc",
        model_revision=model_revision,
        weight_license_spdx="Apache-2.0",
        descriptor_version=descriptor_version,
        descriptor_dimension=8448,
        preprocessing_id="rgb-imagenet-max560-multiple14-v1",
        maximum_edge=560,
        device=device,  # type: ignore[arg-type]
        max_batch_size=max_batch_size,
        max_input_bytes=1024 * 1024,
        max_image_pixels=1_000_000,
        deterministic_algorithms=True,
        reduce_batch_on_cuda_oom=True,
        approval=MegaLocApproval(
            artifact_hash_approved=True,
            source_code_approved=True,
            source_license_approved=True,
            weight_license_approved=True,
            descriptor_contract_approved=True,
            preprocessing_approved=True,
            production_use_approved=True,
            approval_record_id="synthetic-approval",
            weight_license_record_id="synthetic-license",
        ),
    )


class _SyntheticMegaLocBackend:
    def __init__(
        self,
        device: str,
        *,
        invalid: str | None = None,
        oom_above: int | None = None,
        always_oom: bool = False,
    ) -> None:
        self._device = device
        self.invalid = invalid
        self.oom_above = oom_above
        self.always_oom = always_oom
        self.calls: list[int] = []
        self.recoveries = 0
        self.closed = False

    @property
    def device(self) -> str:
        return self._device

    def describe_batch(self, locators: Sequence[Path]) -> NDArray[np.float32]:
        self.calls.append(len(locators))
        if self.always_oom or (self.oom_above is not None and len(locators) > self.oom_above):
            raise MegaLocAdapterError("MEGALOC_CUDA_OOM")
        columns = 16 if self.invalid == "dimension" else 8448
        value = np.zeros((len(locators), columns), dtype=np.float32)
        value[:, 0] = 1.0
        if self.invalid == "nan":
            value[0, 0] = np.nan
        elif self.invalid == "inf":
            value[0, 0] = np.inf
        return value

    def recover_cuda_oom(self) -> None:
        self.recoveries += 1

    def close(self) -> None:
        self.closed = True


def _provider_with_backend(
    config: MegaLocAdapterConfig,
    **backend_options: object,
) -> tuple[MegaLocDescriptorProvider, list[_SyntheticMegaLocBackend]]:
    created: list[_SyntheticMegaLocBackend] = []

    def factory(
        _config: MegaLocAdapterConfig,
        device: str,
    ) -> _SyntheticMegaLocBackend:
        backend = _SyntheticMegaLocBackend(device, **backend_options)  # type: ignore[arg-type]
        created.append(backend)
        return backend

    return MegaLocDescriptorProvider(config, backend_factory=factory), created


def _write_gpx(path: Path, points: Sequence[tuple[str, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f'<trkpt lat="{latitude}" lon="{longitude}"><time>{timestamp}</time></trkpt>'
        for timestamp, latitude, longitude in points
    )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<gpx version="1.1"><trk><trkseg>' + rows + "</trkseg></trk></gpx>",
        encoding="utf-8",
    )


def _deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Phase 3B2 attempted network access")

    monkeypatch.setattr(socket, "create_connection", refuse_network)
    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_network)


def _parser_commands(parser: argparse.ArgumentParser) -> set[str]:
    action = next(item for item in parser._actions if isinstance(item, argparse._SubParsersAction))
    return set(action.choices)


def test_capture_cli_exposes_exact_six_user_commands() -> None:
    assert CAPTURE_COMMANDS == EXPECTED_CAPTURE_COMMANDS
    assert set(EXPECTED_CAPTURE_COMMANDS) <= _parser_commands(build_parser())


def test_phase3b2_tests_use_only_pytest_temporary_storage(tmp_path: Path) -> None:
    fixture = tmp_path / "synthetic-only.fixture"
    fixture.write_bytes(b"atlaslens-phase3b2-generated-test-fixture")
    assert fixture.is_relative_to(tmp_path)
    assert fixture.read_bytes().startswith(b"atlaslens-phase3b2")


def test_capture_plan_rejects_traversal_and_timestamp_count_mismatch() -> None:
    traversal = _capture_plan_payload()
    traversal["media_locators"] = ["../outside.png"]
    with pytest.raises(ValidationError, match="contained relative path"):
        CapturePlan.model_validate(traversal)

    mismatch = _capture_plan_payload()
    mismatch["frame_timestamps"] = ["2026-07-17T08:00:00Z"]
    with pytest.raises(ValidationError, match="timestamp count"):
        CapturePlan.model_validate(mismatch)


def test_capture_models_reject_invalid_gps_and_require_explicit_privacy_receipts() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 90"):
        TrackPoint(
            timestamp=datetime(2026, 7, 17, tzinfo=UTC),
            latitude=91.0,
            longitude=35.0,
            accuracy_m=5.0,
        )
    assert PrivacyReview().state == "pending"
    with pytest.raises(ValidationError, match="explicit review receipt"):
        PrivacyReview(state="approved")
    approved = PrivacyReview(
        state="approved",
        reviewed_by="synthetic-reviewer",
        reviewed_at=datetime(2026, 7, 17, tzinfo=UTC),
        decision_receipt_sha256="d" * 64,
    )
    assert approved.state == "approved"
    revoked = RevocationUpdate(
        asset_id="synthetic-asset",
        request_id="synthetic-request",
        receipt_sha256="e" * 64,
        revoked_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    assert revoked.asset_id == "synthetic-asset"


def test_megaloc_config_rejects_wrong_dimension_and_path_traversal(tmp_path: Path) -> None:
    wrong_dimension = _megaloc_config_payload()
    descriptor = dict(wrong_dimension["descriptor"])  # type: ignore[arg-type]
    descriptor["dimension"] = 1024
    wrong_dimension["descriptor"] = descriptor
    config_path = tmp_path / "wrong-dimension.json"
    config_path.write_text(json.dumps(wrong_dimension), encoding="utf-8")
    with pytest.raises(MegaLocAdapterError) as dimension_error:
        load_megaloc_config(config_path, project_root=tmp_path)
    assert dimension_error.value.code == "MEGALOC_CONFIG_INVALID"

    traversal = _megaloc_config_payload()
    paths = dict(traversal["paths"])  # type: ignore[arg-type]
    paths["artifact_path"] = "../outside.safetensors"
    traversal["paths"] = paths
    config_path = tmp_path / "traversal.json"
    config_path.write_text(json.dumps(traversal), encoding="utf-8")
    with pytest.raises(MegaLocAdapterError) as traversal_error:
        load_megaloc_config(config_path, project_root=tmp_path)
    assert traversal_error.value.code == "MEGALOC_PATH_INVALID"
    assert str(tmp_path) not in str(traversal_error.value)


def test_capture_error_is_safe_and_path_free() -> None:
    error = CaptureImportError("capture_input_missing")
    assert str(error) == "capture_input_missing"


def test_missing_ffmpeg_message_is_exact_unicode_and_never_offers_download() -> None:
    expected = (
        "FFMPEG_NOT_AVAILABLE "
        + chr(0x2014)
        + " provide an explicit existing safe executable path; "
        "AtlasLens will not download FFmpeg."
    )
    assert expected == FFMPEG_NOT_AVAILABLE_MESSAGE
    assert chr(0x2014) in FFMPEG_NOT_AVAILABLE_MESSAGE
    assert chr(0x00E2) not in FFMPEG_NOT_AVAILABLE_MESSAGE


def test_capture_cli_missing_ffmpeg_emits_exact_safe_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = tmp_path / "private-video-input"
    work_root = tmp_path / "private-video-work"
    input_root.mkdir()
    plan = _capture_plan_payload()
    plan.update(
        {
            "input_mode": "video",
            "media_locators": [],
            "track_locator": None,
            "frame_timestamps": None,
            "video_locator": "private-video.mp4",
            "capture_started_at": "2026-07-17T08:00:00Z",
            "video_duration_seconds": 5.0,
            "video_latitude": 38.7205,
            "video_longitude": 35.4826,
            "video_coordinate_accuracy_m": 5.0,
        }
    )
    (input_root / "capture-plan.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )

    exit_code = corpus_cli_main(
        [
            "inspect-capture",
            "--input-root",
            str(input_root),
            "--work-root",
            str(work_root),
            "--plan",
            "capture-plan.json",
        ]
    )

    captured = capsys.readouterr()
    expected = {
        "status": "error",
        "code": "ffmpeg_not_available",
        "message": FFMPEG_NOT_AVAILABLE_MESSAGE,
    }
    assert exit_code == 4
    assert captured.out == ""
    assert captured.err == json.dumps(expected, sort_keys=True) + "\n"
    assert str(input_root) not in captured.err
    assert "38.7205" not in captured.err
    assert "35.4826" not in captured.err


def test_capture_cli_other_errors_never_leak_paths_or_raw_gps(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = tmp_path / "private-raw-gps-input"
    input_root.mkdir()
    sentinel = "../private-gps-38.123456-35.654321.json"

    exit_code = corpus_cli_main(
        [
            "inspect-capture",
            "--input-root",
            str(input_root),
            "--work-root",
            str(tmp_path / "private-work"),
            "--plan",
            sentinel,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 4
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "error",
        "code": "capture_plan_invalid",
    }
    assert str(input_root) not in captured.err
    assert sentinel not in captured.err
    assert "38.123456" not in captured.err
    assert "35.654321" not in captured.err


def test_gpx_interpolation_timestamp_bounds_invalid_gps_and_impossible_speed(
    tmp_path: Path,
) -> None:
    plan = CapturePlan.model_validate(_capture_plan_payload())
    track_path = tmp_path / "tracks" / "route.gpx"
    _write_gpx(
        track_path,
        (
            ("2026-07-17T08:00:00Z", 38.7205, 35.4826),
            ("2026-07-17T08:00:10Z", 38.7207, 35.4828),
        ),
    )
    points = read_gpx_track(tmp_path, "tracks/route.gpx", plan)
    validated = validate_track(
        points,
        max_speed_kmh=plan.max_speed_kmh,
        max_route_distance_km=plan.max_route_distance_km,
    )
    interpolated = interpolate_track(
        validated,
        datetime(2026, 7, 17, 8, 0, 5, tzinfo=UTC),
        max_gap_seconds=plan.max_interpolation_gap_seconds,
    )
    assert interpolated.latitude == pytest.approx(38.7206)
    assert interpolated.longitude == pytest.approx(35.4827)
    with pytest.raises(CaptureImportError, match="frame_timestamp_outside_track"):
        interpolate_track(
            validated,
            datetime(2026, 7, 17, 7, 59, 59, tzinfo=UTC),
            max_gap_seconds=plan.max_interpolation_gap_seconds,
        )

    _write_gpx(track_path, (("2026-07-17T08:00:00Z", 91.0, 35.0),))
    with pytest.raises(CaptureImportError, match="gpx_point_invalid"):
        read_gpx_track(tmp_path, "tracks/route.gpx", plan)

    impossible = (
        TrackPoint(
            timestamp=datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC),
            latitude=38.0,
            longitude=35.0,
            accuracy_m=5.0,
        ),
        TrackPoint(
            timestamp=datetime(2026, 7, 17, 8, 0, 1, tzinfo=UTC),
            latitude=39.0,
            longitude=35.0,
            accuracy_m=5.0,
        ),
    )
    with pytest.raises(CaptureImportError, match="track_speed_limit_exceeded"):
        validate_track(impossible, max_speed_kmh=180.0, max_route_distance_km=5000.0)


def test_route_sampling_uses_50m_and_deduplicates_stationary_frames() -> None:
    frames = tuple(
        SyncedFrame(
            source_ordinal=index,
            source_fingerprint=f"{index + 1:064x}",
            media_kind="image",
            capture_timestamp=datetime(2026, 7, 17, 8, 0, index, tzinfo=UTC),
            latitude=38.7205,
            longitude=longitude,
            coordinate_accuracy_m=5.0,
        )
        for index, longitude in enumerate((35.4826, 35.4826, 35.4829, 35.4832))
    )
    result = sample_frames(
        frames,
        plan_sha256="f" * 64,
        distance_m=50,
        stationary_radius_m=3.0,
    )
    assert result.distance_m == 50
    assert result.input_count == 4
    assert result.sampled_count == 2
    assert result.stationary_deduplicated_count == 1
    assert result.distance_excluded_count == 1


def test_gpx_reader_rejects_symlinks_and_never_uses_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = CapturePlan.model_validate(_capture_plan_payload())
    real = tmp_path / "real.gpx"
    link = tmp_path / "tracks" / "route.gpx"
    _write_gpx(real, (("2026-07-17T08:00:00Z", 38.0, 35.0),))
    link.parent.mkdir()
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")

    _deny_network(monkeypatch)
    with pytest.raises(AssetSafetyError, match="path_link_rejected"):
        read_gpx_track(tmp_path, "tracks/route.gpx", plan)


def test_megaloc_offline_smoke_and_descriptor_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _approved_megaloc_config(tmp_path)
    provider, backends = _provider_with_backend(config)
    locator = tmp_path / "synthetic-input.bin"
    locator.write_bytes(b"generated test-only image handle")

    _deny_network(monkeypatch)
    assert provider.runtime_kind == "test_only"
    receipt = provider.run_smoke(locator)
    assert receipt.descriptor_dimension == 8448
    assert provider.readiness.ready
    matrix = provider.describe_batch((locator,))
    assert matrix.shape == (1, 8448)
    assert provider.spec.artifact_sha256 == config.artifact_sha256
    assert provider.spec.preprocessing_version == config.preprocessing_id
    provider.close()
    assert backends[0].closed


@pytest.mark.parametrize("invalid", ["dimension", "nan", "inf"])
def test_megaloc_rejects_wrong_dimension_nan_and_inf(
    tmp_path: Path,
    invalid: str,
) -> None:
    config = _approved_megaloc_config(tmp_path)
    provider, _ = _provider_with_backend(config, invalid=invalid)
    locator = tmp_path / "synthetic-input.bin"
    locator.write_bytes(b"generated test-only image handle")
    with pytest.raises(MegaLocAdapterError) as error:
        provider.run_smoke(locator)
    assert error.value.code == "MEGALOC_INVALID_DESCRIPTOR"


def test_megaloc_missing_corrupt_and_wrong_hash_fail_closed(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    missing = _approved_megaloc_config(missing_root)
    missing.artifact_path.unlink()
    provider, _ = _provider_with_backend(missing)
    assert provider.readiness.reason_code == "MEGALOC_ARTIFACT_MISSING"

    hash_root = tmp_path / "hash"
    wrong_hash = _approved_megaloc_config(hash_root)
    wrong_hash.artifact_path.write_bytes(b"x" * wrong_hash.artifact_size)
    provider, _ = _provider_with_backend(wrong_hash)
    locator = hash_root / "input.bin"
    locator.write_bytes(b"generated fixture")
    with pytest.raises(MegaLocAdapterError) as hash_error:
        provider.run_smoke(locator)
    assert hash_error.value.code == "MEGALOC_ARTIFACT_HASH_MISMATCH"

    receipt_root = tmp_path / "receipt"
    corrupt = _approved_megaloc_config(receipt_root)
    corrupt.receipt_path.write_text("{corrupt", encoding="utf-8")
    provider, _ = _provider_with_backend(corrupt)
    assert provider.readiness.reason_code == "MEGALOC_RECEIPT_INVALID"


def test_megaloc_cuda_oom_reduces_batch_without_cpu_fallback(tmp_path: Path) -> None:
    config = _approved_megaloc_config(tmp_path, device="cuda", max_batch_size=4)
    provider, backends = _provider_with_backend(config, oom_above=2)
    locators = tuple(tmp_path / f"input-{index}.bin" for index in range(4))
    for locator in locators:
        locator.write_bytes(b"generated fixture")
    provider.run_smoke(locators[0])
    matrix = provider.describe_batch(locators)
    assert matrix.shape == (4, 8448)
    assert backends[0].device == "cuda"
    assert backends[0].calls == [1, 4, 2, 2]
    assert backends[0].recoveries == 1


def test_megaloc_singleton_cuda_oom_never_silently_falls_back_to_cpu(
    tmp_path: Path,
) -> None:
    config = _approved_megaloc_config(tmp_path, device="cuda")
    provider, backends = _provider_with_backend(config, always_oom=True)
    locator = tmp_path / "input.bin"
    locator.write_bytes(b"generated fixture")
    with pytest.raises(MegaLocAdapterError) as error:
        provider.run_smoke(locator)
    assert error.value.code == "MEGALOC_CUDA_OOM"
    assert backends[0].device == "cuda"


def test_megaloc_source_has_no_torch_hub_load_call() -> None:
    source = Path(__file__).parents[1] / "src" / "atlaslens_api" / "megaloc_adapter" / "provider.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "hub"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "torch"
    ]
    assert forbidden == []


def test_technical_smoke_cannot_override_revoked_production_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _approved_megaloc_config(tmp_path)
    created: list[_SyntheticMegaLocBackend] = []

    def factory(
        _config: MegaLocAdapterConfig,
        device: str,
    ) -> _SyntheticMegaLocBackend:
        backend = _SyntheticMegaLocBackend(device)
        created.append(backend)
        return backend

    monkeypatch.setattr(megaloc_provider_module, "_create_torch_backend", factory)
    provider = MegaLocDescriptorProvider(config)
    locator = tmp_path / "smoke-input.bin"
    locator.write_bytes(b"generated technical smoke fixture")
    provider.run_smoke(locator)
    assert created[0].device == "cpu"

    revoked_approval = replace(config.approval, production_use_approved=False)
    object.__setattr__(config, "approval", revoked_approval)
    with pytest.raises(
        MegaLocAdapterError,
        match="MEGALOC_PRODUCTION_APPROVAL_MISSING",
    ):
        _ = provider.runtime_kind
    registry = ProductionDescriptorRegistry()
    approval = ProviderApproval(
        provider_id=provider.spec.provider_id,
        version=provider.spec.version,
        dimension=provider.spec.dimension,
        artifact_sha256=config.artifact_sha256,
        license_record_id="phase3b2-synthetic-license",
        rights_approved=True,
    )
    with pytest.raises(
        MegaLocAdapterError,
        match="MEGALOC_PRODUCTION_APPROVAL_MISSING",
    ):
        registry.register(provider, approval)
