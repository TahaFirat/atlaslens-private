from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from atlaslens_api.phase3f import supervisor as supervisor_module
from atlaslens_api.phase3f.packaging import SourceArchive, SourceManifest
from atlaslens_api.phase3f.remote_environment import (
    FrameworkWheelArtifact,
    FrameworkWheelhouse,
)
from atlaslens_api.phase3f.runpod import RunPodInventory
from atlaslens_api.phase3f.safety import PodRecord
from atlaslens_api.phase3f.supervisor import (
    Phase3FSupervisorError,
    prepare_transfer_bundle,
    require_empty_inventory,
    require_inventory_restored,
    verify_output_archive,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_transfer_bundle_checksum_binds_offline_framework_wheels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    model_sha = _sha256(b"model")
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"wheel")
    wheel_sha = _sha256(b"wheel")
    artifact = FrameworkWheelArtifact(
        path=wheel,
        distribution="torchvision",
        version="0.24.1+cu128",
        filename="torchvision-fixture.whl",
        size_bytes=5,
        sha256=wheel_sha,
        source="official_pytorch",
    )
    inventory = {
        "schema": "atlaslens-phase3f-framework-transfer-inventory-v1",
        "artifact_count": 1,
        "artifacts": [{"filename": artifact.filename, "sha256": wheel_sha}],
    }
    wheelhouse = FrameworkWheelhouse(
        root=tmp_path,
        artifacts=(artifact,),
        inventory=inventory,
        inventory_sha256=_sha256(
            json.dumps(inventory, separators=(",", ":"), sort_keys=True).encode()
        ),
        companion=artifact,
    )
    monkeypatch.setattr(supervisor_module, "MODEL_SIZE", 5)
    monkeypatch.setattr(supervisor_module, "MODEL_SHA256", model_sha)
    monkeypatch.setattr(supervisor_module, "SOURCE_LF_SHA256", "1" * 64)
    monkeypatch.setattr(supervisor_module, "LICENSE_LF_SHA256", "2" * 64)
    monkeypatch.setattr(
        supervisor_module,
        "build_source_manifest",
        lambda *_args, **_kwargs: SourceManifest("a" * 40, (), 0),
    )
    monkeypatch.setattr(
        supervisor_module,
        "write_source_archive",
        lambda *_args, **_kwargs: SourceArchive("source.tar", "3" * 64, 1, "4" * 64, 0),
    )
    monkeypatch.setattr(supervisor_module, "verify_artifact", lambda *_a, **_k: None)

    def materialize(
        _source: Path,
        _name: str,
        destination: Path,
        output_name: str,
        _trust: object,
    ) -> object:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / output_name).write_text("fixture", encoding="ascii")
        digest = "1" * 64 if output_name.endswith(".py") else "2" * 64
        return type("Result", (), {"canonical_lf_sha256": digest})()

    monkeypatch.setattr(supervisor_module, "materialize_trusted_lf", materialize)
    bundle = prepare_transfer_bundle(
        tmp_path,
        tmp_path / "transfer",
        commit_sha="a" * 40,
        tracked_paths=(),
        model_path=model,
        vendor_root=tmp_path,
        framework_wheelhouse=wheelhouse,
    )
    assert bundle.framework_wheelhouse_path is not None
    assert (bundle.framework_wheelhouse_path / artifact.filename).read_bytes() == b"wheel"
    assert bundle.framework_inventory_path is not None
    assert json.loads(bundle.framework_inventory_path.read_text()) == inventory


def _valid_output_tar(path: Path) -> tuple[str, int]:
    execution = (
        json.dumps(
            {
                "schema": "atlaslens-phase3f-cloud-execution-v1",
                "outcome": "COVERAGE_INSUFFICIENT",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    inventory = (
        json.dumps(
            {
                "schema": "atlaslens-phase3f-checksum-inventory-v1",
                "total_size_bytes": len(execution),
                "files": [
                    {
                        "relative_path": "execution-receipt.json",
                        "size_bytes": len(execution),
                        "sha256": _sha256(execution),
                    }
                ],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    with tarfile.open(path, "w") as archive:
        for name, payload in (
            ("phase3f-output/execution-receipt.json", execution),
            ("phase3f-output/checksum-inventory.json", inventory),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return _sha256(inventory), len(execution)


def test_output_archive_is_checksum_bound_and_contains_no_unlisted_file(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "output.tar"
    expected_inventory, expected_size = _valid_output_tar(archive)

    verified = verify_output_archive(archive, tmp_path / "extract")

    assert verified.outcome == "COVERAGE_INSUFFICIENT"
    assert verified.publication_hash is None
    assert verified.inventory_sha256 == expected_inventory
    assert verified.total_size_bytes == expected_size


def test_output_archive_rejects_path_traversal_before_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as output:
        payload = b"unsafe"
        info = tarfile.TarInfo("phase3f-output/../../escape.json")
        info.size = len(payload)
        output.addfile(info, io.BytesIO(payload))

    with pytest.raises(Phase3FSupervisorError, match="OUTPUT_ARCHIVE_MEMBER_INVALID"):
        verify_output_archive(archive, tmp_path / "extract")

    assert not (tmp_path / "escape.json").exists()


def test_preprovision_inventory_must_be_completely_empty() -> None:
    require_empty_inventory(RunPodInventory((), (), (), ()))

    with pytest.raises(
        Phase3FSupervisorError,
        match="ACTIVE_ENDPOINT_INVENTORY_NOT_ZERO",
    ):
        require_empty_inventory(RunPodInventory((), ("endpoint-1",), (), ()))


def test_capacity_race_cleanup_requires_all_four_inventories_restored() -> None:
    empty = RunPodInventory((), (), (), ())
    require_inventory_restored(empty, empty)

    for not_empty in (
        RunPodInventory((PodRecord("pod-1", "phase3f-run"),), (), (), ()),
        RunPodInventory((), ("endpoint-1",), (), ()),
        RunPodInventory((), (), ("volume-1",), ()),
        RunPodInventory((), (), (), ("template-1",)),
    ):
        with pytest.raises(Phase3FSupervisorError, match="CLOUD_CLEANUP_UNVERIFIED"):
            require_inventory_restored(empty, not_empty)
