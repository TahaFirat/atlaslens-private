from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from atlaslens_api.phase3f.runpod import RunPodInventory
from atlaslens_api.phase3f.supervisor import (
    Phase3FSupervisorError,
    require_empty_inventory,
    verify_output_archive,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
