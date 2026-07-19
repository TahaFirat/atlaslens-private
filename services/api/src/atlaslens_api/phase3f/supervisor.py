"""Local-only preparation and verification primitives for the Phase 3F supervisor."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final
from uuid import uuid4

from atlaslens_api.phase3f.packaging import (
    LineEndingTrust,
    SourceArchive,
    SourceManifest,
    build_source_manifest,
    materialize_trusted_lf,
    verify_artifact,
    write_source_archive,
)
from atlaslens_api.phase3f.pipeline import MAX_LOCAL_DERIVED_ARTIFACT_BYTES
from atlaslens_api.phase3f.runpod import RunPodInventory

MODEL_SHA256: Final = "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
MODEL_SIZE: Final = 914_577_436
SOURCE_LF_SHA256: Final = "3cbf1d20515b1da423998a8edab787031eaa7bb273c5a86a5c41c4f6d84e2a6d"
SOURCE_CRLF_SHA256: Final = "c0848dfb287ba15b519d7b54415db824e16ec2f2b5a6899507b0476cf3379767"
LICENSE_LF_SHA256: Final = "0a906f9a65db6f645483f6cbf56b01e20615b9b943df3f70112f3d0fe0521e2a"
LICENSE_CRLF_SHA256: Final = "40c6c4894aecc5b676f0fb93697a6c1f82b08df71b25e485b662779b2c899667"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_OUTPUT_SUFFIXES = {".json", ".npy", ".faiss"}


class Phase3FSupervisorError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Phase3FSupervisorError(code)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TransferBundle:
    root: Path
    source_archive: SourceArchive
    source_manifest: SourceManifest
    model_path: Path
    canonical_source_path: Path
    canonical_license_path: Path


def prepare_transfer_bundle(
    repository_root: Path,
    transfer_root: Path,
    *,
    commit_sha: str,
    tracked_paths: tuple[str, ...],
    model_path: Path,
    vendor_root: Path,
) -> TransferBundle:
    transfer = transfer_root.resolve()
    _require(not transfer.exists(), "TRANSFER_ROOT_ALREADY_EXISTS")
    transfer.mkdir(parents=True)
    manifest = build_source_manifest(
        repository_root,
        commit_sha=commit_sha,
        tracked_paths=tracked_paths,
    )
    archive = write_source_archive(
        repository_root,
        transfer,
        "atlaslens-phase3f-source.tar",
        manifest,
    )
    verify_artifact(
        model_path.parent,
        model_path.name,
        expected_sha256=MODEL_SHA256,
        expected_size_bytes=MODEL_SIZE,
        max_bytes=MODEL_SIZE,
    )
    canonical_vendor = transfer / "vendor"
    source_result = materialize_trusted_lf(
        vendor_root,
        "megaloc_model.py",
        canonical_vendor,
        "megaloc_model.py",
        LineEndingTrust(SOURCE_LF_SHA256, SOURCE_CRLF_SHA256, 32 * 1024),
    )
    license_result = materialize_trusted_lf(
        vendor_root,
        "LICENSE",
        canonical_vendor,
        "LICENSE",
        LineEndingTrust(LICENSE_LF_SHA256, LICENSE_CRLF_SHA256, 4 * 1024),
    )
    _require(
        source_result.canonical_lf_sha256 == SOURCE_LF_SHA256
        and license_result.canonical_lf_sha256 == LICENSE_LF_SHA256,
        "CANONICAL_VENDOR_MISMATCH",
    )
    transfer_model = transfer / "model.safetensors"
    shutil.copy2(model_path, transfer_model)
    _require(
        transfer_model.stat().st_size == MODEL_SIZE
        and _sha256_path(transfer_model) == MODEL_SHA256,
        "TRANSFER_MODEL_MISMATCH",
    )
    return TransferBundle(
        root=transfer,
        source_archive=archive,
        source_manifest=manifest,
        model_path=transfer_model,
        canonical_source_path=canonical_vendor / "megaloc_model.py",
        canonical_license_path=canonical_vendor / "LICENSE",
    )


def require_empty_inventory(inventory: RunPodInventory) -> None:
    _require(not inventory.pods, "ACTIVE_POD_INVENTORY_NOT_ZERO")
    _require(not inventory.endpoint_ids, "ACTIVE_ENDPOINT_INVENTORY_NOT_ZERO")
    _require(not inventory.network_volume_ids, "NETWORK_VOLUME_INVENTORY_NOT_ZERO")
    _require(not inventory.template_ids, "TEMPLATE_INVENTORY_NOT_ZERO")


def require_inventory_restored(
    before: RunPodInventory,
    after: RunPodInventory,
) -> None:
    _require(after == before, "CLOUD_CLEANUP_UNVERIFIED")


@dataclass(frozen=True, slots=True)
class VerifiedOutput:
    extracted_root: Path
    publication_hash: str | None
    outcome: str
    total_size_bytes: int
    inventory_sha256: str


def _safe_member(member: tarfile.TarInfo) -> PurePosixPath:
    path = PurePosixPath(member.name)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and bool(path.parts)
        and path.parts[0] == "phase3f-output"
        and not member.issym()
        and not member.islnk()
        and (member.isdir() or member.isfile()),
        "OUTPUT_ARCHIVE_MEMBER_INVALID",
    )
    return path


def verify_output_archive(archive_path: Path, extraction_parent: Path) -> VerifiedOutput:
    _require(
        archive_path.is_file() and not archive_path.is_symlink(),
        "OUTPUT_ARCHIVE_MISSING",
    )
    destination = extraction_parent.resolve() / f"verified-{uuid4().hex}"
    destination.mkdir(parents=True)
    total = 0
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            _require(bool(members), "OUTPUT_ARCHIVE_EMPTY")
            for member in members:
                path = _safe_member(member)
                total += member.size
                _require(
                    total <= MAX_LOCAL_DERIVED_ARTIFACT_BYTES,
                    "OUTPUT_ARCHIVE_TOO_LARGE",
                )
                if member.isfile():
                    _require(
                        path.suffix.lower() in _ALLOWED_OUTPUT_SUFFIXES,
                        "OUTPUT_ARCHIVE_FILE_FORBIDDEN",
                    )
            archive.extractall(destination, members=members, filter="data")
        root = destination / "phase3f-output"
        inventory_path = root / "checksum-inventory.json"
        _require(
            inventory_path.is_file()
            and not inventory_path.is_symlink()
            and inventory_path.stat().st_size <= 4 * 1024 * 1024,
            "OUTPUT_INVENTORY_INVALID",
        )
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        _require(isinstance(inventory, dict), "OUTPUT_INVENTORY_INVALID")
        rows = inventory.get("files")
        expected_total = inventory.get("total_size_bytes")
        _require(
            inventory.get("schema") == "atlaslens-phase3f-checksum-inventory-v1"
            and isinstance(rows, list)
            and isinstance(expected_total, int)
            and not isinstance(expected_total, bool),
            "OUTPUT_INVENTORY_INVALID",
        )
        listed: set[str] = set()
        measured_total = 0
        for row in rows:
            _require(isinstance(row, dict), "OUTPUT_INVENTORY_INVALID")
            relative = row.get("relative_path")
            size = row.get("size_bytes")
            sha256 = row.get("sha256")
            _require(
                isinstance(relative, str)
                and isinstance(size, int)
                and not isinstance(size, bool)
                and isinstance(sha256, str)
                and bool(_SHA256.fullmatch(sha256)),
                "OUTPUT_INVENTORY_INVALID",
            )
            relative_path = PurePosixPath(relative)
            _require(
                not relative_path.is_absolute() and ".." not in relative_path.parts,
                "OUTPUT_INVENTORY_INVALID",
            )
            file_path = root.joinpath(*relative_path.parts)
            _require(
                file_path.is_file()
                and not file_path.is_symlink()
                and file_path.stat().st_size == size
                and _sha256_path(file_path) == sha256,
                "OUTPUT_CHECKSUM_MISMATCH",
            )
            listed.add(relative_path.as_posix())
            measured_total += size
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "checksum-inventory.json"
        }
        _require(actual == listed, "OUTPUT_INVENTORY_INCOMPLETE")
        _require(measured_total == expected_total, "OUTPUT_SIZE_MISMATCH")
        execution = json.loads((root / "execution-receipt.json").read_text(encoding="utf-8"))
        _require(isinstance(execution, dict), "OUTPUT_EXECUTION_RECEIPT_INVALID")
        outcome = execution.get("outcome")
        _require(
            outcome
            in {
                "COVERAGE_INSUFFICIENT",
                "COMPLETE_BENCHMARK_ONLY",
                "COMPLETE_ACCEPTED_PRIVATE_PILOT",
            },
            "OUTPUT_OUTCOME_INVALID",
        )
        publication_hash: str | None = None
        publication_path = root / "descriptor-publication.json"
        if publication_path.exists():
            publication = json.loads(publication_path.read_text(encoding="utf-8"))
            _require(isinstance(publication, dict), "OUTPUT_PUBLICATION_INVALID")
            candidate = publication.get("descriptor_publication_sha256")
            _require(
                isinstance(candidate, str) and bool(_SHA256.fullmatch(candidate)),
                "OUTPUT_PUBLICATION_INVALID",
            )
            publication_hash = candidate
        return VerifiedOutput(
            extracted_root=root,
            publication_hash=publication_hash,
            outcome=str(outcome),
            total_size_bytes=measured_total,
            inventory_sha256=_sha256_path(inventory_path),
        )
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def publish_verified_output(verified: VerifiedOutput, runtime_root: Path) -> Path:
    publication = verified.publication_hash or verified.inventory_sha256
    _require(bool(_SHA256.fullmatch(publication)), "OUTPUT_PUBLICATION_INVALID")
    root = runtime_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / publication
    _require(not destination.exists(), "OUTPUT_PUBLICATION_ALREADY_EXISTS")
    staging = root / f".{publication}.partial-{uuid4().hex}"
    shutil.copytree(verified.extracted_root, staging)
    os.replace(staging, destination)
    return destination


__all__ = [
    "Phase3FSupervisorError",
    "TransferBundle",
    "VerifiedOutput",
    "prepare_transfer_bundle",
    "publish_verified_output",
    "require_empty_inventory",
    "require_inventory_restored",
    "verify_output_archive",
]
