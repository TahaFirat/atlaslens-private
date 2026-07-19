from __future__ import annotations

import asyncio
import hashlib
import os
import re
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlaslens_api.evaluation.holdout import HoldoutManifestError, HoldoutManifestLoader
from atlaslens_api.phase6b.scheduler import DeviceName
from atlaslens_api.phase6b.worker_client import WorkerClientError
from atlaslens_api.phase6c.megaloc import (
    MEGALOC_DESCRIPTOR_DIMENSION,
    MEGALOC_DESCRIPTOR_VERSION,
    MEGALOC_MODEL_ID,
    MEGALOC_MODEL_REVISION,
    MEGALOC_SOURCE_REVISION,
    MegaLocWorker,
    create_megaloc_http_worker_client,
    validate_megaloc_descriptor,
)
from atlaslens_api.phase6c.reference_index import (
    ReferenceIndexBuildError,
    load_reference_build_input,
)

_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9_]{0,79}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LeakageDescriptorBuildError(RuntimeError):
    """Safe error for the local operator command; messages never contain paths."""

    def __init__(self, code: str) -> None:
        self.code = code if _SAFE_CODE.fullmatch(code) else "leakage_descriptor_build_failed"
        super().__init__(self.code)


class LeakageDescriptorBuildPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_references: int = Field(default=10_000, ge=1, le=100_000)
    max_manifest_bytes: int = Field(default=20 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    max_image_bytes: int = Field(default=20 * 1024 * 1024, ge=1, le=32 * 1024 * 1024)
    max_total_image_bytes: int = Field(
        default=2 * 1024 * 1024 * 1024,
        ge=1,
        le=100 * 1024 * 1024 * 1024,
    )
    max_output_bytes: int = Field(
        default=512 * 1024 * 1024,
        ge=1024 * 1024,
        le=2 * 1024 * 1024 * 1024,
    )
    worker_timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    total_timeout_seconds: float = Field(default=3_600.0, gt=0, le=86_400)

    @model_validator(mode="after")
    def validate_timeouts(self) -> LeakageDescriptorBuildPolicy:
        if self.total_timeout_seconds < self.worker_timeout_seconds:
            raise ValueError("total timeout cannot be shorter than one worker timeout")
        return self


@dataclass(frozen=True, slots=True)
class LeakageDescriptorBuildResult:
    input_image_count: int
    unique_content_count: int
    artifact_size_bytes: int
    provider: Literal["megaloc"] = "megaloc"
    version: str = MEGALOC_DESCRIPTOR_VERSION
    descriptor_dimension: int = MEGALOC_DESCRIPTOR_DIMENSION


@dataclass(frozen=True, slots=True)
class _InputImage:
    path: Path = field(repr=False)
    expected_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _HashedInput:
    path: Path = field(repr=False)
    sha256: str


def _safe_reference_root(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise LeakageDescriptorBuildError("reference_root_invalid")
    try:
        resolved = expanded.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LeakageDescriptorBuildError("reference_root_unavailable") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise LeakageDescriptorBuildError("reference_root_invalid")
    return resolved


def _safe_reference_image(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or relative.drive or not relative.parts or ".." in relative.parts:
        raise LeakageDescriptorBuildError("reference_image_path_invalid")
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise LeakageDescriptorBuildError("reference_image_path_invalid")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LeakageDescriptorBuildError("reference_image_unavailable") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.is_symlink():
        raise LeakageDescriptorBuildError("reference_image_path_invalid")
    return resolved


def _safe_reference_manifest(path: Path, max_bytes: int) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise LeakageDescriptorBuildError("reference_input_invalid")
    try:
        resolved = expanded.resolve(strict=True)
        size = resolved.stat().st_size
    except (OSError, RuntimeError) as exc:
        raise LeakageDescriptorBuildError("reference_input_unavailable") from exc
    if resolved.is_symlink() or not resolved.is_file() or not 0 < size <= max_bytes:
        raise LeakageDescriptorBuildError("reference_input_invalid")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LeakageDescriptorBuildError("input_image_unavailable") from exc
    return digest.hexdigest()


def _prepare_inputs(
    *,
    holdout_manifest: Path,
    holdout_id: str,
    reference_input: Path,
    reference_root: Path,
    policy: LeakageDescriptorBuildPolicy,
) -> tuple[_HashedInput, ...]:
    try:
        prediction_inputs = HoldoutManifestLoader(
            max_image_bytes=policy.max_image_bytes,
        ).load_prediction_inputs(holdout_manifest)
    except HoldoutManifestError as exc:
        raise LeakageDescriptorBuildError("holdout_input_invalid") from exc
    selected = [item for item in prediction_inputs if item.evaluation_id == holdout_id]
    if len(selected) != 1:
        raise LeakageDescriptorBuildError("holdout_selection_invalid")

    root = _safe_reference_root(reference_root)
    manifest = _safe_reference_manifest(reference_input, policy.max_manifest_bytes)
    try:
        build_input = load_reference_build_input(manifest)
    except ReferenceIndexBuildError as exc:
        raise LeakageDescriptorBuildError("reference_input_invalid") from exc
    if not build_input.records:
        raise LeakageDescriptorBuildError("reference_input_empty")
    if len(build_input.records) > policy.max_references:
        raise LeakageDescriptorBuildError("reference_count_limit_exceeded")

    inputs: list[_InputImage] = [_InputImage(selected[0].image_path)]
    reference_paths: set[Path] = set()
    for record in build_input.records:
        image_path = _safe_reference_image(root, record.image_path)
        if image_path in reference_paths:
            raise LeakageDescriptorBuildError("duplicate_reference_path")
        reference_paths.add(image_path)
        inputs.append(_InputImage(image_path, record.expected_sha256))

    total_bytes = 0
    hashed: list[_HashedInput] = []
    for item in inputs:
        try:
            size = item.path.stat().st_size
        except OSError as exc:
            raise LeakageDescriptorBuildError("input_image_unavailable") from exc
        if not 0 < size <= policy.max_image_bytes:
            raise LeakageDescriptorBuildError("input_image_size_limit_exceeded")
        total_bytes += size
        if total_bytes > policy.max_total_image_bytes:
            raise LeakageDescriptorBuildError("input_total_bytes_limit_exceeded")
        digest = _sha256_file(item.path)
        if item.expected_sha256 is not None and item.expected_sha256 != digest:
            raise LeakageDescriptorBuildError("reference_sha256_mismatch")
        hashed.append(_HashedInput(path=item.path, sha256=digest))
    return tuple(hashed)


def _estimated_artifact_bytes(row_count: int) -> int:
    # NumPy U64 keys use four bytes per code point; one MiB covers fixed ZIP metadata.
    return row_count * (MEGALOC_DESCRIPTOR_DIMENSION * 4 + 64 * 4) + 1024 * 1024


def _require_unchanged_inputs(inputs: tuple[_HashedInput, ...]) -> None:
    if any(_sha256_file(item.path) != item.sha256 for item in inputs):
        raise LeakageDescriptorBuildError("input_image_changed")


async def _infer_descriptors(
    inputs: tuple[_HashedInput, ...],
    *,
    worker: MegaLocWorker,
    device: DeviceName,
    policy: LeakageDescriptorBuildPolicy,
) -> tuple[tuple[str, ...], NDArray[np.float32]]:
    unique: list[_HashedInput] = []
    seen_hashes: set[str] = set()
    for item in inputs:
        if item.sha256 not in seen_hashes:
            seen_hashes.add(item.sha256)
            unique.append(item)
    if _estimated_artifact_bytes(len(unique)) > policy.max_output_bytes:
        raise LeakageDescriptorBuildError("artifact_size_limit_exceeded")

    vectors = np.empty(
        (len(unique), MEGALOC_DESCRIPTOR_DIMENSION),
        dtype=np.float32,
    )
    loaded = False
    try:
        async with asyncio.timeout(policy.total_timeout_seconds):
            health = await worker.health(timeout_seconds=policy.worker_timeout_seconds)
            if (
                health.provider != "megaloc"
                or health.provider_revision != MEGALOC_SOURCE_REVISION
                or health.model_revision != MEGALOC_MODEL_REVISION
            ):
                raise LeakageDescriptorBuildError("worker_identity_mismatch")
            if not health.process_running or not health.import_ok or not health.weights_available:
                raise LeakageDescriptorBuildError("worker_not_ready")
            await worker.load(device, timeout_seconds=policy.worker_timeout_seconds)
            loaded = True
            for index, item in enumerate(unique):
                try:
                    image_bytes = item.path.read_bytes()
                except OSError as exc:
                    raise LeakageDescriptorBuildError("input_image_unavailable") from exc
                if (
                    not 0 < len(image_bytes) <= policy.max_image_bytes
                    or hashlib.sha256(image_bytes).hexdigest() != item.sha256
                ):
                    raise LeakageDescriptorBuildError("input_image_changed")
                raw = await worker.describe(
                    image_bytes,
                    device=device,
                    timeout_seconds=policy.worker_timeout_seconds,
                )
                try:
                    descriptor = validate_megaloc_descriptor(raw)
                except ValueError as exc:
                    raise LeakageDescriptorBuildError("worker_descriptor_invalid") from exc
                vectors[index] = np.asarray(descriptor, dtype=np.float32)
                del image_bytes, descriptor
                await asyncio.sleep(0)
    except TimeoutError as exc:
        raise LeakageDescriptorBuildError("worker_timeout") from exc
    except WorkerClientError as exc:
        raise LeakageDescriptorBuildError("worker_request_failed") from exc
    finally:
        if loaded:
            with suppress(TimeoutError, WorkerClientError, OSError):
                await asyncio.wait_for(
                    worker.unload(device, timeout_seconds=5.0),
                    timeout=6.0,
                )
        with suppress(TimeoutError, WorkerClientError, OSError):
            await asyncio.wait_for(worker.close(), timeout=6.0)
    return tuple(item.sha256 for item in unique), vectors


def _write_atomic_artifact(
    output: Path,
    hashes: tuple[str, ...],
    vectors: NDArray[np.float32],
    *,
    max_output_bytes: int,
) -> int:
    expanded = output.expanduser()
    if expanded.suffix.casefold() != ".npz" or expanded.is_symlink():
        raise LeakageDescriptorBuildError("artifact_output_invalid")
    if expanded.exists() and not expanded.is_file():
        raise LeakageDescriptorBuildError("artifact_output_invalid")
    try:
        expanded.parent.mkdir(parents=True, exist_ok=True)
        destination = expanded.resolve()
    except (OSError, RuntimeError) as exc:
        raise LeakageDescriptorBuildError("artifact_output_invalid") from exc
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez(
                stream,
                provider=np.asarray("megaloc"),
                version=np.asarray(MEGALOC_DESCRIPTOR_VERSION),
                model_id=np.asarray(MEGALOC_MODEL_ID),
                model_revision=np.asarray(MEGALOC_MODEL_REVISION),
                source_revision=np.asarray(MEGALOC_SOURCE_REVISION),
                dimension=np.asarray(MEGALOC_DESCRIPTOR_DIMENSION, dtype=np.int64),
                normalization=np.asarray("l2"),
                content_sha256=np.asarray(hashes, dtype="<U64"),
                vectors=np.ascontiguousarray(vectors, dtype=np.float32),
            )
            stream.flush()
            os.fsync(stream.fileno())
        size = temporary.stat().st_size
        if size > max_output_bytes:
            raise LeakageDescriptorBuildError("artifact_size_limit_exceeded")
        os.replace(temporary, destination)
        return size
    except LeakageDescriptorBuildError:
        raise
    except (OSError, ValueError) as exc:
        raise LeakageDescriptorBuildError("artifact_write_failed") from exc
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


async def build_phase6c_leakage_descriptor_artifact(
    *,
    holdout_manifest: Path,
    holdout_id: str,
    reference_input: Path,
    reference_root: Path,
    output: Path,
    worker_port: int = 8794,
    device: DeviceName = "cuda",
    policy: LeakageDescriptorBuildPolicy | None = None,
    worker: MegaLocWorker | None = None,
) -> LeakageDescriptorBuildResult:
    """Infer real target/reference descriptors without loading holdout truth."""

    active_policy = policy or LeakageDescriptorBuildPolicy()
    if not 1 <= worker_port <= 65_535:
        raise LeakageDescriptorBuildError("worker_port_invalid")
    inputs = _prepare_inputs(
        holdout_manifest=holdout_manifest,
        holdout_id=holdout_id,
        reference_input=reference_input,
        reference_root=reference_root,
        policy=active_policy,
    )
    active_worker = worker or create_megaloc_http_worker_client(
        host="127.0.0.1",
        port=worker_port,
        timeout_seconds=active_policy.worker_timeout_seconds,
        max_image_bytes=active_policy.max_image_bytes,
    )
    hashes, vectors = await _infer_descriptors(
        inputs,
        worker=active_worker,
        device=device,
        policy=active_policy,
    )
    _require_unchanged_inputs(inputs)
    if any(_SHA256.fullmatch(value) is None for value in hashes):
        raise LeakageDescriptorBuildError("artifact_hash_invalid")
    size = _write_atomic_artifact(
        output,
        hashes,
        vectors,
        max_output_bytes=active_policy.max_output_bytes,
    )
    return LeakageDescriptorBuildResult(
        input_image_count=len(inputs),
        unique_content_count=len(hashes),
        artifact_size_bytes=size,
    )
