from __future__ import annotations

import hashlib
import os
import shutil
import statistics
import time
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from .errors import ModelManagementError
from .models import (
    EmbeddingInstallationReceipt,
    EmbeddingModelInfo,
    EmbeddingModelManifest,
    InstallationReceipt,
    ModelBenchmarkResult,
    ModelInfo,
    ModelManifest,
    ModelSmokeObservation,
    ModelTestResult,
)

_RECEIPT = "installation.json"
_MAX_WHEEL_BYTES = 64 * 1024 * 1024
_MAX_EXTRACTED_ASSET_BYTES = 256 * 1024 * 1024
_MAX_EXTRACTED_TOTAL_BYTES = 512 * 1024 * 1024
_MINIMUM_FREE_BYTES = 4 * 1024 * 1024 * 1024
_GEOCLIP_ASSETS = (
    "geoclip/model/weights/image_encoder_mlp_weights.pth",
    "geoclip/model/weights/location_encoder_weights.pth",
    "geoclip/model/weights/logit_scale_weights.pth",
    "geoclip/model/gps_gallery/coordinates_100K.csv",
)


class SnapshotFetcher(Protocol):
    def __call__(self, manifest: ModelManifest, destination: Path) -> str: ...


class WheelFetcher(Protocol):
    def __call__(self, url: str, destination: Path) -> None: ...


class EmbeddingSnapshotFetcher(Protocol):
    def __call__(self, manifest: EmbeddingModelManifest, destination: Path) -> str: ...


PredictionHook = Callable[[Path], int | ModelSmokeObservation]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_wheel_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.endswith(".whl")
        or parsed.query
        or parsed.fragment
    ):
        raise ModelManagementError("unapproved_artifact_source")


class _SafeWheelRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_wheel_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url: str, destination: Path) -> None:
    _validate_wheel_url(url)
    request = urllib.request.Request(  # noqa: S310 - HTTPS host is allowlisted above.
        url, headers={"User-Agent": "AtlasLens-model-installer/1"}
    )
    opener = urllib.request.build_opener(_SafeWheelRedirectHandler())
    with (
        opener.open(request, timeout=60) as response,  # noqa: S310
        destination.open("xb") as target,
    ):
        _validate_wheel_url(response.geturl())
        size = 0
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > _MAX_WHEEL_BYTES:
                raise ModelManagementError("artifact_too_large")
            target.write(chunk)
        if size <= 0:
            raise ModelManagementError("weights_incomplete")


def _snapshot(manifest: ModelManifest, destination: Path) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelManagementError("missing_huggingface_hub") from exc
    resolved = Path(
        snapshot_download(
            repo_id=manifest.clip_repo_id,
            revision=manifest.clip_revision,
            local_dir=destination,
            allow_patterns=list(manifest.required_clip_files),
        )
    )
    if resolved.resolve() != destination.resolve():
        raise ModelManagementError("unexpected_snapshot_destination")
    return manifest.clip_revision


def _embedding_snapshot(manifest: EmbeddingModelManifest, destination: Path) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelManagementError("missing_huggingface_hub") from exc
    resolved = Path(
        snapshot_download(
            repo_id=manifest.repo_id,
            revision=manifest.revision,
            local_dir=destination,
            allow_patterns=list(manifest.required_files),
        )
    )
    if resolved.resolve() != destination.resolve():
        raise ModelManagementError("unexpected_snapshot_destination")
    return manifest.revision


class ModelManagementService:
    def __init__(
        self,
        cache_root: Path,
        manifest: ModelManifest,
        *,
        wheel_fetcher: WheelFetcher = _download,
        snapshot_fetcher: SnapshotFetcher = _snapshot,
    ) -> None:
        self.cache_root = cache_root.expanduser().resolve()
        self.manifest = manifest
        self._wheel_fetcher = wheel_fetcher
        self._snapshot_fetcher = snapshot_fetcher

    @property
    def model_directory(self) -> Path:
        return self.cache_root / self.manifest.model_id

    def is_installed(self) -> bool:
        receipt = self.model_directory / _RECEIPT
        return (
            self.model_directory.is_dir()
            and not self.model_directory.is_symlink()
            and receipt.is_file()
            and not receipt.is_symlink()
        )

    def install(self) -> InstallationReceipt:
        if self.model_directory.exists() or self.model_directory.is_symlink():
            raise ModelManagementError("model_already_installed")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.cache_root).free < _MINIMUM_FREE_BYTES:
            raise ModelManagementError("insufficient_disk_space")
        staging = self.cache_root / f".{self.manifest.model_id}.installing-{os.getpid()}"
        if staging.exists() or staging.is_symlink():
            raise ModelManagementError("partial_installation_exists")
        staging.mkdir()
        try:
            wheel = staging / self.manifest.wheel_filename
            self._wheel_fetcher(self.manifest.wheel_url, wheel)
            if _sha256(wheel) != self.manifest.wheel_sha256:
                raise ModelManagementError("checksum_mismatch")
            asset_hashes = self._extract_official_assets(wheel, staging / "geoclip-assets")
            snapshot = staging / "clip-snapshot"
            snapshot.mkdir()
            resolved_revision = self._snapshot_fetcher(self.manifest, snapshot)
            if resolved_revision != self.manifest.clip_revision:
                raise ModelManagementError("revision_mismatch")
            self._verify_snapshot(snapshot)
            receipt = InstallationReceipt(
                model_id="geoclip",
                model_version=self.manifest.model_version,
                implementation_revision=self.manifest.implementation_revision,
                installed_at=datetime.now(UTC),
                wheel_filename=wheel.name,
                wheel_sha256=self.manifest.wheel_sha256,
                clip_repo_id=self.manifest.clip_repo_id,
                requested_clip_revision=self.manifest.clip_revision,
                resolved_clip_revision=resolved_revision,
                clip_snapshot_directory="clip-snapshot",
                clip_weights_filename=self.manifest.clip_weights_filename,
                clip_weights_sha256=self.manifest.clip_weights_sha256,
                clip_file_sha256={
                    relative: _sha256(snapshot / relative)
                    for relative in self.manifest.required_clip_files
                },
                geoclip_assets_directory="geoclip-assets",
                geoclip_asset_sha256=asset_hashes,
            )
            (staging / _RECEIPT).write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
            os.replace(staging, self.model_directory)
            return receipt
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _receipt(self) -> InstallationReceipt:
        receipt_path = self.model_directory / _RECEIPT
        if self.model_directory.is_symlink() or receipt_path.is_symlink():
            raise ModelManagementError("unsafe_installation_path")
        try:
            receipt = InstallationReceipt.model_validate_json(
                receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ModelManagementError("weights_incomplete") from exc
        self._validate_receipt(receipt)
        return receipt

    def _validate_receipt(self, receipt: InstallationReceipt) -> None:
        if (
            receipt.wheel_filename != self.manifest.wheel_filename
            or receipt.clip_snapshot_directory != "clip-snapshot"
            or receipt.geoclip_assets_directory != "geoclip-assets"
            or receipt.clip_weights_filename != self.manifest.clip_weights_filename
        ):
            raise ModelManagementError("unsafe_installation_path")
        if (
            receipt.model_id != self.manifest.model_id
            or receipt.model_version != self.manifest.model_version
            or receipt.implementation_revision != self.manifest.implementation_revision
            or receipt.clip_repo_id != self.manifest.clip_repo_id
        ):
            raise ModelManagementError("weights_incomplete")
        if (
            receipt.wheel_sha256 != self.manifest.wheel_sha256
            or receipt.clip_weights_sha256 != self.manifest.clip_weights_sha256
        ):
            raise ModelManagementError("checksum_mismatch")
        expected_assets = {Path(item).name for item in _GEOCLIP_ASSETS}
        if (
            set(receipt.clip_file_sha256) != set(self.manifest.required_clip_files)
            or set(receipt.geoclip_asset_sha256) != expected_assets
            or any(
                not _is_sha256(value)
                for value in (
                    *receipt.clip_file_sha256.values(),
                    *receipt.geoclip_asset_sha256.values(),
                )
            )
        ):
            raise ModelManagementError("weights_incomplete")

    def _extract_official_assets(self, wheel: Path, destination: Path) -> dict[str, str]:
        destination.mkdir()
        hashes: dict[str, str] = {}
        try:
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                if any(name not in names for name in _GEOCLIP_ASSETS):
                    raise ModelManagementError("weights_incomplete")
                total_size = sum(archive.getinfo(name).file_size for name in _GEOCLIP_ASSETS)
                if (
                    total_size > _MAX_EXTRACTED_TOTAL_BYTES
                    or any(
                        archive.getinfo(name).file_size > _MAX_EXTRACTED_ASSET_BYTES
                        for name in _GEOCLIP_ASSETS
                    )
                ):
                    raise ModelManagementError("artifact_too_large")
                for name in _GEOCLIP_ASSETS:
                    output = destination / Path(name).name
                    with archive.open(name) as source, output.open("xb") as target:
                        copied = 0
                        while chunk := source.read(1024 * 1024):
                            copied += len(chunk)
                            if copied > _MAX_EXTRACTED_ASSET_BYTES:
                                raise ModelManagementError("artifact_too_large")
                            target.write(chunk)
                    hashes[output.name] = _sha256(output)
        except zipfile.BadZipFile as exc:
            raise ModelManagementError("invalid_wheel") from exc
        return hashes

    def _official_asset_hashes_from_wheel(self, wheel: Path) -> dict[str, str]:
        hashes: dict[str, str] = {}
        try:
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                if any(name not in names for name in _GEOCLIP_ASSETS):
                    raise ModelManagementError("weights_incomplete")
                for name in _GEOCLIP_ASSETS:
                    info = archive.getinfo(name)
                    if info.file_size > _MAX_EXTRACTED_ASSET_BYTES:
                        raise ModelManagementError("artifact_too_large")
                    digest = hashlib.sha256()
                    size = 0
                    with archive.open(name) as source:
                        while chunk := source.read(1024 * 1024):
                            size += len(chunk)
                            if size > _MAX_EXTRACTED_ASSET_BYTES:
                                raise ModelManagementError("artifact_too_large")
                            digest.update(chunk)
                    hashes[Path(name).name] = digest.hexdigest()
        except zipfile.BadZipFile as exc:
            raise ModelManagementError("invalid_wheel") from exc
        return hashes

    def snapshot_directory(self) -> Path:
        receipt = self._receipt()
        snapshot = (self.model_directory / receipt.clip_snapshot_directory).resolve()
        if snapshot.parent != self.model_directory.resolve():
            raise ModelManagementError("unsafe_installation_path")
        return snapshot

    def geoclip_assets_directory(self) -> Path:
        receipt = self._receipt()
        assets = (self.model_directory / receipt.geoclip_assets_directory).resolve()
        if assets.parent != self.model_directory.resolve():
            raise ModelManagementError("unsafe_installation_path")
        return assets

    def _verify_snapshot(self, snapshot: Path) -> None:
        for relative in self.manifest.required_clip_files:
            path = snapshot / relative
            if not path.is_file() or path.is_symlink():
                raise ModelManagementError("weights_incomplete")
        weights = snapshot / self.manifest.clip_weights_filename
        if _sha256(weights) != self.manifest.clip_weights_sha256:
            raise ModelManagementError("checksum_mismatch")

    def verify(self) -> InstallationReceipt:
        receipt = self._receipt()
        wheel = self.model_directory / receipt.wheel_filename
        if not wheel.is_file() or wheel.is_symlink():
            raise ModelManagementError("weights_incomplete")
        if _sha256(wheel) != self.manifest.wheel_sha256:
            raise ModelManagementError("checksum_mismatch")
        expected_asset_hashes = self._official_asset_hashes_from_wheel(wheel)
        if receipt.geoclip_asset_sha256 != expected_asset_hashes:
            raise ModelManagementError("checksum_mismatch")
        self._verify_snapshot(self.snapshot_directory())
        snapshot = self.snapshot_directory()
        if set(receipt.clip_file_sha256) != set(self.manifest.required_clip_files):
            raise ModelManagementError("weights_incomplete")
        for relative, expected in receipt.clip_file_sha256.items():
            path = snapshot / relative
            if _sha256(path) != expected:
                raise ModelManagementError("checksum_mismatch")
        assets = self.geoclip_assets_directory()
        for filename, expected in expected_asset_hashes.items():
            path = assets / filename
            if not path.is_file() or path.is_symlink():
                raise ModelManagementError("weights_incomplete")
            if _sha256(path) != expected:
                raise ModelManagementError("checksum_mismatch")
        if (
            receipt.requested_clip_revision != self.manifest.clip_revision
            or receipt.resolved_clip_revision != self.manifest.clip_revision
        ):
            raise ModelManagementError("revision_mismatch")
        return receipt

    def info(self, *, verify: bool = False) -> ModelInfo:
        if not self.is_installed():
            return self._info("not_installed")
        try:
            receipt = self.verify() if verify else self._receipt()
        except ModelManagementError as exc:
            return self._info("invalid", reason_code=exc.code)
        return self._info(
            "verified" if verify else "installed",
            resolved_revision=receipt.resolved_clip_revision,
        )

    def _info(
        self,
        status: str,
        *,
        resolved_revision: str | None = None,
        reason_code: str | None = None,
    ) -> ModelInfo:
        size = 0
        if self.model_directory.is_dir():
            size = sum(
                path.stat().st_size for path in self.model_directory.rglob("*") if path.is_file()
            )
        return ModelInfo(
            model_id=self.manifest.model_id,
            status=status,
            model_version=self.manifest.model_version,
            implementation_revision=self.manifest.implementation_revision,
            license=self.manifest.license,
            score_type=self.manifest.score_type,
            requested_clip_revision=self.manifest.clip_revision,
            resolved_clip_revision=resolved_revision,
            storage_size_bytes=size,
            reason_code=reason_code,
        )

    def remove(self, *, confirmed: bool) -> None:
        if not confirmed:
            raise ModelManagementError("confirmation_required")
        directory = self.model_directory
        if directory.is_symlink() or not (directory / _RECEIPT).is_file():
            raise ModelManagementError("unrecognized_model_directory")
        shutil.rmtree(directory)

    def test(self, image: Path, predictor: PredictionHook) -> ModelTestResult:
        if not image.is_file():
            raise ModelManagementError("image_not_found")
        started = time.monotonic()
        try:
            observation = predictor(image)
        except ModelManagementError as exc:
            return ModelTestResult(
                status="failed",
                hypothesis_count=0,
                duration_ms=round((time.monotonic() - started) * 1000),
                reason_code=exc.code,
                subreason_code=exc.subreason_code,
            )
        except Exception:
            return ModelTestResult(
                status="failed",
                hypothesis_count=0,
                duration_ms=round((time.monotonic() - started) * 1000),
                reason_code="provider_test_failed",
            )
        count = (
            observation
            if isinstance(observation, int)
            else observation.hypothesis_count
        )
        metadata = (
            {}
            if isinstance(observation, int)
            else {
                "provider_id": observation.provider_id,
                "model_revision": observation.model_revision,
                "implementation_revision": observation.implementation_revision,
                "device": observation.device,
                "score_type": observation.score_type,
                "calibration_state": observation.calibration_state,
                "coordinate_validity": observation.coordinate_validity,
            }
        )
        return ModelTestResult(
            status="passed" if count >= 3 else "failed",
            hypothesis_count=max(0, min(count, 20)),
            duration_ms=round((time.monotonic() - started) * 1000),
            reason_code=None if count >= 3 else "insufficient_hypotheses",
            **metadata,
        )

    def benchmark(self, images: Sequence[Path], predictor: PredictionHook) -> ModelBenchmarkResult:
        durations: list[int] = []
        succeeded = 0
        for image in images:
            result = self.test(image, predictor)
            durations.append(result.duration_ms)
            succeeded += result.status == "passed"
        return ModelBenchmarkResult(
            status="completed" if succeeded == len(images) else "failed",
            attempted=len(images),
            succeeded=succeeded,
            durations_ms=tuple(durations),
            runs=max(1, len(images)),
            warmup_runs=0,
            cold_load_ms=durations[0] if durations else 0,
            warm_median_ms=float(statistics.median(durations)) if durations else 0,
            warm_p95_ms=float(max(durations, default=0)),
            device="test_hook",
            peak_gpu_allocated_bytes=0,
            model_revision=self.manifest.model_version,
        )


class EmbeddingModelManagementService:
    """Explicit, receipt-backed management for the pinned SigLIP2 snapshot."""

    def __init__(
        self,
        cache_root: Path,
        manifest: EmbeddingModelManifest,
        *,
        snapshot_fetcher: EmbeddingSnapshotFetcher = _embedding_snapshot,
    ) -> None:
        self.cache_root = cache_root.expanduser().resolve()
        self.manifest = manifest
        self._snapshot_fetcher = snapshot_fetcher

    @property
    def model_directory(self) -> Path:
        return self.cache_root / self.manifest.model_id

    def is_installed(self) -> bool:
        receipt = self.model_directory / _RECEIPT
        return (
            self.model_directory.is_dir()
            and not self.model_directory.is_symlink()
            and receipt.is_file()
            and not receipt.is_symlink()
        )

    def install(self) -> EmbeddingInstallationReceipt:
        if self.model_directory.exists() or self.model_directory.is_symlink():
            raise ModelManagementError("model_already_installed")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.cache_root).free < _MINIMUM_FREE_BYTES:
            raise ModelManagementError("insufficient_disk_space")
        staging = self.cache_root / f".{self.manifest.model_id}.installing-{os.getpid()}"
        if staging.exists() or staging.is_symlink():
            raise ModelManagementError("partial_installation_exists")
        staging.mkdir()
        try:
            snapshot = staging / "snapshot"
            snapshot.mkdir()
            resolved_revision = self._snapshot_fetcher(self.manifest, snapshot)
            if resolved_revision != self.manifest.revision:
                raise ModelManagementError("revision_mismatch")
            self._verify_snapshot_files(snapshot)
            hashes = {
                relative: _sha256(snapshot / relative)
                for relative in self.manifest.required_files
            }
            receipt = EmbeddingInstallationReceipt(
                model_id=self.manifest.model_id,
                model_version=self.manifest.model_version,
                repo_id=self.manifest.repo_id,
                requested_revision=self.manifest.revision,
                resolved_revision=resolved_revision,
                installed_at=datetime.now(UTC),
                weights_filename=self.manifest.weights_filename,
                weights_sha256=self.manifest.weights_sha256,
                file_sha256=hashes,
                embedding_dimension=self.manifest.embedding_dimension,
                preprocessing_version=self.manifest.preprocessing_version,
            )
            (staging / _RECEIPT).write_text(
                receipt.model_dump_json(indent=2), encoding="utf-8"
            )
            os.replace(staging, self.model_directory)
            return receipt
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _receipt(self) -> EmbeddingInstallationReceipt:
        receipt_path = self.model_directory / _RECEIPT
        if self.model_directory.is_symlink() or receipt_path.is_symlink():
            raise ModelManagementError("unsafe_installation_path")
        try:
            receipt = EmbeddingInstallationReceipt.model_validate_json(
                receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ModelManagementError("weights_incomplete") from exc
        if (
            receipt.model_id != self.manifest.model_id
            or receipt.model_version != self.manifest.model_version
            or receipt.repo_id != self.manifest.repo_id
            or receipt.requested_revision != self.manifest.revision
            or receipt.resolved_revision != self.manifest.revision
            or receipt.snapshot_directory != "snapshot"
            or receipt.weights_filename != self.manifest.weights_filename
            or receipt.weights_sha256 != self.manifest.weights_sha256
            or receipt.embedding_dimension != self.manifest.embedding_dimension
            or receipt.preprocessing_version != self.manifest.preprocessing_version
        ):
            raise ModelManagementError("weights_incomplete")
        if (
            set(receipt.file_sha256) != set(self.manifest.required_files)
            or any(not _is_sha256(value) for value in receipt.file_sha256.values())
        ):
            raise ModelManagementError("weights_incomplete")
        return receipt

    def snapshot_directory(self) -> Path:
        receipt = self._receipt()
        snapshot = (self.model_directory / receipt.snapshot_directory).resolve()
        if snapshot.parent != self.model_directory.resolve():
            raise ModelManagementError("unsafe_installation_path")
        return snapshot

    def _verify_snapshot_files(self, snapshot: Path) -> None:
        for relative in self.manifest.required_files:
            path = snapshot / relative
            if not path.is_file() or path.is_symlink():
                raise ModelManagementError("weights_incomplete")
        if _sha256(snapshot / self.manifest.weights_filename) != self.manifest.weights_sha256:
            raise ModelManagementError("checksum_mismatch")

    def verify(self) -> EmbeddingInstallationReceipt:
        receipt = self._receipt()
        snapshot = self.snapshot_directory()
        self._verify_snapshot_files(snapshot)
        for relative, expected in receipt.file_sha256.items():
            if _sha256(snapshot / relative) != expected:
                raise ModelManagementError("checksum_mismatch")
        return receipt

    def info(self, *, verify: bool = False) -> EmbeddingModelInfo:
        if not self.is_installed():
            return self._info("not_installed")
        try:
            receipt = self.verify() if verify else self._receipt()
        except ModelManagementError as exc:
            return self._info("invalid", reason_code=exc.code)
        return self._info(
            "verified" if verify else "installed",
            resolved_revision=receipt.resolved_revision,
        )

    def _info(
        self,
        status: str,
        *,
        resolved_revision: str | None = None,
        reason_code: str | None = None,
    ) -> EmbeddingModelInfo:
        size = 0
        if self.model_directory.is_dir():
            size = sum(
                path.stat().st_size for path in self.model_directory.rglob("*") if path.is_file()
            )
        return EmbeddingModelInfo(
            model_id=self.manifest.model_id,
            status=status,
            model_version=self.manifest.model_version,
            revision=self.manifest.revision,
            resolved_revision=resolved_revision,
            license=self.manifest.license,
            embedding_dimension=self.manifest.embedding_dimension,
            preprocessing_version=self.manifest.preprocessing_version,
            storage_size_bytes=size,
            reason_code=reason_code,
        )

    def remove(self, *, confirmed: bool) -> None:
        if not confirmed:
            raise ModelManagementError("confirmation_required")
        if self.model_directory.is_symlink() or not (
            self.model_directory / _RECEIPT
        ).is_file():
            raise ModelManagementError("unrecognized_model_directory")
        shutil.rmtree(self.model_directory)
