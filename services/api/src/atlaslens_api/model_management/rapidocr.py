from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from atlaslens_api.model_management.errors import ModelManagementError
from atlaslens_api.providers.rapidocr import (
    RapidOCRArtifactReceipt,
    RapidOCRInstallationReceipt,
    RapidOCRProfileReceipt,
    load_verified_rapidocr_runtime,
)

_INSTALLATION = "rapidocr-3.9.1"
_DETECTOR = "PP-OCRv6_det_small.onnx"
_RECOGNIZER = "PP-OCRv6_rec_small.onnx"


class RapidOCRModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "rapidocr"
    status: str
    installed: bool
    verified: bool
    provider_version: str = "3.9.1"
    runtime_family: str | None = None
    runtime_version: str | None = None
    profiles: int = 0
    reason_code: str | None = None


class RapidOCRModelManagementService:
    """Stages only models bundled in the explicitly installed RapidOCR wheel."""

    def __init__(self, cache_root: Path) -> None:
        self._root = cache_root.expanduser().resolve()
        self._target = self._root / _INSTALLATION

    def install(self) -> RapidOCRModelInfo:
        current = self.info()
        if current.verified:
            return current
        if self._target.exists() or self._target.is_symlink():
            raise ModelManagementError("invalid_existing_installation")
        package_root = self._package_root()
        detector = self._find_bundled_model(package_root, _DETECTOR)
        recognizer = self._find_bundled_model(package_root, _RECOGNIZER)
        runtime_family, runtime_version = self._runtime()
        self._root.mkdir(parents=True, exist_ok=True)
        staging = self._root / f".rapidocr-staging-{uuid4().hex}"
        try:
            model_directory = staging / "models"
            model_directory.mkdir(parents=True, exist_ok=False)
            artifacts: list[RapidOCRArtifactReceipt] = []
            for source in (detector, recognizer):
                destination = model_directory / source.name
                shutil.copyfile(source, destination)
                with destination.open("rb+") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
                digest, size = self._hash_file(destination)
                artifacts.append(
                    RapidOCRArtifactReceipt(
                        path=f"models/{destination.name}",
                        sha256=digest,
                        size_bytes=size,
                    )
                )
            receipt = RapidOCRInstallationReceipt(
                schema_version="atlaslens-rapidocr-v1",
                provider_version="3.9.1",
                runtime_family=runtime_family,
                runtime_version=runtime_version,
                profiles=(
                    RapidOCRProfileReceipt(
                        profile_id="ppocrv6-small-multi",
                        detector_file=f"models/{_DETECTOR}",
                        recognizer_file=f"models/{_RECOGNIZER}",
                        detector_language="multi",
                        recognizer_language="multi",
                        ocr_version="PP-OCRv6",
                        model_type="small",
                    ),
                ),
                artifacts=tuple(artifacts),
            )
            receipt_path = staging / "receipt.json"
            receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
            with receipt_path.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            device = "cuda" if runtime_family == "onnxruntime-gpu" else "cpu"
            if load_verified_rapidocr_runtime(staging, device=device) is None:
                raise ModelManagementError("staged_model_verification_failed")
            os.replace(staging, self._target)
        finally:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)
        return self.verify()

    def verify(self) -> RapidOCRModelInfo:
        info = self.info()
        if not info.verified:
            raise ModelManagementError(info.reason_code or "model_not_installed")
        return info

    def info(self) -> RapidOCRModelInfo:
        if not self._target.exists():
            return RapidOCRModelInfo(
                status="not_installed", installed=False, verified=False
            )
        if self._target.is_symlink() or not self._target.is_dir():
            return self._invalid("unsafe_installation_path")
        try:
            receipt = RapidOCRInstallationReceipt.model_validate_json(
                (self._target / "receipt.json").read_text(encoding="utf-8")
            )
            device = "cuda" if receipt.runtime_family == "onnxruntime-gpu" else "cpu"
            runtime = load_verified_rapidocr_runtime(self._target, device=device)
            if runtime is None:
                return self._invalid("receipt_or_checksum_invalid")
        except (OSError, ValueError):
            return self._invalid("receipt_or_checksum_invalid")
        return RapidOCRModelInfo(
            status="ready",
            installed=True,
            verified=True,
            runtime_family=receipt.runtime_family,
            runtime_version=receipt.runtime_version,
            profiles=len(receipt.profiles),
        )

    def remove(self, *, confirmed: bool) -> RapidOCRModelInfo:
        if not confirmed:
            raise ModelManagementError("removal_confirmation_required")
        if not self._target.exists():
            return self.info()
        if self._target.is_symlink() or self._target.resolve().parent != self._root:
            raise ModelManagementError("unsafe_installation_path")
        shutil.rmtree(self._target)
        return self.info()

    @staticmethod
    def _package_root() -> Path:
        try:
            if importlib.metadata.version("rapidocr") != "3.9.1":
                raise ModelManagementError("rapidocr_version_mismatch")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ModelManagementError("missing_dependency") from exc
        spec = importlib.util.find_spec("rapidocr")
        if spec is None or spec.origin is None:
            raise ModelManagementError("missing_dependency")
        root = Path(spec.origin).resolve(strict=True).parent
        if root.is_symlink() or not root.is_dir():
            raise ModelManagementError("unsafe_package_path")
        return root

    @staticmethod
    def _find_bundled_model(package_root: Path, filename: str) -> Path:
        matches = [
            path
            for path in package_root.rglob(filename)
            if path.is_file() and not path.is_symlink()
        ]
        if len(matches) != 1:
            raise ModelManagementError("bundled_model_missing_or_ambiguous")
        path = matches[0].resolve(strict=True)
        if package_root not in path.parents or not 0 < path.stat().st_size <= 512 * 1024 * 1024:
            raise ModelManagementError("bundled_model_invalid")
        return path

    @staticmethod
    def _runtime() -> tuple[str, str]:
        for distribution in ("onnxruntime-gpu", "onnxruntime"):
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
            if version != "1.27.0":
                raise ModelManagementError("onnxruntime_version_mismatch")
            family = "onnxruntime-gpu" if distribution == "onnxruntime-gpu" else "onnxruntime-cpu"
            return family, version
        raise ModelManagementError("missing_dependency")

    @staticmethod
    def _hash_file(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            while chunk := source.read(64 * 1024):
                size += len(chunk)
                digest.update(chunk)
        return digest.hexdigest(), size

    @staticmethod
    def _invalid(reason: str) -> RapidOCRModelInfo:
        return RapidOCRModelInfo(
            status="invalid",
            installed=True,
            verified=False,
            reason_code=reason,
        )
