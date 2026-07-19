from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from atlaslens_api.trained_artifacts.errors import TrainedArtifactError
from atlaslens_api.trained_artifacts.models import (
    ArtifactInfo,
    DeploymentMode,
    PromotionReceipt,
    PromotionReport,
    RegisteredArtifactReceipt,
    TrainedArtifactManifest,
)

_MANIFEST = "manifest.json"
_RECEIPT = "registration.json"
_PROMOTION = "promotion.json"
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_PROMOTION_REPORT_BYTES = 2 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    manifest: TrainedArtifactManifest
    receipt: RegisteredArtifactReceipt
    artifact_path: Path

    def __repr__(self) -> str:
        return (
            f"VerifiedArtifact(model_id={self.manifest.model_id!r}, "
            "artifact_path=<redacted>)"
        )


@dataclass(frozen=True, slots=True)
class VerifiedArtifactPayload:
    manifest: TrainedArtifactManifest
    receipt: RegisteredArtifactReceipt
    artifact_bytes: bytes

    def __repr__(self) -> str:
        return (
            f"VerifiedArtifactPayload(model_id={self.manifest.model_id!r}, "
            "artifact_bytes=<redacted>)"
        )


def _sha256(path: Path, *, maximum_bytes: int | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_COPY_CHUNK_BYTES):
            size += len(chunk)
            if maximum_bytes is not None and size > maximum_bytes:
                raise TrainedArtifactError("artifact_too_large")
            digest.update(chunk)
    return digest.hexdigest(), size


def _canonical_manifest(manifest: TrainedArtifactManifest) -> bytes:
    return json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _artifact_identity(manifest_bytes: bytes, artifact_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(manifest_bytes)
    digest.update(b"\n")
    digest.update(artifact_sha256.encode("ascii"))
    return digest.hexdigest()


def load_trained_manifest(path: Path) -> TrainedArtifactManifest:
    if path.is_symlink() or not path.is_file():
        raise TrainedArtifactError("manifest_unavailable")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TrainedArtifactError("manifest_unavailable") from exc
    if not payload or len(payload) > _MAX_MANIFEST_BYTES:
        raise TrainedArtifactError("manifest_size_invalid")
    try:
        parsed: Any
        if path.suffix.casefold() == ".json":
            parsed = json.loads(payload)
        else:
            parsed = yaml.safe_load(payload)
        if not isinstance(parsed, dict):
            raise ValueError("manifest must contain a mapping")
        return TrainedArtifactManifest.model_validate(parsed)
    except (UnicodeError, json.JSONDecodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise TrainedArtifactError("invalid_manifest") from exc


class TrainedArtifactManager:
    def __init__(self, cache_root: Path) -> None:
        expanded = cache_root.expanduser()
        if expanded.is_symlink():
            raise TrainedArtifactError("unsafe_cache_root")
        self.cache_root = expanded.resolve()
        self.registry_root = self.cache_root / "trained-artifacts"

    def _model_directory(self, model_id: str) -> Path:
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789_.-"
        if not model_id or any(character not in allowed for character in model_id):
            raise TrainedArtifactError("invalid_model_id")
        candidate = (self.registry_root / model_id).resolve()
        try:
            candidate.relative_to(self.registry_root)
        except ValueError as exc:
            raise TrainedArtifactError("unsafe_registry_path") from exc
        return candidate

    @staticmethod
    def _ensure_safe_file(path: Path, *, code: str) -> None:
        if path.is_symlink() or not path.is_file():
            raise TrainedArtifactError(code)

    def _prepare_root(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        if self.cache_root.is_symlink():
            raise TrainedArtifactError("unsafe_cache_root")
        self.registry_root.mkdir(parents=True, exist_ok=True)
        if self.registry_root.is_symlink():
            raise TrainedArtifactError("unsafe_registry_root")

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, object]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists() or temporary.is_symlink():
            raise TrainedArtifactError("partial_registry_write")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as target:
                json.dump(payload, target, ensure_ascii=False, indent=2, sort_keys=True)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _copy_verified(source: Path, destination: Path, maximum_bytes: int) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as reader, destination.open("xb") as writer:
            while chunk := reader.read(_COPY_CHUNK_BYTES):
                size += len(chunk)
                if size > maximum_bytes:
                    raise TrainedArtifactError("artifact_too_large")
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        return digest.hexdigest(), size

    def register_local(self, manifest_path: Path) -> RegisteredArtifactReceipt:
        manifest_path = manifest_path.expanduser()
        if manifest_path.is_symlink():
            raise TrainedArtifactError("unsafe_manifest_path")
        manifest_path = manifest_path.resolve()
        manifest = load_trained_manifest(manifest_path)
        self._prepare_root()
        model_directory = self._model_directory(manifest.model_id)
        if model_directory.exists() or model_directory.is_symlink():
            raise TrainedArtifactError("model_already_registered")

        source = manifest_path.parent / manifest.artifact_file
        self._ensure_safe_file(source, code="artifact_unavailable")
        if source.parent.resolve() != manifest_path.parent.resolve():
            raise TrainedArtifactError("unsafe_artifact_path")
        staging = self.registry_root / f".{manifest.model_id}.registering-{os.getpid()}"
        if staging.exists() or staging.is_symlink():
            raise TrainedArtifactError("partial_registration_exists")
        staging.mkdir()
        try:
            artifact_target = staging / manifest.artifact_file
            digest, size = self._copy_verified(
                source, artifact_target, manifest.artifact_size_bytes
            )
            if digest != manifest.artifact_sha256 or size != manifest.artifact_size_bytes:
                raise TrainedArtifactError("artifact_identity_mismatch")
            manifest_bytes = _canonical_manifest(manifest)
            manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
            receipt = RegisteredArtifactReceipt(
                model_id=manifest.model_id,
                provider_id=manifest.provider_id,
                model_version=manifest.model_version,
                artifact_file=manifest.artifact_file,
                artifact_sha256=digest,
                artifact_size_bytes=size,
                manifest_sha256=manifest_digest,
                artifact_identity=_artifact_identity(manifest_bytes, digest),
                registered_at=datetime.now(UTC),
            )
            self._atomic_json(
                staging / _MANIFEST, manifest.model_dump(mode="json")
            )
            self._atomic_json(staging / _RECEIPT, receipt.model_dump(mode="json"))
            os.replace(staging, model_directory)
            return receipt
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _load(
        self, model_id: str
    ) -> tuple[TrainedArtifactManifest, RegisteredArtifactReceipt, Path]:
        directory = self._model_directory(model_id)
        if directory.is_symlink() or not directory.is_dir():
            raise TrainedArtifactError("model_not_registered")
        manifest_path = directory / _MANIFEST
        receipt_path = directory / _RECEIPT
        self._ensure_safe_file(manifest_path, code="registration_incomplete")
        self._ensure_safe_file(receipt_path, code="registration_incomplete")
        try:
            manifest = TrainedArtifactManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            receipt = RegisteredArtifactReceipt.model_validate_json(
                receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise TrainedArtifactError("registration_invalid") from exc
        artifact = directory / receipt.artifact_file
        self._ensure_safe_file(artifact, code="registration_incomplete")
        if artifact.parent.resolve() != directory.resolve():
            raise TrainedArtifactError("unsafe_registry_path")
        return manifest, receipt, artifact

    def verify(self, model_id: str) -> RegisteredArtifactReceipt:
        manifest, receipt, artifact = self._load(model_id)
        if manifest.license.status != "approved":
            raise TrainedArtifactError("license_not_approved")
        manifest_bytes = _canonical_manifest(manifest)
        digest, size = _sha256(artifact, maximum_bytes=manifest.artifact_size_bytes)
        expected_identity = _artifact_identity(manifest_bytes, digest)
        if (
            hashlib.sha256(manifest_bytes).hexdigest() != receipt.manifest_sha256
            or digest != manifest.artifact_sha256
            or digest != receipt.artifact_sha256
            or size != manifest.artifact_size_bytes
            or size != receipt.artifact_size_bytes
            or expected_identity != receipt.artifact_identity
            or receipt.model_id != manifest.model_id
            or receipt.provider_id != manifest.provider_id
            or receipt.model_version != manifest.model_version
        ):
            raise TrainedArtifactError("artifact_identity_mismatch")
        verified = receipt.model_copy(
            update={
                "verified": True,
                "verified_at": datetime.now(UTC),
                "mode": (
                    receipt.mode if receipt.verified else DeploymentMode.SHADOW
                ),
            }
        )
        self._atomic_json(
            self._model_directory(model_id) / _RECEIPT,
            verified.model_dump(mode="json"),
        )
        return verified

    def verified_artifact(self, model_id: str) -> VerifiedArtifact:
        manifest, receipt, artifact = self._load(model_id)
        if not receipt.verified or receipt.mode == DeploymentMode.DISABLED:
            raise TrainedArtifactError("artifact_not_verified")
        if manifest.license.status != "approved":
            raise TrainedArtifactError("license_not_approved")
        self.verify(model_id)
        _, current, current_artifact = self._load(model_id)
        return VerifiedArtifact(manifest=manifest, receipt=current, artifact_path=current_artifact)

    def verified_artifact_payload(
        self, model_id: str, *, maximum_bytes: int
    ) -> VerifiedArtifactPayload:
        """Read and authenticate the exact immutable bytes passed to a runtime."""
        manifest, receipt, artifact = self._load(model_id)
        if not receipt.verified or receipt.mode == DeploymentMode.DISABLED:
            raise TrainedArtifactError("artifact_not_verified")
        limit = min(maximum_bytes, manifest.artifact_size_bytes)
        digest = hashlib.sha256()
        size = 0
        payload = bytearray()
        try:
            with artifact.open("rb") as source:
                while chunk := source.read(_COPY_CHUNK_BYTES):
                    size += len(chunk)
                    if size > limit:
                        raise TrainedArtifactError("artifact_too_large")
                    digest.update(chunk)
                    payload.extend(chunk)
        except OSError as exc:
            raise TrainedArtifactError("registration_incomplete") from exc
        manifest_bytes = _canonical_manifest(manifest)
        actual_digest = digest.hexdigest()
        if (
            size != manifest.artifact_size_bytes
            or size != receipt.artifact_size_bytes
            or actual_digest != manifest.artifact_sha256
            or actual_digest != receipt.artifact_sha256
            or hashlib.sha256(manifest_bytes).hexdigest() != receipt.manifest_sha256
            or _artifact_identity(manifest_bytes, actual_digest) != receipt.artifact_identity
            or receipt.model_id != manifest.model_id
            or receipt.provider_id != manifest.provider_id
            or receipt.model_version != manifest.model_version
        ):
            raise TrainedArtifactError("artifact_identity_mismatch")
        return VerifiedArtifactPayload(
            manifest=manifest,
            receipt=receipt,
            artifact_bytes=bytes(payload),
        )

    def registered_artifact(self, model_id: str) -> VerifiedArtifact:
        """Return receipted metadata without loading or rehashing model bytes."""
        manifest, receipt, artifact = self._load(model_id)
        if not receipt.verified or receipt.mode == DeploymentMode.DISABLED:
            raise TrainedArtifactError("artifact_not_verified")
        return VerifiedArtifact(manifest=manifest, receipt=receipt, artifact_path=artifact)

    def info(self, model_id: str) -> ArtifactInfo:
        try:
            manifest, receipt, _ = self._load(model_id)
        except TrainedArtifactError as exc:
            if exc.code == "model_not_registered":
                return ArtifactInfo(model_id=model_id, status="not_registered")
            return ArtifactInfo(model_id=model_id, status="invalid", reason_code=exc.code)
        return ArtifactInfo(
            model_id=manifest.model_id,
            provider_id=manifest.provider_id,
            model_version=manifest.model_version,
            artifact_identity=receipt.artifact_identity,
            mode=receipt.mode,
            verified=receipt.verified,
            status="verified" if receipt.verified else "registered",
        )

    def list(self) -> tuple[ArtifactInfo, ...]:
        if not self.registry_root.is_dir() or self.registry_root.is_symlink():
            return ()
        model_ids = sorted(
            path.name
            for path in self.registry_root.iterdir()
            if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
        )
        return tuple(self.info(model_id) for model_id in model_ids)

    def promote(
        self,
        model_id: str,
        *,
        from_mode: DeploymentMode,
        to_mode: DeploymentMode,
        report_path: Path,
    ) -> PromotionReceipt:
        self.verify(model_id)
        manifest, receipt, _ = self._load(model_id)
        if not receipt.verified:
            raise TrainedArtifactError("artifact_not_verified")
        if receipt.mode != from_mode:
            raise TrainedArtifactError("deployment_mode_mismatch")
        allowed = {
            (DeploymentMode.SHADOW, DeploymentMode.CANDIDATE),
            (DeploymentMode.CANDIDATE, DeploymentMode.PRIMARY),
        }
        if (from_mode, to_mode) not in allowed:
            raise TrainedArtifactError("invalid_promotion_transition")
        report_path = report_path.expanduser()
        if report_path.is_symlink():
            raise TrainedArtifactError("unsafe_promotion_report")
        report_path = report_path.resolve()
        self._ensure_safe_file(report_path, code="promotion_report_unavailable")
        if report_path.stat().st_size > _MAX_PROMOTION_REPORT_BYTES:
            raise TrainedArtifactError("promotion_report_too_large")
        try:
            report_bytes = report_path.read_bytes()
            parsed = yaml.safe_load(report_bytes)
            report = PromotionReport.model_validate(parsed)
        except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
            raise TrainedArtifactError("promotion_report_invalid") from exc
        minimum_count = (
            manifest.evaluation.primary_minimum_sample_count
            if to_mode == DeploymentMode.PRIMARY
            else manifest.evaluation.minimum_sample_count
        )
        required_checks = [
            report.safety_checks_passed,
            report.output_schema_compatible,
            report.lineage_verified,
            report.license_approved,
            report.subgroup_regressions_passed,
            report.operator_approved,
        ]
        if to_mode == DeploymentMode.PRIMARY:
            required_checks.append(report.runtime_isolation_verified)
        latency_limit = manifest.evaluation.maximum_latency_p95_ms
        if (
            report.model_id != manifest.model_id
            or report.model_version != manifest.model_version
            or report.candidate_provider_id != manifest.provider_id
            or report.artifact_identity != receipt.artifact_identity
            or report.sample_count < minimum_count
            or not all(required_checks)
            or report.country_top1_delta
            < -manifest.evaluation.maximum_country_top1_regression
            or report.recall_200km_delta
            < -manifest.evaluation.maximum_recall_200km_regression
            or report.median_error_increase_km
            > manifest.evaluation.maximum_median_error_increase
            or (latency_limit is not None and report.latency_p95_ms > latency_limit)
        ):
            raise TrainedArtifactError("promotion_gate_failed")
        report_digest = hashlib.sha256(report_bytes).hexdigest()
        promotion = PromotionReceipt(
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            artifact_identity=receipt.artifact_identity,
            from_mode=from_mode,
            to_mode=to_mode,
            report_sha256=report_digest,
            evaluation_fingerprint=report.evaluation_fingerprint,
            promoted_at=datetime.now(UTC),
        )
        directory = self._model_directory(model_id)
        self._atomic_json(directory / _PROMOTION, promotion.model_dump(mode="json"))
        self._atomic_json(
            directory / _RECEIPT,
            receipt.model_copy(update={"mode": to_mode}).model_dump(mode="json"),
        )
        return promotion

    def unregister(self, model_id: str, *, confirmed: bool) -> None:
        if not confirmed:
            raise TrainedArtifactError("confirmation_required")
        directory = self._model_directory(model_id)
        if directory.is_symlink() or not (directory / _RECEIPT).is_file():
            raise TrainedArtifactError("unrecognized_model_directory")
        shutil.rmtree(directory)
