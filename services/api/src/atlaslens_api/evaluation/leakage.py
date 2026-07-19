from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlaslens_api.evaluation.holdout import (
    HoldoutManifestLoader,
    ValidatedHoldoutRecord,
)
from atlaslens_api.phase6c.reference_index import ReferenceBuildInput
from atlaslens_api.retrieval.errors import ManifestValidationError
from atlaslens_api.retrieval.manifest import validate_manifest
from atlaslens_api.retrieval.models import ManifestRecord
from atlaslens_api.verification.models import VerificationLimits
from atlaslens_api.verification.opencv import OpenCvGeometricVerifier


class LeakageAuditError(ValueError):
    """Safe error for an incomplete or invalid leakage audit input."""


class _AuditModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class LeakageAuditPolicy(_AuditModel):
    audit_version: Literal["atlaslens-leakage-audit-v1"] = "atlaslens-leakage-audit-v1"
    max_references: int = Field(default=10_000, ge=1, le=100_000)
    max_manifest_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_image_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    max_total_image_bytes: int = Field(default=100 * 1024 * 1024 * 1024, ge=1)
    max_decoded_pixels: int = Field(default=40_000_000, ge=1)
    perceptual_distance: int = Field(default=4, ge=0, le=16)
    crop_transform_distance: int = Field(default=3, ge=0, le=16)
    geometric_prefilter_distance: int = Field(default=14, ge=0, le=32)
    max_geometric_checks: int = Field(default=1_000, ge=0, le=10_000)
    geometric_timeout_ms: int = Field(default=3_000, ge=1, le=30_000)
    descriptor_similarity_threshold: float = Field(default=0.995, gt=0, le=1)

    @model_validator(mode="after")
    def validate_distances(self) -> LeakageAuditPolicy:
        if self.crop_transform_distance > self.geometric_prefilter_distance:
            raise ValueError("crop distance cannot exceed the geometric prefilter")
        if self.perceptual_distance > self.geometric_prefilter_distance:
            raise ValueError("perceptual distance cannot exceed the geometric prefilter")
        return self


class LeakageExclusion(_AuditModel):
    reference_key: str = Field(min_length=1, max_length=256)
    reasons: tuple[str, ...] = Field(min_length=1)
    sha256_equal: bool
    perceptual_distance: int = Field(ge=0, le=64)
    crop_transform_distance: int = Field(ge=0, le=64)
    geometric_supported: bool
    descriptor_similarity: float | None = Field(default=None, ge=-1, le=1)


class LeakageAuditReport(_AuditModel):
    audit_version: Literal["atlaslens-leakage-audit-v1"]
    status: Literal["passed", "failed", "incomplete"]
    holdout_id: str
    holdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    holdout_perceptual_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    checked_reference_count: int = Field(ge=0)
    excluded_reference_count: int = Field(ge=0)
    checks: dict[str, str]
    descriptor_provider: str | None = None
    descriptor_version: str | None = None
    descriptor_checked_count: int = Field(ge=0)
    audit_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    exclusions: tuple[LeakageExclusion, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DescriptorArtifact:
    """Caller-supplied real descriptor output; this class never creates embeddings."""

    provider: str
    version: str
    vectors: dict[str, NDArray[np.float32]]

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.version.strip() or not self.vectors:
            raise ValueError("descriptor artifact metadata is incomplete")
        normalized: dict[str, NDArray[np.float32]] = {}
        dimension: int | None = None
        for content_sha256, raw in self.vectors.items():
            if (
                len(content_sha256) != 64
                or any(character not in "0123456789abcdef" for character in content_sha256)
            ):
                raise ValueError("descriptor key is not a SHA-256 digest")
            vector = np.asarray(raw, dtype=np.float32)
            if vector.ndim != 1 or not 1 <= len(vector) <= 65_536:
                raise ValueError("descriptor vector has an invalid shape")
            if not np.isfinite(vector).all():
                raise ValueError("descriptor vector contains non-finite values")
            if dimension is None:
                dimension = len(vector)
            elif dimension != len(vector):
                raise ValueError("descriptor dimensions are inconsistent")
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("descriptor vector norm is invalid")
            value = np.ascontiguousarray(vector / norm, dtype=np.float32)
            value.setflags(write=False)
            normalized[content_sha256] = value
        object.__setattr__(self, "vectors", normalized)


@dataclass(frozen=True, slots=True)
class _ImageFingerprint:
    sha256: str
    canonical_hash: int
    full_hashes: tuple[int, ...]
    crop_hashes: tuple[int, ...]


def load_descriptor_artifact(
    path: Path,
    *,
    max_rows: int = 100_001,
    max_file_bytes: int = 2 * 1024 * 1024 * 1024,
) -> DescriptorArtifact:
    """Read normalized descriptor output produced by a real external model/index job."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise LeakageAuditError("descriptor_artifact_unsafe_or_oversized")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise LeakageAuditError("descriptor_artifact_unavailable") from exc
    if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size > max_file_bytes:
        raise LeakageAuditError("descriptor_artifact_unsafe_or_oversized")
    try:
        with np.load(resolved, allow_pickle=False) as payload:
            required = {"provider", "version", "content_sha256", "vectors"}
            if not required.issubset(payload.files):
                raise LeakageAuditError("descriptor_artifact_schema_invalid")
            provider = str(np.asarray(payload["provider"]).item())
            version = str(np.asarray(payload["version"]).item())
            hashes = np.asarray(payload["content_sha256"])
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise LeakageAuditError("descriptor_artifact_invalid") from exc
    if (
        hashes.ndim != 1
        or vectors.ndim != 2
        or len(hashes) != len(vectors)
        or not 1 <= len(hashes) <= max_rows
        or not 1 <= vectors.shape[1] <= 65_536
    ):
        raise LeakageAuditError("descriptor_artifact_shape_invalid")
    keys = [str(item) for item in hashes.tolist()]
    if len(keys) != len(set(keys)):
        raise LeakageAuditError("descriptor_artifact_duplicate_key")
    try:
        return DescriptorArtifact(
            provider=provider,
            version=version,
            vectors={key: vectors[index] for index, key in enumerate(keys)},
        )
    except ValueError as exc:
        raise LeakageAuditError("descriptor_artifact_invalid") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _perceptual_hash(image: Image.Image) -> int:
    """Match the repository's established 64-bit average-hash representation."""

    resized = image.resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.uint8)
    average = float(np.mean(pixels))
    comparisons = pixels >= average
    value = 0
    for bit in comparisons.ravel():
        value = (value << 1) | int(bit)
    return value


def _oriented_images(image: Image.Image) -> tuple[Image.Image, ...]:
    rotations = tuple(image.rotate(angle, expand=True) for angle in (0, 90, 180, 270))
    mirrors = tuple(ImageOps.mirror(item) for item in rotations)
    return rotations + mirrors


def _crop_hashes(image: Image.Image) -> tuple[int, ...]:
    values: set[int] = set()
    anchors = ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (1.0, 1.0))
    for ratio in (0.9, 0.75, 0.6):
        crop_width = max(8, round(image.width * ratio))
        crop_height = max(8, round(image.height * ratio))
        crop_width = min(image.width, crop_width)
        crop_height = min(image.height, crop_height)
        for anchor_x, anchor_y in anchors:
            left = round((image.width - crop_width) * anchor_x)
            top = round((image.height - crop_height) * anchor_y)
            crop = image.crop((left, top, left + crop_width, top + crop_height))
            values.add(_perceptual_hash(crop))
            crop.close()
    return tuple(sorted(values))


def _fingerprint(path: Path, policy: LeakageAuditPolicy) -> _ImageFingerprint:
    size = path.stat().st_size
    if size < 1 or size > policy.max_image_bytes:
        raise LeakageAuditError("leakage_image_size_invalid")
    try:
        with Image.open(path) as source:
            if (
                source.width < 1
                or source.height < 1
                or source.width * source.height > policy.max_decoded_pixels
            ):
                raise LeakageAuditError("leakage_image_dimensions_invalid")
            image = ImageOps.exif_transpose(source).convert("L")
            image.load()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise LeakageAuditError("leakage_image_invalid") from exc
    oriented = _oriented_images(image)
    full_hashes = tuple(sorted({_perceptual_hash(item) for item in oriented}))
    crop_hashes = tuple(
        sorted({value for item in oriented for value in _crop_hashes(item)})
    )
    for item in oriented:
        item.close()
    canonical_hash = _perceptual_hash(image)
    image.close()
    return _ImageFingerprint(
        sha256=_sha256(path),
        canonical_hash=canonical_hash,
        full_hashes=full_hashes,
        crop_hashes=crop_hashes,
    )


def _minimum_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return min((first ^ second).bit_count() for first in left for second in right)


def _reference_key(record: ManifestRecord, index: int) -> str:
    return (
        record.asset_key
        or record.source_record_id
        or f"reference-{index + 1:06d}-{record.content_hash[:12]}"
    )


def _safe_reference_input_image(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or relative.drive or not relative.parts or ".." in relative.parts:
        raise LeakageAuditError("reference_input_image_path_invalid")
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise LeakageAuditError("reference_input_image_path_invalid")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LeakageAuditError("reference_input_image_unavailable") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise LeakageAuditError("reference_input_image_path_invalid")
    return resolved


def write_filtered_phase6c_reference_input(
    *,
    source_path: Path,
    destination_path: Path,
    excluded_reference_keys: tuple[str, ...],
    max_manifest_bytes: int = 20 * 1024 * 1024,
) -> int:
    """Atomically copy a Phase 6C input minus unambiguously excluded records."""

    expanded_source = source_path.expanduser()
    if expanded_source.is_symlink():
        raise LeakageAuditError("reference_input_unsafe_or_oversized")
    try:
        source = expanded_source.resolve(strict=True)
        source_size = source.stat().st_size
    except (OSError, RuntimeError) as exc:
        raise LeakageAuditError("reference_input_unavailable") from exc
    if not source.is_file() or not 0 < source_size <= max_manifest_bytes:
        raise LeakageAuditError("reference_input_unsafe_or_oversized")
    try:
        document = ReferenceBuildInput.model_validate_json(
            source.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise LeakageAuditError("reference_input_invalid") from exc

    positions_by_key: dict[str, list[int]] = {}
    for position, record in enumerate(document.records):
        positions_by_key.setdefault(record.asset_key, []).append(position)
    excluded_positions: set[int] = set()
    for key in set(excluded_reference_keys):
        positions = positions_by_key.get(key, [])
        if len(positions) != 1:
            raise LeakageAuditError("filtered_reference_mapping_ambiguous")
        excluded_positions.add(positions[0])
    if excluded_positions and len(excluded_positions) == len(document.records):
        raise LeakageAuditError("filtered_reference_input_empty")

    filtered = ReferenceBuildInput(
        schema_version=document.schema_version,
        records=tuple(
            record
            for position, record in enumerate(document.records)
            if position not in excluded_positions
        ),
    )
    expanded_destination = destination_path.expanduser()
    if expanded_destination.is_symlink():
        raise LeakageAuditError("filtered_reference_output_unsafe")
    try:
        expanded_destination.parent.mkdir(parents=True, exist_ok=True)
        parent = expanded_destination.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LeakageAuditError("filtered_reference_output_unavailable") from exc
    destination = parent / expanded_destination.name
    if (
        destination == source
        or not destination.name
        or (destination.exists() and (destination.is_symlink() or not destination.is_file()))
    ):
        raise LeakageAuditError("filtered_reference_output_unsafe")

    temporary = parent / f".{destination.name}.{uuid4().hex}.tmp"
    serialized = (
        json.dumps(filtered.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        raise LeakageAuditError("filtered_reference_output_unavailable") from exc
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return len(filtered.records)


class ReferenceLeakageAuditor:
    def __init__(
        self,
        policy: LeakageAuditPolicy | None = None,
        *,
        verifier: OpenCvGeometricVerifier | None = None,
    ) -> None:
        self.policy = policy or LeakageAuditPolicy()
        self._verifier = verifier or OpenCvGeometricVerifier()

    def load_reference_manifest(
        self,
        manifest_path: Path,
        asset_root: Path,
    ) -> tuple[ManifestRecord, ...]:
        expanded = manifest_path.expanduser()
        if expanded.is_symlink():
            raise LeakageAuditError("reference_manifest_unsafe_or_oversized")
        try:
            resolved = expanded.resolve(strict=True)
        except OSError as exc:
            raise LeakageAuditError("reference_manifest_unavailable") from exc
        if (
            resolved.is_symlink()
            or not resolved.is_file()
            or resolved.stat().st_size > self.policy.max_manifest_bytes
        ):
            raise LeakageAuditError("reference_manifest_unsafe_or_oversized")
        try:
            records = tuple(validate_manifest(resolved, asset_root))
        except ManifestValidationError as exc:
            raise LeakageAuditError("reference_manifest_invalid") from exc
        if not records:
            raise LeakageAuditError("reference_manifest_empty")
        if len(records) > self.policy.max_references:
            raise LeakageAuditError("reference_count_limit_exceeded")
        total_bytes = sum(record.image_path.stat().st_size for record in records)
        if total_bytes > self.policy.max_total_image_bytes:
            raise LeakageAuditError("reference_total_bytes_limit_exceeded")
        if any(
            record.image_path.stat().st_size > self.policy.max_image_bytes
            for record in records
        ):
            raise LeakageAuditError("reference_image_size_limit_exceeded")
        return records

    def load_phase6c_reference_input(
        self,
        manifest_path: Path,
        asset_root: Path,
    ) -> tuple[ManifestRecord, ...]:
        """Adapt the checked Phase 6C builder input without weakening path checks."""

        expanded_manifest = manifest_path.expanduser()
        expanded_root = asset_root.expanduser()
        if expanded_manifest.is_symlink() or expanded_root.is_symlink():
            raise LeakageAuditError("reference_input_unsafe_or_oversized")
        try:
            resolved_manifest = expanded_manifest.resolve(strict=True)
            root = expanded_root.resolve(strict=True)
            manifest_size = resolved_manifest.stat().st_size
        except (OSError, RuntimeError) as exc:
            raise LeakageAuditError("reference_input_unavailable") from exc
        if (
            not resolved_manifest.is_file()
            or not root.is_dir()
            or manifest_size < 1
            or manifest_size > self.policy.max_manifest_bytes
        ):
            raise LeakageAuditError("reference_input_unsafe_or_oversized")
        try:
            payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
            build_input = ReferenceBuildInput.model_validate(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise LeakageAuditError("reference_input_invalid") from exc
        if not build_input.records:
            raise LeakageAuditError("reference_manifest_empty")
        if len(build_input.records) > self.policy.max_references:
            raise LeakageAuditError("reference_count_limit_exceeded")

        references: list[ManifestRecord] = []
        total_bytes = 0
        for record in build_input.records:
            image_path = _safe_reference_input_image(root, record.image_path)
            try:
                image_size = image_path.stat().st_size
            except OSError as exc:
                raise LeakageAuditError("reference_input_image_unavailable") from exc
            if image_size < 1 or image_size > self.policy.max_image_bytes:
                raise LeakageAuditError("reference_image_size_limit_exceeded")
            total_bytes += image_size
            if total_bytes > self.policy.max_total_image_bytes:
                raise LeakageAuditError("reference_total_bytes_limit_exceeded")
            references.append(
                ManifestRecord(
                    image_path=image_path,
                    latitude=record.latitude,
                    longitude=record.longitude,
                    country=record.country,
                    region=record.province,
                    city=record.city,
                    license=record.license,
                    source=record.source,
                    content_hash=record.expected_sha256 or _sha256(image_path),
                    asset_key=record.asset_key,
                    source_record_id=record.source_image_id,
                    source_url=record.source_url,
                    license_url=record.license_url,
                    attribution=record.attribution,
                    display_allowed=False,
                    capture_type="street_level_reference",
                    capture_family_id=record.source_sequence_id,
                    coordinate_kind="camera_raw",
                    coordinate_uncertainty_m=record.coordinate_uncertainty_m,
                    captured_at=(
                        record.captured_at.isoformat()
                        if record.captured_at is not None
                        else None
                    ),
                    heading_degrees=record.heading_degrees,
                    # Phase 6C stores dHash64; the auditor independently computes
                    # its established aHash signatures instead of conflating them.
                    perceptual_hash=None,
                    geographic_cell=record.province,
                )
            )
        return tuple(references)

    def audit_manifest(
        self,
        *,
        holdout_manifest: Path,
        reference_manifest: Path,
        reference_root: Path,
        holdout_id: str | None = None,
        descriptor_artifact: DescriptorArtifact | None = None,
    ) -> LeakageAuditReport:
        holdout = HoldoutManifestLoader(
            max_image_bytes=self.policy.max_image_bytes,
            max_decoded_pixels=self.policy.max_decoded_pixels,
        ).load(holdout_manifest)
        selected = [
            item
            for item in holdout.records
            if holdout_id is None or item.metadata.id == holdout_id
        ]
        if len(selected) != 1:
            raise LeakageAuditError("holdout_selection_must_resolve_to_one_record")
        references = self.load_reference_manifest(reference_manifest, reference_root)
        return self.audit(selected[0], references, descriptor_artifact=descriptor_artifact)

    def audit_phase6c_reference_input(
        self,
        *,
        holdout_manifest: Path,
        reference_input: Path,
        reference_root: Path,
        holdout_id: str | None = None,
        descriptor_artifact: DescriptorArtifact | None = None,
    ) -> LeakageAuditReport:
        holdout = HoldoutManifestLoader(
            max_image_bytes=self.policy.max_image_bytes,
            max_decoded_pixels=self.policy.max_decoded_pixels,
        ).load(holdout_manifest)
        selected = [
            item
            for item in holdout.records
            if holdout_id is None or item.metadata.id == holdout_id
        ]
        if len(selected) != 1:
            raise LeakageAuditError("holdout_selection_must_resolve_to_one_record")
        references = self.load_phase6c_reference_input(reference_input, reference_root)
        return self.audit(selected[0], references, descriptor_artifact=descriptor_artifact)

    def audit(
        self,
        holdout: ValidatedHoldoutRecord,
        references: tuple[ManifestRecord, ...],
        *,
        descriptor_artifact: DescriptorArtifact | None = None,
    ) -> LeakageAuditReport:
        if not references:
            raise LeakageAuditError("reference_manifest_empty")
        if len(references) > self.policy.max_references:
            raise LeakageAuditError("reference_count_limit_exceeded")
        try:
            total_bytes = sum(item.image_path.stat().st_size for item in references)
        except OSError as exc:
            raise LeakageAuditError("reference_image_unavailable") from exc
        if total_bytes > self.policy.max_total_image_bytes:
            raise LeakageAuditError("reference_total_bytes_limit_exceeded")
        target = _fingerprint(holdout.image_path, self.policy)
        target_bytes = holdout.image_path.read_bytes()
        exclusions: list[LeakageExclusion] = []
        limitations: list[str] = [
            "crop_transform_detection_uses_bounded_orientation_and_anchor_signatures",
            "geometric_verification_can_be_inconclusive_for_low_texture_scenes",
        ]
        checks = {
            "sha256": "completed",
            "perceptual_hash": "completed",
            "crop_transform_hash": "completed",
            "geometric_verification": "completed",
            "source_image_identity": (
                "completed"
                if holdout.metadata.source_image_id is not None
                else "not_available_in_holdout_metadata"
            ),
            "capture_family": (
                "completed"
                if holdout.metadata.capture_family_id is not None
                else "not_available_in_holdout_metadata"
            ),
            "descriptor_similarity": (
                "completed"
                if descriptor_artifact is not None
                else "not_run_no_real_descriptor_artifact"
            ),
        }
        if descriptor_artifact is None:
            limitations.append("descriptor_similarity_not_run_without_real_artifact")
        if holdout.metadata.source_image_id is None:
            limitations.append("holdout_source_image_identity_not_available")
        if holdout.metadata.capture_family_id is None:
            limitations.append("holdout_capture_family_not_available")
        target_vector = (
            descriptor_artifact.vectors.get(target.sha256)
            if descriptor_artifact is not None
            else None
        )
        descriptor_incomplete = descriptor_artifact is not None and target_vector is None
        if descriptor_incomplete:
            checks["descriptor_similarity"] = "incomplete_target_descriptor_missing"
            limitations.append("target_descriptor_missing_from_supplied_artifact")

        geometry_checks = 0
        geometry_limit_hit = False
        descriptor_checked = 0
        for index, record in enumerate(references):
            reference = _fingerprint(record.image_path, self.policy)
            full_distance = _minimum_distance(target.full_hashes, reference.full_hashes)
            crop_distance = min(
                _minimum_distance(target.full_hashes, reference.crop_hashes),
                _minimum_distance(target.crop_hashes, reference.full_hashes),
            )
            reasons: list[str] = []
            if record.content_hash != reference.sha256:
                reasons.append("reference_declared_sha256_mismatch")
            if target.sha256 == reference.sha256:
                reasons.append("target_sha256_match")
            if full_distance <= self.policy.perceptual_distance:
                reasons.append("target_perceptual_near_duplicate")
            if crop_distance <= self.policy.crop_transform_distance:
                reasons.append("target_crop_or_transform_near_duplicate")
            if (
                record.perceptual_hash is not None
                and int(record.perceptual_hash, 16) != reference.canonical_hash
            ):
                reasons.append("reference_declared_perceptual_hash_mismatch")
            if (
                holdout.metadata.source is not None
                and holdout.metadata.source_image_id is not None
                and record.source.casefold() == holdout.metadata.source.casefold()
                and record.source_record_id == holdout.metadata.source_image_id
            ):
                reasons.append("target_source_image_match")
            if (
                holdout.metadata.capture_family_id is not None
                and record.capture_family_id == holdout.metadata.capture_family_id
            ):
                reasons.append("target_capture_family_match")

            geometric_supported = False
            if (
                not reasons
                and min(full_distance, crop_distance)
                <= self.policy.geometric_prefilter_distance
            ):
                if geometry_checks >= self.policy.max_geometric_checks:
                    geometry_limit_hit = True
                else:
                    geometry_checks += 1
                    result = self._verifier.verify(
                        target_bytes,
                        record.image_path.read_bytes(),
                        limits=VerificationLimits(
                            timeout_ms=self.policy.geometric_timeout_ms
                        ),
                    )
                    geometric_supported = result.status == "supported"
                    if geometric_supported:
                        reasons.append("target_geometric_near_duplicate")

            similarity: float | None = None
            if descriptor_artifact is not None and target_vector is not None:
                reference_vector = descriptor_artifact.vectors.get(reference.sha256)
                if reference_vector is None:
                    descriptor_incomplete = True
                else:
                    descriptor_checked += 1
                    similarity = float(np.dot(target_vector, reference_vector))
                    if similarity >= self.policy.descriptor_similarity_threshold:
                        reasons.append("target_descriptor_near_duplicate")

            if reasons:
                exclusions.append(
                    LeakageExclusion(
                        reference_key=_reference_key(record, index),
                        reasons=tuple(sorted(set(reasons))),
                        sha256_equal=target.sha256 == reference.sha256,
                        perceptual_distance=full_distance,
                        crop_transform_distance=crop_distance,
                        geometric_supported=geometric_supported,
                        descriptor_similarity=(
                            round(similarity, 6) if similarity is not None else None
                        ),
                    )
                )

        if geometry_limit_hit:
            checks["geometric_verification"] = "incomplete_check_limit_exceeded"
            limitations.append("geometric_candidate_limit_exceeded_audit_is_incomplete")
        if descriptor_artifact is not None and descriptor_incomplete:
            checks["descriptor_similarity"] = "incomplete_descriptor_coverage"
            limitations.append("supplied_descriptor_artifact_did_not_cover_every_reference")

        incomplete = geometry_limit_hit or (
            descriptor_artifact is not None and descriptor_incomplete
        )
        status: Literal["passed", "failed", "incomplete"] = (
            "failed" if exclusions else "incomplete" if incomplete else "passed"
        )
        canonical = {
            "audit_version": self.policy.audit_version,
            "holdout_sha256": target.sha256,
            "references": [record.content_hash for record in references],
            "exclusions": [item.model_dump(mode="json") for item in exclusions],
            "checks": checks,
        }
        audit_fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return LeakageAuditReport(
            audit_version=self.policy.audit_version,
            status=status,
            holdout_id=holdout.metadata.id,
            holdout_sha256=target.sha256,
            holdout_perceptual_hash=f"{target.canonical_hash:016x}",
            checked_reference_count=len(references),
            excluded_reference_count=len(exclusions),
            checks=checks,
            descriptor_provider=(
                descriptor_artifact.provider if descriptor_artifact is not None else None
            ),
            descriptor_version=(
                descriptor_artifact.version if descriptor_artifact is not None else None
            ),
            descriptor_checked_count=descriptor_checked,
            audit_fingerprint=audit_fingerprint,
            exclusions=tuple(exclusions),
            limitations=tuple(sorted(set(limitations))),
        )
