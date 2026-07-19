from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Protocol, cast
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REFERENCE_INPUT_SCHEMA_VERSION: Final = "atlaslens-megaloc-reference-input-v1"
REFERENCE_INDEX_SCHEMA_VERSION: Final = "atlaslens-megaloc-reference-index-v1"
REFERENCE_METADATA_SCHEMA_VERSION: Final = "atlaslens-megaloc-reference-metadata-v1"
REFERENCE_EXCLUSIONS_SCHEMA_VERSION: Final = "atlaslens-megaloc-reference-exclusions-v1"
MANIFEST_FILENAME = "manifest.json"
VECTORS_FILENAME = "descriptors.npy"
METADATA_FILENAME = "references.json"
EXCLUSIONS_FILENAME = "exclusions.json"
LEAKAGE_ATTESTATION_FILENAME = "leakage-attestation.json"

_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PERCEPTUAL_HASH = re.compile(r"^[0-9a-f]{16}$")
_SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

FloatVector = NDArray[np.float32]
ExclusionReason = Literal[
    "excluded_sha256",
    "excluded_perceptual_hash",
    "duplicate_sha256",
    "duplicate_source_image",
    "near_duplicate_perceptual_hash",
    "near_duplicate_descriptor",
    "sequence_limit",
    "province_limit",
    "index_limit",
]


class ReferenceIndexError(RuntimeError):
    """Base error whose message is safe for local operator tooling."""


class ReferenceIndexUnavailableError(ReferenceIndexError):
    pass


class ReferenceIndexIntegrityError(ReferenceIndexError):
    pass


class ReferenceIndexBuildError(ReferenceIndexError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class MegaLocDescriptorSpec(_StrictModel):
    model_id: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(pattern=_VERSION.pattern)
    source_revision: str = Field(pattern=_VERSION.pattern)
    descriptor_version: str = Field(pattern=_VERSION.pattern)
    dimension: int = Field(gt=0, le=100_000)
    normalization: Literal["l2"] = "l2"
    metric: Literal["cosine"] = "cosine"
    dtype: Literal["float32"] = "float32"


class ReferenceProvenance(_StrictModel):
    reference_id: str = Field(pattern=_OPAQUE_ID.pattern)
    source: Literal["mapillary", "kartaview", "manual"]
    source_family: str = Field(pattern=_OPAQUE_ID.pattern)
    source_image_id: str = Field(pattern=_OPAQUE_ID.pattern)
    source_sequence_id: str = Field(pattern=_OPAQUE_ID.pattern)
    source_url: str = Field(min_length=1, max_length=800)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    coordinate_uncertainty_m: float = Field(gt=0, le=1_000_000)
    heading_degrees: float | None = Field(ge=0, lt=360)
    captured_at: datetime | None
    country: str = Field(min_length=1, max_length=120)
    province: str = Field(min_length=1, max_length=160)
    city: str | None = Field(max_length=160)
    license: str = Field(min_length=1, max_length=500)
    license_url: str = Field(min_length=1, max_length=800)
    attribution: str = Field(min_length=1, max_length=500)
    asset_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")

    @field_validator("source_url", "license_url")
    @classmethod
    def require_safe_https_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme.casefold() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.query
        ):
            raise ValueError("provenance URLs must be credential-free stable HTTPS URLs")
        return value

    @field_validator("asset_key")
    @classmethod
    def require_safe_asset_key(cls, value: str) -> str:
        if ".." in Path(value).parts:
            raise ValueError("asset key must not contain parent traversal")
        return value

    @field_validator("captured_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("capture time must include a timezone")
        return value


class ReferenceBuildRecord(ReferenceProvenance):
    image_path: str = Field(min_length=1, max_length=800)
    descriptor_path: str | None = Field(default=None, min_length=1, max_length=800)
    expected_sha256: str | None = Field(default=None, pattern=_SHA256.pattern)
    expected_perceptual_hash: str | None = Field(
        default=None, pattern=_PERCEPTUAL_HASH.pattern
    )


class ReferenceBuildInput(_StrictModel):
    schema_version: Literal["atlaslens-megaloc-reference-input-v1"]
    records: tuple[ReferenceBuildRecord, ...] = Field(max_length=100_000)

    @model_validator(mode="after")
    def require_unique_reference_ids(self) -> ReferenceBuildInput:
        reference_ids = [record.reference_id for record in self.records]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference IDs must be unique")
        return self


class ReferenceRecord(ReferenceProvenance):
    sha256: str = Field(pattern=_SHA256.pattern)
    perceptual_hash: str = Field(pattern=_PERCEPTUAL_HASH.pattern)
    perceptual_hash_algorithm: Literal["dhash64-v1"] = "dhash64-v1"
    descriptor_version: str = Field(pattern=_VERSION.pattern)


class ReferenceMetadataEnvelope(_StrictModel):
    schema_version: Literal["atlaslens-megaloc-reference-metadata-v1"]
    index_version: str = Field(pattern=_VERSION.pattern)
    records: tuple[ReferenceRecord, ...]


class ReferenceExclusion(_StrictModel):
    reference_id: str = Field(pattern=_OPAQUE_ID.pattern)
    reason: ExclusionReason


class ReferenceExclusionsEnvelope(_StrictModel):
    schema_version: Literal["atlaslens-megaloc-reference-exclusions-v1"]
    index_version: str = Field(pattern=_VERSION.pattern)
    exclusions: tuple[ReferenceExclusion, ...]


class IndexArtifact(_StrictModel):
    filename: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")
    sha256: str = Field(pattern=_SHA256.pattern)
    size_bytes: int = Field(ge=0)


class ReferenceDeduplicationSummary(_StrictModel):
    input_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    counts_by_reason: dict[ExclusionReason, int]
    max_per_sequence: int = Field(gt=0)
    max_per_province: int = Field(gt=0)
    perceptual_hash_hamming_threshold: int = Field(ge=0, le=64)
    descriptor_cosine_distance_threshold: float | None = Field(default=None, ge=0, le=2)

    @model_validator(mode="after")
    def validate_counts(self) -> ReferenceDeduplicationSummary:
        if any(count < 0 for count in self.counts_by_reason.values()):
            raise ValueError("deduplication reason count must be non-negative")
        if self.accepted_count + self.excluded_count != self.input_count:
            raise ValueError("deduplication counts do not cover the input")
        if sum(self.counts_by_reason.values()) != self.excluded_count:
            raise ValueError("deduplication reason counts are inconsistent")
        return self


class ReferenceIndexManifest(_StrictModel):
    schema_version: Literal["atlaslens-megaloc-reference-index-v1"]
    index_version: str = Field(pattern=_VERSION.pattern)
    built_at: datetime
    descriptor: MegaLocDescriptorSpec
    count: int = Field(ge=0)
    independent_sequences: int = Field(ge=0)
    covered_countries: tuple[str, ...]
    covered_provinces: tuple[str, ...]
    images_per_province: dict[str, int]
    source_distribution: dict[str, int]
    deduplication: ReferenceDeduplicationSummary
    vectors: IndexArtifact
    metadata: IndexArtifact
    exclusions: IndexArtifact

    @field_validator("built_at")
    @classmethod
    def require_utc_build_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("build time must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_artifacts_and_counts(self) -> ReferenceIndexManifest:
        if (
            self.vectors.filename,
            self.metadata.filename,
            self.exclusions.filename,
        ) != (VECTORS_FILENAME, METADATA_FILENAME, EXCLUSIONS_FILENAME):
            raise ValueError("index artifact filenames are invalid")
        if self.deduplication.accepted_count != self.count:
            raise ValueError("manifest count does not match deduplication summary")
        if sum(self.images_per_province.values()) != self.count:
            raise ValueError("province counts do not match index count")
        if sum(self.source_distribution.values()) != self.count:
            raise ValueError("source counts do not match index count")
        return self


class ReferenceIndexLeakageAttestation(_StrictModel):
    """Safe index-bound projection of a truth-bearing leakage report."""

    schema_version: Literal["atlaslens-reference-leakage-attestation-v1"] = (
        "atlaslens-reference-leakage-attestation-v1"
    )
    status: Literal["passed"] = "passed"
    audit_version: Literal["atlaslens-leakage-audit-v1"] = (
        "atlaslens-leakage-audit-v1"
    )
    audit_fingerprint: str = Field(pattern=_SHA256.pattern)
    source_report_sha256: str = Field(pattern=_SHA256.pattern)
    index_version: str = Field(pattern=_VERSION.pattern)
    descriptor_version: str = Field(pattern=_VERSION.pattern)
    checked_reference_count: int = Field(ge=0, le=100_000)
    descriptor_checked_count: int = Field(ge=0, le=100_000)
    excluded_reference_count: Literal[0] = 0
    pre_index_excluded_reference_count: int | None = Field(
        default=None,
        ge=0,
        le=100_000,
    )

    @model_validator(mode="after")
    def require_complete_descriptor_check(self) -> ReferenceIndexLeakageAttestation:
        if self.descriptor_checked_count != self.checked_reference_count:
            raise ValueError("leakage attestation descriptor coverage is incomplete")
        return self


class ReferenceSearchHit(_StrictModel):
    reference_id: str = Field(pattern=_OPAQUE_ID.pattern)
    rank: int = Field(ge=1, le=1_000)
    similarity: float = Field(ge=-1, le=1)
    similarity_semantics: Literal["cosine_similarity_not_confidence"] = (
        "cosine_similarity_not_confidence"
    )
    distance: float = Field(ge=0, le=2)
    distance_semantics: Literal["cosine_distance_not_confidence"] = (
        "cosine_distance_not_confidence"
    )
    confidence: None = None
    confidence_semantics: Literal["uncalibrated_unavailable"] = "uncalibrated_unavailable"
    uncertainty_radius_m: float = Field(gt=0)
    reference: ReferenceRecord


class ReferenceIndexDiagnostics(_StrictModel):
    status: Literal["ready", "empty", "unavailable", "invalid"]
    reason_code: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{0,79}$")
    index_version: str | None = Field(default=None, max_length=160)
    descriptor_version: str | None = Field(default=None, max_length=160)
    count: int = Field(ge=0)
    independent_sequences: int = Field(ge=0)
    disk_usage_bytes: int = Field(ge=0)
    country_count: int | None = Field(default=None, ge=0)
    province_count: int | None = Field(default=None, ge=0)
    images_per_province: dict[str, int] | None = Field(default=None, max_length=1_000)
    source_distribution: dict[str, int] | None = Field(default=None, max_length=32)
    built_at: datetime | None = None
    duplicate_count: int | None = Field(default=None, ge=0)
    excluded_count: int | None = Field(default=None, ge=0)
    attributions: tuple[str, ...] | None = Field(default=None, max_length=100)
    leakage_status: Literal["passed", "failed", "not_run", "unavailable"] = "not_run"
    leakage_audit: ReferenceIndexLeakageAttestation | None = None


class ReferenceIndexBuildPolicy(_StrictModel):
    max_images: int = Field(default=5_000, gt=0, le=100_000)
    max_per_sequence: int = Field(default=10, gt=0, le=1_000)
    max_per_province: int = Field(default=100, gt=0, le=10_000)
    max_disk_bytes: int = Field(default=2 * 1024 * 1024 * 1024, gt=0)
    max_image_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=40_000_000, gt=0)
    perceptual_hash_hamming_threshold: int = Field(default=4, ge=0, le=64)
    descriptor_cosine_distance_threshold: float | None = Field(default=None, ge=0, le=2)
    excluded_sha256: frozenset[str] = frozenset()
    excluded_perceptual_hashes: frozenset[str] = frozenset()

    @field_validator("excluded_sha256")
    @classmethod
    def validate_excluded_hashes(cls, values: frozenset[str]) -> frozenset[str]:
        if any(_SHA256.fullmatch(value) is None for value in values):
            raise ValueError("excluded SHA-256 value is invalid")
        return values

    @field_validator("excluded_perceptual_hashes")
    @classmethod
    def validate_excluded_perceptual_hashes(
        cls, values: frozenset[str]
    ) -> frozenset[str]:
        if any(_PERCEPTUAL_HASH.fullmatch(value) is None for value in values):
            raise ValueError("excluded perceptual hash is invalid")
        return values


class DescriptorProvider(Protocol):
    def describe(self, image_path: Path) -> FloatVector: ...


@dataclass(frozen=True, slots=True)
class ReferenceIndexBuildResult:
    manifest: ReferenceIndexManifest
    output_directory: Path


@dataclass(frozen=True, slots=True)
class ReferenceIndexOpenResult:
    diagnostics: ReferenceIndexDiagnostics
    index: MegaLocReferenceIndex | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, model: BaseModel) -> None:
    payload = model.model_dump_json(indent=None, by_alias=False)
    path.write_text(payload + "\n", encoding="utf-8", newline="\n")
    with path.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceIndexIntegrityError("index_json_invalid") from exc


def _attestation_matches_manifest(
    attestation: ReferenceIndexLeakageAttestation,
    manifest: ReferenceIndexManifest,
) -> bool:
    return bool(
        attestation.index_version == manifest.index_version
        and attestation.descriptor_version
        == manifest.descriptor.descriptor_version
        and attestation.checked_reference_count == manifest.count
        and attestation.descriptor_checked_count == manifest.count
        and attestation.excluded_reference_count == 0
    )


def _load_leakage_attestation(
    directory: Path,
    manifest: ReferenceIndexManifest,
) -> tuple[
    Literal["passed", "failed", "not_run", "unavailable"],
    ReferenceIndexLeakageAttestation | None,
]:
    path = directory / LEAKAGE_ATTESTATION_FILENAME
    try:
        if not path.exists():
            return "not_run", None
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            return "failed", None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        return "unavailable", None
    except json.JSONDecodeError:
        return "failed", None
    try:
        attestation = ReferenceIndexLeakageAttestation.model_validate(payload)
    except ValueError:
        return "failed", None
    if not _attestation_matches_manifest(attestation, manifest):
        return "failed", None
    return "passed", attestation


def _safe_input_path(root: Path, relative: str, *, suffix: str | None = None) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or not candidate_relative.parts:
        raise ReferenceIndexBuildError("input_path_invalid")
    candidate = (root / candidate_relative).resolve()
    if (
        not candidate.is_relative_to(root)
        or not candidate.is_file()
        or candidate.is_symlink()
        or (suffix is not None and candidate.suffix.casefold() != suffix)
    ):
        raise ReferenceIndexBuildError("input_path_invalid")
    return candidate


def load_reference_build_input(path: Path) -> ReferenceBuildInput:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ReferenceIndexBuildError("input_manifest_unavailable")
    try:
        return ReferenceBuildInput.model_validate(_read_json(resolved))
    except (ReferenceIndexIntegrityError, ValueError) as exc:
        raise ReferenceIndexBuildError("input_manifest_invalid") from exc


def _hash_and_perceptual_hash(
    path: Path,
    *,
    max_bytes: int,
    max_pixels: int,
) -> tuple[str, str]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ReferenceIndexBuildError("input_image_unavailable") from exc
    if not 0 < size <= max_bytes:
        raise ReferenceIndexBuildError("input_image_invalid")
    digest = _sha256_file(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                source.load()
                if (
                    source.format not in _SUPPORTED_IMAGE_FORMATS
                    or getattr(source, "n_frames", 1) != 1
                    or source.width <= 0
                    or source.height <= 0
                    or source.width * source.height > max_pixels
                ):
                    raise ReferenceIndexBuildError("input_image_invalid")
                normalized = ImageOps.exif_transpose(source).convert("L").resize((9, 8))
                values = np.asarray(normalized, dtype=np.uint8)
    except ReferenceIndexBuildError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ReferenceIndexBuildError("input_image_invalid") from exc
    bits = values[:, 1:] > values[:, :-1]
    number = 0
    for bit in bits.reshape(-1).tolist():
        number = (number << 1) | int(bit)
    return digest, f"{number:016x}"


def _hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _normalized_descriptor(value: object, spec: MegaLocDescriptorSpec) -> FloatVector:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.shape[0] != spec.dimension or not np.isfinite(vector).all():
        raise ReferenceIndexBuildError("descriptor_invalid")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ReferenceIndexBuildError("descriptor_invalid")
    normalized = np.ascontiguousarray(vector / norm, dtype=np.float32)
    normalized.setflags(write=False)
    return normalized


def load_descriptor_matrix(path: Path, spec: MegaLocDescriptorSpec) -> NDArray[np.float32]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.suffix.casefold() != ".npy":
        raise ReferenceIndexBuildError("descriptor_matrix_unavailable")
    try:
        matrix = np.load(resolved, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ReferenceIndexBuildError("descriptor_matrix_invalid") from exc
    if (
        matrix.ndim != 2
        or matrix.shape[1] != spec.dimension
        or matrix.dtype != np.float32
        or not np.isfinite(matrix).all()
    ):
        raise ReferenceIndexBuildError("descriptor_matrix_invalid")
    return cast(NDArray[np.float32], matrix)


def _artifact(path: Path) -> IndexArtifact:
    return IndexArtifact(
        filename=path.name,
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def build_reference_index(
    *,
    input_manifest: Path,
    input_root: Path,
    output_directory: Path,
    index_version: str,
    descriptor_spec: MegaLocDescriptorSpec,
    policy: ReferenceIndexBuildPolicy | None = None,
    descriptor_provider: DescriptorProvider | None = None,
    descriptor_matrix: NDArray[np.float32] | None = None,
    built_at: datetime | None = None,
    replace: bool = False,
) -> ReferenceIndexBuildResult:
    policy = policy or ReferenceIndexBuildPolicy()
    if _VERSION.fullmatch(index_version) is None:
        raise ReferenceIndexBuildError("index_version_invalid")
    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise ReferenceIndexBuildError("input_root_unavailable")
    build_input = load_reference_build_input(input_manifest)
    if descriptor_matrix is not None and descriptor_provider is not None:
        raise ReferenceIndexBuildError("descriptor_source_ambiguous")
    if descriptor_matrix is not None and (
        descriptor_matrix.ndim != 2
        or descriptor_matrix.shape != (len(build_input.records), descriptor_spec.dimension)
    ):
        raise ReferenceIndexBuildError("descriptor_matrix_invalid")

    destination = output_directory.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise ReferenceIndexBuildError("output_exists")
    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    backup = destination.parent / f".{destination.name}.{uuid4().hex}.backup"
    temporary.mkdir(parents=False)

    accepted_records: list[ReferenceRecord] = []
    accepted_vectors: list[FloatVector] = []
    exclusions: list[ReferenceExclusion] = []
    seen_hashes: set[str] = set()
    seen_source_images: set[tuple[str, str]] = set()
    seen_perceptual_hashes: list[str] = []
    sequence_counts: Counter[tuple[str, str]] = Counter()
    province_counts: Counter[str] = Counter()

    def exclude(record: ReferenceBuildRecord, reason: ExclusionReason) -> None:
        exclusions.append(ReferenceExclusion(reference_id=record.reference_id, reason=reason))

    try:
        for position, record in enumerate(build_input.records):
            image_path = _safe_input_path(root, record.image_path)
            digest, perceptual_hash = _hash_and_perceptual_hash(
                image_path,
                max_bytes=policy.max_image_bytes,
                max_pixels=policy.max_image_pixels,
            )
            if record.expected_sha256 is not None and record.expected_sha256 != digest:
                raise ReferenceIndexBuildError("expected_sha256_mismatch")
            if (
                record.expected_perceptual_hash is not None
                and record.expected_perceptual_hash != perceptual_hash
            ):
                raise ReferenceIndexBuildError("expected_perceptual_hash_mismatch")
            source_key = (record.source, record.source_image_id)
            sequence_key = (record.source, record.source_sequence_id)
            reason: ExclusionReason | None = None
            if digest in policy.excluded_sha256:
                reason = "excluded_sha256"
            elif perceptual_hash in policy.excluded_perceptual_hashes:
                reason = "excluded_perceptual_hash"
            elif digest in seen_hashes:
                reason = "duplicate_sha256"
            elif source_key in seen_source_images:
                reason = "duplicate_source_image"
            elif any(
                _hamming_distance(perceptual_hash, existing)
                <= policy.perceptual_hash_hamming_threshold
                for existing in seen_perceptual_hashes
            ):
                reason = "near_duplicate_perceptual_hash"
            elif sequence_counts[sequence_key] >= policy.max_per_sequence:
                reason = "sequence_limit"
            elif province_counts[record.province] >= policy.max_per_province:
                reason = "province_limit"
            elif len(accepted_records) >= policy.max_images:
                reason = "index_limit"
            if reason is not None:
                exclude(record, reason)
                continue

            if descriptor_matrix is not None:
                raw_descriptor: object = descriptor_matrix[position]
            elif record.descriptor_path is not None:
                descriptor_path = _safe_input_path(root, record.descriptor_path, suffix=".npy")
                try:
                    raw_descriptor = np.load(descriptor_path, allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise ReferenceIndexBuildError("descriptor_invalid") from exc
            elif descriptor_provider is not None:
                raw_descriptor = descriptor_provider.describe(image_path)
            else:
                raise ReferenceIndexBuildError("descriptor_source_unavailable")
            descriptor = _normalized_descriptor(raw_descriptor, descriptor_spec)
            duplicate_threshold = policy.descriptor_cosine_distance_threshold
            if duplicate_threshold is not None and accepted_vectors:
                previous = np.stack(accepted_vectors)
                maximum_similarity = float(np.max(previous @ descriptor))
                if 1.0 - maximum_similarity <= duplicate_threshold:
                    exclude(record, "near_duplicate_descriptor")
                    continue

            accepted_records.append(
                ReferenceRecord(
                    **record.model_dump(
                        exclude={
                            "image_path",
                            "descriptor_path",
                            "expected_sha256",
                            "expected_perceptual_hash",
                        }
                    ),
                    sha256=digest,
                    perceptual_hash=perceptual_hash,
                    descriptor_version=descriptor_spec.descriptor_version,
                )
            )
            accepted_vectors.append(descriptor)
            seen_hashes.add(digest)
            seen_source_images.add(source_key)
            seen_perceptual_hashes.append(perceptual_hash)
            sequence_counts[sequence_key] += 1
            province_counts[record.province] += 1

        vectors = (
            np.stack(accepted_vectors).astype(np.float32, copy=False)
            if accepted_vectors
            else np.empty((0, descriptor_spec.dimension), dtype=np.float32)
        )
        estimated_bytes = int(vectors.nbytes)
        if estimated_bytes > policy.max_disk_bytes:
            raise ReferenceIndexBuildError("index_disk_limit_exceeded")
        vectors_path = temporary / VECTORS_FILENAME
        with vectors_path.open("wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

        metadata = ReferenceMetadataEnvelope(
            schema_version=REFERENCE_METADATA_SCHEMA_VERSION,
            index_version=index_version,
            records=tuple(accepted_records),
        )
        metadata_path = temporary / METADATA_FILENAME
        _write_json(metadata_path, metadata)
        exclusion_envelope = ReferenceExclusionsEnvelope(
            schema_version=REFERENCE_EXCLUSIONS_SCHEMA_VERSION,
            index_version=index_version,
            exclusions=tuple(exclusions),
        )
        exclusions_path = temporary / EXCLUSIONS_FILENAME
        _write_json(exclusions_path, exclusion_envelope)
        reason_counts: Counter[ExclusionReason] = Counter(item.reason for item in exclusions)
        manifest = ReferenceIndexManifest(
            schema_version=REFERENCE_INDEX_SCHEMA_VERSION,
            index_version=index_version,
            built_at=(built_at or datetime.now(UTC)),
            descriptor=descriptor_spec,
            count=len(accepted_records),
            independent_sequences=len(sequence_counts),
            covered_countries=tuple(sorted({item.country for item in accepted_records})),
            covered_provinces=tuple(sorted(province_counts)),
            images_per_province=dict(sorted(province_counts.items())),
            source_distribution=dict(
                sorted(Counter(item.source for item in accepted_records).items())
            ),
            deduplication=ReferenceDeduplicationSummary(
                input_count=len(build_input.records),
                accepted_count=len(accepted_records),
                excluded_count=len(exclusions),
                counts_by_reason=dict(sorted(reason_counts.items())),
                max_per_sequence=policy.max_per_sequence,
                max_per_province=policy.max_per_province,
                perceptual_hash_hamming_threshold=policy.perceptual_hash_hamming_threshold,
                descriptor_cosine_distance_threshold=(
                    policy.descriptor_cosine_distance_threshold
                ),
            ),
            vectors=_artifact(vectors_path),
            metadata=_artifact(metadata_path),
            exclusions=_artifact(exclusions_path),
        )
        manifest_path = temporary / MANIFEST_FILENAME
        _write_json(manifest_path, manifest)
        actual_disk = sum(path.stat().st_size for path in temporary.iterdir() if path.is_file())
        if actual_disk > policy.max_disk_bytes:
            raise ReferenceIndexBuildError("index_disk_limit_exceeded")
        MegaLocReferenceIndex.open(temporary)

        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except Exception:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return ReferenceIndexBuildResult(manifest=manifest, output_directory=destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise


class MegaLocReferenceIndex:
    def __init__(
        self,
        directory: Path,
        manifest: ReferenceIndexManifest,
        records: tuple[ReferenceRecord, ...],
        vectors: NDArray[np.float32],
    ) -> None:
        self.directory = directory
        self.manifest = manifest
        self.records = records
        self._vectors = vectors

    @classmethod
    def open(cls, directory: Path) -> MegaLocReferenceIndex:
        resolved = directory.expanduser().resolve()
        manifest_path = resolved / MANIFEST_FILENAME
        if not resolved.is_dir() or not manifest_path.is_file():
            raise ReferenceIndexUnavailableError("reference_index_unavailable")
        try:
            manifest = ReferenceIndexManifest.model_validate(_read_json(manifest_path))
        except ValueError as exc:
            raise ReferenceIndexIntegrityError("reference_index_manifest_invalid") from exc
        for artifact in (manifest.vectors, manifest.metadata, manifest.exclusions):
            path = resolved / artifact.filename
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != artifact.size_bytes
                or _sha256_file(path) != artifact.sha256
            ):
                raise ReferenceIndexIntegrityError("reference_index_checksum_mismatch")
        try:
            metadata = ReferenceMetadataEnvelope.model_validate(
                _read_json(resolved / METADATA_FILENAME)
            )
            exclusions = ReferenceExclusionsEnvelope.model_validate(
                _read_json(resolved / EXCLUSIONS_FILENAME)
            )
            vectors = np.load(resolved / VECTORS_FILENAME, mmap_mode="r", allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise ReferenceIndexIntegrityError("reference_index_artifact_invalid") from exc
        if metadata.index_version != manifest.index_version or (
            exclusions.index_version != manifest.index_version
        ):
            raise ReferenceIndexIntegrityError("reference_index_version_mismatch")
        if (
            vectors.dtype != np.float32
            or vectors.ndim != 2
            or vectors.shape != (manifest.count, manifest.descriptor.dimension)
            or len(metadata.records) != manifest.count
        ):
            raise ReferenceIndexIntegrityError("reference_index_shape_mismatch")
        if vectors.size and (
            not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4, rtol=1e-4)
        ):
            raise ReferenceIndexIntegrityError("reference_index_descriptor_invalid")
        records = metadata.records
        if any(
            record.descriptor_version != manifest.descriptor.descriptor_version
            for record in records
        ):
            raise ReferenceIndexIntegrityError("reference_index_descriptor_version_mismatch")
        if len({item.reference_id for item in records}) != len(records):
            raise ReferenceIndexIntegrityError("reference_index_duplicate_reference_id")
        if len({item.sha256 for item in records}) != len(records):
            raise ReferenceIndexIntegrityError("reference_index_duplicate_sha256")
        if len({(item.source, item.source_image_id) for item in records}) != len(records):
            raise ReferenceIndexIntegrityError("reference_index_duplicate_source_image")
        sequence_count = len({(item.source, item.source_sequence_id) for item in records})
        province_counts = dict(sorted(Counter(item.province for item in records).items()))
        source_counts = dict(sorted(Counter(item.source for item in records).items()))
        actual_exclusion_counts = dict(
            sorted(Counter(item.reason for item in exclusions.exclusions).items())
        )
        sequence_sizes = Counter((item.source, item.source_sequence_id) for item in records)
        exclusion_ids = [item.reference_id for item in exclusions.exclusions]
        if (
            sequence_count != manifest.independent_sequences
            or province_counts != manifest.images_per_province
            or source_counts != manifest.source_distribution
            or tuple(sorted({item.country for item in records})) != manifest.covered_countries
            or tuple(sorted(province_counts)) != manifest.covered_provinces
            or len(exclusions.exclusions) != manifest.deduplication.excluded_count
            or actual_exclusion_counts != manifest.deduplication.counts_by_reason
            or any(
                count > manifest.deduplication.max_per_sequence
                for count in sequence_sizes.values()
            )
            or any(
                count > manifest.deduplication.max_per_province
                for count in province_counts.values()
            )
            or len(exclusion_ids) != len(set(exclusion_ids))
            or set(exclusion_ids).intersection(item.reference_id for item in records)
        ):
            raise ReferenceIndexIntegrityError("reference_index_statistics_mismatch")
        return cls(resolved, manifest, records, vectors)

    @property
    def size(self) -> int:
        return self.manifest.count

    @property
    def available(self) -> bool:
        return self.size > 0

    @property
    def index_version(self) -> str:
        return self.manifest.index_version

    @property
    def descriptor_version(self) -> str:
        return self.manifest.descriptor.descriptor_version

    def search(
        self,
        query_descriptor: object,
        *,
        top_k: int,
        max_per_sequence: int = 2,
    ) -> tuple[ReferenceSearchHit, ...]:
        if not 1 <= top_k <= 1_000 or not 1 <= max_per_sequence <= 1_000:
            raise ValueError("search bounds are invalid")
        query = _normalized_descriptor(query_descriptor, self.manifest.descriptor)
        if self.size == 0:
            return ()
        similarities = np.asarray(self._vectors @ query, dtype=np.float32)
        order = np.lexsort((np.arange(self.size, dtype=np.int64), -similarities))
        sequence_counts: Counter[tuple[str, str]] = Counter()
        hits: list[ReferenceSearchHit] = []
        for raw_position in order.tolist():
            record = self.records[int(raw_position)]
            sequence_key = (record.source, record.source_sequence_id)
            if sequence_counts[sequence_key] >= max_per_sequence:
                continue
            similarity = min(1.0, max(-1.0, float(similarities[int(raw_position)])))
            distance = 1.0 - similarity
            hits.append(
                ReferenceSearchHit(
                    reference_id=record.reference_id,
                    rank=len(hits) + 1,
                    similarity=similarity,
                    distance=distance,
                    uncertainty_radius_m=record.coordinate_uncertainty_m,
                    reference=record,
                )
            )
            sequence_counts[sequence_key] += 1
            if len(hits) >= top_k:
                break
        return tuple(hits)

    def diagnostics(self) -> ReferenceIndexDiagnostics:
        leakage_status, leakage_audit = _load_leakage_attestation(
            self.directory,
            self.manifest,
        )
        disk_usage = sum(
            (self.directory / filename).stat().st_size
            for filename in (
                MANIFEST_FILENAME,
                VECTORS_FILENAME,
                METADATA_FILENAME,
                EXCLUSIONS_FILENAME,
            )
        )
        attestation_path = self.directory / LEAKAGE_ATTESTATION_FILENAME
        if attestation_path.is_file() and not attestation_path.is_symlink():
            disk_usage += attestation_path.stat().st_size
        duplicate_reasons = {
            "duplicate_sha256",
            "duplicate_source_image",
            "near_duplicate_perceptual_hash",
            "near_duplicate_descriptor",
        }
        duplicate_count = sum(
            count
            for reason, count in self.manifest.deduplication.counts_by_reason.items()
            if reason in duplicate_reasons
        )
        return ReferenceIndexDiagnostics(
            status="ready" if self.size else "empty",
            reason_code="index_verified" if self.size else "index_empty",
            index_version=self.manifest.index_version,
            descriptor_version=self.manifest.descriptor.descriptor_version,
            count=self.size,
            independent_sequences=self.manifest.independent_sequences,
            disk_usage_bytes=disk_usage,
            country_count=len(self.manifest.covered_countries),
            province_count=len(self.manifest.covered_provinces),
            images_per_province=dict(self.manifest.images_per_province),
            source_distribution=dict(self.manifest.source_distribution),
            built_at=self.manifest.built_at,
            duplicate_count=duplicate_count,
            excluded_count=self.manifest.deduplication.excluded_count,
            attributions=tuple(
                dict.fromkeys(
                    f"{record.source}: {record.attribution}"[:500]
                    for record in self.records
                )
            )[:100],
            leakage_status=leakage_status,
            leakage_audit=leakage_audit,
        )


def open_reference_index(directory: Path) -> ReferenceIndexOpenResult:
    try:
        index = MegaLocReferenceIndex.open(directory)
    except ReferenceIndexUnavailableError:
        return ReferenceIndexOpenResult(
            diagnostics=ReferenceIndexDiagnostics(
                status="unavailable",
                reason_code="index_unavailable",
                count=0,
                independent_sequences=0,
                disk_usage_bytes=0,
            ),
            index=None,
        )
    except (ReferenceIndexIntegrityError, OSError, ValueError):
        return ReferenceIndexOpenResult(
            diagnostics=ReferenceIndexDiagnostics(
                status="invalid",
                reason_code="index_verification_failed",
                count=0,
                independent_sequences=0,
                disk_usage_bytes=0,
            ),
            index=None,
        )
    return ReferenceIndexOpenResult(diagnostics=index.diagnostics(), index=index)


def write_reference_leakage_attestation(
    directory: Path,
    attestation: ReferenceIndexLeakageAttestation,
) -> Path:
    """Atomically bind a safe leakage projection to an already verified index."""

    index = MegaLocReferenceIndex.open(directory)
    if not index.available or not _attestation_matches_manifest(
        attestation,
        index.manifest,
    ):
        raise ReferenceIndexBuildError("leakage_attestation_index_mismatch")
    destination = index.directory / LEAKAGE_ATTESTATION_FILENAME
    if destination.is_symlink():
        raise ReferenceIndexBuildError("leakage_attestation_path_unsafe")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        _write_json(temporary, attestation)
        os.replace(temporary, destination)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return destination


def estimate_index_bytes(record_count: int, descriptor_dimension: int) -> int:
    if record_count < 0 or descriptor_dimension <= 0:
        raise ValueError("index estimate inputs are invalid")
    vector_bytes = record_count * descriptor_dimension * np.dtype(np.float32).itemsize
    conservative_metadata_bytes = record_count * 4_096
    return vector_bytes + conservative_metadata_bytes + 64 * 1024


def descriptor_for_record(
    record: ReferenceBuildRecord,
    *,
    input_root: Path,
    spec: MegaLocDescriptorSpec,
) -> FloatVector:
    if record.descriptor_path is None:
        raise ReferenceIndexBuildError("descriptor_source_unavailable")
    path = _safe_input_path(input_root.resolve(), record.descriptor_path, suffix=".npy")
    try:
        descriptor = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ReferenceIndexBuildError("descriptor_invalid") from exc
    return _normalized_descriptor(descriptor, spec)


class CallableDescriptorProvider:
    """Small adapter for tests and batch providers without widening the build API."""

    def __init__(self, function: Callable[[Path], object], spec: MegaLocDescriptorSpec) -> None:
        self._function = function
        self._spec = spec

    def describe(self, image_path: Path) -> FloatVector:
        return _normalized_descriptor(self._function(image_path), self._spec)
