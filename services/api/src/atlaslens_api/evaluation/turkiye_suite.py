from __future__ import annotations

import csv
import hashlib
import io
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from atlaslens_api.evaluation.models import EvaluationRecord
from atlaslens_api.phase6c.reference_index import ReferenceBuildInput, ReferenceBuildRecord

EVALUATION_SUITE_RECEIPT_VERSION: Final = "atlaslens-turkiye-evaluation-suite-receipt-v1"
_CSV_HEADERS: Final = tuple(EvaluationRecord.model_fields)
_SUPPORTED_IMAGE_FORMATS: Final = frozenset({"JPEG", "PNG", "WEBP"})

EvaluationSuiteStatus = Literal["ready", "insufficient", "empty"]
ExclusionReason = Literal[
    "country_not_turkiye",
    "license_not_allowed",
    "missing_acquisition_hashes",
    "missing_locally_resolved_city",
    "not_phase6c_acquisition_source",
    "retrieval_capture_family_overlap",
    "retrieval_dhash_near_duplicate",
    "retrieval_sha256_overlap",
    "retrieval_source_image_overlap",
    "candidate_ahash_near_duplicate",
    "candidate_dhash_near_duplicate",
    "candidate_sha256_duplicate",
    "candidate_source_image_duplicate",
]
_EXCLUSION_REASONS: Final[tuple[ExclusionReason, ...]] = (
    "country_not_turkiye",
    "license_not_allowed",
    "missing_acquisition_hashes",
    "missing_locally_resolved_city",
    "not_phase6c_acquisition_source",
    "retrieval_capture_family_overlap",
    "retrieval_dhash_near_duplicate",
    "retrieval_sha256_overlap",
    "retrieval_source_image_overlap",
    "candidate_ahash_near_duplicate",
    "candidate_dhash_near_duplicate",
    "candidate_sha256_duplicate",
    "candidate_source_image_duplicate",
)


class EvaluationSuiteBuildError(RuntimeError):
    """Safe operator-facing error for evaluation-suite construction."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class EvaluationSuiteBuildPolicy(_StrictModel):
    allowed_licenses: frozenset[str] = Field(min_length=1, max_length=32)
    minimum_records: int = Field(default=100, ge=2, le=10_000)
    development_fraction: float = Field(default=0.7, gt=0, lt=1)
    dhash_hamming_threshold: int = Field(default=4, ge=0, le=16)
    max_candidate_records: int = Field(default=10_000, ge=1, le=100_000)
    max_retrieval_records: int = Field(default=10_000, ge=0, le=100_000)
    max_manifest_bytes: int = Field(default=16 * 1024 * 1024, ge=1)
    max_image_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    max_decoded_pixels: int = Field(default=40_000_000, ge=1)
    split_seed: str = Field(default="phase6c-turkiye-suite-v1", min_length=1, max_length=160)

    @field_validator("allowed_licenses")
    @classmethod
    def validate_license_allowlist(cls, values: frozenset[str]) -> frozenset[str]:
        if any(not value.strip() for value in values):
            raise ValueError("license allowlist contains a blank value")
        return values


class EvaluationSuiteBuildReceipt(_StrictModel):
    schema_version: Literal[
        "atlaslens-turkiye-evaluation-suite-receipt-v1"
    ] = EVALUATION_SUITE_RECEIPT_VERSION
    status: EvaluationSuiteStatus
    candidate_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_input_count: int = Field(ge=0)
    retrieval_input_count: int = Field(ge=0)
    metadata_eligible_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    exclusion_counts: dict[str, int]
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    province_count: int = Field(ge=0, le=81)
    independent_capture_family_count: int = Field(ge=0)
    minimum_records: int = Field(ge=2)
    dhash_hamming_threshold: int = Field(ge=0, le=16)
    coordinate_uncertainty_verified_positive: bool
    ground_truth_scope: Literal["evaluation_only"] = "evaluation_only"
    accuracy_claim_allowed: Literal[False] = False


@dataclass(frozen=True, slots=True)
class EvaluationSuiteBuildResult:
    records: tuple[EvaluationRecord, ...]
    manifest_csv: str
    receipt: EvaluationSuiteBuildReceipt


@dataclass(frozen=True, slots=True)
class _AssetEvidence:
    record: ReferenceBuildRecord
    path: Path
    sha256: str
    dhash: str
    ahash: str


@dataclass(slots=True)
class _HammingNode:
    value: int
    children: dict[int, _HammingNode]


class _HammingTree:
    """Exact bounded-radius lookup for 64-bit image hashes."""

    def __init__(self) -> None:
        self._root: _HammingNode | None = None

    def add(self, value: int) -> None:
        if self._root is None:
            self._root = _HammingNode(value=value, children={})
            return
        node = self._root
        while True:
            distance = (node.value ^ value).bit_count()
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _HammingNode(value=value, children={})
                return
            node = child

    def contains_within(self, value: int, radius: int) -> bool:
        root = self._root
        if root is None:
            return False
        pending = [root]
        while pending:
            node = pending.pop()
            distance = (node.value ^ value).bit_count()
            if distance <= radius:
                return True
            lower = distance - radius
            upper = distance + radius
            pending.extend(
                child
                for edge, child in node.children.items()
                if lower <= edge <= upper
            )
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise EvaluationSuiteBuildError("input_unreadable") from exc
    return digest.hexdigest()


def _load_input(
    path: Path,
    *,
    maximum_records: int,
    maximum_bytes: int,
) -> ReferenceBuildInput:
    try:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size > maximum_bytes:
            raise EvaluationSuiteBuildError("reference_input_invalid")
        value = ReferenceBuildInput.model_validate_json(resolved.read_text(encoding="utf-8"))
    except EvaluationSuiteBuildError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise EvaluationSuiteBuildError("reference_input_invalid") from exc
    if len(value.records) > maximum_records:
        raise EvaluationSuiteBuildError("reference_input_record_limit_exceeded")
    return value


def _resolve_asset(root: Path, record: ReferenceBuildRecord) -> Path:
    relative = Path(record.image_path)
    unresolved = root / relative
    try:
        candidate = unresolved.resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvaluationSuiteBuildError("reference_asset_path_invalid") from exc
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or unresolved.is_symlink()
        or not candidate.is_file()
    ):
        raise EvaluationSuiteBuildError("reference_asset_path_invalid")
    return candidate


def _load_asset(
    root: Path,
    record: ReferenceBuildRecord,
    *,
    maximum_bytes: int,
    maximum_pixels: int,
) -> _AssetEvidence:
    path = _resolve_asset(root, record)
    try:
        if path.stat().st_size < 1 or path.stat().st_size > maximum_bytes:
            raise EvaluationSuiteBuildError("reference_asset_size_invalid")
        digest = _sha256(path)
        with Image.open(path) as source:
            source.load()
            image_format = cast(str | None, source.format)
            if (
                image_format not in _SUPPORTED_IMAGE_FORMATS
                or getattr(source, "n_frames", 1) != 1
                or source.width < 1
                or source.height < 1
                or source.width * source.height > maximum_pixels
            ):
                raise EvaluationSuiteBuildError("reference_asset_image_invalid")
            grayscale = ImageOps.exif_transpose(source).convert("L")
            # Keep dhash64-v1 byte-for-byte compatible with acquisition and the
            # reference-index builder. Those established boundaries intentionally
            # use Pillow's default nearest-neighbour resize for the 9x8 dHash.
            dhash_pixels = list(grayscale.resize((9, 8)).getdata())
            ahash_pixels = list(grayscale.resize((8, 8), Image.Resampling.LANCZOS).getdata())
    except EvaluationSuiteBuildError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise EvaluationSuiteBuildError("reference_asset_image_invalid") from exc

    dhash_value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            dhash_value = (dhash_value << 1) | int(
                dhash_pixels[offset + column + 1] > dhash_pixels[offset + column]
            )
    average = sum(ahash_pixels) / len(ahash_pixels)
    ahash_value = 0
    for pixel in ahash_pixels:
        ahash_value = (ahash_value << 1) | int(pixel >= average)
    dhash = f"{dhash_value:016x}"
    ahash = f"{ahash_value:016x}"
    if record.expected_sha256 is not None and record.expected_sha256 != digest:
        raise EvaluationSuiteBuildError("reference_asset_sha256_mismatch")
    if record.expected_perceptual_hash is not None and record.expected_perceptual_hash != dhash:
        raise EvaluationSuiteBuildError("reference_asset_dhash_mismatch")
    return _AssetEvidence(
        record=record,
        path=path,
        sha256=digest,
        dhash=dhash,
        ahash=ahash,
    )


def _family_key(record: ReferenceBuildRecord) -> tuple[str, str, str]:
    return (record.source, record.source_family, record.source_sequence_id)


def _source_image_key(record: ReferenceBuildRecord) -> tuple[str, str]:
    return (record.source, record.source_image_id)


def _opaque(prefix: str, *values: str) -> str:
    material = "\x00".join(values).encode()
    return f"{prefix}-{hashlib.sha256(material).hexdigest()[:32]}"


def _split_groups(
    values: tuple[_AssetEvidence, ...],
    *,
    development_fraction: float,
    seed: str,
) -> dict[tuple[str, str, str], Literal["calibration", "validation"]]:
    grouped: defaultdict[tuple[str, str, str], list[_AssetEvidence]] = defaultdict(list)
    for item in values:
        grouped[_family_key(item.record)].append(item)
    ordered = sorted(
        grouped,
        key=lambda key: hashlib.sha256(
            f"{seed}\x00{'|'.join(key)}".encode()
        ).hexdigest(),
    )
    if not ordered:
        return {}
    assignments: dict[
        tuple[str, str, str], Literal["calibration", "validation"]
    ] = {ordered[0]: "calibration"}
    development_count = len(grouped[ordered[0]])
    validation_count = 0
    if len(ordered) > 1:
        assignments[ordered[1]] = "validation"
        validation_count = len(grouped[ordered[1]])
    for key in ordered[2:]:
        group_size = len(grouped[key])
        new_total = development_count + validation_count + group_size
        development_error = abs(
            (development_count + group_size) / new_total - development_fraction
        )
        validation_error = abs(development_count / new_total - development_fraction)
        if development_error <= validation_error:
            assignments[key] = "calibration"
            development_count += group_size
        else:
            assignments[key] = "validation"
            validation_count += group_size
    return assignments


def _csv_text(records: tuple[EvaluationRecord, ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(record.model_dump(mode="json") for record in records)
    return output.getvalue()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except OSError as exc:
        raise EvaluationSuiteBuildError("evaluation_output_write_failed") from exc


class TurkiyeEvaluationSuiteBuilder:
    """Build a non-holdout evaluation CSV from a separate acquisition input."""

    def __init__(self, policy: EvaluationSuiteBuildPolicy) -> None:
        self._policy = policy

    def build(
        self,
        *,
        candidate_reference_input: Path,
        candidate_root: Path,
        retrieval_reference_input: Path,
        retrieval_root: Path,
    ) -> EvaluationSuiteBuildResult:
        candidate_input_path = candidate_reference_input.expanduser().resolve()
        retrieval_input_path = retrieval_reference_input.expanduser().resolve()
        if candidate_input_path == retrieval_input_path:
            raise EvaluationSuiteBuildError("evaluation_and_retrieval_inputs_must_be_separate")
        try:
            candidate_asset_root = candidate_root.expanduser().resolve(strict=True)
            retrieval_asset_root = retrieval_root.expanduser().resolve(strict=True)
        except OSError as exc:
            raise EvaluationSuiteBuildError("reference_asset_root_unavailable") from exc
        if not candidate_asset_root.is_dir() or not retrieval_asset_root.is_dir():
            raise EvaluationSuiteBuildError("reference_asset_root_unavailable")

        candidates = _load_input(
            candidate_input_path,
            maximum_records=self._policy.max_candidate_records,
            maximum_bytes=self._policy.max_manifest_bytes,
        )
        retrieval = _load_input(
            retrieval_input_path,
            maximum_records=self._policy.max_retrieval_records,
            maximum_bytes=self._policy.max_manifest_bytes,
        )
        retrieval_evidence = tuple(
            _load_asset(
                retrieval_asset_root,
                record,
                maximum_bytes=self._policy.max_image_bytes,
                maximum_pixels=self._policy.max_decoded_pixels,
            )
            for record in retrieval.records
        )
        retrieval_sha256 = {item.sha256 for item in retrieval_evidence}
        retrieval_dhash = _HammingTree()
        for item in retrieval_evidence:
            retrieval_dhash.add(int(item.dhash, 16))
        retrieval_source_images = {
            _source_image_key(item.record) for item in retrieval_evidence
        }
        retrieval_source_urls = {item.record.source_url for item in retrieval_evidence}
        retrieval_source_sequences: set[tuple[str, str]] = {
            (str(item.record.source), item.record.source_sequence_id)
            for item in retrieval_evidence
        }
        retrieval_capture_families = {
            (item.record.source_family, item.record.source_sequence_id)
            for item in retrieval_evidence
        }

        exclusions: Counter[ExclusionReason] = Counter()
        metadata_eligible_count = 0
        accepted: list[_AssetEvidence] = []
        candidate_sha256: set[str] = set()
        candidate_dhash = _HammingTree()
        candidate_ahash = _HammingTree()
        candidate_source_images: set[tuple[str, str]] = set()
        for record in sorted(candidates.records, key=lambda item: item.reference_id):
            reason = self._metadata_exclusion(record)
            if reason is not None:
                exclusions[reason] += 1
                continue
            metadata_eligible_count += 1
            evidence = _load_asset(
                candidate_asset_root,
                record,
                maximum_bytes=self._policy.max_image_bytes,
                maximum_pixels=self._policy.max_decoded_pixels,
            )
            cross_index_reason = self._retrieval_exclusion(
                evidence,
                retrieval_sha256=retrieval_sha256,
                retrieval_dhash=retrieval_dhash,
                retrieval_source_images=retrieval_source_images,
                retrieval_source_urls=retrieval_source_urls,
                retrieval_source_sequences=retrieval_source_sequences,
                retrieval_capture_families=retrieval_capture_families,
            )
            if cross_index_reason is not None:
                exclusions[cross_index_reason] += 1
                continue
            internal_reason = self._candidate_duplicate_exclusion(
                evidence,
                sha256_values=candidate_sha256,
                dhash_values=candidate_dhash,
                ahash_values=candidate_ahash,
                source_images=candidate_source_images,
            )
            if internal_reason is not None:
                exclusions[internal_reason] += 1
                continue
            accepted.append(evidence)
            candidate_sha256.add(evidence.sha256)
            candidate_dhash.add(int(evidence.dhash, 16))
            candidate_ahash.add(int(evidence.ahash, 16))
            candidate_source_images.add(_source_image_key(record))

        accepted_values = tuple(accepted)
        assignments = _split_groups(
            accepted_values,
            development_fraction=self._policy.development_fraction,
            seed=self._policy.split_seed,
        )
        rows = tuple(
            sorted(
                (
                    self._evaluation_record(
                        item,
                        assignments[_family_key(item.record)],
                        candidate_asset_root,
                    )
                    for item in accepted_values
                ),
                key=lambda item: (
                    0 if item.split == "calibration" else 1,
                    item.image_asset_key,
                ),
            )
        )
        manifest_csv = _csv_text(rows)
        manifest_digest = hashlib.sha256(manifest_csv.encode()).hexdigest()
        development_count = sum(record.split == "calibration" for record in rows)
        validation_count = sum(record.split == "validation" for record in rows)
        status: EvaluationSuiteStatus
        if not rows:
            status = "empty"
        elif (
            len(rows) < self._policy.minimum_records
            or development_count == 0
            or validation_count == 0
        ):
            status = "insufficient"
        else:
            status = "ready"
        receipt = EvaluationSuiteBuildReceipt(
            status=status,
            candidate_input_sha256=_sha256(candidate_input_path),
            retrieval_input_sha256=_sha256(retrieval_input_path),
            output_manifest_sha256=manifest_digest,
            candidate_input_count=len(candidates.records),
            retrieval_input_count=len(retrieval.records),
            metadata_eligible_count=metadata_eligible_count,
            accepted_count=len(rows),
            excluded_count=sum(exclusions.values()),
            exclusion_counts={reason: exclusions[reason] for reason in _EXCLUSION_REASONS},
            split_counts={
                "development": development_count,
                "validation": validation_count,
            },
            source_counts=dict(Counter(item.record.source for item in accepted_values)),
            province_count=len({item.record.province for item in accepted_values}),
            independent_capture_family_count=len(
                {_family_key(item.record) for item in accepted_values}
            ),
            minimum_records=self._policy.minimum_records,
            dhash_hamming_threshold=self._policy.dhash_hamming_threshold,
            coordinate_uncertainty_verified_positive=bool(accepted_values)
            and all(item.record.coordinate_uncertainty_m > 0 for item in accepted_values),
        )
        return EvaluationSuiteBuildResult(
            records=rows,
            manifest_csv=manifest_csv,
            receipt=receipt,
        )

    def write(
        self,
        result: EvaluationSuiteBuildResult,
        *,
        manifest_path: Path,
        receipt_path: Path,
        protected_inputs: tuple[Path, ...] = (),
    ) -> None:
        manifest_destination = manifest_path.expanduser().resolve()
        receipt_destination = receipt_path.expanduser().resolve()
        protected = {path.expanduser().resolve() for path in protected_inputs}
        if (
            manifest_destination == receipt_destination
            or manifest_destination in protected
            or receipt_destination in protected
        ):
            raise EvaluationSuiteBuildError("evaluation_output_path_conflict")
        _atomic_text(manifest_destination, result.manifest_csv)
        _atomic_text(
            receipt_destination,
            result.receipt.model_dump_json(indent=2) + "\n",
        )

    def _metadata_exclusion(self, record: ReferenceBuildRecord) -> ExclusionReason | None:
        if record.source not in {"mapillary", "kartaview"}:
            return "not_phase6c_acquisition_source"
        if record.country != "TR":
            return "country_not_turkiye"
        if record.city is None or not record.city.strip():
            return "missing_locally_resolved_city"
        if record.license not in self._policy.allowed_licenses:
            return "license_not_allowed"
        if record.expected_sha256 is None or record.expected_perceptual_hash is None:
            return "missing_acquisition_hashes"
        return None

    def _retrieval_exclusion(
        self,
        item: _AssetEvidence,
        *,
        retrieval_sha256: set[str],
        retrieval_dhash: _HammingTree,
        retrieval_source_images: set[tuple[str, str]],
        retrieval_source_urls: set[str],
        retrieval_source_sequences: set[tuple[str, str]],
        retrieval_capture_families: set[tuple[str, str]],
    ) -> ExclusionReason | None:
        record = item.record
        if item.sha256 in retrieval_sha256:
            return "retrieval_sha256_overlap"
        if retrieval_dhash.contains_within(
            int(item.dhash, 16), self._policy.dhash_hamming_threshold
        ):
            return "retrieval_dhash_near_duplicate"
        if (
            _source_image_key(record) in retrieval_source_images
            or record.source_url in retrieval_source_urls
        ):
            return "retrieval_source_image_overlap"
        if (
            (record.source, record.source_sequence_id) in retrieval_source_sequences
            or (record.source_family, record.source_sequence_id)
            in retrieval_capture_families
        ):
            return "retrieval_capture_family_overlap"
        return None

    def _candidate_duplicate_exclusion(
        self,
        item: _AssetEvidence,
        *,
        sha256_values: set[str],
        dhash_values: _HammingTree,
        ahash_values: _HammingTree,
        source_images: set[tuple[str, str]],
    ) -> ExclusionReason | None:
        if item.sha256 in sha256_values:
            return "candidate_sha256_duplicate"
        if dhash_values.contains_within(
            int(item.dhash, 16), self._policy.dhash_hamming_threshold
        ):
            return "candidate_dhash_near_duplicate"
        if ahash_values.contains_within(int(item.ahash, 16), 4):
            return "candidate_ahash_near_duplicate"
        if _source_image_key(item.record) in source_images:
            return "candidate_source_image_duplicate"
        return None

    @staticmethod
    def _evaluation_record(
        item: _AssetEvidence,
        split: Literal["calibration", "validation"],
        root: Path,
    ) -> EvaluationRecord:
        record = item.record
        family = _family_key(record)
        return EvaluationRecord(
            image_asset_key=_opaque("phase6c-eval", record.source, record.source_image_id),
            local_reference=item.path.relative_to(root).as_posix(),
            true_latitude=record.latitude,
            true_longitude=record.longitude,
            country_code="TR",
            region=record.province,
            city_or_area=record.city,
            continent="Europe/Asia",
            source=record.source,
            source_record_id=record.source_image_id,
            license=record.license,
            attribution=record.attribution,
            split=split,
            scene_category="street_imagery",
            geographic_cell=_opaque("tr-cell", record.province),
            capture_family_id=_opaque("capture-family", *family),
            content_sha256=item.sha256,
            perceptual_hash=item.ahash,
        )
