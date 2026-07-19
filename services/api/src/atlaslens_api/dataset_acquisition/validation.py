from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from atlaslens_api.dataset_acquisition.commons import DatasetAcquisitionError, average_hash
from atlaslens_api.dataset_acquisition.models import DatasetValidationReport
from atlaslens_api.evaluation.manifest import EvaluationManifestLoader
from atlaslens_api.retrieval.manifest import EXTENDED_MANIFEST_COLUMNS, validate_manifest


@dataclass(frozen=True, slots=True)
class EvaluationExclusions:
    fingerprint: str
    content_hashes: frozenset[str]
    perceptual_hashes: tuple[int, ...]
    capture_families: frozenset[str]

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        asset_root: Path,
        *,
        expected_fingerprint: str,
        expected_count: int,
    ) -> EvaluationExclusions:
        if (
            len(expected_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in expected_fingerprint)
            or expected_count < 1
        ):
            raise DatasetAcquisitionError("evaluation_expectation_invalid")
        try:
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise DatasetAcquisitionError("evaluation_manifest_unavailable") from exc
        licenses = frozenset(row.get("license", "") for row in rows if row.get("license"))
        if not licenses:
            raise DatasetAcquisitionError("evaluation_manifest_invalid")
        try:
            validated = EvaluationManifestLoader(allowed_licenses=licenses).load(
                manifest_path, asset_root
            )
        except ValueError as exc:
            raise DatasetAcquisitionError("evaluation_manifest_invalid") from exc
        if (
            validated.report.fingerprint != expected_fingerprint
            or len(validated.assets) != expected_count
        ):
            raise DatasetAcquisitionError("evaluation_fingerprint_mismatch")
        return cls(
            fingerprint=validated.report.fingerprint,
            content_hashes=frozenset(
                asset.record.content_sha256 for asset in validated.assets
            ),
            perceptual_hashes=tuple(
                int(asset.record.perceptual_hash or average_hash(asset.path.read_bytes()), 16)
                for asset in validated.assets
            ),
            capture_families=frozenset(
                asset.record.capture_family_id for asset in validated.assets
            ),
        )


class LicensedDatasetValidator:
    def __init__(
        self,
        *,
        allowed_licenses: frozenset[str],
        minimum_count: int = 10_000,
        required_continents: int = 6,
        minimum_countries: int = 30,
    ):
        if (
            not allowed_licenses
            or minimum_count < 1
            or not 1 <= required_continents <= 6
            or minimum_countries < 1
        ):
            raise ValueError("dataset validation requires licenses and a positive target")
        self.allowed_licenses = allowed_licenses
        self.minimum_count = minimum_count
        self.required_continents = required_continents
        self.minimum_countries = minimum_countries

    def validate(
        self,
        manifest_path: Path,
        asset_root: Path,
        exclusions: EvaluationExclusions,
    ) -> DatasetValidationReport:
        try:
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = tuple(next(reader))
        except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
            raise DatasetAcquisitionError("dataset_manifest_unreadable") from exc
        if header != EXTENDED_MANIFEST_COLUMNS:
            raise DatasetAcquisitionError("extended_manifest_required")
        records = validate_manifest(manifest_path, asset_root)
        if len(records) < self.minimum_count:
            raise DatasetAcquisitionError("reference_target_not_met")
        hashes: set[str] = set()
        perceptual: list[int] = []
        families: set[str] = set()
        canonical: list[dict[str, object]] = []
        for record in records:
            if record.license not in self.allowed_licenses:
                raise DatasetAcquisitionError("license_not_allowlisted")
            if (
                record.asset_key is None
                or record.source_record_id is None
                or record.source_url is None
                or record.license_url is None
                or record.attribution == "attribution unavailable"
                or record.capture_family_id is None
                or record.coordinate_kind == "unknown"
                or record.perceptual_hash is None
                or record.continent is None
                or record.geographic_cell is None
                or record.country is None
            ):
                raise DatasetAcquisitionError("incomplete_reference_provenance")
            computed = average_hash(record.image_path.read_bytes())
            if computed != record.perceptual_hash:
                raise DatasetAcquisitionError("perceptual_hash_mismatch")
            value = int(computed, 16)
            if record.content_hash in hashes:
                raise DatasetAcquisitionError("duplicate_reference_content")
            if any((value ^ previous).bit_count() <= 4 for previous in perceptual):
                raise DatasetAcquisitionError("near_duplicate_reference_content")
            if record.content_hash in exclusions.content_hashes:
                raise DatasetAcquisitionError("evaluation_content_overlap")
            if any(
                (value ^ excluded).bit_count() <= 4
                for excluded in exclusions.perceptual_hashes
            ):
                raise DatasetAcquisitionError("evaluation_near_duplicate_overlap")
            if record.capture_family_id in exclusions.capture_families:
                raise DatasetAcquisitionError("evaluation_capture_family_overlap")
            if record.capture_family_id in families:
                raise DatasetAcquisitionError("duplicate_capture_family")
            hashes.add(record.content_hash)
            perceptual.append(value)
            families.add(record.capture_family_id)
            canonical.append(
                {
                    "asset_key": record.asset_key,
                    "content_sha256": record.content_hash,
                    "perceptual_hash": record.perceptual_hash,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "source": record.source,
                    "source_record_id": record.source_record_id,
                    "license": record.license,
                    "attribution": record.attribution,
                    "capture_family_id": record.capture_family_id,
                    "coordinate_kind": record.coordinate_kind,
                    "continent": record.continent,
                    "country": record.country,
                    "geographic_cell": record.geographic_cell,
                }
            )
        continent_counts = dict(Counter(record.continent for record in records))
        country_counts = dict(Counter(record.country for record in records))
        if (
            len(continent_counts) < self.required_continents
            or len(country_counts) < self.minimum_countries
        ):
            raise DatasetAcquisitionError("geographic_distribution_unmet")
        fingerprint = hashlib.sha256(
            json.dumps(
                sorted(canonical, key=lambda item: str(item["asset_key"])),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return DatasetValidationReport(
            fingerprint=fingerprint,
            image_count=len(records),
            source_counts=dict(Counter(record.source for record in records)),
            license_counts=dict(Counter(record.license for record in records)),
            continent_counts=continent_counts,
            country_counts=country_counts,
            geographic_cell_counts=dict(Counter(record.geographic_cell for record in records)),
            evaluation_fingerprint=exclusions.fingerprint,
        )
