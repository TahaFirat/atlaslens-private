"""Deterministic, leakage-aware selection for the bounded Mapillary demo."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from atlaslens_api.corpus_index.artifacts import sha256_json
from atlaslens_api.corpus_index.benchmark import haversine_distance_m
from atlaslens_api.corpus_pipeline.hashing import (
    inspect_image,
    perceptual_hamming_distance,
)
from atlaslens_api.mapillary_demo.models import (
    AcquiredImage,
    AcquisitionManifest,
    CoverageAudit,
    CoverageSummary,
)

REFERENCE_TARGET = 1_500
HOLDOUT_TARGET = 100
HARD_IMAGE_CAP = 2_000
DEFAULT_SAMPLE_DISTANCE_M = 50.0
POSITIVE_THRESHOLDS_M = (25.0, 100.0, 500.0, 1_000.0)

type SplitName = Literal["reference", "holdout"]


@dataclass(frozen=True, slots=True)
class PilotCityDecision:
    aoi_id: str
    aoi_version: str
    display_name: str
    city: str
    province: str
    reason: str
    kayseri_eligible: bool


@dataclass(frozen=True, slots=True)
class SelectionLimits:
    reference_target: int = REFERENCE_TARGET
    holdout_target: int = HOLDOUT_TARGET
    hard_image_cap: int = HARD_IMAGE_CAP
    minimum_sequence_spacing_m: float = DEFAULT_SAMPLE_DISTANCE_M
    near_duplicate_hamming_threshold: int = 4

    def __post_init__(self) -> None:
        if self.reference_target <= 0 or self.reference_target > REFERENCE_TARGET:
            raise ValueError("reference target must be in [1, 1500]")
        if self.holdout_target <= 0 or self.holdout_target > HOLDOUT_TARGET:
            raise ValueError("holdout target must be in [1, 100]")
        if (
            self.hard_image_cap <= 0
            or self.hard_image_cap > HARD_IMAGE_CAP
            or self.reference_target + self.holdout_target > self.hard_image_cap
        ):
            raise ValueError("selection exceeds the 2000-image hard cap")
        if not math.isfinite(self.minimum_sequence_spacing_m) or not (
            50.0 <= self.minimum_sequence_spacing_m <= 100.0
        ):
            raise ValueError("sequence spacing must be between 50 and 100 metres")
        if not 0 <= self.near_duplicate_hamming_threshold <= 64:
            raise ValueError("near-duplicate threshold must be in [0, 64]")


@dataclass(frozen=True, slots=True)
class SelectedMapillaryAsset:
    asset_id: str
    mapillary_image_id: str
    relative_path: str
    raw_sha256: str
    normalized_sha256: str
    perceptual_hash: str
    width_px: int
    height_px: int
    latitude: float
    longitude: float
    captured_at: datetime
    acquired_at: datetime
    compass_angle: float | None
    sequence_id: str | None
    creator_id: str | None
    source_page_url: str
    attribution_text: str
    license_identifier: str
    license_url: str
    city: str
    province: str
    province_code: str
    spatial_cell: str
    split: SplitName

    @property
    def lineage_sequence(self) -> str:
        return self.sequence_id or f"image:{self.mapillary_image_id}"

    @property
    def capture_year(self) -> int:
        return self.captured_at.year

    def lock_record(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "mapillary_image_id": self.mapillary_image_id,
            "raw_sha256": self.raw_sha256,
            "normalized_sha256": self.normalized_sha256,
            "perceptual_hash": self.perceptual_hash,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "captured_at": self.captured_at.isoformat(),
            "sequence_id": self.sequence_id,
            "creator_id": self.creator_id,
            "spatial_cell": self.spatial_cell,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class LockedMapillarySplit:
    decision: PilotCityDecision
    acquisition_manifest_sha256: str
    source_policy_sha256: str
    references: tuple[SelectedMapillaryAsset, ...]
    holdout: tuple[SelectedMapillaryAsset, ...]
    reference_target: int
    holdout_target: int
    reference_shortfall: int
    holdout_shortfall: int
    positive_reference_ids: dict[str, dict[str, tuple[str, ...]]]
    suppressed_stationary_or_adjacent: int
    suppressed_exact_or_near_duplicate: int
    lock_sha256: str

    @property
    def all_assets(self) -> tuple[SelectedMapillaryAsset, ...]:
        return self.references + self.holdout

    def lock_basis(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-mapillary-split-lock-v1",
            "aoi_id": self.decision.aoi_id,
            "aoi_version": self.decision.aoi_version,
            "display_name": self.decision.display_name,
            "city": self.decision.city,
            "province": self.decision.province,
            "acquisition_manifest_sha256": self.acquisition_manifest_sha256,
            "source_policy_sha256": self.source_policy_sha256,
            "references": [asset.lock_record() for asset in self.references],
            "holdout": [asset.lock_record() for asset in self.holdout],
            "reference_target": self.reference_target,
            "holdout_target": self.holdout_target,
            "reference_shortfall": self.reference_shortfall,
            "holdout_shortfall": self.holdout_shortfall,
            "positive_reference_ids": {
                query_id: {
                    threshold: list(asset_ids)
                    for threshold, asset_ids in sorted(thresholds.items())
                }
                for query_id, thresholds in sorted(self.positive_reference_ids.items())
            },
        }

    def verify(self) -> None:
        if sha256_json(self.lock_basis()) != self.lock_sha256:
            raise ValueError("Mapillary split lock checksum mismatch")
        _verify_cross_split(self.references, self.holdout)


@dataclass(frozen=True, slots=True)
class _Candidate:
    image: AcquiredImage
    perceptual_hash: str
    spatial_cell: str

    @property
    def lineage_sequence(self) -> str:
        return self.image.sequence_id or f"image:{self.image.mapillary_image_id}"


def choose_pilot_city(audit: CoverageAudit) -> PilotCityDecision:
    """Apply the prompt's deterministic Kayseri-first selection rule."""

    urban = {
        summary.aoi_id: summary
        for summary in audit.regions
        if summary.region_kind == "urban"
        and summary.aoi_id in {"kayseri-urban-v1", "ankara-urban-v1", "sivas-urban-v1"}
    }
    if set(urban) != {"kayseri-urban-v1", "ankara-urban-v1", "sivas-urban-v1"}:
        raise ValueError("coverage audit lacks one or more required urban regions")
    kayseri = urban["kayseri-urban-v1"]
    kayseri_eligible = (
        kayseri.image_count >= 1_000
        and kayseri.eligible_for_selection
        and kayseri.spatial_cell_count >= 10
        and kayseri.sequence_count >= 2
    )
    if kayseri_eligible:
        selected = kayseri
        reason = "Kayseri has at least 1000 eligible spatially distributed images"
    else:
        eligible = [
            summary
            for summary in urban.values()
            if summary.eligible_for_selection
            and summary.spatial_cell_count >= 10
            and summary.sequence_count >= 2
        ]
        if not eligible:
            raise ValueError("coverage audit has no eligible urban pilot region")
        selected = max(
            eligible,
            key=lambda summary: (
                summary.image_count,
                summary.spatial_cell_count,
                summary.sequence_count,
                summary.contributor_count,
                _stable_aoi_tiebreak(summary),
            ),
        )
        reason = (
            "Kayseri did not meet the 1000-image distributed gate; selected the "
            "best-covered eligible urban region"
        )
    if audit.selected_aoi_id is not None and audit.selected_aoi_id != selected.aoi_id:
        raise ValueError("coverage audit selection disagrees with deterministic policy")
    if selected.city is None:
        raise ValueError("selected urban coverage summary has no truthful city name")
    return PilotCityDecision(
        aoi_id=selected.aoi_id,
        aoi_version=selected.aoi_version,
        display_name=selected.display_name,
        city=selected.city,
        province=selected.province,
        reason=reason,
        kayseri_eligible=kayseri_eligible,
    )


def select_locked_split(
    manifest: AcquisitionManifest,
    audit: CoverageAudit,
    *,
    acquisition_root: Path,
    province_code: str,
    limits: SelectionLimits | None = None,
) -> LockedMapillarySplit:
    """Inspect local assets, suppress leakage, and lock a geographic split."""

    selected_limits = limits or SelectionLimits()
    decision = choose_pilot_city(audit)
    if manifest.aoi_id != decision.aoi_id or manifest.aoi_version != decision.aoi_version:
        raise ValueError("acquisition manifest does not match the selected pilot city")
    if not province_code.startswith("TR-") or len(province_code) != 5:
        raise ValueError("province_code must be an ISO 3166-2 Türkiye code")
    if len(manifest.assets) > selected_limits.hard_image_cap:
        raise ValueError("acquisition manifest exceeds the selection hard cap")
    candidates = _inspect_candidates(
        manifest.assets,
        acquisition_root=acquisition_root,
        spacing_m=selected_limits.minimum_sequence_spacing_m,
        near_duplicate_hamming_threshold=selected_limits.near_duplicate_hamming_threshold,
    )
    pool, stationary_count, duplicate_count = candidates
    if len(pool) < 2:
        raise ValueError("insufficient non-duplicate assets for a reference/holdout split")

    reference_candidates, holdout_candidates = _assign_candidates(
        pool,
        reference_target=selected_limits.reference_target,
        holdout_target=selected_limits.holdout_target,
    )
    references = tuple(
        _selected(candidate, decision, province_code, split="reference")
        for candidate in sorted(
            reference_candidates,
            key=lambda item: item.image.mapillary_image_id,
        )
    )
    holdout = tuple(
        _selected(candidate, decision, province_code, split="holdout")
        for candidate in sorted(holdout_candidates, key=lambda item: item.image.mapillary_image_id)
    )
    _verify_cross_split(references, holdout)
    positives = _positive_reference_map(references, holdout)
    acquisition_hash = sha256_json(manifest.model_dump(mode="json"))
    provisional = LockedMapillarySplit(
        decision=decision,
        acquisition_manifest_sha256=acquisition_hash,
        source_policy_sha256=manifest.source_policy_receipt_sha256,
        references=references,
        holdout=holdout,
        reference_target=selected_limits.reference_target,
        holdout_target=selected_limits.holdout_target,
        reference_shortfall=max(0, selected_limits.reference_target - len(references)),
        holdout_shortfall=max(0, selected_limits.holdout_target - len(holdout)),
        positive_reference_ids=positives,
        suppressed_stationary_or_adjacent=stationary_count,
        suppressed_exact_or_near_duplicate=duplicate_count,
        lock_sha256="0" * 64,
    )
    result = LockedMapillarySplit(
        decision=provisional.decision,
        acquisition_manifest_sha256=provisional.acquisition_manifest_sha256,
        source_policy_sha256=provisional.source_policy_sha256,
        references=provisional.references,
        holdout=provisional.holdout,
        reference_target=provisional.reference_target,
        holdout_target=provisional.holdout_target,
        reference_shortfall=provisional.reference_shortfall,
        holdout_shortfall=provisional.holdout_shortfall,
        positive_reference_ids=provisional.positive_reference_ids,
        suppressed_stationary_or_adjacent=stationary_count,
        suppressed_exact_or_near_duplicate=duplicate_count,
        lock_sha256=sha256_json(provisional.lock_basis()),
    )
    result.verify()
    return result


def _stable_aoi_tiebreak(summary: CoverageSummary) -> tuple[int, ...]:
    # max() must still choose the lexicographically smallest AOI ID on a full tie.
    return tuple(-ord(character) for character in summary.aoi_id)


def _inspect_candidates(
    images: Sequence[AcquiredImage],
    *,
    acquisition_root: Path,
    spacing_m: float,
    near_duplicate_hamming_threshold: int,
) -> tuple[tuple[_Candidate, ...], int, int]:
    root = acquisition_root.resolve()
    grouped: defaultdict[str, list[AcquiredImage]] = defaultdict(list)
    for image in images:
        if image.reconciliation_state != "active":
            continue
        grouped[image.sequence_id or f"image:{image.mapillary_image_id}"].append(image)

    sampled: list[AcquiredImage] = []
    stationary = 0
    for sequence_id in sorted(grouped):
        previous: AcquiredImage | None = None
        for image in sorted(
            grouped[sequence_id],
            key=lambda item: (item.captured_at, item.mapillary_image_id),
        ):
            if previous is not None and _distance_images(previous, image) < spacing_m:
                stationary += 1
                continue
            sampled.append(image)
            previous = image

    accepted: list[_Candidate] = []
    raw_hashes: set[str] = set()
    normalized_hashes: set[str] = set()
    duplicate = 0
    for image in sorted(sampled, key=lambda item: item.mapillary_image_id):
        path = _contained_regular_file(root, image.relative_path)
        identity = inspect_image(path, max_bytes=64 * 1024 * 1024, max_pixels=100_000_000)
        if (
            identity.sha256 != image.normalized_sha256
            or identity.width_px != image.width_px
            or identity.height_px != image.height_px
            or path.stat().st_size != image.normalized_byte_size
        ):
            raise ValueError("acquired image identity does not match its manifest")
        if image.raw_sha256 in raw_hashes or image.normalized_sha256 in normalized_hashes:
            duplicate += 1
            continue
        if any(
            perceptual_hamming_distance(identity.perceptual_hash, item.perceptual_hash)
            <= near_duplicate_hamming_threshold
            for item in accepted
        ):
            duplicate += 1
            continue
        candidate = _Candidate(
            image=image,
            perceptual_hash=identity.perceptual_hash,
            spatial_cell=_spatial_cell(*reversed(image.computed_geometry.coordinates)),
        )
        accepted.append(candidate)
        raw_hashes.add(image.raw_sha256)
        normalized_hashes.add(image.normalized_sha256)
    return tuple(accepted), stationary, duplicate


def _assign_candidates(
    candidates: Sequence[_Candidate],
    *,
    reference_target: int,
    holdout_target: int,
) -> tuple[tuple[_Candidate, ...], tuple[_Candidate, ...]]:
    ordered = _spatial_round_robin(candidates)
    holdout: list[_Candidate] = []
    reserved_references: dict[str, _Candidate] = {}
    holdout_sequences: set[str] = set()
    reference_sequences: set[str] = set()

    for query in ordered:
        if len(holdout) >= holdout_target:
            break
        query_sequence = query.lineage_sequence
        if query_sequence in holdout_sequences or query_sequence in reference_sequences:
            continue
        neighbor = _preferred_positive_neighbor(
            query,
            candidates,
            forbidden_sequences=holdout_sequences | {query_sequence},
        )
        if neighbor is None or neighbor.lineage_sequence in holdout_sequences:
            continue
        holdout.append(query)
        holdout_sequences.add(query_sequence)
        reference_sequences.add(neighbor.lineage_sequence)
        reserved_references.setdefault(neighbor.image.mapillary_image_id, neighbor)

    if not holdout:
        raise ValueError("no leakage-safe holdout with a geographic positive was found")
    references = list(reserved_references.values())
    reference_ids = set(reserved_references)
    for candidate in ordered:
        if len(references) >= reference_target:
            break
        if (
            candidate.image.mapillary_image_id in reference_ids
            or candidate.lineage_sequence in holdout_sequences
        ):
            continue
        references.append(candidate)
        reference_ids.add(candidate.image.mapillary_image_id)
    if not references:
        raise ValueError("no leakage-safe reference asset was found")
    return tuple(references), tuple(holdout)


def _preferred_positive_neighbor(
    query: _Candidate,
    candidates: Sequence[_Candidate],
    *,
    forbidden_sequences: set[str],
) -> _Candidate | None:
    choices: list[tuple[tuple[object, ...], _Candidate]] = []
    for candidate in candidates:
        if (
            candidate.image.mapillary_image_id == query.image.mapillary_image_id
            or candidate.lineage_sequence in forbidden_sequences
        ):
            continue
        distance = _distance_images(query.image, candidate.image)
        if distance <= 0.01 or distance > 1_000.0:
            continue
        bucket = next(
            index for index, threshold in enumerate(POSITIVE_THRESHOLDS_M) if distance <= threshold
        )
        same_contributor = (
            query.image.creator_id is not None
            and query.image.creator_id == candidate.image.creator_id
        )
        same_year = query.image.captured_at.year == candidate.image.captured_at.year
        choices.append(
            (
                (
                    bucket,
                    same_contributor,
                    same_year,
                    abs(distance - 75.0),
                    candidate.image.mapillary_image_id,
                ),
                candidate,
            )
        )
    return min(choices, key=lambda item: item[0])[1] if choices else None


def _spatial_round_robin(candidates: Sequence[_Candidate]) -> tuple[_Candidate, ...]:
    cells: defaultdict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        cells[candidate.spatial_cell].append(candidate)
    for values in cells.values():
        values.sort(
            key=lambda item: (
                item.image.captured_at.year,
                item.lineage_sequence,
                item.image.mapillary_image_id,
            )
        )
    result: list[_Candidate] = []
    depth = 0
    while True:
        added = False
        for cell in sorted(cells):
            values = cells[cell]
            if depth < len(values):
                result.append(values[depth])
                added = True
        if not added:
            return tuple(result)
        depth += 1


def _selected(
    candidate: _Candidate,
    decision: PilotCityDecision,
    province_code: str,
    *,
    split: SplitName,
) -> SelectedMapillaryAsset:
    image = candidate.image
    longitude, latitude = image.computed_geometry.coordinates
    return SelectedMapillaryAsset(
        asset_id=f"mapillary-{image.mapillary_image_id}",
        mapillary_image_id=image.mapillary_image_id,
        relative_path=image.relative_path,
        raw_sha256=image.raw_sha256,
        normalized_sha256=image.normalized_sha256,
        perceptual_hash=candidate.perceptual_hash,
        width_px=image.width_px,
        height_px=image.height_px,
        latitude=latitude,
        longitude=longitude,
        captured_at=image.captured_at,
        acquired_at=image.acquired_at,
        compass_angle=image.compass_angle,
        sequence_id=image.sequence_id,
        creator_id=image.creator_id,
        source_page_url=image.source_page_url,
        attribution_text=image.attribution_text,
        license_identifier=image.license_identifier,
        license_url=image.license_url,
        city=decision.city,
        province=decision.province,
        province_code=province_code,
        spatial_cell=candidate.spatial_cell,
        split=split,
    )


def _positive_reference_map(
    references: Sequence[SelectedMapillaryAsset],
    holdout: Sequence[SelectedMapillaryAsset],
) -> dict[str, dict[str, tuple[str, ...]]]:
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for query in holdout:
        thresholds: dict[str, tuple[str, ...]] = {}
        for threshold in POSITIVE_THRESHOLDS_M:
            ids = tuple(
                sorted(
                    reference.asset_id
                    for reference in references
                    if haversine_distance_m(
                        query.latitude,
                        query.longitude,
                        reference.latitude,
                        reference.longitude,
                    )
                    <= threshold
                )
            )
            thresholds[_threshold_key(threshold)] = ids
        if not thresholds["1000"]:
            raise ValueError("holdout query has no positive reference within 1 kilometre")
        result[query.asset_id] = thresholds
    return result


def _verify_cross_split(
    references: Sequence[SelectedMapillaryAsset],
    holdout: Sequence[SelectedMapillaryAsset],
) -> None:
    reference_ids = {asset.mapillary_image_id for asset in references}
    holdout_ids = {asset.mapillary_image_id for asset in holdout}
    if reference_ids & holdout_ids:
        raise ValueError("Mapillary image ID crosses reference and holdout")
    for attribute, message in (
        ("raw_sha256", "raw hash crosses reference and holdout"),
        ("normalized_sha256", "normalized hash crosses reference and holdout"),
        ("lineage_sequence", "sequence crosses reference and holdout"),
    ):
        left = {getattr(asset, attribute) for asset in references}
        right = {getattr(asset, attribute) for asset in holdout}
        if left & right:
            raise ValueError(message)
    if any(
        perceptual_hamming_distance(reference.perceptual_hash, query.perceptual_hash) <= 4
        for reference in references
        for query in holdout
    ):
        raise ValueError("perceptual near duplicate crosses reference and holdout")


def _contained_regular_file(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("acquired image path escapes the pilot root") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("acquired image must be a regular non-symlink file")
    return candidate


def _spatial_cell(latitude: float, longitude: float, *, cell_m: float = 75.0) -> str:
    latitude_step = cell_m / 111_320.0
    longitude_step = cell_m / max(1.0, 111_320.0 * math.cos(math.radians(latitude)))
    return f"cell-{math.floor(latitude / latitude_step)}-{math.floor(longitude / longitude_step)}"


def _distance_images(left: AcquiredImage, right: AcquiredImage) -> float:
    left_longitude, left_latitude = left.computed_geometry.coordinates
    right_longitude, right_latitude = right.computed_geometry.coordinates
    return haversine_distance_m(
        left_latitude,
        left_longitude,
        right_latitude,
        right_longitude,
    )


def _threshold_key(threshold: float) -> str:
    return str(int(threshold))


def selected_asset_ids(assets: Iterable[SelectedMapillaryAsset]) -> tuple[str, ...]:
    return tuple(sorted(asset.asset_id for asset in assets))


__all__ = [
    "HARD_IMAGE_CAP",
    "HOLDOUT_TARGET",
    "LockedMapillarySplit",
    "POSITIVE_THRESHOLDS_M",
    "PilotCityDecision",
    "REFERENCE_TARGET",
    "SelectedMapillaryAsset",
    "SelectionLimits",
    "choose_pilot_city",
    "select_locked_split",
    "selected_asset_ids",
]
