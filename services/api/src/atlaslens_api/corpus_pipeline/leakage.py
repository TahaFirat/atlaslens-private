from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Final

from atlaslens_api.reranking.geo import EARTH_RADIUS_KM, geodesic_km

from .errors import LeakageValidationError
from .hashing import perceptual_hamming_distance
from .models import (
    IngestedAsset,
    LeakageFinding,
    LeakageReport,
    RevocationImpact,
    SplitAssignment,
    SplitLock,
    canonical_json_sha256,
)

_ROLE_PRIORITY: Final = {"holdout": 0, "validation": 1, "development": 2, "train": 3}
_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")


@dataclass(frozen=True, slots=True)
class LeakageDecision:
    accepted: tuple[IngestedAsset, ...]
    excluded_reason_by_asset: Mapping[str, str]
    report: LeakageReport


@dataclass(slots=True)
class _BKNode:
    value: int
    asset: IngestedAsset
    children: dict[int, _BKNode]


class _HammingIndex:
    def __init__(self) -> None:
        self._root: _BKNode | None = None

    def add(self, asset: IngestedAsset) -> None:
        value = int(asset.perceptual_hash, 16)
        if self._root is None:
            self._root = _BKNode(value=value, asset=asset, children={})
            return
        node = self._root
        while True:
            distance = (node.value ^ value).bit_count()
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value=value, asset=asset, children={})
                return
            node = child

    def nearest_within(self, value_hex: str, threshold: int) -> tuple[IngestedAsset, int] | None:
        if self._root is None:
            return None
        value = int(value_hex, 16)
        pending = [self._root]
        matches: list[tuple[int, IngestedAsset]] = []
        while pending:
            node = pending.pop()
            distance = (node.value ^ value).bit_count()
            if distance <= threshold:
                matches.append((distance, node.asset))
            lower = distance - threshold
            upper = distance + threshold
            pending.extend(
                child
                for edge, child in node.children.items()
                if lower <= edge <= upper
            )
        if not matches:
            return None
        distance, asset = min(matches, key=lambda item: (item[0], item[1].asset_id))
        return asset, distance


def _ecef_bucket(asset: IngestedAsset, bucket_size_m: float) -> tuple[int, int, int]:
    latitude = math.radians(asset.latitude)
    longitude = math.radians(asset.longitude)
    radius_m = EARTH_RADIUS_KM * 1000.0
    x = radius_m * math.cos(latitude) * math.cos(longitude)
    y = radius_m * math.cos(latitude) * math.sin(longitude)
    z = radius_m * math.sin(latitude)
    return (
        math.floor(x / bucket_size_m),
        math.floor(y / bucket_size_m),
        math.floor(z / bucket_size_m),
    )


def _cross_role(left: IngestedAsset, right: IngestedAsset) -> bool:
    return left.role != right.role


def filter_leakage(
    assets: Sequence[IngestedAsset],
    *,
    near_duplicate_hamming_threshold: int,
    spatial_leakage_radius_m: float,
) -> LeakageDecision:
    """Retain locked holdout/validation rows first, then exclude conflicting rows."""

    if not 0 <= near_duplicate_hamming_threshold <= 64:
        raise ValueError("near duplicate threshold must be in [0, 64]")
    if spatial_leakage_radius_m <= 0:
        raise ValueError("spatial leakage radius must be positive")
    ordered = sorted(assets, key=lambda item: (_ROLE_PRIORITY[item.role], item.asset_id))
    exact: dict[str, IngestedAsset] = {}
    perceptual = _HammingIndex()
    sequence: dict[tuple[str, str], IngestedAsset] = {}
    capture_run: dict[tuple[str, str], IngestedAsset] = {}
    parent_source: dict[tuple[str, str], IngestedAsset] = {}
    source_tile: dict[tuple[str, str], IngestedAsset] = {}
    spatial: dict[tuple[int, int, int], list[IngestedAsset]] = {}
    accepted: list[IngestedAsset] = []
    findings: list[LeakageFinding] = []
    reasons: dict[str, str] = {}

    for candidate in ordered:
        retained = exact.get(candidate.image_sha256)
        if retained is not None:
            findings.append(
                LeakageFinding(
                    kind="exact_duplicate",
                    retained_asset_id=retained.asset_id,
                    excluded_asset_id=candidate.asset_id,
                    hamming_distance=perceptual_hamming_distance(
                        retained.perceptual_hash,
                        candidate.perceptual_hash,
                    ),
                )
            )
            reasons[candidate.asset_id] = "exact_duplicate"
            continue

        near = perceptual.nearest_within(
            candidate.perceptual_hash,
            near_duplicate_hamming_threshold,
        )
        if near is not None:
            retained, distance = near
            findings.append(
                LeakageFinding(
                    kind="near_duplicate",
                    retained_asset_id=retained.asset_id,
                    excluded_asset_id=candidate.asset_id,
                    hamming_distance=distance,
                )
            )
            reasons[candidate.asset_id] = "near_duplicate"
            continue

        if candidate.sequence_id is not None:
            sequence_key = (candidate.source_name, candidate.sequence_id)
            retained = sequence.get(sequence_key)
            if retained is not None and retained.spatial_split != candidate.spatial_split:
                findings.append(
                    LeakageFinding(
                        kind="sequence_cross_spatial_split",
                        retained_asset_id=retained.asset_id,
                        excluded_asset_id=candidate.asset_id,
                    )
                )
                reasons[candidate.asset_id] = "sequence_cross_spatial_split"
                continue
            if retained is not None and _cross_role(retained, candidate):
                findings.append(
                    LeakageFinding(
                        kind="sequence_cross_split",
                        retained_asset_id=retained.asset_id,
                        excluded_asset_id=candidate.asset_id,
                    )
                )
                reasons[candidate.asset_id] = "sequence_cross_split"
                continue

        if candidate.capture_run_id is not None:
            run_key = (
                candidate.contributor_or_owner,
                candidate.capture_run_id,
            )
            retained = capture_run.get(run_key)
            if retained is not None and _cross_role(retained, candidate):
                findings.append(
                    LeakageFinding(
                        kind="capture_run_cross_split",
                        retained_asset_id=retained.asset_id,
                        excluded_asset_id=candidate.asset_id,
                    )
                )
                reasons[candidate.asset_id] = "capture_run_cross_split"
                continue

        identity_conflicts: list[IngestedAsset] = []
        if candidate.parent_source_asset_id is not None:
            retained = parent_source.get(
                (candidate.source_name, candidate.parent_source_asset_id)
            )
            if retained is not None and _cross_role(retained, candidate):
                identity_conflicts.append(retained)
        if candidate.source_tile_id is not None:
            retained = source_tile.get((candidate.source_name, candidate.source_tile_id))
            if retained is not None and _cross_role(retained, candidate):
                identity_conflicts.append(retained)
        if identity_conflicts:
            retained = min(identity_conflicts, key=lambda item: item.asset_id)
            findings.append(
                LeakageFinding(
                    kind="source_parent_or_tile_cross_role",
                    retained_asset_id=retained.asset_id,
                    excluded_asset_id=candidate.asset_id,
                )
            )
            reasons[candidate.asset_id] = "source_parent_or_tile_cross_role"
            continue

        bucket = _ecef_bucket(candidate, spatial_leakage_radius_m)
        spatial_conflicts: list[tuple[float, IngestedAsset]] = []
        for delta in product((-1, 0, 1), repeat=3):
            neighbor = (
                bucket[0] + delta[0],
                bucket[1] + delta[1],
                bucket[2] + delta[2],
            )
            for retained_candidate in spatial.get(neighbor, ()):
                if not _cross_role(retained_candidate, candidate):
                    continue
                distance_m = 1000.0 * geodesic_km(
                    (candidate.latitude, candidate.longitude),
                    (retained_candidate.latitude, retained_candidate.longitude),
                )
                if distance_m <= spatial_leakage_radius_m:
                    spatial_conflicts.append((distance_m, retained_candidate))
        if spatial_conflicts:
            distance_m, retained = min(
                spatial_conflicts,
                key=lambda item: (item[0], item[1].asset_id),
            )
            findings.append(
                LeakageFinding(
                    kind="spatial_cross_split",
                    retained_asset_id=retained.asset_id,
                    excluded_asset_id=candidate.asset_id,
                    distance_m=round(distance_m, 6),
                )
            )
            reasons[candidate.asset_id] = "spatial_cross_split"
            continue

        accepted.append(candidate)
        exact[candidate.image_sha256] = candidate
        perceptual.add(candidate)
        if candidate.sequence_id is not None:
            sequence.setdefault(
                (candidate.source_name, candidate.sequence_id),
                candidate,
            )
        if candidate.capture_run_id is not None:
            capture_run.setdefault(
                (
                    candidate.contributor_or_owner,
                    candidate.capture_run_id,
                ),
                candidate,
            )
        if candidate.parent_source_asset_id is not None:
            parent_source.setdefault(
                (candidate.source_name, candidate.parent_source_asset_id),
                candidate,
            )
        if candidate.source_tile_id is not None:
            source_tile.setdefault(
                (candidate.source_name, candidate.source_tile_id),
                candidate,
            )
        spatial.setdefault(bucket, []).append(candidate)

    accepted_tuple = tuple(sorted(accepted, key=lambda item: item.asset_id))
    findings_tuple = tuple(
        sorted(findings, key=lambda item: (item.excluded_asset_id, item.kind))
    )
    return LeakageDecision(
        accepted=accepted_tuple,
        excluded_reason_by_asset=dict(sorted(reasons.items())),
        report=LeakageReport(
            status="passed" if not findings_tuple else "excluded_conflicts",
            input_count=len(assets),
            accepted_count=len(accepted_tuple),
            excluded_count=len(findings_tuple),
            findings=findings_tuple,
        ),
    )


def build_split_lock(
    assets: Sequence[IngestedAsset],
    *,
    manifest_sha256: str,
    config_sha256: str,
) -> SplitLock:
    assignments = tuple(
        sorted(
            (
                SplitAssignment(
                    asset_id=asset.asset_id,
                    image_sha256=asset.image_sha256,
                    role=asset.role,
                    tier=asset.tier,
                    spatial_split=asset.spatial_split,
                )
                for asset in assets
            ),
            key=lambda item: item.asset_id,
        )
    )
    base = {
        "schema_version": "atlaslens-corpus-split-lock-v1",
        "manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
        "assignments": [item.model_dump(mode="json") for item in assignments],
    }
    return SplitLock(
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
        assignments=assignments,
        lock_sha256=canonical_json_sha256(base),
    )


def plan_revocation(
    corpus: Sequence[IngestedAsset],
    revoked_asset_ids: Sequence[str],
    *,
    artifact_membership: Mapping[str, Sequence[str]],
) -> RevocationImpact:
    known = {asset.asset_id for asset in corpus}
    asset_ids = tuple(sorted(set(revoked_asset_ids)))
    if not asset_ids or any(asset_id not in known for asset_id in asset_ids):
        raise LeakageValidationError("revocation_asset_unknown")
    artifacts: set[str] = set()
    for asset_id in asset_ids:
        for artifact in artifact_membership.get(asset_id, ()):
            if _ARTIFACT_NAME.fullmatch(artifact) is None or ".." in artifact.split("/"):
                raise LeakageValidationError("revocation_artifact_identifier_invalid")
            artifacts.add(artifact)
    actions = (
        "exclude_assets",
        "rebuild_descriptors",
        "rebuild_indexes",
        "verify_checksum_inventory",
    )
    base = {
        "schema_version": "atlaslens-revocation-impact-v1",
        "asset_ids": asset_ids,
        "impacted_artifacts": tuple(sorted(artifacts)),
        "rebuild_required": True,
        "actions": actions,
        "evidence_semantics": "operational_plan_not_legal_or_cryptographic_certification",
    }
    return RevocationImpact(
        asset_ids=asset_ids,
        impacted_artifacts=tuple(sorted(artifacts)),
        actions=(
            "exclude_assets",
            "rebuild_descriptors",
            "rebuild_indexes",
            "verify_checksum_inventory",
        ),
        plan_sha256=canonical_json_sha256(base),
    )


__all__ = [
    "LeakageDecision",
    "build_split_lock",
    "filter_leakage",
    "plan_revocation",
]
