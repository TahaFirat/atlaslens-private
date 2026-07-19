"""Fail-closed split sealing and leakage validation for Phase 3F."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Final, Literal

SplitRole = Literal["reference", "calibration", "sealed_holdout", "ood_holdout"]

MIN_REFERENCE_PER_CITY: Final = 75
MIN_CALIBRATION_PER_CITY: Final = 25
MIN_HOLDOUT_PER_CITY: Final = 25
MIN_OOD_PER_CITY: Final = 40
SPATIAL_EXCLUSION_METERS: Final = 1_000.0
NEAR_DUPLICATE_HAMMING: Final = 4
_OPAQUE_PATH = re.compile(r"^assets/[0-9a-f]{2}/[0-9a-f]{32}\.(?:jpg|png|webp)$")


class LeakageViolation(RuntimeError):
    code = "PHASE3F_LEAKAGE_VIOLATION"

    def __init__(self, invariant: str) -> None:
        super().__init__(self.code)
        self.invariant = invariant


@dataclass(frozen=True, slots=True)
class SplitAsset:
    opaque_id: str
    city: str
    role: SplitRole
    relative_path: str
    contributor_id: str
    sequence_id: str
    capture_run_id: str
    content_sha256: str
    perceptual_hash: str
    parent_or_tile_id: str
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{32}", self.opaque_id):
            raise ValueError("opaque_id must be a random-looking 128-bit hex identifier")
        if not _OPAQUE_PATH.fullmatch(self.relative_path):
            raise ValueError("asset path must be opaque and city-free")
        if self.opaque_id not in self.relative_path:
            raise ValueError("asset path must bind its opaque identifier")
        if len(self.content_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.content_sha256
        ):
            raise ValueError("content_sha256 must be lowercase SHA-256")
        if not re.fullmatch(r"[0-9a-f]{16}", self.perceptual_hash):
            raise ValueError("perceptual_hash must be 64-bit lowercase hex")
        identifiers = (
            self.city,
            self.contributor_id,
            self.sequence_id,
            self.capture_run_id,
            self.parent_or_tile_id,
        )
        if any(not value or len(value) > 256 for value in identifiers):
            raise ValueError("split identifiers must be bounded and non-empty")
        if not (-90.0 <= self.latitude <= 90.0 and -180.0 <= self.longitude <= 180.0):
            raise ValueError("ground truth is outside WGS84 bounds")

    def inference_record(self) -> dict[str, object]:
        """Return only fields permitted outside the evaluator process."""

        return {
            "opaque_id": self.opaque_id,
            "role": self.role,
            "relative_path": self.relative_path,
            "content_sha256": self.content_sha256,
        }

    def truth_record(self) -> dict[str, object]:
        return {
            "opaque_id": self.opaque_id,
            "city": self.city,
            "role": self.role,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    contributor_overlap: int
    sequence_overlap: int
    capture_run_overlap: int
    exact_sha256_overlap: int
    perceptual_near_duplicate_overlap: int
    parent_or_tile_overlap: int
    sequence_neighbour_overlap: int
    spatial_exclusion_violations: int

    @property
    def passed(self) -> bool:
        return not any(asdict(self).values())

    def document(self) -> dict[str, object]:
        return {"passed": self.passed, **asdict(self)}


@dataclass(frozen=True, slots=True)
class SealedSplit:
    assets: tuple[SplitAsset, ...]
    in_domain_cities: tuple[str, ...]
    ood_cities: tuple[str, ...]
    leakage: LeakageAudit

    def __post_init__(self) -> None:
        if not self.leakage.passed:
            raise LeakageViolation("leakage_audit")

    def inference_document(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-phase3f-inference-split-v1",
            "ground_truth_present": False,
            "sealed_holdout_open_count": 0,
            "assets": [asset.inference_record() for asset in self.assets],
            "leakage": self.leakage.document(),
        }

    def evaluator_truth_document(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-phase3f-evaluator-truth-v1",
            "access": "evaluator_only_after_inference_complete",
            "assets": [asset.truth_record() for asset in self.assets],
        }

    @property
    def split_lock_sha256(self) -> str:
        return _canonical_sha256(self.inference_document())

    @property
    def holdout_seal_sha256(self) -> str:
        holdout = [
            asset.inference_record()
            for asset in self.assets
            if asset.role in {"sealed_holdout", "ood_holdout"}
        ]
        return _canonical_sha256(
            {"schema": "atlaslens-phase3f-holdout-seal-v1", "assets": holdout}
        )

    @property
    def evaluator_truth_sha256(self) -> str:
        return _canonical_sha256(self.evaluator_truth_document())


def seal_split(
    assets: Sequence[SplitAsset],
    *,
    in_domain_cities: Sequence[str],
    ood_cities: Sequence[str],
    spatial_exclusion_meters: float = SPATIAL_EXCLUSION_METERS,
) -> SealedSplit:
    if spatial_exclusion_meters < SPATIAL_EXCLUSION_METERS:
        raise ValueError("spatial exclusion cannot be weakened below one kilometre")
    in_scope = tuple(dict.fromkeys(in_domain_cities))
    ood_scope = tuple(dict.fromkeys(ood_cities))
    if len(in_scope) < 5 or len(ood_scope) < 2 or set(in_scope) & set(ood_scope):
        raise ValueError("Phase 3F city scopes are invalid")
    if len({asset.opaque_id for asset in assets}) != len(assets):
        raise LeakageViolation("opaque_id_duplicate")
    _validate_role_scope(assets, set(in_scope), set(ood_scope))
    _validate_minimums(assets, in_scope, ood_scope)
    leakage = audit_leakage(assets, spatial_exclusion_meters=spatial_exclusion_meters)
    if not leakage.passed:
        raise LeakageViolation(_first_failed_invariant(leakage))
    ordered = tuple(sorted(assets, key=lambda asset: (asset.role, asset.opaque_id)))
    return SealedSplit(
        assets=ordered,
        in_domain_cities=in_scope,
        ood_cities=ood_scope,
        leakage=leakage,
    )


def audit_leakage(
    assets: Sequence[SplitAsset], *, spatial_exclusion_meters: float
) -> LeakageAudit:
    contributor = _cross_role_overlap(assets, "contributor_id")
    sequence = _cross_role_overlap(assets, "sequence_id")
    capture_run = _cross_role_overlap(assets, "capture_run_id")
    exact = _cross_role_overlap(assets, "content_sha256")
    parent = _cross_role_overlap(assets, "parent_or_tile_id")
    near_duplicate = 0
    spatial = 0
    for index, left in enumerate(assets):
        for right in assets[index + 1 :]:
            if left.role == right.role:
                continue
            if _hamming(left.perceptual_hash, right.perceptual_hash) <= NEAR_DUPLICATE_HAMMING:
                near_duplicate += 1
            roles = {left.role, right.role}
            if (
                roles == {"reference", "sealed_holdout"}
                and left.city == right.city
                and _distance_m(left, right) < spatial_exclusion_meters
            ):
                spatial += 1
    return LeakageAudit(
        contributor_overlap=contributor,
        sequence_overlap=sequence,
        capture_run_overlap=capture_run,
        exact_sha256_overlap=exact,
        perceptual_near_duplicate_overlap=near_duplicate,
        parent_or_tile_overlap=parent,
        sequence_neighbour_overlap=sequence,
        spatial_exclusion_violations=spatial,
    )


def _validate_role_scope(
    assets: Sequence[SplitAsset], in_scope: set[str], ood_scope: set[str]
) -> None:
    for asset in assets:
        if asset.city in in_scope and asset.role == "ood_holdout":
            raise LeakageViolation("in_domain_asset_in_ood_role")
        if asset.city in ood_scope and asset.role != "ood_holdout":
            raise LeakageViolation("ood_asset_outside_ood_role")
        if asset.city not in in_scope | ood_scope:
            raise LeakageViolation("asset_city_outside_locked_scope")


def _validate_minimums(
    assets: Sequence[SplitAsset], in_scope: Sequence[str], ood_scope: Sequence[str]
) -> None:
    counts: dict[tuple[str, SplitRole], int] = {}
    for asset in assets:
        key = (asset.city, asset.role)
        counts[key] = counts.get(key, 0) + 1
    for city in in_scope:
        required: tuple[tuple[SplitRole, int], ...] = (
            ("reference", MIN_REFERENCE_PER_CITY),
            ("calibration", MIN_CALIBRATION_PER_CITY),
            ("sealed_holdout", MIN_HOLDOUT_PER_CITY),
        )
        if any(counts.get((city, role), 0) < minimum for role, minimum in required):
            raise LeakageViolation("in_domain_city_minimum")
    if any(counts.get((city, "ood_holdout"), 0) < MIN_OOD_PER_CITY for city in ood_scope):
        raise LeakageViolation("ood_city_minimum")


def _cross_role_overlap(assets: Sequence[SplitAsset], field: str) -> int:
    roles_by_value: dict[str, set[SplitRole]] = {}
    for asset in assets:
        value = str(getattr(asset, field))
        roles_by_value.setdefault(value, set()).add(asset.role)
    return sum(1 for roles in roles_by_value.values() if len(roles) > 1)


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _distance_m(left: SplitAsset, right: SplitAsset) -> float:
    radius = 6_371_008.8
    lat1, lat2 = math.radians(left.latitude), math.radians(right.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(right.longitude - left.longitude)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _first_failed_invariant(audit: LeakageAudit) -> str:
    for name, value in asdict(audit).items():
        if value:
            return name
    return "unknown"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "LeakageAudit",
    "LeakageViolation",
    "SealedSplit",
    "SplitAsset",
    "audit_leakage",
    "seal_split",
]
