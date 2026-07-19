"""Deterministic metadata-only city selection for the Phase 3F private pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

SELECTION_SCHEMA: Final = "atlaslens-phase3f-selection-lock-v1"
MIN_IN_DOMAIN_ASSETS: Final = 125
MIN_OOD_ASSETS: Final = 40
MIN_IN_DOMAIN_CITIES: Final = 5
MIN_OOD_CITIES: Final = 2
MIN_MACRO_REGIONS: Final = 4

CANDIDATE_CITY_REGIONS: Mapping[str, str] = MappingProxyType(
    {
        "İstanbul": "Marmara",
        "Ankara": "İç Anadolu",
        "İzmir": "Ege",
        "Bursa": "Marmara",
        "Antalya": "Akdeniz",
        "Konya": "İç Anadolu",
        "Kayseri": "İç Anadolu",
        "Gaziantep": "Güneydoğu Anadolu",
        "Samsun": "Karadeniz",
        "Trabzon": "Karadeniz",
        "Diyarbakır": "Güneydoğu Anadolu",
        "Erzurum": "Doğu Anadolu",
        "Mersin": "Akdeniz",
        "Adana": "Akdeniz",
        "Eskişehir": "İç Anadolu",
        "Sivas": "İç Anadolu",
    }
)


class CoverageInsufficient(RuntimeError):
    """Raised before imagery acquisition when the metadata gate cannot be met."""

    code = "COVERAGE_INSUFFICIENT"

    def __init__(self, reason: str) -> None:
        super().__init__(self.code)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class CityCoverageRecord:
    city: str
    macro_region: str
    image_count: int
    sequence_count: int
    contributor_count: int
    spatial_cell_count: int
    capture_year_count: int
    eligible_asset_count: int
    metadata_sha256: str

    def __post_init__(self) -> None:
        expected_region = CANDIDATE_CITY_REGIONS.get(self.city)
        if expected_region is None or expected_region != self.macro_region:
            raise ValueError("city is outside the locked Phase 3F candidate pool")
        values = (
            self.image_count,
            self.sequence_count,
            self.contributor_count,
            self.spatial_cell_count,
            self.capture_year_count,
            self.eligible_asset_count,
        )
        if any(value < 0 for value in values):
            raise ValueError("coverage counts must be non-negative")
        if len(self.metadata_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.metadata_sha256
        ):
            raise ValueError("metadata_sha256 must be lowercase SHA-256")

    @property
    def score_micros(self) -> int:
        """Return a stable integer score; similarity or probability semantics do not apply."""

        components = (
            (min(self.sequence_count, 120), 120, 22_000),
            (min(self.contributor_count, 80), 80, 18_000),
            (min(self.spatial_cell_count, 160), 160, 22_000),
            (min(self.capture_year_count, 8), 8, 13_000),
            (min(self.eligible_asset_count, 250), 250, 25_000),
        )
        return sum(value * weight // cap for value, cap, weight in components)

    def public_record(self) -> dict[str, object]:
        return {
            "city": self.city,
            "macro_region": self.macro_region,
            "image_count": self.image_count,
            "sequence_count": self.sequence_count,
            "contributor_count": self.contributor_count,
            "spatial_cell_count": self.spatial_cell_count,
            "capture_year_count": self.capture_year_count,
            "eligible_asset_count": self.eligible_asset_count,
            "metadata_sha256": self.metadata_sha256,
            "selection_score_micros": self.score_micros,
        }


@dataclass(frozen=True, slots=True)
class CoverageSelectionLock:
    in_domain: tuple[CityCoverageRecord, ...]
    ood: tuple[CityCoverageRecord, ...]
    audited_city_count: int
    policy_sha256: str

    def __post_init__(self) -> None:
        if len(self.in_domain) < MIN_IN_DOMAIN_CITIES or len(self.in_domain) > 8:
            raise ValueError("in-domain city count is outside [5, 8]")
        if len(self.ood) < MIN_OOD_CITIES:
            raise ValueError("at least two OOD cities are required")
        identities = [record.city for record in (*self.in_domain, *self.ood)]
        if len(identities) != len(set(identities)):
            raise ValueError("a city cannot cross in-domain and OOD scopes")
        if len({record.macro_region for record in self.in_domain}) < MIN_MACRO_REGIONS:
            raise ValueError("in-domain scope must cover at least four macro regions")

    def document(self) -> dict[str, object]:
        return {
            "schema": SELECTION_SCHEMA,
            "selection_stage": "metadata_only_before_image_acquisition",
            "candidate_city_count": len(CANDIDATE_CITY_REGIONS),
            "audited_city_count": self.audited_city_count,
            "minimum_in_domain_assets": MIN_IN_DOMAIN_ASSETS,
            "minimum_ood_assets": MIN_OOD_ASSETS,
            "minimum_macro_regions": MIN_MACRO_REGIONS,
            "policy_sha256": self.policy_sha256,
            "in_domain": [record.public_record() for record in self.in_domain],
            "ood": [record.public_record() for record in self.ood],
        }

    @property
    def lock_sha256(self) -> str:
        return _canonical_sha256(self.document())


def select_city_scope(
    records: Sequence[CityCoverageRecord],
    *,
    policy_sha256: str,
    in_domain_target: int = 6,
    ood_target: int = 2,
) -> CoverageSelectionLock:
    """Lock a diverse city scope without alphabetical or operator-history ordering."""

    if not 5 <= in_domain_target <= 8 or ood_target < 2:
        raise ValueError("city selection targets are outside Phase 3F policy")
    if len(policy_sha256) != 64:
        raise ValueError("policy_sha256 must be SHA-256")
    by_city = {record.city: record for record in records}
    if len(by_city) != len(records):
        raise ValueError("coverage records must be unique by city")
    if set(by_city) != set(CANDIDATE_CITY_REGIONS):
        raise CoverageInsufficient("all sixteen candidate cities require metadata audit")

    eligible = [record for record in records if record.eligible_asset_count >= MIN_IN_DOMAIN_ASSETS]
    if len(eligible) < MIN_IN_DOMAIN_CITIES:
        raise CoverageInsufficient("fewer than five cities meet in-domain asset minimums")

    ranked = sorted(eligible, key=_rank_key)
    selected: list[CityCoverageRecord] = []
    represented: set[str] = set()
    best_by_region: dict[str, CityCoverageRecord] = {}
    for record in ranked:
        best_by_region.setdefault(record.macro_region, record)
    regional_seeds = sorted(best_by_region.values(), key=_rank_key)[:MIN_MACRO_REGIONS]
    selected.extend(regional_seeds)
    represented.update(record.macro_region for record in regional_seeds)
    for record in ranked:
        if len(selected) >= in_domain_target:
            break
        if record not in selected:
            selected.append(record)

    if len(selected) < MIN_IN_DOMAIN_CITIES or len(represented) < MIN_MACRO_REGIONS:
        raise CoverageInsufficient("macro-region diversity gate cannot be met")

    remaining = [
        record
        for record in records
        if record not in selected and record.eligible_asset_count >= MIN_OOD_ASSETS
    ]
    if len(remaining) < MIN_OOD_CITIES:
        raise CoverageInsufficient("fewer than two held-out cities meet OOD minimums")
    ood: list[CityCoverageRecord] = []
    for record in sorted(remaining, key=_rank_key):
        if not ood or record.macro_region not in {item.macro_region for item in ood}:
            ood.append(record)
        if len(ood) >= ood_target:
            break
    for record in sorted(remaining, key=_rank_key):
        if len(ood) >= ood_target:
            break
        if record not in ood:
            ood.append(record)

    return CoverageSelectionLock(
        in_domain=tuple(selected),
        ood=tuple(ood),
        audited_city_count=len(records),
        policy_sha256=policy_sha256,
    )


def _rank_key(record: CityCoverageRecord) -> tuple[int, str, str]:
    city_digest = hashlib.sha256(record.city.encode("utf-8")).hexdigest()
    return (-record.score_micros, record.metadata_sha256, city_digest)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CANDIDATE_CITY_REGIONS",
    "CityCoverageRecord",
    "CoverageInsufficient",
    "CoverageSelectionLock",
    "select_city_scope",
]
