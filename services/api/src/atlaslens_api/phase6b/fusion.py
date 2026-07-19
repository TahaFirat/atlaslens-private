from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from atlaslens_api.phase6b.models import (
    GeographicCandidate,
    GeographicProviderResult,
    GeographicSourceFamily,
    Phase6BModel,
)
from atlaslens_api.reranking.geo import geodesic_km, percentile, spherical_center
from atlaslens_api.schemas import PlaceEvidenceSummary


class Phase6BDistanceBands(Phase6BModel):
    point_km: float = Field(default=10.0, gt=0, le=100)
    city_km: float = Field(default=75.0, gt=0, le=500)
    regional_km: float = Field(default=350.0, gt=0, le=2_000)
    country_km: float = Field(default=1_500.0, gt=0, le=5_000)
    strong_disagreement_km: float = Field(default=3_000.0, gt=0, le=20_000)

    @model_validator(mode="after")
    def ascending(self) -> Phase6BDistanceBands:
        values = (
            self.point_km,
            self.city_km,
            self.regional_km,
            self.country_km,
            self.strong_disagreement_km,
        )
        if tuple(sorted(values)) != values or len(set(values)) != len(values):
            raise ValueError("Phase 6B distance bands must be strictly ascending")
        return self


class Phase6BWeights(Phase6BModel):
    within_provider_rank: float = Field(default=0.34, ge=0, le=1)
    sample_density: float = Field(default=0.08, ge=0, le=1)
    provider_diversity: float = Field(default=0.06, ge=0, le=1)
    independent_family_agreement: float = Field(default=0.14, ge=0, le=1)
    same_family_support: float = Field(default=0.03, ge=0, le=1)
    candidate_margin: float = Field(default=0.05, ge=0, le=1)
    ocr_place_agreement: float = Field(default=0.20, ge=0, le=1)
    geographic_spread_penalty: float = Field(default=0.10, ge=0, le=1)
    ocr_contradiction_penalty: float = Field(default=0.30, ge=0, le=1)


class Phase6BFusionConfig(Phase6BModel):
    version: str = Field(default="phase6b-v1", pattern=r"^phase6b-v1$")
    distance_bands: Phase6BDistanceBands = Field(default_factory=Phase6BDistanceBands)
    weights: Phase6BWeights = Field(default_factory=Phase6BWeights)
    max_candidates: int = Field(default=5, ge=1, le=20)
    maximum_provider_diversity: int = Field(default=4, ge=1, le=16)
    maximum_independent_families: int = Field(default=3, ge=1, le=6)
    minimum_ocr_strength: float = Field(default=0.55, ge=0, le=1)
    strong_ocr_strength: float = Field(default=0.75, ge=0, le=1)
    provider_family_defaults: dict[str, GeographicSourceFamily] = Field(
        default_factory=dict, max_length=16
    )
    model_family_overrides: dict[str, GeographicSourceFamily] = Field(
        default_factory=dict, max_length=16
    )

    @field_validator("provider_family_defaults")
    @classmethod
    def safe_provider_keys(
        cls, value: dict[str, GeographicSourceFamily]
    ) -> dict[str, GeographicSourceFamily]:
        if any(not key or len(key) > 80 for key in value):
            raise ValueError("provider family keys must be bounded")
        return value

    @field_validator("model_family_overrides")
    @classmethod
    def safe_model_keys(
        cls, value: dict[str, GeographicSourceFamily]
    ) -> dict[str, GeographicSourceFamily]:
        if any(not key or len(key) > 160 for key in value):
            raise ValueError("model family keys must be bounded")
        return value


def load_phase6b_fusion_config(path: Path) -> Phase6BFusionConfig:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("Phase 6B fusion config is missing or unsafe")
    return Phase6BFusionConfig.model_validate_json(path.read_text(encoding="utf-8"))


class FusionMember(Phase6BModel):
    provider: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=160)
    candidate_id: str = Field(min_length=1, max_length=128)
    source_family: GeographicSourceFamily
    provider_rank: int = Field(ge=1, le=100)
    sample_support: int = Field(ge=1, le=4096)


class FusionContribution(Phase6BModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    raw_value: float
    weight: float
    contribution: float
    reason: str = Field(min_length=1, max_length=240)

    @field_validator("raw_value", "weight", "contribution")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fusion contribution values must be finite")
        return value


class Phase6BFusedCandidate(Phase6BModel):
    cluster_id: str = Field(min_length=1, max_length=128)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0)
    relative_rank_score: float = Field(ge=0, le=1)
    score_semantics: str = "uncalibrated_relative_rank_not_probability"
    provider_count: int = Field(ge=1, le=16)
    independent_family_count: int = Field(ge=1, le=6)
    same_family_duplicate_support: int = Field(ge=0, le=16)
    spread_km: float = Field(ge=0)
    members: tuple[FusionMember, ...] = Field(min_length=1, max_length=100)
    contributions: tuple[FusionContribution, ...] = Field(min_length=1, max_length=20)
    ocr_agreement: bool
    ocr_contradiction: bool


class Phase6BFusionResult(Phase6BModel):
    version: str = Field(pattern=r"^phase6b-v1$")
    candidates: tuple[Phase6BFusedCandidate, ...] = Field(max_length=20)
    provider_failures: tuple[str, ...] = Field(default=(), max_length=24)
    abstained: bool
    warnings: tuple[str, ...] = ("warning.confidence_not_calibrated",)

    @model_validator(mode="after")
    def coherent(self) -> Phase6BFusionResult:
        if self.abstained == bool(self.candidates):
            raise ValueError("fusion abstention and candidates are inconsistent")
        return self


class _Member:
    __slots__ = ("candidate", "family", "model_id", "provider")

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        family: GeographicSourceFamily,
        candidate: GeographicCandidate,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.family = family
        self.candidate = candidate


_OCR_RADIUS_KM: Mapping[str, float] = {
    "country": 1_500.0,
    "domain_suffix": 1_500.0,
    "region": 350.0,
    "city": 100.0,
    "district": 50.0,
    "road": 35.0,
    "airport": 50.0,
    "station": 35.0,
    "public_landmark": 25.0,
    "public_institution": 35.0,
}


class Phase6BGeographicFusionEngine:
    """Rank provider coordinates by rank, geography and independent evidence.

    Raw model scores are intentionally never read by this engine.
    """

    def __init__(self, config: Phase6BFusionConfig | None = None) -> None:
        self._config = config or Phase6BFusionConfig()

    def fuse(
        self,
        providers: Sequence[GeographicProviderResult],
        *,
        ocr_places: Sequence[PlaceEvidenceSummary] = (),
    ) -> Phase6BFusionResult:
        failures: list[str] = []
        members: list[_Member] = []
        provider_max_support: dict[str, int] = {}
        for result in providers:
            if result.status != "completed":
                failures.append(f"{result.provider}:{result.reason_code or result.status}")
                continue
            expected = self._expected_family(result)
            if expected != result.source_family:
                failures.append(f"{result.provider}:source_family_corrected")
            provider_max_support[result.provider] = max(
                item.sample_support for item in result.candidates
            )
            members.extend(
                _Member(
                    provider=result.provider,
                    model_id=result.model_id,
                    family=expected,
                    candidate=candidate,
                )
                for candidate in result.candidates
            )
        if not members:
            return Phase6BFusionResult(
                version=self._config.version,
                candidates=(),
                provider_failures=tuple(dict.fromkeys(failures))[:24],
                abstained=True,
            )
        clusters = self._cluster(members)
        fused = [
            self._score_cluster(
                cluster,
                provider_max_support=provider_max_support,
                ocr_places=ocr_places,
            )
            for cluster in clusters
        ]
        fused.sort(
            key=lambda item: (
                -item.relative_rank_score,
                -item.independent_family_count,
                -item.provider_count,
                item.cluster_id,
            )
        )
        return Phase6BFusionResult(
            version=self._config.version,
            candidates=tuple(fused[: self._config.max_candidates]),
            provider_failures=tuple(dict.fromkeys(failures))[:24],
            abstained=False,
        )

    def _expected_family(self, result: GeographicProviderResult) -> GeographicSourceFamily:
        return self._config.model_family_overrides.get(
            result.model_id,
            self._config.provider_family_defaults.get(result.provider, result.source_family),
        )

    def _cluster(self, members: Sequence[_Member]) -> tuple[tuple[_Member, ...], ...]:
        parents = list(range(len(members)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        for left in range(len(members)):
            left_point = (
                members[left].candidate.latitude,
                members[left].candidate.longitude,
            )
            for right in range(left + 1, len(members)):
                right_point = (
                    members[right].candidate.latitude,
                    members[right].candidate.longitude,
                )
                if geodesic_km(left_point, right_point) <= self._config.distance_bands.city_km:
                    union(left, right)
        grouped: dict[int, list[_Member]] = defaultdict(list)
        for index, member in enumerate(members):
            grouped[find(index)].append(member)
        return tuple(
            tuple(
                sorted(
                    group,
                    key=lambda item: (
                        item.provider,
                        item.candidate.provider_rank,
                        item.candidate.candidate_id,
                    ),
                )
            )
            for _, group in sorted(grouped.items())
        )

    def _score_cluster(
        self,
        members: tuple[_Member, ...],
        *,
        provider_max_support: Mapping[str, int],
        ocr_places: Sequence[PlaceEvidenceSummary],
    ) -> Phase6BFusedCandidate:
        points = [(item.candidate.latitude, item.candidate.longitude) for item in members]
        center = spherical_center(points)
        distances = [geodesic_km(center, point) for point in points]
        spread = percentile(distances, 0.80)
        providers = sorted({item.provider for item in members})
        family_providers: dict[GeographicSourceFamily, set[str]] = defaultdict(set)
        for item in members:
            family_providers[item.family].add(item.provider)
        families = sorted(family_providers)
        same_family_duplicates = sum(max(0, len(value) - 1) for value in family_providers.values())

        best_rank_by_provider: dict[str, int] = {}
        best_density_by_provider: dict[str, float] = {}
        for item in members:
            rank = item.candidate.provider_rank
            best_rank_by_provider[item.provider] = min(
                rank, best_rank_by_provider.get(item.provider, rank)
            )
            density = item.candidate.sample_support / provider_max_support[item.provider]
            best_density_by_provider[item.provider] = max(
                density, best_density_by_provider.get(item.provider, 0.0)
            )
        rank_value = sum(1.0 / value for value in best_rank_by_provider.values()) / len(
            best_rank_by_provider
        )
        density_value = sum(best_density_by_provider.values()) / len(best_density_by_provider)
        provider_value = min(1.0, len(providers) / self._config.maximum_provider_diversity)
        independent_value = min(
            1.0,
            max(0, len(families) - 1) / max(1, self._config.maximum_independent_families - 1),
        )
        same_family_value = min(1.0, same_family_duplicates / 2.0)
        rank_margin = sum(
            (1.0 / rank) - (1.0 / (rank + 1)) for rank in best_rank_by_provider.values()
        ) / len(best_rank_by_provider)
        margin_value = min(1.0, rank_margin * 2.0)
        spread_value = min(1.0, spread / self._config.distance_bands.city_km)
        ocr_agreement, ocr_contradiction = self._ocr_values(center, ocr_places)
        weights = self._config.weights
        contributions = (
            self._contribution(
                "within_provider_rank",
                rank_value,
                weights.within_provider_rank,
                "Best rank is normalized only within each provider.",
            ),
            self._contribution(
                "sample_density",
                density_value,
                weights.sample_density,
                "Sample support is normalized inside its originating provider.",
            ),
            self._contribution(
                "provider_diversity",
                provider_value,
                weights.provider_diversity,
                f"{len(providers)} distinct geographic providers support this cluster.",
            ),
            self._contribution(
                "independent_model_agreement",
                independent_value,
                weights.independent_family_agreement,
                f"{len(families)} independent source families support nearby coordinates.",
            ),
            self._contribution(
                "same_family_support",
                same_family_value,
                weights.same_family_support,
                (
                    f"{same_family_duplicates} additional provider supports share a model family; "
                    "their contribution is reduced."
                ),
            ),
            self._contribution(
                "candidate_margin",
                margin_value,
                weights.candidate_margin,
                "Reciprocal-rank separation is computed without raw provider scores.",
            ),
            self._contribution(
                "ocr_place_agreement",
                ocr_agreement,
                weights.ocr_place_agreement,
                "Strong local OCR place evidence is geographically compatible.",
            ),
            self._contribution(
                "geographic_spread_penalty",
                spread_value,
                -weights.geographic_spread_penalty,
                "Wider member dispersion reduces relative support.",
            ),
            self._contribution(
                "ocr_contradiction_penalty",
                ocr_contradiction,
                -weights.ocr_contradiction_penalty,
                "Strong OCR place evidence points well outside this cluster.",
            ),
        )
        relative_score = min(1.0, max(0.0, sum(item.contribution for item in contributions)))
        stable_payload = "|".join(
            f"{item.provider}:{item.candidate.candidate_id}" for item in members
        )
        stable = hashlib.sha256(stable_payload.encode()).hexdigest()[:24]
        return Phase6BFusedCandidate(
            cluster_id=f"phase6b-{stable}",
            latitude=center[0],
            longitude=center[1],
            radius_km=max(25.0, spread * 2.0),
            relative_rank_score=relative_score,
            provider_count=len(providers),
            independent_family_count=len(families),
            same_family_duplicate_support=same_family_duplicates,
            spread_km=spread,
            members=tuple(
                FusionMember(
                    provider=item.provider,
                    model_id=item.model_id,
                    candidate_id=item.candidate.candidate_id,
                    source_family=item.family,
                    provider_rank=item.candidate.provider_rank,
                    sample_support=item.candidate.sample_support,
                )
                for item in members
            ),
            contributions=contributions,
            ocr_agreement=ocr_agreement > 0,
            ocr_contradiction=ocr_contradiction > 0,
        )

    def _ocr_values(
        self,
        center: tuple[float, float],
        ocr_places: Sequence[PlaceEvidenceSummary],
    ) -> tuple[float, float]:
        agreement = 0.0
        contradiction = 0.0
        for place in ocr_places:
            if place.evidence_strength < self._config.minimum_ocr_strength:
                continue
            radius = _OCR_RADIUS_KM[place.match_type]
            distance = geodesic_km(center, (place.center.latitude, place.center.longitude))
            if distance <= radius:
                agreement = max(
                    agreement,
                    place.evidence_strength * max(0.5, 1.0 - distance / radius),
                )
            elif (
                place.evidence_strength >= self._config.strong_ocr_strength
                and place.ambiguity_count <= 3
                and distance > radius * 2
            ):
                contradiction = max(contradiction, place.evidence_strength)
        return agreement, contradiction

    @staticmethod
    def _contribution(
        name: str, raw_value: float, weight: float, reason: str
    ) -> FusionContribution:
        return FusionContribution(
            name=name,
            raw_value=raw_value,
            weight=weight,
            contribution=raw_value * weight,
            reason=reason,
        )
