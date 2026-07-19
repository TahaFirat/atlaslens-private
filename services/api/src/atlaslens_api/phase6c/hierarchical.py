from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.phase6c.catalogue import CoordinateCatalogue, CoordinateCatalogueRecord
from atlaslens_api.phase6c.grid import (
    ScoredPoint,
    approximate_grid_resolution_km,
    fibonacci_global_grid,
    geodesic_refinement_points,
    select_geographically_diverse,
)
from atlaslens_api.reranking.geo import geodesic_km


class CoordinateScorer(Protocol):
    async def score_coordinates(
        self,
        image_bytes: bytes,
        coordinates: Sequence[tuple[float, float]],
        *,
        cache_key: str,
        cancellation: asyncio.Event,
    ) -> tuple[float, ...]: ...


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GeoCLIPHierarchicalConfig(_FrozenModel):
    version: Literal["geoclip-hierarchical-v1"] = "geoclip-hierarchical-v1"
    global_grid_points: int = Field(default=2_048, ge=256, le=16_384)
    global_modes: int = Field(default=12, ge=2, le=32)
    catalogue_modes: int = Field(default=24, ge=4, le=64)
    refinement_levels_km: tuple[float, ...] = Field(
        default=(900.0, 300.0, 90.0), min_length=1, max_length=4
    )
    refinement_anchors: int = Field(default=8, ge=2, le=16)
    refinement_bearings: int = Field(default=8, ge=4, le=16)
    turkiye_refinement_min_signals: int = Field(default=2, ge=2, le=4)
    turkiye_province_anchors: int = Field(default=12, ge=4, le=24)
    output_candidates: int = Field(default=32, ge=5, le=64)
    output_diversity_km: float = Field(default=40.0, ge=1, le=500)
    timeout_seconds: float = Field(default=60.0, gt=0, le=180)

    @field_validator("refinement_levels_km")
    @classmethod
    def descending_levels(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(item) or item <= 0 for item in value):
            raise ValueError("refinement radii must be finite and positive")
        if tuple(sorted(value, reverse=True)) != value:
            raise ValueError("refinement radii must be coarse-to-fine")
        return value


class HierarchicalSearchCandidate(_FrozenModel):
    candidate_id: str = Field(pattern=r"^hier-[a-f0-9]{20}$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    search_level: Literal["global", "catalogue", "regional_refinement", "turkiye_refinement"]
    grid_resolution_km: float = Field(gt=0)
    provider_score: float
    provider_rank: int = Field(ge=1, le=64)
    diversity_cluster: str = Field(pattern=r"^mode-[0-9]{2}$")
    nearest_name: str | None = Field(default=None, max_length=160)
    nearest_kind: str | None = Field(default=None, max_length=40)
    nearest_distance_km: float | None = Field(default=None, ge=0)

    @field_validator("provider_score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("provider score must be finite")
        return value


class HierarchicalSearchResult(_FrozenModel):
    provider: Literal["geoclip_hierarchical_search"] = "geoclip_hierarchical_search"
    provider_revision: Literal["geoclip-hierarchical-v1"] = "geoclip-hierarchical-v1"
    model_id: Literal["GeoCLIP-1.2.0"] = "GeoCLIP-1.2.0"
    score_semantics: Literal["raw_cosine_similarity_not_confidence"] = (
        "raw_cosine_similarity_not_confidence"
    )
    catalogue_version: str
    global_grid_points_scored: int = Field(ge=256, le=16_384)
    catalogue_points_scored: int = Field(ge=400, le=2_000)
    turkiye_provinces_scored: Literal[81] = 81
    turkiye_refinement_triggered: bool
    trigger_signal_count: int = Field(ge=0, le=8)
    candidates: tuple[HierarchicalSearchCandidate, ...] = Field(min_length=1, max_length=64)
    duration_ms: int = Field(ge=0)
    cache_keys: tuple[str, ...] = Field(min_length=2, max_length=12)
    warnings: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def ranks_are_contiguous(self) -> HierarchicalSearchResult:
        if [item.provider_rank for item in self.candidates] != list(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("hierarchical candidate ranks must be contiguous")
        return self


@dataclass(frozen=True, slots=True)
class _SearchEntry:
    point: ScoredPoint
    catalogue: CoordinateCatalogueRecord | None = None


class GeoCLIPHierarchicalSearchProvider:
    """Bounded, diversity-preserving arbitrary-coordinate search using GeoCLIP."""

    def __init__(
        self,
        scorer: CoordinateScorer,
        catalogue: CoordinateCatalogue,
        config: GeoCLIPHierarchicalConfig | None = None,
    ) -> None:
        self._scorer = scorer
        self._catalogue = catalogue
        self._config = config or GeoCLIPHierarchicalConfig()
        self._global_grid = fibonacci_global_grid(self._config.global_grid_points)

    async def search(
        self,
        image_bytes: bytes,
        *,
        turkiye_signal_count: int = 0,
        cancellation: asyncio.Event | None = None,
    ) -> HierarchicalSearchResult:
        if not image_bytes:
            raise ValueError("hierarchical search requires image bytes")
        if not 0 <= turkiye_signal_count <= 8:
            raise ValueError("Türkiye signal count is outside the bounded range")
        stopped = cancellation or asyncio.Event()
        return await asyncio.wait_for(
            self._search(
                image_bytes,
                turkiye_signal_count=turkiye_signal_count,
                cancellation=stopped,
            ),
            timeout=self._config.timeout_seconds,
        )

    async def _search(
        self,
        image_bytes: bytes,
        *,
        turkiye_signal_count: int,
        cancellation: asyncio.Event,
    ) -> HierarchicalSearchResult:
        started = time.monotonic()
        cache_keys: list[str] = []
        global_resolution = approximate_grid_resolution_km(len(self._global_grid))
        global_key = f"global-fibonacci-{len(self._global_grid)}-v1"
        global_scores = await self._score(
            image_bytes, self._global_grid, cache_key=global_key, cancellation=cancellation
        )
        cache_keys.append(global_key)
        global_points = tuple(
            ScoredPoint(lat, lon, score, f"global-{index}", "global", global_resolution)
            for index, ((lat, lon), score) in enumerate(
                zip(self._global_grid, global_scores, strict=True)
            )
        )
        global_modes = select_geographically_diverse(
            global_points,
            limit=self._config.global_modes,
            minimum_distance_km=max(600.0, global_resolution * 1.5),
        )

        catalogue_coordinates = tuple(
            (item.latitude, item.longitude) for item in self._catalogue.records
        )
        catalogue_key = f"catalogue-{self._catalogue.catalogue_version}"
        catalogue_scores = await self._score(
            image_bytes,
            catalogue_coordinates,
            cache_key=catalogue_key,
            cancellation=cancellation,
        )
        cache_keys.append(catalogue_key)
        catalogue_entries = tuple(
            _SearchEntry(
                ScoredPoint(
                    item.latitude,
                    item.longitude,
                    score,
                    item.id,
                    "catalogue",
                    100.0 if item.kind == "populated_place" else 250.0,
                ),
                item,
            )
            for item, score in zip(self._catalogue.records, catalogue_scores, strict=True)
        )
        catalogue_modes = select_geographically_diverse(
            tuple(item.point for item in catalogue_entries),
            limit=self._config.catalogue_modes,
            minimum_distance_km=150.0,
        )
        by_source = {item.point.source_id: item for item in catalogue_entries}

        pool: list[_SearchEntry] = [*(_SearchEntry(item) for item in global_modes)]
        pool.extend(by_source[item.source_id] for item in catalogue_modes)
        anchors = select_geographically_diverse(
            tuple(item.point for item in pool),
            limit=self._config.refinement_anchors,
            minimum_distance_km=300.0,
        )
        for level_index, radius_km in enumerate(self._config.refinement_levels_km, start=1):
            coordinates = geodesic_refinement_points(
                tuple((item.latitude, item.longitude) for item in anchors),
                radius_km=radius_km,
                bearings=self._config.refinement_bearings,
            )
            key = self._coordinate_cache_key(f"regional-{level_index}", coordinates)
            scores = await self._score(
                image_bytes, coordinates, cache_key=key, cancellation=cancellation
            )
            cache_keys.append(key)
            level_points = tuple(
                ScoredPoint(
                    latitude,
                    longitude,
                    score,
                    f"regional-{level_index}-{index}",
                    "regional_refinement",
                    radius_km,
                )
                for index, ((latitude, longitude), score) in enumerate(
                    zip(coordinates, scores, strict=True)
                )
            )
            selected = select_geographically_diverse(
                level_points,
                limit=self._config.refinement_anchors,
                minimum_distance_km=max(20.0, radius_km * 0.45),
            )
            pool.extend(_SearchEntry(item) for item in selected)
            anchors = selected

        turkiye_triggered = (
            turkiye_signal_count >= self._config.turkiye_refinement_min_signals
        )
        if turkiye_triggered:
            province_entries = sorted(
                (
                    item
                    for item in catalogue_entries
                    if item.catalogue is not None
                    and item.catalogue.kind == "turkiye_province"
                ),
                key=lambda item: (-item.point.score, item.point.source_id),
            )
            province_anchors = province_entries[: self._config.turkiye_province_anchors]
            coordinates = geodesic_refinement_points(
                tuple((item.point.latitude, item.point.longitude) for item in province_anchors),
                radius_km=80.0,
                bearings=self._config.refinement_bearings,
            )
            key = self._coordinate_cache_key("turkiye", coordinates)
            scores = await self._score(
                image_bytes, coordinates, cache_key=key, cancellation=cancellation
            )
            cache_keys.append(key)
            turkey_points = tuple(
                ScoredPoint(
                    latitude,
                    longitude,
                    score,
                    f"turkiye-{index}",
                    "turkiye_refinement",
                    80.0,
                )
                for index, ((latitude, longitude), score) in enumerate(
                    zip(coordinates, scores, strict=True)
                )
            )
            pool.extend(
                _SearchEntry(item)
                for item in select_geographically_diverse(
                    turkey_points,
                    limit=self._config.turkiye_province_anchors,
                    minimum_distance_km=35.0,
                )
            )

        final_points = self._preserve_levels(pool)
        candidates = tuple(
            self._candidate(point, rank=index, catalogue_entries=catalogue_entries)
            for index, point in enumerate(final_points, start=1)
        )
        return HierarchicalSearchResult(
            catalogue_version=self._catalogue.catalogue_version,
            global_grid_points_scored=len(self._global_grid),
            catalogue_points_scored=len(catalogue_coordinates),
            turkiye_refinement_triggered=turkiye_triggered,
            trigger_signal_count=turkiye_signal_count,
            candidates=candidates,
            duration_ms=round((time.monotonic() - started) * 1000),
            cache_keys=tuple(dict.fromkeys(cache_keys)),
        )

    async def _score(
        self,
        image_bytes: bytes,
        coordinates: Sequence[tuple[float, float]],
        *,
        cache_key: str,
        cancellation: asyncio.Event,
    ) -> tuple[float, ...]:
        if cancellation.is_set():
            raise asyncio.CancelledError
        scores = await self._scorer.score_coordinates(
            image_bytes,
            coordinates,
            cache_key=cache_key,
            cancellation=cancellation,
        )
        if len(scores) != len(coordinates) or any(not math.isfinite(item) for item in scores):
            raise ValueError("GeoCLIP coordinate scorer returned invalid output")
        return scores

    def _preserve_levels(self, entries: Sequence[_SearchEntry]) -> tuple[ScoredPoint, ...]:
        level_order = ("catalogue", "global", "regional_refinement", "turkiye_refinement")
        selected: list[ScoredPoint] = []
        for level in level_order:
            level_points = [item.point for item in entries if item.point.level == level]
            preserved = select_geographically_diverse(
                level_points,
                limit=min(4, len(level_points)),
                minimum_distance_km=self._config.output_diversity_km,
            )
            selected.extend(item for item in preserved if item not in selected)
        remaining = sorted(
            (item.point for item in entries if item.point not in selected),
            key=lambda item: (-item.score, item.source_id),
        )
        for point in remaining:
            if len(selected) >= self._config.output_candidates:
                break
            if all(
                geodesic_km(
                    (point.latitude, point.longitude),
                    (other.latitude, other.longitude),
                )
                >= self._config.output_diversity_km
                for other in selected
            ):
                selected.append(point)
        for point in remaining:
            if len(selected) >= self._config.output_candidates:
                break
            if point not in selected:
                selected.append(point)
        bounded = selected[: self._config.output_candidates]
        return tuple(sorted(bounded, key=lambda item: (-item.score, item.source_id)))

    def _candidate(
        self,
        point: ScoredPoint,
        *,
        rank: int,
        catalogue_entries: Sequence[_SearchEntry],
    ) -> HierarchicalSearchCandidate:
        nearest = min(
            (item for item in catalogue_entries if item.catalogue is not None),
            key=lambda item: geodesic_km(
                (point.latitude, point.longitude),
                (item.point.latitude, item.point.longitude),
            ),
        )
        nearest_distance = geodesic_km(
            (point.latitude, point.longitude),
            (nearest.point.latitude, nearest.point.longitude),
        )
        identity = hashlib.sha256(
            f"{point.level}|{point.source_id}|{point.latitude:.7f}|{point.longitude:.7f}".encode()
        ).hexdigest()[:20]
        return HierarchicalSearchCandidate(
            candidate_id=f"hier-{identity}",
            latitude=point.latitude,
            longitude=point.longitude,
            search_level=point.level,
            grid_resolution_km=point.resolution_km,
            provider_score=point.score,
            provider_rank=rank,
            diversity_cluster=f"mode-{rank:02d}",
            nearest_name=nearest.catalogue.name if nearest.catalogue is not None else None,
            nearest_kind=nearest.catalogue.kind if nearest.catalogue is not None else None,
            nearest_distance_km=nearest_distance,
        )

    @staticmethod
    def _coordinate_cache_key(
        prefix: str, coordinates: Sequence[tuple[float, float]]
    ) -> str:
        digest = hashlib.sha256()
        for latitude, longitude in coordinates:
            digest.update(f"{latitude:.7f},{longitude:.7f};".encode())
        return f"{prefix}-{digest.hexdigest()[:20]}"
