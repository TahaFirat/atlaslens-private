from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from atlaslens_api.phase6c import (
    GeoCLIPHierarchicalConfig,
    GeoCLIPHierarchicalSearchProvider,
    load_coordinate_catalogue,
)
from atlaslens_api.phase6c.grid import (
    destination_point,
    fibonacci_global_grid,
    geodesic_refinement_points,
)
from atlaslens_api.providers.geoclip import GeoCLIPGlobalGeolocationProvider
from atlaslens_api.reranking.geo import geodesic_km
from test_model_management import build_installed_management


class DistanceScorer:
    def __init__(self, target: tuple[float, float]) -> None:
        self.target = target
        self.calls: list[tuple[str, int]] = []

    async def score_coordinates(
        self,
        _image_bytes: bytes,
        coordinates: Sequence[tuple[float, float]],
        *,
        cache_key: str,
        cancellation: asyncio.Event,
    ) -> tuple[float, ...]:
        assert not cancellation.is_set()
        self.calls.append((cache_key, len(coordinates)))
        return tuple(-geodesic_km(self.target, item) / 20_000.0 for item in coordinates)


class CoordinateRuntime:
    device = "cpu"
    dtype = "float32"

    def __init__(self) -> None:
        self.location_encodes = 0

    def predict(self, _image_payload: bytes, top_k: int) -> tuple[list[list[float]], list[float]]:
        return [[float(index), float(index)] for index in range(top_k)], [0.5] * top_k

    def encode_coordinates(
        self, coordinates: Sequence[tuple[float, float]]
    ) -> tuple[tuple[float, float], ...]:
        self.location_encodes += 1
        return tuple(coordinates)

    def score_coordinate_features(
        self,
        _image_payload: bytes,
        location_features: Sequence[tuple[float, float]],
    ) -> list[float]:
        return [(latitude + longitude) / 360.0 for latitude, longitude in location_features]


def test_checked_in_catalogue_has_global_and_all_turkiye_province_coverage() -> None:
    catalogue = load_coordinate_catalogue()
    provinces = catalogue.records_of_kind("turkiye_province")
    assert len(provinces) == 81
    assert len({item.admin1_code for item in provinces}) == 81
    assert len(catalogue.records_of_kind("country")) >= 170
    assert len(catalogue.records_of_kind("populated_place")) >= 200
    assert {item.country_code for item in provinces} == {"TR"}


def test_fibonacci_grid_is_global_equal_area_and_refinement_is_geodesic() -> None:
    grid = fibonacci_global_grid(2_048)
    assert len(grid) == 2_048
    assert max(latitude for latitude, _ in grid) > 88
    assert min(latitude for latitude, _ in grid) < -88
    assert min(longitude for _, longitude in grid) < -179
    assert max(longitude for _, longitude in grid) > 179

    across_dateline = destination_point((10.0, 179.9), bearing_degrees=90, distance_km=50)
    near_pole = destination_point((89.9, 0.0), bearing_degrees=0, distance_km=50)
    assert -180 <= across_dateline[1] <= 180
    assert -90 <= near_pole[0] <= 90
    points = geodesic_refinement_points(((89.9, 179.9),), radius_km=25, bearings=8)
    assert all(-90 <= latitude <= 90 and -180 <= longitude <= 180 for latitude, longitude in points)


@pytest.mark.asyncio
async def test_hierarchical_search_scores_every_province_and_preserves_modes() -> None:
    scorer = DistanceScorer((39.0, 35.0))
    provider = GeoCLIPHierarchicalSearchProvider(
        scorer,
        load_coordinate_catalogue(),
        GeoCLIPHierarchicalConfig(
            global_grid_points=256,
            global_modes=6,
            catalogue_modes=12,
            refinement_levels_km=(400.0, 100.0),
            refinement_anchors=4,
            turkiye_province_anchors=4,
            output_candidates=16,
        ),
    )
    result = await provider.search(b"private-image", turkiye_signal_count=2)
    assert result.turkiye_provinces_scored == 81
    assert result.catalogue_points_scored == len(load_coordinate_catalogue().records)
    assert result.turkiye_refinement_triggered is True
    assert [item.provider_rank for item in result.candidates] == list(range(1, 17))
    assert {item.search_level for item in result.candidates} >= {
        "catalogue",
        "global",
        "regional_refinement",
        "turkiye_refinement",
    }
    assert any(key.startswith("catalogue-") for key, _ in scorer.calls)
    catalogue_call = next(size for key, size in scorer.calls if key.startswith("catalogue-"))
    assert catalogue_call == result.catalogue_points_scored


@pytest.mark.asyncio
async def test_geoclip_location_embedding_cache_is_digest_bound(tmp_path: Path) -> None:
    management = build_installed_management(tmp_path)
    runtime = CoordinateRuntime()
    provider = GeoCLIPGlobalGeolocationProvider(
        management,
        model_loader=lambda _snapshot, _assets, _device: runtime,
        device_selector=lambda: "cpu",
    )
    cancellation = asyncio.Event()
    coordinates = ((10.0, 20.0), (30.0, 40.0))
    first = await provider.score_coordinates(
        b"private", coordinates, cache_key="static-test-v1", cancellation=cancellation
    )
    second = await provider.score_coordinates(
        b"private", coordinates, cache_key="static-test-v1", cancellation=cancellation
    )
    assert first == second
    assert runtime.location_encodes == 1
    diagnostics = provider.diagnostics()
    assert diagnostics["coordinate_embedding_cache_hits"] == 1
    assert diagnostics["coordinate_embedding_cache_misses"] == 1

    await provider.score_coordinates(
        b"private",
        ((10.0, 20.0), (31.0, 41.0)),
        cache_key="static-test-v1",
        cancellation=cancellation,
    )
    assert runtime.location_encodes == 2


def test_production_hierarchical_module_contains_no_holdout_specific_branch() -> None:
    import atlaslens_api.phase6c.hierarchical as module

    source = inspect.getsource(module).casefold()
    forbidden = (
        "user-holdout-001",
        "private-operator-image.example.jpg",
        "province id 38",
        "talas",
        "erciyes",
    )
    assert all(item not in source for item in forbidden)
    assert "kayseri" not in source


def test_grid_latitudes_are_not_naive_rectangular_rows() -> None:
    grid = fibonacci_global_grid(512)
    latitudes = [round(latitude, 8) for latitude, _ in grid]
    assert len(set(latitudes)) == len(latitudes)
    sine_steps = [
        math.sin(math.radians(latitudes[index])) - math.sin(math.radians(latitudes[index + 1]))
        for index in range(len(latitudes) - 1)
    ]
    assert max(sine_steps) - min(sine_steps) < 1e-6
