from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from atlaslens_api.gazetteer import ResolvedPlace
from atlaslens_api.global_prediction.clustering import (
    GeoClipCandidateCluster,
    GeoClipCandidateClusterer,
)
from atlaslens_api.global_prediction.reverse_geocoding import (
    CachedClusterReverseGeocoder,
    SQLiteReverseGeocodeCache,
)
from atlaslens_api.providers.base import GlobalPredictionHypothesis


def _cluster() -> GeoClipCandidateCluster:
    hypothesis = GlobalPredictionHypothesis(
        rank=1,
        original_rank=1,
        latitude=40.0,
        longitude=29.0,
        raw_score=0.9,
        score_type="uncalibrated_gallery_softmax",
        normalization_method="softmax_over_fixed_gallery",
        calibration_state="uncalibrated",
        limitations=["test.relative_only"],
    )
    return GeoClipCandidateClusterer().cluster([hypothesis])[0]


class CountingResolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, latitude: float, longitude: float) -> ResolvedPlace:
        del latitude, longitude
        self.calls += 1
        return ResolvedPlace(
            label="Reviewed Place",
            country_code="TR",
            country="Reviewed Country",
            region="Reviewed Region",
            city="Reviewed City",
            distance_km=1.0,
            source="installed-gazetteer",
            dataset_version="test-v1",
            license="test-license",
        )


@pytest.mark.asyncio
async def test_reverse_geocoding_is_bounded_and_persistently_cached(tmp_path: Path) -> None:
    resolver = CountingResolver()
    path = tmp_path / "reverse-cache.sqlite3"
    service = CachedClusterReverseGeocoder(resolver, SQLiteReverseGeocodeCache(path), top_k=1)
    first = await service.enrich([_cluster()])
    assert first[0].place is not None
    assert resolver.calls == 1

    second_service = CachedClusterReverseGeocoder(
        resolver, SQLiteReverseGeocodeCache(path), top_k=1
    )
    second = await second_service.enrich([_cluster()])
    assert second[0].place is not None
    assert resolver.calls == 1


@pytest.mark.asyncio
async def test_unavailable_reverse_geocoder_keeps_raw_cluster(tmp_path: Path) -> None:
    result = await CachedClusterReverseGeocoder(
        None, SQLiteReverseGeocodeCache(tmp_path / "cache.sqlite3"), top_k=1
    ).enrich([_cluster()])
    assert result[0].cluster.summary.source == "geoclip"
    assert result[0].place is None
    assert result[0].warning == "provider.reverse_geocoding.unavailable"


@pytest.mark.asyncio
async def test_locked_cache_degrades_without_losing_resolver_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver = CountingResolver()
    cache = SQLiteReverseGeocodeCache(tmp_path / "locked-cache.sqlite3")

    async def locked_initialize() -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cache, "initialize", locked_initialize)
    result = await CachedClusterReverseGeocoder(resolver, cache, top_k=1).enrich([_cluster()])

    assert result[0].place is not None
    assert resolver.calls == 1
    assert result[0].warning == "provider.reverse_geocoding.cache_unavailable"
