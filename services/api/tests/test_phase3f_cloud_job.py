from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from atlaslens_api.mapillary_demo.client import RemoteImage
from atlaslens_api.mapillary_demo.models import GeoPoint, ImageMetadata
from atlaslens_api.phase3f.cloud_job import (
    MetadataAsset,
    MetadataAudit,
    load_city_areas,
    plan_locked_roles,
)
from atlaslens_api.phase3f.coverage import (
    CANDIDATE_CITY_REGIONS,
    CityCoverageRecord,
    CoverageSelectionLock,
)
from atlaslens_api.phase3f.pipeline import Phase3FPipeline, PipelineStage

ROOT = Path(__file__).parents[3]
AOI_CONFIG = ROOT / "config" / "phase3f" / "city-coverage-aoi-v1.json"


def _record(city: str) -> CityCoverageRecord:
    return CityCoverageRecord(
        city=city,
        macro_region=CANDIDATE_CITY_REGIONS[city],
        image_count=225,
        sequence_count=225,
        contributor_count=3,
        spatial_cell_count=20,
        capture_year_count=5,
        eligible_asset_count=225,
        metadata_sha256="a" * 64,
    )


def _assets(
    city: str,
    longitude: float,
    latitude: float,
    *,
    creator_count: int,
) -> tuple[MetadataAsset, ...]:
    rows: list[MetadataAsset] = []
    per_creator = 75 if creator_count == 3 else 40
    offsets = (-0.04, 0.0, 0.04) if creator_count == 3 else (0.0,)
    for creator_index, offset in enumerate(offsets):
        for index in range(per_creator):
            image_id = f"{city.encode().hex()[:16]}-{creator_index}-{index}"
            metadata = ImageMetadata(
                mapillary_image_id=image_id,
                computed_geometry=GeoPoint(
                    coordinates=(longitude + offset, latitude + index * 0.000001)
                ),
                captured_at=datetime(2020 + index % 5, 1, 1, tzinfo=UTC),
                sequence_id=f"{city}-sequence-{creator_index}-{index}",
                creator_id=f"{city}-creator-{creator_index}",
                width_px=1024,
                height_px=768,
            )
            rows.append(MetadataAsset(city, RemoteImage(metadata)))
    return tuple(rows)


def test_aoi_config_covers_exact_candidate_pool_with_bounded_tiles() -> None:
    areas = load_city_areas(AOI_CONFIG)

    assert {area.city for area in areas} == set(CANDIDATE_CITY_REGIONS)
    assert len(areas) == 16
    assert all(len(area.boxes) == 4 for area in areas)
    assert all(box.area_square_degrees <= 0.01 for area in areas for box in area.boxes)


def test_role_planner_is_deterministic_and_globally_disjoint() -> None:
    in_domain = ("İstanbul", "İzmir", "Antalya", "Ankara", "Trabzon", "Erzurum")
    ood = ("Bursa", "Gaziantep")
    selection = CoverageSelectionLock(
        in_domain=tuple(_record(city) for city in in_domain),
        ood=tuple(_record(city) for city in ood),
        audited_city_count=16,
        policy_sha256="b" * 64,
    )
    areas = {area.city: area for area in load_city_areas(AOI_CONFIG)}
    assets_by_city = {
        city: _assets(
            city,
            areas[city].longitude,
            areas[city].latitude,
            creator_count=3 if city in in_domain else 1,
        )
        for city in (*in_domain, *ood)
    }
    audit = MetadataAudit(selection, assets_by_city, request_count=64, rejected_item_count=0)

    first = plan_locked_roles(audit)
    second = plan_locked_roles(audit)

    assert first == second
    assert len(first) == 6 * 125 + 2 * 40
    counts = {
        (city, role): sum(
            item.metadata.city == city and item.role == role for item in first
        )
        for city in (*in_domain, *ood)
        for role in ("reference", "calibration", "sealed_holdout", "ood_holdout")
    }
    assert all(counts[(city, "reference")] == 75 for city in in_domain)
    assert all(counts[(city, "calibration")] == 25 for city in in_domain)
    assert all(counts[(city, "sealed_holdout")] == 25 for city in in_domain)
    assert all(counts[(city, "ood_holdout")] == 40 for city in ood)
    roles_by_creator: dict[str, set[str]] = {}
    roles_by_sequence: dict[str, set[str]] = {}
    for item in first:
        assert item.metadata.creator_id is not None
        assert item.metadata.sequence_id is not None
        roles_by_creator.setdefault(item.metadata.creator_id, set()).add(item.role)
        roles_by_sequence.setdefault(item.metadata.sequence_id, set()).add(item.role)
    assert all(len(roles) == 1 for roles in roles_by_creator.values())
    assert all(len(roles) == 1 for roles in roles_by_sequence.values())


def test_coverage_insufficient_is_a_terminal_resumable_state(tmp_path: Path) -> None:
    state = tmp_path / "phase3f-state.json"
    pipeline = Phase3FPipeline.create(state, run_id="c" * 32)

    pipeline.finalize_coverage_insufficient(reason_code="COVERAGE_INSUFFICIENT")

    resumed = Phase3FPipeline.resume(state)
    assert resumed.stage is PipelineStage.COVERAGE_INSUFFICIENT
    assert resumed.document["image_acquisition_started"] is False
    assert resumed.document["outcome"] == "COVERAGE_INSUFFICIENT"
