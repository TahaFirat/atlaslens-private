from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from atlaslens_api.mapillary_demo.client import RemoteImage
from atlaslens_api.mapillary_demo.models import BoundingBox, GeoPoint, ImageMetadata
from atlaslens_api.phase3f import cloud_job as worker
from atlaslens_api.phase3f.acquisition import AcquisitionGuard
from atlaslens_api.phase3f.cloud_job import (
    MEDIA_PLAN_REVISION_SCHEMA,
    MEDIA_REPLENISHMENT_MAX_ROUNDS,
    MEDIA_REPLENISHMENT_MAX_TOTAL,
    MediaSplitPlan,
    MetadataAsset,
    MetadataAudit,
    PlannedAsset,
    acquire_replenished_media,
    finalize_media_split,
    finalize_provenance_aggregate,
    load_city_areas,
    load_media_plan_revision,
    plan_locked_roles,
    plan_metadata_split,
    quarter_bbox,
    replenish_media_reserves,
)
from atlaslens_api.phase3f.coverage import (
    CANDIDATE_CITY_REGIONS,
    CityCoverageRecord,
    CoverageSelectionLock,
)
from atlaslens_api.phase3f.pipeline import Phase3FPipeline, PipelineStage
from atlaslens_api.phase3f.splits import SplitAsset, audit_leakage

ROOT = Path(__file__).parents[3]
AOI_CONFIG = ROOT / "config" / "phase3f" / "city-coverage-aoi-v1.json"
_CANDIDATE_ORDER = tuple(CANDIDATE_CITY_REGIONS)
_TEST_IN_DOMAIN = tuple(_CANDIDATE_ORDER[index] for index in (0, 2, 4, 1, 9, 11))
_TEST_OOD = tuple(_CANDIDATE_ORDER[index] for index in (3, 7))


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


def _assets_with_creator_sizes(
    city: str,
    longitude: float,
    latitude: float,
    sizes: tuple[int, ...],
    *,
    colocated: bool = False,
) -> tuple[MetadataAsset, ...]:
    rows: list[MetadataAsset] = []
    midpoint = (len(sizes) - 1) / 2
    for creator_index, size in enumerate(sizes):
        offset = 0.0 if colocated else (creator_index - midpoint) * 0.05
        for index in range(size):
            image_id = f"{city.encode().hex()[:16]}-s{creator_index}-{index}"
            metadata = ImageMetadata(
                mapillary_image_id=image_id,
                computed_geometry=GeoPoint(
                    coordinates=(longitude + offset, latitude + index * 0.000001)
                ),
                captured_at=datetime(2020 + index % 5, 1, 1, tzinfo=UTC),
                sequence_id=f"{city}-s{creator_index}-sequence-{index // 5}",
                creator_id=f"{city}-s{creator_index}-creator",
                width_px=1024,
                height_px=768,
            )
            rows.append(MetadataAsset(city, RemoteImage(metadata)))
    return tuple(rows)


def _split_asset(item: PlannedAsset, *, duplicate: SplitAsset | None = None) -> SplitAsset:
    planned = item
    metadata = planned.metadata
    image_id = metadata.image_id
    opaque_id = hashlib.sha256(f"opaque:{image_id}".encode()).hexdigest()[:32]
    longitude, latitude = metadata.remote.metadata.computed_geometry.coordinates
    return SplitAsset(
        opaque_id=opaque_id,
        city=metadata.city,
        role=planned.role,
        relative_path=f"assets/{opaque_id[:2]}/{opaque_id}.jpg",
        contributor_id=metadata.creator_id or "missing",
        sequence_id=metadata.sequence_id or "missing",
        capture_run_id=metadata.sequence_id or "missing",
        content_sha256=(
            duplicate.content_sha256
            if duplicate is not None
            else hashlib.sha256(f"content:{image_id}".encode()).hexdigest()
        ),
        perceptual_hash=(
            duplicate.perceptual_hash
            if duplicate is not None
            else hashlib.sha256(f"phash:{image_id}".encode()).hexdigest()[:16]
        ),
        parent_or_tile_id=image_id,
        longitude=longitude,
        latitude=latitude,
    )


def _sized_audit(
    *,
    in_domain_sizes: tuple[int, ...],
    ood_sizes: tuple[int, ...] = (50,),
    colocated_city: str | None = None,
) -> MetadataAudit:
    selection = CoverageSelectionLock(
        in_domain=tuple(_record(city) for city in _TEST_IN_DOMAIN),
        ood=tuple(_record(city) for city in _TEST_OOD),
        audited_city_count=16,
        policy_sha256="b" * 64,
    )
    areas = {area.city: area for area in load_city_areas(AOI_CONFIG)}
    assets_by_city = {
        city: _assets_with_creator_sizes(
            city,
            areas[city].longitude,
            areas[city].latitude,
            in_domain_sizes if city in _TEST_IN_DOMAIN else ood_sizes,
            colocated=city == colocated_city,
        )
        for city in (*_TEST_IN_DOMAIN, *_TEST_OOD)
    }
    return MetadataAudit(selection, assets_by_city, request_count=64, rejected_item_count=0)


def _write_task_ledger(
    path: Path,
    plan: tuple[PlannedAsset, ...],
    states: dict[str, str],
    *,
    run_id: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "atlaslens-phase3f-media-task-ledger-v2",
                "run_id": run_id,
                "tasks": [
                    {
                        "image_id": item.metadata.image_id,
                        "city": item.metadata.city,
                        "role": item.role,
                        "state": states.get(item.metadata.image_id, "REJECTED"),
                    }
                    for item in plan
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _media_deficit(city: str, role: str, deficit: int) -> dict[str, object]:
    return {
        "failed_constraints": [
            {
                "constraint": "media_city_role_minimum_after_decode_and_dedup",
                "city": city,
                "role": role,
                "required_records": 75,
                "available_records": 75 - deficit,
                "deficit_records": deficit,
            }
        ]
    }


def test_aoi_config_covers_exact_candidate_pool_with_bounded_tiles() -> None:
    areas = load_city_areas(AOI_CONFIG)

    assert {area.city for area in areas} == set(CANDIDATE_CITY_REGIONS)
    assert len(areas) == 16
    assert all(len(area.boxes) == 16 for area in areas)
    assert all(box.area_square_degrees <= 0.01 for area in areas for box in area.boxes)


def test_city_bbox_quarters_have_stable_southwest_first_order() -> None:
    parent = BoundingBox(west=28.9, south=40.9, east=28.95, north=40.95)
    cells = quarter_bbox(parent)

    assert [cell.as_query_value() for cell in cells] == [
        "28.9000000,40.9000000,28.9250000,40.9250000",
        "28.9250000,40.9000000,28.9500000,40.9250000",
        "28.9000000,40.9250000,28.9250000,40.9500000",
        "28.9250000,40.9250000,28.9500000,40.9500000",
    ]
    expected_area = pytest.approx(parent.area_square_degrees / 4)
    assert all(cell.area_square_degrees == expected_area for cell in cells)


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
        (city, role): sum(item.metadata.city == city and item.role == role for item in first)
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


def test_constraint_solver_handles_feasible_corpus_that_consumptive_greedy_misses() -> None:
    audit = _sized_audit(in_domain_sizes=(100, 25, 25), ood_sizes=(40,))

    first = plan_metadata_split(audit)
    second = plan_metadata_split(audit)

    assert first.ready is True
    assert first == second
    assert len(first.primary) == 6 * 125 + 2 * 40
    roles_by_creator: dict[str, set[str]] = {}
    roles_by_sequence: dict[str, set[str]] = {}
    for item in first.download_assets:
        assert item.metadata.creator_id is not None
        assert item.metadata.sequence_id is not None
        roles_by_creator.setdefault(item.metadata.creator_id, set()).add(item.role)
        roles_by_sequence.setdefault(item.metadata.sequence_id, set()).add(item.role)
    assert all(len(roles) == 1 for roles in roles_by_creator.values())
    assert all(len(roles) == 1 for roles in roles_by_sequence.values())


def test_genuinely_infeasible_corpus_gets_bounded_repeatable_supplemental_plan() -> None:
    audit = _sized_audit(in_domain_sizes=(100, 24), ood_sizes=(40,))

    first = plan_metadata_split(audit)
    second = plan_metadata_split(audit)

    assert first.ready is False
    assert first.readiness == second.readiness
    assert first.readiness["next_automatic_action"] == "ACQUIRE_BOUNDED_TARGETED_METADATA"
    supplemental = first.readiness["supplemental_acquisition"]
    assert isinstance(supplemental, dict)
    assert supplemental["request_cap"] == 512
    assert supplemental["metadata_record_cap"] == 600
    assert supplemental["media_byte_cap"] == 0
    assert supplemental["query_completed_cells"] is False
    assert supplemental["provider_failures_quarantined"] is True
    assert json.dumps(first.readiness, sort_keys=True) == json.dumps(
        second.readiness, sort_keys=True
    )


def test_spatial_conflict_is_reported_without_weakening_the_distance_gate() -> None:
    audit = _sized_audit(
        in_domain_sizes=(100, 40, 40),
        colocated_city=_TEST_IN_DOMAIN[0],
    )

    plan = plan_metadata_split(audit)

    assert plan.ready is False
    assert any(
        row["constraint"] == "spatial_separation_allocation"
        for row in plan.readiness["failed_constraints"]
    )
    isolation = plan.readiness["isolation"]
    assert isinstance(isolation, dict)
    assert isolation["spatial_exclusion_meters"] == 1_000.0


def test_shared_sequence_connects_contributors_into_one_split_role() -> None:
    audit = _sized_audit(in_domain_sizes=(100, 40, 40, 40), ood_sizes=(50,))
    city = _TEST_IN_DOMAIN[0]
    rows = list(audit.assets_by_city[city])
    linked_creators = {
        rows[100].creator_id,
        rows[140].creator_id,
    }
    for index in (100, 140):
        metadata = rows[index].remote.metadata.model_copy(
            update={"sequence_id": "shared-sequence-isolation-edge"}
        )
        rows[index] = MetadataAsset(city, RemoteImage(metadata))
    assets_by_city = dict(audit.assets_by_city)
    assets_by_city[city] = tuple(rows)
    linked_audit = MetadataAudit(
        audit.selection,
        assets_by_city,
        audit.request_count,
        audit.rejected_item_count,
    )

    plan = plan_metadata_split(linked_audit)

    assert plan.ready is True
    linked_roles = {
        item.role for item in plan.download_assets if item.metadata.creator_id in linked_creators
    }
    assert len(linked_roles) == 1


def test_media_failures_and_cross_role_duplicates_are_replaced_from_reserves() -> None:
    audit = _sized_audit(in_domain_sizes=(100, 40, 40), ood_sizes=(50,))
    plan = plan_metadata_split(audit)
    assert plan.ready is True
    assert plan.reserves
    primary_by_role = {
        role: next(item for item in plan.primary if item.role == role)
        for role in ("calibration", "sealed_holdout", "ood_holdout")
    }
    duplicate_source = _split_asset(primary_by_role["calibration"])
    duplicate_target_id = primary_by_role["ood_holdout"].metadata.image_id
    omitted_id = primary_by_role["sealed_holdout"].metadata.image_id
    downloaded = tuple(
        _split_asset(
            item,
            duplicate=duplicate_source if item.metadata.image_id == duplicate_target_id else None,
        )
        for item in plan.download_assets
        if item.metadata.image_id != omitted_id
    )

    media = finalize_media_split(plan, downloaded)

    assert media.ready is True
    assert len(media.assets) == 6 * 125 + 2 * 40
    leakage = audit_leakage(media.assets, spatial_exclusion_meters=1_000.0)
    assert leakage.passed is True
    duplicate_roles = {
        item.role for item in media.assets if item.content_sha256 == duplicate_source.content_sha256
    }
    assert len(duplicate_roles) == 1
    assert all(item.parent_or_tile_id != omitted_id for item in media.assets)


def test_three_missing_media_selects_deterministic_unused_same_bucket_reserves(
    tmp_path: Path,
) -> None:
    audit = _sized_audit(in_domain_sizes=(100, 40, 40), ood_sizes=(50,))
    base_plan = plan_metadata_split(audit)
    assert base_plan.ready is True
    city = _TEST_IN_DOMAIN[0]
    target_primary = [
        item
        for item in base_plan.primary
        if item.metadata.city == city and item.role == "reference"
    ]
    omitted = {item.metadata.image_id for item in target_primary[-3:]}
    omitted.update(
        item.metadata.image_id
        for item in base_plan.reserves
        if item.metadata.city == city and item.role == "reference"
    )
    accepted_ids = {
        item.metadata.image_id
        for item in base_plan.download_assets
        if item.metadata.image_id not in omitted
    }
    states = {
        item.metadata.image_id: (
            "ACCEPTED" if item.metadata.image_id in accepted_ids else "REJECTED"
        )
        for item in base_plan.download_assets
    }
    run_id = "e" * 32
    ledger = tmp_path / "media-task-ledger.json"
    revision = tmp_path / "media-plan-revisions.json"
    _write_task_ledger(ledger, base_plan.download_assets, states, run_id=run_id)
    ledger_before = ledger.read_bytes()
    state = load_media_plan_revision(audit, base_plan, revision, run_id=run_id)

    first, report = replenish_media_reserves(
        audit,
        state,
        _media_deficit(city, "reference", 3),
        run_id=run_id,
        ledger_path=ledger,
        revision_path=revision,
    )

    assert report["action"] == "ACQUIRE_INCREMENTAL_RESERVES"
    assert report["selected_this_round"] == 12
    assert first.plan.revision == 1
    assert first.plan.incremental_reserve_count == 12
    incremental = first.plan.reserves[len(base_plan.reserves) :]
    assert all(item.metadata.city == city and item.role == "reference" for item in incremental)
    assert not ({item.metadata.image_id for item in incremental} & set(states))
    assert ledger.read_bytes() == ledger_before
    document = json.loads(revision.read_text())
    assert document["schema"] == MEDIA_PLAN_REVISION_SCHEMA
    assert document["revision"] == 1
    assert document["total_added"] == 12
    assert document["bounds"]["max_rounds"] == MEDIA_REPLENISHMENT_MAX_ROUNDS
    assert document["bounds"]["max_total_candidates"] == MEDIA_REPLENISHMENT_MAX_TOTAL

    second_root = tmp_path / "second"
    second_root.mkdir()
    second_ledger = second_root / "media-task-ledger.json"
    second_revision = second_root / "media-plan-revisions.json"
    _write_task_ledger(second_ledger, base_plan.download_assets, states, run_id=run_id)
    repeated, repeated_report = replenish_media_reserves(
        audit,
        load_media_plan_revision(audit, base_plan, second_revision, run_id=run_id),
        _media_deficit(city, "reference", 3),
        run_id=run_id,
        ledger_path=second_ledger,
        revision_path=second_revision,
    )
    assert repeated_report["selected_this_round"] == 12
    assert [item.metadata.image_id for item in repeated.plan.reserves] == [
        item.metadata.image_id for item in first.plan.reserves
    ]

    roles_by_contributor: dict[str, set[str]] = {}
    roles_by_sequence: dict[str, set[str]] = {}
    for item in first.plan.download_assets:
        assert item.metadata.creator_id is not None
        assert item.metadata.sequence_id is not None
        roles_by_contributor.setdefault(item.metadata.creator_id, set()).add(item.role)
        roles_by_sequence.setdefault(item.metadata.sequence_id, set()).add(item.role)
    assert all(len(roles) == 1 for roles in roles_by_contributor.values())
    assert all(len(roles) == 1 for roles in roles_by_sequence.values())


def test_failed_first_replenishment_round_adds_disjoint_second_round(
    tmp_path: Path,
) -> None:
    audit = _sized_audit(in_domain_sizes=(100, 40, 40), ood_sizes=(50,))
    base_plan = plan_metadata_split(audit)
    city = _TEST_IN_DOMAIN[0]
    target = [
        item
        for item in base_plan.primary
        if item.metadata.city == city and item.role == "reference"
    ]
    accepted_ids = {item.metadata.image_id for item in base_plan.download_assets} - {
        item.metadata.image_id for item in target[-3:]
    }
    states = {
        item.metadata.image_id: (
            "ACCEPTED" if item.metadata.image_id in accepted_ids else "REJECTED"
        )
        for item in base_plan.download_assets
    }
    run_id = "f" * 32
    ledger = tmp_path / "media-task-ledger.json"
    revision = tmp_path / "media-plan-revisions.json"
    _write_task_ledger(ledger, base_plan.download_assets, states, run_id=run_id)
    first, _report = replenish_media_reserves(
        audit,
        load_media_plan_revision(audit, base_plan, revision, run_id=run_id),
        _media_deficit(city, "reference", 3),
        run_id=run_id,
        ledger_path=ledger,
        revision_path=revision,
    )
    first_ids = {item.metadata.image_id for item in first.plan.reserves[len(base_plan.reserves) :]}
    failed_states = dict(states)
    failed_states.update({image_id: "REJECTED" for image_id in first_ids})
    _write_task_ledger(ledger, first.plan.download_assets, failed_states, run_id=run_id)

    second, report = replenish_media_reserves(
        audit,
        load_media_plan_revision(audit, base_plan, revision, run_id=run_id),
        _media_deficit(city, "reference", 3),
        run_id=run_id,
        ledger_path=ledger,
        revision_path=revision,
    )
    all_incremental = second.plan.reserves[len(base_plan.reserves) :]
    second_ids = {
        item.metadata.image_id
        for item in all_incremental
        if item.metadata.image_id not in first_ids
    }
    assert report["action"] == "ACQUIRE_INCREMENTAL_RESERVES"
    assert second.plan.revision == 2
    assert first_ids
    assert second_ids
    assert first_ids.isdisjoint(second_ids)


def test_acquisition_loop_automatically_completes_on_second_replenishment_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _sized_audit(in_domain_sizes=(100, 40, 40), ood_sizes=(50,))
    base_plan = plan_metadata_split(audit)
    city = _TEST_IN_DOMAIN[0]
    target = [
        item
        for item in base_plan.primary
        if item.metadata.city == city and item.role == "reference"
    ]
    states = {item.metadata.image_id: "ACCEPTED" for item in base_plan.download_assets}
    for item in target[-3:]:
        states[item.metadata.image_id] = "REJECTED"
    for item in base_plan.reserves:
        if item.metadata.city == city and item.role == "reference":
            states[item.metadata.image_id] = "REJECTED"
    run_id = "3" * 32
    checkpoint = tmp_path / "acquisition-checkpoint.json"
    ledger = tmp_path / "media-task-ledger.json"
    revision = tmp_path / "media-plan-revisions.json"
    _write_task_ledger(ledger, base_plan.download_assets, states, run_id=run_id)
    calls = 0

    def fake_acquire(
        _client: object,
        _areas: object,
        planned: tuple[PlannedAsset, ...],
        *_args: object,
        **_kwargs: object,
    ) -> tuple[tuple[SplitAsset, ...], dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            new_ids = [
                item.metadata.image_id for item in planned if item.metadata.image_id not in states
            ]
            states.update({image_id: "REJECTED" for image_id in new_ids})
            states.update({image_id: "ACCEPTED" for image_id in new_ids[:2]})
        elif calls == 3:
            new_ids = [
                item.metadata.image_id for item in planned if item.metadata.image_id not in states
            ]
            states.update({image_id: "REJECTED" for image_id in new_ids})
            if new_ids:
                states[new_ids[0]] = "ACCEPTED"
        _write_task_ledger(ledger, planned, states, run_id=run_id)
        return (), {"media_status": "MEDIA_SPLIT_MINIMUM_UNAVAILABLE"}

    def fake_finalize(
        _plan: object,
        _assets: object,
    ) -> MediaSplitPlan:
        if calls == 1:
            return MediaSplitPlan((), _media_deficit(city, "reference", 3))
        if calls == 2:
            return MediaSplitPlan((), _media_deficit(city, "reference", 1))
        return MediaSplitPlan((), {"ready": True})

    monkeypatch.setattr(worker, "acquire_planned_assets", fake_acquire)
    monkeypatch.setattr(worker, "finalize_media_split", fake_finalize)

    _assets, _provenance, media_plan, report = acquire_replenished_media(
        object(),  # type: ignore[arg-type]
        (),
        audit,
        base_plan,
        tmp_path / "media",
        run_id,
        AcquisitionGuard(),
        checkpoint,
        revision,
    )

    assert calls == 3
    assert media_plan.ready is True
    assert report is not None
    assert report["revision"] == 2
    assert json.loads(revision.read_text())["revision"] == 2


def test_media_replenishment_reports_true_corpus_exhaustion(tmp_path: Path) -> None:
    audit = _sized_audit(in_domain_sizes=(85, 25, 25), ood_sizes=(40,))
    base_plan = plan_metadata_split(audit)
    assert base_plan.ready is True
    assert {item.metadata.image_id for item in base_plan.download_assets} == {
        item.image_id for rows in audit.assets_by_city.values() for item in rows
    }
    city = _TEST_IN_DOMAIN[0]
    target = next(
        item
        for item in base_plan.primary
        if item.metadata.city == city and item.role == "reference"
    )
    states = {item.metadata.image_id: "ACCEPTED" for item in base_plan.download_assets}
    states[target.metadata.image_id] = "REJECTED"
    run_id = "1" * 32
    ledger = tmp_path / "media-task-ledger.json"
    revision = tmp_path / "media-plan-revisions.json"
    _write_task_ledger(ledger, base_plan.download_assets, states, run_id=run_id)

    unchanged, report = replenish_media_reserves(
        audit,
        load_media_plan_revision(audit, base_plan, revision, run_id=run_id),
        _media_deficit(city, "reference", 1),
        run_id=run_id,
        ledger_path=ledger,
        revision_path=revision,
    )

    assert unchanged.plan == base_plan
    assert report["action"] == "MEDIA_CORPUS_EXHAUSTED"
    selectable = report["selectable_unused"]
    assert isinstance(selectable, list)
    assert selectable[0]["selectable_unused_records"] == 0


def test_replenishment_does_not_mutate_existing_827_accepted_ledger(
    tmp_path: Path,
) -> None:
    audit = _sized_audit(in_domain_sizes=(100, 40, 40), ood_sizes=(50,))
    base_plan = plan_metadata_split(audit)
    city = _TEST_IN_DOMAIN[0]
    states = {
        item.metadata.image_id: ("ACCEPTED" if index < 827 else "REJECTED")
        for index, item in enumerate(base_plan.download_assets)
    }
    target_rows = [
        item
        for item in base_plan.download_assets
        if item.metadata.city == city and item.role == "reference"
    ]
    for item in target_rows[-3:]:
        states[item.metadata.image_id] = "REJECTED"
    accepted_now = sum(state == "ACCEPTED" for state in states.values())
    for item in base_plan.download_assets:
        if accepted_now >= 827:
            break
        if states[item.metadata.image_id] == "REJECTED" and item not in target_rows:
            states[item.metadata.image_id] = "ACCEPTED"
            accepted_now += 1
    assert accepted_now == 827
    run_id = "2" * 32
    ledger = tmp_path / "media-task-ledger.json"
    revision = tmp_path / "media-plan-revisions.json"
    _write_task_ledger(ledger, base_plan.download_assets, states, run_id=run_id)
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()

    revised, _report = replenish_media_reserves(
        audit,
        load_media_plan_revision(audit, base_plan, revision, run_id=run_id),
        _media_deficit(city, "reference", 3),
        run_id=run_id,
        ledger_path=ledger,
        revision_path=revision,
    )

    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before
    assert sum(state == "ACCEPTED" for state in states.values()) == 827
    assert not (
        {item.metadata.image_id for item in revised.plan.download_assets}
        - {item.metadata.image_id for item in base_plan.download_assets}
    ) & set(states)


def test_provenance_aggregate_is_rebound_to_selected_assets_only(tmp_path: Path) -> None:
    audit = _sized_audit(in_domain_sizes=(100, 40, 40), ood_sizes=(50,))
    plan = plan_metadata_split(audit)
    assets = tuple(_split_asset(item) for item in plan.primary[:2])
    sidecars = tmp_path / "private-sidecars"
    sidecars.mkdir()
    for asset in assets:
        (sidecars / f"{asset.opaque_id}.json").write_text("{}", encoding="utf-8")

    aggregate = finalize_provenance_aggregate(
        {"asset_count": 3, "secrets_included": False},
        assets,
        tmp_path,
    )

    assert aggregate["asset_count"] == 2
    assert aggregate["download_candidate_count"] == 3
    assert aggregate["discarded_or_reserve_count"] == 1
    assert aggregate["selected_split_only"] is True


def test_pipeline_records_post_download_dataset_not_ready_as_terminal(
    tmp_path: Path,
) -> None:
    audit = _sized_audit(in_domain_sizes=(100, 40, 40), ood_sizes=(50,))
    state = tmp_path / "phase3f-state.json"
    pipeline = Phase3FPipeline.create(state, run_id="d" * 32)
    pipeline.lock_selection(audit.selection)
    pipeline.complete_acquisition(AcquisitionGuard())

    pipeline.finalize_dataset_not_ready(reason_code="MEDIA_SPLIT_MINIMUM_UNAVAILABLE")

    resumed = Phase3FPipeline.resume(state)
    assert resumed.stage is PipelineStage.COVERAGE_INSUFFICIENT
    assert resumed.document["outcome"] == "DATASET_NOT_READY_FOR_TRAINING"
    assert resumed.document["image_acquisition_started"] is True


def test_coverage_insufficient_is_a_terminal_resumable_state(tmp_path: Path) -> None:
    state = tmp_path / "phase3f-state.json"
    pipeline = Phase3FPipeline.create(state, run_id="c" * 32)

    pipeline.finalize_coverage_insufficient(reason_code="COVERAGE_INSUFFICIENT")

    resumed = Phase3FPipeline.resume(state)
    assert resumed.stage is PipelineStage.COVERAGE_INSUFFICIENT
    assert resumed.document["image_acquisition_started"] is False
    assert resumed.document["outcome"] == "COVERAGE_INSUFFICIENT"
