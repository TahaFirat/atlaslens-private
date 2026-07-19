from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from atlaslens_api.phase3f.acquisition import (
    MAX_REQUESTS,
    AcquisitionGuard,
    AcquisitionRefused,
    PrivacyReview,
    ProvenanceSidecar,
)
from atlaslens_api.phase3f.benchmark import (
    CalibrationObservation,
    RetrievalObservation,
    RetrievalSignals,
    evaluate_holdout_once,
    fit_abstention_threshold,
)
from atlaslens_api.phase3f.coverage import (
    CANDIDATE_CITY_REGIONS,
    CityCoverageRecord,
    select_city_scope,
)
from atlaslens_api.phase3f.pipeline import (
    MODEL_REVISION,
    MODEL_SHA256,
    PROVIDER_REVISION,
    DescriptorPublication,
    Phase3FPipeline,
    PipelineStage,
    PipelineStateError,
)
from atlaslens_api.phase3f.splits import LeakageViolation, SplitAsset, seal_split


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _coverage_records() -> list[CityCoverageRecord]:
    records: list[CityCoverageRecord] = []
    for index, (city, region) in enumerate(CANDIDATE_CITY_REGIONS.items()):
        records.append(
            CityCoverageRecord(
                city=city,
                macro_region=region,
                image_count=500 - index,
                sequence_count=120 - index,
                contributor_count=80 - index,
                spatial_cell_count=160 - index,
                capture_year_count=8,
                eligible_asset_count=250 - index,
                metadata_sha256=_sha(f"metadata-{index}"),
            )
        )
    return records


def _split_assets(
    in_domain: tuple[str, ...], ood: tuple[str, ...]
) -> list[SplitAsset]:
    assets: list[SplitAsset] = []
    counter = 0

    def add(city: str, role: str, count: int, city_index: int) -> None:
        nonlocal counter
        role_offset = {
            "reference": 0.0,
            "calibration": 0.4,
            "sealed_holdout": 1.0,
            "ood_holdout": 1.4,
        }[role]
        for item_index in range(count):
            seed = f"asset-{counter}"
            opaque_id = _sha(seed)[:32]
            assets.append(
                SplitAsset(
                    opaque_id=opaque_id,
                    city=city,
                    role=role,  # type: ignore[arg-type]
                    relative_path=f"assets/{opaque_id[:2]}/{opaque_id}.jpg",
                    contributor_id=f"contributor-{counter}",
                    sequence_id=f"sequence-{counter}",
                    capture_run_id=f"run-{counter}",
                    content_sha256=_sha(f"content-{counter}"),
                    perceptual_hash=_sha(f"phash-{counter}")[:16],
                    parent_or_tile_id=f"parent-{counter}",
                    latitude=30.0 + city_index * 1.5 + role_offset,
                    longitude=25.0 + item_index * 0.001,
                )
            )
            counter += 1

    for city_index, city in enumerate(in_domain):
        add(city, "reference", 75, city_index)
        add(city, "calibration", 25, city_index)
        add(city, "sealed_holdout", 25, city_index)
    for city_index, city in enumerate(ood, start=len(in_domain)):
        add(city, "ood_holdout", 40, city_index)
    return assets


def _calibration() -> list[CalibrationObservation]:
    return [
        CalibrationObservation(
            signals=RetrievalSignals(
                similarity=0.80 if index < 60 else 0.20,
                top1_top2_margin=0.20 if index < 60 else 0.01,
                reference_density=20 if index < 60 else 1,
            ),
            city_correct=index < 60,
        )
        for index in range(100)
    ]


def _observations(
    in_domain: tuple[str, ...], ood: tuple[str, ...]
) -> list[RetrievalObservation]:
    rows: list[RetrievalObservation] = []
    index = 0
    for city in in_domain:
        for item in range(25):
            accepted = item < 20
            rows.append(
                RetrievalObservation(
                    opaque_query_id=_sha(f"query-{index}")[:32],
                    role="sealed_holdout",
                    city=city,
                    province=city,
                    predicted_cities=(city, "redacted-a", "redacted-b"),
                    predicted_provinces=(city, "redacted-a", "redacted-b"),
                    relevant_reference_rank=1,
                    geodesic_error_km=10.0,
                    signals=RetrievalSignals(
                        similarity=0.80 if accepted else 0.10,
                        top1_top2_margin=0.20 if accepted else 0.0,
                        reference_density=20 if accepted else 0,
                    ),
                    contributor_group_sha256=_sha(f"contributor-group-{index}"),
                    sequence_group_sha256=_sha(f"sequence-group-{index}"),
                )
            )
            index += 1
    for city in ood:
        for item in range(40):
            false_accept = item < 4
            rows.append(
                RetrievalObservation(
                    opaque_query_id=_sha(f"query-{index}")[:32],
                    role="ood_holdout",
                    city=city,
                    province=city,
                    predicted_cities=("redacted-a", "redacted-b", "redacted-c"),
                    predicted_provinces=("redacted-a", "redacted-b", "redacted-c"),
                    relevant_reference_rank=None,
                    geodesic_error_km=None,
                    signals=RetrievalSignals(
                        similarity=0.80 if false_accept else 0.10,
                        top1_top2_margin=0.20 if false_accept else 0.0,
                        reference_density=20 if false_accept else 0,
                    ),
                    contributor_group_sha256=_sha(f"contributor-group-{index}"),
                    sequence_group_sha256=_sha(f"sequence-group-{index}"),
                    failure_category="coverage_gap",
                )
            )
            index += 1
    return rows


def test_metadata_selection_is_deterministic_diverse_and_preacquisition() -> None:
    records = _coverage_records()
    first = select_city_scope(records, policy_sha256=_sha("policy"))
    second = select_city_scope(list(reversed(records)), policy_sha256=_sha("policy"))

    assert first.lock_sha256 == second.lock_sha256
    assert 5 <= len(first.in_domain) <= 8
    assert len(first.ood) == 2
    assert len({record.macro_region for record in first.in_domain}) >= 4
    assert not set(record.city for record in first.in_domain) & set(
        record.city for record in first.ood
    )
    assert first.document()["selection_stage"] == "metadata_only_before_image_acquisition"


def test_acquisition_guard_caps_token_failure_and_provenance() -> None:
    guard = AcquisitionGuard(request_count=MAX_REQUESTS - 1)
    guard.begin_request()
    guard.finish_request(status_code=429)
    assert not guard.should_retry(status_code=429, completed_attempts=1)
    with pytest.raises(AcquisitionRefused, match="MAPILLARY_REQUEST_CAP_REACHED"):
        guard.begin_request()

    token_guard = AcquisitionGuard()
    token_guard.begin_request()
    with pytest.raises(AcquisitionRefused, match="MAPILLARY_TOKEN_REJECTED"):
        token_guard.finish_request(status_code=403)
    assert token_guard.stopped

    admitted = AcquisitionGuard()
    admitted.begin_request()
    admitted.finish_request(
        status_code=200,
        media_bytes=1234,
        admitted_image=True,
        provenance=ProvenanceSidecar(
            mapillary_image_id="opaque-mapillary-id",
            source_page="https://www.mapillary.com/app/?pKey=opaque-mapillary-id",
            contributor_attribution="Mapillary contributor attribution",
            capture_date=datetime(2025, 1, 1, tzinfo=UTC),
            source_policy_receipt_sha256=_sha("source-policy"),
        ),
        privacy_review=PrivacyReview(passed=True),
    )
    assert admitted.image_count == 1
    assert admitted.media_bytes == 1234


def test_split_seal_enforces_every_leakage_invariant() -> None:
    selection = select_city_scope(_coverage_records(), policy_sha256=_sha("policy"))
    in_domain = tuple(record.city for record in selection.in_domain[:5])
    ood = tuple(record.city for record in selection.ood)
    assets = _split_assets(in_domain, ood)
    split = seal_split(assets, in_domain_cities=in_domain, ood_cities=ood)

    assert split.leakage.passed
    assert split.inference_document()["ground_truth_present"] is False
    assert "latitude" not in json.dumps(split.inference_document())
    assert split.holdout_seal_sha256 != split.evaluator_truth_sha256

    bad = list(assets)
    reference = next(item for item in bad if item.role == "reference")
    holdout_index = next(index for index, item in enumerate(bad) if item.role == "sealed_holdout")
    holdout = bad[holdout_index]
    bad[holdout_index] = SplitAsset(
        opaque_id=holdout.opaque_id,
        city=holdout.city,
        role=holdout.role,
        relative_path=holdout.relative_path,
        contributor_id=reference.contributor_id,
        sequence_id=holdout.sequence_id,
        capture_run_id=holdout.capture_run_id,
        content_sha256=holdout.content_sha256,
        perceptual_hash=holdout.perceptual_hash,
        parent_or_tile_id=holdout.parent_or_tile_id,
        latitude=holdout.latitude,
        longitude=holdout.longitude,
    )
    with pytest.raises(LeakageViolation):
        seal_split(bad, in_domain_cities=in_domain, ood_cities=ood)


def test_calibration_holdout_gates_and_resumable_full_pipeline(tmp_path: Path) -> None:
    selection = select_city_scope(_coverage_records(), policy_sha256=_sha("policy"))
    in_domain = tuple(record.city for record in selection.in_domain[:5])
    ood = tuple(record.city for record in selection.ood)
    split = seal_split(
        _split_assets(in_domain, ood),
        in_domain_cities=in_domain,
        ood_cities=ood,
    )
    threshold = fit_abstention_threshold(
        _calibration(),
        selection_lock_sha256=selection.lock_sha256,
        split_lock_sha256=split.split_lock_sha256,
    )
    assert threshold.calibration_coverage == 0.6
    assert threshold.accepted_city_accuracy == 1.0

    benchmark = evaluate_holdout_once(
        _observations(in_domain, ood),
        threshold=threshold,
        selection_lock_sha256=selection.lock_sha256,
        split_lock_sha256=split.split_lock_sha256,
        in_domain_cities=in_domain,
        ood_cities=ood,
        leakage_passed=True,
        city_minimums_passed=True,
        security_integrity_passed=True,
        holdout_open_count_before=0,
    )
    assert benchmark.outcome == "COMPLETE_ACCEPTED_PRIVATE_PILOT"
    assert all(benchmark.gates.values())
    assert benchmark.metrics["ood_false_accept_rate"] == 0.1
    assert benchmark.document()["confidence"] is None

    state_path = tmp_path / "phase3f-state.json"
    pipeline = Phase3FPipeline.create(state_path, run_id="1" * 32)
    pipeline.lock_selection(selection)
    pipeline.complete_acquisition(AcquisitionGuard())
    pipeline.record_split(split)
    pipeline.record_descriptor_publication(
        DescriptorPublication(
            selection_lock_sha256=selection.lock_sha256,
            split_lock_sha256=split.split_lock_sha256,
            source_policy_sha256=_sha("source-policy"),
            descriptor_publication_sha256=_sha("descriptors"),
            index_sha256=_sha("index"),
            city_scope=in_domain,
            created_at=datetime.now(UTC),
        )
    )
    pipeline.lock_threshold(threshold)
    pipeline.record_benchmark(benchmark)
    pipeline.finalize()

    resumed = Phase3FPipeline.resume(state_path)
    assert resumed.stage == PipelineStage.FINALIZED
    assert resumed.document["holdout_open_count"] == 1
    assert resumed.document["outcome"] == "COMPLETE_ACCEPTED_PRIVATE_PILOT"
    state_text = state_path.read_text(encoding="utf-8")
    assert "latitude" not in state_text
    assert "longitude" not in state_text
    with pytest.raises(PipelineStateError, match="PHASE3F_STATE_TRANSITION_REFUSED"):
        resumed.finalize()

    assert MODEL_SHA256 in pipeline.document["descriptor_publication"]["model_sha256"]
    assert pipeline.document["descriptor_publication"]["model_revision"] == MODEL_REVISION
    assert pipeline.document["descriptor_publication"]["provider_revision"] == PROVIDER_REVISION
