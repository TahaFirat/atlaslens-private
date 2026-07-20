from __future__ import annotations

import importlib.util
import inspect
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from atlaslens_api.phase3f import scheduler, training
from atlaslens_api.phase3f.splits import SplitAsset

ROOT = Path(__file__).parents[3]
LAUNCHER = ROOT / "scripts" / "phase3f-end-to-end.ps1"
SUPERVISOR = ROOT / "scripts" / "phase3f" / "supervisor.py"


def _load_supervisor() -> object:
    spec = importlib.util.spec_from_file_location("phase3f_e2e_budget_supervisor", SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset(index: int, *, sequence: str, city: str = "Ankara") -> SplitAsset:
    opaque = f"{index:032x}"
    return SplitAsset(
        opaque_id=opaque,
        city=city,
        role="reference",
        relative_path=f"assets/{opaque[:2]}/{opaque}.jpg",
        contributor_id=f"contributor-{sequence}",
        sequence_id=sequence,
        capture_run_id=sequence,
        content_sha256=f"{index + 1:064x}",
        perceptual_hash=f"{index + 1:016x}",
        parent_or_tile_id=f"image-{index}",
        latitude=39.9,
        longitude=32.8,
    )


def test_scheduler_v4_migrates_rows_queue_quotas_and_budgets(tmp_path: Path) -> None:
    run_id = "a" * 32
    metadata = tmp_path / "metadata-pages.json"
    counters = tmp_path / "client-counters.json"
    destination = tmp_path / "acquisition-scheduler-v4.json"
    metadata.write_text(
        json.dumps(
            {
                "schema": "atlaslens-phase3f-metadata-pages-v3",
                "run_id": run_id,
                "city_index": 0,
                "box_index": 0,
                "next_url": None,
                "cells_by_city": {
                    "İstanbul": [
                        {"bbox": [28.8, 40.9, 28.9, 41.0], "depth": 1},
                    ]
                },
                "rows_by_city": {
                    "İstanbul": [
                        {"mapillary_image_id": "first-image"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    counters.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "request_count": 18,
                "page_count": 10,
                "rejected_item_count": 0,
            }
        ),
        encoding="utf-8",
    )

    document = scheduler.sync_scheduler_checkpoint(
        destination,
        run_id=run_id,
        metadata_checkpoint=metadata,
        client_counters=counters,
        acquisition_checkpoint=None,
        request_cap=10_000,
        media_byte_cap=7 * 1024**3,
        max_wall_seconds=720 * 60,
        status="ACQUISITION_RUNNING",
        city_order=["İstanbul"],
    )

    assert document["schema"] == scheduler.SCHEDULER_CHECKPOINT_SCHEMA
    assert document["first_seen_image_ids"] == ["first-image"]
    assert cast(dict[str, object], document["quotas"])["admitted_metadata_rows"] == 1
    assert cast(dict[str, object], document["budgets"])["request_count"] == 18
    assert len(cast(list[object], document["pending_cell_queue"])) == 1
    assert document["migrated_from_schema"] == "atlaslens-phase3f-metadata-pages-v3"
    assert json.loads(destination.read_text(encoding="utf-8"))["secrets_included"] is False


@pytest.mark.parametrize(
    ("code", "state", "terminal"),
    (
        ("MAPILLARY_TOKEN_REJECTED", "TERMINAL_AUTH", True),
        ("MAPILLARY_API_BAD_REQUEST", "TERMINAL_REQUEST_CONTRACT", True),
        ("MAPILLARY_API_RATE_LIMIT_RETRY_EXHAUSTED", "PAUSED_RATE_LIMIT", False),
        (
            "MAPILLARY_API_SERVER_RETRY_EXHAUSTED",
            "PAUSED_PROVIDER_RETRY_EXHAUSTED",
            False,
        ),
        ("MAPILLARY_PAGING_LOOP_DETECTED", "CELL_FAILED_PAGING", False),
    ),
)
def test_scheduler_failure_matrix_is_typed(
    code: str, state: str, terminal: bool
) -> None:
    assert scheduler.classify_failure(code) == (state, terminal)


def test_dataset_not_ready_report_precedes_any_gpu_or_cloud_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _asset(1, sequence="sequence-a")
    fake = SimpleNamespace(
        root=tmp_path,
        selection=SimpleNamespace(
            in_domain=(
                SimpleNamespace(macro_region="Marmara"),
                SimpleNamespace(macro_region="Ege"),
                SimpleNamespace(macro_region="Akdeniz"),
                SimpleNamespace(macro_region="İç Anadolu"),
                SimpleNamespace(macro_region="Karadeniz"),
            )
        ),
        split=SimpleNamespace(
            assets=(asset,),
            leakage=SimpleNamespace(passed=True),
        ),
        provenance={"asset_count": 1},
    )
    monkeypatch.setattr(training, "verify_sealed_acquisition", lambda _path: fake)
    report_path = tmp_path / "training-readiness.json"

    with pytest.raises(training.DatasetNotReadyError) as error:
        training.require_training_ready(tmp_path / "sealed", report_path=report_path)

    assert error.value.code == "DATASET_NOT_READY_FOR_TRAINING"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ready"] is False
    assert report["gpu_started"] is False
    assert report["cloud_mutations"] == 0
    assert report["secrets_included"] is False


def test_metric_batches_have_positive_pairs_and_negative_groups() -> None:
    assets = tuple(
        _asset(index, sequence=f"sequence-{index // 2}")
        for index in range(12)
    )
    batches = training._epoch_batches(assets, batch_size=4, seed=7, epoch=0)
    assert batches
    for batch in batches:
        counts: dict[str, int] = {}
        for asset in batch:
            counts[asset.sequence_id] = counts.get(asset.sequence_id, 0) + 1
        assert len(counts) >= 2
        assert all(count >= 2 for count in counts.values())


def test_training_and_operator_contracts_are_real_and_secret_safe() -> None:
    train_source = inspect.getsource(training.TrainableMegaLocRuntime.train)
    job_source = inspect.getsource(training.run_training_job)
    launcher = LAUNCHER.read_text(encoding="utf-8")
    supervisor = _load_supervisor()
    execute_source = inspect.getsource(supervisor._run_execute)

    assert ".backward()" in train_source
    assert "scaler.step(optimizer)" in train_source
    assert "AdamW" in train_source
    assert "GradScaler" in train_source
    assert "optimizer.load_state_dict" in inspect.getsource(
        training.TrainableMegaLocRuntime._load_checkpoint
    )
    assert "trainable.safetensors" in inspect.getsource(
        training.TrainableMegaLocRuntime._write_checkpoint
    )
    assert "TRAINING_WEIGHTS_UNCHANGED" in train_source
    assert job_source.index("require_training_ready") < job_source.index("import torch")
    assert job_source.index("fit_abstention_threshold") < job_source.index("holdout_matrix =")
    assert launcher.index('Invoke-Control -ControlAction "readiness"') < launcher.rindex(
        "Invoke-CloudTraining -RunId"
    )
    assert execute_source.index("_require_e2e_budget") < execute_source.index(
        "with RunPodV1Client"
    )
    assert execute_source.index("prepare_transfer_bundle") < execute_source.index(
        "with RunPodV1Client"
    )
    assert 'os.environ.pop("MAPILLARY_ACCESS_TOKEN", None)' in job_source
    assert "automatic_promotion_performed" in job_source
    for action in (
        "Preflight",
        "Execute",
        "Status",
        "Resume",
        "EmergencyStop",
        "Cleanup",
    ):
        assert action in launcher
    assert "Read-Host" in launcher and "-AsSecureString" in launcher
    assert "ZeroFreeBSTR" in launcher
    assert "Remove-Item Env:MAPILLARY_ACCESS_TOKEN" in launcher
    assert "Remove-Item Env:RUNPOD_API_KEY" in launcher
    assert ".env" not in launcher
    assert "asda.html" not in launcher
    assert "Invoke-WebRequest" not in launcher
    assert "Invoke-RestMethod" not in launcher


def test_training_module_never_requires_mapillary_token() -> None:
    source = inspect.getsource(training)
    assert 'os.environ.pop("MAPILLARY_ACCESS_TOKEN", None)' in source
    assert "MapillaryClient" not in source
    assert 'os.environ.get("RUNPOD_API_KEY")' not in source
    assert "thumbnail_url" not in source


def test_end_to_end_python_control_has_no_secret_inputs() -> None:
    source = (ROOT / "scripts" / "phase3f" / "end_to_end.py").read_text(
        encoding="utf-8"
    )
    assert "MAPILLARY_ACCESS_TOKEN" not in source
    assert "RUNPOD_API_KEY" not in source
    assert "cloud_mutations" in source
    assert "FROZEN_START_HEAD" in source
    preflight_source = source[source.index("def preflight(") : source.index("def readiness(")]
    assert "httpx" not in preflight_source
    assert "requests" not in preflight_source
    assert "RunPodV1Client" not in preflight_source
    assert "MapillaryClient" not in preflight_source


def test_e2e_budget_is_three_usd_with_ten_usd_historical_ceiling(tmp_path: Path) -> None:
    module = _load_supervisor()
    policy = module.BudgetPolicy(
        target_usd=Decimal("2.94"),
        soft_stop_usd=Decimal("2.95"),
        terminate_usd=Decimal("2.99"),
        absolute_usd=Decimal("3"),
        max_hourly_cost_usd=Decimal("0.50"),
        max_runtime_seconds=345 * 60,
    )

    historical, projected = module._require_e2e_budget(policy, tmp_path)

    assert historical == Decimal("0")
    assert projected == Decimal("3")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "[decimal]$MaxSpendUsd = 3" in launcher
    assert "E2E_HISTORICAL_BUDGET_USD = Decimal(\"10\")" in SUPERVISOR.read_text(
        encoding="utf-8"
    )


def test_e2e_historical_budget_refuses_projected_total_over_ten(tmp_path: Path) -> None:
    module = _load_supervisor()
    receipt_root = tmp_path / "_receipts"
    receipt_root.mkdir()
    (receipt_root / "prior.json").write_text(
        json.dumps(
            {
                "schema": "atlaslens-phase3f-local-supervisor-receipt-v1",
                "run_id": "b" * 32,
                "conservative_incremental_upper_usd": "7.01",
            }
        ),
        encoding="utf-8",
    )
    policy = module.BudgetPolicy(
        target_usd=Decimal("2.94"),
        soft_stop_usd=Decimal("2.95"),
        terminate_usd=Decimal("2.99"),
        absolute_usd=Decimal("3"),
        max_hourly_cost_usd=Decimal("0.50"),
        max_runtime_seconds=345 * 60,
    )

    with pytest.raises(module.SupervisorExecutionError) as error:
        module._require_e2e_budget(policy, tmp_path)

    assert error.value.code == "HISTORICAL_BUDGET_WOULD_EXCEED_10_USD"
