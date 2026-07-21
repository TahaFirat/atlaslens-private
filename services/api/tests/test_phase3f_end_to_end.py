from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import subprocess
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from atlaslens_api.phase3f import local_first, scheduler, training
from atlaslens_api.phase3f.splits import SplitAsset

ROOT = Path(__file__).parents[3]
LAUNCHER = ROOT / "scripts" / "phase3f-end-to-end.ps1"
SUPERVISOR = ROOT / "scripts" / "phase3f" / "supervisor.py"
CONTROL = ROOT / "scripts" / "phase3f" / "end_to_end.py"


def _load_supervisor() -> object:
    spec = importlib.util.spec_from_file_location("phase3f_e2e_budget_supervisor", SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_control() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase3f_e2e_resume_control", CONTROL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _ready_document(asset_count: int = 830) -> dict[str, object]:
    return {
        "schema": training.TRAINING_READINESS_SCHEMA,
        "outcome": "READY_FOR_TRAINING",
        "ready": True,
        "asset_count": asset_count,
        "checks": {"fixture_integrity": True},
        "gpu_started": False,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def _write_resume_runtime(
    runtime_root: Path,
    *,
    asset_count: int = 830,
    sealed: bool = True,
) -> tuple[str, Path, dict[str, object]]:
    run_id = "a" * 32
    run_root = runtime_root / run_id
    run_root.mkdir(parents=True)
    (runtime_root / "current.json").write_bytes(
        _canonical_bytes({"schema": local_first.LOCAL_CURRENT_SCHEMA, "run_id": run_id})
    )
    report = _ready_document(asset_count)
    report_payload = _canonical_bytes(report)
    (run_root / "training-readiness.json").write_bytes(report_payload)
    (run_root / "state.json").write_bytes(
        _canonical_bytes(
            {
                "schema": local_first.LOCAL_STATE_SCHEMA,
                "run_id": run_id,
                "stage": "ACQUISITION_SEALED",
                "asset_count": asset_count,
                "model_loaded": False,
                "readiness_report_sha256": hashlib.sha256(report_payload).hexdigest(),
                "secrets_included": False,
            }
        )
    )
    sealed_root = run_root / "sealed-acquisition"
    if sealed:
        sealed_root.mkdir()
        (sealed_root / "sealed-assets.json").write_bytes(b"fixture-sealed-assets\n")
    return run_id, run_root, report


def _patch_valid_seal(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
    report: dict[str, object],
    asset_count: int = 830,
) -> None:
    verified = SimpleNamespace(
        run_id=run_id,
        split=SimpleNamespace(assets=tuple(range(asset_count))),
    )
    monkeypatch.setattr(local_first, "verify_sealed_acquisition", lambda _root: verified)
    monkeypatch.setattr(training, "training_readiness_document", lambda _sealed: report)


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _budget_policy(module: object) -> object:
    return module.BudgetPolicy(
        target_usd=Decimal("2.94"),
        soft_stop_usd=Decimal("2.95"),
        terminate_usd=Decimal("2.99"),
        absolute_usd=Decimal("3"),
        max_hourly_cost_usd=Decimal("0.50"),
        max_runtime_seconds=345 * 60,
    )


def _billing_snapshot(
    module: object,
    *,
    balance: str = "10",
    current_spend: str = "0",
) -> object:
    return module.RunPodBillingSnapshot(
        client_balance_usd=Decimal(balance),
        current_spend_per_hour_usd=Decimal(current_spend),
    )


def _closed_operator_receipt(
    module: object,
    *,
    run_id: str,
    pod_id: str,
    max_spend_usd: str = "3",
    duration_seconds: int = 120,
    cleanup_verified: bool = True,
) -> object:
    return module.OperatorReceipt(
        run_id=run_id,
        run_marker=f"atlaslens-phase3f-{run_id}",
        pod_id=pod_id,
        supervisor_pid=1234,
        stage="terminated" if cleanup_verified else "failed",
        started_at="2026-07-20T00:00:00+00:00",
        finished_at=f"2026-07-20T00:{duration_seconds // 60:02d}:{duration_seconds % 60:02d}+00:00",
        max_spend_usd=Decimal(max_spend_usd),
        soft_stop_usd=Decimal("2.95"),
        hard_stop_usd=Decimal("2.99"),
        max_gpu_hourly_usd=Decimal("0.50"),
        max_wall_minutes=345,
        cleanup_verified=cleanup_verified,
    )


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
    assert execute_source.index("reconciliation_inventory = billing_client.inventory()") < (
        execute_source.index("require_empty_inventory(reconciliation_inventory)")
    )
    assert execute_source.index("require_empty_inventory(reconciliation_inventory)") < (
        execute_source.index("billing_client.account_billing_snapshot()")
    )
    assert execute_source.index("billing_client.account_billing_snapshot()") < (
        execute_source.index("_require_e2e_budget")
    )
    assert execute_source.index("_require_e2e_budget") < execute_source.index(
        "archive_completed_operator_receipt"
    )
    assert execute_source.index("_require_e2e_budget") < execute_source.index("ssh-keygen")
    assert execute_source.index("prepare_transfer_bundle") < execute_source.index(
        "select_gpu_offer"
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


def test_sealed_ready_resume_is_read_only_and_skips_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load_control()
    runtime = tmp_path / "runtime"
    run_id, _run_root, report = _write_resume_runtime(runtime)
    _patch_valid_seal(monkeypatch, run_id=run_id, report=report)
    before = _tree_sha256(runtime)

    plan = control.resume_plan(ROOT, runtime, tmp_path / "cloud")

    assert plan["asset_count"] == 830
    assert plan["acquisition_required"] is False
    assert plan["mapillary_required"] is False
    assert plan["dataset_write_count"] == 0
    assert plan["cloud_readiness_transition_count"] == 1
    assert plan["next_phase"] == "cloud_inventory"
    assert plan["runpod_api_calls"] == 0
    assert plan["cloud_mutations"] == 0
    assert _tree_sha256(runtime) == before


def test_repeated_sealed_resume_keeps_hash_and_state_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load_control()
    runtime = tmp_path / "runtime"
    run_id, _run_root, report = _write_resume_runtime(runtime)
    _patch_valid_seal(monkeypatch, run_id=run_id, report=report)
    before = _tree_sha256(runtime)

    first = control.resume_plan(ROOT, runtime, tmp_path / "cloud")
    middle = _tree_sha256(runtime)
    second = control.resume_plan(ROOT, runtime, tmp_path / "cloud")

    assert first == second
    assert before == middle == _tree_sha256(runtime)
    assert first["acquisition_required"] is False


def test_resume_missing_seal_is_exact_typed_blocker(tmp_path: Path) -> None:
    control = _load_control()
    runtime = tmp_path / "runtime"
    _write_resume_runtime(runtime, sealed=False)

    with pytest.raises(control.EndToEndError) as error:
        control.resume_plan(ROOT, runtime, tmp_path / "cloud")

    assert error.value.code == "SEALED_BUNDLE_MISSING"
    assert error.value.artifact == "sealed-acquisition"


def test_resume_seal_hash_mismatch_is_not_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load_control()
    runtime = tmp_path / "runtime"
    _write_resume_runtime(runtime)

    def fail_verification(_root: Path) -> object:
        raise local_first.LocalFirstError("SEALED_INVENTORY_MISMATCH")

    monkeypatch.setattr(local_first, "verify_sealed_acquisition", fail_verification)
    with pytest.raises(control.EndToEndError) as error:
        control.resume_plan(ROOT, runtime, tmp_path / "cloud")

    assert error.value.code == "SEALED_INVENTORY_MISMATCH"
    assert error.value.artifact == "checksum-inventory.json"
    assert error.value.field == "sha256"


def test_resume_readiness_hash_mismatch_reports_expected_and_actual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load_control()
    runtime = tmp_path / "runtime"
    run_id, run_root, report = _write_resume_runtime(runtime)
    _patch_valid_seal(monkeypatch, run_id=run_id, report=report)
    (run_root / "training-readiness.json").write_bytes(
        _canonical_bytes({**report, "ready": False})
    )

    with pytest.raises(control.EndToEndError) as error:
        control.resume_plan(ROOT, runtime, tmp_path / "cloud")

    assert error.value.code == "TRAINING_READINESS_HASH_MISMATCH"
    assert error.value.artifact == "training-readiness.json"
    assert error.value.field == "sha256"
    assert len(error.value.expected) == 64
    assert len(error.value.actual) == 64


def test_resume_requires_exactly_830_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load_control()
    runtime = tmp_path / "runtime"
    run_id, _run_root, report = _write_resume_runtime(runtime, asset_count=829)
    _patch_valid_seal(
        monkeypatch,
        run_id=run_id,
        report=report,
        asset_count=829,
    )

    with pytest.raises(control.EndToEndError) as error:
        control.resume_plan(ROOT, runtime, tmp_path / "cloud")

    assert error.value.code == "PHASE3F_DATASET_ASSET_COUNT_MISMATCH"
    assert error.value.expected == 830
    assert error.value.actual == 829


def test_resume_does_not_restart_running_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load_control()
    runtime = tmp_path / "runtime"
    cloud = tmp_path / "cloud"
    run_id, _run_root, report = _write_resume_runtime(runtime)
    _patch_valid_seal(monkeypatch, run_id=run_id, report=report)
    operator = cloud / "_operator" / "phase3f-current.json"
    operator.parent.mkdir(parents=True)
    operator.write_bytes(
        _canonical_bytes({"run_id": run_id, "stage": "running", "secrets_included": False})
    )

    plan = control.resume_plan(ROOT, runtime, cloud)

    assert plan["next_phase"] == "training_running"
    assert plan["cloud_readiness_transition_count"] == 0
    assert plan["acquisition_required"] is False


def test_resume_does_not_restart_completed_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load_control()
    runtime = tmp_path / "runtime"
    cloud = tmp_path / "cloud"
    run_id, _run_root, report = _write_resume_runtime(runtime)
    _patch_valid_seal(monkeypatch, run_id=run_id, report=report)
    receipt = cloud / "_receipts" / "completed.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(
        _canonical_bytes(
            {
                "schema": "atlaslens-phase3f-local-supervisor-receipt-v1",
                "run_id": run_id,
                "outcome": "TRAINING_COMPLETE",
                "pod_termination_verified": True,
                "secret_values_included": False,
            }
        )
    )

    plan = control.resume_plan(ROOT, runtime, cloud)

    assert plan["next_phase"] == "training_completed"
    assert plan["cloud_readiness_transition_count"] == 0
    assert plan["acquisition_required"] is False


def test_genuinely_resumable_acquisition_is_the_only_mapillary_path(tmp_path: Path) -> None:
    control = _load_control()
    runtime = tmp_path / "runtime"
    run_id, run_root, _report = _write_resume_runtime(runtime, sealed=False)
    state_path = run_root / "state.json"
    state_path.write_bytes(
        _canonical_bytes(
            {
                "schema": local_first.LOCAL_STATE_SCHEMA,
                "run_id": run_id,
                "stage": "ACQUISITION_FAILED_RESUMABLE",
                "error_code": "MAPILLARY_API_SERVER_RETRY_EXHAUSTED",
                "model_loaded": False,
                "secrets_included": False,
            }
        )
    )
    work = run_root / "acquisition-work"
    work.mkdir()
    (work / "metadata-pages.json").write_bytes(
        _canonical_bytes({"rows_by_city": {"fixture-city": []}})
    )

    plan = control.resume_plan(ROOT, runtime, tmp_path / "cloud")

    assert plan["next_phase"] == "local_acquisition"
    assert plan["acquisition_required"] is True
    assert plan["mapillary_required"] is True
    assert plan["dataset_write_count"] == 0


def test_launcher_preserves_typed_child_failure_and_gates_cloud_before_key() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    execute_flow = launcher[launcher.index('{ $_ -in @("Execute", "Resume") }') :]

    assert 'throw "PHASE3F_LOCAL_ACQUISITION_FAILED"' not in launcher
    assert "Get-SanitizedChildFailure" in launcher
    assert "PHASE3F_LOCAL_ACQUISITION_CHILD_FAILED" in launcher
    assert execute_flow.index('Invoke-Control -ControlAction "resume-plan"') < (
        execute_flow.index("Invoke-LocalAcquisition")
    )
    assert execute_flow.index('Invoke-Control -ControlAction "resume-plan"') < (
        execute_flow.index("Invoke-CloudTraining")
    )
    assert execute_flow.index("Invoke-CloudTraining") < execute_flow.index(
        'Invoke-Control -ControlAction "status"'
    )


def test_child_error_sanitizer_preserves_code_and_redacts_raw_output() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    function_source = launcher[
        launcher.index("function Get-SanitizedChildFailure") : launcher.index(
            "function Throw-SanitizedChildFailure"
        )
    ]
    detail = json.dumps(
        {
            "error_code": "SEALED_INVENTORY_MISMATCH",
            "artifact": "checksum-inventory.json",
            "field": "sha256",
            "expected": "a" * 64,
            "actual": "b" * 64,
            "raw": "raw-secret-must-not-escape",
        },
        separators=(",", ":"),
    )
    probe = (
        function_source
        + "\n$typed=Get-SanitizedChildFailure -ChildOutput "
        + "@('raw-secret-must-not-escape','ACQUISITION_ALREADY_SEALED') "
        + "-FallbackCode 'FALLBACK'; "
        + f"$detailed=Get-SanitizedChildFailure -ChildOutput @('{detail}') "
        + "-FallbackCode 'FALLBACK'; "
        + "[ordered]@{typed=$typed.Code;detail=$detailed.Diagnostic}|ConvertTo-Json -Compress"
    )

    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", probe],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=30,
        text=True,
    )

    assert completed.returncode == 0
    output = json.loads(completed.stdout)
    assert output["typed"] == "ACQUISITION_ALREADY_SEALED"
    assert "SEALED_INVENTORY_MISMATCH" in output["detail"]
    assert "raw-secret-must-not-escape" not in completed.stdout
    assert "raw-secret-must-not-escape" not in completed.stderr


def test_untyped_control_failure_redacts_raw_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    control = _load_control()
    secret = "raw-secret-value-must-not-escape"

    def fail(*_args: object) -> dict[str, object]:
        raise RuntimeError(secret)

    monkeypatch.setattr(control, "resume_plan", fail)
    result = control.main(
        [
            "resume-plan",
            "--repository-root",
            str(ROOT),
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--cloud-runtime-root",
            str(tmp_path / "cloud"),
        ]
    )

    captured = capsys.readouterr().out
    assert result == 1
    assert captured.strip() == "PHASE3F_END_TO_END_FAILED"
    assert secret not in captured


def test_e2e_budget_is_three_usd_with_ten_usd_historical_ceiling(tmp_path: Path) -> None:
    module = _load_supervisor()
    policy = _budget_policy(module)

    reconciliation, receipt_path = module._require_e2e_budget(
        policy,
        tmp_path,
        run_id="a" * 32,
        billing=_billing_snapshot(module),
        api_request_count=5,
        cloud_mutation_count=0,
    )

    assert reconciliation.actual_billed_usd == Decimal("0")
    assert reconciliation.projected_total_usd == Decimal("3")
    assert reconciliation.remaining_authorized_usd == Decimal("10")
    assert receipt_path.is_file()
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
    policy = _budget_policy(module)

    with pytest.raises(module.SupervisorExecutionError) as error:
        module._require_e2e_budget(
            policy,
            tmp_path,
            run_id="a" * 32,
            billing=_billing_snapshot(module),
            api_request_count=5,
            cloud_mutation_count=0,
        )

    assert error.value.code == "BUDGET_INSUFFICIENT"
    assert list((tmp_path / "_budget" / "reconciliations").glob("*.json"))


def test_four_short_cleaned_runs_release_unused_reservations_and_allow_point83_plus_three(
    tmp_path: Path,
) -> None:
    module = _load_supervisor()
    archive = tmp_path / "_operator" / "archive"
    archive.mkdir(parents=True)
    for index in range(4):
        run_id = f"{index + 1:032x}"
        module.write_operator_receipt(
            archive / f"{run_id}.json",
            _closed_operator_receipt(
                module,
                run_id=run_id,
                pod_id=f"closed-pod-{index}",
            ),
        )

    reconciliation, _path = module._require_e2e_budget(
        _budget_policy(module),
        tmp_path,
        run_id="f" * 32,
        billing=_billing_snapshot(module, balance="9.17"),
        api_request_count=5,
        cloud_mutation_count=0,
    )

    assert reconciliation.legacy_closed_reservation_total_usd == Decimal("12")
    assert reconciliation.local_conservative_historical_usd > Decimal("0.46")
    assert reconciliation.local_conservative_historical_usd < Decimal("0.47")
    assert reconciliation.actual_billed_usd == Decimal("0.83")
    assert reconciliation.conservative_unbilled_estimate_usd == Decimal("0")
    assert reconciliation.closed_unused_reservation_released_usd == Decimal("11.17")
    assert reconciliation.projected_total_usd == Decimal("3.83")
    assert reconciliation.remaining_authorized_usd == Decimal("9.17")


def test_billing_lag_keeps_local_lifecycle_estimate_nonzero(tmp_path: Path) -> None:
    module = _load_supervisor()
    archive = tmp_path / "_operator" / "archive"
    archive.mkdir(parents=True)
    module.write_operator_receipt(
        archive / "closed.json",
        _closed_operator_receipt(
            module,
            run_id="1" * 32,
            pod_id="billing-lag-pod",
        ),
    )

    reconciliation = module._budget_reconciliation(
        _budget_policy(module),
        tmp_path,
        _billing_snapshot(module, balance="10"),
    )

    assert reconciliation.actual_billed_usd == Decimal("0")
    assert reconciliation.conservative_unbilled_estimate_usd > Decimal("0.11")
    assert reconciliation.projected_total_usd > Decimal("3.11")


def test_duplicate_provider_billing_receipt_is_counted_once(tmp_path: Path) -> None:
    module = _load_supervisor()
    receipts = tmp_path / "_receipts"
    receipts.mkdir()
    pod_hash = "a" * 64
    for name, amount in (("first", "0.20"), ("duplicate", "0.25")):
        (receipts / f"{name}.json").write_text(
            json.dumps(
                {
                    "schema": "atlaslens-phase3f-local-supervisor-receipt-v1",
                    "run_id": "2" * 32,
                    "pod_id_sha256": pod_hash,
                    "conservative_incremental_upper_usd": amount,
                    "actual_spend_available": True,
                    "actual_spend_usd": amount,
                }
            ),
            encoding="utf-8",
        )

    reconciliation = module._budget_reconciliation(
        _budget_policy(module),
        tmp_path,
        _billing_snapshot(module),
    )

    assert reconciliation.actual_billed_usd == Decimal("0.25")
    assert reconciliation.local_conservative_historical_usd == Decimal("0.25")
    assert reconciliation.duplicate_billing_record_count == 1


def test_cleanup_unverified_run_keeps_full_active_exposure(tmp_path: Path) -> None:
    module = _load_supervisor()
    operator_root = tmp_path / "_operator"
    operator_root.mkdir()
    module.write_operator_receipt(
        operator_root / "phase3f-current.json",
        _closed_operator_receipt(
            module,
            run_id="3" * 32,
            pod_id="unverified-pod",
            cleanup_verified=False,
        ),
    )

    reconciliation = module._budget_reconciliation(
        _budget_policy(module),
        tmp_path,
        _billing_snapshot(module),
    )

    assert reconciliation.active_exposure_usd == Decimal("3")
    assert reconciliation.closed_unused_reservation_released_usd == Decimal("0")
    assert reconciliation.projected_total_usd == Decimal("6")


def test_unrecognized_provider_current_spend_blocks_after_sanitized_receipt(
    tmp_path: Path,
) -> None:
    module = _load_supervisor()

    with pytest.raises(module.SupervisorExecutionError) as error:
        module._require_e2e_budget(
            _budget_policy(module),
            tmp_path,
            run_id="8" * 32,
            billing=_billing_snapshot(module, current_spend="0.01"),
            api_request_count=5,
            cloud_mutation_count=0,
        )

    assert error.value.code == "UNRECOGNIZED_ACTIVE_BILLING_EXPOSURE"
    receipts = list((tmp_path / "_budget" / "reconciliations").glob("*.json"))
    assert len(receipts) == 1
    document = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert document["unrecognized_active_billing"] is True
    assert document["active_exposure_usd"] == "10"
    assert document["cloud_mutations"] == 0


@pytest.mark.parametrize(
    ("balance", "expected_actual", "expect_unbilled"),
    (("9.90", Decimal("0.10"), True), ("9.50", Decimal("0.50"), False)),
)
def test_provider_balance_delta_and_local_estimate_use_the_safe_higher_total(
    tmp_path: Path,
    balance: str,
    expected_actual: Decimal,
    expect_unbilled: bool,
) -> None:
    module = _load_supervisor()
    archive = tmp_path / "_operator" / "archive"
    archive.mkdir(parents=True)
    module.write_operator_receipt(
        archive / "closed.json",
        _closed_operator_receipt(
            module,
            run_id="4" * 32,
            pod_id="mismatch-pod",
        ),
    )

    reconciliation = module._budget_reconciliation(
        _budget_policy(module),
        tmp_path,
        _billing_snapshot(module, balance=balance),
    )

    assert reconciliation.actual_billed_usd == expected_actual
    assert (reconciliation.conservative_unbilled_estimate_usd > 0) is expect_unbilled
    assert (
        reconciliation.actual_billed_usd
        + reconciliation.conservative_unbilled_estimate_usd
    ) == max(expected_actual, reconciliation.local_conservative_historical_usd)


def test_verified_actual_over_ten_uses_historical_exceeded_code(tmp_path: Path) -> None:
    module = _load_supervisor()
    receipts = tmp_path / "_receipts"
    receipts.mkdir()
    (receipts / "actual.json").write_text(
        json.dumps(
            {
                "schema": "atlaslens-phase3f-local-supervisor-receipt-v1",
                "run_id": "5" * 32,
                "pod_id_sha256": "b" * 64,
                "conservative_incremental_upper_usd": "10.01",
                "actual_spend_available": True,
                "actual_spend_usd": "10.01",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(module.SupervisorExecutionError) as error:
        module._require_e2e_budget(
            _budget_policy(module),
            tmp_path,
            run_id="f" * 32,
            billing=_billing_snapshot(module),
            api_request_count=5,
            cloud_mutation_count=0,
        )

    assert error.value.code == "HISTORICAL_BUDGET_ALREADY_EXCEEDED"


def test_reconciliation_refuses_any_prior_cloud_mutation(tmp_path: Path) -> None:
    module = _load_supervisor()

    with pytest.raises(module.SupervisorExecutionError) as error:
        module._require_e2e_budget(
            _budget_policy(module),
            tmp_path,
            run_id="6" * 32,
            billing=_billing_snapshot(module),
            api_request_count=5,
            cloud_mutation_count=1,
        )

    assert error.value.code == "BUDGET_RECONCILIATION_AFTER_MUTATION"
    assert not (tmp_path / "_budget").exists()


def test_reconciliation_receipt_contains_only_sanitized_totals_and_hashes(
    tmp_path: Path,
) -> None:
    module = _load_supervisor()
    archive = tmp_path / "_operator" / "archive"
    archive.mkdir(parents=True)
    module.write_operator_receipt(
        archive / "closed.json",
        _closed_operator_receipt(
            module,
            run_id="7" * 32,
            pod_id="raw-provider-pod-id",
        ),
    )

    _reconciliation, receipt_path = module._require_e2e_budget(
        _budget_policy(module),
        tmp_path,
        run_id="f" * 32,
        billing=_billing_snapshot(module, balance="9.17"),
        api_request_count=5,
        cloud_mutation_count=0,
    )
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    serialized = json.dumps(document, sort_keys=True)

    assert document["schema"] == "atlaslens-phase3f-budget-reconciliation-v1"
    assert len(document["receipt_id"]) == 32
    assert len(document["receipt_sha256"]) == 64
    assert document["provider_response_body_included"] is False
    assert document["secret_values_included"] is False
    assert document["cloud_mutations"] == 0
    assert "raw-provider-pod-id" not in serialized
    assert "runpod-test-token-never-log" not in serialized
