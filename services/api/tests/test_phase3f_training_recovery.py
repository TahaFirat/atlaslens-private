from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tarfile
import time
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

from atlaslens_api.phase3f.operator import OperatorReceipt, write_operator_receipt
from atlaslens_api.phase3f.runpod import PodConnection
from atlaslens_api.phase3f.safety import (
    BudgetPolicy,
    PodRecord,
    PodRequest,
    RunPodLease,
)
from atlaslens_api.phase3f.training_recovery import (
    CHECKPOINT_MANIFEST_SCHEMA,
    CHECKPOINT_POINTER_SCHEMA,
    TrainingRecoveryError,
    atomic_json,
    classify_remote_failure,
    inspect_checkpoint_archive,
    inspect_local_checkpoint,
    sanitize_log_line,
    store_verified_checkpoint_archive,
)

RUN_ID = "a" * 32
READINESS = "b" * 64
SEALED = "c" * 64
CONFIG = "d" * 64


def _load_supervisor() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts/phase3f/supervisor.py"
    spec = importlib.util.spec_from_file_location("phase3f_recovery_supervisor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("return_code", "message", "expected"),
    (
        (1, "TRAINING_CUDA_OOM_EXHAUSTED", "REMOTE_TRAINING_CUDA_OOM"),
        (124, "", "REMOTE_TRAINING_DEADLINE"),
        (1, "OSError: ENOSPC", "REMOTE_TRAINING_DISK_FULL"),
        (1, "TRAINING_IMAGE_INVALID", "REMOTE_TRAINING_DATALOADER_FAILED"),
        (1, "TRAINING_NONFINITE_LOSS", "REMOTE_TRAINING_NONFINITE_LOSS"),
        (1, "TRAINING_CHECKPOINT_WRITE_FAILED", "REMOTE_TRAINING_CHECKPOINT_FAILED"),
        (1, "RUNTIME_MISSING", "REMOTE_TRAINING_DEPENDENCY_FAILED"),
        (-15, "", "REMOTE_TRAINING_PROCESS_SIGNALLED"),
        (1, "RuntimeError", "REMOTE_TRAINING_UNKNOWN_FAILURE"),
    ),
)
def test_remote_failure_is_typed(
    return_code: int, message: str, expected: str
) -> None:
    result = classify_remote_failure(return_code, stderr=message)
    assert result.failure_code == expected
    assert result.process_signal == (15 if return_code == -15 else None)


def test_remote_failure_redacts_secret_ip_url_and_private_path() -> None:
    line = (
        "api_key=super-secret-value https://signed.example/path?token=abc "
        "198.51.100.2 C:\\Users\\person\\private"
    )
    sanitized = sanitize_log_line(line)
    assert "super-secret" not in sanitized
    assert "198.51.100.2" not in sanitized
    assert "signed.example" not in sanitized
    assert "person" not in sanitized


def _checkpoint_tree(root: Path) -> Path:
    generation = "generation-00000001-aaaaaaaaaaaaaaaa"
    generation_root = root / generation
    generation_root.mkdir(parents=True)
    state = {
        "schema": "atlaslens-phase3f-training-checkpoint-v2",
        "completed_epoch": 0,
        "next_epoch": 1,
        "next_batch_index": 0,
        "optimizer_steps": 1,
        "holdout_open_count": 0,
    }
    artifacts = []
    for name, payload in (
        ("trainable.safetensors", b"model"),
        ("optimizer.pt", b"optimizer"),
        ("scheduler.pt", b"scheduler"),
        ("scaler.pt", b"scaler"),
        ("rng.pt", b"rng"),
        ("training-state.json", (json.dumps(state) + "\n").encode()),
    ):
        path = generation_root / name
        path.write_bytes(payload)
        artifacts.append(
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    atomic_json(
        generation_root / "checkpoint-manifest.json",
        {
            "schema": CHECKPOINT_MANIFEST_SCHEMA,
            "run_id": RUN_ID,
            "dataset_readiness_sha256": READINESS,
            "sealed_assets_sha256": SEALED,
            "training_config_sha256": CONFIG,
            "holdout_open_count": 0,
            "artifacts": artifacts,
        },
    )
    atomic_json(
        root / "latest.json",
        {"schema": CHECKPOINT_POINTER_SCHEMA, "generation": generation},
    )
    return generation_root


def _archive(tree: Path, destination: Path) -> Path:
    with tarfile.open(destination, "w:") as archive:
        for path in sorted(tree.rglob("*")):
            archive.add(path, arcname=path.relative_to(tree).as_posix(), recursive=False)
    return destination


def test_valid_checkpoint_resume_and_duplicate_dedup(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    generation = _checkpoint_tree(tree)
    first = _archive(tree, tmp_path / "first.tar")
    status = inspect_checkpoint_archive(
        first,
        expected_run_id=RUN_ID,
        expected_readiness_sha256=READINESS,
        expected_sealed_assets_sha256=SEALED,
        expected_training_config_sha256=CONFIG,
    )
    assert status.valid and status.next_epoch == 1 and status.optimizer_steps == 1
    store = tmp_path / "store"
    stored = store_verified_checkpoint_archive(
        first,
        store,
        expected_run_id=RUN_ID,
        expected_readiness_sha256=READINESS,
        expected_sealed_assets_sha256=SEALED,
        expected_training_config_sha256=CONFIG,
    )
    duplicate = tmp_path / "duplicate.tar"
    shutil.copy2(stored.archive_path, duplicate)
    again = store_verified_checkpoint_archive(
        duplicate,
        store,
        expected_run_id=RUN_ID,
        expected_readiness_sha256=READINESS,
        expected_sealed_assets_sha256=SEALED,
        expected_training_config_sha256=CONFIG,
    )
    assert again.archive_sha256 == stored.archive_sha256
    assert not duplicate.exists()
    assert generation.is_dir()


def test_corrupt_checkpoint_and_dataset_config_mismatch(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    generation = _checkpoint_tree(tree)
    (generation / "optimizer.pt").write_bytes(b"corrupt")
    archive = _archive(tree, tmp_path / "corrupt.tar")
    with pytest.raises(TrainingRecoveryError, match="TRAINING_CHECKPOINT_HASH_MISMATCH"):
        inspect_checkpoint_archive(
            archive,
            expected_run_id=RUN_ID,
            expected_readiness_sha256=READINESS,
            expected_sealed_assets_sha256=SEALED,
            expected_training_config_sha256=CONFIG,
        )
    clean_tree = tmp_path / "clean"
    _checkpoint_tree(clean_tree)
    clean = _archive(clean_tree, tmp_path / "clean.tar")
    with pytest.raises(TrainingRecoveryError, match="TRAINING_CHECKPOINT_BINDING_MISMATCH"):
        inspect_checkpoint_archive(
            clean,
            expected_run_id=RUN_ID,
            expected_readiness_sha256="e" * 64,
            expected_sealed_assets_sha256=SEALED,
            expected_training_config_sha256=CONFIG,
        )


def test_partial_checkpoint_is_never_valid(tmp_path: Path) -> None:
    archive = tmp_path / "partial.tar"
    partial = tmp_path / "latest.json.partial"
    partial.write_text("{}", encoding="utf-8")
    with tarfile.open(archive, "w:") as output:
        output.add(partial, arcname=partial.name)
    with pytest.raises(TrainingRecoveryError, match="TRAINING_CHECKPOINT_ARCHIVE_INVALID"):
        inspect_checkpoint_archive(
            archive,
            expected_run_id=RUN_ID,
            expected_readiness_sha256=READINESS,
            expected_sealed_assets_sha256=SEALED,
            expected_training_config_sha256=CONFIG,
        )


def test_checkpoint_after_holdout_open_is_not_resumable(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    _checkpoint_tree(tree)
    atomic_json(
        tree / "holdout-state.json",
        {
            "schema": "atlaslens-phase3f-holdout-state-v1",
            "run_id": RUN_ID,
            "holdout_open_count": 1,
        },
    )
    archive = _archive(tree, tmp_path / "holdout.tar")
    status = inspect_checkpoint_archive(
        archive,
        expected_run_id=RUN_ID,
        expected_readiness_sha256=READINESS,
        expected_sealed_assets_sha256=SEALED,
        expected_training_config_sha256=CONFIG,
    )
    assert status.present and not status.valid
    assert status.failure_code == "TRAINING_HOLDOUT_ALREADY_OPENED"
    assert status.holdout_open_count == 1


def test_invalid_local_checkpoint_is_fail_closed(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / "latest.json").write_text("{}", encoding="utf-8")
    status = inspect_local_checkpoint(
        store,
        expected_run_id=RUN_ID,
        expected_readiness_sha256=READINESS,
        expected_sealed_assets_sha256=SEALED,
        expected_training_config_sha256=CONFIG,
    )
    assert status.present and not status.valid
    assert status.failure_code == "TRAINING_CHECKPOINT_INVALID"


def test_training_sources_preserve_full_state_and_holdout_invariant() -> None:
    root = Path(__file__).resolve().parents[3]
    training = (root / "services/api/src/atlaslens_api/phase3f/training.py").read_text()
    supervisor = (root / "scripts/phase3f/supervisor.py").read_text()
    wrapper = (root / "scripts/phase3f/remote_training.py").read_text()
    for marker in ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng.pt"):
        assert marker in training
    assert "CHECKPOINT_INTERVAL_SECONDS: Final = 5 * 60" in training
    assert '"holdout_open_count": 0' in training
    assert "_salvage_remote_failure" in supervisor
    assert "CHECKPOINT_SYNC_SECONDS = 5 * 60" in supervisor
    assert "classify_remote_failure" in wrapper


def test_failure_salvage_downloads_hashes_and_keeps_typed_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_supervisor()
    recovery_tree = tmp_path / "recovery-source" / "recovery"
    recovery_tree.mkdir(parents=True)
    atomic_json(
        recovery_tree / "failure.json",
        {
            "schema": "atlaslens-phase3f-remote-training-failure-v1",
            "failure_code": "REMOTE_TRAINING_CUDA_OOM",
            "process_exit_code": 1,
            "process_signal": None,
        },
    )
    recovery_archive = _archive(recovery_tree.parent, tmp_path / "recovery.tar")
    checkpoint_tree = tmp_path / "checkpoint-source"
    _checkpoint_tree(checkpoint_tree)
    checkpoint_archive = _archive(checkpoint_tree, tmp_path / "checkpoint.tar")

    def fake_command(arguments: list[str], *, timeout_seconds: float) -> None:
        assert timeout_seconds > 0
        if arguments[0] != "scp":
            return
        source = recovery_archive if "failure-recovery.tar" in arguments[-2] else checkpoint_archive
        shutil.copy2(source, Path(arguments[-1]))

    monkeypatch.setattr(module, "_run_command", fake_command)
    monkeypatch.setattr(module, "_run_sha256_command", lambda *_args, **_kwargs: "e" * 64)
    result = module._salvage_remote_failure(
        ["ssh"],
        ["scp"],
        public_ip="192.0.2.1",
        remote_transfer="/workspace/phase3f-transfer",
        attempt_root=tmp_path / "attempt",
        checkpoint_store=tmp_path / "store",
        run_id=RUN_ID,
        readiness_sha256=READINESS,
        sealed_assets_sha256=SEALED,
        config_sha256=CONFIG,
        process_result=module.RemoteProcessResult(1, 5.0),
        timeout_seconds=300,
    )
    assert result.succeeded and result.checkpoint_present
    assert result.failure_code == "REMOTE_TRAINING_CUDA_OOM"
    receipt = json.loads((tmp_path / "attempt/salvage-receipt.json").read_text())
    assert receipt["succeeded"] is True
    assert receipt["checkpoint_sha256"] == result.checkpoint_sha256


def test_nonzero_remote_code_is_not_masked_and_cleanup_still_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_supervisor()
    receipt = OperatorReceipt(
        run_id=RUN_ID,
        attempt_id="f" * 32,
        run_marker=f"atlaslens-phase3f-{RUN_ID}",
        pod_id="receipt-bound-pod",
        supervisor_pid=123,
        stage="running",
        started_at="2026-07-21T00:00:00+00:00",
        finished_at=None,
        max_spend_usd=Decimal("10"),
        soft_stop_usd=Decimal("8"),
        hard_stop_usd=Decimal("9"),
        max_gpu_hourly_usd=Decimal("0.50"),
        max_wall_minutes=345,
        cleanup_verified=False,
    )
    receipt_path = tmp_path / "operator.json"
    write_operator_receipt(receipt_path, receipt)
    request = PodRequest(
        run_marker=receipt.run_marker,
        idempotency_key="one-create-only",
        hourly_cost_usd=Decimal("0.50"),
        max_runtime_seconds=20_700,
        gpu_type_id="NVIDIA L4",
        public_ports=(22,),
    )
    lease = RunPodLease(PodRecord("receipt-bound-pod", receipt.run_marker), request, BudgetPolicy())

    class Client:
        def await_pod_connectivity(self, *_args: object, **_kwargs: object) -> PodConnection:
            return PodConnection(
                pod_id="receipt-bound-pod",
                public_ip="192.0.2.1",
                public_ssh_port=22,
                gpu_display_name="NVIDIA L4",
                hourly_price=Decimal("0.39"),
            )

    commands: list[list[str]] = []
    watched: list[list[str]] = []
    monkeypatch.setattr(
        module,
        "_run_command",
        lambda arguments, **_kwargs: commands.append(arguments),
    )
    def fail_bootstrap(arguments: list[str], **_kwargs: object) -> object:
        watched.append(arguments)
        return module.RemoteProcessResult(1, 10.0)

    monkeypatch.setattr(module, "_run_watched", fail_bootstrap)
    monkeypatch.setattr(
        module,
        "_salvage_remote_failure",
        lambda *_args, **_kwargs: module.SalvageResult(
            True, True, 2, False, None, "REMOTE_TRAINING_NONFINITE_LOSS"
        ),
    )
    with pytest.raises(module.SupervisorExecutionError, match="REMOTE_TRAINING_NONFINITE_LOSS"):
        module._operation(
            Client(),
            lease,
            bundle_root=tmp_path / "bundle",
            key=tmp_path / "key",
            known_hosts=tmp_path / "known",
            download_path=tmp_path / "output.tar",
            run_id=RUN_ID,
            started=time.monotonic(),
            operator_receipt=receipt,
            operator_receipt_path=receipt_path,
            remote_job_seconds=60,
            training_dataset=tmp_path / "dataset.tar",
            checkpoint_store=tmp_path / "store",
            recovery_attempt_root=tmp_path / "attempt",
            readiness_sha256=READINESS,
            sealed_assets_sha256=SEALED,
            config_sha256=CONFIG,
            environment_contract_sha256="e" * 64,
            dependency_lock_sha256="f" * 64,
        )
    assert any("rm -rf /workspace/phase3f-work" in " ".join(row) for row in commands)
    assert len(watched) == 1
    assert "remote_bootstrap.py" in " ".join(watched[0])
    assert "training_job.py" not in " ".join(watched[0])


def _dependency_salvage_fixture(root: Path) -> tuple[Path, ...]:
    documents: dict[str, object] = {
        "failure.json": {
            "schema": "atlaslens-phase3f-remote-training-failure-v1",
            "failure_code": "REMOTE_TRAINING_DEPENDENCY_FAILED",
            "dependency_failure_code": "REMOTE_DEPENDENCY_MODULE_MISSING",
            "failed_module": "pydantic",
            "secrets_included": False,
        },
        "dependency-report.json": {
            "schema": "atlaslens-phase3f-dependency-report-v1",
            "outcome": "failed",
            "dependency_failure_code": "REMOTE_DEPENDENCY_MODULE_MISSING",
            "failed_module": "pydantic",
            "failure_exception_class": "ModuleNotFoundError",
            "secrets_included": False,
        },
        "environment-receipt.json": {
            "schema": "atlaslens-phase3f-environment-receipt-v1",
            "outcome": "failed",
            "dependency_failure_code": "REMOTE_DEPENDENCY_MODULE_MISSING",
            "failed_module": "pydantic",
            "secrets_included": False,
        },
    }
    root.mkdir()
    for name, value in documents.items():
        atomic_json(root / name, value)
    (root / "bootstrap-stdout.log").write_text("", encoding="utf-8")
    (root / "bootstrap-stderr.log").write_text("", encoding="utf-8")
    inventory_rows = []
    for path in sorted(root.iterdir()):
        inventory_rows.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    atomic_json(
        root / "checksum-inventory.json",
        {
            "schema": "atlaslens-phase3f-bootstrap-checksum-inventory-v1",
            "files": inventory_rows,
            "file_count": len(inventory_rows),
            "secrets_included": False,
        },
    )
    return tuple(root.iterdir())


def test_dependency_salvage_requires_report_and_preserves_exact_code(
    tmp_path: Path,
) -> None:
    module = _load_supervisor()
    files = _dependency_salvage_fixture(tmp_path / "recovery")
    assert module._dependency_failure_evidence(files) == (
        "REMOTE_DEPENDENCY_MODULE_MISSING",
        "pydantic",
    )
    incomplete = tuple(path for path in files if path.name != "dependency-report.json")
    with pytest.raises(
        module.SupervisorExecutionError,
        match="REMOTE_DEPENDENCY_SALVAGE_INCOMPLETE",
    ):
        module._dependency_failure_evidence(incomplete)
