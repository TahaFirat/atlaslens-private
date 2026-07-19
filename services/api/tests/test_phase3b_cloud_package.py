from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config" / "cloud" / "runpod-phase3b-job-v1.json"
LOCK_PATH = ROOT / "services" / "api" / "uv.lock"
SCRIPT_PATH = ROOT / "scripts" / "phase3b" / "cloud_job.py"
DOCKERFILE_PATH = ROOT / "infra" / "phase3b" / "Dockerfile"


def _load_cloud_job() -> ModuleType:
    spec = importlib.util.spec_from_file_location("atlaslens_phase3b_cloud_job", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLOUD_JOB = _load_cloud_job()


def _config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_canonical_config_is_disabled_bounded_and_offline() -> None:
    result = CLOUD_JOB.validate_config(_config())

    assert result["status"] == "passed"
    assert result["execution_enabled"] is False
    assert result["network_calls"] == 0
    assert result["runpod_api_calls"] == 0


def test_dependency_lock_hash_and_container_digest_are_pinned() -> None:
    config = _config()
    actual_lock_hash = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert config["container"]["dependency_lock_sha256"] == actual_lock_hash
    assert f"API_LOCK_SHA256={actual_lock_hash}" in dockerfile
    assert config["container"]["base_image"] in dockerfile
    assert "--frozen" in dockerfile


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value["budget"].update({"hard_limit_usd": 14}), "budget_limits"),
        (
            lambda value: value["runtime"].update({"hard_deadline_seconds": 3600}),
            "deadline_below_maximum_hours",
        ),
        (
            lambda value: value["paths"].update(
                {"output_root": "/workspace/input/output/<run-id>"}
            ),
            "mutable_under_input",
        ),
        (
            lambda value: value["paths"].update(
                {"manifest": "/workspace/input/manifests/corpus.jsonl"}
            ),
            "manifest_outside_corpus",
        ),
        (
            lambda value: value["paths"].update({"input_read_only": False}),
            "input_not_read_only",
        ),
        (
            lambda value: value["controls"].update({"network_calls_in_dry_run": True}),
            "control_network_calls_in_dry_run",
        ),
    ],
)
def test_invalid_runtime_or_budget_config_fails_closed(mutation: Any, code: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)

    with pytest.raises(CLOUD_JOB.CloudJobError, match=code):
        CLOUD_JOB.validate_config(config)


def test_execution_enablement_requires_matching_approval_state() -> None:
    config = copy.deepcopy(_config())
    config["execution_enabled"] = True

    with pytest.raises(CLOUD_JOB.CloudJobError, match="approval_state"):
        CLOUD_JOB.validate_config(config)

    config["approval_state"] = "USER_APPROVED"
    assert CLOUD_JOB.validate_config(config)["execution_enabled"] is True


def test_dry_run_builds_exact_explicit_offline_pipeline_command(monkeypatch: Any) -> None:
    def fail_if_spawned(*_args: object, **_kwargs: object) -> None:
        pytest.fail("dry-run attempted to spawn a child process")

    monkeypatch.setattr(CLOUD_JOB.subprocess, "Popen", fail_if_spawned)
    result = CLOUD_JOB.run_job(
        _config(),
        run_id="synthetic-001",
        execute=False,
        approval_receipt=None,
    )
    command = result["command"]

    assert result["status"] == "dry-run"
    assert result["network_calls"] == 0
    assert result["runpod_api_calls"] == 0
    assert command[:2] == ["atlaslens-corpus", "run-pipeline"]
    for option in (
        "--manifest",
        "--manifest-schema",
        "--source-policy",
        "--corpus-root",
        "--work-dir",
        "--output-dir",
        "--checkpoint",
        "--config",
        "--resume",
        "--dry-run",
    ):
        assert option in command
    assert all(".env" not in value for value in command)


def test_execute_is_refused_while_canonical_config_is_disabled() -> None:
    with pytest.raises(CLOUD_JOB.CloudJobError, match="execution_not_enabled"):
        CLOUD_JOB.run_job(
            _config(),
            run_id="synthetic-001",
            execute=True,
            approval_receipt="approved-fixture",
        )


def test_cli_uses_stable_success_and_refusal_exit_codes(capsys: Any) -> None:
    assert CLOUD_JOB.main(["validate-config", "--config", str(CONFIG_PATH)]) == CLOUD_JOB.EXIT_OK
    success = json.loads(capsys.readouterr().out)
    assert success["status"] == "passed"

    exit_code = CLOUD_JOB.main(
        [
            "run",
            "--config",
            str(CONFIG_PATH),
            "--run-id",
            "synthetic-001",
            "--execute",
            "--approval-receipt",
            "approved-fixture",
        ]
    )
    refusal = json.loads(capsys.readouterr().out)
    assert exit_code == CLOUD_JOB.EXIT_REFUSED
    assert refusal == {"status": "refused", "code": "execution_not_enabled"}


def test_approved_execution_still_refuses_unset_provider_and_holdout() -> None:
    config = copy.deepcopy(_config())
    config["execution_enabled"] = True
    config["approval_state"] = "USER_APPROVED"

    with pytest.raises(CLOUD_JOB.CloudJobError, match="production_descriptor_provider_required"):
        CLOUD_JOB.run_job(
            config,
            run_id="synthetic-001",
            execute=True,
            approval_receipt="approved-fixture",
        )


def test_explicit_production_pipeline_settings_validate_and_test_provider_refuses() -> None:
    config = copy.deepcopy(_config())
    pipeline = config["pipeline"]
    pipeline.update(
        {
            "production_descriptor_provider": "rights-approved-adapter-v1",
            "descriptor_batch_size": 64,
            "index_backend": "faiss",
            "index_version": "turkiye-reference-v1",
            "shard_size": 10000,
            "holdout_path": "/workspace/input/holdout/locked.jsonl",
            "expected_holdout_hash": "a" * 64,
            "abstain_if_distance_gt": 0.4,
        }
    )

    CLOUD_JOB.validate_execution_pipeline_settings(config)
    pipeline["production_descriptor_provider"] = "deterministic-test-provider"
    with pytest.raises(CLOUD_JOB.CloudJobError, match="test_provider_forbidden"):
        CLOUD_JOB.validate_execution_pipeline_settings(config)


def test_inventory_is_deterministic_and_hashes_only_regular_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint"
    output.mkdir()
    checkpoint.mkdir()
    (output / "index.bin").write_bytes(b"synthetic-index")
    (checkpoint / "state.json").write_text('{"state":"INDEX_BUILT"}\n', encoding="utf-8")

    first = CLOUD_JOB.write_artifact_inventory(output, checkpoint)
    second = CLOUD_JOB.write_artifact_inventory(output, checkpoint)
    inventory = json.loads((output / "artifact-inventory.json").read_text(encoding="utf-8"))
    checksum_lines = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()

    assert first == second == inventory
    assert inventory["artifact_count"] == 2
    assert {record["root"] for record in inventory["artifacts"]} == {
        "output",
        "checkpoint",
    }
    assert len(checksum_lines) == 2
    assert all(len(line.split("  ", 1)[0]) == 64 for line in checksum_lines)
    assert str(tmp_path) not in json.dumps(inventory)


def test_inventory_refuses_symlinks(tmp_path: Path) -> None:
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint"
    output.mkdir()
    checkpoint.mkdir()
    external = tmp_path / "external.bin"
    external.write_bytes(b"synthetic")
    try:
        (output / "link.bin").symlink_to(external)
    except OSError:
        pytest.skip("host cannot create a test symlink")

    with pytest.raises(CLOUD_JOB.CloudJobError, match="artifact_symlink_refused"):
        CLOUD_JOB.write_artifact_inventory(output, checkpoint)


def test_job_receipts_hash_approval_and_do_not_claim_provider_billing(
    tmp_path: Path,
) -> None:
    approval = "operator-approval-fixture"
    result = CLOUD_JOB.SupervisorResult(
        exit_code=0,
        reason="completed",
        checkpoint_hooks=2,
        elapsed_seconds=12.5,
    )
    CLOUD_JOB.write_job_receipts(
        tmp_path,
        _config(),
        run_id="synthetic-001",
        approval_receipt=approval,
        result=result,
    )
    supervisor = json.loads((tmp_path / "supervisor-receipt.json").read_text(encoding="utf-8"))
    cost = json.loads((tmp_path / "cost-receipt.json").read_text(encoding="utf-8"))

    assert supervisor["approval_receipt_sha256"] == hashlib.sha256(approval.encode()).hexdigest()
    assert approval not in json.dumps(supervisor)
    assert cost["provider_billed_cost_usd"] is None
    assert cost["provider_billing_observed"] is False
    assert cost["elapsed_compute_estimate_usd"] > 0


def test_supervisor_enforces_deadline_and_writes_checkpoint_receipts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    result = CLOUD_JOB.supervise_command(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        checkpoint_root=checkpoint,
        deadline_seconds=0.3,
        checkpoint_interval_seconds=0.05,
        termination_grace_seconds=1,
    )
    receipt = json.loads((checkpoint / "supervisor-final-receipt.json").read_text(encoding="utf-8"))
    hook = json.loads(
        (checkpoint / "supervisor-checkpoint-request.json").read_text(encoding="utf-8")
    )

    assert result.exit_code == CLOUD_JOB.EXIT_DEADLINE
    assert result.reason == "deadline_exceeded"
    assert result.checkpoint_hooks >= 1
    assert receipt["reason"] == "deadline_exceeded"
    assert receipt["exit_code"] == CLOUD_JOB.EXIT_DEADLINE
    assert hook["reason"] == "deadline"


def test_supervisor_reports_child_failure_without_marking_completion(tmp_path: Path) -> None:
    result = CLOUD_JOB.supervise_command(
        [sys.executable, "-c", "raise SystemExit(7)"],
        checkpoint_root=tmp_path / "checkpoint",
        deadline_seconds=5,
        checkpoint_interval_seconds=1,
        termination_grace_seconds=1,
    )

    assert result.exit_code == 7
    assert result.reason == "child_failed"


def test_cleanup_defaults_to_preview_and_requires_exact_confirmation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_id = "synthetic-cleanup"

    def map_runtime_path(template: object, received_run_id: str) -> Path:
        assert received_run_id == run_id
        category = PurePosixPath(str(template)).parent.name
        return tmp_path / category / received_run_id

    monkeypatch.setattr(CLOUD_JOB, "_local_path", map_runtime_path)
    targets = [
        map_runtime_path(_config()["paths"][name], run_id)
        for name in ("work_root", "output_root", "checkpoint_root")
    ]
    for target in targets:
        target.mkdir(parents=True)
        (target / "synthetic.txt").write_text("temporary", encoding="utf-8")

    preview = CLOUD_JOB.cleanup_run(_config(), run_id=run_id, execute=False, confirm_run_id=None)
    assert preview == {
        "status": "dry-run",
        "run_id": run_id,
        "targets": ["<workspace-path>"] * 3,
        "target_count": 3,
    }
    assert all(target.exists() for target in targets)

    with pytest.raises(CLOUD_JOB.CloudJobError, match="cleanup_confirmation_mismatch"):
        CLOUD_JOB.cleanup_run(_config(), run_id=run_id, execute=True, confirm_run_id="wrong-run")
    deleted = CLOUD_JOB.cleanup_run(_config(), run_id=run_id, execute=True, confirm_run_id=run_id)
    assert deleted["status"] == "deleted"
    assert all(not target.exists() for target in targets)


def test_offline_child_environment_drops_unrelated_values(monkeypatch: Any) -> None:
    monkeypatch.setenv("UNRELATED_PRIVATE_VALUE", "fixture-only")
    environment = CLOUD_JOB._offline_environment()

    assert "UNRELATED_PRIVATE_VALUE" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["UV_OFFLINE"] == "1"
    assert environment["PIP_NO_INDEX"] == "1"
