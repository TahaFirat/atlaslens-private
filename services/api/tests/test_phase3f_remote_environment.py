from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from atlaslens_api.phase3f import remote_environment as environment
from atlaslens_api.phase3f.remote_environment import (
    DependencyCheck,
    DependencyPreflightResult,
    FrameworkWheelArtifact,
    FrameworkWheelhouse,
    FrameworkWheelSpec,
    RemoteEnvironmentContract,
    RemoteEnvironmentError,
    canonical_environment_identity,
    capture_torch_identity,
    evaluate_base_runtime_checks,
    evaluate_remote_environment,
    load_remote_environment_contract,
    require_torch_identity_unchanged,
    verify_framework_wheelhouse,
)
from atlaslens_api.phase3f.training import (
    aggregate_training_smoke_failure,
    training_config_sha256,
)
from atlaslens_api.phase3f.training_recovery import atomic_json

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "config" / "phase3f-remote-environment.json"
LOCK_PATH = ROOT / "config" / "phase3f-training-requirements.lock"
BOOTSTRAP_PATH = ROOT / "scripts" / "phase3f" / "remote_bootstrap.py"
CONTROL_PATH = ROOT / "scripts" / "phase3f" / "end_to_end.py"


class _Cuda:
    def __init__(self, *, available: bool = True, name: str = "NVIDIA L4") -> None:
        self._available = available
        self._name = name

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, _index: int) -> str:
        return self._name


def _contract() -> RemoteEnvironmentContract:
    return load_remote_environment_contract(CONTRACT_PATH, LOCK_PATH)


def _versions(contract: RemoteEnvironmentContract) -> dict[str, str]:
    versions = {
        cast(str, check.distribution): cast(str, check.required_version)
        for check in contract.checks
        if check.distribution is not None and check.required_version is not None
    }
    versions.update({"torch": "2.9.1+cu128", "torchvision": "0.24.1+cu128"})
    return versions


def _evaluate(
    contract: RemoteEnvironmentContract,
    *,
    missing: str | None = None,
    broken_vendor: bool = False,
    cuda_available: bool = True,
    cuda_version: str = "12.8",
    version_overrides: dict[str, str] | None = None,
    expected_interpreter_class: str = "project_venv",
    actual_interpreter_class: str = "project_venv",
    stage: str = "full",
    allow_missing_torchvision_companion: bool = False,
) -> DependencyPreflightResult:
    versions = _versions(contract)
    versions.update(version_overrides or {})

    def importer(check: DependencyCheck, _vendor: Path | None) -> object:
        if check.module == missing:
            raise ModuleNotFoundError(check.module)
        if check.source == "vendor_source" and broken_vendor:
            raise RuntimeError("vendor import failed")
        if check.module == "torch":
            return SimpleNamespace(
                cuda=_Cuda(available=cuda_available),
                version=SimpleNamespace(cuda=cuda_version),
            )
        return object()

    return evaluate_remote_environment(
        contract,
        expected_interpreter_class=cast(object, expected_interpreter_class),
        vendor_root=ROOT / ".local" / "vendor" / "megaloc",
        stage=cast(object, stage),
        importer=importer,
        version_reader=versions.__getitem__,
        actual_interpreter_class=actual_interpreter_class,
        python_version=(3, 12),
        allow_missing_torchvision_companion=allow_missing_torchvision_companion,
    )


@pytest.mark.parametrize("module", ("pydantic", "timm"))
def test_missing_dependency_is_exact_even_without_stderr(module: str) -> None:
    contract = _contract()
    if module == "timm":
        contract = replace(
            contract,
            checks=(
                DependencyCheck(
                    name="fixture-timm",
                    module="timm",
                    distribution="timm",
                    required_version="1.0.19",
                    source="locked_project",
                ),
            ),
        )
    result = _evaluate(contract, missing=module, stage="project")
    assert result.failure_code == "REMOTE_DEPENDENCY_MODULE_MISSING"
    assert result.failed_module == module
    assert result.exception_class == "ModuleNotFoundError"
    assert result.report["secrets_included"] is False


@pytest.mark.parametrize(
    ("distribution", "observed"),
    (("torch", "2.6.0"), ("torchvision", "0.21.0")),
)
def test_base_version_mismatch_is_typed(distribution: str, observed: str) -> None:
    result = _evaluate(_contract(), version_overrides={distribution: observed})
    assert result.failure_code == "REMOTE_TORCHVISION_PAIR_UNSUPPORTED"
    assert result.report["check_count"] == 20


def test_observed_live_torch_pair_is_compatibility_approved() -> None:
    result = _evaluate(
        _contract(),
        version_overrides={"torch": "2.9.1+cu128", "torchvision": "0.24.1+cu128"},
    )
    assert result.passed is True
    assert result.environment_receipt["torchvision_pair_compatible"] is True
    assert result.environment_receipt["torch_version"] == "2.9.1+cu128"


def test_live_image_missing_torchvision_is_bootstrap_eligible() -> None:
    result = _evaluate(
        _contract(),
        missing="torchvision",
        stage="base",
        allow_missing_torchvision_companion=True,
    )
    assert result.failure_code == "REMOTE_TORCHVISION_COMPANION_MISSING"
    assert result.environment_receipt["bootstrap_eligible"] is True
    assert result.environment_receipt["torchvision_companion_required"] is True


@pytest.mark.parametrize(
    ("torch_version", "torchvision_version"),
    (("2.7.1", "0.22.1"), ("2.8.0", "0.23.0"), ("2.9.1+cu128", "0.24.1+cu128")),
)
def test_every_official_production_pair_requires_and_passes_the_gate(
    torch_version: str,
    torchvision_version: str,
) -> None:
    result = _evaluate(
        _contract(),
        version_overrides={"torch": torch_version, "torchvision": torchvision_version},
    )
    assert result.passed is True
    assert result.environment_receipt["torchvision_pair_compatible"] is True


def test_full_report_continues_after_first_failure() -> None:
    result = _evaluate(
        _contract(),
        missing="pydantic",
        version_overrides={"torchvision": "0.23.1"},
    )
    assert result.passed is False
    assert result.report["check_count"] == 20
    assert len(cast(list[object], result.report["failure_codes"])) >= 2


def test_base_runtime_probe_collects_ops_after_earlier_failures() -> None:
    base = _evaluate(
        _contract(),
        expected_interpreter_class="system_python",
        actual_interpreter_class="system_python",
        stage="base",
    )
    result = evaluate_base_runtime_checks(
        _contract(), base, torch_module=object(), torchvision_module=object()
    )
    rows = cast(list[dict[str, object]], result.report["base_runtime_checks"])
    assert len(rows) == 11
    assert rows[-1]["check_name"] == "cuda-optimizer-step"
    assert any(
        row["dependency_failure_code"] == "REMOTE_TORCHVISION_OPS_FAILED"
        for row in rows
    )


def test_runtime_identity_binds_actual_versions_and_artifacts() -> None:
    contract = _contract()
    identity, digest = canonical_environment_identity(
        contract,
        python_version="3.12",
        torch_version="2.9.1+cu128",
        torchvision_version="0.24.1+cu128",
        cuda_version="12.8",
        gpu_class="NVIDIA RTX A5000",
        vendor_source_sha256="1" * 64,
        model_sha256="2" * 64,
    )
    assert identity["image"] == contract.image
    assert identity["torch"] == "2.9.1+cu128"
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("check_name", "code"),
    (
        ("torchvision-compiled-ops", "REMOTE_TORCHVISION_OPS_FAILED"),
        ("megaloc-vendor-import", "REMOTE_VENDOR_IMPORT_FAILED"),
        ("model-safetensors-hash", "REMOTE_MODEL_HASH_MISMATCH"),
        ("forward", "REMOTE_CUDA_FORWARD_FAILED"),
        ("backward", "REMOTE_CUDA_BACKWARD_FAILED"),
        ("finite-loss", "REMOTE_TRAINING_SMOKE_FAILED"),
    ),
)
def test_smoke_aggregate_preserves_exact_failure(check_name: str, code: str) -> None:
    rows = [
        {"check_name": "earlier", "failure_code": None},
        {"check_name": check_name, "failure_code": code},
        {"check_name": "later", "failure_code": None},
    ]
    assert aggregate_training_smoke_failure(rows) == code
    assert len(rows) == 3


def test_cuda_unavailable_is_typed() -> None:
    result = _evaluate(_contract(), cuda_available=False)
    assert result.failure_code == "REMOTE_TORCH_CUDA_UNAVAILABLE"
    assert result.failed_module == "torch"
    assert result.environment_receipt["cuda_available"] is False


def test_cuda_abi_mismatch_is_typed() -> None:
    result = _evaluate(_contract(), cuda_version="12.6")
    assert result.failure_code == "REMOTE_TORCH_CUDA_ABI_MISMATCH"
    assert result.failed_module == "torch"
    assert result.environment_receipt["abi_compatible"] is False


def test_wrong_python_interpreter_is_typed() -> None:
    result = _evaluate(
        _contract(),
        expected_interpreter_class="project_venv",
        actual_interpreter_class="system_python",
    )
    assert result.failure_code == "REMOTE_PYTHON_INTERPRETER_MISMATCH"
    assert result.report["check_count"] == 0


def test_vendor_import_failure_is_typed() -> None:
    result = _evaluate(_contract(), broken_vendor=True)
    assert result.failure_code == "REMOTE_VENDOR_IMPORT_FAILED"
    assert result.failed_module == "megaloc_model"


def test_lock_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    changed_lock = tmp_path / "requirements.lock"
    changed_lock.write_bytes(LOCK_PATH.read_bytes() + b"\n")
    with pytest.raises(RemoteEnvironmentError, match="REMOTE_DEPENDENCY_LOCK_MISMATCH"):
        load_remote_environment_contract(CONTRACT_PATH, changed_lock)


def _tiny_wheel_contract(tmp_path: Path) -> tuple[RemoteEnvironmentContract, Path]:
    root = tmp_path / "wheelhouse"
    root.mkdir()
    filename = "torchvision-test-cp312-cp312-manylinux_2_28_x86_64.whl"
    payload = b"official-wheel-fixture"
    path = root / filename
    path.write_bytes(payload)
    digest = environment.hashlib.sha256(payload).hexdigest()
    spec = FrameworkWheelSpec(
        distribution="torchvision",
        version="0.24.1+cu128",
        filename=filename,
        size_bytes=len(payload),
        sha256=digest,
        source="official_pytorch",
    )
    return (
        replace(
            _contract(),
            framework_wheel_artifacts=(spec,),
            torchvision_companion_filename=filename,
            torchvision_companion_size_bytes=len(payload),
            torchvision_companion_sha256=digest,
        ),
        root,
    )


def test_hash_pinned_framework_wheelhouse_is_verified(tmp_path: Path) -> None:
    contract, root = _tiny_wheel_contract(tmp_path)
    wheelhouse = verify_framework_wheelhouse(contract, root)
    assert wheelhouse.companion.sha256 == contract.torchvision_companion_sha256
    assert wheelhouse.inventory["package_index_resolution_on_pod"] is False


def test_missing_or_changed_companion_wheel_is_typed(tmp_path: Path) -> None:
    contract, root = _tiny_wheel_contract(tmp_path)
    companion = root / contract.torchvision_companion_filename
    companion.unlink()
    with pytest.raises(
        RemoteEnvironmentError, match="REMOTE_TORCHVISION_COMPANION_MISSING"
    ):
        verify_framework_wheelhouse(contract, root)
    companion.write_bytes(b"changed")
    with pytest.raises(
        RemoteEnvironmentError, match="REMOTE_TORCHVISION_WHEEL_HASH_MISMATCH"
    ):
        verify_framework_wheelhouse(contract, root)


def test_torch_identity_detects_post_install_change(tmp_path: Path) -> None:
    module_path = tmp_path / "torch.py"
    module_path.write_text("identity", encoding="ascii")
    module = SimpleNamespace(
        version=SimpleNamespace(cuda="12.8"),
        __version__="2.9.1+cu128",
        __file__=str(module_path),
    )
    before = capture_torch_identity(
        torch_module=module,
        version_reader=lambda _name: "2.9.1+cu128",
        module_path=module_path,
    )
    after = capture_torch_identity(
        torch_module=module,
        version_reader=lambda _name: "2.9.1+cu128",
        module_path=module_path,
    )
    require_torch_identity_unchanged(before, after)
    changed = dict(after)
    changed["version"] = "2.9.2+cu128"
    with pytest.raises(RemoteEnvironmentError, match="REMOTE_TORCH_IDENTITY_CHANGED"):
        require_torch_identity_unchanged(before, changed)


def test_real_project_import_graph_passes_without_cuda_probe() -> None:
    contract = _contract()
    result = evaluate_remote_environment(
        contract,
        expected_interpreter_class="project_venv",
        vendor_root=ROOT / ".local" / "vendor" / "megaloc",
        stage="project",
    )
    assert result.passed is True
    assert result.report["check_count"] == 18
    assert result.environment_receipt["cuda_available"] is None


def _load_bootstrap() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase3f_remote_bootstrap_test", BOOTSTRAP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_control() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "phase3f_remote_environment_plan_test", CONTROL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_environment_plan_is_mutation_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_bootstrap_framework(monkeypatch, tmp_path)
    module = _load_control()
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    plan = module.remote_environment_plan(ROOT)
    assert plan["schema"] == "atlaslens-phase3f-remote-environment-plan-v1"
    assert plan["all_local_imports_passed"] is True
    assert plan["local_blockers"] == []
    assert plan["ready_for_live_bootstrap"] is True
    assert plan["linux_verification_status"] == "not_verified_cache_absent"
    assert plan["runpod_api_calls"] == 0
    assert plan["cloud_mutations"] == 0
    assert plan["network_calls"] == 0
    assert plan["torch_download_prohibited"] is True
    assert plan["compatibility_smoke_required"] is True
    assert plan["historical_observed_torch"] == "2.9.1+cu128"
    assert plan["framework_wheelhouse_artifact_count"] == 1
    assert plan["torchvision_companion_present"] is True
    assert plan["framework_transfer_ready"] is True
    assert plan["package_index_resolution_on_pod"] is False
    assert plan["create_attempts"] == 0
    assert plan["dataset_writes"] == 0


def test_remote_environment_plan_blocks_before_cloud_when_companion_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*_args: object, **_kwargs: object) -> object:
        raise RemoteEnvironmentError("REMOTE_TORCHVISION_COMPANION_MISSING")

    monkeypatch.setattr(environment, "verify_framework_wheelhouse", missing)
    module = _load_control()
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    plan = module.remote_environment_plan(ROOT)
    assert plan["local_blockers"] == ["REMOTE_TORCHVISION_COMPANION_MISSING"]
    assert plan["ready_for_live_bootstrap"] is False
    assert plan["runpod_api_calls"] == 0
    assert plan["cloud_mutations"] == 0
    assert plan["network_calls"] == 0
    assert plan["create_attempts"] == 0
    assert plan["dataset_writes"] == 0


def test_environment_contract_hash_binds_checkpoint_configuration() -> None:
    legacy = training_config_sha256(seed=20260720, max_epochs=8)
    bound = training_config_sha256(
        seed=20260720,
        max_epochs=8,
        environment_contract_sha256=_contract().contract_sha256,
    )
    assert bound != legacy
    assert len(bound) == 64


def test_bootstrap_environment_excludes_cloud_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_bootstrap()
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", "MLY-secret")
    monkeypatch.setenv("RUNPOD_API_KEY", "runpod-secret")
    safe = module._safe_environment()
    assert "MAPILLARY_ACCESS_TOKEN" not in safe
    assert "RUNPOD_API_KEY" not in safe
    assert safe["HF_HUB_OFFLINE"] == "1"
    assert safe["TRANSFORMERS_OFFLINE"] == "1"
    assert "MLY-secret" not in repr(safe)
    assert "runpod-secret" not in repr(safe)


def _bootstrap_args(tmp_path: Path) -> list[str]:
    contract = _contract()
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(exist_ok=True)
    inventory = wheelhouse / "framework-checksum-inventory.json"
    inventory.write_text("{}\n", encoding="ascii")
    return [
        "--repository-root",
        str(ROOT),
        "--contract",
        str(CONTRACT_PATH),
        "--requirements-lock",
        str(LOCK_PATH),
        "--expected-contract-sha256",
        contract.contract_sha256,
        "--expected-lock-sha256",
        contract.requirements_lock_sha256,
        "--wheelhouse",
        str(wheelhouse),
        "--wheelhouse-inventory",
        str(inventory),
        "--vendor-root",
        str(ROOT / ".local" / "vendor" / "megaloc"),
        "--model",
        str(ROOT / ".local" / "models" / "phase6c" / "megaloc" / "model.safetensors"),
        "--venv-root",
        str(tmp_path / "venv"),
        "--recovery-root",
        str(tmp_path / "recovery"),
        "--timeout-seconds",
        "600",
    ]


def _mock_bootstrap_framework(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FrameworkWheelhouse:
    contract = _contract()
    wheelhouse_root = tmp_path / "wheelhouse"
    wheelhouse_root.mkdir(exist_ok=True)
    companion_path = wheelhouse_root / contract.torchvision_companion_filename
    companion_path.write_bytes(b"companion-fixture")
    companion = FrameworkWheelArtifact(
        path=companion_path,
        distribution="torchvision",
        version=contract.torchvision_companion_version,
        filename=contract.torchvision_companion_filename,
        size_bytes=contract.torchvision_companion_size_bytes,
        sha256=contract.torchvision_companion_sha256,
        source="official_pytorch",
    )
    wheelhouse = FrameworkWheelhouse(
        root=wheelhouse_root,
        artifacts=(companion,),
        inventory={"schema": "fixture"},
        inventory_sha256="9" * 64,
        companion=companion,
    )
    identity = {
        "schema": "atlaslens-phase3f-torch-identity-v1",
        "version": "2.9.1+cu128",
        "cuda": "12.8",
        "python_abi": "cp312",
        "module_path_sha256": "1" * 64,
        "module_file_sha256": "2" * 64,
        "module_device": 1,
        "module_inode": 2,
        "secrets_included": False,
    }
    monkeypatch.setattr(
        environment, "verify_framework_wheelhouse", lambda *_a, **_k: wheelhouse
    )
    monkeypatch.setattr(environment, "capture_torch_identity", lambda: identity)
    return wheelhouse


def test_live_image_shape_installs_companion_then_reaches_cuda_megaloc_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_bootstrap()
    contract = _contract()
    wheelhouse = _mock_bootstrap_framework(monkeypatch, tmp_path)
    base = _evaluate(
        contract,
        missing="torchvision",
        expected_interpreter_class="system_python",
        actual_interpreter_class="system_python",
        stage="base",
        allow_missing_torchvision_companion=True,
    )
    full = _evaluate(contract)
    commands: list[tuple[list[str], str]] = []
    monkeypatch.setattr(environment, "evaluate_remote_environment", lambda *_a, **_k: base)
    monkeypatch.setattr(environment, "evaluate_base_runtime_checks", lambda *_a, **_k: base)
    monkeypatch.setattr(environment, "require_dependency_report", lambda *_a, **_k: {})
    monkeypatch.setattr(environment, "require_environment_receipt", lambda *_a, **_k: {})

    def fake_run(command: list[str], **kwargs: object) -> tuple[int, str, str, None]:
        stage = cast(str, kwargs["stage"])
        commands.append((command, stage))
        if stage == "validating_project_imports":
            recovery = tmp_path / "recovery"
            atomic_json(recovery / "dependency-report.json", full.report)
            atomic_json(recovery / "environment-receipt.json", full.environment_receipt)
        return 0, "", "", None

    monkeypatch.setattr(module, "_run_logged", fake_run)
    assert module.main(_bootstrap_args(tmp_path)) == 0
    assert len(commands) == 5
    assert "--system-site-packages" in commands[0][0]
    companion_install = commands[1][0]
    assert commands[1][1] == "installing_torchvision_companion"
    assert "--no-deps" in companion_install
    assert "--no-index" in companion_install
    assert str(wheelhouse.companion.path) in companion_install
    locked_install = commands[2][0]
    assert "--isolated" in locked_install
    assert "--no-deps" in locked_install
    assert "--require-hashes" in locked_install
    assert "--only-binary=:all:" in locked_install
    assert "--no-index" in locked_install
    assert "--find-links" in locked_install
    assert "https://pypi.org/simple" not in locked_install
    assert commands[1][0][0] == commands[2][0][0]
    assert commands[2][0][0] == commands[3][0][0]
    assert commands[3][0][0] == commands[4][0][0]
    assert "--expected-torch-identity" in commands[3][0]
    assert "--framework-install-receipt" in commands[3][0]
    assert commands[4][1] == "validating_training_smoke"
    assert full.environment_receipt["cuda_available"] is True


def test_cached_pinned_image_runs_offline_companion_and_cuda_megaloc_smoke(
    tmp_path: Path,
) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("local Docker is unavailable")
    contract = _contract()
    inspect = subprocess.run(  # noqa: S603
        [docker, "image", "inspect", contract.image],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if inspect.returncode != 0:
        pytest.skip("pinned RunPod image is not present in the local Docker cache")
    wheelhouse = verify_framework_wheelhouse(
        contract, ROOT / contract.framework_wheelhouse_local_path
    )
    inventory = tmp_path / "framework-checksum-inventory.json"
    atomic_json(inventory, wheelhouse.inventory)
    model = ROOT / ".local" / "models" / "phase6c" / "megaloc" / "model.safetensors"
    vendor = ROOT / ".local" / "vendor" / "megaloc"
    if not model.is_file() or not vendor.is_dir():
        pytest.skip("local MegaLoc artifacts are unavailable")
    command = [
        docker,
        "run",
        "--rm",
        "--network=none",
        "--gpus=all",
        "--mount",
        f"type=bind,source={ROOT},target=/repo,readonly",
        "--mount",
        f"type=bind,source={wheelhouse.root},target=/wheelhouse,readonly",
        "--mount",
        f"type=bind,source={tmp_path},target=/state",
        contract.image,
        "python",
        "/repo/scripts/phase3f/remote_bootstrap.py",
        "--repository-root",
        "/repo",
        "--contract",
        "/repo/config/phase3f-remote-environment.json",
        "--requirements-lock",
        "/repo/config/phase3f-training-requirements.lock",
        "--expected-contract-sha256",
        contract.contract_sha256,
        "--expected-lock-sha256",
        contract.requirements_lock_sha256,
        "--wheelhouse",
        "/wheelhouse",
        "--wheelhouse-inventory",
        "/state/framework-checksum-inventory.json",
        "--vendor-root",
        "/repo/.local/vendor/megaloc",
        "--model",
        "/repo/.local/models/phase6c/megaloc/model.safetensors",
        "--venv-root",
        "/state/venv",
        "--recovery-root",
        "/state/recovery",
        "--timeout-seconds",
        "600",
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=630,
    )
    typed_lines = [
        line
        for line in completed.stdout.splitlines()
        if re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", line)
    ]
    assert completed.returncode == 0, (
        f"offline container bootstrap failed: "
        f"{typed_lines[-1] if typed_lines else 'TYPED_CODE_MISSING'}"
    )
    recovery = tmp_path / "recovery"
    framework = json.loads(
        (recovery / "framework-install-receipt.json").read_text(encoding="utf-8")
    )
    smoke = json.loads(
        (recovery / "training-smoke-report.json").read_text(encoding="utf-8")
    )
    assert framework["outcome"] == "passed"
    assert framework["torch_identity_unchanged"] is True
    assert framework["package_index_resolution_count"] == 0
    assert framework["torch_download_count"] == 0
    assert smoke["outcome"] == "passed"
    assert smoke["failure_code"] is None


@pytest.mark.parametrize(
    ("return_code", "exception_class"), ((7, None), (124, "TimeoutExpired"))
)
def test_install_failure_or_timeout_stops_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    return_code: int,
    exception_class: str | None,
) -> None:
    module = _load_bootstrap()
    _mock_bootstrap_framework(monkeypatch, tmp_path)
    base = _evaluate(
        _contract(),
        missing="torchvision",
        expected_interpreter_class="system_python",
        actual_interpreter_class="system_python",
        stage="base",
        allow_missing_torchvision_companion=True,
    )
    stages: list[str] = []
    monkeypatch.setattr(environment, "evaluate_remote_environment", lambda *_a, **_k: base)
    monkeypatch.setattr(environment, "evaluate_base_runtime_checks", lambda *_a, **_k: base)

    def fake_run(_command: list[str], **kwargs: object) -> tuple[int, str, str, str | None]:
        stage = cast(str, kwargs["stage"])
        stages.append(stage)
        if stage == "installing_torchvision_companion":
            return return_code, "", "", exception_class
        return 0, "", "", None

    monkeypatch.setattr(module, "_run_logged", fake_run)
    assert module.main(_bootstrap_args(tmp_path)) == return_code
    assert stages == ["creating_venv", "installing_torchvision_companion"]
    failure = json.loads((tmp_path / "recovery" / "failure.json").read_text())
    assert failure["dependency_failure_code"] == "REMOTE_TORCHVISION_INSTALL_FAILED"
    assert failure["failed_module"] == "torchvision"
    assert failure["exception_class"] == (
        exception_class or "DependencyBootstrapError"
    )
    assert failure["training_started"] is False


def test_bootstrap_wheel_hash_mismatch_stops_before_any_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_bootstrap()

    def mismatch(*_args: object, **_kwargs: object) -> object:
        raise RemoteEnvironmentError("REMOTE_TORCHVISION_WHEEL_HASH_MISMATCH")

    monkeypatch.setattr(environment, "verify_framework_wheelhouse", mismatch)
    monkeypatch.setattr(
        module,
        "_run_logged",
        lambda *_args, **_kwargs: pytest.fail("install must not start"),
    )
    assert module.main(_bootstrap_args(tmp_path)) == 90
    failure = json.loads((tmp_path / "recovery" / "failure.json").read_text())
    assert failure["dependency_failure_code"] == (
        "REMOTE_TORCHVISION_WHEEL_HASH_MISMATCH"
    )
    assert failure["failed_module"] == "torchvision"


def test_bootstrap_preserves_exact_module_with_empty_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_bootstrap()
    _mock_bootstrap_framework(monkeypatch, tmp_path)
    contract = _contract()
    base = _evaluate(
        contract,
        expected_interpreter_class="system_python",
        actual_interpreter_class="system_python",
        stage="base",
    )
    missing = _evaluate(contract, missing="pydantic", stage="project")
    monkeypatch.setattr(environment, "evaluate_remote_environment", lambda *_a, **_k: base)
    monkeypatch.setattr(environment, "evaluate_base_runtime_checks", lambda *_a, **_k: base)

    def fake_run(_command: list[str], **kwargs: object) -> tuple[int, str, str, None]:
        if kwargs["stage"] == "validating_project_imports":
            recovery = tmp_path / "recovery"
            atomic_json(recovery / "dependency-report.json", missing.report)
            atomic_json(recovery / "environment-receipt.json", missing.environment_receipt)
            return 92, "", "", None
        return 0, "", "", None

    monkeypatch.setattr(module, "_run_logged", fake_run)
    assert module.main(_bootstrap_args(tmp_path)) == 92
    failure = json.loads((tmp_path / "recovery" / "failure.json").read_text())
    assert failure["dependency_failure_code"] == "REMOTE_DEPENDENCY_MODULE_MISSING"
    assert failure["failed_module"] == "pydantic"
    assert (tmp_path / "recovery" / "bootstrap-stderr.log").read_text() == ""
    assert failure["secrets_included"] is False
