from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from atlaslens_api.phase3f import remote_environment as environment
from atlaslens_api.phase3f.remote_environment import (
    DependencyCheck,
    DependencyPreflightResult,
    RemoteEnvironmentContract,
    RemoteEnvironmentError,
    evaluate_remote_environment,
    load_remote_environment_contract,
)
from atlaslens_api.phase3f.training import training_config_sha256
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
    return {
        cast(str, check.distribution): cast(str, check.required_version)
        for check in contract.checks
        if check.distribution is not None
    }


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
    assert result.failure_code == "REMOTE_DEPENDENCY_VERSION_MISMATCH"
    assert result.failed_module == distribution


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
) -> None:
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
        "--vendor-root",
        str(ROOT / ".local" / "vendor" / "megaloc"),
        "--venv-root",
        str(tmp_path / "venv"),
        "--recovery-root",
        str(tmp_path / "recovery"),
        "--timeout-seconds",
        "600",
    ]


def test_pinned_bootstrap_success_uses_one_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_bootstrap()
    contract = _contract()
    base = _evaluate(
        contract,
        expected_interpreter_class="system_python",
        actual_interpreter_class="system_python",
        stage="base",
    )
    full = _evaluate(contract)
    commands: list[tuple[list[str], str]] = []
    monkeypatch.setattr(environment, "evaluate_remote_environment", lambda *_a, **_k: base)

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
    assert len(commands) == 3
    assert "--system-site-packages" in commands[0][0]
    install = commands[1][0]
    assert "--isolated" in install
    assert "--no-deps" in install
    assert "--require-hashes" in install
    assert "--only-binary=:all:" in install
    assert "https://pypi.org/simple" in install
    assert commands[1][0][0] == commands[2][0][0]
    assert full.environment_receipt["cuda_available"] is True


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
    base = _evaluate(
        _contract(),
        expected_interpreter_class="system_python",
        actual_interpreter_class="system_python",
        stage="base",
    )
    stages: list[str] = []
    monkeypatch.setattr(environment, "evaluate_remote_environment", lambda *_a, **_k: base)

    def fake_run(_command: list[str], **kwargs: object) -> tuple[int, str, str, str | None]:
        stage = cast(str, kwargs["stage"])
        stages.append(stage)
        if stage == "installing_locked_packages":
            return return_code, "", "", exception_class
        return 0, "", "", None

    monkeypatch.setattr(module, "_run_logged", fake_run)
    assert module.main(_bootstrap_args(tmp_path)) == return_code
    assert stages == ["creating_venv", "installing_locked_packages"]
    failure = json.loads((tmp_path / "recovery" / "failure.json").read_text())
    assert failure["dependency_failure_code"] == "REMOTE_DEPENDENCY_IMPORT_FAILED"
    assert failure["failed_module"] == "phase3f-training-lock"
    assert failure["exception_class"] == (
        exception_class or "DependencyBootstrapError"
    )
    assert failure["training_started"] is False


def test_bootstrap_preserves_exact_module_with_empty_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_bootstrap()
    contract = _contract()
    base = _evaluate(
        contract,
        expected_interpreter_class="system_python",
        actual_interpreter_class="system_python",
        stage="base",
    )
    missing = _evaluate(contract, missing="pydantic", stage="project")
    monkeypatch.setattr(environment, "evaluate_remote_environment", lambda *_a, **_k: base)

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
