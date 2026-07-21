"""Deterministic, secret-free Phase 3F remote dependency contract checks."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, cast

REMOTE_ENVIRONMENT_CONTRACT_SCHEMA: Final = (
    "atlaslens-phase3f-remote-environment-contract-v1"
)
REMOTE_DEPENDENCY_REPORT_SCHEMA: Final = "atlaslens-phase3f-dependency-report-v1"
REMOTE_ENVIRONMENT_RECEIPT_SCHEMA: Final = "atlaslens-phase3f-environment-receipt-v1"
REMOTE_ENVIRONMENT_PLAN_SCHEMA: Final = "atlaslens-phase3f-remote-environment-plan-v1"
DEPENDENCY_FAILURE_CODES: Final = frozenset(
    {
        "REMOTE_DEPENDENCY_MODULE_MISSING",
        "REMOTE_DEPENDENCY_VERSION_MISMATCH",
        "REMOTE_DEPENDENCY_IMPORT_FAILED",
        "REMOTE_PYTHON_INTERPRETER_MISMATCH",
        "REMOTE_TORCH_CUDA_UNAVAILABLE",
        "REMOTE_TORCH_CUDA_ABI_MISMATCH",
        "REMOTE_VENDOR_IMPORT_FAILED",
        "REMOTE_DEPENDENCY_LOCK_MISMATCH",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_GPU = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,127}$")
_LOCK_LINE = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9-]*)==(?P<version>[A-Za-z0-9.+!-]+) "
    r"--hash=sha256:(?P<sha256>[0-9a-f]{64})$"
)
_REMOTE_IMAGE = (
    "runpod/pytorch@sha256:"
    "60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f"
)


class RemoteEnvironmentError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str = "REMOTE_DEPENDENCY_LOCK_MISMATCH") -> None:
    if not condition:
        raise RemoteEnvironmentError(code)


def sha256_file(path: Path, *, max_bytes: int = 1024 * 1024) -> str:
    _require(path.is_file() and not path.is_symlink())
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            size += len(chunk)
            _require(size <= max_bytes)
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    name: str
    module: str
    distribution: str | None
    required_version: str | None
    source: Literal["base_image", "locked_project", "project_source", "vendor_source"]


@dataclass(frozen=True, slots=True)
class RemoteEnvironmentContract:
    image: str
    platform: str
    python_major: int
    python_minor: int
    torch_requirement: str
    torchvision_requirement: str
    cuda_requirement: str
    official_index_url: str
    bootstrap_timeout_seconds: int
    requirements_lock_sha256: str
    system_site_packages_required: bool
    checks: tuple[DependencyCheck, ...]
    contract_sha256: str


@dataclass(frozen=True, slots=True)
class DependencyPreflightResult:
    passed: bool
    failure_code: str | None
    failed_module: str | None
    exception_class: str | None
    report: dict[str, object]
    environment_receipt: dict[str, object]


ModuleImporter = Callable[[DependencyCheck, Path | None], object]
VersionReader = Callable[[str], str]


def interpreter_class() -> Literal["system_python", "project_venv"]:
    return "project_venv" if sys.prefix != sys.base_prefix else "system_python"


def _read_json(path: Path) -> dict[str, object]:
    try:
        _require(path.is_file() and not path.is_symlink())
        _require(path.stat().st_size <= 1024 * 1024)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteEnvironmentError("REMOTE_DEPENDENCY_LOCK_MISMATCH") from exc
    _require(isinstance(value, dict))
    return cast(dict[str, object], value)


def _locked_rows(path: Path) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RemoteEnvironmentError("REMOTE_DEPENDENCY_LOCK_MISMATCH") from exc
    _require(bool(lines) and len(lines) <= 128)
    for line in lines:
        matched = _LOCK_LINE.fullmatch(line)
        _require(matched is not None)
        name = cast(re.Match[str], matched).group("name")
        _require(name not in rows and name not in {"torch", "torchvision"})
        rows[name] = (
            cast(re.Match[str], matched).group("version"),
            cast(re.Match[str], matched).group("sha256"),
        )
    return rows


def load_remote_environment_contract(
    contract_path: Path,
    requirements_path: Path,
    *,
    expected_contract_sha256: str | None = None,
    expected_lock_sha256: str | None = None,
) -> RemoteEnvironmentContract:
    contract_sha256 = sha256_file(contract_path)
    lock_sha256 = sha256_file(requirements_path)
    if expected_contract_sha256 is not None:
        _require(
            bool(_SHA256.fullmatch(expected_contract_sha256))
            and contract_sha256 == expected_contract_sha256
        )
    if expected_lock_sha256 is not None:
        _require(
            bool(_SHA256.fullmatch(expected_lock_sha256))
            and lock_sha256 == expected_lock_sha256
        )
    value = _read_json(contract_path)
    _require(value.get("schema") == REMOTE_ENVIRONMENT_CONTRACT_SCHEMA)
    _require(value.get("requirements_lock_sha256") == lock_sha256)
    _require(value.get("python_requirement") == "==3.12.*")
    _require(value.get("torch_requirement") == "==2.7.1")
    _require(value.get("torchvision_requirement") == "==0.22.1")
    _require(value.get("cuda_requirement") == "12.8")
    _require(value.get("platform") == "linux_x86_64")
    _require(value.get("system_site_packages_required") is True)
    _require(value.get("official_index_url") == "https://pypi.org/simple")
    _require(value.get("bootstrap_timeout_seconds") == 600)
    _require(value.get("forbidden_locked_distributions") == ["torch", "torchvision"])
    _require(value.get("requirements_lock") == "config/phase3f-training-requirements.lock")
    image = value.get("image")
    raw_checks = value.get("checks")
    _require(image == _REMOTE_IMAGE)
    if not isinstance(raw_checks, list) or len(raw_checks) != 20:
        raise RemoteEnvironmentError("REMOTE_DEPENDENCY_LOCK_MISMATCH")
    checks: list[DependencyCheck] = []
    for row in raw_checks:
        _require(isinstance(row, dict))
        raw = cast(dict[str, object], row)
        name = raw.get("name")
        module = raw.get("module")
        distribution = raw.get("distribution")
        version = raw.get("required_version")
        source = raw.get("source")
        _require(
            isinstance(name, str)
            and bool(_SAFE_NAME.fullmatch(name))
            and isinstance(module, str)
            and bool(_SAFE_NAME.fullmatch(module))
            and (distribution is None or isinstance(distribution, str))
            and (version is None or isinstance(version, str))
            and source in {"base_image", "locked_project", "project_source", "vendor_source"}
        )
        checked_name = cast(str, name)
        checked_module = cast(str, module)
        checks.append(
            DependencyCheck(
                name=checked_name,
                module=checked_module,
                distribution=cast(str | None, distribution),
                required_version=cast(str | None, version),
                source=cast(
                    Literal[
                        "base_image", "locked_project", "project_source", "vendor_source"
                    ],
                    source,
                ),
            )
        )
    locked = _locked_rows(requirements_path)
    expected_locked = {
        cast(str, check.distribution): cast(str, check.required_version)
        for check in checks
        if check.source == "locked_project"
    }
    _require(set(locked) == set(expected_locked))
    _require(all(locked[name][0] == version for name, version in expected_locked.items()))
    return RemoteEnvironmentContract(
        image=cast(str, image),
        platform="linux_x86_64",
        python_major=3,
        python_minor=12,
        torch_requirement="2.7.1",
        torchvision_requirement="0.22.1",
        cuda_requirement="12.8",
        official_index_url="https://pypi.org/simple",
        bootstrap_timeout_seconds=600,
        requirements_lock_sha256=lock_sha256,
        system_site_packages_required=True,
        checks=tuple(checks),
        contract_sha256=contract_sha256,
    )


def _default_importer(check: DependencyCheck, vendor_root: Path | None) -> object:
    if check.source != "vendor_source":
        return importlib.import_module(check.module)
    if vendor_root is None:
        raise RemoteEnvironmentError("REMOTE_VENDOR_IMPORT_FAILED")
    source = vendor_root / "megaloc_model.py"
    _require(
        source.is_file() and not source.is_symlink(),
        "REMOTE_VENDOR_IMPORT_FAILED",
    )
    spec = importlib.util.spec_from_file_location("phase3f_megaloc_preflight", source)
    if spec is None or spec.loader is None:
        raise RemoteEnvironmentError("REMOTE_VENDOR_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_gpu_name(value: object) -> str | None:
    return value if isinstance(value, str) and bool(_SAFE_GPU.fullmatch(value)) else None


def _document(
    contract: RemoteEnvironmentContract,
    *,
    expected_interpreter_class: str,
    actual_interpreter_class: str,
    python_version: tuple[int, int],
    rows: list[dict[str, object]],
    failure_code: str | None,
    failed_module: str | None,
    exception_class: str | None,
    cuda_available: bool | None,
    torch_cuda_version: str | None,
    gpu_name: str | None,
    preflight_scope: Literal["base", "project", "full"],
) -> DependencyPreflightResult:
    passed = failure_code is None
    abi_compatible = (
        None
        if cuda_available is None
        else cuda_available and torch_cuda_version == contract.cuda_requirement
    )
    report: dict[str, object] = {
        "schema": REMOTE_DEPENDENCY_REPORT_SCHEMA,
        "outcome": "passed" if passed else "failed",
        "dependency_failure_code": failure_code,
        "failed_module": failed_module,
        "failure_exception_class": exception_class,
        "checks": rows,
        "check_count": len(rows),
        "preflight_scope": preflight_scope,
        "contract_sha256": contract.contract_sha256,
        "dependency_lock_sha256": contract.requirements_lock_sha256,
        "secrets_included": False,
    }
    receipt: dict[str, object] = {
        "schema": REMOTE_ENVIRONMENT_RECEIPT_SCHEMA,
        "outcome": "passed" if passed else "failed",
        "interpreter_class": actual_interpreter_class,
        "expected_interpreter_class": expected_interpreter_class,
        "preflight_scope": preflight_scope,
        "python_major": python_version[0],
        "python_minor": python_version[1],
        "cuda_available": cuda_available,
        "torch_cuda_version": torch_cuda_version,
        "gpu_name": gpu_name,
        "gpu_name_present": gpu_name is not None,
        "abi_compatible": abi_compatible,
        "contract_sha256": contract.contract_sha256,
        "dependency_lock_sha256": contract.requirements_lock_sha256,
        "dependency_failure_code": failure_code,
        "failed_module": failed_module,
        "failure_exception_class": exception_class,
        "all_imports_passed": passed,
        "system_site_packages": contract.system_site_packages_required,
        "secrets_included": False,
    }
    return DependencyPreflightResult(
        passed=passed,
        failure_code=failure_code,
        failed_module=failed_module,
        exception_class=exception_class,
        report=report,
        environment_receipt=receipt,
    )


def evaluate_remote_environment(
    contract: RemoteEnvironmentContract,
    *,
    expected_interpreter_class: Literal["system_python", "project_venv"],
    vendor_root: Path | None,
    stage: Literal["base", "project", "full"] = "full",
    importer: ModuleImporter = _default_importer,
    version_reader: VersionReader = importlib.metadata.version,
    actual_interpreter_class: str | None = None,
    python_version: tuple[int, int] | None = None,
) -> DependencyPreflightResult:
    actual_class = actual_interpreter_class or interpreter_class()
    actual_python = python_version or (sys.version_info.major, sys.version_info.minor)
    rows: list[dict[str, object]] = []
    cuda_available: bool | None = None
    torch_cuda_version: str | None = None
    gpu_name: str | None = None
    if actual_class != expected_interpreter_class or actual_python != (
        contract.python_major,
        contract.python_minor,
    ):
        return _document(
            contract,
            expected_interpreter_class=expected_interpreter_class,
            actual_interpreter_class=actual_class,
            python_version=actual_python,
            rows=rows,
            failure_code="REMOTE_PYTHON_INTERPRETER_MISMATCH",
            failed_module="python",
            exception_class="InterpreterContractError",
            cuda_available=None,
            torch_cuda_version=None,
            gpu_name=None,
            preflight_scope=stage,
        )
    selected = tuple(
        check
        for check in contract.checks
        if stage == "full"
        or (stage == "base" and check.source == "base_image")
        or (stage == "project" and check.source != "base_image")
    )
    for check in selected:
        observed_version: str | None = None
        failure_code: str | None = None
        exception_class: str | None = None
        try:
            module = importer(check, vendor_root)
            if check.distribution is not None:
                observed_version = version_reader(check.distribution)
                required = cast(str, check.required_version)
                if observed_version.split("+", 1)[0] != required:
                    failure_code = "REMOTE_DEPENDENCY_VERSION_MISMATCH"
                    exception_class = "VersionContractError"
            if failure_code is None and check.module == "torch":
                torch_module = cast(ModuleType, module)
                cuda = torch_module.cuda
                cuda_available = bool(cuda.is_available())
                version = torch_module.version
                raw_cuda_version = getattr(version, "cuda", None)
                torch_cuda_version = (
                    raw_cuda_version if isinstance(raw_cuda_version, str) else None
                )
                if not cuda_available:
                    failure_code = "REMOTE_TORCH_CUDA_UNAVAILABLE"
                    exception_class = "CudaContractError"
                elif torch_cuda_version != contract.cuda_requirement:
                    failure_code = "REMOTE_TORCH_CUDA_ABI_MISMATCH"
                    exception_class = "CudaAbiContractError"
                else:
                    gpu_name = _safe_gpu_name(cuda.get_device_name(0))
        except (ModuleNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
            failure_code = "REMOTE_DEPENDENCY_MODULE_MISSING"
            exception_class = type(exc).__name__
        except RemoteEnvironmentError as exc:
            failure_code = (
                "REMOTE_VENDOR_IMPORT_FAILED"
                if check.source == "vendor_source"
                else exc.code
            )
            exception_class = type(exc).__name__
        except Exception as exc:  # sanitized import boundary
            failure_code = (
                "REMOTE_VENDOR_IMPORT_FAILED"
                if check.source == "vendor_source"
                else "REMOTE_DEPENDENCY_IMPORT_FAILED"
            )
            exception_class = type(exc).__name__
        row: dict[str, object] = {
            "check_name": check.name,
            "module": check.module,
            "distribution": check.distribution,
            "required_version": check.required_version,
            "observed_version": (
                observed_version
                if observed_version is not None
                else "not_applicable"
                if check.distribution is None and failure_code is None
                else "missing"
            ),
            "import_result": "passed" if failure_code is None else "failed",
            "failure_exception_class": exception_class,
            "dependency_failure_code": failure_code,
            "interpreter_class": actual_class,
            "python_major": actual_python[0],
            "python_minor": actual_python[1],
            "cuda_available": cuda_available,
            "torch_cuda_version": torch_cuda_version,
            "gpu_name": gpu_name,
            "gpu_name_present": gpu_name is not None,
            "abi_compatible": (
                None
                if cuda_available is None
                else cuda_available
                and torch_cuda_version == contract.cuda_requirement
            ),
            "secrets_included": False,
        }
        rows.append(row)
        if failure_code is not None:
            _require(failure_code in DEPENDENCY_FAILURE_CODES, failure_code)
            return _document(
                contract,
                expected_interpreter_class=expected_interpreter_class,
                actual_interpreter_class=actual_class,
                python_version=actual_python,
                rows=rows,
                failure_code=failure_code,
                failed_module=check.module,
                exception_class=exception_class,
                cuda_available=cuda_available,
                torch_cuda_version=torch_cuda_version,
                gpu_name=gpu_name,
                preflight_scope=stage,
            )
    return _document(
        contract,
        expected_interpreter_class=expected_interpreter_class,
        actual_interpreter_class=actual_class,
        python_version=actual_python,
        rows=rows,
        failure_code=None,
        failed_module=None,
        exception_class=None,
        cuda_available=cuda_available,
        torch_cuda_version=torch_cuda_version,
        gpu_name=gpu_name,
        preflight_scope=stage,
    )


def require_environment_receipt(
    path: Path,
    *,
    expected_contract_sha256: str,
    expected_lock_sha256: str,
    validate_current_interpreter: bool = True,
) -> dict[str, object]:
    value = _read_json(path)
    _require(
        value.get("schema") == REMOTE_ENVIRONMENT_RECEIPT_SCHEMA
        and value.get("outcome") == "passed"
        and value.get("interpreter_class") == "project_venv"
        and value.get("expected_interpreter_class") == "project_venv"
        and value.get("preflight_scope") == "full"
        and value.get("python_major") == 3
        and value.get("python_minor") == 12
        and value.get("cuda_available") is True
        and value.get("abi_compatible") is True
        and value.get("all_imports_passed") is True
        and value.get("system_site_packages") is True
        and value.get("contract_sha256") == expected_contract_sha256
        and value.get("dependency_lock_sha256") == expected_lock_sha256
        and value.get("dependency_failure_code") is None
        and value.get("secrets_included") is False
    )
    if validate_current_interpreter:
        _require(
            interpreter_class() == "project_venv",
            "REMOTE_PYTHON_INTERPRETER_MISMATCH",
        )
    return value


def require_dependency_report(
    path: Path,
    *,
    expected_contract_sha256: str,
    expected_lock_sha256: str,
    expected_check_count: int,
) -> dict[str, object]:
    value = _read_json(path)
    rows = value.get("checks")
    _require(
        value.get("schema") == REMOTE_DEPENDENCY_REPORT_SCHEMA
        and value.get("outcome") == "passed"
        and value.get("dependency_failure_code") is None
        and value.get("failed_module") is None
        and value.get("preflight_scope") == "full"
        and isinstance(rows, list)
        and len(rows) == expected_check_count
        and value.get("check_count") == expected_check_count
        and value.get("contract_sha256") == expected_contract_sha256
        and value.get("dependency_lock_sha256") == expected_lock_sha256
        and value.get("secrets_included") is False
    )
    for row in cast(list[object], rows):
        _require(
            isinstance(row, dict)
            and row.get("import_result") == "passed"
            and row.get("dependency_failure_code") is None
            and row.get("secrets_included") is False
        )
    return value


__all__ = [
    "DEPENDENCY_FAILURE_CODES",
    "DependencyCheck",
    "DependencyPreflightResult",
    "REMOTE_DEPENDENCY_REPORT_SCHEMA",
    "REMOTE_ENVIRONMENT_CONTRACT_SCHEMA",
    "REMOTE_ENVIRONMENT_PLAN_SCHEMA",
    "REMOTE_ENVIRONMENT_RECEIPT_SCHEMA",
    "RemoteEnvironmentContract",
    "RemoteEnvironmentError",
    "evaluate_remote_environment",
    "interpreter_class",
    "load_remote_environment_contract",
    "require_dependency_report",
    "require_environment_receipt",
    "sha256_file",
]
