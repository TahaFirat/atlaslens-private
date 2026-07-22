"""Deterministic, secret-free Phase 3F remote dependency contract checks."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, cast

REMOTE_ENVIRONMENT_CONTRACT_SCHEMA: Final = (
    "atlaslens-phase3f-remote-environment-contract-v3"
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
        "REMOTE_TORCHVISION_PAIR_UNSUPPORTED",
        "REMOTE_TORCHVISION_OPS_FAILED",
        "REMOTE_CUDA_TENSOR_ALLOCATION_FAILED",
        "REMOTE_CUDA_AUTOCAST_FAILED",
        "REMOTE_CUDA_FORWARD_FAILED",
        "REMOTE_CUDA_BACKWARD_FAILED",
        "REMOTE_CUDA_OPTIMIZER_FAILED",
        "REMOTE_MODEL_HASH_MISMATCH",
        "REMOTE_TRAINING_SMOKE_FAILED",
        "REMOTE_TRAINING_SMOKE_TIMEOUT",
        "REMOTE_BASE_PACKAGE_IDENTITY_CHANGED",
        "REMOTE_VENDOR_IMPORT_FAILED",
        "REMOTE_DEPENDENCY_LOCK_MISMATCH",
        "REMOTE_TORCHVISION_COMPANION_MISSING",
        "REMOTE_TORCHVISION_WHEEL_HASH_MISMATCH",
        "REMOTE_TORCHVISION_INSTALL_FAILED",
        "REMOTE_TORCH_IDENTITY_CHANGED",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_GPU = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,127}$")
_LOCK_LINE = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9-]*)==(?P<version>[A-Za-z0-9.+!-]+) "
    r"--hash=sha256:(?P<sha256>[0-9a-f]{64})$"
)
_WHEEL_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{1,240}\.whl$")
_REMOTE_IMAGE = (
    "runpod/pytorch@sha256:"
    "60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f"
)
SUPPORTED_TORCHVISION_PAIRS: Final = (
    ("2.7", "0.22"),
    ("2.8", "0.23"),
    ("2.9", "0.24"),
)
TORCHVISION_COMPATIBILITY_SOURCE: Final = (
    "https://github.com/pytorch/vision#installation"
)
FRAMEWORK_WHEELHOUSE_SCHEMA: Final = "atlaslens-phase3f-framework-wheelhouse-lock-v1"
FRAMEWORK_TRANSFER_INVENTORY_SCHEMA: Final = (
    "atlaslens-phase3f-framework-transfer-inventory-v1"
)
TORCHVISION_COMPANION_VERSION: Final = "0.24.1+cu128"
TORCHVISION_COMPANION_FILENAME: Final = (
    "torchvision-0.24.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"
)
TORCHVISION_COMPANION_SHA256: Final = (
    "cf84eae1d2d12a7d261a7496eca00dd927b71792011b1e84d4162c950eb3201d"
)
TORCHVISION_COMPANION_SIZE: Final = 8_049_914
TORCHVISION_COMPANION_SOURCE: Final = (
    "https://download-r2.pytorch.org/whl/cu128/"
    "torchvision-0.24.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
)
TORCHVISION_COMPANION_INDEX: Final = (
    "https://download.pytorch.org/whl/cu128/torchvision/"
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
class FrameworkWheelSpec:
    distribution: str
    version: str
    filename: str
    size_bytes: int
    sha256: str
    source: Literal["locked_project", "official_pytorch"]


@dataclass(frozen=True, slots=True)
class FrameworkWheelArtifact(FrameworkWheelSpec):
    path: Path


@dataclass(frozen=True, slots=True)
class FrameworkWheelhouse:
    root: Path
    artifacts: tuple[FrameworkWheelArtifact, ...]
    inventory: dict[str, object]
    inventory_sha256: str
    companion: FrameworkWheelArtifact


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
    compatibility_smoke_timeout_seconds: int
    requirements_lock_sha256: str
    framework_wheelhouse_lock_sha256: str
    framework_wheelhouse_local_path: str
    framework_wheelhouse_lock_path: Path
    framework_wheel_artifacts: tuple[FrameworkWheelSpec, ...]
    torchvision_companion_version: str
    torchvision_companion_filename: str
    torchvision_companion_sha256: str
    torchvision_companion_size_bytes: int
    torchvision_companion_source_url: str
    system_site_packages_required: bool
    supported_torchvision_pairs: tuple[tuple[str, str], ...]
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


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _framework_wheel_specs(
    path: Path,
    locked: dict[str, tuple[str, str]],
) -> tuple[FrameworkWheelSpec, ...]:
    value = _read_json(path)
    target = value.get("target")
    rows = value.get("artifacts")
    _require(
        value.get("schema") == FRAMEWORK_WHEELHOUSE_SCHEMA
        and isinstance(target, dict)
        and target.get("python_abi") == "cp312"
        and target.get("platform") == "linux_x86_64"
        and target.get("cuda") == "12.8"
        and isinstance(rows, list)
        and len(rows) == len(locked) + 1
    )
    specs: list[FrameworkWheelSpec] = []
    observed_locked: dict[str, tuple[str, str]] = {}
    companion_count = 0
    for raw in cast(list[object], rows):
        _require(isinstance(raw, dict))
        row = cast(dict[str, object], raw)
        distribution = row.get("distribution")
        version = row.get("version")
        filename = row.get("filename")
        size_bytes = row.get("size_bytes")
        sha256 = row.get("sha256")
        source = row.get("source")
        _require(
            isinstance(distribution, str)
            and bool(_SAFE_NAME.fullmatch(distribution))
            and isinstance(version, str)
            and bool(version)
            and isinstance(filename, str)
            and bool(_WHEEL_FILENAME.fullmatch(filename))
            and isinstance(size_bytes, int)
            and not isinstance(size_bytes, bool)
            and 0 < size_bytes <= 256 * 1024 * 1024
            and isinstance(sha256, str)
            and bool(_SHA256.fullmatch(sha256))
            and source in {"locked_project", "official_pytorch"}
        )
        checked_source = cast(Literal["locked_project", "official_pytorch"], source)
        if checked_source == "official_pytorch":
            companion_count += 1
            _require(
                distribution == "torchvision"
                and version == TORCHVISION_COMPANION_VERSION
                and filename == TORCHVISION_COMPANION_FILENAME
                and size_bytes == TORCHVISION_COMPANION_SIZE
                and sha256 == TORCHVISION_COMPANION_SHA256
                and row.get("source_url") == TORCHVISION_COMPANION_SOURCE
                and row.get("source_index") == TORCHVISION_COMPANION_INDEX
            )
        else:
            checked_distribution = cast(str, distribution)
            _require(checked_distribution in locked)
            observed_locked[checked_distribution] = (
                cast(str, version),
                cast(str, sha256),
            )
        specs.append(
            FrameworkWheelSpec(
                distribution=cast(str, distribution),
                version=cast(str, version),
                filename=cast(str, filename),
                size_bytes=cast(int, size_bytes),
                sha256=cast(str, sha256),
                source=checked_source,
            )
        )
    _require(companion_count == 1 and observed_locked == locked)
    _require(len({spec.filename for spec in specs}) == len(specs))
    return tuple(specs)


def verify_framework_wheelhouse(
    contract: RemoteEnvironmentContract,
    root: Path,
    *,
    expected_inventory_path: Path | None = None,
) -> FrameworkWheelhouse:
    resolved = root.resolve()
    companion_path = resolved / contract.torchvision_companion_filename
    if not companion_path.is_file() or companion_path.is_symlink():
        raise RemoteEnvironmentError("REMOTE_TORCHVISION_COMPANION_MISSING")
    artifacts: list[FrameworkWheelArtifact] = []
    for spec in contract.framework_wheel_artifacts:
        path = resolved / spec.filename
        if not path.is_file() or path.is_symlink():
            code = (
                "REMOTE_TORCHVISION_COMPANION_MISSING"
                if spec.source == "official_pytorch"
                else "REMOTE_DEPENDENCY_LOCK_MISMATCH"
            )
            raise RemoteEnvironmentError(code)
        try:
            size_bytes = path.stat().st_size
            digest = sha256_file(path, max_bytes=256 * 1024 * 1024)
        except (OSError, RemoteEnvironmentError) as exc:
            code = (
                "REMOTE_TORCHVISION_WHEEL_HASH_MISMATCH"
                if spec.source == "official_pytorch"
                else "REMOTE_DEPENDENCY_LOCK_MISMATCH"
            )
            raise RemoteEnvironmentError(code) from exc
        if size_bytes != spec.size_bytes or digest != spec.sha256:
            code = (
                "REMOTE_TORCHVISION_WHEEL_HASH_MISMATCH"
                if spec.source == "official_pytorch"
                else "REMOTE_DEPENDENCY_LOCK_MISMATCH"
            )
            raise RemoteEnvironmentError(code)
        artifacts.append(
            FrameworkWheelArtifact(
                path=path,
                distribution=spec.distribution,
                version=spec.version,
                filename=spec.filename,
                size_bytes=spec.size_bytes,
                sha256=spec.sha256,
                source=spec.source,
            )
        )
    actual_names = {
        path.name
        for path in resolved.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == ".whl"
    }
    _require(
        actual_names == {artifact.filename for artifact in artifacts},
        "REMOTE_DEPENDENCY_LOCK_MISMATCH",
    )
    rows = [
        {
            "distribution": artifact.distribution,
            "filename": artifact.filename,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
            "source": artifact.source,
            "version": artifact.version,
        }
        for artifact in sorted(artifacts, key=lambda item: item.filename)
    ]
    inventory: dict[str, object] = {
        "schema": FRAMEWORK_TRANSFER_INVENTORY_SCHEMA,
        "artifacts": rows,
        "artifact_count": len(rows),
        "framework_wheelhouse_lock_sha256": (
            contract.framework_wheelhouse_lock_sha256
        ),
        "package_index_resolution_on_pod": False,
        "torch_download_prohibited": True,
        "secrets_included": False,
    }
    inventory_sha256 = _canonical_sha256(inventory)
    if expected_inventory_path is not None:
        expected = _read_json(expected_inventory_path)
        _require(
            expected == inventory,
            "REMOTE_TORCHVISION_WHEEL_HASH_MISMATCH",
        )
    companion = next(
        artifact for artifact in artifacts if artifact.source == "official_pytorch"
    )
    return FrameworkWheelhouse(
        root=resolved,
        artifacts=tuple(artifacts),
        inventory=inventory,
        inventory_sha256=inventory_sha256,
        companion=companion,
    )


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
    framework = value.get("framework_wheelhouse")
    _require(isinstance(framework, dict))
    checked_framework = cast(dict[str, object], framework)
    companion = checked_framework.get("torchvision_companion")
    _require(
        checked_framework.get("local_path")
        == ".local/phase3f/wheelhouse/linux-cp312-cu128"
        and checked_framework.get("lock")
        == "config/phase3f-framework-wheelhouse.lock.json"
        and checked_framework.get("package_index_resolution_on_pod") is False
        and isinstance(companion, dict)
        and companion.get("version") == TORCHVISION_COMPANION_VERSION
        and companion.get("filename") == TORCHVISION_COMPANION_FILENAME
        and companion.get("size_bytes") == TORCHVISION_COMPANION_SIZE
        and companion.get("sha256") == TORCHVISION_COMPANION_SHA256
        and companion.get("source_url") == TORCHVISION_COMPANION_SOURCE
        and companion.get("source_index") == TORCHVISION_COMPANION_INDEX
    )
    framework_lock_path = (
        contract_path.resolve().parent / "phase3f-framework-wheelhouse.lock.json"
    )
    framework_lock_sha256 = sha256_file(framework_lock_path)
    _require(
        checked_framework.get("lock_sha256") == framework_lock_sha256,
        "REMOTE_DEPENDENCY_LOCK_MISMATCH",
    )
    _require(value.get("python_requirement") == "==3.12.*")
    _require(value.get("torch_requirement") == ">=2.7,<2.10")
    _require(value.get("torchvision_requirement") == ">=0.22,<0.25")
    _require(value.get("cuda_requirement") == "12.8")
    _require(value.get("platform") == "linux_x86_64")
    _require(value.get("system_site_packages_required") is True)
    _require(value.get("image_identity_policy") == "exact_digest")
    _require(value.get("torch_download_prohibited") is True)
    _require(value.get("official_index_url") == "https://pypi.org/simple")
    _require(value.get("bootstrap_timeout_seconds") == 600)
    _require(value.get("compatibility_smoke_timeout_seconds") == 180)
    _require(value.get("compatibility_source") == TORCHVISION_COMPATIBILITY_SOURCE)
    raw_pairs = value.get("supported_torchvision_pairs")
    _require(
        isinstance(raw_pairs, list)
        and tuple(
            (row.get("torch_minor"), row.get("torchvision_minor"))
            for row in raw_pairs
            if isinstance(row, dict)
        )
        == SUPPORTED_TORCHVISION_PAIRS
    )
    history = value.get("historical_observations")
    _require(
        isinstance(history, list)
        and len(history) == 1
        and isinstance(history[0], dict)
        and history[0].get("image_digest") == _REMOTE_IMAGE.split("@", 1)[1]
        and history[0].get("python") == "3.12"
        and history[0].get("torch") == "2.9.1+cu128"
        and history[0].get("torchvision") is None
        and history[0].get("cuda") == "12.8"
    )
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
    framework_specs = _framework_wheel_specs(framework_lock_path, locked)
    return RemoteEnvironmentContract(
        image=cast(str, image),
        platform="linux_x86_64",
        python_major=3,
        python_minor=12,
        torch_requirement=">=2.7,<2.10",
        torchvision_requirement=">=0.22,<0.25",
        cuda_requirement="12.8",
        official_index_url="https://pypi.org/simple",
        bootstrap_timeout_seconds=600,
        compatibility_smoke_timeout_seconds=180,
        requirements_lock_sha256=lock_sha256,
        framework_wheelhouse_lock_sha256=framework_lock_sha256,
        framework_wheelhouse_local_path=cast(str, checked_framework["local_path"]),
        framework_wheelhouse_lock_path=framework_lock_path,
        framework_wheel_artifacts=framework_specs,
        torchvision_companion_version=TORCHVISION_COMPANION_VERSION,
        torchvision_companion_filename=TORCHVISION_COMPANION_FILENAME,
        torchvision_companion_sha256=TORCHVISION_COMPANION_SHA256,
        torchvision_companion_size_bytes=TORCHVISION_COMPANION_SIZE,
        torchvision_companion_source_url=TORCHVISION_COMPANION_SOURCE,
        system_site_packages_required=True,
        supported_torchvision_pairs=SUPPORTED_TORCHVISION_PAIRS,
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


def _version_minor(value: str) -> str | None:
    matched = re.fullmatch(r"(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)"
                             r"\.(?:0|[1-9][0-9]*)(?:\+[A-Za-z0-9.]+)?", value)
    if matched is None:
        return None
    return f"{matched.group('major')}.{matched.group('minor')}"


def canonical_environment_identity(
    contract: RemoteEnvironmentContract,
    *,
    python_version: str,
    torch_version: str,
    torchvision_version: str,
    cuda_version: str,
    gpu_class: str,
    torchvision_wheel_sha256: str | None = None,
    vendor_source_sha256: str | None = None,
    model_sha256: str | None = None,
) -> tuple[dict[str, object], str]:
    """Return the canonical, secret-free runtime identity and its SHA-256."""
    _require(bool(re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", python_version)))
    _require(_version_minor(torch_version) is not None)
    _require(_version_minor(torchvision_version) is not None)
    _require(bool(re.fullmatch(r"[0-9]+\.[0-9]+", cuda_version)))
    _require(bool(_SAFE_GPU.fullmatch(gpu_class)))
    for digest in (torchvision_wheel_sha256, vendor_source_sha256, model_sha256):
        _require(digest is None or bool(_SHA256.fullmatch(digest)))
    document: dict[str, object] = {
        "schema": "atlaslens-phase3f-runtime-environment-identity-v1",
        "image": contract.image,
        "python": python_version,
        "python_abi": "cp312",
        "torch": torch_version,
        "torchvision": torchvision_version,
        "torchvision_wheel_sha256": torchvision_wheel_sha256,
        "cuda": cuda_version,
        "gpu_class": gpu_class,
        "dependency_lock_sha256": contract.requirements_lock_sha256,
        "framework_wheelhouse_lock_sha256": (
            contract.framework_wheelhouse_lock_sha256
        ),
        "torch_download_prohibited": True,
        "vendor_source_sha256": vendor_source_sha256,
        "model_sha256": model_sha256,
    }
    encoded = json.dumps(
        document, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return document, hashlib.sha256(encoded).hexdigest()


def capture_torch_identity(
    *,
    torch_module: object | None = None,
    version_reader: VersionReader = importlib.metadata.version,
    module_path: Path | None = None,
) -> dict[str, object]:
    try:
        module = torch_module or importlib.import_module("torch")
        checked_module = cast(ModuleType, module)
        version = version_reader("torch")
        module_version = vars(checked_module).get("__version__")
        raw_cuda = getattr(checked_module.version, "cuda", None)
        raw_path = module_path or Path(cast(str, checked_module.__file__))
        resolved = raw_path.resolve(strict=True)
        stat_result = resolved.stat()
        _require(
            _version_minor(version) is not None
            and module_version == version
            and isinstance(raw_cuda, str)
            and bool(re.fullmatch(r"[0-9]+\.[0-9]+", raw_cuda))
            and resolved.is_file()
            and not resolved.is_symlink(),
            "REMOTE_TORCH_IDENTITY_CHANGED",
        )
        path_sha256 = hashlib.sha256(
            os.fsencode(str(resolved).replace("\\", "/"))
        ).hexdigest()
        return {
            "schema": "atlaslens-phase3f-torch-identity-v1",
            "version": version,
            "cuda": raw_cuda,
            "python_abi": "cp312",
            "module_path_sha256": path_sha256,
            "module_file_sha256": sha256_file(resolved, max_bytes=16 * 1024 * 1024),
            "module_device": int(stat_result.st_dev),
            "module_inode": int(stat_result.st_ino),
            "secrets_included": False,
        }
    except RemoteEnvironmentError:
        raise
    except Exception as exc:
        raise RemoteEnvironmentError("REMOTE_TORCH_IDENTITY_CHANGED") from exc


def require_torch_identity_unchanged(
    before: dict[str, object], after: dict[str, object]
) -> None:
    _require(
        before == after
        and before.get("schema") == "atlaslens-phase3f-torch-identity-v1"
        and before.get("secrets_included") is False,
        "REMOTE_TORCH_IDENTITY_CHANGED",
    )


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
    observed_torch_version: str | None = None,
    observed_torchvision_version: str | None = None,
) -> DependencyPreflightResult:
    passed = failure_code is None
    companion_required = failure_code == "REMOTE_TORCHVISION_COMPANION_MISSING"
    abi_compatible = (
        None
        if cuda_available is None
        else cuda_available and torch_cuda_version == contract.cuda_requirement
    )
    failure_codes = [
        cast(str, row["dependency_failure_code"])
        for row in rows
        if isinstance(row.get("dependency_failure_code"), str)
    ]
    pair = (
        (_version_minor(observed_torch_version), _version_minor(observed_torchvision_version))
        if observed_torch_version is not None and observed_torchvision_version is not None
        else (None, None)
    )
    pair_compatible = (
        pair in contract.supported_torchvision_pairs
        if pair[0] is not None and pair[1] is not None
        else None
    )
    identity: dict[str, object] | None = None
    identity_sha256: str | None = None
    if (
        observed_torch_version is not None
        and observed_torchvision_version is not None
        and torch_cuda_version is not None
        and gpu_name is not None
    ):
        identity, identity_sha256 = canonical_environment_identity(
            contract,
            python_version=f"{python_version[0]}.{python_version[1]}",
            torch_version=observed_torch_version,
            torchvision_version=observed_torchvision_version,
            cuda_version=torch_cuda_version,
            gpu_class=gpu_name,
        )
    report: dict[str, object] = {
        "schema": REMOTE_DEPENDENCY_REPORT_SCHEMA,
        "outcome": "passed" if passed else "failed",
        "dependency_failure_code": failure_code,
        "failed_module": failed_module,
        "failure_exception_class": exception_class,
        "checks": rows,
        "check_count": len(rows),
        "all_checks_collected": len(rows) > 0,
        "failure_codes": failure_codes,
        "preflight_scope": preflight_scope,
        "contract_sha256": contract.contract_sha256,
        "dependency_lock_sha256": contract.requirements_lock_sha256,
        "framework_wheelhouse_lock_sha256": (
            contract.framework_wheelhouse_lock_sha256
        ),
        "torchvision_companion_required": companion_required,
        "bootstrap_eligible": passed or companion_required,
        "secrets_included": False,
        "torch_download_prohibited": True,
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
        "torch_version": observed_torch_version,
        "torchvision_version": observed_torchvision_version,
        "torchvision_pair_compatible": pair_compatible,
        "supported_torchvision_pairs": [
            list(item) for item in contract.supported_torchvision_pairs
        ],
        "compatibility_source": TORCHVISION_COMPATIBILITY_SOURCE,
        "gpu_name": gpu_name,
        "gpu_name_present": gpu_name is not None,
        "abi_compatible": abi_compatible,
        "contract_sha256": contract.contract_sha256,
        "dependency_lock_sha256": contract.requirements_lock_sha256,
        "framework_wheelhouse_lock_sha256": (
            contract.framework_wheelhouse_lock_sha256
        ),
        "torchvision_companion_required": companion_required,
        "bootstrap_eligible": passed or companion_required,
        "dependency_failure_code": failure_code,
        "failed_module": failed_module,
        "failure_exception_class": exception_class,
        "all_imports_passed": passed,
        "system_site_packages": contract.system_site_packages_required,
        "torch_download_prohibited": True,
        "runtime_environment_identity": identity,
        "runtime_environment_sha256": identity_sha256,
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
    allow_missing_torchvision_companion: bool = False,
) -> DependencyPreflightResult:
    actual_class = actual_interpreter_class or interpreter_class()
    actual_python = python_version or (sys.version_info.major, sys.version_info.minor)
    rows: list[dict[str, object]] = []
    cuda_available: bool | None = None
    torch_cuda_version: str | None = None
    gpu_name: str | None = None
    observed_torch_version: str | None = None
    observed_torchvision_version: str | None = None
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
    failures: list[tuple[str, str, str | None]] = []
    for check in selected:
        observed_version: str | None = None
        failure_code: str | None = None
        exception_class: str | None = None
        try:
            module = importer(check, vendor_root)
            if check.distribution is not None:
                observed_version = version_reader(check.distribution)
                if check.module == "torch":
                    observed_torch_version = observed_version
                elif check.module == "torchvision":
                    observed_torchvision_version = observed_version
                required = check.required_version
                if (
                    required is not None
                    and observed_version.split("+", 1)[0] != required
                ):
                    failure_code = "REMOTE_DEPENDENCY_VERSION_MISMATCH"
                    exception_class = "VersionContractError"
            if check.module == "torch":
                torch_module = cast(ModuleType, module)
                cuda = torch_module.cuda
                cuda_available = bool(cuda.is_available())
                version = torch_module.version
                raw_cuda_version = getattr(version, "cuda", None)
                torch_cuda_version = (
                    raw_cuda_version if isinstance(raw_cuda_version, str) else None
                )
                if not cuda_available and failure_code is None:
                    failure_code = "REMOTE_TORCH_CUDA_UNAVAILABLE"
                    exception_class = "CudaContractError"
                elif torch_cuda_version != contract.cuda_requirement and failure_code is None:
                    failure_code = "REMOTE_TORCH_CUDA_ABI_MISMATCH"
                    exception_class = "CudaAbiContractError"
                elif cuda_available:
                    gpu_name = _safe_gpu_name(cuda.get_device_name(0))
        except (ModuleNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
            failure_code = (
                "REMOTE_TORCHVISION_COMPANION_MISSING"
                if allow_missing_torchvision_companion
                and check.source == "base_image"
                and check.module == "torchvision"
                else "REMOTE_DEPENDENCY_MODULE_MISSING"
            )
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
            failures.append((failure_code, check.module, exception_class))
    if observed_torch_version is not None and observed_torchvision_version is not None:
        pair = (
            _version_minor(observed_torch_version),
            _version_minor(observed_torchvision_version),
        )
        if pair not in contract.supported_torchvision_pairs:
            pair_code = "REMOTE_TORCHVISION_PAIR_UNSUPPORTED"
            for row in rows:
                if (
                    row.get("module") == "torchvision"
                    and row.get("dependency_failure_code") is None
                ):
                    row["import_result"] = "failed"
                    row["failure_exception_class"] = "TorchVisionPairContractError"
                    row["dependency_failure_code"] = pair_code
                    break
            failures.append((pair_code, "torchvision", "TorchVisionPairContractError"))
    failure_code, failed_module, exception_class = (
        failures[0] if failures else (None, None, None)
    )
    return _document(
        contract,
        expected_interpreter_class=expected_interpreter_class,
        actual_interpreter_class=actual_class,
        python_version=actual_python,
        rows=rows,
        failure_code=failure_code,
        failed_module=failed_module,
        exception_class=exception_class,
        cuda_available=cuda_available,
        torch_cuda_version=torch_cuda_version,
        gpu_name=gpu_name,
        preflight_scope=stage,
        observed_torch_version=observed_torch_version,
        observed_torchvision_version=observed_torchvision_version,
    )


def evaluate_base_runtime_checks(
    contract: RemoteEnvironmentContract,
    base: DependencyPreflightResult,
    *,
    torch_module: object | None = None,
    torchvision_module: object | None = None,
) -> DependencyPreflightResult:
    """Collect every small base-runtime check without stopping at the first failure."""
    torch = torch_module
    torchvision = torchvision_module
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except Exception:  # reported by every dependent probe below
            torch = None
    if torchvision is None:
        try:
            torchvision = importlib.import_module("torchvision")
        except Exception:  # reported by the compiled-ops probe below
            torchvision = None
    checks: list[dict[str, object]] = []

    def collect(name: str, code: str, operation: Callable[[], object]) -> None:
        outcome = "passed"
        exception_class: str | None = None
        observed: object = "compatible"
        try:
            observed = operation()
        except Exception as exc:  # sanitized probe boundary
            outcome = "failed"
            exception_class = type(exc).__name__
            observed = "failed"
        checks.append(
            {
                "check_name": name,
                "outcome": outcome,
                "observed": observed,
                "dependency_failure_code": None if outcome == "passed" else code,
                "failure_exception_class": exception_class,
                "secrets_included": False,
            }
        )

    collect("python-runtime", "REMOTE_PYTHON_INTERPRETER_MISMATCH", lambda: sys.version_info[:2])
    collect(
        "torch-version-cuda",
        "REMOTE_TORCH_CUDA_ABI_MISMATCH",
        lambda: (
            contract.cuda_requirement
            if getattr(cast(ModuleType, torch).version, "cuda", None)
            == contract.cuda_requirement
            else (_ for _ in ()).throw(RuntimeError("cuda abi"))
        ),
    )
    collect(
        "cuda-available",
        "REMOTE_TORCH_CUDA_UNAVAILABLE",
        lambda: True
        if cast(ModuleType, torch).cuda.is_available()
        else (_ for _ in ()).throw(RuntimeError("cuda unavailable")),
    )
    def initialize_cuda_runtime() -> str:
        cast(ModuleType, torch).cuda.init()
        return "initialized"

    collect(
        "cuda-runtime",
        "REMOTE_TORCH_CUDA_UNAVAILABLE",
        initialize_cuda_runtime,
    )
    collect(
        "gpu-capability",
        "REMOTE_TORCH_CUDA_UNAVAILABLE",
        lambda: list(cast(ModuleType, torch).cuda.get_device_capability(0)),
    )
    if (
        torchvision is None
        and base.failure_code == "REMOTE_TORCHVISION_COMPANION_MISSING"
    ):
        checks.append(
            {
                "check_name": "torchvision-compiled-ops",
                "outcome": "pending",
                "observed": "companion_install_required",
                "dependency_failure_code": None,
                "failure_exception_class": None,
                "secrets_included": False,
            }
        )
    else:
        collect(
            "torchvision-compiled-ops",
            "REMOTE_TORCHVISION_OPS_FAILED",
            lambda: int(
                cast(ModuleType, torchvision).ops.nms(
                    cast(ModuleType, torch).tensor([[0.0, 0.0, 1.0, 1.0]]),
                    cast(ModuleType, torch).tensor([1.0]),
                    0.5,
                ).numel()
            ),
        )
    collect(
        "cuda-tensor-allocation",
        "REMOTE_CUDA_TENSOR_ALLOCATION_FAILED",
        lambda: list(cast(ModuleType, torch).zeros(2, device="cuda").shape),
    )

    def training_operation(stage: str) -> object:
        module = cast(ModuleType, torch)
        layer = module.nn.Linear(4, 2).to("cuda")
        optimizer = module.optim.SGD(layer.parameters(), lr=0.01)
        before = layer.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        with module.amp.autocast("cuda", dtype=module.float16):
            output = layer(module.ones((2, 4), device="cuda"))
            loss = output.float().square().mean()
        if stage == "autocast":
            return str(output.dtype)
        loss.backward()
        if stage == "backward":
            return bool(layer.weight.grad is not None)
        optimizer.step()
        if stage == "optimizer":
            return bool(not module.equal(before, layer.weight.detach()))
        return list(output.shape)

    collect("cuda-autocast", "REMOTE_CUDA_AUTOCAST_FAILED", lambda: training_operation("autocast"))
    collect("cuda-forward", "REMOTE_CUDA_FORWARD_FAILED", lambda: training_operation("forward"))
    collect("cuda-backward", "REMOTE_CUDA_BACKWARD_FAILED", lambda: training_operation("backward"))
    collect(
        "cuda-optimizer-step",
        "REMOTE_CUDA_OPTIMIZER_FAILED",
        lambda: training_operation("optimizer"),
    )
    failed = next(
        (row for row in checks if row["dependency_failure_code"] is not None),
        None,
    )
    failure_code = base.failure_code or (
        cast(str, failed["dependency_failure_code"]) if failed is not None else None
    )
    failed_module = base.failed_module or (
        cast(str, failed["check_name"]) if failed is not None else None
    )
    exception_class = base.exception_class or (
        cast(str | None, failed["failure_exception_class"])
        if failed is not None
        else None
    )
    report = dict(base.report)
    report["base_runtime_checks"] = checks
    report["base_runtime_check_count"] = len(checks)
    report["full_report_check_count"] = cast(int, report["check_count"]) + len(checks)
    report["all_checks_collected"] = True
    report["outcome"] = "passed" if failure_code is None else "failed"
    report["dependency_failure_code"] = failure_code
    report["failed_module"] = failed_module
    report["failure_exception_class"] = exception_class
    report["failure_codes"] = [
        code
        for code in (
            *cast(list[str], report.get("failure_codes", [])),
            *(
                cast(str, row["dependency_failure_code"])
                for row in checks
                if row["dependency_failure_code"] is not None
            ),
        )
    ]
    receipt = dict(base.environment_receipt)
    receipt["base_runtime_checks_passed"] = failed is None
    receipt["base_runtime_check_count"] = len(checks)
    receipt["outcome"] = "passed" if failure_code is None else "failed"
    receipt["dependency_failure_code"] = failure_code
    receipt["failed_module"] = failed_module
    receipt["failure_exception_class"] = exception_class
    receipt["all_imports_passed"] = failure_code is None
    return DependencyPreflightResult(
        passed=failure_code is None,
        failure_code=failure_code,
        failed_module=failed_module,
        exception_class=exception_class,
        report=report,
        environment_receipt=receipt,
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
        and value.get("torchvision_pair_compatible") is True
        and value.get("compatibility_smoke_passed") is True
        and value.get("compatibility_smoke_event")
        == "PHASE3F_REMOTE_TRAINING_SMOKE_PASSED"
        and isinstance(value.get("runtime_environment_identity"), dict)
        and cast(dict[str, object], value["runtime_environment_identity"]).get(
            "torchvision_wheel_sha256"
        )
        == TORCHVISION_COMPANION_SHA256
        and isinstance(value.get("runtime_environment_sha256"), str)
        and bool(_SHA256.fullmatch(cast(str, value["runtime_environment_sha256"])))
        and isinstance(value.get("vendor_source_sha256"), str)
        and bool(_SHA256.fullmatch(cast(str, value["vendor_source_sha256"])))
        and isinstance(value.get("model_sha256"), str)
        and bool(_SHA256.fullmatch(cast(str, value["model_sha256"])))
        and value.get("all_imports_passed") is True
        and value.get("system_site_packages") is True
        and value.get("contract_sha256") == expected_contract_sha256
        and value.get("dependency_lock_sha256") == expected_lock_sha256
        and isinstance(value.get("framework_wheelhouse_lock_sha256"), str)
        and bool(
            _SHA256.fullmatch(cast(str, value["framework_wheelhouse_lock_sha256"]))
        )
        and value.get("torchvision_wheel_sha256")
        == TORCHVISION_COMPANION_SHA256
        and value.get("torch_identity_unchanged") is True
        and value.get("package_index_resolution_count") == 0
        and value.get("torch_download_count") == 0
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
    base_runtime_checks = value.get("base_runtime_checks")
    smoke = value.get("training_smoke")
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
        and value.get("torchvision_wheel_sha256")
        == TORCHVISION_COMPANION_SHA256
        and value.get("torch_identity_unchanged") is True
        and value.get("package_index_resolution_count") == 0
        and value.get("torch_download_count") == 0
        and value.get("secrets_included") is False
        and isinstance(base_runtime_checks, list)
        and len(base_runtime_checks) == 11
        and all(
            isinstance(row, dict)
            and row.get("outcome") == "passed"
            and row.get("dependency_failure_code") is None
            and row.get("secrets_included") is False
            for row in base_runtime_checks
        )
        and isinstance(smoke, dict)
        and smoke.get("outcome") == "passed"
        and smoke.get("event") == "PHASE3F_REMOTE_TRAINING_SMOKE_PASSED"
        and smoke.get("holdout_open_count") == 0
        and smoke.get("checkpoint_write_count") == 0
        and smoke.get("secrets_included") is False
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
    "FrameworkWheelArtifact",
    "FrameworkWheelSpec",
    "FrameworkWheelhouse",
    "REMOTE_DEPENDENCY_REPORT_SCHEMA",
    "REMOTE_ENVIRONMENT_CONTRACT_SCHEMA",
    "REMOTE_ENVIRONMENT_PLAN_SCHEMA",
    "REMOTE_ENVIRONMENT_RECEIPT_SCHEMA",
    "RemoteEnvironmentContract",
    "RemoteEnvironmentError",
    "SUPPORTED_TORCHVISION_PAIRS",
    "TORCHVISION_COMPATIBILITY_SOURCE",
    "TORCHVISION_COMPANION_FILENAME",
    "TORCHVISION_COMPANION_SHA256",
    "TORCHVISION_COMPANION_SOURCE",
    "TORCHVISION_COMPANION_VERSION",
    "canonical_environment_identity",
    "capture_torch_identity",
    "evaluate_base_runtime_checks",
    "evaluate_remote_environment",
    "interpreter_class",
    "load_remote_environment_contract",
    "require_dependency_report",
    "require_environment_receipt",
    "require_torch_identity_unchanged",
    "sha256_file",
    "verify_framework_wheelhouse",
]
