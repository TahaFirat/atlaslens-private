"""Offline RunPod entry point for sealed-corpus MegaLoc fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Literal, cast


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-training-job")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--sealed-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--vendor-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--dependency-preflight-only", action="store_true")
    parser.add_argument("--compatibility-smoke-only", action="store_true")
    parser.add_argument("--compatibility-smoke-timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--dependency-preflight-scope",
        choices=("project", "full"),
        default="full",
    )
    parser.add_argument("--environment-contract", type=Path)
    parser.add_argument("--requirements-lock", type=Path)
    parser.add_argument("--expected-contract-sha256")
    parser.add_argument("--expected-lock-sha256")
    parser.add_argument("--expected-torch-identity", type=Path)
    parser.add_argument("--framework-install-receipt", type=Path)
    parser.add_argument("--environment-receipt", type=Path)
    return parser


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_]+", code):
        return code.upper()
    return "PHASE3F_TRAINING_JOB_FAILED"


def _dependency_preflight(args: argparse.Namespace) -> int:
    from atlaslens_api.phase3f.remote_environment import (  # noqa: PLC0415
        REMOTE_DEPENDENCY_REPORT_SCHEMA,
        REMOTE_ENVIRONMENT_RECEIPT_SCHEMA,
        RemoteEnvironmentError,
        capture_torch_identity,
        evaluate_base_runtime_checks,
        evaluate_remote_environment,
        load_remote_environment_contract,
        require_torch_identity_unchanged,
    )
    from atlaslens_api.phase3f.training_recovery import atomic_json  # noqa: PLC0415

    required_values = [
        args.vendor_root,
        args.recovery_root,
        args.environment_contract,
        args.requirements_lock,
        args.expected_contract_sha256,
        args.expected_lock_sha256,
    ]
    if args.dependency_preflight_scope == "full":
        required_values.extend(
            (args.expected_torch_identity, args.framework_install_receipt)
        )
    if any(value is None for value in required_values):
        print("REMOTE_DEPENDENCY_LOCK_MISMATCH")
        return 92
    recovery_root = cast(Path, args.recovery_root)
    recovery_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    framework_receipt: dict[str, object] | None = None
    torch_identity_before: dict[str, object] | None = None
    torch_identity_after: dict[str, object] | None = None
    try:
        contract = load_remote_environment_contract(
            cast(Path, args.environment_contract),
            cast(Path, args.requirements_lock),
            expected_contract_sha256=cast(str, args.expected_contract_sha256),
            expected_lock_sha256=cast(str, args.expected_lock_sha256),
        )
        preflight_stage: Literal["project", "full"] = (
            "project" if args.dependency_preflight_scope == "project" else "full"
        )
        if preflight_stage == "full":
            torch_identity_before_raw = json.loads(
                cast(Path, args.expected_torch_identity).read_text(encoding="utf-8")
            )
            framework_receipt_raw = json.loads(
                cast(Path, args.framework_install_receipt).read_text(encoding="utf-8")
            )
            if not isinstance(torch_identity_before_raw, dict) or not isinstance(
                framework_receipt_raw, dict
            ):
                raise RemoteEnvironmentError("REMOTE_TORCH_IDENTITY_CHANGED")
            torch_identity_before = cast(
                dict[str, object], torch_identity_before_raw
            )
            framework_receipt = cast(dict[str, object], framework_receipt_raw)
            torch_identity_after = capture_torch_identity()
            require_torch_identity_unchanged(
                torch_identity_before, torch_identity_after
            )
            if not (
                framework_receipt.get("schema")
                == "atlaslens-phase3f-framework-install-receipt-v1"
                and framework_receipt.get("torchvision_version")
                == contract.torchvision_companion_version
                and framework_receipt.get("torchvision_wheel_sha256")
                == contract.torchvision_companion_sha256
                and framework_receipt.get("framework_wheelhouse_lock_sha256")
                == contract.framework_wheelhouse_lock_sha256
                and framework_receipt.get("package_index_resolution_count") == 0
                and framework_receipt.get("torch_download_count") == 0
                and (
                    framework_receipt.get("torchvision_companion_required") is False
                    or framework_receipt.get("torchvision_companion_installed") is True
                )
            ):
                raise RemoteEnvironmentError("REMOTE_TORCHVISION_INSTALL_FAILED")
            framework_receipt.update(
                {
                    "outcome": "torch_identity_verified",
                    "torch_identity_unchanged": True,
                    "torch_identity_before": torch_identity_before,
                    "torch_identity_after": torch_identity_after,
                    "dependency_failure_code": None,
                }
            )
            atomic_json(cast(Path, args.framework_install_receipt), framework_receipt)
            atomic_json(
                recovery_root / "torch-identity-after.json", torch_identity_after
            )
        result = evaluate_remote_environment(
            contract,
            expected_interpreter_class="project_venv",
            vendor_root=cast(Path, args.vendor_root),
            stage=preflight_stage,
        )
        if preflight_stage == "full":
            result = evaluate_base_runtime_checks(contract, result)
        report = result.report
        receipt = result.environment_receipt
        if framework_receipt is not None:
            framework_receipt.update(
                {
                    "outcome": "passed" if result.passed else "failed",
                    "dependency_failure_code": result.failure_code,
                }
            )
            atomic_json(cast(Path, args.framework_install_receipt), framework_receipt)
            framework_fields = {
                "framework_wheelhouse_inventory_sha256": framework_receipt.get(
                    "framework_wheelhouse_inventory_sha256"
                ),
                "framework_wheelhouse_lock_sha256": (
                    contract.framework_wheelhouse_lock_sha256
                ),
                "torchvision_companion_installed": framework_receipt.get(
                    "torchvision_companion_installed"
                ),
                "torchvision_wheel_sha256": contract.torchvision_companion_sha256,
                "package_index_resolution_count": 0,
                "torch_download_count": 0,
                "torch_identity_unchanged": True,
                "torch_identity_before": torch_identity_before,
                "torch_identity_after": torch_identity_after,
            }
            report.update(framework_fields)
            receipt.update(framework_fields)
        failure_code = result.failure_code
    except RemoteEnvironmentError as exc:
        failure_code = exc.code
        failed_module = (
            "torch"
            if failure_code == "REMOTE_TORCH_IDENTITY_CHANGED"
            else "torchvision"
            if failure_code.startswith("REMOTE_TORCHVISION_")
            else "dependency-lock"
        )
        if framework_receipt is not None and args.framework_install_receipt is not None:
            framework_receipt.update(
                {
                    "outcome": "failed",
                    "torch_identity_unchanged": False,
                    "dependency_failure_code": failure_code,
                }
            )
            atomic_json(cast(Path, args.framework_install_receipt), framework_receipt)
        report = {
            "schema": REMOTE_DEPENDENCY_REPORT_SCHEMA,
            "outcome": "failed",
            "dependency_failure_code": failure_code,
            "failed_module": failed_module,
            "failure_exception_class": type(exc).__name__,
            "checks": [],
            "check_count": 0,
            "secrets_included": False,
        }
        receipt = {
            "schema": REMOTE_ENVIRONMENT_RECEIPT_SCHEMA,
            "outcome": "failed",
            "interpreter_class": "project_venv",
            "dependency_failure_code": failure_code,
            "failed_module": failed_module,
            "failure_exception_class": type(exc).__name__,
            "all_imports_passed": False,
            "secrets_included": False,
        }
    atomic_json(recovery_root / "dependency-report.json", report)
    atomic_json(recovery_root / "environment-receipt.json", receipt)
    if failure_code is not None:
        print(failure_code)
        return 92
    print(
        "PHASE3F_LOCAL_PROJECT_DEPENDENCIES_READY"
        if args.dependency_preflight_scope == "project"
        else "PHASE3F_REMOTE_DEPENDENCIES_READY"
    )
    return 0


def _compatibility_smoke(args: argparse.Namespace) -> int:
    from atlaslens_api.phase3f.pipeline import MODEL_SHA256  # noqa: PLC0415
    from atlaslens_api.phase3f.remote_environment import (  # noqa: PLC0415
        RemoteEnvironmentError,
        canonical_environment_identity,
        load_remote_environment_contract,
    )
    from atlaslens_api.phase3f.training import run_remote_training_smoke  # noqa: PLC0415
    from atlaslens_api.phase3f.training_recovery import atomic_json  # noqa: PLC0415

    if any(
        value is None
        for value in (
            args.model,
            args.vendor_root,
            args.recovery_root,
            args.environment_contract,
            args.requirements_lock,
            args.expected_contract_sha256,
            args.expected_lock_sha256,
        )
    ):
        print("REMOTE_DEPENDENCY_LOCK_MISMATCH")
        return 95
    recovery_root = cast(Path, args.recovery_root)
    receipt_path = recovery_root / "environment-receipt.json"
    report_path = recovery_root / "dependency-report.json"
    try:
        contract = load_remote_environment_contract(
            cast(Path, args.environment_contract),
            cast(Path, args.requirements_lock),
            expected_contract_sha256=cast(str, args.expected_contract_sha256),
            expected_lock_sha256=cast(str, args.expected_lock_sha256),
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        base_receipt = json.loads(
            (recovery_root / "base-environment-receipt.json").read_text(encoding="utf-8")
        )
        framework_receipt = json.loads(
            (recovery_root / "framework-install-receipt.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(receipt, dict) or not isinstance(report, dict):
            raise RemoteEnvironmentError("REMOTE_DEPENDENCY_LOCK_MISMATCH")
        if (
            not isinstance(base_receipt, dict)
            or not isinstance(framework_receipt, dict)
            or any(
                receipt.get(field) != base_receipt.get(field)
                for field in (
                    "python_major",
                    "python_minor",
                    "torch_version",
                    "torch_cuda_version",
                )
            )
            or framework_receipt.get("outcome") != "passed"
            or framework_receipt.get("torch_identity_unchanged") is not True
            or framework_receipt.get("torchvision_wheel_sha256")
            != contract.torchvision_companion_sha256
            or framework_receipt.get("package_index_resolution_count") != 0
            or framework_receipt.get("torch_download_count") != 0
        ):
            raise RemoteEnvironmentError("REMOTE_BASE_PACKAGE_IDENTITY_CHANGED")
        smoke = run_remote_training_smoke(
            cast(Path, args.model),
            cast(Path, args.vendor_root),
            timeout_seconds=args.compatibility_smoke_timeout_seconds,
        )
        atomic_json(recovery_root / "training-smoke-report.json", smoke)
        failure_code = smoke.get("failure_code")
        if failure_code is None:
            torch_version = receipt.get("torch_version")
            torchvision_version = receipt.get("torchvision_version")
            cuda_version = receipt.get("torch_cuda_version")
            gpu_class = receipt.get("gpu_name")
            if not all(
                isinstance(value, str)
                for value in (torch_version, torchvision_version, cuda_version, gpu_class)
            ):
                raise RemoteEnvironmentError("REMOTE_TRAINING_SMOKE_FAILED")
            vendor_sha256 = _sha256_path(
                cast(Path, args.vendor_root) / "megaloc_model.py"
            )
            identity, identity_sha256 = canonical_environment_identity(
                contract,
                python_version=f"{receipt['python_major']}.{receipt['python_minor']}",
                torch_version=cast(str, torch_version),
                torchvision_version=cast(str, torchvision_version),
                cuda_version=cast(str, cuda_version),
                gpu_class=cast(str, gpu_class),
                torchvision_wheel_sha256=contract.torchvision_companion_sha256,
                vendor_source_sha256=vendor_sha256,
                model_sha256=MODEL_SHA256,
            )
            receipt.update(
                {
                    "compatibility_smoke_passed": True,
                    "compatibility_smoke_event": "PHASE3F_REMOTE_TRAINING_SMOKE_PASSED",
                    "vendor_source_sha256": vendor_sha256,
                    "model_sha256": MODEL_SHA256,
                    "runtime_environment_identity": identity,
                    "runtime_environment_sha256": identity_sha256,
                }
            )
            report["training_smoke"] = smoke
            report["full_report_check_count"] = cast(
                int, report.get("full_report_check_count", report["check_count"])
            ) + cast(int, smoke["check_count"])
            atomic_json(receipt_path, receipt)
            atomic_json(report_path, report)
            print("PHASE3F_REMOTE_TRAINING_SMOKE_PASSED")
            return 0
        exact = cast(str, failure_code)
        receipt.update(
            {
                "outcome": "failed",
                "compatibility_smoke_passed": False,
                "dependency_failure_code": exact,
                "failed_module": "megaloc-training-smoke",
            }
        )
        report.update(
            {
                "outcome": "failed",
                "dependency_failure_code": exact,
                "failed_module": "megaloc-training-smoke",
                "training_smoke": smoke,
            }
        )
        atomic_json(receipt_path, receipt)
        atomic_json(report_path, report)
        print(exact)
        return 95
    except (OSError, UnicodeError, json.JSONDecodeError, RemoteEnvironmentError) as exc:
        code = getattr(exc, "code", "REMOTE_TRAINING_SMOKE_FAILED")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(receipt, dict) and isinstance(report, dict):
                receipt.update(
                    {
                        "outcome": "failed",
                        "compatibility_smoke_passed": False,
                        "dependency_failure_code": code,
                        "failed_module": "environment-identity",
                    }
                )
                report.update(
                    {
                        "outcome": "failed",
                        "dependency_failure_code": code,
                        "failed_module": "environment-identity",
                    }
                )
                atomic_json(receipt_path, receipt)
                atomic_json(report_path, report)
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        print(code)
        return 95


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ.pop("MAPILLARY_ACCESS_TOKEN", None)
    source = args.repository_root.resolve() / "services" / "api" / "src"
    if not source.is_dir() or source.is_symlink():
        print("PHASE3F_SOURCE_ROOT_INVALID")
        return 2
    sys.path.insert(0, str(source))
    if args.dependency_preflight_only:
        return _dependency_preflight(args)
    if args.compatibility_smoke_only:
        return _compatibility_smoke(args)
    if any(
        value is None
        for value in (
            args.run_id,
            args.sealed_root,
            args.model,
            args.vendor_root,
            args.work_root,
            args.output_root,
            args.recovery_root,
            args.environment_contract,
            args.requirements_lock,
            args.expected_contract_sha256,
            args.expected_lock_sha256,
            args.environment_receipt,
            args.deadline_epoch,
        )
    ):
        print("REMOTE_DEPENDENCY_LOCK_MISMATCH")
        return 2
    from atlaslens_api.phase3f.remote_environment import (  # noqa: PLC0415
        RemoteEnvironmentError,
        require_environment_receipt,
    )
    try:
        require_environment_receipt(
            cast(Path, args.environment_receipt),
            expected_contract_sha256=cast(str, args.expected_contract_sha256),
            expected_lock_sha256=cast(str, args.expected_lock_sha256),
        )
    except RemoteEnvironmentError as exc:
        from atlaslens_api.phase3f.training_recovery import atomic_json  # noqa: PLC0415

        atomic_json(
            cast(Path, args.recovery_root) / "failure.json",
            {
                "schema": "atlaslens-phase3f-remote-training-failure-v1",
                "failure_code": "REMOTE_TRAINING_DEPENDENCY_FAILED",
                "dependency_failure_code": exc.code,
                "failed_module": "environment-receipt",
                "exception_class": type(exc).__name__,
                "stdout_tail": [],
                "stderr_tail": [],
                "training_started": False,
                "secrets_included": False,
            },
        )
        print(exc.code)
        return 93
    from atlaslens_api.phase3f.training import (  # noqa: PLC0415
        TrainingJobConfig,
        run_training_job,
    )

    try:
        result = run_training_job(
            TrainingJobConfig(
                run_id=cast(str, args.run_id),
                sealed_root=cast(Path, args.sealed_root),
                model_path=cast(Path, args.model),
                vendor_root=cast(Path, args.vendor_root),
                work_root=cast(Path, args.work_root),
                output_root=cast(Path, args.output_root),
                deadline_epoch=cast(float, args.deadline_epoch),
                recovery_root=cast(Path, args.recovery_root),
                seed=args.seed,
                max_epochs=args.max_epochs,
                environment_contract_sha256=cast(
                    str, args.expected_contract_sha256
                ),
                environment_receipt_path=cast(Path, args.environment_receipt),
            )
        )
    except Exception as exc:  # sanitized process boundary
        safe_code = _safe_code(exc)
        if args.recovery_root is not None:
            from atlaslens_api.phase3f.training_recovery import (  # noqa: PLC0415
                atomic_json,
                sanitized_tail,
            )

            peak_cuda_bytes: int | None = None
            torch_module = sys.modules.get("torch")
            try:
                if torch_module is not None and torch_module.cuda.is_available():
                    peak_cuda_bytes = int(torch_module.cuda.max_memory_allocated())
            except RuntimeError:
                pass
            disk_free_bytes = shutil.disk_usage(args.work_root.parent).free
            atomic_json(
                args.recovery_root / "child-failure.json",
                {
                    "schema": "atlaslens-phase3f-training-child-failure-v1",
                    "exception_class": type(exc).__name__,
                    "message_code": safe_code,
                    "traceback_tail": list(
                        sanitized_tail("".join(traceback.format_exception(exc)))
                    ),
                    "peak_cuda_bytes": peak_cuda_bytes,
                    "disk_free_bytes": disk_free_bytes,
                    "secrets_included": False,
                },
            )
        print(safe_code)
        return 1
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
