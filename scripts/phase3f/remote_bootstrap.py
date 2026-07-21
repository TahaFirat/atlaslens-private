"""Bounded deterministic dependency bootstrap for the Phase 3F RunPod job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol


class AtomicWriter(Protocol):
    def __call__(self, path: Path, value: object) -> str: ...


class TailSanitizer(Protocol):
    def __call__(
        self, value: str, *, maximum_lines: int = 100
    ) -> tuple[str, ...]: ...


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-remote-bootstrap")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--requirements-lock", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser


def _safe_environment() -> dict[str, str]:
    removed = {
        "MAPILLARY_ACCESS_TOKEN",
        "RUNPOD_API_KEY",
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in removed and not key.startswith("PIP_")
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _write_log(path: Path, value: str, sanitized_tail: TailSanitizer) -> None:
    lines = sanitized_tail(value, maximum_lines=100)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _remove_raw_logs(recovery_root: Path) -> None:
    for raw in recovery_root.parent.glob(".phase3f-bootstrap-*.raw"):
        raw.unlink(missing_ok=True)


def _bounded_raw_text(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 256 * 1024))
            return stream.read(256 * 1024).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _inventory(recovery_root: Path, atomic_json: AtomicWriter) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(recovery_root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "checksum-inventory.json":
            continue
        rows.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    atomic_json(
        recovery_root / "checksum-inventory.json",
        {
            "schema": "atlaslens-phase3f-bootstrap-checksum-inventory-v1",
            "files": rows,
            "file_count": len(rows),
            "secrets_included": False,
        },
    )


def _progress(
    recovery_root: Path, atomic_json: AtomicWriter, *, stage: str, started: float
) -> None:
    elapsed = time.monotonic() - started
    atomic_json(
        recovery_root / "bootstrap-progress.json",
        {
            "schema": "atlaslens-phase3f-bootstrap-progress-v1",
            "stage": stage,
            "elapsed_minutes": int(elapsed // 60),
            "elapsed_seconds": elapsed,
            "training_started": False,
            "secrets_included": False,
        },
    )


def _run_logged(
    command: list[str],
    *,
    recovery_root: Path,
    environment: dict[str, str],
    started: float,
    deadline: float,
    stage: str,
    atomic_json: AtomicWriter,
) -> tuple[int, str, str, str | None]:
    stdout_raw = recovery_root.parent / ".phase3f-bootstrap-stdout.raw"
    stderr_raw = recovery_root.parent / ".phase3f-bootstrap-stderr.raw"
    stdout_raw.unlink(missing_ok=True)
    stderr_raw.unlink(missing_ok=True)
    exception_class: str | None = None
    return_code = 1
    try:
        with stdout_raw.open("xb") as stdout, stderr_raw.open("xb") as stderr:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
            next_minute = int((time.monotonic() - started) // 60) + 1
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    exception_class = "TimeoutExpired"
                    return_code = 124
                    break
                elapsed_minute = int((now - started) // 60)
                if elapsed_minute >= next_minute:
                    _progress(recovery_root, atomic_json, stage=stage, started=started)
                    next_minute = elapsed_minute + 1
                time.sleep(1)
            else:
                return_code = process.returncode if process.returncode is not None else 1
            if process.poll() is not None and return_code != 124:
                return_code = process.returncode if process.returncode is not None else 1
    except OSError as exc:
        exception_class = type(exc).__name__
        return_code = 1
    stdout_text = _bounded_raw_text(stdout_raw)
    stderr_text = _bounded_raw_text(stderr_raw)
    return return_code, stdout_text, stderr_text, exception_class


def _failure_documents(
    recovery_root: Path,
    *,
    atomic_json: AtomicWriter,
    sanitized_tail: TailSanitizer,
    failure_code: str,
    failed_module: str,
    exception_class: str,
    report: dict[str, object],
    receipt: dict[str, object],
    stdout_text: str,
    stderr_text: str,
    process_exit_code: int | None,
) -> None:
    report.update(
        {
            "outcome": "failed",
            "dependency_failure_code": failure_code,
            "failed_module": failed_module,
            "failure_exception_class": exception_class,
            "secrets_included": False,
        }
    )
    receipt.update(
        {
            "outcome": "failed",
            "dependency_failure_code": failure_code,
            "failed_module": failed_module,
            "failure_exception_class": exception_class,
            "all_imports_passed": False,
            "secrets_included": False,
        }
    )
    atomic_json(recovery_root / "dependency-report.json", report)
    atomic_json(recovery_root / "environment-receipt.json", receipt)
    _write_log(recovery_root / "bootstrap-stdout.log", stdout_text, sanitized_tail)
    _write_log(recovery_root / "bootstrap-stderr.log", stderr_text, sanitized_tail)
    atomic_json(
        recovery_root / "failure.json",
        {
            "schema": "atlaslens-phase3f-remote-training-failure-v1",
            "failure_code": "REMOTE_TRAINING_DEPENDENCY_FAILED",
            "dependency_failure_code": failure_code,
            "failed_module": failed_module,
            "exception_class": exception_class,
            "process_exit_code": process_exit_code,
            "stdout_tail": list(sanitized_tail(stdout_text)),
            "stderr_tail": list(sanitized_tail(stderr_text)),
            "training_started": False,
            "secrets_included": False,
        },
    )
    _remove_raw_logs(recovery_root)
    _inventory(recovery_root, atomic_json)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 60 <= args.timeout_seconds <= 600:
        print("REMOTE_DEPENDENCY_LOCK_MISMATCH")
        return 2
    source = args.repository_root.resolve() / "services" / "api" / "src"
    if not source.is_dir() or source.is_symlink():
        print("REMOTE_DEPENDENCY_LOCK_MISMATCH")
        return 2
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.remote_environment import (  # noqa: PLC0415
        REMOTE_DEPENDENCY_REPORT_SCHEMA,
        REMOTE_ENVIRONMENT_RECEIPT_SCHEMA,
        RemoteEnvironmentError,
        evaluate_remote_environment,
        load_remote_environment_contract,
    )
    from atlaslens_api.phase3f.training_recovery import (  # noqa: PLC0415
        atomic_json,
        sanitized_tail,
    )

    os.environ.pop("MAPILLARY_ACCESS_TOKEN", None)
    os.environ.pop("RUNPOD_API_KEY", None)
    args.recovery_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    empty_report: dict[str, object] = {
        "schema": REMOTE_DEPENDENCY_REPORT_SCHEMA,
        "checks": [],
        "check_count": 0,
    }
    empty_receipt: dict[str, object] = {
        "schema": REMOTE_ENVIRONMENT_RECEIPT_SCHEMA,
        "interpreter_class": "system_python",
    }
    try:
        contract = load_remote_environment_contract(
            args.contract,
            args.requirements_lock,
            expected_contract_sha256=args.expected_contract_sha256,
            expected_lock_sha256=args.expected_lock_sha256,
        )
    except RemoteEnvironmentError as exc:
        _failure_documents(
            args.recovery_root,
            atomic_json=atomic_json,
            sanitized_tail=sanitized_tail,
            failure_code=exc.code,
            failed_module="dependency-lock",
            exception_class=type(exc).__name__,
            report=empty_report,
            receipt=empty_receipt,
            stdout_text="",
            stderr_text="",
            process_exit_code=90,
        )
        print(exc.code)
        return 90
    base = evaluate_remote_environment(
        contract,
        expected_interpreter_class="system_python",
        vendor_root=None,
        stage="base",
    )
    if not base.passed:
        _failure_documents(
            args.recovery_root,
            atomic_json=atomic_json,
            sanitized_tail=sanitized_tail,
            failure_code=base.failure_code or "REMOTE_DEPENDENCY_IMPORT_FAILED",
            failed_module=base.failed_module or "base-runtime",
            exception_class=base.exception_class or "DependencyBootstrapError",
            report=base.report,
            receipt=base.environment_receipt,
            stdout_text="",
            stderr_text="",
            process_exit_code=91,
        )
        print(base.failure_code)
        return 91
    environment = _safe_environment()
    commands = (
        (
            [sys.executable, "-m", "venv", "--system-site-packages", str(args.venv_root)],
            "creating_venv",
            "venv",
        ),
        (
            [
                str(_venv_python(args.venv_root)),
                "-m",
                "pip",
                "install",
                "--isolated",
                "--no-deps",
                "--require-hashes",
                "--only-binary=:all:",
                "--index-url",
                contract.official_index_url,
                "--timeout",
                "60",
                "--retries",
                "2",
                "-r",
                str(args.requirements_lock),
            ],
            "installing_locked_packages",
            "phase3f-training-lock",
        ),
        (
            [
                str(_venv_python(args.venv_root)),
                str(args.repository_root / "scripts" / "phase3f" / "training_job.py"),
                "--dependency-preflight-only",
                "--repository-root",
                str(args.repository_root),
                "--vendor-root",
                str(args.vendor_root),
                "--recovery-root",
                str(args.recovery_root),
                "--environment-contract",
                str(args.contract),
                "--requirements-lock",
                str(args.requirements_lock),
                "--expected-contract-sha256",
                args.expected_contract_sha256,
                "--expected-lock-sha256",
                args.expected_lock_sha256,
            ],
            "validating_project_imports",
            "project-import-graph",
        ),
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    for command, stage, failed_module in commands:
        _progress(args.recovery_root, atomic_json, stage=stage, started=started)
        return_code, stdout_text, stderr_text, exception_class = _run_logged(
            command,
            recovery_root=args.recovery_root,
            environment=environment,
            started=started,
            deadline=deadline,
            stage=stage,
            atomic_json=atomic_json,
        )
        stdout_parts.append(stdout_text)
        stderr_parts.append(stderr_text)
        if return_code != 0:
            report = base.report
            receipt = base.environment_receipt
            dependency_report = args.recovery_root / "dependency-report.json"
            environment_receipt = args.recovery_root / "environment-receipt.json"
            try:
                candidate_report = json.loads(dependency_report.read_text(encoding="utf-8"))
                candidate_receipt = json.loads(environment_receipt.read_text(encoding="utf-8"))
                if isinstance(candidate_report, dict) and isinstance(candidate_receipt, dict):
                    report = candidate_report
                    receipt = candidate_receipt
                    exact = candidate_report.get("dependency_failure_code")
                    module = candidate_report.get("failed_module")
                    observed_exception = candidate_report.get("failure_exception_class")
                    failure_code = (
                        exact if isinstance(exact, str) else "REMOTE_DEPENDENCY_IMPORT_FAILED"
                    )
                    failed_module = module if isinstance(module, str) else failed_module
                    exception_class = (
                        observed_exception
                        if isinstance(observed_exception, str)
                        else exception_class
                    )
                else:
                    failure_code = "REMOTE_DEPENDENCY_IMPORT_FAILED"
            except (OSError, UnicodeError, json.JSONDecodeError):
                failure_code = "REMOTE_DEPENDENCY_IMPORT_FAILED"
            _failure_documents(
                args.recovery_root,
                atomic_json=atomic_json,
                sanitized_tail=sanitized_tail,
                failure_code=failure_code,
                failed_module=failed_module,
                exception_class=exception_class or "DependencyBootstrapError",
                report=report,
                receipt=receipt,
                stdout_text="\n".join(stdout_parts),
                stderr_text="\n".join(stderr_parts),
                process_exit_code=return_code,
            )
            print(failure_code)
            return return_code if 0 < return_code <= 255 else 1
    _write_log(
        args.recovery_root / "bootstrap-stdout.log",
        "\n".join(stdout_parts),
        sanitized_tail,
    )
    _write_log(
        args.recovery_root / "bootstrap-stderr.log",
        "\n".join(stderr_parts),
        sanitized_tail,
    )
    try:
        from atlaslens_api.phase3f.remote_environment import (  # noqa: PLC0415
            require_dependency_report,
            require_environment_receipt,
        )

        require_dependency_report(
            args.recovery_root / "dependency-report.json",
            expected_contract_sha256=args.expected_contract_sha256,
            expected_lock_sha256=args.expected_lock_sha256,
            expected_check_count=len(contract.checks),
        )
        require_environment_receipt(
            args.recovery_root / "environment-receipt.json",
            expected_contract_sha256=args.expected_contract_sha256,
            expected_lock_sha256=args.expected_lock_sha256,
            validate_current_interpreter=False,
        )
    except RemoteEnvironmentError as exc:
        _failure_documents(
            args.recovery_root,
            atomic_json=atomic_json,
            sanitized_tail=sanitized_tail,
            failure_code=exc.code,
            failed_module="environment-receipt",
            exception_class=type(exc).__name__,
            report=base.report,
            receipt=base.environment_receipt,
            stdout_text="\n".join(stdout_parts),
            stderr_text="\n".join(stderr_parts),
            process_exit_code=94,
        )
        print(exc.code)
        return 94
    _progress(args.recovery_root, atomic_json, stage="ready", started=started)
    _remove_raw_logs(args.recovery_root)
    _inventory(args.recovery_root, atomic_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
