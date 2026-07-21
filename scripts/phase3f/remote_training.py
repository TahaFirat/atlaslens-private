"""Bounded remote process wrapper that persists sanitized Phase 3F failure evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-remote-training")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--environment-contract", type=Path, required=True)
    parser.add_argument("--requirements-lock", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    return parser


def _bounded_text(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 256 * 1024))
        return stream.read(256 * 1024).decode("utf-8", errors="replace")


def _explicit_code(stdout: str, stderr: str) -> str | None:
    for line in reversed((*stdout.splitlines(), *stderr.splitlines())):
        candidate = line.strip()
        if candidate.isascii() and candidate.replace("_", "A").isalnum():
            if candidate.upper() == candidate and 3 <= len(candidate) <= 128:
                return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.repository_root.resolve() / "services" / "api" / "src"
    if not source.is_dir() or source.is_symlink():
        print("PHASE3F_SOURCE_ROOT_INVALID")
        return 2
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.training_recovery import (  # noqa: PLC0415
        REMOTE_FAILURE_SCHEMA,
        REMOTE_TELEMETRY_SCHEMA,
        atomic_json,
        classify_remote_failure,
    )

    args.recovery_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stdout_path = args.recovery_root / "stdout.log"
    stderr_path = args.recovery_root / "stderr.log"
    command = [
        sys.executable,
        str(args.repository_root / "scripts" / "phase3f" / "training_job.py"),
        "--repository-root",
        str(args.repository_root),
        "--run-id",
        args.run_id,
        "--sealed-root",
        str(args.sealed_root),
        "--model",
        str(args.model),
        "--vendor-root",
        str(args.vendor_root),
        "--work-root",
        str(args.work_root),
        "--output-root",
        str(args.output_root),
        "--recovery-root",
        str(args.recovery_root),
        "--deadline-epoch",
        str(args.deadline_epoch),
        "--environment-contract",
        str(args.environment_contract),
        "--requirements-lock",
        str(args.requirements_lock),
        "--expected-contract-sha256",
        args.expected_contract_sha256,
        "--expected-lock-sha256",
        args.expected_lock_sha256,
        "--environment-receipt",
        str(args.environment_receipt),
    ]
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env={
                key: value
                for key, value in os.environ.items()
                if key
                not in {
                    "MAPILLARY_ACCESS_TOKEN",
                    "RUNPOD_API_KEY",
                    "PYTHONHOME",
                    "PYTHONPATH",
                    "VIRTUAL_ENV",
                    "CONDA_PREFIX",
                }
            },
        )
        try:
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait(timeout=30)
    stdout_text = _bounded_text(stdout_path)
    stderr_text = _bounded_text(stderr_path)
    child_failure: dict[str, object] = {}
    try:
        candidate = json.loads(
            (args.recovery_root / "child-failure.json").read_text(encoding="utf-8")
        )
        if isinstance(candidate, dict):
            child_failure = candidate
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    child_code = child_failure.get("message_code")
    exception_class = child_failure.get("exception_class")
    diagnostic = classify_remote_failure(
        124 if timed_out else return_code,
        stdout=stdout_text,
        stderr=stderr_text,
        explicit_code=(
            "REMOTE_TRAINING_DEADLINE"
            if timed_out
            else (
                child_code
                if isinstance(child_code, str)
                else _explicit_code(stdout_text, stderr_text)
            )
        ),
        exception_class=(exception_class if isinstance(exception_class, str) else None),
    )
    disk = shutil.disk_usage(args.work_root.parent)
    peak_value = child_failure.get("peak_cuda_bytes")
    peak_cuda_bytes = (
        peak_value
        if isinstance(peak_value, int) and not isinstance(peak_value, bool) and peak_value >= 0
        else None
    )
    atomic_json(
        args.recovery_root / "telemetry.json",
        {
            "schema": REMOTE_TELEMETRY_SCHEMA,
            "elapsed_seconds": time.monotonic() - started,
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
            "peak_cuda_bytes": peak_cuda_bytes,
            "secrets_included": False,
        },
    )
    if return_code != 0 or timed_out:
        progress: dict[str, object] = {}
        progress_path = args.recovery_root / "progress.json"
        try:
            candidate = json.loads(progress_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                progress = candidate
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        atomic_json(
            args.recovery_root / "failure.json",
            {
                "schema": REMOTE_FAILURE_SCHEMA,
                **diagnostic.to_dict(),
                "last_completed_epoch": progress.get("completed_epoch"),
                "last_epoch": progress.get("epoch"),
                "last_step": progress.get("optimizer_steps"),
                "next_batch_index": progress.get("next_batch_index"),
                "traceback_tail": child_failure.get("traceback_tail", []),
                "holdout_open_count": progress.get("holdout_open_count", 0),
                "peak_cuda_bytes": peak_cuda_bytes,
                "disk_free_bytes": disk.free,
                "secrets_included": False,
            },
        )
        print(diagnostic.failure_code)
        return return_code if 0 < return_code <= 255 and not timed_out else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
