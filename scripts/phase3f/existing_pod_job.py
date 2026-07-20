"""In-Pod, local-only Phase 3F job controls; never calls the RunPod API."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import signal
import secrets
import stat
import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Callable
from typing import BinaryIO, cast
from uuid import uuid4

_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MAX_WALL_SECONDS = 16_200
_MODEL_SIZE = 914_577_436
_MODEL_SHA256 = "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
_SOURCE_SHA256 = "3cbf1d20515b1da423998a8edab787031eaa7bb273c5a86a5c41c4f6d84e2a6d"
_LICENSE_SHA256 = "0a906f9a65db6f645483f6cbf56b01e20615b9b943df3f70112f3d0fe0521e2a"
_SAFE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_SAFE_PROGRESS_STAGES = frozenset(
    {"metadata_page_checkpointed", "metadata_city_complete"}
)
_SAFE_CHILD_ENV_NAMES = frozenset(
    {
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "TMPDIR",
        "TZ",
    }
)


class ManualJobError(RuntimeError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ManualJobError(code)


def _child_environment(*, include_mapillary_token: bool) -> dict[str, str]:
    environment = {
        name: value
        for name in _SAFE_CHILD_ENV_NAMES
        if (value := os.environ.get(name)) is not None
    }
    if include_mapillary_token:
        token = os.environ.get("MAPILLARY_ACCESS_TOKEN")
        _require(bool(token), "MAPILLARY_ACCESS_TOKEN_MISSING")
        environment["MAPILLARY_ACCESS_TOKEN"] = cast(str, token)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _open_private_regular(path: Path, flags: int, code: str) -> int:
    descriptor = os.open(
        path,
        flags | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        information = os.fstat(descriptor)
        _require(
            stat.S_ISREG(information.st_mode) and information.st_nlink == 1,
            code,
        )
        descriptor_chmod = getattr(os, "fchmod", None)
        if descriptor_chmod is not None:
            cast(Callable[[int, int], None], descriptor_chmod)(descriptor, 0o600)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _fsync_parent(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if directory_flag == 0:
        return
    descriptor = os.open(path.parent, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_private_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _proc_identity(pid: int) -> tuple[str, str]:
    _require(pid > 1, "PHASE3F_JOB_PID_INVALID")
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text(encoding="ascii")
        cmdline = (root / "cmdline").read_bytes()
    except OSError as exc:
        raise ManualJobError("PHASE3F_JOB_NOT_RUNNING") from exc
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split()
    _require(closing > 0 and len(fields) > 19, "PHASE3F_JOB_IDENTITY_INVALID")
    start_ticks = fields[19]
    command_sha = hashlib.sha256(cmdline).hexdigest()
    return start_ticks, command_sha


def _read_state(path: Path, run_id: str) -> dict[str, object]:
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 16_384,
        "PHASE3F_JOB_STATE_MISSING",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManualJobError("PHASE3F_JOB_STATE_INVALID") from exc
    _require(isinstance(value, dict), "PHASE3F_JOB_STATE_INVALID")
    state = {str(key): item for key, item in value.items()}
    _require(
        set(state)
        == {
            "schema",
            "run_id",
            "pid",
            "proc_start_ticks",
            "command_sha256",
            "output_name",
            "deadline_epoch",
            "resume",
        }
        and state.get("schema") == "atlaslens-phase3f-in-pod-job-v1"
        and state.get("run_id") == run_id
        and isinstance(state.get("pid"), int)
        and not isinstance(state.get("pid"), bool)
        and isinstance(state.get("deadline_epoch"), int)
        and not isinstance(state.get("deadline_epoch"), bool)
        and isinstance(state.get("resume"), bool),
        "PHASE3F_JOB_STATE_INVALID",
    )
    return state


def _validated_process(state: dict[str, object]) -> int:
    pid = cast(int, state["pid"])
    start_ticks, command_sha = _proc_identity(pid)
    _require(
        start_ticks == state.get("proc_start_ticks")
        and command_sha == state.get("command_sha256"),
        "PHASE3F_JOB_STALE_PID",
    )
    return pid


def _terminate_process_group(pid: int) -> None:
    kill_process_group = cast(Callable[[int, int], None], getattr(os, "killpg"))
    kill_process_group(pid, signal.SIGTERM)


def _existing_cloud_jobs() -> tuple[int, ...]:
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"scripts/phase3f/cloud_job.py" in command:
            found.append(int(entry.name))
    return tuple(sorted(found))


def _paths(runtime_root: Path, run_id: str) -> tuple[Path, Path, Path, Path, Path]:
    resolved_runtime = runtime_root.resolve()
    _require(
        resolved_runtime.is_dir() and not resolved_runtime.is_symlink(),
        "RUNTIME_ROOT_INVALID",
    )
    resolved = (resolved_runtime / run_id).resolve()
    _require(
        resolved.parent == resolved_runtime
        and resolved.name == run_id
        and not resolved.is_symlink(),
        "RUN_ROOT_INVALID",
    )
    return (
        resolved,
        resolved / "job-state.json",
        resolved / "job.log",
        resolved / "work",
        resolved_runtime / ".phase3f-global-job.lock",
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_lf_sha256(path: Path, *, max_bytes: int) -> str:
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= max_bytes,
        "PHASE3F_VENDOR_ARTIFACT_INVALID",
    )
    payload = path.read_bytes()
    _require(
        not payload.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")),
        "PHASE3F_VENDOR_BOM_REFUSED",
    )
    without_crlf = payload.replace(b"\r\n", b"")
    _require(b"\r" not in without_crlf, "PHASE3F_VENDOR_LONE_CR_REFUSED")
    crlf_count = payload.count(b"\r\n")
    _require(
        crlf_count == 0 or payload.count(b"\n") == crlf_count,
        "PHASE3F_VENDOR_MIXED_NEWLINES_REFUSED",
    )
    return hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()


def _prepare_readiness(
    repository: Path,
    model: Path,
    vendor_root: Path,
    source_commit: str,
) -> None:
    _require(
        repository.is_dir()
        and not repository.is_symlink()
        and (repository / "scripts" / "phase3f" / "cloud_job.py").is_file(),
        "PHASE3F_SOURCE_ROOT_INVALID",
    )
    _require(bool(_SOURCE_COMMIT.fullmatch(source_commit)), "PHASE3F_SOURCE_COMMIT_INVALID")
    git_environment = _child_environment(include_mapillary_token=False)
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD^{commit}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=git_environment,
        check=False,
        timeout=30,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=git_environment,
        check=False,
        timeout=30,
        text=True,
    )
    _require(
        head.returncode == 0
        and head.stdout.strip() == source_commit
        and status.returncode == 0
        and not status.stdout,
        "PHASE3F_SOURCE_COMMIT_MISMATCH",
    )
    _require(
        model.is_file()
        and not model.is_symlink()
        and model.stat().st_size == _MODEL_SIZE
        and _sha256_path(model) == _MODEL_SHA256,
        "PHASE3F_MODEL_HASH_MISMATCH",
    )
    _require(
        _canonical_lf_sha256(vendor_root / "megaloc_model.py", max_bytes=32_768)
        == _SOURCE_SHA256
        and _canonical_lf_sha256(vendor_root / "LICENSE", max_bytes=4096)
        == _LICENSE_SHA256,
        "PHASE3F_VENDOR_HASH_MISMATCH",
    )
    _require(bool(os.environ.get("MAPILLARY_ACCESS_TOKEN")), "MAPILLARY_ACCESS_TOKEN_MISSING")
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import faiss,httpx,numpy,PIL,pydantic,torch;"
                "raise SystemExit(0 if torch.cuda.is_available() else 1)"
            ),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_child_environment(include_mapillary_token=False),
        check=False,
        timeout=60,
    )
    _require(probe.returncode == 0, "PHASE3F_DEPENDENCY_OR_CUDA_NOT_READY")


def _read_current_run(runtime_root: Path) -> str:
    path = runtime_root / "current-root.json"
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 4096,
        "PHASE3F_CURRENT_ROOT_MISSING",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManualJobError("PHASE3F_CURRENT_ROOT_INVALID") from exc
    _require(
        isinstance(value, dict)
        and set(value) == {"schema", "run_id"}
        and value.get("schema") == "atlaslens-phase3f-current-root-v1"
        and isinstance(value.get("run_id"), str)
        and bool(_RUN_ID.fullmatch(str(value.get("run_id")))),
        "PHASE3F_CURRENT_ROOT_INVALID",
    )
    return str(value["run_id"])


def _safe_log_line(line: str) -> str:
    if _SAFE_CODE.fullmatch(line):
        return line
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return "[REDACTED_UNSAFE_LOG_LINE]"
    if not isinstance(value, dict):
        return "[REDACTED_UNSAFE_LOG_LINE]"
    if value.get("event") == "PHASE3F_MAPILLARY_METADATA_PROGRESS":
        stage = value.get("stage")
        page_count = value.get("page_count")
        region = value.get("region")
        if (
            stage in _SAFE_PROGRESS_STAGES
            and isinstance(page_count, int)
            and not isinstance(page_count, bool)
            and isinstance(region, str)
            and 1 <= len(region) <= 40
            and value.get("secrets_included") is False
        ):
            return json.dumps(
                {
                    "event": value["event"],
                    "page_count": page_count,
                    "region": region,
                    "secrets_included": False,
                    "stage": stage,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
    if value.get("finished") is True and value.get("secrets_included") is False:
        outcome = value.get("outcome")
        run_id = value.get("run_id")
        if isinstance(outcome, str) and _SAFE_CODE.fullmatch(outcome) and run_id is not None:
            return json.dumps(
                {
                    "finished": True,
                    "outcome": outcome,
                    "run_id_sha256": hashlib.sha256(str(run_id).encode()).hexdigest(),
                    "secrets_included": False,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
    return "[REDACTED_UNSAFE_LOG_LINE]"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-existing-pod-job")
    parser.add_argument(
        "action",
        choices=("prepare-check", "start-job", "status-job", "tail-log", "stop-job"),
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--vendor-root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--lines", type=int, default=80)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lock_stream: BinaryIO | None = None
    try:
        runtime_root = args.runtime_root.resolve()
        _require(runtime_root.is_dir() and not runtime_root.is_symlink(), "RUNTIME_ROOT_INVALID")
        os.chmod(runtime_root, 0o700)
        run_id = secrets.token_hex(16) if args.action == "prepare-check" else _read_current_run(runtime_root)
        run_parent, state_path, log_path, work_root, lock_path = _paths(runtime_root, run_id)
        repository = args.repository_root.resolve() if args.repository_root is not None else None
        if args.action in {"prepare-check", "start-job"}:
            if (
                repository is None
                or args.model is None
                or args.vendor_root is None
                or args.source_commit is None
            ):
                raise ManualJobError("PHASE3F_READINESS_ARGUMENTS_MISSING")
            _prepare_readiness(
                repository,
                args.model.resolve(),
                args.vendor_root.resolve(),
                args.source_commit,
            )
        if args.action == "prepare-check":
            _require(not _existing_cloud_jobs(), "PHASE3F_JOB_ALREADY_RUNNING")
            _require(
                not (runtime_root / "current-root.json").exists(),
                "PHASE3F_CURRENT_ROOT_ALREADY_EXISTS",
            )
            run_parent.mkdir(mode=0o700, exist_ok=False)
            os.chmod(run_parent, 0o700)
            _atomic_private_json(
                runtime_root / "current-root.json",
                {"schema": "atlaslens-phase3f-current-root-v1", "run_id": run_id},
            )
        _require(run_parent.is_dir(), "RUN_ROOT_MISSING")

        if args.action == "prepare-check":
            print("PHASE3F_IN_POD_PREPARE_CHECK_PASS")
            return 0
        if args.action == "start-job":
            if (
                repository is None
                or args.model is None
                or args.vendor_root is None
                or args.source_commit is None
            ):
                raise ManualJobError("PHASE3F_READINESS_ARGUMENTS_MISSING")
            model = args.model.resolve()
            vendor_root = args.vendor_root.resolve()
            fcntl = importlib.import_module("fcntl")

            _require(not _existing_cloud_jobs(), "PHASE3F_JOB_ALREADY_RUNNING")
            lock_descriptor = _open_private_regular(
                lock_path,
                os.O_CREAT | os.O_APPEND | os.O_RDWR,
                "PHASE3F_JOB_LOCK_INVALID",
            )
            lock_stream = os.fdopen(lock_descriptor, "a+b", closefd=True)
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ManualJobError("PHASE3F_JOB_ALREADY_RUNNING") from exc
            os.chmod(lock_path, 0o600)
            if state_path.exists():
                prior_state = _read_state(state_path, run_id)
                try:
                    _validated_process(prior_state)
                except ManualJobError as exc:
                    _require(
                        str(exc) == "PHASE3F_JOB_NOT_RUNNING",
                        "PHASE3F_JOB_STALE_PID",
                    )
                else:
                    raise ManualJobError("PHASE3F_JOB_ALREADY_RUNNING")
            output_name = f"output-{int(time.time())}-{secrets.token_hex(4)}"
            output_root = run_parent / output_name
            _require(not output_root.exists(), "PHASE3F_OUTPUT_ALREADY_EXISTS")
            command = [
                "timeout",
                "--signal=TERM",
                str(_MAX_WALL_SECONDS),
                sys.executable,
                str(repository / "scripts" / "phase3f" / "cloud_job.py"),
                "--repository-root",
                str(repository),
                "--run-id",
                run_id,
                "--model",
                str(model),
                "--vendor-root",
                str(vendor_root),
                "--work-root",
                str(work_root),
                "--output-root",
                str(output_root),
                "--deadline-epoch",
                str(int(time.time() + _MAX_WALL_SECONDS)),
            ]
            metadata_checkpoint = work_root / "metadata-pages.json"
            counter_checkpoint = work_root / "client-counters.json"
            checkpoint_presence = (
                metadata_checkpoint.is_file() and counter_checkpoint.is_file()
            )
            _require(
                checkpoint_presence
                or (not metadata_checkpoint.exists() and not counter_checkpoint.exists()),
                "PHASE3F_RESUME_CHECKPOINT_INCOMPLETE",
            )
            if checkpoint_presence:
                command.append("--resume")
            log_descriptor = _open_private_regular(
                log_path,
                os.O_CREAT | os.O_APPEND | os.O_WRONLY,
                "PHASE3F_JOB_LOG_INVALID",
            )
            os.set_inheritable(lock_stream.fileno(), True)
            with os.fdopen(log_descriptor, "ab", closefd=True) as log_stream:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=log_stream,
                    start_new_session=True,
                    close_fds=True,
                    pass_fds=(lock_stream.fileno(),),
                    env=_child_environment(include_mapillary_token=True),
                )
            time.sleep(0.1)
            _require(process.poll() is None, "PHASE3F_JOB_START_FAILED")
            start_ticks, command_sha = _proc_identity(process.pid)
            try:
                _atomic_private_json(
                    state_path,
                    {
                        "schema": "atlaslens-phase3f-in-pod-job-v1",
                        "run_id": run_id,
                        "pid": process.pid,
                        "proc_start_ticks": start_ticks,
                        "command_sha256": command_sha,
                        "output_name": output_name,
                        "deadline_epoch": int(time.time() + _MAX_WALL_SECONDS),
                        "resume": checkpoint_presence,
                    },
                )
            except OSError:
                _terminate_process_group(process.pid)
                process.wait(timeout=30)
                raise
            print("PHASE3F_IN_POD_JOB_STARTED")
            return 0

        state = _read_state(state_path, run_id)
        if args.action == "status-job":
            try:
                _validated_process(state)
            except ManualJobError as exc:
                if str(exc) == "PHASE3F_JOB_NOT_RUNNING":
                    print("PHASE3F_IN_POD_JOB_STOPPED")
                    return 0
                raise
            print("PHASE3F_IN_POD_JOB_RUNNING")
            return 0
        if args.action == "tail-log":
            _require(1 <= args.lines <= 200, "TAIL_LINE_LIMIT_INVALID")
            _require(
                log_path.is_file()
                and not log_path.is_symlink()
                and log_path.stat().st_size <= 16 * 1024 * 1024,
                "PHASE3F_JOB_LOG_MISSING",
            )
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-args.lines :]:
                print(_safe_log_line(line))
            return 0

        pid = _validated_process(state)
        _terminate_process_group(pid)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                _proc_identity(pid)
            except ManualJobError:
                print("PHASE3F_IN_POD_JOB_STOPPED")
                return 0
            time.sleep(0.25)
        raise ManualJobError("PHASE3F_JOB_STOP_TIMEOUT")
    except (ManualJobError, OSError, subprocess.SubprocessError) as exc:
        code = str(exc)
        print(code if re.fullmatch(r"[A-Z0-9_]+", code) else "PHASE3F_MANUAL_JOB_FAILED")
        return 1
    finally:
        if lock_stream is not None:
            lock_stream.close()  # child retains the inherited lock descriptor


if __name__ == "__main__":
    raise SystemExit(main())
