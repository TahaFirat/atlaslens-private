"""Secret-free local controls for the one-command Phase 3F operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

REQUIRED_BRANCH = "feature/phase3f-multiregion-pilot"
FROZEN_START_HEAD = "17e1a14f2ccc96418a076d451a03fcdf981988bb"
CURRENT_SCHEMA = "atlaslens-phase3f-local-first-current-v1"
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


class EndToEndError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise EndToEndError(code)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=D:/geoSearch", *arguments],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    _require(completed.returncode == 0, "GIT_PREFLIGHT_FAILED")
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _read_json(path: Path, *, max_bytes: int) -> dict[str, object]:
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= max_bytes,
        "RUNTIME_STATE_INVALID",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EndToEndError("RUNTIME_STATE_INVALID") from exc
    _require(isinstance(value, dict), "RUNTIME_STATE_INVALID")
    return cast(dict[str, object], value)


def _runtime_snapshot(runtime_root: Path) -> dict[str, object]:
    current_path = runtime_root / "current.json"
    if not current_path.exists():
        return {
            "current_run_present": False,
            "run_id": None,
            "metadata_row_count": 0,
            "metadata_sha256": None,
            "checkpoint_present": False,
        }
    current = _read_json(current_path, max_bytes=4096)
    run_id = current.get("run_id")
    _require(
        current.get("schema") == CURRENT_SCHEMA
        and isinstance(run_id, str)
        and bool(_RUN_ID.fullmatch(run_id)),
        "CURRENT_RUN_INVALID",
    )
    checkpoint = runtime_root / run_id / "acquisition-work" / "metadata-pages.json"
    sealed = runtime_root / run_id / "sealed-acquisition"
    if checkpoint.exists():
        document = _read_json(checkpoint, max_bytes=64 * 1024 * 1024)
        raw_rows = document.get("rows_by_city")
        _require(isinstance(raw_rows, dict), "METADATA_CHECKPOINT_INVALID")
        rows = cast(dict[str, object], raw_rows)
        _require(all(isinstance(value, list) for value in rows.values()), "METADATA_CHECKPOINT_INVALID")
        payload = checkpoint.read_bytes()
        return {
            "current_run_present": True,
            "run_id": run_id,
            "metadata_row_count": sum(len(cast(list[object], value)) for value in rows.values()),
            "metadata_sha256": hashlib.sha256(payload).hexdigest(),
            "checkpoint_present": True,
            "sealed": False,
        }
    _require(sealed.is_dir() and not sealed.is_symlink(), "CURRENT_RUN_CHECKPOINT_MISSING")
    manifest = _read_json(sealed / "manifest.json", max_bytes=1024 * 1024)
    return {
        "current_run_present": True,
        "run_id": run_id,
        "metadata_row_count": None,
        "metadata_sha256": None,
        "checkpoint_present": False,
        "sealed": manifest.get("sealed") is True,
        "asset_count": manifest.get("asset_count"),
    }


def preflight(repository: Path, runtime_root: Path) -> dict[str, object]:
    repository = repository.resolve()
    _require(repository.is_dir() and not repository.is_symlink(), "REPOSITORY_INVALID")
    branch = _git(repository, "branch", "--show-current")
    head = _git(repository, "rev-parse", "HEAD")
    _require(branch == REQUIRED_BRANCH, "GIT_BRANCH_MISMATCH")
    _git(repository, "merge-base", "--is-ancestor", FROZEN_START_HEAD, head)
    status = tuple(
        row
        for row in _git(repository, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if row
    )
    _require(status in {(), ("?? asda.html",)}, "GIT_WORKTREE_NOT_CLEAN")
    c_free = shutil.disk_usage("C:\\").free
    d_free = shutil.disk_usage("D:\\").free
    _require(c_free >= 35 * 1024**3, "C_DISK_GATE_FAILED")
    _require(d_free >= 8 * 1024**3, "D_DISK_GATE_FAILED")
    snapshot = _runtime_snapshot(runtime_root.resolve()) if runtime_root.exists() else {
        "current_run_present": False,
        "run_id": None,
        "metadata_row_count": 0,
        "metadata_sha256": None,
        "checkpoint_present": False,
    }
    return {
        "action": "preflight",
        "branch": branch,
        "head": head,
        "frozen_start_is_ancestor": True,
        "worktree_clean": True,
        "ignored_untracked": ["asda.html"] if status else [],
        "c_free_bytes": c_free,
        "d_free_bytes": d_free,
        "runtime": snapshot,
        "runpod_api_calls": 0,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def readiness(repository: Path, runtime_root: Path) -> dict[str, object]:
    source = repository.resolve() / "services" / "api" / "src"
    _require(source.is_dir() and not source.is_symlink(), "PHASE3F_SOURCE_ROOT_INVALID")
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.training import require_training_ready  # noqa: PLC0415

    snapshot = _runtime_snapshot(runtime_root.resolve())
    run_id = snapshot["run_id"]
    _require(isinstance(run_id, str), "CURRENT_RUN_INVALID")
    run_root = runtime_root.resolve() / run_id
    report_path = run_root / "training-readiness.json"
    sealed = require_training_ready(
        run_root / "sealed-acquisition",
        report_path=report_path,
    )
    return {
        "action": "readiness",
        "run_id": run_id,
        "asset_count": len(sealed.split.assets),
        "ready_for_training": True,
        "report_path_name": report_path.name,
        "runpod_api_calls": 0,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def status(repository: Path, local_runtime: Path, cloud_runtime: Path) -> dict[str, object]:
    source = repository.resolve() / "services" / "api" / "src"
    _require(source.is_dir(), "PHASE3F_SOURCE_ROOT_INVALID")
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.local_first import status as local_status  # noqa: PLC0415

    local = local_status(local_runtime)
    operator_path = cloud_runtime / "_operator" / "phase3f-current.json"
    operator: dict[str, object] | None = None
    if operator_path.exists():
        raw = _read_json(operator_path, max_bytes=1024 * 1024)
        operator = {
            key: raw.get(key)
            for key in (
                "run_id",
                "stage",
                "started_at",
                "finished_at",
                "cleanup_verified",
            )
        }
    return {
        "action": "status",
        "local": local,
        "cloud_operator": operator,
        "live_cloud_inventory_checked": False,
        "secrets_included": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-end-to-end")
    parser.add_argument("action", choices=("preflight", "readiness", "status"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cloud-runtime-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "preflight":
            result = preflight(args.repository_root, args.runtime_root)
        elif args.action == "readiness":
            result = readiness(args.repository_root, args.runtime_root)
        else:
            _require(args.cloud_runtime_root is not None, "CLOUD_RUNTIME_ROOT_MISSING")
            result = status(args.repository_root, args.runtime_root, args.cloud_runtime_root)
    except Exception as exc:  # sanitized operator boundary
        code = getattr(exc, "code", None)
        print(code if isinstance(code, str) else "PHASE3F_END_TO_END_FAILED")
        return 1
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
