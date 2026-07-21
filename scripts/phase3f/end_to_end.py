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
    def __init__(
        self,
        code: str,
        *,
        artifact: str | None = None,
        field: str | None = None,
        expected: str | int | bool | None = None,
        actual: str | int | bool | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.artifact = artifact
        self.field = field
        self.expected = expected
        self.actual = actual

    def public_document(self) -> dict[str, object]:
        document: dict[str, object] = {"error_code": self.code}
        for key in ("artifact", "field", "expected", "actual"):
            value = getattr(self, key)
            if value is not None:
                document[key] = value
        return document


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


def _sha256_path(path: Path, *, artifact: str, max_bytes: int) -> str:
    if not path.is_file() or path.is_symlink():
        raise EndToEndError(
            "TRAINING_READINESS_REPORT_MISSING",
            artifact=artifact,
        )
    if path.stat().st_size > max_bytes:
        raise EndToEndError(
            "TRAINING_READINESS_REPORT_INVALID",
            artifact=artifact,
            field="size_bytes",
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integrity_json(
    path: Path,
    *,
    artifact: str,
    missing_code: str,
    invalid_code: str,
    max_bytes: int,
) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise EndToEndError(missing_code, artifact=artifact)
    if path.stat().st_size > max_bytes:
        raise EndToEndError(invalid_code, artifact=artifact, field="size_bytes")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EndToEndError(invalid_code, artifact=artifact, field="json") from exc
    if not isinstance(value, dict):
        raise EndToEndError(invalid_code, artifact=artifact, field="document")
    return cast(dict[str, object], value)


def _sealed_error(exc: BaseException) -> EndToEndError:
    raw_code = getattr(exc, "code", None)
    code = (
        raw_code
        if isinstance(raw_code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", raw_code)
        else "SEALED_BUNDLE_INVALID"
    )
    artifact_by_code = {
        "SEALED_BUNDLE_MISSING": "sealed-acquisition",
        "SEALED_INVENTORY_INVALID": "checksum-inventory.json",
        "SEALED_INVENTORY_MISMATCH": "checksum-inventory.json",
        "SEALED_MANIFEST_INVALID": "manifest.json",
        "SEALED_MANIFEST_MISMATCH": "manifest.json",
        "SEALED_SELECTION_INVALID": "selection-lock.json",
        "SEALED_ASSETS_INVALID": "sealed-assets.json",
        "SEALED_MEDIA_MISMATCH": "assets",
        "SEALED_RECEIPT_INVALID": "acquisition-receipt.json",
        "SEALED_PROVENANCE_INVALID": "provenance-aggregate.json",
    }
    return EndToEndError(
        code,
        artifact=artifact_by_code.get(code, "sealed-acquisition"),
        field=(
            "sha256"
            if code in {"SEALED_INVENTORY_MISMATCH", "SEALED_MEDIA_MISMATCH"}
            else None
        ),
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


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
    safe_run_id = cast(str, run_id)
    checkpoint = runtime_root / safe_run_id / "acquisition-work" / "metadata-pages.json"
    sealed = runtime_root / safe_run_id / "sealed-acquisition"
    if checkpoint.exists():
        document = _read_json(checkpoint, max_bytes=64 * 1024 * 1024)
        raw_rows = document.get("rows_by_city")
        _require(isinstance(raw_rows, dict), "METADATA_CHECKPOINT_INVALID")
        rows = cast(dict[str, object], raw_rows)
        _require(all(isinstance(value, list) for value in rows.values()), "METADATA_CHECKPOINT_INVALID")
        payload = checkpoint.read_bytes()
        return {
            "current_run_present": True,
            "run_id": safe_run_id,
            "metadata_row_count": sum(len(cast(list[object], value)) for value in rows.values()),
            "metadata_sha256": hashlib.sha256(payload).hexdigest(),
            "checkpoint_present": True,
            "sealed": False,
        }
    _require(sealed.is_dir() and not sealed.is_symlink(), "CURRENT_RUN_CHECKPOINT_MISSING")
    manifest = _read_json(sealed / "manifest.json", max_bytes=1024 * 1024)
    return {
        "current_run_present": True,
        "run_id": safe_run_id,
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
    from atlaslens_api.phase3f.training import (  # type: ignore[import-untyped] # noqa: PLC0415
        require_training_ready,
    )

    snapshot = _runtime_snapshot(runtime_root.resolve())
    run_id = snapshot["run_id"]
    _require(isinstance(run_id, str), "CURRENT_RUN_INVALID")
    safe_run_id = cast(str, run_id)
    run_root = runtime_root.resolve() / safe_run_id
    report_path = run_root / "training-readiness.json"
    sealed = require_training_ready(
        run_root / "sealed-acquisition",
        report_path=report_path,
    )
    return {
        "action": "readiness",
        "run_id": safe_run_id,
        "asset_count": len(sealed.split.assets),
        "ready_for_training": True,
        "report_path_name": report_path.name,
        "runpod_api_calls": 0,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def _cloud_training_phase(cloud_runtime: Path, run_id: str) -> str:
    if not cloud_runtime.exists():
        return "cloud_inventory"
    if not cloud_runtime.is_dir() or cloud_runtime.is_symlink():
        raise EndToEndError(
            "CLOUD_RUNTIME_STATE_INVALID",
            artifact="cloud-runtime",
        )
    operator_path = cloud_runtime / "_operator" / "phase3f-current.json"
    if operator_path.exists():
        operator = _integrity_json(
            operator_path,
            artifact="phase3f-current.json",
            missing_code="CLOUD_OPERATOR_STATE_MISSING",
            invalid_code="CLOUD_OPERATOR_STATE_INVALID",
            max_bytes=1024 * 1024,
        )
        operator_run_id = operator.get("run_id")
        operator_stage = operator.get("stage")
        if (
            not isinstance(operator_run_id, str)
            or not _RUN_ID.fullmatch(operator_run_id)
            or operator_stage not in {"preflight", "running", "failed", "terminated"}
        ):
            raise EndToEndError(
                "CLOUD_OPERATOR_STATE_INVALID",
                artifact="phase3f-current.json",
                field="run_id_or_stage",
            )
        if operator_stage in {"preflight", "running"}:
            if operator_run_id != run_id:
                raise EndToEndError(
                    "ACTIVE_CLOUD_RUN_CONFLICT",
                    artifact="phase3f-current.json",
                    field="run_id",
                )
            return "training_running"

    receipts_root = cloud_runtime / "_receipts"
    if not receipts_root.exists():
        return "cloud_inventory"
    if not receipts_root.is_dir() or receipts_root.is_symlink():
        raise EndToEndError(
            "CLOUD_TRAINING_RECEIPT_INVALID",
            artifact="_receipts",
        )
    try:
        receipt_paths = tuple(sorted(receipts_root.glob("*.json")))
    except OSError as exc:
        raise EndToEndError(
            "CLOUD_TRAINING_RECEIPT_INVALID",
            artifact="_receipts",
        ) from exc
    for receipt_path in receipt_paths:
        receipt = _integrity_json(
            receipt_path,
            artifact="training-receipt.json",
            missing_code="CLOUD_TRAINING_RECEIPT_INVALID",
            invalid_code="CLOUD_TRAINING_RECEIPT_INVALID",
            max_bytes=1024 * 1024,
        )
        if receipt.get("run_id") != run_id:
            continue
        if (
            receipt.get("schema") != "atlaslens-phase3f-local-supervisor-receipt-v1"
            or not isinstance(receipt.get("outcome"), str)
            or receipt.get("pod_termination_verified") is not True
            or receipt.get("secret_values_included") is not False
        ):
            raise EndToEndError(
                "CLOUD_TRAINING_RECEIPT_INVALID",
                artifact="training-receipt.json",
                field="completion_evidence",
            )
        return "training_completed"
    return "cloud_inventory"


def resume_plan(
    repository: Path,
    runtime_root: Path,
    cloud_runtime: Path,
) -> dict[str, object]:
    source = repository.resolve() / "services" / "api" / "src"
    _require(source.is_dir() and not source.is_symlink(), "PHASE3F_SOURCE_ROOT_INVALID")
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.local_first import (  # type: ignore[import-untyped] # noqa: PLC0415
        verify_sealed_acquisition,
    )
    from atlaslens_api.phase3f.training import (  # noqa: PLC0415
        training_readiness_document,
    )

    resolved_runtime = runtime_root.resolve()
    current = _integrity_json(
        resolved_runtime / "current.json",
        artifact="current.json",
        missing_code="PHASE3F_RESUME_STATE_MISSING",
        invalid_code="CURRENT_RUN_INVALID",
        max_bytes=4096,
    )
    run_id = current.get("run_id")
    if (
        current.get("schema") != CURRENT_SCHEMA
        or not isinstance(run_id, str)
        or not _RUN_ID.fullmatch(run_id)
    ):
        raise EndToEndError(
            "CURRENT_RUN_INVALID",
            artifact="current.json",
            field="schema_or_run_id",
        )
    run_root = resolved_runtime / run_id
    state = _integrity_json(
        run_root / "state.json",
        artifact="state.json",
        missing_code="LOCAL_STATE_MISSING",
        invalid_code="LOCAL_STATE_INVALID",
        max_bytes=1024 * 1024,
    )
    stage = state.get("stage")
    if (
        state.get("schema") != "atlaslens-phase3f-local-first-state-v1"
        or state.get("run_id") != run_id
        or not isinstance(stage, str)
    ):
        raise EndToEndError(
            "LOCAL_STATE_INVALID",
            artifact="state.json",
            field="schema_run_id_or_stage",
        )

    sealed_root = run_root / "sealed-acquisition"
    if not sealed_root.exists():
        work_root = run_root / "acquisition-work"
        metadata_checkpoint = work_root / "metadata-pages.json"
        if stage == "ACQUISITION_SEALED" or not work_root.is_dir():
            raise EndToEndError(
                "SEALED_BUNDLE_MISSING",
                artifact="sealed-acquisition",
                field="directory",
                expected=True,
                actual=False,
            )
        if work_root.is_symlink() or not metadata_checkpoint.is_file():
            raise EndToEndError(
                "ACQUISITION_WORK_MISSING",
                artifact="acquisition-work",
                field="metadata-pages.json",
            )
        _runtime_snapshot(resolved_runtime)
        return {
            "schema": "atlaslens-phase3f-resume-plan-v1",
            "action": "resume-plan",
            "run_id": run_id,
            "state_stage": stage,
            "next_phase": "local_acquisition",
            "acquisition_required": True,
            "mapillary_required": True,
            "dataset_write_count": 0,
            "cloud_readiness_transition_count": 0,
            "runpod_api_calls": 0,
            "cloud_mutations": 0,
            "secrets_included": False,
        }
    if not sealed_root.is_dir() or sealed_root.is_symlink():
        raise EndToEndError(
            "SEALED_BUNDLE_INVALID",
            artifact="sealed-acquisition",
            field="directory",
        )

    try:
        sealed = verify_sealed_acquisition(sealed_root)
    except Exception as exc:
        raise _sealed_error(exc) from exc
    if sealed.run_id != run_id:
        raise EndToEndError(
            "SEALED_RUN_ID_MISMATCH",
            artifact="manifest.json",
            field="run_id",
        )
    asset_count = len(sealed.split.assets)
    if asset_count != 830:
        raise EndToEndError(
            "PHASE3F_DATASET_ASSET_COUNT_MISMATCH",
            artifact="sealed-assets.json",
            field="asset_count",
            expected=830,
            actual=asset_count,
        )
    if state.get("asset_count") != asset_count:
        raise EndToEndError(
            "LOCAL_STATE_ASSET_COUNT_MISMATCH",
            artifact="state.json",
            field="asset_count",
            expected=asset_count,
            actual=cast(int | None, state.get("asset_count")),
        )
    if state.get("model_loaded") is not False:
        raise EndToEndError(
            "LOCAL_STATE_MODEL_LOADED_INVALID",
            artifact="state.json",
            field="model_loaded",
            expected=False,
            actual=cast(bool | None, state.get("model_loaded")),
        )

    report_path = run_root / "training-readiness.json"
    report = _integrity_json(
        report_path,
        artifact="training-readiness.json",
        missing_code="TRAINING_READINESS_REPORT_MISSING",
        invalid_code="TRAINING_READINESS_REPORT_INVALID",
        max_bytes=1024 * 1024,
    )
    actual_report_sha256 = _sha256_path(
        report_path,
        artifact="training-readiness.json",
        max_bytes=1024 * 1024,
    )
    try:
        expected_report = training_readiness_document(sealed)
    except Exception as exc:
        raw_code = getattr(exc, "code", None)
        code = raw_code if isinstance(raw_code, str) else "TRAINING_READINESS_RECOMPUTE_FAILED"
        raise EndToEndError(
            code,
            artifact="sealed-acquisition",
            field="training_readiness",
        ) from exc
    expected_report_bytes = _canonical_bytes(expected_report)
    expected_report_sha256 = hashlib.sha256(expected_report_bytes).hexdigest()
    if report_path.read_bytes() != expected_report_bytes:
        raise EndToEndError(
            "TRAINING_READINESS_HASH_MISMATCH",
            artifact="training-readiness.json",
            field="sha256",
            expected=expected_report_sha256,
            actual=actual_report_sha256,
        )
    if (
        report.get("ready") is not True
        or report.get("outcome") != "READY_FOR_TRAINING"
        or report.get("asset_count") != 830
        or report.get("gpu_started") is not False
        or report.get("cloud_mutations") != 0
        or report.get("secrets_included") is not False
        or not isinstance(report.get("checks"), dict)
        or not all(cast(dict[str, object], report["checks"]).values())
    ):
        raise EndToEndError(
            "TRAINING_READINESS_NOT_READY",
            artifact="training-readiness.json",
            field="ready_or_safety_fields",
        )

    next_phase = _cloud_training_phase(cloud_runtime.resolve(), run_id)
    sealed_assets_path = sealed_root / "sealed-assets.json"
    state_readiness_sha256 = state.get("readiness_report_sha256")
    return {
        "schema": "atlaslens-phase3f-resume-plan-v1",
        "action": "resume-plan",
        "run_id": run_id,
        "state_stage": stage,
        "next_phase": next_phase,
        "acquisition_required": False,
        "mapillary_required": False,
        "dataset_write_count": 0,
        "cloud_readiness_transition_count": 1 if next_phase == "cloud_inventory" else 0,
        "asset_count": asset_count,
        "readiness_sha256": actual_report_sha256,
        "sealed_assets_sha256": hashlib.sha256(sealed_assets_path.read_bytes()).hexdigest(),
        "readiness_recomputed_exact": True,
        "state_readiness_hash_matches": state_readiness_sha256 == actual_report_sha256,
        "model_loaded": False,
        "gpu_started": False,
        "runpod_api_calls": 0,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def status(repository: Path, local_runtime: Path, cloud_runtime: Path) -> dict[str, object]:
    source = repository.resolve() / "services" / "api" / "src"
    _require(source.is_dir(), "PHASE3F_SOURCE_ROOT_INVALID")
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.local_first import (  # noqa: PLC0415
        status as local_status,
    )

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
    parser.add_argument(
        "action",
        choices=("preflight", "readiness", "resume-plan", "status"),
    )
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
        elif args.action == "resume-plan":
            _require(args.cloud_runtime_root is not None, "CLOUD_RUNTIME_ROOT_MISSING")
            result = resume_plan(
                args.repository_root,
                args.runtime_root,
                args.cloud_runtime_root,
            )
        else:
            _require(args.cloud_runtime_root is not None, "CLOUD_RUNTIME_ROOT_MISSING")
            result = status(args.repository_root, args.runtime_root, args.cloud_runtime_root)
    except Exception as exc:  # sanitized operator boundary
        code = getattr(exc, "code", None)
        if isinstance(exc, EndToEndError) and len(exc.public_document()) > 1:
            print(json.dumps(exc.public_document(), separators=(",", ":"), sort_keys=True))
        else:
            print(code if isinstance(code, str) else "PHASE3F_END_TO_END_FAILED")
        return 1
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
