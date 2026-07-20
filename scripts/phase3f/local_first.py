"""CLI entry point for Phase 3F local acquisition and offline compute."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-local-first")
    parser.add_argument(
        "action",
        choices=(
            "acquire-only",
            "diagnose-acquisition",
            "status",
            "resume",
            "compute-only",
            "cleanup",
        ),
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--vendor-root", type=Path)
    parser.add_argument("--max-wall-minutes", type=int, default=720)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-run-id")
    return parser


def _safe_error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_]+", code):
        return code.upper()
    return "PHASE3F_LOCAL_FIRST_FAILED"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository_root.resolve()
    source = repository / "services" / "api" / "src"
    if not source.is_dir() or source.is_symlink():
        print("PHASE3F_SOURCE_ROOT_INVALID")
        return 2
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.local_first import (  # noqa: PLC0415
        AcquisitionConfig,
        ComputeConfig,
        DiagnoseAcquisitionConfig,
        cleanup,
        diagnose_acquisition,
        run_acquisition,
        run_compute,
        status,
    )

    try:
        if args.action in {"acquire-only", "resume"}:
            result = run_acquisition(
                AcquisitionConfig(
                    runtime_root=args.runtime_root,
                    aoi_config_path=(
                        repository / "config" / "phase3f" / "city-coverage-aoi-v1.json"
                    ),
                    source_policy_path=(
                        repository
                        / "config"
                        / "phase3b3"
                        / "mapillary-source-policy-v1.json"
                    ),
                    resume=args.action == "resume",
                    max_wall_seconds=args.max_wall_minutes * 60,
                )
            )
        elif args.action == "diagnose-acquisition":
            result = diagnose_acquisition(
                DiagnoseAcquisitionConfig(
                    runtime_root=args.runtime_root,
                    aoi_config_path=(
                        repository / "config" / "phase3f" / "city-coverage-aoi-v1.json"
                    ),
                )
            )
        elif args.action == "status":
            result = status(args.runtime_root)
        elif args.action == "compute-only":
            if args.model is None or args.vendor_root is None:
                print("LOCAL_COMPUTE_ARTIFACT_PATH_MISSING")
                return 2
            result = run_compute(
                ComputeConfig(
                    runtime_root=args.runtime_root,
                    model_path=args.model,
                    vendor_root=args.vendor_root,
                    max_wall_seconds=args.max_wall_minutes * 60,
                )
            )
        else:
            result = cleanup(
                args.runtime_root,
                execute=args.execute,
                confirm_run_id=args.confirm_run_id,
            )
    except Exception as exc:  # sanitized boundary for the operator console
        print(_safe_error_code(exc))
        return 1
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
