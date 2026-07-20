"""Offline RunPod entry point for sealed-corpus MegaLoc fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-training-job")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser


def _safe_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_]+", code):
        return code.upper()
    return "PHASE3F_TRAINING_JOB_FAILED"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ.pop("MAPILLARY_ACCESS_TOKEN", None)
    source = args.repository_root.resolve() / "services" / "api" / "src"
    if not source.is_dir() or source.is_symlink():
        print("PHASE3F_SOURCE_ROOT_INVALID")
        return 2
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.training import (  # noqa: PLC0415
        TrainingJobConfig,
        run_training_job,
    )

    try:
        result = run_training_job(
            TrainingJobConfig(
                run_id=args.run_id,
                sealed_root=args.sealed_root,
                model_path=args.model,
                vendor_root=args.vendor_root,
                work_root=args.work_root,
                output_root=args.output_root,
                deadline_epoch=args.deadline_epoch,
                seed=args.seed,
                max_epochs=args.max_epochs,
            )
        )
    except Exception as exc:  # sanitized process boundary
        print(_safe_code(exc))
        return 1
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
