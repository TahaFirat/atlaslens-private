"""Offline RunPod entry point for sealed-corpus MegaLoc fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import traceback
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
    parser.add_argument("--recovery-root", type=Path)
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
                recovery_root=args.recovery_root,
                seed=args.seed,
                max_epochs=args.max_epochs,
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
