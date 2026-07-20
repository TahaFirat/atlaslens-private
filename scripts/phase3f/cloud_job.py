"""CLI entry point for the bounded Phase 3F cloud worker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-cloud-job")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository_root.resolve()
    source = repository / "services" / "api" / "src"
    if not source.is_dir() or source.is_symlink():
        print("PHASE3F_SOURCE_ROOT_INVALID")
        return 2
    sys.path.insert(0, str(source))
    from atlaslens_api.phase3f.cloud_job import (  # noqa: PLC0415
        CloudJobConfig,
        Phase3FCloudJobError,
        run_cloud_job,
    )
    from atlaslens_api.mapillary_demo.errors import MapillaryDemoError  # noqa: PLC0415

    try:
        result = run_cloud_job(
            CloudJobConfig(
                run_id=args.run_id,
                aoi_config_path=repository
                / "config"
                / "phase3f"
                / "city-coverage-aoi-v1.json",
                source_policy_path=repository
                / "config"
                / "phase3b3"
                / "mapillary-source-policy-v1.json",
                model_path=args.model,
                vendor_root=args.vendor_root,
                work_root=args.work_root,
                output_root=args.output_root,
                deadline_epoch=args.deadline_epoch,
                resume=args.resume,
            )
        )
    except Phase3FCloudJobError as exc:
        print(exc.code)
        return 1
    except MapillaryDemoError as exc:
        print(exc.code.upper())
        return 1
    print(
        json.dumps(
            {
                "outcome": result["outcome"],
                "run_id": result["run_id"],
                "finished": True,
                "secrets_included": False,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
