#!/usr/bin/env python3
"""Build a real pinned-MegaLoc NPZ for the Phase 6C leakage audit."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import cast

from atlaslens_api.evaluation.descriptors import (
    LeakageDescriptorBuildError,
    LeakageDescriptorBuildPolicy,
    build_phase6c_leakage_descriptor_artifact,
)
from atlaslens_api.phase6b.scheduler import DeviceName
from pydantic import ValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        default=Path(".local/evaluation/turkey_holdout.json"),
    )
    parser.add_argument("--holdout-id", default="user-holdout-001")
    parser.add_argument("--reference-input", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-port", type=int, default=8794)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-references", type=int, default=10_000)
    parser.add_argument("--max-manifest-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--max-image-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--max-total-image-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--max-output-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--worker-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--total-timeout-seconds", type=float, default=3_600.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        policy = LeakageDescriptorBuildPolicy(
            max_references=args.max_references,
            max_manifest_bytes=args.max_manifest_bytes,
            max_image_bytes=args.max_image_bytes,
            max_total_image_bytes=args.max_total_image_bytes,
            max_output_bytes=args.max_output_bytes,
            worker_timeout_seconds=args.worker_timeout_seconds,
            total_timeout_seconds=args.total_timeout_seconds,
        )
        result = asyncio.run(
            build_phase6c_leakage_descriptor_artifact(
                holdout_manifest=args.holdout_manifest,
                holdout_id=args.holdout_id,
                reference_input=args.reference_input,
                reference_root=args.reference_root,
                output=args.output,
                worker_port=args.worker_port,
                device=cast(DeviceName, args.device),
                policy=policy,
            )
        )
    except ValidationError:
        print(
            json.dumps(
                {
                    "event": "leakage_descriptor_build_failed",
                    "reason_code": "build_bounds_invalid",
                },
                sort_keys=True,
            )
        )
        return 2
    except LeakageDescriptorBuildError as exc:
        print(
            json.dumps(
                {
                    "event": "leakage_descriptor_build_failed",
                    "reason_code": exc.code,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "event": "leakage_descriptors_built",
                "provider": result.provider,
                "version": result.version,
                "descriptor_dimension": result.descriptor_dimension,
                "input_image_count": result.input_image_count,
                "unique_content_count": result.unique_content_count,
                "artifact_size_bytes": result.artifact_size_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
