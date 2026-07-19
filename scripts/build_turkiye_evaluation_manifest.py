#!/usr/bin/env python3
"""Build a leakage-safe Türkiye development/validation evaluation CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atlaslens_api.evaluation.turkiye_suite import (
    EvaluationSuiteBuildError,
    EvaluationSuiteBuildPolicy,
    TurkiyeEvaluationSuiteBuilder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-reference-input", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--retrieval-reference-input", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument(
        "--allowed-license",
        action="append",
        required=True,
        help="Exact reviewed license string; repeat to allow more than one.",
    )
    parser.add_argument("--minimum-records", type=int, default=100)
    parser.add_argument("--development-fraction", type=float, default=0.7)
    parser.add_argument("--dhash-hamming-threshold", type=int, default=4)
    parser.add_argument("--max-candidate-records", type=int, default=10_000)
    parser.add_argument("--max-retrieval-records", type=int, default=10_000)
    parser.add_argument("--max-manifest-mb", type=int, default=16)
    parser.add_argument("--max-image-mb", type=int, default=50)
    parser.add_argument("--max-decoded-pixels", type=int, default=40_000_000)
    parser.add_argument("--split-seed", default="phase6c-turkiye-suite-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = EvaluationSuiteBuildPolicy(
            allowed_licenses=frozenset(args.allowed_license),
            minimum_records=args.minimum_records,
            development_fraction=args.development_fraction,
            dhash_hamming_threshold=args.dhash_hamming_threshold,
            max_candidate_records=args.max_candidate_records,
            max_retrieval_records=args.max_retrieval_records,
            max_manifest_bytes=args.max_manifest_mb * 1024 * 1024,
            max_image_bytes=args.max_image_mb * 1024 * 1024,
            max_decoded_pixels=args.max_decoded_pixels,
            split_seed=args.split_seed,
        )
        builder = TurkiyeEvaluationSuiteBuilder(policy)
        result = builder.build(
            candidate_reference_input=args.candidate_reference_input,
            candidate_root=args.candidate_root,
            retrieval_reference_input=args.retrieval_reference_input,
            retrieval_root=args.retrieval_root,
        )
        builder.write(
            result,
            manifest_path=args.output_manifest,
            receipt_path=args.output_receipt,
            protected_inputs=(
                args.candidate_reference_input,
                args.retrieval_reference_input,
            ),
        )
    except (EvaluationSuiteBuildError, OSError, ValueError) as exc:
        reason = str(exc) if isinstance(exc, EvaluationSuiteBuildError) else "build_failed"
        print(
            json.dumps({"event": "evaluation_manifest_build_failed", "reason_code": reason}),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "event": "evaluation_manifest_built",
                "status": result.receipt.status,
                "accepted_count": result.receipt.accepted_count,
                "excluded_count": result.receipt.excluded_count,
                "split_counts": result.receipt.split_counts,
            },
            sort_keys=True,
        )
    )
    return 0 if result.receipt.status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
