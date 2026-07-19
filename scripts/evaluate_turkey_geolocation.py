#!/usr/bin/env python3
"""Run geolocation predictions behind a process boundary, then score private truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlaslens_api.evaluation.holdout import HoldoutManifestError
from atlaslens_api.evaluation.isolation import (
    LeakageSafeEvaluationRunner,
    PredictionBoundaryError,
    ProcessPredictionClient,
    prediction_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".local/evaluation/turkey_holdout.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--max-samples", type=int, default=1_000)
    parser.add_argument("--minimum-claim-samples", type=int, default=100)
    parser.add_argument(
        "--prediction-command",
        nargs=argparse.REMAINDER,
        required=True,
        help="argv for a worker that reads one isolated request from stdin",
    )
    args = parser.parse_args()

    try:
        command = prediction_command(args.prediction_command)
        client = ProcessPredictionClient(
            command=command,
            timeout_seconds=args.timeout_seconds,
        )
        report = LeakageSafeEvaluationRunner(
            max_samples=args.max_samples,
            minimum_claim_samples=args.minimum_claim_samples,
        ).run(args.manifest, client)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (HoldoutManifestError, PredictionBoundaryError, OSError, ValueError):
        print(
            json.dumps(
                {"status": "error", "code": "isolated_evaluation_failed"},
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "sample_count": report.sample_count,
                "accuracy_claim_allowed": report.accuracy_claim_allowed,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
