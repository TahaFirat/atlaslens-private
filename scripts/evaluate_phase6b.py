#!/usr/bin/env python3
"""Aggregate a user-provided Phase 6B manifest and real run records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlaslens_api.phase6b.evaluation import EvaluationError, Phase6BEvaluationHarness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/geolocation/manifest.json"),
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/geolocation/phase6b-report.json"),
    )
    parser.add_argument("--minimum-claim-samples", type=int, default=100)
    args = parser.parse_args()

    harness = Phase6BEvaluationHarness()
    try:
        manifest, fingerprint = harness.load_manifest(args.manifest)
        observations = harness.load_observations(args.results)
        report = harness.evaluate(
            manifest,
            observations,
            manifest_fingerprint=fingerprint,
            minimum_claim_samples=args.minimum_claim_samples,
        )
        harness.write_report(report, args.output)
    except EvaluationError as exc:
        print(json.dumps({"status": "error", "code": str(exc)}, sort_keys=True))
        return 2
    print(f"Phase 6B evaluation sample count: {report.sample_count}")
    print(
        json.dumps(
            {
                "status": "completed",
                "sample_count": report.sample_count,
                "improvement_claim_allowed": report.improvement_claim_allowed,
                "claim_note": report.claim_note,
                "profiles_complete": sum(item.complete for item in report.profiles),
                "profiles_total": len(report.profiles),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
