#!/usr/bin/env python3
"""Evaluate a bounded sequence of precomputed generic Phase 6C configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlaslens_api.phase6c.improvement import (
    ControlledImprovementEvaluator,
    ImprovementCandidate,
    ImprovementPolicy,
)
from pydantic import TypeAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-wall-clock-seconds", type=int, default=14_400)
    args = parser.parse_args()
    try:
        payload = json.loads(args.iterations.read_text(encoding="utf-8"))
        candidates = TypeAdapter(tuple[ImprovementCandidate, ...]).validate_python(payload)
        policy = ImprovementPolicy(
            max_iterations=args.max_iterations,
            max_wall_clock_seconds=args.max_wall_clock_seconds,
        )
        decisions = ControlledImprovementEvaluator(policy).evaluate_sequence(candidates)
        output = {
            "schema_version": "atlaslens-phase6c-improvement-report-v1",
            "policy": policy.model_dump(mode="json"),
            "decisions": [item.model_dump(mode="json") for item in decisions],
            "holdout_results_influenced_decisions": False,
            "status": "accepted_configuration_available"
            if any(item.accepted for item in decisions)
            else "no_valid_improvement",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        print(json.dumps({"status": "error", "reason_code": "improvement_input_invalid"}))
        return 2
    print(json.dumps({"status": output["status"], "iterations": len(decisions)}))
    return 0 if any(item.accepted for item in decisions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
