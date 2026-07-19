#!/usr/bin/env python3
"""Audit a reference manifest against private evaluation images without mutating it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlaslens_api.evaluation.leakage import (
    LeakageAuditError,
    LeakageAuditPolicy,
    ReferenceLeakageAuditor,
    load_descriptor_artifact,
    write_filtered_phase6c_reference_input,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        default=Path(".local/evaluation/turkey_holdout.json"),
    )
    parser.add_argument("--holdout-id", default="user-holdout-001")
    reference_source = parser.add_mutually_exclusive_group(required=True)
    reference_source.add_argument(
        "--reference-manifest",
        type=Path,
        help="Licensed extended retrieval CSV (existing interface).",
    )
    reference_source.add_argument(
        "--reference-input",
        type=Path,
        help="Phase 6C atlaslens-megaloc-reference-input-v1 JSON.",
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--descriptor-npz", type=Path)
    parser.add_argument(
        "--write-filtered-reference-input",
        type=Path,
        help=(
            "Atomically write the Phase 6C input without excluded references; "
            "the source is never modified."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-references", type=int, default=10_000)
    parser.add_argument("--max-geometric-checks", type=int, default=1_000)
    args = parser.parse_args()
    if (
        args.write_filtered_reference_input is not None
        and args.reference_input is None
    ):
        parser.error("--write-filtered-reference-input requires --reference-input")

    try:
        policy = LeakageAuditPolicy(
            max_references=args.max_references,
            max_geometric_checks=args.max_geometric_checks,
        )
        descriptors = (
            load_descriptor_artifact(args.descriptor_npz)
            if args.descriptor_npz is not None
            else None
        )
        auditor = ReferenceLeakageAuditor(policy)
        if args.reference_input is not None:
            report = auditor.audit_phase6c_reference_input(
                holdout_manifest=args.holdout_manifest,
                holdout_id=args.holdout_id,
                reference_input=args.reference_input,
                reference_root=args.reference_root,
                descriptor_artifact=descriptors,
            )
        else:
            report = auditor.audit_manifest(
                holdout_manifest=args.holdout_manifest,
                holdout_id=args.holdout_id,
                reference_manifest=args.reference_manifest,
                reference_root=args.reference_root,
                descriptor_artifact=descriptors,
            )
        input_path = args.reference_input or args.reference_manifest
        input_destination = input_path.expanduser().resolve()
        report_destination = args.output.expanduser().resolve()
        if input_destination == report_destination:
            raise LeakageAuditError("audit_output_conflicts_with_reference_input")
        if (
            args.write_filtered_reference_input is not None
            and args.write_filtered_reference_input.expanduser().resolve()
            == report_destination
        ):
            raise LeakageAuditError("filtered_reference_output_conflicts_with_report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        filtered_reference_count: int | None = None
        if args.write_filtered_reference_input is not None:
            filtered_reference_count = write_filtered_phase6c_reference_input(
                source_path=args.reference_input,
                destination_path=args.write_filtered_reference_input,
                excluded_reference_keys=tuple(
                    item.reference_key for item in report.exclusions
                ),
                max_manifest_bytes=policy.max_manifest_bytes,
            )
    except (LeakageAuditError, OSError, RuntimeError, ValueError):
        print(
            json.dumps({"status": "error", "code": "leakage_audit_failed"}),
        )
        return 2
    summary: dict[str, object] = {
        "status": report.status,
        "checked_reference_count": report.checked_reference_count,
        "excluded_reference_count": report.excluded_reference_count,
        "descriptor_check": report.checks["descriptor_similarity"],
    }
    if filtered_reference_count is not None:
        summary.update(
            {
                "filtered_output_written": True,
                "filtered_reference_count": filtered_reference_count,
            }
        )
    print(json.dumps(summary, sort_keys=True))
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
