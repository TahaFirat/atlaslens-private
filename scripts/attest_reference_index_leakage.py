#!/usr/bin/env python3
"""Bind a passed leakage audit to a verified Phase 6C reference index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from atlaslens_api.evaluation.leakage import LeakageAuditReport
from atlaslens_api.phase6c.reference_index import (
    ReferenceIndexBuildError,
    ReferenceIndexLeakageAttestation,
    open_reference_index,
    write_reference_leakage_attestation,
)
from pydantic import ValidationError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(path: Path) -> LeakageAuditReport:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError("leakage_report_unsafe")
    resolved = requested.resolve(strict=True)
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or not 1 <= resolved.stat().st_size <= 20 * 1024 * 1024
    ):
        raise ValueError("leakage_report_unsafe")
    return LeakageAuditReport.model_validate_json(resolved.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--prior-report",
        type=Path,
        help="Optional pre-filter audit; only its exclusion count is attested.",
    )
    args = parser.parse_args()

    try:
        opened = open_reference_index(args.index)
        diagnostics = opened.diagnostics
        if opened.index is None or diagnostics.status != "ready":
            raise ReferenceIndexBuildError("reference_index_not_ready")
        report = _load_report(args.report)
        if (
            report.status != "passed"
            or report.checked_reference_count != diagnostics.count
            or report.descriptor_checked_count != diagnostics.count
            or report.excluded_reference_count != 0
            or report.exclusions
            or report.descriptor_version != diagnostics.descriptor_version
            or report.checks.get("descriptor_similarity") != "completed"
        ):
            raise ValueError("leakage_report_not_complete_or_clean")

        prior_excluded: int | None = None
        if args.prior_report is not None:
            prior = _load_report(args.prior_report)
            if (
                prior.holdout_id != report.holdout_id
                or prior.holdout_sha256 != report.holdout_sha256
                or prior.holdout_perceptual_hash != report.holdout_perceptual_hash
                or prior.excluded_reference_count != len(prior.exclusions)
            ):
                raise ValueError("prior_leakage_report_mismatch")
            prior_excluded = prior.excluded_reference_count

        report_path = args.report.expanduser().resolve(strict=True)
        attestation = ReferenceIndexLeakageAttestation(
            audit_fingerprint=report.audit_fingerprint,
            source_report_sha256=_sha256(report_path),
            index_version=diagnostics.index_version or "unavailable",
            descriptor_version=diagnostics.descriptor_version or "unavailable",
            checked_reference_count=report.checked_reference_count,
            descriptor_checked_count=report.descriptor_checked_count,
            pre_index_excluded_reference_count=prior_excluded,
        )
        write_reference_leakage_attestation(args.index, attestation)
    except (
        OSError,
        UnicodeError,
        ValueError,
        ValidationError,
        ReferenceIndexBuildError,
    ):
        print(json.dumps({"status": "error", "code": "leakage_attestation_failed"}))
        return 2

    print(
        json.dumps(
            {
                "status": "passed",
                "checked_reference_count": attestation.checked_reference_count,
                "descriptor_checked_count": attestation.descriptor_checked_count,
                "excluded_reference_count": attestation.excluded_reference_count,
                "pre_index_excluded_reference_count": (
                    attestation.pre_index_excluded_reference_count
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
