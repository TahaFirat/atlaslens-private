from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from atlaslens_api.dataset_acquisition.commons import (
    CommonsAcquirer,
    CommonsApiClient,
    DatasetAcquisitionError,
    read_review,
    write_review,
)
from atlaslens_api.dataset_acquisition.models import SamplingCell
from atlaslens_api.dataset_acquisition.validation import (
    EvaluationExclusions,
    LicensedDatasetValidator,
)
from atlaslens_api.dataset_qa import (
    DatasetQAError,
    DatasetQAReportCatalog,
    DatasetQAReportWriter,
    DatasetQAScanner,
    safe_output_directory,
)


def _licenses(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-license", action="append", required=True)


def build_dataset_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas dataset")
    commands = parser.add_subparsers(dest="command", required=True)

    discover = commands.add_parser("discover")
    discover.add_argument("source", choices=("commons",))
    discover.add_argument("--sampling-cells", type=Path, required=True)
    discover.add_argument("--target", type=int, default=10_000)
    discover.add_argument("--output", type=Path, required=True)
    discover.add_argument("--user-agent", required=True)
    _licenses(discover)

    review = commands.add_parser("review")
    review.add_argument("source", choices=("commons",), nargs="?")
    review.add_argument("--titles", type=Path)
    review.add_argument("--output", type=Path)
    review.add_argument("--user-agent")
    review.add_argument("--country")
    review.add_argument("--region")
    review.add_argument(
        "--continent",
        choices=("Africa", "Asia", "Europe", "North America", "South America", "Oceania"),
    )
    review.add_argument("--geographic-cell")
    review.add_argument("--allow-license", action="append")
    review.add_argument("--qa-report", type=Path)
    review.add_argument("--severity", choices=("error", "warning", "info"))

    acquire = commands.add_parser("acquire")
    acquire.add_argument("source", choices=("commons", "kartaview"))
    acquire.add_argument("--review", type=Path)
    acquire.add_argument("--output-root", type=Path)
    acquire.add_argument("--manifest", type=Path)
    acquire.add_argument("--user-agent")
    acquire.add_argument("--evaluation-manifest", type=Path)
    acquire.add_argument("--evaluation-root", type=Path)
    acquire.add_argument("--expected-evaluation-fingerprint")
    acquire.add_argument("--expected-evaluation-count", type=int)
    acquire.add_argument("--terms-approval-file", type=Path)
    _licenses(acquire)

    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--asset-root", type=Path, required=True)
    validate.add_argument("--evaluation-manifest", type=Path, required=True)
    validate.add_argument("--evaluation-root", type=Path, required=True)
    validate.add_argument("--expected-evaluation-fingerprint", required=True)
    validate.add_argument("--expected-evaluation-count", type=int, required=True)
    validate.add_argument("--minimum-count", type=int, default=10_000)
    validate.add_argument("--required-continents", type=int, default=6)
    validate.add_argument("--minimum-countries", type=int, default=30)
    _licenses(validate)

    qa = commands.add_parser("qa")
    qa.add_argument("--images", type=Path, required=True)
    qa.add_argument("--masks", type=Path)
    qa.add_argument("--manifest", type=Path)
    qa.add_argument("--asset-root", type=Path)
    qa.add_argument("--output", type=Path, required=True)
    qa.add_argument("--report-id")
    qa.add_argument("--class-id", type=int, action="append", default=[])
    qa.add_argument("--contact-sheet", action="store_true")
    qa.add_argument("--apply", action="store_true")

    qa_report = commands.add_parser("qa-report")
    qa_report.add_argument("--output", type=Path, required=True)
    return parser


def _explicit_licenses(args: argparse.Namespace) -> frozenset[str]:
    values = frozenset(value.strip() for value in (args.allow_license or []) if value.strip())
    if not values:
        raise DatasetAcquisitionError("license_allowlist_required")
    return values


def _review(args: argparse.Namespace) -> dict[str, object]:
    if args.qa_report is not None:
        if args.source is not None:
            raise DatasetQAError("qa_review_arguments_conflict")
        report_path = args.qa_report.expanduser()
        report = DatasetQAReportCatalog(report_path.parent).get(report_path.name)
        selected = [
            issue
            for issue in report.issues
            if args.severity is None or issue.severity == args.severity
        ]
        return {
            "status": "review",
            "report_id": report.summary.report_id,
            "issues": [issue.model_dump(mode="json") for issue in selected],
            "issue_count": len(selected),
            "read_only": True,
        }
    if (
        args.source != "commons"
        or args.titles is None
        or args.output is None
        or args.user_agent is None
        or args.country is None
        or args.continent is None
        or args.geographic_cell is None
    ):
        raise DatasetAcquisitionError("commons_review_arguments_required")
    try:
        titles = [
            line.strip()
            for line in args.titles.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as exc:
        raise DatasetAcquisitionError("titles_file_unreadable") from exc
    if not titles or len(titles) != len(set(titles)):
        raise DatasetAcquisitionError("titles_file_invalid")
    client = CommonsApiClient(user_agent=args.user_agent, allowed_licenses=_explicit_licenses(args))
    records = [
        client.review_title(
            title,
            fallback_country=args.country,
            fallback_region=args.region,
            continent=args.continent,
            geographic_cell=args.geographic_cell,
        )
        for title in titles
    ]
    write_review(args.output, records)
    return {
        "status": "review_required",
        "source": "Wikimedia Commons",
        "records": len(records),
        "approved": 0,
    }


def _sampling_cells(path: Path) -> list[SamplingCell]:
    headers = tuple(SamplingCell.model_fields)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != headers:
                raise DatasetAcquisitionError("sampling_cell_headers_invalid")
            cells = [SamplingCell.model_validate(row) for row in reader]
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        if isinstance(exc, DatasetAcquisitionError):
            raise
        raise DatasetAcquisitionError("sampling_cells_invalid") from exc
    if not cells or len({cell.cell_id for cell in cells}) != len(cells):
        raise DatasetAcquisitionError("sampling_cells_invalid")
    return sorted(cells, key=lambda item: item.cell_id)


def _discover(args: argparse.Namespace) -> dict[str, object]:
    if args.target < 1:
        raise DatasetAcquisitionError("discovery_target_invalid")
    client = CommonsApiClient(user_agent=args.user_agent, allowed_licenses=_explicit_licenses(args))
    existing = read_review(args.output) if args.output.exists() else []
    by_page = {record.page_id: record for record in existing}
    for cell in _sampling_cells(args.sampling_cells):
        if len(by_page) >= args.target:
            break
        for record in client.discover_cell(cell):
            by_page.setdefault(record.page_id, record)
            write_review(args.output, sorted(by_page.values(), key=lambda item: item.page_id))
            if len(by_page) >= args.target:
                break
    if len(by_page) < args.target:
        raise DatasetAcquisitionError("discovery_target_not_met")
    return {
        "status": "review_required",
        "source": "Wikimedia Commons",
        "records": len(by_page),
        "approved": sum(record.approved for record in by_page.values()),
    }


def _exclusions(args: argparse.Namespace) -> EvaluationExclusions:
    if (
        args.evaluation_manifest is None
        or args.evaluation_root is None
        or args.expected_evaluation_fingerprint is None
        or args.expected_evaluation_count is None
    ):
        raise DatasetAcquisitionError("evaluation_exclusions_required")
    return EvaluationExclusions.from_manifest(
        args.evaluation_manifest,
        args.evaluation_root,
        expected_fingerprint=args.expected_evaluation_fingerprint,
        expected_count=args.expected_evaluation_count,
    )


def _kartaview_gate(args: argparse.Namespace) -> None:
    if args.terms_approval_file is None:
        raise DatasetAcquisitionError("kartaview_terms_approval_required")
    try:
        approval = json.loads(args.terms_approval_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise DatasetAcquisitionError("kartaview_terms_approval_invalid") from exc
    if approval != {
        "approved": True,
        "source": "KartaView",
        "terms_version": "2025-06-17",
    }:
        raise DatasetAcquisitionError("kartaview_terms_approval_invalid")
    raise DatasetAcquisitionError("kartaview_adapter_unavailable")


def _acquire(args: argparse.Namespace) -> dict[str, object]:
    if args.source == "kartaview":
        _kartaview_gate(args)
    required = (args.review, args.output_root, args.manifest, args.user_agent)
    if any(value is None for value in required):
        raise DatasetAcquisitionError("commons_acquisition_arguments_required")
    exclusions = _exclusions(args)
    client = CommonsApiClient(user_agent=args.user_agent, allowed_licenses=_explicit_licenses(args))
    receipt = CommonsAcquirer(client).acquire(
        args.review,
        args.output_root,
        args.manifest,
        excluded_hashes=exclusions.content_hashes,
        excluded_perceptual_hashes=exclusions.perceptual_hashes,
        excluded_capture_families=exclusions.capture_families,
    )
    return {
        "status": "acquired",
        "source": "Wikimedia Commons",
        "records": len(receipt.entries),
        "evaluation_fingerprint": exclusions.fingerprint,
    }


def _validate(args: argparse.Namespace) -> dict[str, object]:
    exclusions = _exclusions(args)
    report = LicensedDatasetValidator(
        allowed_licenses=_explicit_licenses(args),
        minimum_count=args.minimum_count,
        required_continents=args.required_continents,
        minimum_countries=args.minimum_countries,
    ).validate(args.manifest, args.asset_root, exclusions)
    return report.model_dump(mode="json")


def _qa(args: argparse.Namespace) -> dict[str, object]:
    if args.apply:
        raise DatasetQAError("qa_apply_not_supported")
    report_id = args.report_id or args.output.name
    if report_id != args.output.name:
        raise DatasetQAError("qa_report_id_must_match_output_directory")
    roots = tuple(
        value for value in (args.images, args.masks, args.asset_root) if isinstance(value, Path)
    )
    destination = safe_output_directory(args.output, roots)
    scanner = DatasetQAScanner()
    run = scanner.scan(
        images=args.images,
        masks=args.masks,
        manifest=args.manifest,
        asset_root=args.asset_root,
        report_id=report_id,
        allowed_class_ids=(frozenset({0, *args.class_id}) if args.class_id else frozenset()),
        contact_sheet=args.contact_sheet,
    )
    DatasetQAReportWriter(scanner.policy).write(run, destination)
    return run.report.model_dump(mode="json")


def _qa_report(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser()
    report = DatasetQAReportCatalog(output.parent).get(output.name)
    return report.model_dump(mode="json")


def run_dataset_command(argv: Sequence[str]) -> dict[str, object]:
    args = build_dataset_parser().parse_args(argv)
    if args.command == "discover":
        return _discover(args)
    if args.command == "review":
        return _review(args)
    if args.command == "acquire":
        return _acquire(args)
    if args.command == "validate":
        return _validate(args)
    if args.command == "qa":
        return _qa(args)
    return _qa_report(args)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run_dataset_command(sys.argv[1:] if argv is None else argv)
    except (DatasetAcquisitionError, DatasetQAError) as exc:
        print(json.dumps({"status": "error", "code": str(exc)}), file=sys.stderr)
        return 2
    except (OSError, ValueError):
        print(
            json.dumps({"status": "error", "code": "dataset_operation_failed"}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
