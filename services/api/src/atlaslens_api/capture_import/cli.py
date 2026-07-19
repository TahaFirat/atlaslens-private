from __future__ import annotations

import argparse
from pathlib import Path

from .errors import CaptureImportError
from .models import PrivacyDecision, RevocationUpdate
from .service import (
    CaptureImporter,
    build_capture_manifest,
    load_capture_plan,
    revoke_capture_asset,
    update_capture_privacy,
)
from .sources import parse_aware_timestamp

CAPTURE_COMMANDS = (
    "inspect-capture",
    "import-capture",
    "sync-gpx",
    "sample-route",
    "privacy-review",
    "build-capture-manifest",
)


def _add_capture_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)


def add_capture_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    inspect_parser = subparsers.add_parser(
        "inspect-capture",
        help="Inspect bounded first-party capture inputs without network access.",
    )
    _add_capture_paths(inspect_parser)

    import_parser = subparsers.add_parser(
        "import-capture",
        help="Atomically import sampled first-party media with privacy pending.",
    )
    _add_capture_paths(import_parser)
    import_parser.add_argument("--no-resume", action="store_true")
    import_parser.add_argument("--max-new-assets", type=int)
    import_parser.add_argument("--dry-run", action="store_true")

    sync_parser = subparsers.add_parser(
        "sync-gpx",
        help="Synchronize ordered media with bounded GPS interpolation.",
    )
    _add_capture_paths(sync_parser)

    sample_parser = subparsers.add_parser(
        "sample-route",
        help="Apply the approved distance sampling and stationary deduplication policy.",
    )
    _add_capture_paths(sample_parser)

    privacy_parser = subparsers.add_parser(
        "privacy-review",
        help="Record an explicit privacy review or revocation receipt.",
    )
    privacy_parser.add_argument("--work-root", type=Path, required=True)
    privacy_parser.add_argument("--asset-id", required=True)
    privacy_parser.add_argument(
        "--state",
        choices=("approved", "rejected", "needs_redaction"),
    )
    privacy_parser.add_argument("--reviewed-by")
    privacy_parser.add_argument("--reviewed-at", type=parse_aware_timestamp)
    privacy_parser.add_argument("--decision-receipt-sha256")
    privacy_parser.add_argument("--redaction-applied", action="store_true")
    privacy_parser.add_argument("--redaction-receipt-sha256")
    privacy_parser.add_argument("--redacted-input-root", type=Path)
    privacy_parser.add_argument("--redacted-copy", type=Path)
    privacy_parser.add_argument("--revoke-request-id")
    privacy_parser.add_argument("--revocation-receipt-sha256")
    privacy_parser.add_argument("--revoked-at", type=parse_aware_timestamp)

    manifest_parser = subparsers.add_parser(
        "build-capture-manifest",
        help="Build a Phase 3B1 manifest containing approved active assets only.",
    )
    manifest_parser.add_argument("--work-root", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, default=Path("manifest.json"))


def _importer(args: argparse.Namespace) -> CaptureImporter:
    plan = load_capture_plan(args.input_root, args.plan)
    return CaptureImporter(args.input_root, args.work_root, plan)


def _privacy_dispatch(args: argparse.Namespace) -> dict[str, object]:
    revocation_values = (
        args.revoke_request_id,
        args.revocation_receipt_sha256,
        args.revoked_at,
    )
    if any(value is not None for value in revocation_values):
        if (
            not all(value is not None for value in revocation_values)
            or args.state is not None
            or args.redacted_input_root is not None
            or args.redacted_copy is not None
        ):
            raise CaptureImportError("revocation_arguments_invalid")
        inventory = revoke_capture_asset(
            args.work_root,
            RevocationUpdate(
                asset_id=args.asset_id,
                request_id=args.revoke_request_id,
                receipt_sha256=args.revocation_receipt_sha256,
                revoked_at=args.revoked_at,
            ),
        )
        return {
            "status": "revoked",
            "asset_count": len(inventory.assets),
            "revoked_count": sum(
                asset.revocation_status == "REVOKED" for asset in inventory.assets
            ),
        }
    if (
        args.state is None
        or args.reviewed_by is None
        or args.reviewed_at is None
        or args.decision_receipt_sha256 is None
    ):
        raise CaptureImportError("privacy_arguments_invalid")
    inventory = update_capture_privacy(
        args.work_root,
        PrivacyDecision(
            asset_id=args.asset_id,
            state=args.state,
            reviewed_by=args.reviewed_by,
            reviewed_at=args.reviewed_at,
            decision_receipt_sha256=args.decision_receipt_sha256,
            redaction_applied=args.redaction_applied,
            redaction_receipt_sha256=args.redaction_receipt_sha256,
        ),
        redacted_input_root=args.redacted_input_root,
        redacted_copy=args.redacted_copy,
    )
    return {
        "status": "privacy_recorded",
        "asset_count": len(inventory.assets),
        "approved_count": sum(
            asset.privacy.state == "approved" for asset in inventory.assets
        ),
        "excluded_count": sum(
            asset.privacy.state != "approved"
            or asset.revocation_status != "ACTIVE"
            for asset in inventory.assets
        ),
    }


def dispatch_capture_command(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "inspect-capture":
        inspection = _importer(args).inspect()
        return {
            "status": inspection.status,
            "plan_sha256": inspection.plan_sha256,
            "input_mode": inspection.input_mode,
            "media_count": inspection.media_count,
            "track_point_count": inspection.track_point_count,
            "synchronized_count": inspection.synchronized_count,
            "ffmpeg_required": inspection.ffmpeg_required,
            "privacy_default": inspection.privacy_default,
            "accuracy_claim": inspection.accuracy_claim,
        }
    if args.command == "sync-gpx":
        synchronized = _importer(args).synchronize()
        return {
            "status": "synchronized",
            "plan_sha256": synchronized.plan_sha256,
            "track_point_count": synchronized.track_point_count,
            "synchronized_count": len(synchronized.frames),
        }
    if args.command == "sample-route":
        sampled = _importer(args).sample()
        return {
            "status": "sampled",
            "plan_sha256": sampled.plan_sha256,
            "input_count": sampled.input_count,
            "sampled_count": sampled.sampled_count,
            "stationary_deduplicated_count": sampled.stationary_deduplicated_count,
            "distance_excluded_count": sampled.distance_excluded_count,
        }
    if args.command == "import-capture":
        importer = _importer(args)
        if args.dry_run:
            preview = importer.preview()
            return {
                "status": "dry_run",
                "plan_sha256": preview.plan_sha256,
                "input_count": preview.input_count,
                "sampled_count": preview.sampled_count,
                "would_import_count": preview.sampled_count,
                "stationary_deduplicated_count": (
                    preview.stationary_deduplicated_count
                ),
                "distance_excluded_count": preview.distance_excluded_count,
                "rejection_reason_codes": [],
            }
        imported = importer.import_capture(
            resume=not args.no_resume,
            max_new_assets=args.max_new_assets,
        )
        return {
            "status": imported.status,
            "plan_sha256": imported.plan_sha256,
            "input_count": imported.input_count,
            "sampled_count": imported.sampled_count,
            "imported_count": imported.imported_count,
            "resumed_count": imported.resumed_count,
            "pending_privacy_count": imported.pending_privacy_count,
        }
    if args.command == "privacy-review":
        return _privacy_dispatch(args)
    if args.command == "build-capture-manifest":
        manifest = build_capture_manifest(args.work_root, args.output)
        return {
            "status": manifest.status,
            "eligible_count": manifest.eligible_count,
            "excluded_count": manifest.excluded_count,
            "excluded_by_reason": manifest.excluded_by_reason,
            "manifest_sha256": manifest.manifest_sha256,
        }
    raise CaptureImportError("capture_command_unsupported")


__all__ = ["CAPTURE_COMMANDS", "add_capture_subparsers", "dispatch_capture_command"]
