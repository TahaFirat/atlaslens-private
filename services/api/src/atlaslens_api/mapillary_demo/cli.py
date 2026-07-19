"""Argument parsers and dispatch seam for the seven Phase 3B3 commands."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import SecretStr

from .acquisition import (
    MapillaryAcquisition,
    MapillaryCoverageAuditor,
    _safe_root,
    atomic_write_model,
    build_acquisition_plan,
    canonical_sha256,
    load_acquisition_manifest,
    load_acquisition_plan,
    load_aoi_catalog,
    load_coverage_audit,
)
from .client import MapillaryClient, validate_access_token
from .errors import MapillarySafetyError, MapillaryTokenError
from .models import MAPILLARY_PILOT_ROOT, ClientLimits

MAPILLARY_COMMANDS = (
    "mapillary-check-token",
    "mapillary-coverage",
    "mapillary-plan",
    "mapillary-acquire",
    "mapillary-status",
    "mapillary-reconcile",
    "mapillary-cleanup",
)
MAPILLARY_TOKEN_ENV_NAME = "MAPILLARY_ACCESS_TOKEN"  # noqa: S105 - key name, not a secret
_MAX_ENV_FILE_BYTES = 1024 * 1024
_MAX_ENV_LINE_CHARACTERS = 16 * 1024

ClientFactory = Callable[[str, ClientLimits], MapillaryClient]


def load_mapillary_access_token(
    env_file: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> SecretStr:
    """Load only the named runtime secret without parsing the rest of `.env`."""

    selected = os.environ if environ is None else environ
    value = selected.get(MAPILLARY_TOKEN_ENV_NAME)
    if value is not None:
        return validate_access_token(value)
    if (
        env_file.is_symlink()
        or not env_file.is_file()
        or env_file.stat().st_size > _MAX_ENV_FILE_BYTES
    ):
        raise MapillaryTokenError("mapillary_token_not_configured")
    try:
        with env_file.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if len(line) > _MAX_ENV_LINE_CHARACTERS:
                    raise MapillaryTokenError("mapillary_token_source_invalid")
                key, separator, raw = line.partition("=")
                if not separator or key.strip() != MAPILLARY_TOKEN_ENV_NAME:
                    continue
                candidate = raw.strip()
                if candidate.startswith(("'", '"')):
                    if len(candidate) < 2 or candidate[-1] != candidate[0]:
                        raise MapillaryTokenError("mapillary_token_source_invalid")
                    candidate = candidate[1:-1]
                return validate_access_token(candidate)
    except (OSError, UnicodeError):
        raise MapillaryTokenError("mapillary_token_source_invalid") from None
    raise MapillaryTokenError("mapillary_token_not_configured")


def _client_factory(token: str, limits: ClientLimits) -> MapillaryClient:
    return MapillaryClient(token, limits=limits)


def _add_client_limits(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request-cap", type=int, default=2_500)
    parser.add_argument("--page-cap", type=int, default=300)
    parser.add_argument("--item-cap", type=int, default=20_000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--retry-cap", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)


def _add_pilot_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pilot-root", type=Path, default=Path(MAPILLARY_PILOT_ROOT))


def _add_catalog(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--aoi-config", type=Path, required=True)


def add_mapillary_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    check = subparsers.add_parser(
        "mapillary-check-token",
        help="Check the runtime Mapillary token while returning status only.",
    )
    _add_client_limits(check)

    coverage = subparsers.add_parser(
        "mapillary-coverage",
        help="Run the mandatory metadata-only audit for all five versioned AOIs.",
    )
    _add_pilot_root(coverage)
    _add_catalog(coverage)
    _add_client_limits(coverage)
    coverage.add_argument("--output", type=Path, default=Path("metadata/coverage-audit.json"))

    plan = subparsers.add_parser(
        "mapillary-plan",
        help="Choose one audited urban AOI and bind a rights-policy receipt.",
    )
    _add_pilot_root(plan)
    _add_catalog(plan)
    plan.add_argument("--audit", type=Path, required=True)
    plan.add_argument("--source-policy", type=Path, required=True)
    plan.add_argument("--target-reference-images", type=int, default=1_500)
    plan.add_argument("--target-holdout-images", type=int, default=100)
    plan.add_argument("--request-cap", type=int, default=2_500)
    plan.add_argument("--concurrency", type=int, default=2)
    plan.add_argument("--output", type=Path, default=Path("metadata/acquisition-plan.json"))

    acquire = subparsers.add_parser(
        "mapillary-acquire",
        help="Acquire the single audited pilot with hard image, request and byte caps.",
    )
    _add_pilot_root(acquire)
    _add_catalog(acquire)
    _add_client_limits(acquire)
    acquire.add_argument("--audit", type=Path, required=True)
    acquire.add_argument("--plan", type=Path, required=True)
    acquire.add_argument("--no-resume", action="store_true")

    status = subparsers.add_parser(
        "mapillary-status",
        help="Inspect the credential-free acquisition state.",
    )
    _add_pilot_root(status)

    reconcile = subparsers.add_parser(
        "mapillary-reconcile",
        help="Recheck local hashes and official remote image presence.",
    )
    _add_pilot_root(reconcile)
    _add_client_limits(reconcile)

    cleanup = subparsers.add_parser(
        "mapillary-cleanup",
        help="Dry-run or execute bounded raw-cache cleanup inside the pilot root.",
    )
    _add_pilot_root(cleanup)
    _add_client_limits(cleanup)
    cleanup.add_argument("--retain-image-id", action="append", default=[])
    cleanup.add_argument("--execute", action="store_true")
    cleanup.add_argument("--validation-receipt", type=Path)


def _limits(args: argparse.Namespace) -> ClientLimits:
    return ClientLimits(
        request_cap=args.request_cap,
        page_cap=args.page_cap,
        metadata_item_cap=args.item_cap,
        page_size=args.page_size,
        timeout_seconds=args.timeout_seconds,
        retry_cap=args.retry_cap,
        concurrency=args.concurrency,
    )


def _relative_output(value: Path) -> PurePosixPath:
    path = PurePosixPath(value.as_posix())
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise MapillarySafetyError("mapillary_output_path_invalid")
    return path


def _token(value: str | None) -> str:
    if value is None:
        raise MapillaryTokenError("mapillary_token_not_configured")
    return value


def _read_validation_receipt(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise MapillarySafetyError("mapillary_cleanup_validation_invalid")
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MapillarySafetyError("mapillary_cleanup_validation_invalid") from exc
    if not isinstance(value, dict):
        raise MapillarySafetyError("mapillary_cleanup_validation_invalid")
    return value


def _status(root: Path) -> dict[str, object]:
    manifest_path = root / "metadata" / "acquisition-manifest.json"
    checkpoint_path = root / "checkpoints" / "acquisition.json"
    if not manifest_path.exists() and not checkpoint_path.exists():
        return {"state": "not_started", "image_count": 0, "downloaded_bytes": 0}
    if manifest_path.exists() != checkpoint_path.exists():
        raise MapillarySafetyError("mapillary_acquisition_state_incomplete")
    manifest = load_acquisition_manifest(manifest_path)
    if checkpoint_path.is_symlink() or checkpoint_path.stat().st_size > 4 * 1024 * 1024:
        raise MapillarySafetyError("mapillary_acquisition_state_incomplete")
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MapillarySafetyError("mapillary_acquisition_state_incomplete") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != (
        "atlaslens-mapillary-acquisition-checkpoint-v1"
    ):
        raise MapillarySafetyError("mapillary_acquisition_state_incomplete")
    state = checkpoint.get("status")
    if state not in {"in_progress", "completed", "cancelled", "failed"}:
        raise MapillarySafetyError("mapillary_acquisition_state_incomplete")
    return {
        "state": state,
        "aoi_id": manifest.aoi_id,
        "image_count": len(manifest.assets),
        "downloaded_bytes": manifest.downloaded_bytes,
        "request_count": checkpoint.get("request_count"),
        "page_count": checkpoint.get("page_count"),
        "failure_code": checkpoint.get("failure_code"),
    }


def dispatch_mapillary_command(
    args: argparse.Namespace,
    *,
    access_token: str | None,
    client_factory: ClientFactory = _client_factory,
    approved_root: Path = Path(MAPILLARY_PILOT_ROOT),
) -> dict[str, object]:
    """Dispatch a parsed command; access_token is runtime-only and never returned."""

    if args.command == "mapillary-status":
        root = _safe_root(args.pilot_root, approved_root)
        return _status(root)
    if args.command == "mapillary-plan":
        root = _safe_root(args.pilot_root, approved_root)
        audit = load_coverage_audit(args.audit)
        catalog = load_aoi_catalog(args.aoi_config)
        plan = build_acquisition_plan(
            audit,
            catalog,
            source_policy_path=args.source_policy,
            target_reference_images=args.target_reference_images,
            target_holdout_images=args.target_holdout_images,
            hard_request_cap=args.request_cap,
            concurrency=args.concurrency,
        )
        atomic_write_model(root, _relative_output(args.output), plan)
        return {
            "status": "planned",
            "aoi_id": plan.aoi_id,
            "planned_images": (plan.target_reference_images + plan.target_holdout_images),
            "plan_sha256": canonical_sha256(plan),
        }

    limits = _limits(args)
    client = client_factory(_token(access_token), limits)
    try:
        if args.command == "mapillary-check-token":
            return {"token_status": client.check_token()}
        if args.command == "mapillary-coverage":
            root = _safe_root(args.pilot_root, approved_root)
            catalog = load_aoi_catalog(args.aoi_config)
            audit = MapillaryCoverageAuditor(client).audit(catalog)
            atomic_write_model(root, _relative_output(args.output), audit)
            return {
                "status": "metadata_audited",
                "region_count": len(audit.regions),
                "selected_aoi_id": audit.selected_aoi_id,
                "request_count": audit.request_count,
                "page_count": audit.page_count,
                "rejected_item_count": sum(
                    region.rejected_item_count for region in audit.regions
                ),
                "imagery_downloaded": False,
            }
        acquisition = MapillaryAcquisition(
            client,
            args.pilot_root,
            approved_root=approved_root,
        )
        if args.command == "mapillary-acquire":
            manifest = acquisition.acquire(
                load_aoi_catalog(args.aoi_config),
                load_coverage_audit(args.audit),
                load_acquisition_plan(args.plan),
                resume=not args.no_resume,
            )
            return {
                "status": "acquired",
                "aoi_id": manifest.aoi_id,
                "image_count": len(manifest.assets),
                "downloaded_bytes": manifest.downloaded_bytes,
                "rejected_item_count": client.rejected_item_count,
            }
        if args.command == "mapillary-reconcile":
            manifest = acquisition.reconcile()
            states: dict[str, int] = {}
            for asset in manifest.assets:
                states[asset.reconciliation_state] = states.get(asset.reconciliation_state, 0) + 1
            return {"status": "reconciled", "states": states}
        if args.command == "mapillary-cleanup":
            report = acquisition.cleanup(
                retain_image_ids=args.retain_image_id,
                execute=args.execute,
                validation_receipt=_read_validation_receipt(args.validation_receipt),
            )
            return report.model_dump(mode="json")
    finally:
        client.close()
    raise MapillarySafetyError("mapillary_command_unsupported")


__all__ = [
    "MAPILLARY_COMMANDS",
    "MAPILLARY_TOKEN_ENV_NAME",
    "add_mapillary_subparsers",
    "dispatch_mapillary_command",
    "load_mapillary_access_token",
]
