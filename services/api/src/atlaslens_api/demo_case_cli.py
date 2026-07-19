from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.engine import make_url

from atlaslens_api.cases import CaseInvestigationService, SQLAlchemyCaseRepository
from atlaslens_api.corpus_index.errors import ArtifactIntegrityError
from atlaslens_api.database import create_database_engine
from atlaslens_api.demo_case import (
    InvestorDemoCaseError,
    LockedPilotAssetIdentity,
    load_retained_pilot_asset,
    prepare_investor_demo_case,
)
from atlaslens_api.mapillary_demo.api import (
    MapillaryDemoRuntime,
    PublishedMapillaryDemoSearchIndex,
)
from atlaslens_api.mapillary_demo.errors import MapillarySafetyError
from atlaslens_api.mapillary_demo.indexing import (
    MapillaryAttribution,
    PublishedMapillaryDemoIndex,
)
from atlaslens_api.mapillary_demo.models import MAPILLARY_PILOT_ROOT
from atlaslens_api.phase6c.megaloc import create_megaloc_http_worker_client
from atlaslens_api.repository import SQLAlchemyAnalysisRepository

_BUNDLE_RELATIVE_PATH = Path("derived") / "mapillary-faiss-index"


def _sha256(value: str) -> str:
    normalized = value.strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas private-demo",
        description="Prepare one canonical private investor demo case.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="Validate the retained pilot and idempotently prepare the runtime case.",
    )
    prepare.add_argument("--database-url", required=True)
    prepare.add_argument(
        "--pilot-root", type=Path, default=Path(MAPILLARY_PILOT_ROOT)
    )
    prepare.add_argument("--expected-publication-sha256", required=True, type=_sha256)
    prepare.add_argument("--expected-source-policy-sha256", required=True, type=_sha256)
    prepare.add_argument("--expected-selection-lock-sha256", required=True, type=_sha256)
    prepare.add_argument("--worker-host", default="127.0.0.1")
    prepare.add_argument("--worker-port", type=int, default=8794)
    prepare.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prepare.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def _validate_database_url(value: str) -> str:
    try:
        url = make_url(value)
    except ValueError as exc:
        raise InvestorDemoCaseError("demo_database_url_invalid") from exc
    if (
        url.get_backend_name() != "sqlite"
        or not url.database
        or url.database == ":memory:"
        or url.query
    ):
        raise InvestorDemoCaseError("demo_database_must_be_dedicated_sqlite")
    path = Path(url.database).expanduser()
    if not path.is_absolute() or path.suffix.casefold() not in {
        ".db",
        ".sqlite",
        ".sqlite3",
    }:
        raise InvestorDemoCaseError("demo_database_must_be_dedicated_sqlite")
    return value


async def _prepare(args: argparse.Namespace) -> dict[str, object]:
    database_url = _validate_database_url(args.database_url)
    approved_root = Path(MAPILLARY_PILOT_ROOT)
    bundle_path = args.pilot_root / _BUNDLE_RELATIVE_PATH
    index = PublishedMapillaryDemoSearchIndex(
        enabled=True,
        bundle_path=bundle_path,
        expected_publication_sha256=args.expected_publication_sha256,
        expected_source_policy_sha256=args.expected_source_policy_sha256,
        expected_selection_lock_sha256=args.expected_selection_lock_sha256,
        uncertainty_radius_m=1_000.0,
    )
    index_status = index.status()
    if index_status.state != "active":
        raise InvestorDemoCaseError("demo_index_not_ready")
    try:
        publication = PublishedMapillaryDemoIndex.open(
            bundle_path,
            expected_source_policy_sha256=args.expected_source_policy_sha256,
            expected_selection_lock_sha256=args.expected_selection_lock_sha256,
        )
    except (ArtifactIntegrityError, OSError, ValueError) as exc:
        raise InvestorDemoCaseError("demo_index_not_ready") from exc
    def identity(record: MapillaryAttribution) -> LockedPilotAssetIdentity:
        return LockedPilotAssetIdentity(
            source_asset_id=record.mapillary_image_id,
            raw_sha256=record.raw_sha256,
            normalized_sha256=record.normalized_sha256,
            perceptual_hash=record.perceptual_hash,
            sequence_id=record.sequence_id,
            creator_id=record.creator_id,
        )

    references = tuple(
        identity(item)
        for item in publication.attribution.values()
        if item.split == "reference"
    )
    holdout = {
        item.mapillary_image_id: identity(item)
        for item in publication.attribution.values()
        if item.split == "holdout"
    }
    asset = load_retained_pilot_asset(
        args.pilot_root,
        approved_root=approved_root,
        reference_image_ids={item.source_asset_id for item in references},
        reference_assets=references,
        locked_holdout_assets=holdout,
    )
    if not 1 <= args.worker_port <= 65_535:
        raise InvestorDemoCaseError("demo_worker_configuration_invalid")
    if not 0 < args.timeout_seconds <= 180:
        raise InvestorDemoCaseError("demo_worker_configuration_invalid")
    try:
        worker = create_megaloc_http_worker_client(
            host=args.worker_host,
            port=args.worker_port,
            timeout_seconds=args.timeout_seconds,
            max_image_bytes=20 * 1024 * 1024,
        )
    except ValueError as exc:
        raise InvestorDemoCaseError("demo_worker_configuration_invalid") from exc
    runtime = MapillaryDemoRuntime(
        index=index,
        worker=worker,
        device=args.device,
        timeout_seconds=args.timeout_seconds,
        top_k=5,
    )
    engine = create_database_engine(database_url)
    analysis_repository = SQLAlchemyAnalysisRepository(engine)
    case_repository = SQLAlchemyCaseRepository(engine)
    service = CaseInvestigationService(case_repository, analysis_repository)
    try:
        await analysis_repository.initialize()
        await service.initialize()
        result = await prepare_investor_demo_case(
            service=service,
            analysis_repository=analysis_repository,
            runtime=runtime,
            asset=asset,
            publication_sha256=args.expected_publication_sha256,
        )
        return result.safe_payload()
    finally:
        await runtime.close()
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = asyncio.run(_prepare(args))
    except (
        ArtifactIntegrityError,
        InvestorDemoCaseError,
        MapillarySafetyError,
        OSError,
        ValueError,
    ) as exc:
        code = exc.code if isinstance(exc, InvestorDemoCaseError) else "demo_prepare_failed"
        print(
            json.dumps({"status": "error", "code": code}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
