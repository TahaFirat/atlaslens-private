from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from atlaslens_api.acceptance.runner import AcceptanceError, run_acceptance
from atlaslens_api.asyncio_utils import run_isolated
from atlaslens_api.gazetteer import (
    GazetteerManager,
    GazetteerMetadata,
    SQLiteGazetteerResolver,
)
from atlaslens_api.gazetteer.cli import default_cache_root as default_gazetteer_cache_root
from atlaslens_api.model_management.cli import default_cache_root
from atlaslens_api.model_management.errors import ModelManagementError
from atlaslens_api.model_management.manifest import default_manifest_path, load_manifest
from atlaslens_api.model_management.service import ModelManagementService
from atlaslens_api.providers.geoclip import GeoCLIPGlobalGeolocationProvider, select_device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas acceptance")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--directory", type=Path, required=True)
    run.add_argument("--provider", choices=("geoclip",), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--cache-root", type=Path, default=default_cache_root())
    run.add_argument(
        "--gazetteer-cache", type=Path, default=default_gazetteer_cache_root()
    )
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        management = ModelManagementService(
            args.cache_root, load_manifest(default_manifest_path())
        )
        provider = GeoCLIPGlobalGeolocationProvider(
            management,
            device_selector=lambda: select_device(args.device),
            timeout_seconds=180.0,
        )
        gazetteer_manager = GazetteerManager(args.gazetteer_cache)
        gazetteer_info = gazetteer_manager.info()
        gazetteer = None
        if (
            gazetteer_info.status == "ready"
            and gazetteer_info.schema_version is not None
            and gazetteer_info.license is not None
        ):
            gazetteer = SQLiteGazetteerResolver(
                gazetteer_manager.database_path,
                GazetteerMetadata(
                    dataset=gazetteer_info.dataset,
                    version=gazetteer_info.schema_version,
                    source="https://download.geonames.org/export/dump/",
                    license=gazetteer_info.license,
                ),
            )
        report = run_isolated(
            run_acceptance(
                args.directory, args.output, provider, gazetteer=gazetteer
            )
        )
    except (AcceptanceError, ModelManagementError, OSError, ValueError):
        print(
            json.dumps({"status": "error", "code": "acceptance_operation_failed"}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
