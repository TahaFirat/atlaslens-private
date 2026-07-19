#!/usr/bin/env python3
"""Plan or explicitly acquire a bounded, licensed Türkiye MegaLoc reference input."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from atlaslens_api.gazetteer import GazetteerManager, GazetteerMetadata, SQLiteGazetteerResolver
from atlaslens_api.gazetteer.cli import default_cache_root as default_gazetteer_cache_root
from atlaslens_api.phase6c.catalogue import (
    default_coordinate_catalogue_path,
    load_coordinate_catalogue,
)
from atlaslens_api.phase6c.reference_acquisition import (
    KARTAVIEW_REVIEWED_TERMS_VERSION,
    AcquisitionPolicy,
    KartaViewApiAdapter,
    LocalAdministrativeResolver,
    MapillaryGraphAdapter,
    ReferenceAcquisitionError,
    ReferenceCorpusAcquirer,
    ReferenceSourceAdapter,
    build_acquisition_plan,
    load_mapillary_backend_settings,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ENV_FILE = REPOSITORY_ROOT / ".env"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / ".local" / "phase6c-reference",
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=default_coordinate_catalogue_path(),
    )
    parser.add_argument("--gazetteer-cache", type=Path, default=default_gazetteer_cache_root())
    parser.add_argument("--plan-version", default="turkiye-balanced-v1")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-per-province", type=int, default=25)
    parser.add_argument("--max-per-sequence", type=int, default=4)
    parser.add_argument("--max-disk-gb", type=float, default=2.0)
    parser.add_argument("--estimated-image-kb", type=int, default=512)
    parser.add_argument("--enable-kartaview", action="store_true")
    parser.add_argument(
        "--accept-kartaview-terms-version",
        help=f"Required exact reviewed value when enabled: {KARTAVIEW_REVIEWED_TERMS_VERSION}",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow metadata requests and authorized image downloads.",
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Confirm the printed count/storage estimate before any network request.",
    )
    parser.add_argument("--confirm-large-index", action="store_true")
    return parser


def _write_plan(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _gazetteer(cache_root: Path) -> SQLiteGazetteerResolver:
    manager = GazetteerManager(cache_root)
    info = manager.info()
    if info.status != "ready" or info.schema_version is None or info.license is None:
        raise ReferenceAcquisitionError("gazetteer_unavailable")
    return SQLiteGazetteerResolver(
        manager.database_path,
        GazetteerMetadata(
            dataset=info.dataset,
            version=info.schema_version,
            source="https://download.geonames.org/export/dump/",
            license=info.license,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (
            not math.isfinite(args.max_disk_gb)
            or args.max_disk_gb <= 0
            or args.estimated_image_kb <= 0
        ):
            raise ReferenceAcquisitionError("acquisition_bounds_invalid")
        mapillary = load_mapillary_backend_settings(BACKEND_ENV_FILE)
        maximum_images = args.max_images if args.max_images is not None else mapillary.max_images
        if mapillary.enabled and maximum_images > mapillary.max_images:
            raise ReferenceAcquisitionError("mapillary_max_images_exceeded")
        policy = AcquisitionPolicy(
            max_images=maximum_images,
            max_per_province=args.max_per_province,
            max_per_sequence=args.max_per_sequence,
            max_disk_bytes=round(args.max_disk_gb * 1024**3),
            estimated_image_bytes=args.estimated_image_kb * 1024,
        )
        plan = build_acquisition_plan(
            policy=policy,
            catalogue=load_coordinate_catalogue(args.catalogue),
            plan_version=args.plan_version,
        )
        _write_plan(
            args.output_root.expanduser().resolve() / "acquisition-plan.json",
            plan.model_dump_json(indent=2),
        )
        print(
            json.dumps(
                {
                    "event": "reference_acquisition_estimate",
                    "province_plan_count": len(plan.provinces),
                    "estimated_image_count": plan.estimated_image_count,
                    "estimated_storage_bytes": plan.estimated_storage_bytes,
                    "mapillary_enabled": mapillary.enabled,
                    "kartaview_enabled": bool(args.enable_kartaview),
                    "network_allowed": bool(args.execute and args.confirm_download),
                },
                sort_keys=True,
            )
        )
        if not args.execute:
            return 0
        if not args.confirm_download:
            raise ReferenceAcquisitionError("download_confirmation_required")
        if plan.requires_large_confirmation and not args.confirm_large_index:
            raise ReferenceAcquisitionError("large_index_confirmation_required")

        sources: list[ReferenceSourceAdapter] = []
        if mapillary.enabled:
            sources.append(
                MapillaryGraphAdapter(
                    mapillary,
                    bbox_half_span_degrees=policy.mapillary_bbox_half_span_degrees,
                )
            )
        if args.enable_kartaview:
            sources.append(
                KartaViewApiAdapter(
                    accepted_terms_version=args.accept_kartaview_terms_version or ""
                )
            )
        if not sources:
            raise ReferenceAcquisitionError("no_reference_source_enabled")
        result = ReferenceCorpusAcquirer(
            sources=sources,
            administrative_resolver=LocalAdministrativeResolver(
                _gazetteer(args.gazetteer_cache)
            ),
        ).acquire(plan, args.output_root)
        print(
            json.dumps(
                {
                    "event": "reference_acquisition_finished",
                    **result.model_dump(mode="json"),
                },
                sort_keys=True,
            )
        )
        return 0 if result.status != "empty" else 1
    except (ReferenceAcquisitionError, OSError, ValueError) as exc:
        code = str(exc) if isinstance(exc, ReferenceAcquisitionError) else "acquisition_failed"
        print(
            json.dumps({"event": "reference_acquisition_failed", "reason_code": code}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
