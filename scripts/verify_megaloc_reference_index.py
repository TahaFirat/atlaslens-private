#!/usr/bin/env python3
"""Verify a Phase 6C MegaLoc index without loading reference images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atlaslens_api.phase6c.reference_index import open_reference_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()

    opened = open_reference_index(args.index)
    diagnostics = opened.diagnostics
    payload: dict[str, object] = diagnostics.model_dump(mode="json")
    if opened.index is not None:
        manifest = opened.index.manifest
        payload.update(
            {
                "model_id": manifest.descriptor.model_id,
                "model_revision": manifest.descriptor.model_revision,
                "source_revision": manifest.descriptor.source_revision,
                "covered_country_count": len(manifest.covered_countries),
                "covered_province_count": len(manifest.covered_provinces),
                "source_distribution": manifest.source_distribution,
                "duplicate_exclusion_count": manifest.deduplication.excluded_count,
                "artifact_checksums": {
                    "descriptors": manifest.vectors.sha256,
                    "metadata": manifest.metadata.sha256,
                    "exclusions": manifest.exclusions.sha256,
                },
            }
        )
    print(json.dumps(payload, sort_keys=True))
    return 0 if diagnostics.status in {"ready", "empty"} else 1


if __name__ == "__main__":
    sys.exit(main())
