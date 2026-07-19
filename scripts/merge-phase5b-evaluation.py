"""Merge the reviewed Phase 5 smoke slice and frozen Phase 5B sample exactly once."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = ROOT / ".local"
OUTPUT = LOCAL_ROOT / "phase5b-fixed-evaluation-v2"
MANIFEST = OUTPUT / "manifest.csv"
RECEIPT = OUTPUT / "receipt.json"
SOURCES = (
    (LOCAL_ROOT / "evaluation" / "manifest.csv", "acceptance-images"),
    (LOCAL_ROOT / "phase5b-evaluation" / "manifest.csv", "phase5b-evaluation/images"),
)

PHASE5B_SCENE_CATEGORIES = (
    "text_rich_graffiti",
    "indoor_text_poor",
    "architecture_difficult_viewpoint",
    "indoor_text_poor",
    "urban_aerial_difficult",
    "text_rich_transit",
    "text_rich_urban",
    "text_rich_urban",
    "transit_infrastructure",
    "text_rich_public_institution",
    "natural_urban_park",
    "text_rich_indoor",
    "text_rich_public_institution",
    "text_rich_public_event",
    "wildlife_low_resolution",
    "mountain_industrial_landscape",
    "wildlife_closeup",
    "mountain_lakeside",
    "glacier_closeup_difficult",
    "public_landmark",
    "desert_natural",
    "text_rich_indoor_odd_landmark",
    "mountain_village_road",
    "natural_lake_mountain",
)


def main() -> int:
    if MANIFEST.exists() or RECEIPT.exists():
        raise RuntimeError("the merged Phase 5B evaluation set is already frozen")
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None
    for source_index, (source, prefix) in enumerate(SOURCES):
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            current_headers = list(reader.fieldnames or ())
            if headers is None:
                headers = current_headers
            elif headers != current_headers:
                raise RuntimeError("evaluation manifest schemas differ")
            for row_index, row in enumerate(reader):
                row["local_reference"] = f"{prefix}/{row['local_reference']}"
                if source_index == 1:
                    row["scene_category"] = PHASE5B_SCENE_CATEGORIES[row_index]
                rows.append(row)
    if headers is None or len(rows) != 30:
        raise RuntimeError("the fixed evaluation merge must contain exactly 30 rows")
    hashes = [row["content_sha256"] for row in rows]
    source_ids = [(row["source"], row["source_record_id"]) for row in rows]
    families = [row["capture_family_id"] for row in rows]
    if len(set(hashes)) != len(rows) or len(set(source_ids)) != len(rows):
        raise RuntimeError("fixed evaluation contains an exact duplicate")
    if len(set(families)) != len(rows):
        raise RuntimeError("fixed evaluation contains a repeated capture family")
    perceptual = [int(row["perceptual_hash"], 16) for row in rows]
    for index, first in enumerate(perceptual):
        for second in perceptual[index + 1 :]:
            if (first ^ second).bit_count() <= 4:
                raise RuntimeError("fixed evaluation contains a near duplicate")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with MANIFEST.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    receipt = {
        "schema": "phase5b-fixed-evaluation-merge-v2",
        "row_fingerprint": hashlib.sha256(canonical).hexdigest(),
        "image_count": len(rows),
        "source_slices": [str(path.relative_to(ROOT)) for path, _ in SOURCES],
        "selection_precedes_phase5b_predictions": True,
        "reference_index_overlap_forbidden": True,
        "scene_categories_frozen_after_visual_qa_before_baseline": True,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
