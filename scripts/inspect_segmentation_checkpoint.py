"""Inspect a trusted local SegFormer training checkpoint without printing tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "services" / "api" / "src"))

from atlaslens_api.segmentation_artifacts.checkpoint import (  # noqa: E402
    SegmentationCheckpointError,
    inspect_checkpoint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a trusted local SegFormer checkpoint on CPU."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=REPOSITORY_ROOT / "last_checkpoint.pt",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        inspection = inspect_checkpoint(args.checkpoint)
    except SegmentationCheckpointError as exc:
        print(f"checkpoint inspection failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(inspection.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
