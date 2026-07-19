"""Prepare a trusted training checkpoint as an offline Hugging Face deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "services" / "api" / "src"))

from atlaslens_api.segmentation_artifacts.checkpoint import (  # noqa: E402
    SegmentationCheckpointError,
)
from atlaslens_api.segmentation_artifacts.preparation import (  # noqa: E402
    default_output_directory,
    prepare_segmentation_model,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a trusted SegFormer checkpoint using only an already-installed "
            "local Hugging Face base model. No model or label data is downloaded."
        )
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=REPOSITORY_ROOT / "last_checkpoint.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_directory(REPOSITORY_ROOT),
    )
    parser.add_argument("--base-model-dir", type=Path)
    parser.add_argument("--label-config", type=Path)
    parser.add_argument("--mapillary-zip", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = prepare_segmentation_model(
            args.checkpoint,
            args.output,
            repository_root=REPOSITORY_ROOT,
            base_model_directory=args.base_model_dir,
            label_config=args.label_config,
            mapillary_zip=args.mapillary_zip,
        )
    except SegmentationCheckpointError as exc:
        print(f"segmentation model preparation failed: {exc}", file=sys.stderr)
        return 2
    payload = dict(result.metadata)
    payload["output_directory"] = str(result.output_directory)
    payload["changed"] = result.changed
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
