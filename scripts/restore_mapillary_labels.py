"""Restore reviewed Mapillary labels on an existing safe SegFormer export."""

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
)
from atlaslens_api.segmentation_artifacts.relabeling import (  # noqa: E402
    restore_mapillary_labels,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and restore the reviewed 124-label Mapillary mapping without "
            "loading or changing SegFormer weights. No network access is used."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_output_directory(REPOSITORY_ROOT),
    )
    parser.add_argument(
        "--label-config",
        type=Path,
        help=(
            "Optional repository-contained label config. By default the reviewed "
            "assets/mapillary paths and then root config.json are checked."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = restore_mapillary_labels(
            args.model_dir,
            repository_root=REPOSITORY_ROOT,
            label_config=args.label_config,
        )
    except SegmentationCheckpointError as exc:
        print(f"Mapillary label restoration failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "changed": result.changed,
                "classifier_output_count": result.classifier_output_count,
                "label_config": str(result.label_config),
                "label_mapping_source": result.label_mapping_source,
                "model_directory": str(result.model_directory),
                "semantic_label_names_available": True,
                "training_image_size": result.training_image_size,
                "weights_sha256": result.weights_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
