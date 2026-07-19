from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODEL_WORKERS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(MODEL_WORKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_WORKERS_ROOT))

from common.server import run_worker  # noqa: E402

from adapter import MegaLocAdapter, MegaLocArtifacts  # noqa: E402

SOURCE_REVISION = "1af071c68fc3ab6c6018c5c868391763516e50f7"
MODEL_REVISION = "7cb9f7970d366fdf059963d04d372e503e8e9df9"
WEIGHT_SHA256 = "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="atlaslens-megaloc-worker")
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(os.environ.get("MEGALOC_SOURCE_DIR", PROJECT_ROOT / ".local/vendor/megaloc")),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            os.environ.get("MEGALOC_MODEL_DIR", PROJECT_ROOT / ".local/models/phase6c/megaloc")
        ),
    )
    parser.add_argument("--max-image-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--max-image-pixels", type=int, default=40_000_000)
    parser.add_argument("--maximum-edge", type=int, default=560)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = MegaLocAdapter(
        MegaLocArtifacts(
            source_dir=args.source_dir.resolve(),
            model_dir=args.model_dir.resolve(),
            source_revision=SOURCE_REVISION,
            model_revision=MODEL_REVISION,
            weight_sha256=WEIGHT_SHA256,
        ),
        max_input_bytes=args.max_image_bytes,
        max_image_pixels=args.max_image_pixels,
        maximum_edge=args.maximum_edge,
    )
    run_worker(
        adapter,
        provider="megaloc",
        provider_revision=SOURCE_REVISION,
        model_revision=MODEL_REVISION,
        host=args.host,
        port=args.port,
        max_request_bytes=max(args.max_image_bytes * 2, 1024 * 1024),
        max_image_bytes=args.max_image_bytes,
        max_response_bytes=2 * 1024 * 1024,
        request_timeout_seconds=180,
    )


if __name__ == "__main__":
    main()
