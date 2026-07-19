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

from adapter import OSV5MAdapter, OSV5MArtifacts  # noqa: E402


SOURCE_REVISION = "4e6075387ecde4255410785ffb83830c9aa099f6"
MODEL_REVISION = "71548b90ac4a1aa7c37839841f411a06da82b1a6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="atlaslens-osv5m-worker")
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            os.environ.get("OSV5M_SOURCE_DIR", PROJECT_ROOT / ".local/vendor/osv5m")
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "OSV5M_MODEL_DIR",
                PROJECT_ROOT / ".local/models/phase6b/osv5m",
            )
        ),
    )
    parser.add_argument("--max-image-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--max-image-pixels", type=int, default=40_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65_535:
        raise SystemExit("port must be between 1 and 65535")
    if args.max_image_bytes <= 0 or args.max_image_pixels <= 0:
        raise SystemExit("image bounds must be positive")
    adapter = OSV5MAdapter(
        OSV5MArtifacts(
            source_dir=args.source_dir.resolve(),
            model_dir=args.model_dir.resolve(),
            source_revision=SOURCE_REVISION,
            model_revision=MODEL_REVISION,
        ),
        max_input_bytes=args.max_image_bytes,
        max_image_pixels=args.max_image_pixels,
    )
    run_worker(
        adapter,
        provider="osv5m",
        provider_revision=SOURCE_REVISION,
        model_revision=MODEL_REVISION,
        host=args.host,
        port=args.port,
        max_request_bytes=max(args.max_image_bytes * 2, 1024 * 1024),
        max_image_bytes=args.max_image_bytes,
        max_response_bytes=256 * 1024,
    )


if __name__ == "__main__":
    main()
