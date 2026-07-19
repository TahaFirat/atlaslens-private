from __future__ import annotations

import os
import sys
from pathlib import Path


MODEL_WORKERS_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_WORKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_WORKERS_ROOT))

from common.server import run_worker  # noqa: E402

from adapter import (  # noqa: E402
    MODEL_REVISION,
    PROVIDER,
    PROVIDER_REVISION,
    PaddleOCRAdapter,
)


def _port_from_environment() -> int:
    try:
        port = int(
            os.environ.get(
                "PADDLEOCR_WORKER_PORT",
                os.environ.get("ATLASLENS_PADDLEOCR_WORKER_PORT", "8793"),
            )
        )
    except ValueError as exc:
        raise SystemExit("invalid_worker_port") from exc
    if not 1024 <= port <= 65535:
        raise SystemExit("invalid_worker_port")
    return port


def main() -> None:
    host = os.environ.get(
        "PADDLEOCR_WORKER_HOST",
        os.environ.get("ATLASLENS_PADDLEOCR_WORKER_HOST", "127.0.0.1"),
    )
    if host != "127.0.0.1":
        raise SystemExit("worker_host_must_be_loopback")
    run_worker(
        PaddleOCRAdapter(),
        provider=PROVIDER,
        provider_revision=PROVIDER_REVISION,
        model_revision=MODEL_REVISION,
        host=host,
        port=_port_from_environment(),
        max_request_bytes=29 * 1024 * 1024,
        max_image_bytes=20 * 1024 * 1024,
        max_response_bytes=2 * 1024 * 1024,
        max_decoded_pixels=16_000_000,
        max_image_dimension=4096,
        request_timeout_seconds=60.0,
    )


if __name__ == "__main__":
    main()
