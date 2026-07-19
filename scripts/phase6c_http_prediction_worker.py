#!/usr/bin/env python3
"""Run one leakage-safe Phase 6C prediction through a loopback AtlasLens API."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import BinaryIO, TextIO

from atlaslens_api.evaluation.phase6c_http import (
    Phase6CHTTPPredictionWorker,
    Phase6CHTTPWorkerConfig,
    Phase6CHTTPWorkerError,
    read_isolated_request,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=110.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.25)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-input-bytes", type=int, default=25 * 1024 * 1024)
    parser.add_argument("--max-response-bytes", type=int, default=8 * 1024 * 1024)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: BinaryIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    args = _parser().parse_args(argv)
    input_stream = stdin or sys.stdin.buffer
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    try:
        config = Phase6CHTTPWorkerConfig(
            api_base_url=args.api_base_url,
            timeout_seconds=args.timeout_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            max_candidates=args.max_candidates,
            max_input_bytes=args.max_input_bytes,
            max_response_bytes=args.max_response_bytes,
        )
        request = read_isolated_request(input_stream)
    except (Phase6CHTTPWorkerError, ValueError):
        error_stream.write("phase6c_http_prediction_worker:request_or_config_invalid\n")
        return 2

    response = Phase6CHTTPPredictionWorker(config).predict(request)
    output_stream.write(response.model_dump_json())
    output_stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
