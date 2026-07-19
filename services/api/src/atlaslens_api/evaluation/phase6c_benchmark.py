from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, TextIO

from atlaslens_api.evaluation.manifest import (
    EvaluationManifestError,
    EvaluationManifestLoader,
)
from atlaslens_api.evaluation.models import EvaluationProvider
from atlaslens_api.evaluation.phase6c_http import (
    Phase6CHTTPEvaluationProvider,
    Phase6CHTTPPredictionWorker,
    Phase6CHTTPWorkerConfig,
)
from atlaslens_api.evaluation.runner import BenchmarkRunner


class Phase6CBenchmarkError(ValueError):
    """Safe operator-input failure without sensitive payload details."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise Phase6CBenchmarkError("arguments_invalid")


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Benchmark a licensed Türkiye development/validation manifest against "
            "a loopback Phase 6C API."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--allowed-license", action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--request-timeout-seconds", type=float, required=True)
    parser.add_argument("--poll-interval-seconds", type=float, required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-input-bytes", type=int, default=25 * 1024 * 1024)
    parser.add_argument("--max-response-bytes", type=int, default=8 * 1024 * 1024)
    return parser


def run_phase6c_benchmark(
    argv: Sequence[str],
    *,
    provider: EvaluationProvider | None = None,
) -> dict[str, int | str]:
    args = build_parser().parse_args(argv)
    config = Phase6CHTTPWorkerConfig(
        api_base_url=args.api_base_url,
        timeout_seconds=args.timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        max_candidates=args.max_candidates,
        max_input_bytes=args.max_input_bytes,
        max_response_bytes=args.max_response_bytes,
    )
    manifest = EvaluationManifestLoader(
        allowed_licenses=frozenset(str(item) for item in args.allowed_license)
    ).load(args.manifest, args.asset_root)
    if any(asset.record.split == "test" for asset in manifest.assets):
        raise Phase6CBenchmarkError("test_split_forbidden")
    if any(asset.record.country_code != "TR" for asset in manifest.assets):
        raise Phase6CBenchmarkError("non_turkiye_record_forbidden")

    selected_provider = (
        provider
        if provider is not None
        else Phase6CHTTPEvaluationProvider(Phase6CHTTPPredictionWorker(config))
    )
    run = BenchmarkRunner().run(
        manifest,
        selected_provider,
        output_directory=args.output_directory,
    )
    return {
        "event": "phase6c_benchmark_completed",
        "image_count": run.summary.image_count,
        "successful_inference_count": run.summary.successful_inference.numerator,
        "provider_failure_count": run.summary.provider_failure.numerator,
        "abstention_count": run.summary.abstention.numerator,
        "report_count": len(run.report_paths or ()),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: EvaluationProvider | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    try:
        result = run_phase6c_benchmark(
            tuple(argv) if argv is not None else sys.argv[1:], provider=provider
        )
    except (EvaluationManifestError, Phase6CBenchmarkError, OSError, ValueError):
        error_stream.write("phase6c_benchmark_failed\n")
        return 2
    except Exception:  # The CLI boundary must never print payload-bearing tracebacks.
        error_stream.write("phase6c_benchmark_failed\n")
        return 2
    output_stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0
