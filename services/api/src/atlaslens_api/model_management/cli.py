from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from PIL import Image

from atlaslens_api.asyncio_utils import run_isolated
from atlaslens_api.inference.models import InferenceRequest
from atlaslens_api.model_management.errors import ModelManagementError
from atlaslens_api.model_management.manifest import (
    default_manifest_path,
    load_embedding_manifest,
    load_manifest,
    siglip2_manifest_path,
)
from atlaslens_api.model_management.models import ModelBenchmarkResult, ModelSmokeObservation
from atlaslens_api.model_management.paths import default_cache_root as default_cache_root
from atlaslens_api.model_management.rapidocr import RapidOCRModelManagementService
from atlaslens_api.model_management.service import (
    EmbeddingModelManagementService,
    ModelManagementService,
)
from atlaslens_api.providers.base import InvocationContext, OutcomeStatus
from atlaslens_api.providers.geoclip import GeoCLIPGlobalGeolocationProvider, select_device
from atlaslens_api.schemas import AnalysisMode
from atlaslens_api.storage import LocalImageHandle
from atlaslens_api.trained_artifacts.errors import TrainedArtifactError
from atlaslens_api.trained_artifacts.integration import build_custom_provider
from atlaslens_api.trained_artifacts.manager import TrainedArtifactManager
from atlaslens_api.trained_artifacts.models import DeploymentMode

_MANAGED_MODELS = ("geoclip", "siglip2-b16-384", "rapidocr")


def build_models_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas models")
    parser.add_argument("--cache-root", type=Path, default=default_cache_root())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    install = commands.add_parser("install")
    install.add_argument("model", choices=_MANAGED_MODELS)
    for name in ("verify", "info"):
        item = commands.add_parser(name)
        item.add_argument("model")
    remove = commands.add_parser("remove")
    remove.add_argument("model", choices=_MANAGED_MODELS)
    remove.add_argument("--yes", action="store_true")
    smoke = commands.add_parser("test")
    smoke.add_argument("model")
    smoke.add_argument("--image", type=Path, required=True)
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("model", choices=("geoclip",))
    benchmark.add_argument("--image", type=Path, required=True)
    benchmark.add_argument("--runs", type=int, default=5)
    benchmark.add_argument("--warmup-runs", type=int, default=1)
    benchmark.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    register = commands.add_parser("register-local")
    register.add_argument("--manifest", type=Path, required=True)
    unregister = commands.add_parser("unregister")
    unregister.add_argument("model")
    unregister.add_argument("--yes", action="store_true")
    promote = commands.add_parser("promote")
    promote.add_argument("model")
    promote.add_argument(
        "--from", dest="from_mode", choices=tuple(DeploymentMode), required=True
    )
    promote.add_argument("--to", dest="to_mode", choices=tuple(DeploymentMode), required=True)
    promote.add_argument("--report", type=Path, required=True)
    return parser


def _context() -> InvocationContext:
    return InvocationContext(
        analysis_id=uuid4(),
        request_id="model-cli",
        mode=AnalysisMode.LOCAL_ONLY,
        cloud_consent=False,
        deadline=datetime.now(UTC) + timedelta(minutes=6),
        cancellation=asyncio.Event(),
    )


async def _predict_count(
    service: ModelManagementService, image: Path, requested_device: str
) -> ModelSmokeObservation:
    provider = GeoCLIPGlobalGeolocationProvider(
        service,
        device_selector=lambda: select_device(requested_device),
        timeout_seconds=300.0,
    )
    outcome = await provider.predict(
        LocalImageHandle(key="model-cli.upload", path=image), _context()
    )
    if outcome.status != OutcomeStatus.SUCCEEDED or outcome.value is None:
        raise ModelManagementError(
            outcome.failure.code if outcome.failure else "provider_test_failed",
            subreason_code=(
                outcome.failure.subreason_code if outcome.failure is not None else None
            ),
        )
    result = outcome.value
    return ModelSmokeObservation(
        hypothesis_count=len(result.hypotheses),
        provider_id=result.provider_id,
        model_revision=result.model_revision,
        implementation_revision=result.implementation_revision,
        device=result.device,
        score_type=result.hypotheses[0].score_type,
        calibration_state="uncalibrated",
        coordinate_validity="valid_wgs84",
    )


async def _predict_custom(
    manager: TrainedArtifactManager,
    model_id: str,
    image: Path,
    requested_device: str,
) -> dict[str, object]:
    if image.is_symlink() or not image.is_file() or image.stat().st_size > 50 * 1024 * 1024:
        raise TrainedArtifactError("image_unavailable")
    try:
        with Image.open(image) as source:
            width, height = source.size
            source.verify()
    except (OSError, ValueError) as exc:
        raise TrainedArtifactError("image_invalid") from exc
    selected = select_device(requested_device)
    device: Literal["cpu", "cuda"] = "cuda" if selected == "cuda" else "cpu"
    provider = build_custom_provider(
        manager,
        model_id,
        enabled=True,
        device=device,
        max_input_bytes=50 * 1024 * 1024,
    )
    outcome = await provider.infer(
        InferenceRequest(
            image_handle=LocalImageHandle(key="model-cli.upload", path=image),
            image_bytes=None,
            width=width,
            height=height,
            exif=None,
            analysis_mode=AnalysisMode.LOCAL_ONLY,
            top_k=5,
            cancellation=asyncio.Event(),
            deadline=datetime.now(UTC) + timedelta(minutes=6),
            trace_id="model-cli",
            max_input_bytes=50 * 1024 * 1024,
        )
    )
    if outcome.status != "succeeded":
        raise TrainedArtifactError(
            outcome.failure.code if outcome.failure is not None else "provider_test_failed"
        )
    return {
        "status": "passed",
        "provider_id": outcome.provider_id,
        "model_revision": outcome.model_revision,
        "device": outcome.device,
        "candidate_count": len(outcome.candidates),
        "score_type": outcome.score_semantics,
        "calibration_state": outcome.calibration_state,
        "coordinate_validity": "valid_wgs84",
    }


async def _benchmark(
    service: ModelManagementService,
    image: Path,
    *,
    runs: int,
    warmup_runs: int,
    requested_device: str,
) -> ModelBenchmarkResult:
    if not image.is_file():
        raise ModelManagementError("image_not_found")
    if runs < 1 or warmup_runs < 0:
        raise ModelManagementError("invalid_benchmark_configuration")
    provider = GeoCLIPGlobalGeolocationProvider(
        service,
        device_selector=lambda: select_device(requested_device),
        timeout_seconds=300.0,
    )
    handle = LocalImageHandle(key="model-cli.benchmark", path=image)
    device = select_device(requested_device)
    torch = None
    if device == "cuda":
        import importlib

        torch = importlib.import_module("torch")
        torch.cuda.reset_peak_memory_stats()

    cold_started = time.perf_counter()
    cold = await provider.predict(handle, _context())
    if cold.status != OutcomeStatus.SUCCEEDED:
        raise ModelManagementError(cold.failure.code if cold.failure else "provider_test_failed")
    cold_elapsed_ms = round((time.perf_counter() - cold_started) * 1000)
    load_duration = provider.diagnostics()["load_duration_ms"]
    if not isinstance(load_duration, int):
        raise ModelManagementError("invalid_provider_diagnostics")
    cold_load_ms = load_duration

    for _ in range(warmup_runs):
        warmup = await provider.predict(handle, _context())
        if warmup.status != OutcomeStatus.SUCCEEDED:
            raise ModelManagementError(
                warmup.failure.code if warmup.failure else "provider_test_failed"
            )

    durations: list[int] = []
    for _ in range(runs):
        started = time.perf_counter()
        outcome = await provider.predict(handle, _context())
        durations.append(round((time.perf_counter() - started) * 1000))
        if outcome.status != OutcomeStatus.SUCCEEDED:
            raise ModelManagementError(
                outcome.failure.code if outcome.failure else "provider_test_failed"
            )
    ordered = sorted(durations)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    peak_gpu = int(torch.cuda.max_memory_allocated()) if torch is not None else 0
    return ModelBenchmarkResult(
        status="completed",
        attempted=runs,
        succeeded=runs,
        durations_ms=tuple(durations),
        runs=runs,
        warmup_runs=warmup_runs,
        cold_load_ms=cold_load_ms or cold_elapsed_ms,
        warm_median_ms=float(statistics.median(durations)),
        warm_p95_ms=float(p95),
        device=device,
        peak_gpu_allocated_bytes=peak_gpu,
        model_revision=service.manifest.model_version,
    )


def run_models_command(argv: Sequence[str]) -> dict[str, object]:
    args = build_models_parser().parse_args(argv)
    service = ModelManagementService(args.cache_root, load_manifest(default_manifest_path()))
    embedding_service = EmbeddingModelManagementService(
        args.cache_root, load_embedding_manifest(siglip2_manifest_path())
    )
    rapidocr_service = RapidOCRModelManagementService(args.cache_root)
    trained_manager = TrainedArtifactManager(args.cache_root)
    if args.command == "list":
        return {
            "models": [
                service.info().model_dump(mode="json"),
                embedding_service.info().model_dump(mode="json"),
                rapidocr_service.info().model_dump(mode="json"),
                *[item.model_dump(mode="json") for item in trained_manager.list()],
            ]
        }
    if args.command == "register-local":
        trained_receipt = trained_manager.register_local(args.manifest)
        return {
            "status": "registered",
            "model_id": trained_receipt.model_id,
            "model_version": trained_receipt.model_version,
            "artifact_identity": trained_receipt.artifact_identity,
            "mode": trained_receipt.mode.value,
        }
    if args.command == "unregister":
        trained_manager.unregister(args.model, confirmed=args.yes)
        return {"status": "unregistered", "model_id": args.model}
    if args.command == "promote":
        promotion = trained_manager.promote(
            args.model,
            from_mode=DeploymentMode(args.from_mode),
            to_mode=DeploymentMode(args.to_mode),
            report_path=args.report,
        )
        return {
            "status": "promoted",
            "model_id": promotion.model_id,
            "from": promotion.from_mode.value,
            "to": promotion.to_mode.value,
            "evaluation_fingerprint": promotion.evaluation_fingerprint,
        }
    if args.model not in _MANAGED_MODELS:
        if args.command == "verify":
            trained_manager.verify(args.model)
            return trained_manager.info(args.model).model_dump(mode="json")
        if args.command == "info":
            return trained_manager.info(args.model).model_dump(mode="json")
        if args.command == "test":
            return run_isolated(
                _predict_custom(
                    trained_manager,
                    args.model,
                    args.image,
                    args.device,
                )
            )
        raise TrainedArtifactError("unsupported_custom_model_command")
    if args.model == "rapidocr":
        if args.command == "install":
            return rapidocr_service.install().model_dump(mode="json")
        if args.command == "verify":
            return rapidocr_service.verify().model_dump(mode="json")
        if args.command == "info":
            return rapidocr_service.info().model_dump(mode="json")
        if args.command == "remove":
            return rapidocr_service.remove(confirmed=args.yes).model_dump(mode="json")
        raise ModelManagementError("unsupported_rapidocr_command")
    selected = service if args.model == "geoclip" else embedding_service
    if args.command == "install":
        install_receipt = selected.install()
        resolved_revision = getattr(
            install_receipt,
            "resolved_clip_revision",
            getattr(install_receipt, "resolved_revision", None),
        )
        return {
            "status": "installed",
            "model_id": install_receipt.model_id,
            "model_version": install_receipt.model_version,
            "resolved_revision": resolved_revision,
        }
    if args.command == "verify":
        selected.verify()
        return selected.info(verify=True).model_dump(mode="json")
    if args.command == "info":
        return selected.info().model_dump(mode="json")
    if args.command == "remove":
        selected.remove(confirmed=args.yes)
        return {"status": "removed", "model_id": args.model}
    if args.command == "test":
        result = service.test(
            args.image,
            lambda image: run_isolated(_predict_count(service, image, args.device)),
        )
        return result.model_dump(mode="json")
    benchmark_result = run_isolated(
        _benchmark(
            service,
            args.image,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            requested_device=args.device,
        )
    )
    return benchmark_result.model_dump(mode="json")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run_models_command(sys.argv[1:] if argv is None else argv)
    except (ModelManagementError, TrainedArtifactError) as exc:
        print(json.dumps({"status": "error", "code": exc.code}), file=sys.stderr)
        return 2
    except (OSError, ValueError):
        print(
            json.dumps({"status": "error", "code": "model_operation_failed"}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
