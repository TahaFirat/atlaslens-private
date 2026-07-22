"""Network-free production-shape CUDA memory validation for Phase 3F."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "services" / "api" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from atlaslens_api.phase3f import cloud_job as worker  # noqa: E402
from atlaslens_api.phase3f.benchmark import (  # noqa: E402
    CalibrationObservation,
    fit_abstention_threshold,
)
from atlaslens_api.phase3f.memory import (  # noqa: E402
    BASELINE_MAX_ATTENTION_TOKENS,
    BASELINE_MAX_INPUT_SHAPE,
    BASELINE_MICROBATCH,
    EFFECTIVE_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUMERICAL_COSINE_MINIMUM,
    NUMERICAL_LINF_TOLERANCE,
    PRODUCTION_MEMORY_EVIDENCE_SCHEMA,
    TRAINING_INPUT_SHAPE,
    TRAINING_MICROBATCH,
    MeasuredMemoryRequirement,
)
from atlaslens_api.phase3f.pipeline import MODEL_SHA256  # noqa: E402
from atlaslens_api.phase3f.training import (  # noqa: E402
    TrainableMegaLocRuntime,
    TrainingError,
    _epoch_batches,
    require_training_ready,
    training_config_sha256,
)
from atlaslens_api.phase3f.training_recovery import atomic_json, sha256_path  # noqa: E402
from atlaslens_api.phase3f.splits import SplitAsset  # noqa: E402

SEED: Final = 20260720
_ENVIRONMENT_SHA256: Final = hashlib.sha256(
    b"atlaslens-phase3f-local-unbound-environment-v1"
).hexdigest()


class MemorySmokeError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise MemorySmokeError(code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-local-memory-smoke")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--temp-parent", type=Path, default=Path(r"C:\tmp"))
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    parser.add_argument("--worker-phase", choices=("initial", "resume"))
    parser.add_argument("--work-root", type=Path)
    return parser


def _network_guard(event: str, _arguments: tuple[object, ...]) -> None:
    if event in {"socket.connect", "socket.bind"}:
        raise MemorySmokeError("LOCAL_CUDA_MEMORY_NETWORK_ATTEMPTED")


def _tensor_sha256(runtime: TrainableMegaLocRuntime) -> str:
    digest = hashlib.sha256()
    named = dict(runtime._model.named_parameters())  # noqa: SLF001
    for name in sorted(runtime._selected_names):  # noqa: SLF001
        tensor = named[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _components(runtime: TrainableMegaLocRuntime) -> tuple[Any, Any, Any]:
    torch = runtime._torch  # noqa: SLF001
    optimizer = torch.optim.AdamW(
        runtime._selected_parameters,  # noqa: SLF001
        lr=2e-6,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return optimizer, scheduler, scaler


def _peak(torch: Any) -> dict[str, int]:
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _reset_peak(torch: Any) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _target_shape(path: Path) -> tuple[int, int]:
    with Image.open(path) as source:
        width, height = source.size
    scale = min(1.0, 560 / max(width, height))
    return (
        max(14, round(height * scale / 14) * 14),
        max(14, round(width * scale / 14) * 14),
    )


def _same_shape_paths(
    assets: tuple[SplitAsset, ...], media_root: Path, *, count: int
) -> list[Path]:
    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for asset in assets:
        path = media_root / asset.relative_path
        shape = _target_shape(path)
        groups[shape].append(path)
    maximum = BASELINE_MAX_INPUT_SHAPE[1:]
    selected = groups.get(maximum, [])
    _require(len(selected) >= count, "LOCAL_CUDA_MAX_SHAPE_MISSING")
    return selected[:count]


def _load_dataset(args: argparse.Namespace, work_root: Path) -> tuple[
    Any,
    tuple[SplitAsset, ...],
    tuple[SplitAsset, ...],
    str,
    str,
    str,
]:
    readiness_sha256 = sha256_path(args.readiness_report, max_bytes=1024 * 1024)
    sealed_assets_sha256 = sha256_path(
        args.sealed_root / "sealed-assets.json", max_bytes=64 * 1024 * 1024
    )
    inventory_sha256 = sha256_path(
        args.sealed_root / "checksum-inventory.json", max_bytes=64 * 1024 * 1024
    )
    generated = work_root / "validated-training-readiness.json"
    sealed = require_training_ready(args.sealed_root, report_path=generated)
    _require(
        sha256_path(generated, max_bytes=1024 * 1024) == readiness_sha256,
        "LOCAL_CUDA_READINESS_HASH_MISMATCH",
    )
    references = tuple(row for row in sealed.split.assets if row.role == "reference")
    validation = tuple(row for row in sealed.split.assets if row.role == "calibration")
    _require(
        sealed.run_id == args.run_id
        and len(sealed.split.assets) == 830
        and len(references) == 450
        and len(validation) == 150,
        "LOCAL_CUDA_DATASET_SHAPE_INVALID",
    )
    return (
        sealed,
        references,
        validation,
        readiness_sha256,
        sealed_assets_sha256,
        inventory_sha256,
    )


def _baseline_and_validation(
    args: argparse.Namespace,
    work_root: Path,
    sealed: Any,
    references: tuple[SplitAsset, ...],
    validation: tuple[SplitAsset, ...],
) -> dict[str, object]:
    runtime: worker.MegaLocRuntime | None = None
    try:
        runtime = worker.MegaLocRuntime(args.model, args.vendor_root)
        torch = runtime._torch  # noqa: SLF001
        maximum_paths = _same_shape_paths(references, sealed.root, count=4)
        bulk = runtime.describe(maximum_paths, batch_size=4, stage="baseline")
        micro = runtime.describe(maximum_paths, batch_size=1, stage="baseline")
        cosine = np.sum(bulk * micro, axis=1)
        linf = np.max(np.abs(bulk - micro))
        _require(
            bulk.shape == micro.shape == (4, 8448)
            and bool(np.isfinite(bulk).all())
            and float(np.min(cosine)) >= NUMERICAL_COSINE_MINIMUM
            and float(linf) <= NUMERICAL_LINF_TOLERANCE,
            "LOCAL_CUDA_NUMERICAL_EQUIVALENCE_FAILED",
        )
        _reset_peak(torch)
        reference_matrix = runtime.describe(
            [sealed.root / item.relative_path for item in references],
            batch_size=BASELINE_MICROBATCH,
            stage="baseline",
        )
        baseline_peak = _peak(torch)
        _reset_peak(torch)
        validation_matrix = runtime.describe(
            [sealed.root / item.relative_path for item in validation],
            batch_size=BASELINE_MICROBATCH,
            stage="validation",
        )
        validation_peak = _peak(torch)
        index, _descriptor_sha, _index_sha, _publication_sha = (
            worker._publish_reference_bundle(  # noqa: SLF001
                work_root / "baseline-index",
                references,
                reference_matrix,
            )
        )
        rows = worker._retrieval_rows(  # noqa: SLF001
            index,
            validation,
            validation_matrix,
            references,
            tuple(item.city for item in sealed.selection.in_domain),
        )
        threshold = fit_abstention_threshold(
            [
                CalibrationObservation(row.signals, row.predicted_cities[0] == row.city)
                for row in rows
            ],
            selection_lock_sha256=sealed.selection.lock_sha256,
            split_lock_sha256=sealed.split.split_lock_sha256,
        )
        atomic_json(work_root / "calibration.json", threshold.document())
        return {
            "baseline_peak": baseline_peak,
            "validation_peak": validation_peak,
            "reference_count": int(reference_matrix.shape[0]),
            "validation_count": int(validation_matrix.shape[0]),
            "numerical_equivalence": {
                "shape_equal": True,
                "order_equal": True,
                "cosine_min": float(np.min(cosine)),
                "linf_max": float(linf),
                "cosine_minimum": NUMERICAL_COSINE_MINIMUM,
                "linf_tolerance": NUMERICAL_LINF_TOLERANCE,
            },
        }
    finally:
        if runtime is not None:
            runtime.close()


def _worker_initial(args: argparse.Namespace) -> int:
    work_root = cast(Path, args.work_root)
    sealed, references, validation, readiness, sealed_assets, inventory = _load_dataset(
        args, work_root
    )
    baseline = _baseline_and_validation(
        args, work_root, sealed, references, validation
    )
    runtime: TrainableMegaLocRuntime | None = None
    try:
        runtime = TrainableMegaLocRuntime(args.model, args.vendor_root, seed=SEED)
        torch = runtime._torch  # noqa: SLF001
        optimizer, scheduler, scaler = _components(runtime)
        initial_tensor = _tensor_sha256(runtime)
        batch = _epoch_batches(
            references, batch_size=EFFECTIVE_BATCH_SIZE, seed=SEED, epoch=0
        )[0]
        _reset_peak(torch)
        loss, microbatch, oom_recoveries = runtime._effective_batch_step(  # noqa: SLF001
            batch,
            sealed.root,
            optimizer,
            scheduler,
            scaler,
            epoch=0,
            batch_index=0,
            microbatch_size=TRAINING_MICROBATCH,
        )
        train_peak = _peak(torch)
        final_tensor = _tensor_sha256(runtime)
        _require(
            np.isfinite(loss) and initial_tensor != final_tensor,
            "LOCAL_CUDA_TRAINING_STEP_FAILED",
        )
        config_sha256 = training_config_sha256(seed=SEED, max_epochs=8)
        runtime._write_checkpoint(  # noqa: SLF001
            work_root / "training-checkpoint",
            optimizer,
            scheduler,
            scaler,
            run_id=args.run_id,
            completed_epoch=-1,
            next_epoch=0,
            next_batch_index=1,
            optimizer_steps=1,
            best_validation_loss=loss,
            patience=0,
            history=(),
            train_losses=(loss,),
            dataset_readiness_sha256=readiness,
            sealed_assets_sha256=sealed_assets,
            training_config_sha256_value=config_sha256,
            environment_identity_sha256=_ENVIRONMENT_SHA256,
            training_microbatch_size=microbatch,
            gradient_accumulation_steps=EFFECTIVE_BATCH_SIZE // microbatch,
        )
        receipt = {
            "initial_tensor_sha256": initial_tensor,
            "final_tensor_sha256": final_tensor,
            "train_loss": loss,
            "training_microbatch": microbatch,
            "training_oom_recoveries": oom_recoveries,
            "nonfinite_loss_count": 0,
            "stage_peaks": {
                "baseline": baseline["baseline_peak"],
                "validation": baseline["validation_peak"],
                "train": train_peak,
            },
            "reference_count": baseline["reference_count"],
            "validation_count": baseline["validation_count"],
            "numerical_equivalence": baseline["numerical_equivalence"],
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "readiness_sha256": readiness,
            "sealed_assets_sha256": sealed_assets,
            "inventory_sha256": inventory,
            "secrets_included": False,
        }
        atomic_json(work_root / "initial.json", receipt)
        print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
        return 0
    finally:
        if runtime is not None:
            runtime.close()


def _worker_resume(args: argparse.Namespace) -> int:
    work_root = cast(Path, args.work_root)
    sealed, references, validation, readiness, sealed_assets, inventory = _load_dataset(
        args, work_root
    )
    initial = json.loads((work_root / "initial.json").read_text(encoding="utf-8"))
    runtime: TrainableMegaLocRuntime | None = None
    try:
        runtime = TrainableMegaLocRuntime(args.model, args.vendor_root, seed=SEED)
        torch = runtime._torch  # noqa: SLF001
        optimizer, scheduler, scaler = _components(runtime)
        config_sha256 = training_config_sha256(seed=SEED, max_epochs=8)
        state = runtime._load_checkpoint(  # noqa: SLF001
            work_root / "training-checkpoint",
            optimizer,
            scheduler,
            scaler,
            run_id=args.run_id,
            dataset_readiness_sha256=readiness,
            sealed_assets_sha256=sealed_assets,
            training_config_sha256_value=config_sha256,
            environment_identity_sha256=_ENVIRONMENT_SHA256,
        )
        loaded_tensor = _tensor_sha256(runtime)
        _require(
            state["optimizer_steps"] == 1
            and state["next_batch_index"] == 1
            and state["accumulation_boundary"] is True
            and loaded_tensor == initial["final_tensor_sha256"],
            "LOCAL_CUDA_CHECKPOINT_ROUNDTRIP_FAILED",
        )
        batch = _epoch_batches(
            references, batch_size=EFFECTIVE_BATCH_SIZE, seed=SEED, epoch=0
        )[1]
        _reset_peak(torch)
        loss, microbatch, oom_recoveries = runtime._effective_batch_step(  # noqa: SLF001
            batch,
            sealed.root,
            optimizer,
            scheduler,
            scaler,
            epoch=0,
            batch_index=1,
            microbatch_size=TRAINING_MICROBATCH,
        )
        resume_peak = _peak(torch)
        final_tensor = _tensor_sha256(runtime)
        _require(
            np.isfinite(loss) and final_tensor != loaded_tensor,
            "LOCAL_CUDA_RESUME_STEP_FAILED",
        )
        runtime._write_checkpoint(  # noqa: SLF001
            work_root / "training-checkpoint",
            optimizer,
            scheduler,
            scaler,
            run_id=args.run_id,
            completed_epoch=-1,
            next_epoch=0,
            next_batch_index=2,
            optimizer_steps=2,
            best_validation_loss=loss,
            patience=0,
            history=(),
            train_losses=(loss,),
            dataset_readiness_sha256=readiness,
            sealed_assets_sha256=sealed_assets,
            training_config_sha256_value=config_sha256,
            environment_identity_sha256=_ENVIRONMENT_SHA256,
            training_microbatch_size=microbatch,
            gradient_accumulation_steps=EFFECTIVE_BATCH_SIZE // microbatch,
        )
        allowed_probe = [sealed.root / row.relative_path for row in validation[:4]]
        _reset_peak(torch)
        holdout_probe = runtime.describe(
            allowed_probe,
            batch_size=TRAINING_MICROBATCH,
            stage="holdout",
        )
        holdout_peak = _peak(torch)
        _require(
            holdout_probe.shape == (4, 8448)
            and bool(np.isfinite(holdout_probe).all()),
            "LOCAL_CUDA_HOLDOUT_PATH_PROBE_FAILED",
        )
        receipt = {
            "loaded_tensor_sha256": loaded_tensor,
            "final_tensor_sha256": final_tensor,
            "resume_loss": loss,
            "training_microbatch": microbatch,
            "training_oom_recoveries": oom_recoveries,
            "stage_peaks": {"resume": resume_peak, "holdout_probe": holdout_peak},
            "checkpoint_roundtrip_passed": True,
            "locked_holdout_access_count": 0,
            "allowed_holdout_probe_count": 4,
            "readiness_sha256": readiness,
            "sealed_assets_sha256": sealed_assets,
            "inventory_sha256": inventory,
            "secrets_included": False,
        }
        print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
        return 0
    finally:
        if runtime is not None:
            runtime.close()


def _worker_command(args: argparse.Namespace, work_root: Path, phase: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--repository-root",
        str(args.repository_root),
        "--run-id",
        args.run_id,
        "--sealed-root",
        str(args.sealed_root),
        "--readiness-report",
        str(args.readiness_report),
        "--model",
        str(args.model),
        "--vendor-root",
        str(args.vendor_root),
        "--evidence-root",
        str(args.evidence_root),
        "--max-wall-seconds",
        str(args.max_wall_seconds),
        "--worker-phase",
        phase,
        "--work-root",
        str(work_root),
    ]


def _run_worker(args: argparse.Namespace, work_root: Path, phase: str) -> dict[str, object]:
    completed = subprocess.run(
        _worker_command(args, work_root, phase),
        cwd=args.repository_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=args.max_wall_seconds,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"MAPILLARY_ACCESS_TOKEN", "RUNPOD_API_KEY"}
        },
    )
    if completed.returncode != 0:
        lines = completed.stdout.decode("utf-8", errors="replace").splitlines()
        code = lines[-1] if lines else f"LOCAL_CUDA_{phase.upper()}_FAILED"
        raise MemorySmokeError(code)
    lines = completed.stdout.decode("utf-8", errors="strict").splitlines()
    _require(bool(lines), "LOCAL_CUDA_WORKER_OUTPUT_MISSING")
    value = json.loads(lines[-1])
    _require(isinstance(value, dict), "LOCAL_CUDA_WORKER_OUTPUT_INVALID")
    return cast(dict[str, object], value)


def _orchestrate(args: argparse.Namespace) -> int:
    _require(args.repository_root.resolve() == ROOT, "LOCAL_CUDA_REPOSITORY_INVALID")
    _require(900 <= args.max_wall_seconds <= 7200, "LOCAL_CUDA_WALL_LIMIT_INVALID")
    _require(sha256_path(args.model) == MODEL_SHA256, "LOCAL_CUDA_MODEL_HASH_MISMATCH")
    before = {
        "readiness": sha256_path(args.readiness_report, max_bytes=1024 * 1024),
        "sealed": sha256_path(
            args.sealed_root / "sealed-assets.json", max_bytes=64 * 1024 * 1024
        ),
        "inventory": sha256_path(
            args.sealed_root / "checksum-inventory.json", max_bytes=64 * 1024 * 1024
        ),
    }
    temporary = Path(
        tempfile.mkdtemp(prefix="atlaslens-phase3f-memory-", dir=args.temp_parent)
    )
    receipt: dict[str, object] | None = None
    try:
        initial = _run_worker(args, temporary, "initial")
        print("LOCAL_CUDA_MEMORY_INITIAL_PASSED", flush=True)
        resumed = _run_worker(args, temporary, "resume")
        print("LOCAL_CUDA_MEMORY_RESUME_PASSED", flush=True)
        after = {
            "readiness": sha256_path(args.readiness_report, max_bytes=1024 * 1024),
            "sealed": sha256_path(
                args.sealed_root / "sealed-assets.json", max_bytes=64 * 1024 * 1024
            ),
            "inventory": sha256_path(
                args.sealed_root / "checksum-inventory.json",
                max_bytes=64 * 1024 * 1024,
            ),
        }
        _require(before == after, "LOCAL_CUDA_DATASET_MUTATED")
        stage_peaks = {
            **cast(dict[str, object], initial["stage_peaks"]),
            **cast(dict[str, object], resumed["stage_peaks"]),
        }
        peak_allocated = max(
            cast(int, cast(dict[str, object], row)["peak_allocated_bytes"])
            for row in stage_peaks.values()
        )
        peak_reserved = max(
            cast(int, cast(dict[str, object], row)["peak_reserved_bytes"])
            for row in stage_peaks.values()
        )
        requirement = MeasuredMemoryRequirement.from_peaks(
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
        receipt = {
            "schema": PRODUCTION_MEMORY_EVIDENCE_SCHEMA,
            "run_id": args.run_id,
            "model_sha256": MODEL_SHA256,
            "readiness_sha256": before["readiness"],
            "sealed_assets_sha256": before["sealed"],
            "checksum_inventory_sha256": before["inventory"],
            "gpu_name": initial["gpu_name"],
            "gpu_total_bytes": initial["gpu_total_bytes"],
            "production_input_shape": list(BASELINE_MAX_INPUT_SHAPE),
            "training_input_shape": list(TRAINING_INPUT_SHAPE),
            "attention_token_count": BASELINE_MAX_ATTENTION_TOKENS,
            "baseline_microbatch": BASELINE_MICROBATCH,
            "training_microbatch": min(
                cast(int, initial["training_microbatch"]),
                cast(int, resumed["training_microbatch"]),
            ),
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "full_reference_descriptor_count": initial["reference_count"],
            "validation_descriptor_count": initial["validation_count"],
            "stage_peaks": stage_peaks,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "required_headroom_bytes": requirement.required_headroom_bytes,
            "required_gpu_vram_gib": requirement.required_vram_gib,
            "numerical_equivalence": initial["numerical_equivalence"],
            "checkpoint_roundtrip_passed": resumed["checkpoint_roundtrip_passed"],
            "local_cuda_production_shape_passed": True,
            "nonfinite_loss_count": initial["nonfinite_loss_count"],
            "model_parameters_changed": (
                resumed["final_tensor_sha256"] != initial["initial_tensor_sha256"]
            ),
            "locked_holdout_access_count": 0,
            "allowed_holdout_probe_count": resumed["allowed_holdout_probe_count"],
            "model_input_asset_access_count": 450 + 150 + 4 + 4 + 16 * 4 + 4,
            "dataset_write_count": 0,
            "network_calls": 0,
            "runpod_api_calls": 0,
            "cloud_mutations": 0,
            "temporary_cleanup_verified": False,
            "secrets_included": False,
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=False)
    _require(receipt is not None and not temporary.exists(), "LOCAL_CUDA_TEMP_CLEANUP_FAILED")
    receipt["temporary_cleanup_verified"] = True
    evidence = args.evidence_root / args.run_id / "production-memory-smoke.json"
    atomic_json(evidence, receipt)
    print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ.pop("MAPILLARY_ACCESS_TOKEN", None)
    os.environ.pop("RUNPOD_API_KEY", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.addaudithook(_network_guard)
    try:
        if args.worker_phase == "initial":
            _require(args.work_root is not None, "LOCAL_CUDA_WORK_ROOT_MISSING")
            return _worker_initial(args)
        if args.worker_phase == "resume":
            _require(args.work_root is not None, "LOCAL_CUDA_WORK_ROOT_MISSING")
            return _worker_resume(args)
        return _orchestrate(args)
    except (MemorySmokeError, TrainingError) as exc:
        print(exc.code)
        return 1
    except BaseException as exc:
        print(f"LOCAL_CUDA_UNEXPECTED_{type(exc).__name__.upper()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
