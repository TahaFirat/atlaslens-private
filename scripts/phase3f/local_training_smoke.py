"""Isolated, network-free real-CUDA retry validation for Phase 3F training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Final, cast

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "services" / "api" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from atlaslens_api.phase3f import cloud_job as worker  # noqa: E402
from atlaslens_api.phase3f.benchmark import (  # noqa: E402
    CalibrationObservation,
    fit_abstention_threshold,
)
from atlaslens_api.phase3f.pipeline import MODEL_SHA256  # noqa: E402
from atlaslens_api.phase3f.training import (  # noqa: E402
    TrainableMegaLocRuntime,
    TrainingError,
    _batch_metric_labels,
    _epoch_batches,
    _paired_metric_groups,
    require_training_ready,
    training_config_sha256,
)
from atlaslens_api.phase3f.training_recovery import (  # noqa: E402
    REMOTE_FAILURE_SCHEMA,
    TrainingRecoveryError,
    atomic_json,
    classify_remote_failure,
    extract_recovery_archive,
    sanitized_tail,
    sha256_path,
    store_verified_checkpoint_archive,
)
from atlaslens_api.phase3f.splits import SplitAsset  # noqa: E402

SMOKE_SCHEMA: Final = "atlaslens-phase3f-local-cuda-smoke-v1"
WORKER_SCHEMA: Final = "atlaslens-phase3f-local-cuda-smoke-worker-v1"
SEED: Final = 20260720
INITIAL_OPTIMIZER_STEPS: Final = 12
RESUME_OPTIMIZER_STEPS: Final = 3
BATCH_SIZE: Final = 4
GRADIENT_ACCUMULATION: Final = 4
MINIMUM_TEMP_FREE_BYTES: Final = 12 * 1024**3
_RUN_ID = set("0123456789abcdef")
_CHECKPOINT_GENERATION = re.compile(r"^generation-[0-9]{8}-[0-9a-f]{16}$")


class SmokeError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SmokeError(code if code.startswith("SMOKE_") else f"SMOKE_{code}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-local-training-smoke")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--temp-parent", type=Path, default=Path(r"C:\tmp"))
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    parser.add_argument("--worker-phase", choices=("initial", "resume", "failure"))
    parser.add_argument("--work-root", type=Path)
    return parser


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _tensor_sha256(runtime: TrainableMegaLocRuntime) -> str:
    digest = hashlib.sha256()
    named = dict(runtime._model.named_parameters())  # noqa: SLF001
    for name in sorted(runtime._selected_names):  # noqa: SLF001
        tensor = named[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _training_components(runtime: TrainableMegaLocRuntime) -> tuple[Any, Any, Any]:
    torch = runtime._torch  # noqa: SLF001
    optimizer = torch.optim.AdamW(
        runtime._selected_parameters,  # noqa: SLF001
        lr=2e-6,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return optimizer, scheduler, scaler


def _subset_by_groups(
    assets: tuple[SplitAsset, ...],
    *,
    group_limit: int,
    rows_per_group: int,
) -> tuple[SplitAsset, ...]:
    groups = _paired_metric_groups(assets)
    chosen: list[SplitAsset] = []
    for _key, rows in sorted(groups.items())[:group_limit]:
        chosen.extend(sorted(rows, key=lambda item: item.opaque_id)[:rows_per_group])
    _require(len(_paired_metric_groups(chosen)) >= 2, "SMOKE_GROUP_DIVERSITY_INSUFFICIENT")
    return tuple(chosen)


def _load_smoke_dataset(args: argparse.Namespace, work_root: Path) -> tuple[
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
    generated = work_root / "validated-training-readiness.json"
    sealed = require_training_ready(args.sealed_root, report_path=generated)
    _require(sha256_path(generated) == readiness_sha256, "SMOKE_READINESS_HASH_MISMATCH")
    _require(sealed.run_id == args.run_id, "SMOKE_DATASET_RUN_ID_MISMATCH")
    _require(len(sealed.split.assets) == 830, "SMOKE_ASSET_COUNT_MISMATCH")
    references = tuple(asset for asset in sealed.split.assets if asset.role == "reference")
    validation = tuple(asset for asset in sealed.split.assets if asset.role == "calibration")
    _require(
        len(references) == 450
        and len(validation) == 150
        and all(asset.role == "reference" for asset in references)
        and all(asset.role == "calibration" for asset in validation),
        "SMOKE_ROLE_INVENTORY_INVALID",
    )
    config_sha256 = training_config_sha256(seed=SEED, max_epochs=8)
    return (
        sealed,
        _subset_by_groups(references, group_limit=12, rows_per_group=4),
        _subset_by_groups(validation, group_limit=8, rows_per_group=4),
        readiness_sha256,
        sealed_assets_sha256,
        config_sha256,
    )


def _optimizer_steps(
    runtime: TrainableMegaLocRuntime,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    assets: tuple[SplitAsset, ...],
    media_root: Path,
    *,
    epoch: int,
    target_steps: int,
    deadline: float,
) -> tuple[int, list[float], int]:
    torch = runtime._torch  # noqa: SLF001
    batches = _epoch_batches(assets, batch_size=BATCH_SIZE, seed=SEED, epoch=epoch)
    micro_batches = target_steps * GRADIENT_ACCUMULATION
    selected = tuple(islice(cycle(batches), micro_batches))
    runtime._model.train()  # noqa: SLF001
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    nonfinite = 0
    steps = 0
    for batch_index, batch_assets in enumerate(selected):
        _require(time.monotonic() < deadline, "SMOKE_WALL_LIMIT_REACHED")
        _require(
            all(asset.role in {"reference", "calibration"} for asset in batch_assets),
            "SMOKE_LOCKED_HOLDOUT_ACCESS_ATTEMPTED",
        )
        labels_by_asset = _batch_metric_labels(batch_assets)
        tensors = [
            runtime._tensor(  # noqa: SLF001
                media_root / asset.relative_path,
                training=True,
                salt=epoch * 1_000_000 + batch_index * 100 + index,
            )
            for index, asset in enumerate(batch_assets)
        ]
        batch = torch.stack(tensors).to("cuda", non_blocking=True)
        labels = torch.tensor(
            [labels_by_asset[asset.opaque_id] for asset in batch_assets],
            dtype=torch.long,
            device="cuda",
        )
        with torch.amp.autocast("cuda", dtype=torch.float16):
            embeddings = runtime._model(batch)  # noqa: SLF001
            loss = runtime._metric_loss(embeddings, labels)  # noqa: SLF001
            scaled_loss = loss / GRADIENT_ACCUMULATION
        if not bool(torch.isfinite(loss).item()):
            nonfinite += 1
            raise SmokeError("SMOKE_NONFINITE_LOSS")
        scaler.scale(scaled_loss).backward()
        losses.append(float(loss.detach().cpu()))
        if (batch_index + 1) % GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(runtime._selected_parameters, 1.0)  # noqa: SLF001
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    return steps, losses, nonfinite


def _validation_loss(
    runtime: TrainableMegaLocRuntime,
    assets: tuple[SplitAsset, ...],
    media_root: Path,
) -> tuple[float, int]:
    torch = runtime._torch  # noqa: SLF001
    batches = _epoch_batches(assets, batch_size=BATCH_SIZE, seed=SEED, epoch=0)
    losses: list[float] = []
    runtime._model.eval()  # noqa: SLF001
    with torch.inference_mode():
        for batch_index, batch_assets in enumerate(batches):
            labels_by_asset = _batch_metric_labels(batch_assets)
            tensors = [
                runtime._tensor(  # noqa: SLF001
                    media_root / asset.relative_path,
                    training=False,
                    salt=batch_index * 100 + index,
                )
                for index, asset in enumerate(batch_assets)
            ]
            batch = torch.stack(tensors).to("cuda", non_blocking=True)
            labels = torch.tensor(
                [labels_by_asset[asset.opaque_id] for asset in batch_assets],
                dtype=torch.long,
                device="cuda",
            )
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = runtime._metric_loss(runtime._model(batch), labels)  # noqa: SLF001
            _require(bool(torch.isfinite(loss).item()), "SMOKE_VALIDATION_NONFINITE")
            losses.append(float(loss.cpu()))
    _require(bool(losses), "SMOKE_VALIDATION_EMPTY")
    return sum(losses) / len(losses), len(losses)


def _rng_probe(runtime: TrainableMegaLocRuntime) -> dict[str, float]:
    torch = runtime._torch  # noqa: SLF001
    return {
        "python": random.random(),  # noqa: S311 - deterministic state probe
        "numpy": float(cast(float, __import__("numpy").random.random())),
        "torch_cpu": float(torch.rand(1).item()),
        "torch_cuda": float(torch.rand(1, device="cuda").item()),
    }


def _worker_initial(args: argparse.Namespace) -> int:
    work_root = cast(Path, args.work_root)
    (
        sealed,
        references,
        validation,
        readiness_sha256,
        sealed_assets_sha256,
        config_sha256,
    ) = _load_smoke_dataset(args, work_root)
    checkpoint_root = work_root / "training-checkpoint"
    output_root = work_root / "initial-output"
    output_root.mkdir(mode=0o700)
    runtime: TrainableMegaLocRuntime | None = None
    try:
        runtime = TrainableMegaLocRuntime(args.model, args.vendor_root, seed=SEED)
        torch = runtime._torch  # noqa: SLF001
        torch.cuda.reset_peak_memory_stats()
        optimizer, scheduler, scaler = _training_components(runtime)
        initial_tensor_sha256 = _tensor_sha256(runtime)
        steps, losses, nonfinite = _optimizer_steps(
            runtime,
            optimizer,
            scheduler,
            scaler,
            references,
            sealed.root,
            epoch=0,
            target_steps=INITIAL_OPTIMIZER_STEPS,
            deadline=time.monotonic() + args.max_wall_seconds,
        )
        validation_loss, validation_batches = _validation_loss(
            runtime, validation, sealed.root
        )
        final_tensor_sha256 = _tensor_sha256(runtime)
        _require(steps == INITIAL_OPTIMIZER_STEPS, "SMOKE_INITIAL_STEP_COUNT_INVALID")
        _require(initial_tensor_sha256 != final_tensor_sha256, "SMOKE_PARAMETERS_UNCHANGED")
        runtime._write_checkpoint(  # noqa: SLF001
            checkpoint_root,
            optimizer,
            scheduler,
            scaler,
            run_id=args.run_id,
            completed_epoch=0,
            next_epoch=1,
            next_batch_index=0,
            optimizer_steps=steps,
            best_validation_loss=validation_loss,
            patience=0,
            history=(
                {
                    "epoch": 0,
                    "train_loss": sum(losses) / len(losses),
                    "validation_loss": validation_loss,
                    "improved": True,
                },
            ),
            train_losses=(),
            dataset_readiness_sha256=readiness_sha256,
            sealed_assets_sha256=sealed_assets_sha256,
            training_config_sha256_value=config_sha256,
        )
        rng_probe = _rng_probe(runtime)
        pointer = json.loads((checkpoint_root / "latest.json").read_text(encoding="utf-8"))
        generation = cast(str, pointer["generation"])
        checkpoint_manifest_sha256 = sha256_path(
            checkpoint_root / generation / "checkpoint-manifest.json"
        )
        receipt = {
            "schema": WORKER_SCHEMA,
            "phase": "initial",
            "gpu_name": torch.cuda.get_device_name(0),
            "optimizer_steps": steps,
            "validation_batches": validation_batches,
            "validation_loss": validation_loss,
            "nonfinite_loss_count": nonfinite,
            "initial_tensor_sha256": initial_tensor_sha256,
            "final_tensor_sha256": final_tensor_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "rng_probe_after_checkpoint": rng_probe,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "locked_holdout_access_count": 0,
            "secrets_included": False,
        }
        atomic_json(output_root / "initial-receipt.json", receipt)
        print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
        return 0
    finally:
        if runtime is not None:
            runtime.close()


def _worker_resume(args: argparse.Namespace) -> int:
    work_root = cast(Path, args.work_root)
    (
        sealed,
        references,
        validation,
        readiness_sha256,
        sealed_assets_sha256,
        config_sha256,
    ) = _load_smoke_dataset(args, work_root)
    checkpoint_root = work_root / "training-checkpoint"
    output_root = work_root / "final-output"
    output_root.mkdir(mode=0o700)
    initial = json.loads(
        (work_root / "initial-output" / "initial-receipt.json").read_text(encoding="utf-8")
    )
    runtime: TrainableMegaLocRuntime | None = None
    try:
        runtime = TrainableMegaLocRuntime(args.model, args.vendor_root, seed=SEED)
        torch = runtime._torch  # noqa: SLF001
        torch.cuda.reset_peak_memory_stats()
        optimizer, scheduler, scaler = _training_components(runtime)
        state = runtime._load_checkpoint(  # noqa: SLF001
            checkpoint_root,
            optimizer,
            scheduler,
            scaler,
            run_id=args.run_id,
            dataset_readiness_sha256=readiness_sha256,
            sealed_assets_sha256=sealed_assets_sha256,
            training_config_sha256_value=config_sha256,
        )
        _require(state["next_epoch"] == 1, "SMOKE_RESUME_EPOCH_INVALID")
        _require(
            state["optimizer_steps"] == INITIAL_OPTIMIZER_STEPS,
            "SMOKE_RESUME_STEP_INVALID",
        )
        loaded_tensor_sha256 = _tensor_sha256(runtime)
        _require(
            loaded_tensor_sha256 == initial["final_tensor_sha256"],
            "SMOKE_RESUME_MODEL_STATE_MISMATCH",
        )
        restored_rng_probe = _rng_probe(runtime)
        _require(
            restored_rng_probe == initial["rng_probe_after_checkpoint"],
            "SMOKE_RESUME_RNG_STATE_MISMATCH",
        )
        steps, losses, nonfinite = _optimizer_steps(
            runtime,
            optimizer,
            scheduler,
            scaler,
            references,
            sealed.root,
            epoch=1,
            target_steps=RESUME_OPTIMIZER_STEPS,
            deadline=time.monotonic() + args.max_wall_seconds,
        )
        total_steps = INITIAL_OPTIMIZER_STEPS + steps
        runtime._write_checkpoint(  # noqa: SLF001
            checkpoint_root,
            optimizer,
            scheduler,
            scaler,
            run_id=args.run_id,
            completed_epoch=0,
            next_epoch=1,
            next_batch_index=steps * GRADIENT_ACCUMULATION,
            optimizer_steps=total_steps,
            best_validation_loss=float(state["best_validation_loss"]),
            patience=int(state["patience"]),
            history=cast(list[dict[str, object]], state["history"]),
            train_losses=losses,
            dataset_readiness_sha256=readiness_sha256,
            sealed_assets_sha256=sealed_assets_sha256,
            training_config_sha256_value=config_sha256,
        )
        reference_matrix = runtime.describe(
            [sealed.root / item.relative_path for item in references[:16]],
            batch_size=BATCH_SIZE,
        )
        validation_matrix = runtime.describe(
            [sealed.root / item.relative_path for item in validation[:8]],
            batch_size=BATCH_SIZE,
        )
        index, descriptor_sha, index_sha, publication_sha = worker._publish_reference_bundle(  # noqa: SLF001
            output_root,
            references[:16],
            reference_matrix,
        )
        rows = worker._retrieval_rows(  # noqa: SLF001
            index,
            validation[:8],
            validation_matrix,
            references[:16],
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
        atomic_json(output_root / "calibration.json", threshold.document())
        final_model_sha256 = runtime.save_final(output_root / "megaloc-smoke.safetensors")
        atomic_json(
            output_root / "smoke-provenance.json",
            {
                "schema": "atlaslens-phase3f-local-smoke-provenance-v1",
                "run_id": args.run_id,
                "optimizer_steps": total_steps,
                "locked_holdout_access_count": 0,
                "production_configuration_changed": False,
                "secrets_included": False,
            },
        )
        inventory = worker._inventory_output(output_root)  # noqa: SLF001
        inventory_sha256 = atomic_json(output_root / "checksum-inventory.json", inventory)
        final_tensor_sha256 = _tensor_sha256(runtime)
        receipt = {
            "schema": WORKER_SCHEMA,
            "phase": "resume",
            "gpu_name": torch.cuda.get_device_name(0),
            "restored_epoch": state["next_epoch"],
            "restored_optimizer_steps": state["optimizer_steps"],
            "restored_model_tensor_sha256": loaded_tensor_sha256,
            "restored_rng": True,
            "additional_optimizer_steps": steps,
            "total_optimizer_steps": total_steps,
            "nonfinite_loss_count": nonfinite,
            "model_parameters_changed": final_tensor_sha256 != initial["initial_tensor_sha256"],
            "final_tensor_sha256": final_tensor_sha256,
            "final_model_sha256": final_model_sha256,
            "descriptor_sha256": descriptor_sha,
            "index_sha256": index_sha,
            "publication_sha256": publication_sha,
            "threshold_sha256": sha256_path(output_root / "calibration.json"),
            "inventory_sha256": inventory_sha256,
            "inventory_file_count": len(cast(list[object], inventory["files"])),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "locked_holdout_access_count": 0,
            "secrets_included": False,
        }
        atomic_json(work_root / "resume-worker-receipt.json", receipt)
        print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
        return 0
    finally:
        if runtime is not None:
            runtime.close()


def _worker_failure(args: argparse.Namespace) -> int:
    work_root = cast(Path, args.work_root)
    recovery = work_root / "failure-recovery"
    recovery.mkdir(mode=0o700)
    atomic_json(
        recovery / "progress.json",
        {
            "schema": "atlaslens-phase3f-remote-training-progress-v1",
            "stage": "training",
            "completed_epoch": 0,
            "epoch": 1,
            "next_batch_index": 12,
            "optimizer_steps": 15,
            "holdout_open_count": 0,
            "secrets_included": False,
        },
    )
    (recovery / "metrics.jsonl").write_bytes(
        _canonical_bytes(
            {
                "schema": "atlaslens-phase3f-training-metric-v1",
                "epoch": 0,
                "optimizer_steps": 12,
                "holdout_open_count": 0,
                "secrets_included": False,
            }
        )
    )
    try:
        raise TrainingError("TRAINING_NONFINITE_LOSS")
    except TrainingError as exc:
        code = exc.code
        atomic_json(
            recovery / "child-failure.json",
            {
                "schema": "atlaslens-phase3f-training-child-failure-v1",
                "exception_class": type(exc).__name__,
                "message_code": code,
                "traceback_tail": list(
                    sanitized_tail("".join(traceback.format_exception(exc)))
                ),
                "secrets_included": False,
            },
        )
        print("controlled failure stdout", flush=True)
        print(
            "api_key=smoke-only-secret-value-123456789 198.51.100.9",
            file=sys.stderr,
            flush=True,
        )
        print(code, flush=True)
        return 17


def _command(args: argparse.Namespace, work_root: Path, phase: str) -> list[str]:
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
        _command(args, work_root, phase),
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
        tails = (
            completed.stdout.decode("utf-8", errors="replace").splitlines()
            + completed.stderr.decode("utf-8", errors="replace").splitlines()
        )
        code = next(
            (
                line.strip()
                for line in reversed(tails)
                if re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", line.strip())
            ),
            f"SMOKE_{phase.upper()}_WORKER_FAILED",
        )
        raise SmokeError(code)
    lines = completed.stdout.decode("utf-8", errors="strict").splitlines()
    _require(bool(lines), "SMOKE_WORKER_OUTPUT_MISSING")
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SmokeError("SMOKE_WORKER_OUTPUT_INVALID") from exc
    _require(isinstance(value, dict), "SMOKE_WORKER_OUTPUT_INVALID")
    return cast(dict[str, object], value)


def _tar_tree(source: Path, destination: Path) -> None:
    with tarfile.open(destination, "w:") as archive:
        for path in sorted(source.rglob("*")):
            archive.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)


def _tar_latest_checkpoint(source: Path, destination: Path) -> None:
    pointer_path = source / "latest.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeError("SMOKE_CHECKPOINT_POINTER_INVALID") from exc
    generation = pointer.get("generation") if isinstance(pointer, dict) else None
    _require(
        isinstance(generation, str)
        and bool(_CHECKPOINT_GENERATION.fullmatch(generation)),
        "SMOKE_CHECKPOINT_POINTER_INVALID",
    )
    generation_root = source / cast(str, generation)
    _require(generation_root.is_dir(), "SMOKE_CHECKPOINT_GENERATION_MISSING")
    with tarfile.open(destination, "w:") as archive:
        archive.add(pointer_path, arcname="latest.json", recursive=False)
        archive.add(generation_root, arcname=cast(str, generation), recursive=True)


def _failure_salvage(
    args: argparse.Namespace,
    work_root: Path,
    *,
    readiness_sha256: str,
    sealed_assets_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    completed = subprocess.run(
        _command(args, work_root, "failure"),
        cwd=args.repository_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    explicit = next(
        (line for line in reversed(stdout.splitlines()) if line == "TRAINING_NONFINITE_LOSS"),
        None,
    )
    diagnostic = classify_remote_failure(
        completed.returncode,
        stdout=stdout,
        stderr=stderr,
        explicit_code=explicit,
        exception_class="TrainingError",
    )
    _require(
        completed.returncode == 17
        and diagnostic.failure_code == "REMOTE_TRAINING_NONFINITE_LOSS",
        "SMOKE_FAILURE_CLASSIFICATION_INVALID",
    )
    recovery = work_root / "failure-recovery"
    child = json.loads((recovery / "child-failure.json").read_text(encoding="utf-8"))
    failure = {
        "schema": REMOTE_FAILURE_SCHEMA,
        **diagnostic.to_dict(),
        "traceback_tail": child["traceback_tail"],
        "last_completed_epoch": 0,
        "last_epoch": 1,
        "last_step": 15,
        "holdout_open_count": 0,
        "secrets_included": False,
    }
    atomic_json(recovery / "failure.json", failure)
    recovery_archive = work_root / "failure-recovery.tar"
    _tar_tree(recovery, recovery_archive)
    recovered_files = extract_recovery_archive(
        recovery_archive, work_root / "salvaged-recovery"
    )
    checkpoint_archive = work_root / "failure-checkpoint.tar"
    _tar_latest_checkpoint(work_root / "training-checkpoint", checkpoint_archive)
    checkpoint = store_verified_checkpoint_archive(
        checkpoint_archive,
        work_root / "salvaged-checkpoint-store",
        expected_run_id=args.run_id,
        expected_readiness_sha256=readiness_sha256,
        expected_sealed_assets_sha256=sealed_assets_sha256,
        expected_training_config_sha256=config_sha256,
    )
    serialized = json.dumps(failure, separators=(",", ":"), sort_keys=True)
    _require(
        "smoke-only-secret" not in serialized
        and "198.51.100.9" not in serialized
        and checkpoint.valid,
        "SMOKE_FAILURE_SALVAGE_INVALID",
    )
    return {
        "failure_code": diagnostic.failure_code,
        "process_exit_code": diagnostic.process_exit_code,
        "process_signal": diagnostic.process_signal,
        "exception_class": diagnostic.exception_class,
        "stdout_tail": list(diagnostic.stdout_tail),
        "stderr_tail": list(diagnostic.stderr_tail),
        "artifact_count": len(recovered_files),
        "progress_present": any(path.name == "progress.json" for path in recovered_files),
        "metrics_present": any(path.name == "metrics.jsonl" for path in recovered_files),
        "checkpoint_valid": checkpoint.valid,
        "checkpoint_sha256": checkpoint.archive_sha256,
        "checkpoint_optimizer_steps": checkpoint.optimizer_steps,
        "secret_redaction_passed": True,
    }


def _network_guard(event: str, _arguments: tuple[object, ...]) -> None:
    if event in {"socket.connect", "socket.bind"}:
        raise SmokeError("SMOKE_NETWORK_ATTEMPTED")


def _orchestrate(args: argparse.Namespace) -> int:
    _require(args.repository_root.resolve() == ROOT, "SMOKE_REPOSITORY_ROOT_INVALID")
    _require(
        len(args.run_id) == 32 and all(character in _RUN_ID for character in args.run_id),
        "RUN_ID_INVALID",
    )
    _require(900 <= args.max_wall_seconds <= 7200, "SMOKE_WALL_LIMIT_INVALID")
    _require(sha256_path(args.model) == MODEL_SHA256, "SMOKE_MODEL_HASH_MISMATCH")
    before = {
        "readiness_sha256": sha256_path(args.readiness_report, max_bytes=1024 * 1024),
        "sealed_assets_sha256": sha256_path(
            args.sealed_root / "sealed-assets.json", max_bytes=64 * 1024 * 1024
        ),
        "inventory_sha256": sha256_path(
            args.sealed_root / "checksum-inventory.json", max_bytes=64 * 1024 * 1024
        ),
    }
    args.temp_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require(
        shutil.disk_usage(args.temp_parent).free >= MINIMUM_TEMP_FREE_BYTES,
        "SMOKE_TEMP_SPACE_INSUFFICIENT",
    )
    temporary = Path(
        tempfile.mkdtemp(prefix="atlaslens-phase3f-cuda-smoke-", dir=args.temp_parent)
    )
    evidence_path = args.evidence_root / args.run_id / "retry-smoke.json"
    completed = False
    receipt: dict[str, object] | None = None
    try:
        initial = _run_worker(args, temporary, "initial")
        print("SMOKE_INITIAL_PASSED", flush=True)
        resumed = _run_worker(args, temporary, "resume")
        print("SMOKE_RESUME_PASSED", flush=True)
        config_sha256 = training_config_sha256(seed=SEED, max_epochs=8)
        salvage = _failure_salvage(
            args,
            temporary,
            readiness_sha256=before["readiness_sha256"],
            sealed_assets_sha256=before["sealed_assets_sha256"],
            config_sha256=config_sha256,
        )
        print("SMOKE_FAILURE_SALVAGE_PASSED", flush=True)
        after = {
            "readiness_sha256": sha256_path(
                args.readiness_report, max_bytes=1024 * 1024
            ),
            "sealed_assets_sha256": sha256_path(
                args.sealed_root / "sealed-assets.json", max_bytes=64 * 1024 * 1024
            ),
            "inventory_sha256": sha256_path(
                args.sealed_root / "checksum-inventory.json", max_bytes=64 * 1024 * 1024
            ),
        }
        _require(before == after, "SMOKE_DATASET_MUTATED")
        receipt = {
            "schema": SMOKE_SCHEMA,
            "run_id": args.run_id,
            "model_sha256": MODEL_SHA256,
            "readiness_sha256": before["readiness_sha256"],
            "sealed_assets_sha256": before["sealed_assets_sha256"],
            "checksum_inventory_sha256": before["inventory_sha256"],
            "training_config_sha256": config_sha256,
            "gpu_name": initial["gpu_name"],
            "local_cuda_smoke_passed": True,
            "mini_epoch_passed": True,
            "checkpoint_roundtrip_passed": (
                resumed["restored_optimizer_steps"] == INITIAL_OPTIMIZER_STEPS
                and resumed["restored_rng"] is True
            ),
            "failure_salvage_passed": (
                salvage["checkpoint_valid"] is True
                and salvage["secret_redaction_passed"] is True
            ),
            "initial_optimizer_steps": initial["optimizer_steps"],
            "additional_optimizer_steps": resumed["additional_optimizer_steps"],
            "total_optimizer_steps": resumed["total_optimizer_steps"],
            "validation_batches": initial["validation_batches"],
            "nonfinite_loss_count": cast(int, initial["nonfinite_loss_count"])
            + cast(int, resumed["nonfinite_loss_count"]),
            "model_parameters_changed": resumed["model_parameters_changed"],
            "initial_model_tensor_sha256": initial["initial_tensor_sha256"],
            "final_model_tensor_sha256": resumed["final_tensor_sha256"],
            "final_artifact_sha256": resumed["final_model_sha256"],
            "checkpoint_manifest_sha256": initial["checkpoint_manifest_sha256"],
            "salvaged_checkpoint_sha256": salvage["checkpoint_sha256"],
            "salvaged_checkpoint_optimizer_steps": salvage[
                "checkpoint_optimizer_steps"
            ],
            "failure_diagnostic": salvage,
            "threshold_sha256": resumed["threshold_sha256"],
            "output_inventory_sha256": resumed["inventory_sha256"],
            "output_inventory_file_count": resumed["inventory_file_count"],
            "peak_cuda_bytes": max(
                cast(int, initial["peak_cuda_bytes"]),
                cast(int, resumed["peak_cuda_bytes"]),
            ),
            "temporary_peak_bytes": _tree_bytes(temporary),
            "locked_holdout_access_count": 0,
            "network_calls": 0,
            "runpod_api_calls": 0,
            "mapillary_api_calls": 0,
            "cloud_mutations": 0,
            "production_training_state_advanced": False,
            "production_configuration_changed": False,
            "temporary_cleanup_verified": False,
            "secrets_included": False,
        }
        completed = True
    finally:
        shutil.rmtree(temporary, ignore_errors=False)
    _require(completed and not temporary.exists(), "SMOKE_TEMP_CLEANUP_FAILED")
    _require(receipt is not None, "SMOKE_RECEIPT_MISSING")
    receipt["temporary_cleanup_verified"] = True
    atomic_json(evidence_path, receipt)
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
            _require(args.work_root is not None, "SMOKE_WORK_ROOT_MISSING")
            return _worker_initial(args)
        if args.worker_phase == "resume":
            _require(args.work_root is not None, "SMOKE_WORK_ROOT_MISSING")
            return _worker_resume(args)
        if args.worker_phase == "failure":
            _require(args.work_root is not None, "SMOKE_WORK_ROOT_MISSING")
            return _worker_failure(args)
        return _orchestrate(args)
    except (SmokeError, TrainingError, TrainingRecoveryError) as exc:
        print(exc.code)
        return 1
    except BaseException as exc:
        exception_name = re.sub(r"[^A-Za-z0-9]", "_", type(exc).__name__).upper()
        print(f"SMOKE_UNEXPECTED_{exception_name}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
