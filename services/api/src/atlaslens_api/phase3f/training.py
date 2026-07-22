"""Fail-closed Phase 3F dataset handoff and real MegaLoc fine-tuning.

The locked holdout is described exactly once, after training and threshold lock.
Mapillary credentials are neither required nor accepted by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast
from uuid import uuid4

import numpy as np
from PIL import Image

from atlaslens_api.phase3f import cloud_job as worker
from atlaslens_api.phase3f.acquisition import AcquisitionGuard
from atlaslens_api.phase3f.benchmark import (
    CalibrationObservation,
    RetrievalObservation,
    evaluate_holdout_once,
    fit_abstention_threshold,
)
from atlaslens_api.phase3f.local_first import VerifiedAcquisition, verify_sealed_acquisition
from atlaslens_api.phase3f.pipeline import (
    MODEL_SHA256,
    DescriptorPublication,
    Phase3FPipeline,
)
from atlaslens_api.phase3f.splits import SplitAsset
from atlaslens_api.phase3f.training_recovery import (
    CHECKPOINT_MANIFEST_SCHEMA,
    CHECKPOINT_POINTER_SCHEMA,
    TrainingRecoveryError,
    validate_checkpoint_tree,
)

TRAINING_READINESS_SCHEMA: Final = "atlaslens-phase3f-training-readiness-v1"
TRAINING_SPLIT_SCHEMA: Final = "atlaslens-phase3f-training-split-v1"
TRAINING_RECEIPT_SCHEMA: Final = "atlaslens-phase3f-megaloc-training-v1"
DATASET_ARCHIVE_SCHEMA: Final = "atlaslens-phase3f-dataset-archive-v1"
MAX_DATASET_ARCHIVE_BYTES: Final = 8 * 1024 * 1024 * 1024
MIN_USABLE_ASSETS: Final = 830
MAX_CONCENTRATION: Final = 0.35
DEFAULT_MAX_EPOCHS: Final = 8
DEFAULT_TRAINING_WALL_SECONDS: Final = 4 * 60 * 60 + 30 * 60
CHECKPOINT_INTERVAL_SECONDS: Final = 5 * 60
CHECKPOINT_INTERVAL_STEPS: Final = 10
_SHA256 = set("0123456789abcdef")
_SECRET_MARKERS: Final = (
    b"RUNPOD_API_KEY",
    b"MAPILLARY_ACCESS_TOKEN",
    b"access_token=",
    b"authorization: bearer",
    b"authorization: oauth",
    b"MLY|",
    b"mly|",
)


class TrainingError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DatasetNotReadyError(TrainingError):
    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__("DATASET_NOT_READY_FOR_TRAINING")
        self.reasons = tuple(sorted(set(reasons)))


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise TrainingError(code)


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
    ).encode()


def _sha256_path(path: Path, *, max_bytes: int | None = None) -> str:
    _require(path.is_file() and not path.is_symlink(), "TRAINING_ARTIFACT_INVALID")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(chunk)
            _require(max_bytes is None or total <= max_bytes, "TRAINING_ARTIFACT_TOO_LARGE")
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> str:
    payload = _canonical_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def training_config_sha256(
    *,
    seed: int,
    max_epochs: int,
    environment_contract_sha256: str | None = None,
) -> str:
    """Hash resume-relevant training semantics without machine-adaptive batch size."""
    if environment_contract_sha256 is not None:
        _require(
            bool(re.fullmatch(r"[0-9a-f]{64}", environment_contract_sha256)),
            "TRAINING_ENVIRONMENT_CONTRACT_INVALID",
        )
    document: dict[str, object] = {
        "schema": "atlaslens-phase3f-training-config-v1",
        "seed": seed,
        "maximum_epochs": max_epochs,
        "objective": "batch_hard_metric_learning_softplus_cosine_v1",
        "optimizer": "adamw",
        "learning_rate": "0.000002",
        "weight_decay": "0.0001",
        "scheduler": "constant",
        "mixed_precision": True,
        "checkpoint_interval_seconds": CHECKPOINT_INTERVAL_SECONDS,
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL_STEPS,
    }
    if environment_contract_sha256 is not None:
        document["environment_contract_sha256"] = environment_contract_sha256
    return hashlib.sha256(
        _canonical_bytes(document)
    ).hexdigest()


def _nested_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _maximum_share(values: Sequence[str]) -> float:
    counts = Counter(values)
    return max(counts.values(), default=0) / len(values) if values else 1.0


def _scan_private_json(root: Path) -> tuple[int, int]:
    secret_findings = 0
    private_path_findings = 0
    for path in sorted(root.rglob("*.json")):
        _require(path.is_file() and not path.is_symlink(), "DATASET_JSON_INVALID")
        _require(path.stat().st_size <= 64 * 1024 * 1024, "DATASET_JSON_TOO_LARGE")
        payload = path.read_bytes()
        lowered = payload.lower()
        secret_findings += sum(marker.lower() in lowered for marker in _SECRET_MARKERS)
        private_path_findings += sum(
            marker in lowered
            for marker in (b"c:\\\\users\\\\", b"d:\\\\atlaslensruntime", b"/home/", b"/workspace/")
        )
    return secret_findings, private_path_findings


def training_readiness_document(sealed: VerifiedAcquisition) -> dict[str, object]:
    assets = sealed.split.assets
    role_counts = Counter(asset.role for asset in assets)
    city_counts = Counter(asset.city for asset in assets)
    sequence_share = _maximum_share([asset.sequence_id for asset in assets])
    contributor_share = _maximum_share([asset.contributor_id for asset in assets])
    secret_findings, path_findings = _scan_private_json(sealed.root)
    inventory_file_count = sum(
        path.is_file() and path.name != "checksum-inventory.json"
        for path in sealed.root.rglob("*")
    )
    provenance_count = sealed.provenance.get("asset_count")
    training_positive_groups = len(
        _paired_metric_groups(
            [asset for asset in assets if asset.role == "reference"]
        )
    )
    validation_positive_groups = len(
        _paired_metric_groups(
            [asset for asset in assets if asset.role == "calibration"]
        )
    )
    exact_duplicate_count = len(assets) - len({asset.content_sha256 for asset in assets})
    near_duplicate_pairs = 0
    for index, left in enumerate(assets):
        left_hash = int(left.perceptual_hash, 16)
        for right in assets[index + 1 :]:
            if (left_hash ^ int(right.perceptual_hash, 16)).bit_count() <= 4:
                near_duplicate_pairs += 1
    near_duplicate_ratio = near_duplicate_pairs / len(assets) if assets else 1.0
    reasons: list[str] = []
    checks = {
        "multi_region_coverage": len(sealed.selection.in_domain) >= 5
        and len({item.macro_region for item in sealed.selection.in_domain}) >= 4,
        "city_distribution": len(city_counts) >= 7 and min(city_counts.values(), default=0) >= 40,
        "minimum_usable_assets": len(assets) >= MIN_USABLE_ASSETS,
        "duplicate_invariants": sealed.split.leakage.passed
        and exact_duplicate_count == 0
        and near_duplicate_ratio <= 0.02,
        "sequence_concentration": sequence_share <= MAX_CONCENTRATION,
        "contributor_concentration": contributor_share <= MAX_CONCENTRATION,
        "metric_training_groups": training_positive_groups >= 2,
        "metric_validation_groups": validation_positive_groups >= 2,
        "rights_and_provenance": provenance_count == len(assets),
        "checksum_inventory": inventory_file_count >= len(assets),
        "secret_scan": secret_findings == 0,
        "private_path_scan": path_findings == 0,
        "image_label_binding": all(
            (sealed.root / asset.relative_path).is_file() for asset in assets
        ),
    }
    reasons.extend(name.upper() for name, passed in checks.items() if not passed)
    ready = not reasons
    return {
        "schema": TRAINING_READINESS_SCHEMA,
        "outcome": "READY_FOR_TRAINING" if ready else "DATASET_NOT_READY_FOR_TRAINING",
        "ready": ready,
        "asset_count": len(assets),
        "city_count": len(city_counts),
        "macro_region_count": len(
            {item.macro_region for item in sealed.selection.in_domain}
        ),
        "role_counts": dict(sorted(role_counts.items())),
        "maximum_sequence_share": sequence_share,
        "maximum_contributor_share": contributor_share,
        "metric_training_positive_groups": training_positive_groups,
        "metric_validation_positive_groups": validation_positive_groups,
        "exact_duplicate_count": exact_duplicate_count,
        "perceptual_near_duplicate_pair_count": near_duplicate_pairs,
        "perceptual_near_duplicate_ratio": near_duplicate_ratio,
        "secret_findings": secret_findings,
        "private_path_findings": path_findings,
        "checks": checks,
        "reason_codes": reasons,
        "gpu_started": False,
        "cloud_mutations": 0,
        "raw_paths_included": False,
        "secrets_included": False,
    }


def require_training_ready(sealed_root: Path, *, report_path: Path) -> VerifiedAcquisition:
    sealed = verify_sealed_acquisition(sealed_root)
    report = training_readiness_document(sealed)
    _atomic_json(report_path, report)
    if report["ready"] is not True:
        raise DatasetNotReadyError(cast(list[str], report["reason_codes"]))
    return sealed


@dataclass(frozen=True, slots=True)
class DatasetArchive:
    path: Path
    sha256: str
    size_bytes: int
    file_count: int
    readiness_sha256: str

    def receipt(self) -> dict[str, object]:
        return {
            "schema": DATASET_ARCHIVE_SCHEMA,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "readiness_sha256": self.readiness_sha256,
            "contains_private_corpus": True,
            "mapillary_token_included": False,
            "signed_urls_included": False,
            "secrets_included": False,
        }


def write_dataset_archive(
    sealed_root: Path,
    destination: Path,
    *,
    readiness_report_path: Path,
) -> DatasetArchive:
    sealed = require_training_ready(sealed_root, report_path=readiness_report_path)
    resolved = sealed.root.resolve()
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    _require(
        bool(files) and all(not path.is_symlink() for path in files),
        "DATASET_TREE_INVALID",
    )
    total = sum(path.stat().st_size for path in files)
    _require(0 < total <= MAX_DATASET_ARCHIVE_BYTES, "DATASET_ARCHIVE_CAP_EXCEEDED")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require(not destination.exists() and not destination.is_symlink(), "DATASET_ARCHIVE_EXISTS")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                relative = path.relative_to(resolved)
                _require(".." not in relative.parts, "DATASET_ARCHIVE_PATH_INVALID")
                info = archive.gettarinfo(
                    str(path),
                    arcname=str(PurePosixPath("sealed-acquisition", *relative.parts)),
                )
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.mode = 0o600
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
        _require(
            temporary.stat().st_size <= MAX_DATASET_ARCHIVE_BYTES,
            "DATASET_ARCHIVE_CAP_EXCEEDED",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    readiness_sha = _sha256_path(readiness_report_path, max_bytes=1024 * 1024)
    result = DatasetArchive(
        path=destination,
        sha256=_sha256_path(destination, max_bytes=MAX_DATASET_ARCHIVE_BYTES),
        size_bytes=destination.stat().st_size,
        file_count=len(files),
        readiness_sha256=readiness_sha,
    )
    _atomic_json(destination.with_suffix(".json"), result.receipt())
    return result


def training_split_document(sealed: VerifiedAcquisition) -> dict[str, object]:
    role_names = {
        "reference": "train",
        "calibration": "validation",
        "sealed_holdout": "locked_holdout",
        "ood_holdout": "locked_ood_holdout",
    }
    rows = [
        {
            "opaque_id": asset.opaque_id,
            "split": role_names[asset.role],
            "content_sha256": asset.content_sha256,
        }
        for asset in sealed.split.assets
    ]
    payload: dict[str, object] = {
        "schema": TRAINING_SPLIT_SCHEMA,
        "version": 1,
        "immutable": True,
        "split_lock_sha256": sealed.split.split_lock_sha256,
        "holdout_seal_sha256": sealed.split.holdout_seal_sha256,
        "ground_truth_included": False,
        "filename_labels_included": False,
        "source_context_included": False,
        "assets": rows,
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(quantile * len(ordered)) - 1)])


def _evaluation_summary(rows: Sequence[RetrievalObservation]) -> dict[str, object]:
    errors = [row.geodesic_error_km for row in rows if row.geodesic_error_km is not None]
    per_city: dict[str, dict[str, object]] = {}
    for city in sorted({row.city for row in rows}):
        city_rows = [row for row in rows if row.city == city]
        per_city[city] = {
            "count": len(city_rows),
            "city_top1": sum(row.predicted_cities[0] == city for row in city_rows)
            / len(city_rows),
            "recall_at_1": sum(row.relevant_reference_rank == 1 for row in city_rows)
            / len(city_rows),
        }
    similarities = [row.signals.similarity for row in rows]
    threshold = float(np.quantile(np.asarray(similarities, dtype=np.float64), 0.25))
    accepted = [row for row in rows if row.signals.similarity >= threshold]
    return {
        "query_count": len(rows),
        "recall_at_1": sum(
            row.relevant_reference_rank is not None and row.relevant_reference_rank <= 1
            for row in rows
        )
        / len(rows),
        "recall_at_5": sum(
            row.relevant_reference_rank is not None and row.relevant_reference_rank <= 5
            for row in rows
        )
        / len(rows),
        "recall_at_10": sum(
            row.relevant_reference_rank is not None and row.relevant_reference_rank <= 10
            for row in rows
        )
        / len(rows),
        "city_top1": sum(row.predicted_cities[0] == row.city for row in rows) / len(rows),
        "median_geodesic_error_km": float(np.median(errors)) if errors else None,
        "p90_geodesic_error_km": _percentile(errors, 0.90),
        "p95_geodesic_error_km": _percentile(errors, 0.95),
        "diagnostic_abstention_threshold": threshold,
        "diagnostic_coverage": len(accepted) / len(rows),
        "diagnostic_abstention_rate": 1.0 - len(accepted) / len(rows),
        "per_city": per_city,
        "evaluation_split": "validation_only",
        "locked_holdout_opened": False,
        "confidence": None,
        "similarity_semantics": "uncalibrated_cosine_similarity_not_probability",
    }


def _regressed(baseline: Mapping[str, object], candidate: Mapping[str, object]) -> bool:
    for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "city_top1"):
        if float(cast(float, candidate[metric])) + 1e-12 < float(cast(float, baseline[metric])):
            return True
    baseline_error = baseline.get("median_geodesic_error_km")
    candidate_error = candidate.get("median_geodesic_error_km")
    return bool(
        isinstance(baseline_error, int | float)
        and isinstance(candidate_error, int | float)
        and float(candidate_error) > float(baseline_error) + 1e-12
    )


def _epoch_batches(
    assets: Sequence[SplitAsset], *, batch_size: int, seed: int, epoch: int
) -> tuple[tuple[SplitAsset, ...], ...]:
    _require(batch_size >= 4 and batch_size % 2 == 0, "TRAINING_BATCH_SIZE_INVALID")
    paired_groups = _paired_metric_groups(assets)
    _require(len(paired_groups) >= 2, "TRAINING_GROUP_DIVERSITY_INSUFFICIENT")
    generator = random.Random(seed + epoch)  # noqa: S311 - deterministic training sampler
    pairs: list[tuple[str, SplitAsset, SplitAsset]] = []
    for group_key, rows in sorted(paired_groups.items()):
        shuffled = list(rows)
        generator.shuffle(shuffled)
        if len(shuffled) < 2:
            continue
        if len(shuffled) % 2:
            shuffled.append(shuffled[0])
        pairs.extend(
            (group_key, shuffled[index], shuffled[index + 1])
            for index in range(0, len(shuffled), 2)
        )
    generator.shuffle(pairs)
    pair_count = batch_size // 2
    batches: list[tuple[SplitAsset, ...]] = []
    for offset in range(0, len(pairs), pair_count):
        group = pairs[offset : offset + pair_count]
        if len(group) < 2 or len({row[0] for row in group}) < 2:
            continue
        batches.append(tuple(item for _city, left, right in group for item in (left, right)))
    _require(bool(batches), "TRAINING_BATCHES_EMPTY")
    return tuple(batches)


def _paired_metric_groups(
    assets: Sequence[SplitAsset],
) -> dict[str, list[SplitAsset]]:
    by_sequence: dict[str, list[SplitAsset]] = defaultdict(list)
    for asset in assets:
        by_sequence[asset.sequence_id].append(asset)
    paired = {
        f"sequence:{key}": rows
        for key, rows in by_sequence.items()
        if len(rows) >= 2
    }
    if len(paired) >= 2:
        return paired
    by_spatial_cell: dict[str, list[SplitAsset]] = defaultdict(list)
    for asset in assets:
        key = (
            f"spatial:{asset.city}:"
            f"{math.floor(asset.latitude * 100)}:{math.floor(asset.longitude * 100)}"
        )
        by_spatial_cell[key].append(asset)
    return {
        key: rows for key, rows in by_spatial_cell.items() if len(rows) >= 2
    }


def _batch_metric_labels(assets: Sequence[SplitAsset]) -> dict[str, int]:
    groups = _paired_metric_groups(assets)
    labels: dict[str, int] = {}
    for label, key in enumerate(sorted(groups)):
        for asset in groups[key]:
            labels[asset.opaque_id] = label
    _require(len(labels) == len(assets), "METRIC_BATCH_INVALID")
    return labels


class TrainableMegaLocRuntime:
    """Pinned MegaLoc with a bounded trainable tail and real metric-learning steps."""

    def __init__(self, model_path: Path, vendor_root: Path, *, seed: int) -> None:
        worker.verify_megaloc_artifacts(
            model_path,
            vendor_root / "megaloc_model.py",
            vendor_root / "LICENSE",
        )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            import torchvision.transforms.functional as transform  # type: ignore[import-untyped]
            from safetensors.torch import load_file, save_file
        except ImportError as exc:
            raise TrainingError("MEGALOC_TRAINING_RUNTIME_MISSING") from exc
        _require(bool(torch.cuda.is_available()), "MEGALOC_CUDA_UNAVAILABLE")
        source = str(vendor_root.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            from megaloc_model import MegaLoc  # type: ignore[import-not-found]

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            model = MegaLoc()
            state = load_file(str(model_path), device="cpu")
            model.load_state_dict(state, strict=True)
            del state
            model.requires_grad_(False)
            selected: list[tuple[str, Any]] = []
            selected_parameters = 0
            for name, parameter in reversed(tuple(model.named_parameters())):
                parameter.requires_grad_(True)
                selected.append((name, parameter))
                selected_parameters += int(parameter.numel())
                if selected_parameters >= 1_000_000:
                    break
            _require(bool(selected), "MEGALOC_TRAINABLE_TAIL_EMPTY")
            initial_selected = {
                name: parameter.detach().float().cpu().clone().contiguous()
                for name, parameter in selected
            }
            model.to("cuda")
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError("MEGALOC_TRAINING_MODEL_LOAD_FAILED") from exc
        self._torch: Any = torch
        self._load_file = load_file
        self._save_file = save_file
        self._transform = transform
        self._model = model
        self._selected_names = tuple(name for name, _parameter in selected)
        self._selected_parameters = tuple(parameter for _name, parameter in selected)
        self._initial_selected = initial_selected
        self._seed = seed

    def _tensor(self, path: Path, *, training: bool, salt: int) -> Any:
        _require(path.is_file() and not path.is_symlink(), "TRAINING_IMAGE_INVALID")
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            image = self._transform.resize(image, [392, 392], antialias=True)
            if training:
                generator = random.Random(  # noqa: S311 - deterministic augmentation
                    self._seed + salt
                )
                if generator.random() < 0.5:
                    image = self._transform.hflip(image)
                image = self._transform.adjust_brightness(image, 0.9 + generator.random() * 0.2)
                image = self._transform.adjust_contrast(image, 0.9 + generator.random() * 0.2)
            tensor = self._transform.to_tensor(image)
            return self._transform.normalize(
                tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        except (OSError, ValueError) as exc:
            raise TrainingError("TRAINING_IMAGE_INVALID") from exc

    def describe(
        self, paths: Sequence[Path], *, batch_size: int
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        if not paths:
            return np.empty((0, 8448), dtype=np.float32)
        output = np.empty((len(paths), 8448), dtype=np.float32)
        self._model.eval()
        try:
            with self._torch.inference_mode():
                for offset in range(0, len(paths), batch_size):
                    batch_paths = paths[offset : offset + batch_size]
                    tensors = [
                        self._tensor(path, training=False, salt=offset + index)
                        for index, path in enumerate(batch_paths)
                    ]
                    batch = self._torch.stack(tensors).to("cuda", non_blocking=True)
                    with self._torch.amp.autocast("cuda", dtype=self._torch.float16):
                        matrix = self._model(batch)
                    output[offset : offset + len(batch_paths)] = matrix.float().cpu().numpy()
        except Exception as exc:
            if isinstance(exc, self._torch.cuda.OutOfMemoryError):
                raise
            raise TrainingError("MEGALOC_TRAINING_INFERENCE_FAILED") from exc
        _require(bool(np.isfinite(output).all()), "DESCRIPTOR_NONFINITE")
        norms = np.linalg.norm(output, axis=1)
        _require(bool(np.allclose(norms, 1.0, atol=1e-3, rtol=0.0)), "DESCRIPTOR_NORM_INVALID")
        return np.ascontiguousarray(output, dtype=np.float32)

    def _metric_loss(self, embeddings: Any, labels: Any) -> Any:
        embeddings = self._torch.nn.functional.normalize(embeddings.float(), dim=1)
        similarities = embeddings @ embeddings.T
        identity = self._torch.eye(len(labels), dtype=self._torch.bool, device=labels.device)
        positives = labels[:, None].eq(labels[None, :]) & ~identity
        negatives = ~labels[:, None].eq(labels[None, :])
        _require(
            bool(positives.any().item()) and bool(negatives.any().item()),
            "METRIC_BATCH_INVALID",
        )
        hardest_positive = similarities.masked_fill(~positives, float("inf")).min(dim=1).values
        hardest_negative = similarities.masked_fill(~negatives, float("-inf")).max(dim=1).values
        valid = self._torch.isfinite(hardest_positive) & self._torch.isfinite(hardest_negative)
        _require(bool(valid.any().item()), "METRIC_BATCH_INVALID")
        return self._torch.nn.functional.softplus(
            (hardest_negative[valid] - hardest_positive[valid]) / 0.07
        ).mean()

    def _load_checkpoint(
        self,
        checkpoint_root: Path,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        *,
        run_id: str,
        dataset_readiness_sha256: str,
        sealed_assets_sha256: str,
        training_config_sha256_value: str,
        environment_identity_sha256: str,
    ) -> dict[str, Any]:
        pointer_path = checkpoint_root / "latest.json"
        if not pointer_path.exists():
            return {
                "completed_epoch": -1,
                "next_epoch": 0,
                "next_batch_index": 0,
                "optimizer_steps": 0,
                "best_validation_loss": math.inf,
                "patience": 0,
                "history": [],
                "train_losses": [],
            }
        try:
            status = validate_checkpoint_tree(
                checkpoint_root,
                expected_run_id=run_id,
                expected_readiness_sha256=dataset_readiness_sha256,
                expected_sealed_assets_sha256=sealed_assets_sha256,
                expected_training_config_sha256=training_config_sha256_value,
                expected_environment_identity_sha256=environment_identity_sha256,
            )
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            generation_root = checkpoint_root / cast(str, pointer["generation"])
            state = json.loads(
                (generation_root / "training-state.json").read_text(encoding="utf-8")
            )
            _require(isinstance(state, dict), "TRAINING_CHECKPOINT_INVALID")
            delta = self._load_file(
                str(generation_root / "trainable.safetensors"), device="cpu"
            )
            named = dict(self._model.named_parameters())
            _require(set(delta) == set(self._selected_names), "TRAINING_CHECKPOINT_INVALID")
            with self._torch.no_grad():
                for name in self._selected_names:
                    named[name].copy_(delta[name].to("cuda"))
            optimizer.load_state_dict(
                self._torch.load(
                    generation_root / "optimizer.pt",
                    map_location="cuda",
                    weights_only=True,
                )
            )
            scheduler.load_state_dict(
                self._torch.load(
                    generation_root / "scheduler.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            scaler.load_state_dict(
                self._torch.load(
                    generation_root / "scaler.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            rng = self._torch.load(
                generation_root / "rng.pt", map_location="cpu", weights_only=True
            )
            _require(isinstance(rng, dict), "TRAINING_CHECKPOINT_INVALID")
            self._torch.set_rng_state(rng["torch_cpu"])
            if self._torch.cuda.is_available():
                self._torch.cuda.set_rng_state_all(rng["torch_cuda"])
            python_rng = state.get("python_rng_state")
            numpy_rng = state.get("numpy_rng_state")
            _require(
                isinstance(python_rng, list)
                and isinstance(numpy_rng, dict)
                and isinstance(numpy_rng.get("keys"), list),
                "TRAINING_CHECKPOINT_INVALID",
            )
            random.setstate(cast(tuple[Any, ...], _nested_tuple(python_rng)))
            np.random.set_state(
                (
                    cast(str, numpy_rng["algorithm"]),
                    np.asarray(numpy_rng["keys"], dtype=np.uint32),
                    cast(int, numpy_rng["position"]),
                    cast(int, numpy_rng["has_gauss"]),
                    cast(float, numpy_rng["cached_gaussian"]),
                )
            )
            _require(status.valid, "TRAINING_CHECKPOINT_INVALID")
            return cast(dict[str, Any], state)
        except (TrainingError, TrainingRecoveryError):
            raise TrainingError("TRAINING_CHECKPOINT_INVALID") from None
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise TrainingError("TRAINING_CHECKPOINT_INVALID") from exc

    def _write_checkpoint(
        self,
        checkpoint_root: Path,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        *,
        run_id: str,
        completed_epoch: int,
        next_epoch: int,
        next_batch_index: int,
        optimizer_steps: int,
        best_validation_loss: float,
        patience: int,
        history: Sequence[Mapping[str, object]],
        train_losses: Sequence[float],
        dataset_readiness_sha256: str,
        sealed_assets_sha256: str,
        training_config_sha256_value: str,
        environment_identity_sha256: str,
    ) -> None:
        checkpoint_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = checkpoint_root / f".generation-{uuid4().hex}.partial"
        temporary.mkdir(mode=0o700)
        try:
            named = dict(self._model.named_parameters())
            self._save_file(
                {
                    name: named[name].detach().float().cpu().contiguous()
                    for name in self._selected_names
                },
                str(temporary / "trainable.safetensors"),
            )
            self._torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            self._torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
            self._torch.save(scaler.state_dict(), temporary / "scaler.pt")
            self._torch.save(
                {
                    "torch_cpu": self._torch.get_rng_state(),
                    "torch_cuda": self._torch.cuda.get_rng_state_all(),
                },
                temporary / "rng.pt",
            )
            numpy_state = cast(
                tuple[str, np.ndarray[Any, np.dtype[np.uint32]], int, int, float],
                np.random.get_state(),
            )
            (
                numpy_algorithm,
                numpy_keys,
                numpy_position,
                numpy_has_gauss,
                numpy_cached_gaussian,
            ) = numpy_state
            _atomic_json(
                temporary / "training-state.json",
                {
                    "schema": "atlaslens-phase3f-training-checkpoint-v2",
                    "completed_epoch": completed_epoch,
                    "next_epoch": next_epoch,
                    "next_batch_index": next_batch_index,
                    "optimizer_steps": optimizer_steps,
                    "best_validation_loss": best_validation_loss,
                    "patience": patience,
                    "history": list(history),
                    "train_losses": list(train_losses),
                    "holdout_open_count": 0,
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": {
                        "algorithm": numpy_algorithm,
                        "keys": numpy_keys.tolist(),
                        "position": numpy_position,
                        "has_gauss": numpy_has_gauss,
                        "cached_gaussian": numpy_cached_gaussian,
                    },
                    "selected_parameter_names_sha256": hashlib.sha256(
                        _canonical_bytes(self._selected_names)
                    ).hexdigest(),
                    "environment_identity_sha256": environment_identity_sha256,
                    "secrets_included": False,
                },
            )
            artifacts = []
            for name in (
                "trainable.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "scaler.pt",
                "rng.pt",
                "training-state.json",
            ):
                path = temporary / name
                artifacts.append(
                    {
                        "path": name,
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_path(path),
                    }
                )
            manifest = {
                "schema": CHECKPOINT_MANIFEST_SCHEMA,
                "run_id": run_id,
                "dataset_readiness_sha256": dataset_readiness_sha256,
                "sealed_assets_sha256": sealed_assets_sha256,
                "training_config_sha256": training_config_sha256_value,
                "environment_identity_sha256": environment_identity_sha256,
                "completed_epoch": completed_epoch,
                "next_epoch": next_epoch,
                "next_batch_index": next_batch_index,
                "optimizer_steps": optimizer_steps,
                "holdout_open_count": 0,
                "artifacts": artifacts,
                "secrets_included": False,
            }
            manifest_sha = _atomic_json(temporary / "checkpoint-manifest.json", manifest)
            generation = f"generation-{optimizer_steps:08d}-{manifest_sha[:16]}"
            destination = checkpoint_root / generation
            if destination.exists():
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, destination)
            _atomic_json(
                checkpoint_root / "latest.json",
                {
                    "schema": CHECKPOINT_POINTER_SCHEMA,
                    "generation": generation,
                    "manifest_sha256": manifest_sha,
                    "secrets_included": False,
                },
            )
        except Exception as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            if isinstance(exc, TrainingError):
                raise
            raise TrainingError("TRAINING_CHECKPOINT_WRITE_FAILED") from exc

    def train(
        self,
        train_assets: Sequence[SplitAsset],
        validation_assets: Sequence[SplitAsset],
        media_root: Path,
        checkpoint_root: Path,
        *,
        batch_size: int,
        gradient_accumulation: int,
        max_epochs: int,
        deadline_epoch: float,
        run_id: str,
        dataset_readiness_sha256: str,
        sealed_assets_sha256: str,
        training_config_sha256_value: str,
        environment_identity_sha256: str,
        progress_root: Path | None = None,
    ) -> dict[str, object]:
        _require(
            bool(re.fullmatch(r"[0-9a-f]{64}", environment_identity_sha256)),
            "TRAINING_ENVIRONMENT_RECEIPT_INVALID",
        )
        optimizer = self._torch.optim.AdamW(
            self._selected_parameters,
            lr=2e-6,
            weight_decay=1e-4,
        )
        scaler = self._torch.amp.GradScaler("cuda", enabled=True)
        scheduler = self._torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _step: 1.0
        )
        resumed = self._load_checkpoint(
            checkpoint_root,
            optimizer,
            scheduler,
            scaler,
            run_id=run_id,
            dataset_readiness_sha256=dataset_readiness_sha256,
            sealed_assets_sha256=sealed_assets_sha256,
            training_config_sha256_value=training_config_sha256_value,
            environment_identity_sha256=environment_identity_sha256,
        )
        start_epoch = cast(int, resumed["next_epoch"])
        resume_batch = cast(int, resumed["next_batch_index"])
        optimizer_steps = cast(int, resumed["optimizer_steps"])
        best_loss = float(resumed["best_validation_loss"])
        patience = cast(int, resumed["patience"])
        completed_epoch = cast(int, resumed["completed_epoch"])
        history = cast(list[dict[str, object]], resumed["history"])
        resumed_train_losses = cast(list[float], resumed["train_losses"])
        last_checkpoint_at = time.monotonic()
        for epoch in range(start_epoch, max_epochs):
            _require(time.time() < deadline_epoch, "TRAINING_WALL_LIMIT_REACHED")
            batches = _epoch_batches(
                train_assets,
                batch_size=batch_size,
                seed=self._seed,
                epoch=epoch,
            )
            self._model.train()
            optimizer.zero_grad(set_to_none=True)
            train_losses = list(resumed_train_losses) if epoch == start_epoch else []
            for batch_index, batch_assets in enumerate(batches):
                if epoch == start_epoch and batch_index < resume_batch:
                    continue
                _require(time.time() < deadline_epoch, "TRAINING_WALL_LIMIT_REACHED")
                labels_by_asset = _batch_metric_labels(batch_assets)
                tensors = [
                    self._tensor(
                        media_root / item.relative_path,
                        training=True,
                        salt=epoch * 1_000_000 + batch_index * 100 + index,
                    )
                    for index, item in enumerate(batch_assets)
                ]
                batch = self._torch.stack(tensors).to("cuda", non_blocking=True)
                labels = self._torch.tensor(
                    [labels_by_asset[item.opaque_id] for item in batch_assets],
                    dtype=self._torch.long,
                    device="cuda",
                )
                with self._torch.amp.autocast("cuda", dtype=self._torch.float16):
                    embeddings = self._model(batch)
                    loss = self._metric_loss(embeddings, labels)
                    scaled_loss = loss / gradient_accumulation
                _require(bool(self._torch.isfinite(loss).item()), "TRAINING_NONFINITE_LOSS")
                scaler.scale(scaled_loss).backward()
                train_losses.append(float(loss.detach().cpu()))
                should_step = (batch_index + 1) % gradient_accumulation == 0 or (
                    batch_index + 1 == len(batches)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    self._torch.nn.utils.clip_grad_norm_(self._selected_parameters, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if progress_root is not None:
                        _atomic_json(
                            progress_root / "progress.json",
                            {
                                "schema": "atlaslens-phase3f-remote-training-progress-v1",
                                "stage": "training",
                                "completed_epoch": completed_epoch,
                                "epoch": epoch,
                                "next_batch_index": batch_index + 1,
                                "optimizer_steps": optimizer_steps,
                                "holdout_open_count": 0,
                                "secrets_included": False,
                            },
                        )
                    now = time.monotonic()
                    if (
                        optimizer_steps % CHECKPOINT_INTERVAL_STEPS == 0
                        or now - last_checkpoint_at >= CHECKPOINT_INTERVAL_SECONDS
                    ):
                        self._write_checkpoint(
                            checkpoint_root,
                            optimizer,
                            scheduler,
                            scaler,
                            run_id=run_id,
                            completed_epoch=completed_epoch,
                            next_epoch=epoch,
                            next_batch_index=batch_index + 1,
                            optimizer_steps=optimizer_steps,
                            best_validation_loss=best_loss,
                            patience=patience,
                            history=history,
                            train_losses=train_losses,
                            dataset_readiness_sha256=dataset_readiness_sha256,
                            sealed_assets_sha256=sealed_assets_sha256,
                            training_config_sha256_value=training_config_sha256_value,
                            environment_identity_sha256=environment_identity_sha256,
                        )
                        last_checkpoint_at = now
            validation_batches = _epoch_batches(
                validation_assets,
                batch_size=batch_size,
                seed=self._seed,
                epoch=0,
            )
            validation_losses: list[float] = []
            self._model.eval()
            with self._torch.inference_mode():
                for batch_index, batch_assets in enumerate(validation_batches):
                    labels_by_asset = _batch_metric_labels(batch_assets)
                    tensors = [
                        self._tensor(
                            media_root / item.relative_path,
                            training=False,
                            salt=batch_index * 100 + index,
                        )
                        for index, item in enumerate(batch_assets)
                    ]
                    batch = self._torch.stack(tensors).to("cuda", non_blocking=True)
                    labels = self._torch.tensor(
                        [labels_by_asset[item.opaque_id] for item in batch_assets],
                        dtype=self._torch.long,
                        device="cuda",
                    )
                    with self._torch.amp.autocast("cuda", dtype=self._torch.float16):
                        loss = self._metric_loss(self._model(batch), labels)
                    _require(bool(self._torch.isfinite(loss).item()), "VALIDATION_NONFINITE_LOSS")
                    validation_losses.append(float(loss.cpu()))
            mean_train = float(np.mean(train_losses))
            mean_validation = float(np.mean(validation_losses))
            improved = mean_validation < best_loss - 1e-6
            if improved:
                best_loss = mean_validation
                patience = 0
            else:
                patience += 1
            completed_epoch = epoch
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_train,
                    "validation_loss": mean_validation,
                    "improved": improved,
                }
            )
            if progress_root is not None:
                progress_root.mkdir(mode=0o700, parents=True, exist_ok=True)
                with (progress_root / "metrics.jsonl").open("ab") as metrics_stream:
                    metrics_stream.write(
                        _canonical_bytes(
                            {
                                "schema": "atlaslens-phase3f-training-metric-v1",
                                **history[-1],
                                "optimizer_steps": optimizer_steps,
                                "holdout_open_count": 0,
                                "secrets_included": False,
                            }
                        )
                    )
                    metrics_stream.flush()
                    os.fsync(metrics_stream.fileno())
            self._write_checkpoint(
                checkpoint_root,
                optimizer,
                scheduler,
                scaler,
                run_id=run_id,
                completed_epoch=epoch,
                next_epoch=epoch + 1,
                next_batch_index=0,
                optimizer_steps=optimizer_steps,
                best_validation_loss=best_loss,
                patience=patience,
                history=history,
                train_losses=(),
                dataset_readiness_sha256=dataset_readiness_sha256,
                sealed_assets_sha256=sealed_assets_sha256,
                training_config_sha256_value=training_config_sha256_value,
                environment_identity_sha256=environment_identity_sha256,
            )
            if improved:
                pointer = json.loads(
                    (checkpoint_root / "latest.json").read_text(encoding="utf-8")
                )
                shutil.copy2(
                    checkpoint_root
                    / cast(str, pointer["generation"])
                    / "trainable.safetensors",
                    checkpoint_root / "best-trainable.safetensors",
                )
            resumed_train_losses = []
            resume_batch = 0
            if patience >= 2:
                break
        _require(optimizer_steps > 0, "TRAINING_NO_OPTIMIZER_STEP")
        best_path = checkpoint_root / "best-trainable.safetensors"
        _require(
            best_path.is_file() and not best_path.is_symlink(),
            "TRAINING_BEST_CHECKPOINT_MISSING",
        )
        best_delta = self._load_file(str(best_path), device="cpu")
        named = dict(self._model.named_parameters())
        _require(set(best_delta) == set(self._selected_names), "TRAINING_BEST_CHECKPOINT_INVALID")
        with self._torch.no_grad():
            for name in self._selected_names:
                named[name].copy_(best_delta[name].to("cuda"))
        weights_changed = any(
            not self._torch.equal(
                named[name].detach().float().cpu(),
                self._initial_selected[name],
            )
            for name in self._selected_names
        )
        _require(weights_changed, "TRAINING_WEIGHTS_UNCHANGED")
        return {
            "completed_epoch": completed_epoch,
            "optimizer_steps": optimizer_steps,
            "best_validation_loss": best_loss,
            "early_stopped": completed_epoch + 1 < max_epochs,
            "history": history,
            "trainable_parameter_count": sum(
                int(parameter.numel()) for parameter in self._selected_parameters
            ),
            "trainable_parameter_names_sha256": hashlib.sha256(
                _canonical_bytes(self._selected_names)
            ).hexdigest(),
            "weights_changed": True,
        }

    def save_final(self, destination: Path) -> str:
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self._model.state_dict().items()
        }
        temporary = destination.with_name(f".{destination.name}.partial.safetensors")
        self._save_file(state, str(temporary))
        os.replace(temporary, destination)
        return _sha256_path(destination)

    def close(self) -> None:
        model, self._model = self._model, None
        del model
        self._torch.cuda.empty_cache()


def aggregate_training_smoke_failure(
    checks: Sequence[Mapping[str, object]],
) -> str | None:
    """Return the first typed smoke failure after every check has been collected."""
    for row in checks:
        code = row.get("failure_code")
        if isinstance(code, str):
            return code
    return None


def run_remote_training_smoke(
    model_path: Path,
    vendor_root: Path,
    *,
    timeout_seconds: int = 180,
    seed: int = 20260720,
    runtime_factory: Callable[..., TrainableMegaLocRuntime] = TrainableMegaLocRuntime,
) -> dict[str, object]:
    """Run one isolated CUDA optimizer step without dataset or checkpoint mutation."""
    _require(1 <= timeout_seconds <= 180, "REMOTE_TRAINING_SMOKE_TIMEOUT")
    started = time.monotonic()
    deadline = started + timeout_seconds
    checks: list[dict[str, object]] = []
    failure_code: str | None = None
    runtime: TrainableMegaLocRuntime | None = None

    def record(name: str, passed: bool, code: str, observed: object = "compatible") -> None:
        nonlocal failure_code
        if not passed and failure_code is None:
            failure_code = code
        checks.append(
            {
                "check_name": name,
                "outcome": "passed" if passed else "failed",
                "observed": observed if passed else "failed",
                "failure_code": None if passed else code,
                "secrets_included": False,
            }
        )

    def within_deadline() -> None:
        _require(time.monotonic() < deadline, "REMOTE_TRAINING_SMOKE_TIMEOUT")

    try:
        worker.verify_megaloc_artifacts(
            model_path,
            vendor_root / "megaloc_model.py",
            vendor_root / "LICENSE",
        )
        record("vendor-source-hash", True, "REMOTE_MODEL_HASH_MISMATCH")
        record("model-safetensors-hash", True, "REMOTE_MODEL_HASH_MISMATCH")
    except Exception as exc:
        code = (
            "REMOTE_MODEL_HASH_MISMATCH"
            if "SHA256" in str(getattr(exc, "code", ""))
            else "REMOTE_VENDOR_IMPORT_FAILED"
        )
        record("vendor-source-hash", False, code)
        record("model-safetensors-hash", False, code)
    try:
        within_deadline()
        if failure_code is not None:
            raise TrainingError(failure_code)
        runtime = runtime_factory(model_path, vendor_root, seed=seed)
        record("megaloc-vendor-import", True, "REMOTE_VENDOR_IMPORT_FAILED")
        record("megaloc-model-load", True, "REMOTE_TRAINING_SMOKE_FAILED")
    except Exception as exc:
        code = str(getattr(exc, "code", "REMOTE_TRAINING_SMOKE_FAILED"))
        code = (
            "REMOTE_VENDOR_IMPORT_FAILED"
            if "RUNTIME_MISSING" in code
            else "REMOTE_TRAINING_SMOKE_TIMEOUT"
            if code == "REMOTE_TRAINING_SMOKE_TIMEOUT"
            else "REMOTE_TRAINING_SMOKE_FAILED"
        )
        record("megaloc-vendor-import", False, code)
        record("megaloc-model-load", False, code)
    peak_cuda_bytes: int | None = None
    if runtime is not None:
        torch = runtime._torch  # noqa: SLF001 - same-module isolated compatibility probe
        model = runtime._model  # noqa: SLF001
        parameters = runtime._selected_parameters  # noqa: SLF001
        try:
            within_deadline()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(seed)
            batch = torch.zeros((2, 3, 392, 392), device="cuda")
            record("cuda-tensor-allocation", True, "REMOTE_CUDA_TENSOR_ALLOCATION_FAILED")
        except Exception:
            batch = None
            record("cuda-tensor-allocation", False, "REMOTE_CUDA_TENSOR_ALLOCATION_FAILED")
        output: Any = None
        loss: Any = None
        optimizer: Any = None
        before: Any = None
        try:
            within_deadline()
            optimizer = torch.optim.AdamW(parameters, lr=2e-6, weight_decay=1e-4)
            optimizer.zero_grad(set_to_none=True)
            before = parameters[0].detach().clone()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(batch)
            record("amp-autocast", True, "REMOTE_CUDA_AUTOCAST_FAILED", str(output.dtype))
            record("forward", True, "REMOTE_CUDA_FORWARD_FAILED", list(output.shape))
        except Exception:
            record("amp-autocast", False, "REMOTE_CUDA_AUTOCAST_FAILED")
            record("forward", False, "REMOTE_CUDA_FORWARD_FAILED")
        try:
            within_deadline()
            weights = torch.linspace(0.5, 1.5, output.shape[-1], device="cuda")
            loss = (output.float() * weights).mean()
            finite = bool(torch.isfinite(loss).item())
            record("finite-loss", finite, "REMOTE_TRAINING_SMOKE_FAILED", finite)
        except Exception:
            record("finite-loss", False, "REMOTE_TRAINING_SMOKE_FAILED")
        try:
            within_deadline()
            loss.backward()
            record("backward", True, "REMOTE_CUDA_BACKWARD_FAILED")
        except Exception:
            record("backward", False, "REMOTE_CUDA_BACKWARD_FAILED")
        try:
            within_deadline()
            optimizer.step()
            record("optimizer-step", True, "REMOTE_CUDA_OPTIMIZER_FAILED")
            changed = bool(not torch.equal(before, parameters[0].detach()))
            record("model-tensor-changed", changed, "REMOTE_TRAINING_SMOKE_FAILED", changed)
        except Exception:
            record("optimizer-step", False, "REMOTE_CUDA_OPTIMIZER_FAILED")
            record("model-tensor-changed", False, "REMOTE_TRAINING_SMOKE_FAILED")
        try:
            peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
        except RuntimeError:
            peak_cuda_bytes = None
    else:
        for name, code in (
            ("cuda-tensor-allocation", "REMOTE_CUDA_TENSOR_ALLOCATION_FAILED"),
            ("amp-autocast", "REMOTE_CUDA_AUTOCAST_FAILED"),
            ("forward", "REMOTE_CUDA_FORWARD_FAILED"),
            ("finite-loss", "REMOTE_TRAINING_SMOKE_FAILED"),
            ("backward", "REMOTE_CUDA_BACKWARD_FAILED"),
            ("optimizer-step", "REMOTE_CUDA_OPTIMIZER_FAILED"),
            ("model-tensor-changed", "REMOTE_TRAINING_SMOKE_FAILED"),
        ):
            record(name, False, code)
    if runtime is not None:
        runtime.close()
    elapsed = time.monotonic() - started
    failure_code = aggregate_training_smoke_failure(checks)
    if elapsed > timeout_seconds and failure_code is None:
        failure_code = "REMOTE_TRAINING_SMOKE_TIMEOUT"
    return {
        "schema": "atlaslens-phase3f-remote-training-smoke-v1",
        "outcome": "passed" if failure_code is None else "failed",
        "event": (
            "PHASE3F_REMOTE_TRAINING_SMOKE_PASSED"
            if failure_code is None
            else None
        ),
        "failure_code": failure_code,
        "checks": checks,
        "check_count": len(checks),
        "elapsed_seconds": elapsed,
        "timeout_seconds": timeout_seconds,
        "peak_cuda_bytes": peak_cuda_bytes,
        "dataset_open_count": 0,
        "holdout_open_count": 0,
        "checkpoint_write_count": 0,
        "secrets_included": False,
    }


def _runtime_descriptors(
    runtime: TrainableMegaLocRuntime,
    assets: Sequence[SplitAsset],
    media_root: Path,
    *,
    batch_size: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    return runtime.describe(
        [media_root / item.relative_path for item in assets],
        batch_size=batch_size,
    )


def _adaptive_batch(torch: Any) -> tuple[int, int, int]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_gib = int(properties.total_memory) / 1024**3
    _require(total_gib >= 16.0, "GPU_MEMORY_BELOW_TRAINING_MINIMUM")
    batch_size = 8 if total_gib >= 23.0 else 6 if total_gib >= 20.0 else 4
    if batch_size % 2:
        batch_size -= 1
    accumulation = max(1, math.ceil(16 / batch_size))
    return batch_size, accumulation, int(total_gib)


@dataclass(frozen=True, slots=True)
class TrainingJobConfig:
    run_id: str
    sealed_root: Path
    model_path: Path
    vendor_root: Path
    work_root: Path
    output_root: Path
    deadline_epoch: float
    recovery_root: Path | None = None
    seed: int = 20260720
    max_epochs: int = DEFAULT_MAX_EPOCHS
    environment_contract_sha256: str | None = None
    environment_receipt_path: Path | None = None


def run_training_job(config: TrainingJobConfig) -> dict[str, object]:
    _require(
        len(config.run_id) == 32 and all(character in _SHA256 for character in config.run_id),
        "RUN_ID_INVALID",
    )
    _require(1 <= config.max_epochs <= DEFAULT_MAX_EPOCHS, "TRAINING_EPOCH_LIMIT_INVALID")
    _require(
        time.time() < config.deadline_epoch <= time.time() + DEFAULT_TRAINING_WALL_SECONDS + 60,
        "TRAINING_DEADLINE_INVALID",
    )
    os.environ.pop("MAPILLARY_ACCESS_TOKEN", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _require(not config.output_root.exists(), "TRAINING_OUTPUT_EXISTS")
    config.output_root.mkdir(mode=0o700, parents=True)
    config.work_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment_identity_sha256 = hashlib.sha256(
        b"atlaslens-phase3f-local-unbound-environment-v1"
    ).hexdigest()
    if config.environment_contract_sha256 is not None:
        _require(
            config.environment_receipt_path is not None,
            "TRAINING_ENVIRONMENT_RECEIPT_MISSING",
        )
        environment_receipt = json.loads(
            cast(Path, config.environment_receipt_path).read_text(encoding="utf-8")
        )
        _require(
            isinstance(environment_receipt, dict)
            and environment_receipt.get("contract_sha256")
            == config.environment_contract_sha256
            and environment_receipt.get("all_imports_passed") is True
            and environment_receipt.get("compatibility_smoke_passed") is True
            and isinstance(environment_receipt.get("runtime_environment_sha256"), str)
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}",
                    cast(str, environment_receipt["runtime_environment_sha256"]),
                )
            )
            and environment_receipt.get("secrets_included") is False,
            "TRAINING_ENVIRONMENT_RECEIPT_INVALID",
        )
        _atomic_json(
            config.output_root / "environment-receipt.json",
            environment_receipt,
        )
        environment_identity_sha256 = cast(
            str, environment_receipt["runtime_environment_sha256"]
        )
    holdout_state_path = config.work_root / "training-checkpoint" / "holdout-state.json"
    if holdout_state_path.exists():
        try:
            holdout_state = json.loads(holdout_state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainingError("TRAINING_HOLDOUT_STATE_INVALID") from exc
        _require(
            isinstance(holdout_state, dict)
            and holdout_state.get("schema") == "atlaslens-phase3f-holdout-state-v1"
            and holdout_state.get("run_id") == config.run_id,
            "TRAINING_HOLDOUT_STATE_INVALID",
        )
        _require(
            holdout_state.get("holdout_open_count") == 0,
            "TRAINING_HOLDOUT_ALREADY_OPENED",
        )
    readiness_path = config.output_root / "training-readiness.json"
    sealed = require_training_ready(config.sealed_root, report_path=readiness_path)
    _require(sealed.receipt.get("run_id") == config.run_id, "DATASET_RUN_ID_MISMATCH")
    dataset_readiness_sha256 = _sha256_path(readiness_path)
    sealed_assets_sha256 = _sha256_path(config.sealed_root / "sealed-assets.json")
    config_sha256 = training_config_sha256(
        seed=config.seed,
        max_epochs=config.max_epochs,
        environment_contract_sha256=config.environment_contract_sha256,
    )
    if config.recovery_root is not None:
        _atomic_json(
            config.recovery_root / "training-configuration.json",
            {
                "schema": "atlaslens-phase3f-training-configuration-v1",
                "run_id": config.run_id,
                "dataset_readiness_sha256": dataset_readiness_sha256,
                "sealed_assets_sha256": sealed_assets_sha256,
                "training_config_sha256": config_sha256,
                "environment_contract_sha256": config.environment_contract_sha256,
                "environment_identity_sha256": environment_identity_sha256,
                "seed": config.seed,
                "maximum_epochs": config.max_epochs,
                "holdout_open_count": 0,
                "secrets_included": False,
            },
        )
    worker.verify_megaloc_artifacts(
        config.model_path,
        config.vendor_root / "megaloc_model.py",
        config.vendor_root / "LICENSE",
    )
    split_document = training_split_document(sealed)
    _atomic_json(config.output_root / "training-split-manifest.json", split_document)
    pipeline = Phase3FPipeline.create(
        config.output_root / "pipeline-state.json", run_id=config.run_id
    )
    pipeline.lock_selection(sealed.selection)
    pipeline.complete_acquisition(
        AcquisitionGuard(
            request_count=cast(int, sealed.receipt["mapillary_request_count"]),
            image_count=cast(int, sealed.receipt["image_count"]),
            media_bytes=cast(int, sealed.receipt["media_bytes"]),
        )
    )
    pipeline.record_split(sealed.split)
    references = tuple(asset for asset in sealed.split.assets if asset.role == "reference")
    validation = tuple(asset for asset in sealed.split.assets if asset.role == "calibration")
    holdout = tuple(
        asset
        for asset in sealed.split.assets
        if asset.role in {"sealed_holdout", "ood_holdout"}
    )
    media_root = sealed.root
    baseline_runtime: worker.MegaLocRuntime | None = None
    tuner: TrainableMegaLocRuntime | None = None
    started = datetime.now(UTC)
    try:
        baseline_runtime = worker.MegaLocRuntime(config.model_path, config.vendor_root)
        baseline_reference = baseline_runtime.describe(
            [media_root / item.relative_path for item in references]
        )
        baseline_validation = baseline_runtime.describe(
            [media_root / item.relative_path for item in validation]
        )
        (
            baseline_index,
            _descriptor_sha,
            _index_sha,
            _publication_sha,
        ) = worker._publish_reference_bundle(  # noqa: SLF001
            config.work_root / "baseline-index",
            references,
            baseline_reference,
        )
        baseline_rows = worker._retrieval_rows(  # noqa: SLF001
            baseline_index,
            validation,
            baseline_validation,
            references,
            tuple(item.city for item in sealed.selection.in_domain),
        )
        baseline_summary = _evaluation_summary(baseline_rows)
        _atomic_json(
            config.output_root / "pretrained-baseline.json",
            {
                "schema": "atlaslens-phase3f-pretrained-baseline-v1",
                "model_sha256": MODEL_SHA256,
                "metrics": baseline_summary,
                "locked_holdout_open_count": 0,
                "secrets_included": False,
            },
        )
        baseline_runtime.close()
        baseline_runtime = None
        import torch

        batch_size, accumulation, gpu_memory_gib = _adaptive_batch(torch)
        oom_recoveries = 0
        while True:
            try:
                tuner = TrainableMegaLocRuntime(
                    config.model_path,
                    config.vendor_root,
                    seed=config.seed,
                )
                training = tuner.train(
                    references,
                    validation,
                    media_root,
                    config.work_root / "training-checkpoint",
                    batch_size=batch_size,
                    gradient_accumulation=accumulation,
                    max_epochs=config.max_epochs,
                    deadline_epoch=config.deadline_epoch,
                    run_id=config.run_id,
                    dataset_readiness_sha256=dataset_readiness_sha256,
                    sealed_assets_sha256=sealed_assets_sha256,
                    training_config_sha256_value=config_sha256,
                    environment_identity_sha256=environment_identity_sha256,
                    progress_root=config.recovery_root,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if tuner is not None:
                    tuner.close()
                    tuner = None
                torch.cuda.empty_cache()
                oom_recoveries += 1
                if oom_recoveries > 2 or batch_size <= 4:
                    raise TrainingError("TRAINING_CUDA_OOM_EXHAUSTED") from None
                batch_size = max(4, batch_size - 2)
                accumulation = max(1, math.ceil(16 / batch_size))
        trained_reference = _runtime_descriptors(
            tuner, references, media_root, batch_size=batch_size
        )
        trained_validation = _runtime_descriptors(
            tuner, validation, media_root, batch_size=batch_size
        )
        index, descriptor_sha, index_sha, publication_sha = worker._publish_reference_bundle(  # noqa: SLF001
            config.output_root,
            references,
            trained_reference,
        )
        publication = DescriptorPublication(
            selection_lock_sha256=sealed.selection.lock_sha256,
            split_lock_sha256=sealed.split.split_lock_sha256,
            source_policy_sha256=worker.SOURCE_POLICY_SHA256,
            descriptor_publication_sha256=publication_sha,
            index_sha256=index_sha,
            city_scope=tuple(item.city for item in sealed.selection.in_domain),
            created_at=datetime.now(UTC),
        )
        pipeline.record_descriptor_publication(publication)
        validation_rows = worker._retrieval_rows(  # noqa: SLF001
            index,
            validation,
            trained_validation,
            references,
            publication.city_scope,
        )
        candidate_summary = _evaluation_summary(validation_rows)
        regression = _regressed(baseline_summary, candidate_summary)
        _atomic_json(
            config.output_root / "fine-tuned-validation.json",
            {
                "schema": "atlaslens-phase3f-fine-tuned-validation-v1",
                "metrics": candidate_summary,
                "regressed_against_pretrained": regression,
                "locked_holdout_open_count": 0,
                "secrets_included": False,
            },
        )
        threshold = fit_abstention_threshold(
            [
                CalibrationObservation(row.signals, row.predicted_cities[0] == row.city)
                for row in validation_rows
            ],
            selection_lock_sha256=sealed.selection.lock_sha256,
            split_lock_sha256=sealed.split.split_lock_sha256,
        )
        pipeline.lock_threshold(threshold)
        _atomic_json(config.output_root / "calibration.json", threshold.document())
        _atomic_json(
            holdout_state_path,
            {
                "schema": "atlaslens-phase3f-holdout-state-v1",
                "run_id": config.run_id,
                "holdout_open_count": 1,
                "secrets_included": False,
            },
        )
        if config.recovery_root is not None:
            _atomic_json(
                config.recovery_root / "progress.json",
                {
                    "schema": "atlaslens-phase3f-remote-training-progress-v1",
                    "stage": "holdout",
                    "completed_epoch": training["completed_epoch"],
                    "optimizer_steps": training["optimizer_steps"],
                    "holdout_open_count": 1,
                    "secrets_included": False,
                },
            )
        holdout_matrix = _runtime_descriptors(
            tuner,
            holdout,
            media_root,
            batch_size=batch_size,
        )
        holdout_rows = worker._retrieval_rows(  # noqa: SLF001
            index,
            holdout,
            holdout_matrix,
            references,
            publication.city_scope,
        )
        benchmark = evaluate_holdout_once(
            holdout_rows,
            threshold=threshold,
            selection_lock_sha256=sealed.selection.lock_sha256,
            split_lock_sha256=sealed.split.split_lock_sha256,
            in_domain_cities=publication.city_scope,
            ood_cities=tuple(item.city for item in sealed.selection.ood),
            leakage_passed=sealed.split.leakage.passed,
            city_minimums_passed=True,
            security_integrity_passed=True,
            holdout_open_count_before=0,
        )
        pipeline.record_benchmark(benchmark)
        pipeline.finalize()
        final_path = config.output_root / "megaloc-finetuned.safetensors"
        final_sha = tuner.save_final(final_path)
        _require(training["weights_changed"] is True, "TRAINING_WEIGHTS_UNCHANGED")
        _require(final_sha != MODEL_SHA256, "TRAINING_WEIGHTS_UNCHANGED")
        _atomic_json(config.output_root / "aggregate-benchmark.json", benchmark.document())
        _atomic_json(
            config.output_root / "descriptor-publication.json",
            {**publication.document(), "reference_descriptor_sha256": descriptor_sha},
        )
        training_receipt = {
            "schema": TRAINING_RECEIPT_SCHEMA,
            "run_id": config.run_id,
            "seed": config.seed,
            "objective": "batch_hard_metric_learning_softplus_cosine_v1",
            "optimizer": "adamw",
            "mixed_precision": True,
            "adaptive_batch_size": batch_size,
            "gradient_accumulation": accumulation,
            "gpu_memory_gib_floor": gpu_memory_gib,
            "oom_recoveries": oom_recoveries,
            "maximum_oom_recoveries": 2,
            "maximum_epochs": config.max_epochs,
            "early_stopping_patience": 2,
            "train_only_augmentation": True,
            "pretrained_weight_sha256": MODEL_SHA256,
            "fine_tuned_weight_sha256": final_sha,
            "weights_changed": True,
            "training": training,
            "locked_holdout_open_count": 1,
            "regressed_against_pretrained_validation": regression,
            "production_model_changed": False,
            "automatic_promotion_performed": False,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "mapillary_requests": 0,
            "mapillary_token_available": False,
            "secrets_included": False,
        }
        _atomic_json(config.output_root / "training-provenance.json", training_receipt)
        execution = {
            "schema": "atlaslens-phase3f-training-execution-v1",
            "run_id": config.run_id,
            "outcome": benchmark.outcome,
            "dataset_ready": True,
            "optimizer_steps": training["optimizer_steps"],
            "locked_holdout_open_count": 1,
            "weights_changed": True,
            "regressed_against_pretrained_validation": regression,
            "automatic_promotion_performed": False,
            "mapillary_requests": 0,
            "secrets_included": False,
            "raw_images_included": False,
        }
        _atomic_json(config.output_root / "execution-receipt.json", execution)
        _atomic_json(
            config.output_root / "checksum-inventory.json",
            worker._inventory_output(config.output_root),  # noqa: SLF001
        )
        return execution
    finally:
        if baseline_runtime is not None:
            baseline_runtime.close()
        if tuner is not None:
            tuner.close()


__all__ = [
    "DATASET_ARCHIVE_SCHEMA",
    "DatasetArchive",
    "DatasetNotReadyError",
    "TrainableMegaLocRuntime",
    "TrainingError",
    "TrainingJobConfig",
    "require_training_ready",
    "run_training_job",
    "training_config_sha256",
    "training_readiness_document",
    "training_split_document",
    "write_dataset_archive",
]
