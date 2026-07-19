"""Offline orchestration for the bounded Phase 3B3 private demo run.

This module never loads application settings, a token, or a network client.  It
consumes only the connector's completed credential-free artifacts and reuses
the Phase 3B1/3B2 descriptor and index pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from atlaslens_api.corpus_index.artifacts import (
    canonical_json_bytes,
    read_json,
    sha256_json,
    sha256_path,
)
from atlaslens_api.corpus_pipeline.policy import SourcePolicy
from atlaslens_api.mapillary_demo.acquisition import (
    atomic_write_bytes,
    load_acquisition_manifest,
    load_coverage_audit,
)
from atlaslens_api.mapillary_demo.benchmark import (
    MapillaryBenchmarkResult,
    evaluate_mapillary_holdout,
)
from atlaslens_api.mapillary_demo.indexing import (
    MEGALOC_PHASE3B2_ARTIFACT_SHA256,
    PRIVATE_DEMO_AUTHORIZATION_ID,
    PrivateDemoMegaLocExecutor,
    build_mapillary_demo_index,
)
from atlaslens_api.mapillary_demo.manifest import (
    ingest_with_phase3b1,
    materialize_phase3b1_manifest,
)
from atlaslens_api.mapillary_demo.models import (
    MAPILLARY_PILOT_ROOT,
    AcquisitionCheckpoint,
    AcquisitionManifest,
    CoverageAudit,
)
from atlaslens_api.mapillary_demo.selection import (
    LockedMapillarySplit,
    SelectedMapillaryAsset,
    select_locked_split,
)
from atlaslens_api.megaloc_adapter.models import load_megaloc_config
from atlaslens_api.megaloc_adapter.provider import (
    MegaLocDescriptorProvider,
    PrivateDemoExecutionMetrics,
)

_PROVINCE_CODE_BY_AOI = {
    "kayseri-urban-v1": ("Kayseri", "TR-38"),
    "ankara-urban-v1": ("Ankara", "TR-06"),
    "sivas-urban-v1": ("Sivas", "TR-58"),
}
_BATCH_SIZE = 8
_COORDINATE_UNCERTAINTY_M = 100.0
_ABSTAIN_IF_COSINE_DISTANCE_GT: float | None = None
_SPLIT_SCHEMA = "atlaslens-mapillary-orchestration-split-v1"
_RECEIPT_SCHEMA = "atlaslens-mapillary-private-demo-run-v1"
_PERFORMANCE_RECEIPT_SCHEMA = "atlaslens-mapillary-private-demo-performance-v1"
_MAX_PERFORMANCE_RECEIPT_BYTES = 64 * 1024


class Phase3B3RunError(RuntimeError):
    """Stable, payload-free orchestration failure."""


@dataclass(frozen=True, slots=True)
class Phase3B3RunPaths:
    project_root: Path
    pilot_root: Path
    coverage_audit: Path
    acquisition_manifest: Path
    acquisition_checkpoint: Path
    source_policy: Path
    manifest_schema: Path
    megaloc_config: Path
    locked_split: Path
    phase3b1_corpus: Path
    phase3b1_manifest: Path
    phase3b1_work: Path
    phase3b1_checkpoint_root: Path
    descriptor_work: Path
    descriptor_publication: Path
    index_bundle: Path
    benchmark: Path
    aggregate_receipt: Path
    performance_receipt: Path

    @classmethod
    def fixed(cls, project_root: Path) -> Phase3B3RunPaths:
        return cls._from_roots(
            project_root=project_root,
            pilot_root=Path(MAPILLARY_PILOT_ROOT),
        )

    @classmethod
    def for_test(
        cls,
        *,
        project_root: Path,
        pilot_root: Path,
    ) -> Phase3B3RunPaths:
        """Dependency-injected root for synthetic tests; the CLI never exposes it."""

        return cls._from_roots(project_root=project_root, pilot_root=pilot_root)

    @classmethod
    def _from_roots(
        cls,
        *,
        project_root: Path,
        pilot_root: Path,
    ) -> Phase3B3RunPaths:
        project = project_root.resolve()
        pilot = pilot_root.resolve()
        corpus = pilot / "phase3b1-corpus"
        return cls(
            project_root=project,
            pilot_root=pilot,
            coverage_audit=pilot / "metadata" / "coverage-audit.json",
            acquisition_manifest=pilot / "metadata" / "acquisition-manifest.json",
            acquisition_checkpoint=pilot / "checkpoints" / "acquisition.json",
            source_policy=(
                project / "config" / "phase3b3" / "mapillary-source-policy-v1.json"
            ),
            manifest_schema=project / "config" / "corpus" / "manifest-schema-v1.json",
            megaloc_config=project / "config" / "models" / "megaloc-phase3b2.json",
            locked_split=pilot / "metadata" / "locked-split.json",
            phase3b1_corpus=corpus,
            phase3b1_manifest=corpus / "mapillary-phase3b1-manifest.jsonl",
            phase3b1_work=pilot / "derived" / "phase3b1",
            phase3b1_checkpoint_root=pilot / "checkpoints",
            descriptor_work=pilot / "derived" / "descriptor-work",
            descriptor_publication=pilot / "derived" / "descriptors",
            index_bundle=pilot / "derived" / "mapillary-faiss-index",
            benchmark=pilot / "derived" / "locked-holdout-benchmark.json",
            aggregate_receipt=pilot / "receipts" / "private-demo-run.json",
            performance_receipt=pilot / "receipts" / "private-demo-performance.json",
        )


@dataclass(frozen=True, slots=True)
class PlannedPhase3B3Run:
    audit: CoverageAudit
    acquisition: AcquisitionManifest
    split: LockedMapillarySplit
    source_policy_sha256: str


@dataclass(frozen=True, slots=True)
class AggregateRunReceipt:
    value: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return dict(self.value)


def plan_private_demo_run(paths: Phase3B3RunPaths) -> PlannedPhase3B3Run:
    """Read and validate all credential-free inputs without writing or inference."""

    _validate_paths(paths)
    audit = load_coverage_audit(paths.coverage_audit)
    acquisition = load_acquisition_manifest(paths.acquisition_manifest)
    _regular_file(paths.source_policy)
    source_policy_hash = SourcePolicy.load(paths.source_policy).sha256
    if acquisition.source_policy_receipt_sha256 != source_policy_hash:
        raise Phase3B3RunError("PHASE3B3_SOURCE_POLICY_MISMATCH")
    if acquisition.coverage_audit_sha256 != sha256_path(paths.coverage_audit):
        raise Phase3B3RunError("PHASE3B3_COVERAGE_AUDIT_MISMATCH")
    _verify_completed_acquisition(paths.acquisition_checkpoint, acquisition)
    expected_city, province_code = _province_identity(acquisition.aoi_id)
    split = select_locked_split(
        acquisition,
        audit,
        acquisition_root=paths.pilot_root,
        province_code=province_code,
    )
    if split.decision.city != expected_city or split.decision.province != expected_city:
        raise Phase3B3RunError("PHASE3B3_CITY_IDENTITY_MISMATCH")
    return PlannedPhase3B3Run(
        audit=audit,
        acquisition=acquisition,
        split=split,
        source_policy_sha256=source_policy_hash,
    )


def execute_private_demo_run(
    paths: Phase3B3RunPaths,
    *,
    _executor_for_test: PrivateDemoMegaLocExecutor | None = None,
) -> AggregateRunReceipt:
    """Execute or deterministically resume the offline descriptor/index benchmark."""

    plan = plan_private_demo_run(paths)
    split_file_sha256 = _persist_locked_split(paths, plan.split)
    manifest = materialize_phase3b1_manifest(
        plan.split,
        acquisition_root=paths.pilot_root,
        corpus_root=paths.phase3b1_corpus,
        manifest_path=paths.phase3b1_manifest,
        source_policy_path=paths.source_policy,
        coordinate_uncertainty_m=_COORDINATE_UNCERTAINTY_M,
    )
    ingestion = ingest_with_phase3b1(
        manifest,
        work_root=paths.phase3b1_work,
        checkpoint_root=paths.phase3b1_checkpoint_root,
        manifest_schema_path=paths.manifest_schema,
        source_policy_path=paths.source_policy,
        checkpoint_path="phase3b1-mapillary-ingestion.json",
        resume=(paths.phase3b1_checkpoint_root / "phase3b1-mapillary-ingestion.json").exists(),
    )

    owned_provider: MegaLocDescriptorProvider | None = None
    executor = _executor_for_test
    if executor is None:
        config = load_megaloc_config(paths.megaloc_config, project_root=paths.project_root)
        owned_provider = MegaLocDescriptorProvider(config)
        executor = owned_provider
    try:
        index_result = build_mapillary_demo_index(
            ingestion,
            plan.split,
            executor,
            corpus_root=paths.phase3b1_corpus,
            descriptor_work_root=paths.descriptor_work,
            descriptor_output_root=paths.descriptor_publication,
            output_dir=paths.index_bundle,
            batch_size=_BATCH_SIZE,
        )
        if owned_provider is not None:
            _ensure_performance_receipt(paths, plan, owned_provider)
        benchmark = evaluate_mapillary_holdout(
            index_result.publication,
            index_result.holdout_descriptors.dataset,
            plan.split,
            expected_selection_lock_sha256=plan.split.lock_sha256,
            abstain_if_cosine_distance_gt=_ABSTAIN_IF_COSINE_DISTANCE_GT,
        )
        _freeze_json(paths, paths.benchmark, benchmark.to_json())
        receipt = _aggregate_receipt(
            paths=paths,
            plan=plan,
            split_file_sha256=split_file_sha256,
            manifest_sha256=manifest.manifest_sha256,
            phase3b1_split_lock_sha256=ingestion.split_lock.lock_sha256,
            index_metadata=index_result.publication.metadata,
            benchmark=benchmark,
        )
        _freeze_json(paths, paths.aggregate_receipt, receipt.to_json())
        return receipt
    finally:
        if owned_provider is not None:
            owned_provider.close()
        elif _executor_for_test is not None:
            close = getattr(_executor_for_test, "close", None)
            if callable(close):
                close()


def _ensure_performance_receipt(
    paths: Phase3B3RunPaths,
    plan: PlannedPhase3B3Run,
    provider: MegaLocDescriptorProvider,
) -> None:
    expected_count = len(plan.split.references) + len(plan.split.holdout)
    if paths.performance_receipt.exists() or _is_link(paths.performance_receipt):
        _validate_performance_receipt(paths, plan, expected_count)
        return
    metrics = provider.private_demo_metrics
    if metrics.image_count != expected_count:
        provider.reset_private_demo_metrics()
        assets = (*plan.split.references, *plan.split.holdout)
        for offset in range(0, len(assets), _BATCH_SIZE):
            locators = tuple(
                _locked_asset_path(paths, asset) for asset in assets[offset : offset + _BATCH_SIZE]
            )
            matrix = provider.describe_batch_for_private_demo(
                locators,
                authorization_id=PRIVATE_DEMO_AUTHORIZATION_ID,
                source_policy_sha256=plan.source_policy_sha256,
            )
            del matrix
        metrics = provider.private_demo_metrics
    _write_performance_receipt(paths, plan, metrics, expected_count)


def _locked_asset_path(paths: Phase3B3RunPaths, asset: SelectedMapillaryAsset) -> Path:
    relative = PurePosixPath(asset.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise Phase3B3RunError("PHASE3B3_PERFORMANCE_LOCATOR_INVALID")
    path = paths.pilot_root.joinpath(*relative.parts)
    _relative_to_pilot(paths, path)
    _regular_file(path)
    return path


def _write_performance_receipt(
    paths: Phase3B3RunPaths,
    plan: PlannedPhase3B3Run,
    metrics: PrivateDemoExecutionMetrics,
    expected_count: int,
) -> None:
    if (
        expected_count <= 0
        or metrics.device not in {"cpu", "cuda"}
        or metrics.image_count != expected_count
        or not 1 <= metrics.batch_call_count <= expected_count
        or not math.isfinite(metrics.elapsed_seconds)
        or metrics.elapsed_seconds <= 0.0
        or type(metrics.peak_vram_bytes) is not int
        or metrics.peak_vram_bytes < 0
        or (metrics.device == "cpu" and metrics.peak_vram_bytes != 0)
    ):
        raise Phase3B3RunError("PHASE3B3_PERFORMANCE_MEASUREMENT_INVALID")
    base: dict[str, object] = {
        "schema": _PERFORMANCE_RECEIPT_SCHEMA,
        "technical_demo_only": True,
        "measurement_scope": "real_megaloc_all_locked_reference_and_holdout_images",
        "descriptor_count": metrics.image_count,
        "reference_count": len(plan.split.references),
        "holdout_count": len(plan.split.holdout),
        "descriptor_dimension": 8_448,
        "batch_size": _BATCH_SIZE,
        "batch_call_count": metrics.batch_call_count,
        "runtime_seconds": round(metrics.elapsed_seconds, 6),
        "images_per_second": round(metrics.image_count / metrics.elapsed_seconds, 6),
        "device": metrics.device,
        "peak_vram_bytes": metrics.peak_vram_bytes,
        "vram_semantics": "torch_cuda_max_memory_allocated_in_process; zero_on_cpu",
        "model_artifact_sha256": MEGALOC_PHASE3B2_ARTIFACT_SHA256,
        "source_policy_sha256": plan.source_policy_sha256,
        "selection_lock_sha256": plan.split.lock_sha256,
        "measurement_descriptors_persisted": False,
        "network_used": False,
        "individual_asset_records": False,
    }
    _freeze_json(
        paths,
        paths.performance_receipt,
        {**base, "receipt_sha256": sha256_json(base)},
    )


def _validate_performance_receipt(
    paths: Phase3B3RunPaths,
    plan: PlannedPhase3B3Run,
    expected_count: int,
) -> None:
    _regular_file(paths.performance_receipt)
    if paths.performance_receipt.stat().st_size > _MAX_PERFORMANCE_RECEIPT_BYTES:
        raise Phase3B3RunError("PHASE3B3_PERFORMANCE_RECEIPT_INVALID")
    try:
        value = read_json(paths.performance_receipt)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3B3RunError("PHASE3B3_PERFORMANCE_RECEIPT_INVALID") from exc
    if not isinstance(value, dict):
        raise Phase3B3RunError("PHASE3B3_PERFORMANCE_RECEIPT_INVALID")
    document = cast(dict[str, object], value)
    expected_keys = {
        "schema",
        "technical_demo_only",
        "measurement_scope",
        "descriptor_count",
        "reference_count",
        "holdout_count",
        "descriptor_dimension",
        "batch_size",
        "batch_call_count",
        "runtime_seconds",
        "images_per_second",
        "device",
        "peak_vram_bytes",
        "vram_semantics",
        "model_artifact_sha256",
        "source_policy_sha256",
        "selection_lock_sha256",
        "measurement_descriptors_persisted",
        "network_used",
        "individual_asset_records",
        "receipt_sha256",
    }
    integer_expectations = {
        "descriptor_count": expected_count,
        "reference_count": len(plan.split.references),
        "holdout_count": len(plan.split.holdout),
        "descriptor_dimension": 8_448,
        "batch_size": _BATCH_SIZE,
    }
    batch_call_count = document.get("batch_call_count")
    runtime_seconds = document.get("runtime_seconds")
    images_per_second = document.get("images_per_second")
    device = document.get("device")
    peak_vram_bytes = document.get("peak_vram_bytes")
    base: dict[str, object] = {
        key: item for key, item in document.items() if key != "receipt_sha256"
    }
    if (
        expected_count <= 0
        or set(document) != expected_keys
        or any(
            type(document.get(key)) is not int or document.get(key) != expected
            for key, expected in integer_expectations.items()
        )
        or type(batch_call_count) is not int
        or not 1 <= batch_call_count <= expected_count
        or type(runtime_seconds) is not float
        or not math.isfinite(runtime_seconds)
        or runtime_seconds <= 0.0
        or type(images_per_second) is not float
        or not math.isfinite(images_per_second)
        or images_per_second <= 0.0
        or not math.isclose(
            images_per_second,
            expected_count / runtime_seconds,
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
        or device not in {"cpu", "cuda"}
        or type(peak_vram_bytes) is not int
        or peak_vram_bytes < 0
        or (device == "cpu" and peak_vram_bytes != 0)
        or document.get("schema") != _PERFORMANCE_RECEIPT_SCHEMA
        or document.get("technical_demo_only") is not True
        or document.get("measurement_scope")
        != "real_megaloc_all_locked_reference_and_holdout_images"
        or document.get("vram_semantics")
        != "torch_cuda_max_memory_allocated_in_process; zero_on_cpu"
        or document.get("model_artifact_sha256") != MEGALOC_PHASE3B2_ARTIFACT_SHA256
        or document.get("source_policy_sha256") != plan.source_policy_sha256
        or document.get("selection_lock_sha256") != plan.split.lock_sha256
        or document.get("measurement_descriptors_persisted") is not False
        or document.get("network_used") is not False
        or document.get("individual_asset_records") is not False
        or document.get("receipt_sha256") != sha256_json(base)
    ):
        raise Phase3B3RunError("PHASE3B3_PERFORMANCE_RECEIPT_INVALID")


def _persist_locked_split(paths: Phase3B3RunPaths, split: LockedMapillarySplit) -> str:
    document = _split_document(split)
    _freeze_json(paths, paths.locked_split, document)
    return sha256_path(paths.locked_split)


def _split_document(split: LockedMapillarySplit) -> dict[str, object]:
    base: dict[str, object] = {
        "schema": _SPLIT_SCHEMA,
        "selection_lock_sha256": split.lock_sha256,
        "acquisition_manifest_sha256": split.acquisition_manifest_sha256,
        "source_policy_sha256": split.source_policy_sha256,
        "decision": {
            "aoi_id": split.decision.aoi_id,
            "aoi_version": split.decision.aoi_version,
            "display_name": split.decision.display_name,
            "city": split.decision.city,
            "province": split.decision.province,
            "reason": split.decision.reason,
            "kayseri_eligible": split.decision.kayseri_eligible,
        },
        "reference_target": split.reference_target,
        "holdout_target": split.holdout_target,
        "reference_shortfall": split.reference_shortfall,
        "holdout_shortfall": split.holdout_shortfall,
        "suppressed_stationary_or_adjacent": split.suppressed_stationary_or_adjacent,
        "suppressed_exact_or_near_duplicate": split.suppressed_exact_or_near_duplicate,
        "references": [_asset_record(asset) for asset in split.references],
        "holdout": [_asset_record(asset) for asset in split.holdout],
        "positive_reference_ids": {
            query_id: {
                threshold: list(ids) for threshold, ids in sorted(thresholds.items())
            }
            for query_id, thresholds in sorted(split.positive_reference_ids.items())
        },
        "holdout_policy": {
            "abstain_if_cosine_distance_gt": _ABSTAIN_IF_COSINE_DISTANCE_GT,
            "tuning_after_lock": False,
        },
    }
    return {**base, "document_sha256": sha256_json(base)}


def _asset_record(asset: SelectedMapillaryAsset) -> dict[str, object]:
    return {
        "asset_id": asset.asset_id,
        "mapillary_image_id": asset.mapillary_image_id,
        "relative_path": asset.relative_path,
        "raw_sha256": asset.raw_sha256,
        "normalized_sha256": asset.normalized_sha256,
        "perceptual_hash": asset.perceptual_hash,
        "width_px": asset.width_px,
        "height_px": asset.height_px,
        "latitude": asset.latitude,
        "longitude": asset.longitude,
        "captured_at": asset.captured_at.isoformat(),
        "acquired_at": asset.acquired_at.isoformat(),
        "compass_angle": asset.compass_angle,
        "sequence_id": asset.sequence_id,
        "creator_id": asset.creator_id,
        "source_page_url": asset.source_page_url,
        "attribution_text": asset.attribution_text,
        "license_identifier": asset.license_identifier,
        "license_url": asset.license_url,
        "city": asset.city,
        "province": asset.province,
        "province_code": asset.province_code,
        "spatial_cell": asset.spatial_cell,
        "split": asset.split,
    }


def _aggregate_receipt(
    *,
    paths: Phase3B3RunPaths,
    plan: PlannedPhase3B3Run,
    split_file_sha256: str,
    manifest_sha256: str,
    phase3b1_split_lock_sha256: str,
    index_metadata: dict[str, Any],
    benchmark: MapillaryBenchmarkResult,
) -> AggregateRunReceipt:
    metrics = {
        threshold: recall.to_json()
        for threshold, recall in sorted(benchmark.geographic_recall.items())
    }
    base: dict[str, object] = {
        "schema": _RECEIPT_SCHEMA,
        "state": "completed_private_technical_demo",
        "technical_demo_only": True,
        "production_approved": False,
        "public_distribution_approved": False,
        "aoi_id": plan.split.decision.aoi_id,
        "city": plan.split.decision.city,
        "province": plan.split.decision.province,
        "acquired_image_count": len(plan.acquisition.assets),
        "downloaded_bytes": plan.acquisition.downloaded_bytes,
        "manifest_sha256": sha256_path(paths.acquisition_manifest),
        "reference_count": len(plan.split.references),
        "holdout_count": len(plan.split.holdout),
        "reference_shortfall": plan.split.reference_shortfall,
        "holdout_shortfall": plan.split.holdout_shortfall,
        "source_policy_sha256": plan.source_policy_sha256,
        "selection_lock_sha256": plan.split.lock_sha256,
        "persisted_split_sha256": split_file_sha256,
        "phase3b1_manifest_sha256": manifest_sha256,
        "phase3b1_split_lock_sha256": phase3b1_split_lock_sha256,
        "model_artifact_sha256": index_metadata.get("model_artifact_sha256"),
        "descriptor_dimension": index_metadata.get("descriptor_dimension"),
        "index_version": index_metadata.get("index_version"),
        "index_build_hash": index_metadata.get("base_index_build_hash"),
        "index_metadata_sha256": sha256_path(paths.index_bundle / "metadata.json"),
        "index_sha256": sha256_path(paths.index_bundle / "PUBLISHED.json"),
        "index_validated": True,
        "attribution_validated": True,
        "benchmark_result_sha256": benchmark.result_sha256,
        "geographic_recall": metrics,
        "median_geodesic_error_m": benchmark.median_geodesic_error_m,
        "p90_geodesic_error_m": benchmark.p90_geodesic_error_m,
        "city_accuracy": benchmark.city_accuracy,
        "province_accuracy": benchmark.province_accuracy,
        "abstention_coverage": benchmark.abstention_coverage,
        "individual_asset_records_in_receipt": False,
        "operator_images_used": False,
        "network_used_by_runner": False,
    }
    if base["model_artifact_sha256"] != MEGALOC_PHASE3B2_ARTIFACT_SHA256:
        raise Phase3B3RunError("PHASE3B3_MODEL_ARTIFACT_MISMATCH")
    return AggregateRunReceipt({**base, "receipt_sha256": sha256_json(base)})


def _freeze_json(
    paths: Phase3B3RunPaths,
    destination: Path,
    value: dict[str, object],
) -> None:
    relative = _relative_to_pilot(paths, destination)
    payload = canonical_json_bytes(value)
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise Phase3B3RunError("PHASE3B3_FROZEN_ARTIFACT_MISMATCH")
        return
    atomic_write_bytes(paths.pilot_root, PurePosixPath(relative.as_posix()), payload)


def _verify_completed_acquisition(
    checkpoint_path: Path,
    acquisition: AcquisitionManifest,
) -> None:
    checkpoint = _load_checkpoint(checkpoint_path)
    expected_ids = tuple(sorted(asset.mapillary_image_id for asset in acquisition.assets))
    if (
        checkpoint.status != "completed"
        or checkpoint.failure_code is not None
        or checkpoint.completed_image_ids != expected_ids
        or checkpoint.downloaded_bytes != acquisition.downloaded_bytes
    ):
        raise Phase3B3RunError("PHASE3B3_ACQUISITION_NOT_COMPLETED")


def _load_checkpoint(path: Path) -> AcquisitionCheckpoint:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise Phase3B3RunError("PHASE3B3_ACQUISITION_CHECKPOINT_INVALID")
    try:
        return AcquisitionCheckpoint.model_validate(read_json(path))
    except (OSError, UnicodeError, ValueError) as exc:
        raise Phase3B3RunError("PHASE3B3_ACQUISITION_CHECKPOINT_INVALID") from exc


def _province_identity(aoi_id: str) -> tuple[str, str]:
    try:
        return _PROVINCE_CODE_BY_AOI[aoi_id]
    except KeyError as exc:
        raise Phase3B3RunError("PHASE3B3_UNSUPPORTED_PILOT_CITY") from exc


def _validate_paths(paths: Phase3B3RunPaths) -> None:
    root = paths.pilot_root
    if not root.is_dir() or _is_link(root):
        raise Phase3B3RunError("PHASE3B3_PILOT_ROOT_INVALID")
    for path in (
        paths.coverage_audit,
        paths.acquisition_manifest,
        paths.acquisition_checkpoint,
        paths.locked_split,
        paths.phase3b1_corpus,
        paths.phase3b1_manifest,
        paths.phase3b1_work,
        paths.phase3b1_checkpoint_root,
        paths.descriptor_work,
        paths.descriptor_publication,
        paths.index_bundle,
        paths.benchmark,
        paths.aggregate_receipt,
        paths.performance_receipt,
    ):
        _relative_to_pilot(paths, path)
    _regular_file(paths.source_policy)
    _regular_file(paths.manifest_schema)


def _relative_to_pilot(paths: Phase3B3RunPaths, path: Path) -> Path:
    try:
        return path.resolve(strict=False).relative_to(paths.pilot_root)
    except ValueError as exc:
        raise Phase3B3RunError("PHASE3B3_PATH_OUTSIDE_PILOT_ROOT") from exc


def _regular_file(path: Path) -> Path:
    if not path.is_file() or _is_link(path):
        raise Phase3B3RunError("PHASE3B3_REQUIRED_FILE_INVALID")
    return path


def _is_link(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline AtlasLens Phase 3B3 private demo pipeline."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required")
    try:
        receipt = execute_private_demo_run(Phase3B3RunPaths.fixed(args.project_root))
    except Exception:  # CLI must not expose paths, tokens, URLs, or raw payloads.
        print("PHASE3B3_PRIVATE_DEMO_RUN_FAILED", file=sys.stderr)
        return 1
    print(json.dumps(receipt.to_json(), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AggregateRunReceipt",
    "Phase3B3RunError",
    "Phase3B3RunPaths",
    "PlannedPhase3B3Run",
    "execute_private_demo_run",
    "main",
    "plan_private_demo_run",
]
