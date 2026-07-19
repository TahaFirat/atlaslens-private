"""Rights-gated corpus and bounded private Mapillary pilot command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from atlaslens_api.capture_import import (
    CAPTURE_COMMANDS,
    FFMPEG_NOT_AVAILABLE_MESSAGE,
    CaptureImportError,
    add_capture_subparsers,
    dispatch_capture_command,
)
from atlaslens_api.corpus_index.benchmark import (
    BenchmarkQuery,
    compute_locked_holdout_hash,
)
from atlaslens_api.corpus_index.descriptors import DescriptorDataset
from atlaslens_api.corpus_index.errors import (
    CorpusIndexError,
    ProviderNotConfiguredError,
)
from atlaslens_api.corpus_index.index import PublishedCorpusIndex
from atlaslens_api.corpus_index.providers import (
    DescriptorProvider,
    ProductionDescriptorRegistry,
)
from atlaslens_api.corpus_pipeline import (
    CheckpointStore,
    CorpusPipelineError,
    IngestionConfig,
    filter_leakage,
    load_ingested_corpus,
)
from atlaslens_api.corpus_workflow import CorpusWorkflow, CorpusWorkflowPaths
from atlaslens_api.mapillary_demo.cli import (
    MAPILLARY_COMMANDS,
    add_mapillary_subparsers,
    dispatch_mapillary_command,
    load_mapillary_access_token,
)
from atlaslens_api.mapillary_demo.errors import MapillaryDemoError

COMMANDS = (
    "validate-manifest",
    "ingest",
    "check-leakage",
    "build-descriptors",
    "build-index",
    "evaluate",
    "run-pipeline",
    "inspect-checkpoint",
    *CAPTURE_COMMANDS,
    *MAPILLARY_COMMANDS,
)


def _add_pipeline_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-schema", type=Path, required=True)
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Explicit checkpoint directory; pipeline-state.json is stored inside it.",
    )
    parser.add_argument("--config", type=Path, required=True)


def _add_resume_and_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlaslens-corpus",
        description="Run rights-gated AtlasLens corpus and private pilot workflows.",
        epilog="Exit codes: 0 success, 2 invalid arguments/config, 3 provider unavailable, "
        "4 rights/integrity/state/filesystem failure.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-manifest",
        help="Validate schema, rights, files and leakage without writing artifacts.",
    )
    _add_pipeline_paths(validate)

    ingest = subparsers.add_parser("ingest", help="Ingest through the locked split state.")
    _add_pipeline_paths(ingest)
    _add_resume_and_dry_run(ingest)

    leakage = subparsers.add_parser(
        "check-leakage",
        help="Recheck an ingested corpus for duplicate and split leakage.",
    )
    _add_pipeline_paths(leakage)
    leakage.add_argument("--dry-run", action="store_true")

    descriptors = subparsers.add_parser(
        "build-descriptors",
        help="Build descriptors with an explicitly registered production adapter.",
    )
    _add_pipeline_paths(descriptors)
    _add_resume_and_dry_run(descriptors)
    descriptors.add_argument("--batch-size", type=int)

    index = subparsers.add_parser("build-index", help="Atomically publish an exact/FAISS index.")
    _add_pipeline_paths(index)
    _add_resume_and_dry_run(index)
    index.add_argument("--backend", choices=("exact", "faiss"))
    index.add_argument("--index-version")
    index.add_argument("--shard-size", type=int)
    index.add_argument("--holdout", type=Path)
    index.add_argument("--expected-holdout-hash")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a verified locked holdout.")
    _add_pipeline_paths(evaluate)
    _add_resume_and_dry_run(evaluate)
    evaluate.add_argument("--holdout", type=Path)
    evaluate.add_argument("--expected-holdout-hash")
    evaluate.add_argument("--abstain-if-distance-gt", type=float)

    run = subparsers.add_parser("run-pipeline", help="Run every state through COMPLETED.")
    _add_pipeline_paths(run)
    _add_resume_and_dry_run(run)

    checkpoint = subparsers.add_parser(
        "inspect-checkpoint",
        help="Inspect a safe checkpoint projection without reading corpus bytes.",
    )
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    add_capture_subparsers(subparsers)
    add_mapillary_subparsers(subparsers)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: ProductionDescriptorRegistry | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    selected_registry = registry or ProductionDescriptorRegistry()
    try:
        result = _dispatch(args, selected_registry)
    except ProviderNotConfiguredError:
        _emit(
            {
                "status": "error",
                "code": "production_descriptor_provider_not_configured",
            },
            error=True,
        )
        return 3
    except CaptureImportError as exc:
        payload: dict[str, object] = {"status": "error", "code": exc.code}
        if exc.code == "ffmpeg_not_available":
            payload["message"] = FFMPEG_NOT_AVAILABLE_MESSAGE
        _emit(payload, error=True)
        return 4
    except MapillaryDemoError as exc:
        safe_code = {
            "mapillary_token_not_configured": "MAPILLARY_TOKEN_NOT_CONFIGURED",
            "mapillary_token_source_invalid": "MAPILLARY_TOKEN_SOURCE_INVALID",
        }.get(exc.code, exc.code)
        _emit({"status": "error", "code": safe_code}, error=True)
        return 4
    except CorpusPipelineError as exc:
        _emit({"status": "error", "code": exc.code}, error=True)
        return 4
    except CorpusIndexError as exc:
        _emit(
            {"status": "error", "code": _safe_error_code(type(exc).__name__)},
            error=True,
        )
        return 4
    except (ValidationError, json.JSONDecodeError, TypeError, ValueError):
        _emit({"status": "error", "code": "configuration_or_input_invalid"}, error=True)
        return 2
    except OSError:
        _emit({"status": "error", "code": "filesystem_operation_failed"}, error=True)
        return 4
    _emit({"status": "ok", "command": args.command, **result})
    return 0


def _emit(payload: dict[str, object], *, error: bool = False) -> None:
    print(json.dumps(payload, sort_keys=True), file=sys.stderr if error else sys.stdout)


def _safe_error_code(name: str) -> str:
    characters = [character.lower() if character.isalnum() else "_" for character in name]
    return "".join(characters).strip("_") or "corpus_index_error"


def _load_config(path: Path) -> dict[str, Any]:
    value = _read_json(path, max_bytes=2 * 1024 * 1024)
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")
    return value


def _read_json(path: Path, *, max_bytes: int) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError("input file must be a regular non-symlink file")
    if path.stat().st_size > max_bytes:
        raise ValueError("input file exceeds its byte limit")
    return json.loads(path.read_text(encoding="utf-8"))


def _ingestion_config(config: dict[str, Any]) -> IngestionConfig:
    value = config.get("ingestion", {})
    if not isinstance(value, dict):
        raise ValueError("ingestion configuration must be an object")
    return IngestionConfig.model_validate(value)


def _pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("pipeline", {})
    if not isinstance(value, dict):
        raise ValueError("pipeline configuration must be an object")
    return value


def _workflow(args: argparse.Namespace, config: dict[str, Any]) -> CorpusWorkflow:
    return CorpusWorkflow(
        manifest_path=args.manifest,
        manifest_schema_path=args.manifest_schema,
        source_policy_path=args.source_policy,
        paths=CorpusWorkflowPaths(
            corpus_root=args.corpus_root,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint,
        ),
        ingestion_config=_ingestion_config(config),
    )


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be explicit non-empty text")
    return value


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _required_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be an explicit number")
    return float(value)


def _configured_provider(
    registry: ProductionDescriptorRegistry,
    pipeline: dict[str, Any],
) -> DescriptorProvider:
    provider = registry.active()
    configured_identity = _required_text(
        pipeline.get("production_descriptor_provider"),
        "production_descriptor_provider",
    )
    if configured_identity != provider.spec.identity:
        raise ProviderNotConfiguredError("configured provider identity is not active")
    return provider


def _load_holdout(path: Path) -> tuple[BenchmarkQuery, ...]:
    value = _read_json(path, max_bytes=32 * 1024 * 1024)
    if not isinstance(value, dict) or value.get("schema") != "atlaslens-locked-holdout-v1":
        raise ValueError("holdout schema is invalid")
    rows = value.get("queries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("holdout queries are required")
    queries: list[BenchmarkQuery] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("holdout query must be an object")
        relevant = row.get("relevant_asset_ids")
        descriptor = row.get("descriptor")
        if not isinstance(relevant, list) or not all(isinstance(item, str) for item in relevant):
            raise ValueError("holdout relevant asset IDs are invalid")
        if not isinstance(descriptor, list):
            raise ValueError("holdout descriptor is invalid")
        queries.append(
            BenchmarkQuery(
                query_id=_required_text(row.get("query_id"), "query_id"),
                descriptor=np.asarray(descriptor, dtype=np.float32),
                relevant_asset_ids=tuple(relevant),
                content_sha256=_required_text(row.get("content_sha256"), "content_sha256"),
                holdout_asset_id=_required_text(
                    row.get("holdout_asset_id"), "holdout_asset_id"
                ),
                province=_required_text(row.get("province"), "province"),
                latitude=_required_float(row.get("latitude"), "latitude"),
                longitude=_required_float(row.get("longitude"), "longitude"),
                source_id=_required_text(row.get("source_id"), "source_id"),
                contributor_id=_required_text(
                    row.get("contributor_id"), "contributor_id"
                ),
                capture_run_id=_required_text(
                    row.get("capture_run_id"), "capture_run_id"
                ),
                sequence_id=_required_text(row.get("sequence_id"), "sequence_id"),
                sampling_cell=_required_text(row.get("sampling_cell"), "sampling_cell"),
            )
        )
    return tuple(queries)


def _holdout_settings(
    args: argparse.Namespace,
    pipeline: dict[str, Any],
) -> tuple[tuple[BenchmarkQuery, ...], str]:
    selected_path = getattr(args, "holdout", None) or pipeline.get("holdout_path")
    if isinstance(selected_path, str):
        selected_path = Path(selected_path)
    if not isinstance(selected_path, Path):
        raise ValueError("holdout_path must be explicit")
    expected_hash = getattr(args, "expected_holdout_hash", None) or pipeline.get(
        "expected_holdout_hash"
    )
    expected = _required_text(expected_hash, "expected_holdout_hash")
    queries = _load_holdout(selected_path)
    if compute_locked_holdout_hash(queries) != expected:
        raise ValueError("holdout hash does not match the locked query set")
    return queries, expected


def _dispatch(
    args: argparse.Namespace,
    registry: ProductionDescriptorRegistry,
) -> dict[str, object]:
    if args.command in CAPTURE_COMMANDS:
        return dispatch_capture_command(args)
    if args.command in MAPILLARY_COMMANDS:
        access_token: str | None = None
        if args.command not in {"mapillary-plan", "mapillary-status"}:
            access_token = load_mapillary_access_token(
                Path(__file__).resolve().parents[4] / ".env"
            ).get_secret_value()
        return dispatch_mapillary_command(
            args,
            access_token=access_token,
        )
    if args.command == "inspect-checkpoint":
        checkpoint = CheckpointStore(args.checkpoint, Path("pipeline-state.json")).load()
        return {
            "run_id": checkpoint.run_id,
            "state": checkpoint.state.value,
            "last_successful_state": checkpoint.last_successful_state.value,
            "active_state": (
                checkpoint.active_state.value if checkpoint.active_state is not None else None
            ),
            "failure_code": checkpoint.failure_code,
            "artifact_keys": sorted(checkpoint.artifact_sha256),
            "history_events": len(checkpoint.history),
        }

    config = _load_config(args.config)
    pipeline = _pipeline_config(config)
    workflow = _workflow(args, config)
    if args.command == "validate-manifest":
        return _preparation_result(workflow)
    if args.command == "ingest":
        if args.dry_run:
            return {"dry_run": True, **_preparation_result(workflow)}
        ingestion_result = workflow.ingest(resume=args.resume)
        return {
            "state": ingestion_result.checkpoint.state.value,
            "accepted_assets": len(ingestion_result.corpus.assets),
            "rejected_assets": len(ingestion_result.corpus.rejections),
            "leakage_findings": len(ingestion_result.leakage_report.findings),
            "split_lock_sha256": ingestion_result.split_lock.lock_sha256,
        }
    if args.command == "check-leakage":
        corpus = load_ingested_corpus(args.work_dir)
        decision = filter_leakage(
            corpus.assets,
            near_duplicate_hamming_threshold=_ingestion_config(
                config
            ).near_duplicate_hamming_threshold,
            spatial_leakage_radius_m=_ingestion_config(config).spatial_leakage_radius_m,
        )
        return {
            "dry_run": bool(args.dry_run),
            "status": decision.report.status,
            "input_assets": decision.report.input_count,
            "accepted_assets": decision.report.accepted_count,
            "excluded_assets": decision.report.excluded_count,
        }
    if args.command == "build-descriptors":
        provider = _configured_provider(registry, pipeline)
        batch_size = _positive_int(
            args.batch_size or pipeline.get("descriptor_batch_size"),
            "descriptor_batch_size",
        )
        if args.dry_run:
            corpus = load_ingested_corpus(args.work_dir)
            return {
                "dry_run": True,
                "asset_count": len(corpus.assets),
                "descriptor_spec": provider.spec.identity,
                "batch_size": batch_size,
            }
        descriptor_result = workflow.build_descriptors(
            provider,
            batch_size=batch_size,
            resume=args.resume,
        )
        return {
            "descriptor_count": descriptor_result.dataset.count,
            "generated_assets": descriptor_result.generated_assets,
            "resumed_assets": descriptor_result.resumed_assets,
            "descriptor_spec": descriptor_result.dataset.spec.identity,
        }
    if args.command == "build-index":
        queries, holdout_hash = _holdout_settings(args, pipeline)
        backend = args.backend or pipeline.get("index_backend")
        if backend not in {"exact", "faiss"}:
            raise ValueError("index_backend must be exact or faiss")
        index_version = _required_text(
            args.index_version or pipeline.get("index_version"),
            "index_version",
        )
        shard_size = _positive_int(
            args.shard_size or pipeline.get("shard_size"),
            "shard_size",
        )
        if args.dry_run:
            dataset = DescriptorDataset.open(workflow.paths.descriptor_output_dir)
            split_lock_hash = workflow.validate_holdout(
                queries,
                expected_holdout_hash=holdout_hash,
            )
            return {
                "dry_run": True,
                "descriptor_count": dataset.count,
                "holdout_queries": len(queries),
                "backend": backend,
                "split_lock_hash": split_lock_hash,
            }
        index = workflow.build_index(
            queries,
            backend=backend,
            index_version=index_version,
            shard_size=shard_size,
            locked_holdout_hash=holdout_hash,
            resume=args.resume,
        )
        return {"index_assets": index.size, "backend": index.backend}
    if args.command == "evaluate":
        queries, holdout_hash = _holdout_settings(args, pipeline)
        threshold = _abstention_threshold(args, pipeline)
        if args.dry_run:
            split_lock_hash = workflow.validate_holdout(
                queries,
                expected_holdout_hash=holdout_hash,
            )
            index = PublishedCorpusIndex.open(
                workflow.paths.index_output_dir,
                expected_locked_holdout_hash=holdout_hash,
                expected_split_lock_hash=split_lock_hash,
            )
            return {
                "dry_run": True,
                "index_assets": index.size,
                "holdout_queries": len(queries),
            }
        benchmark_result = workflow.evaluate(
            queries,
            expected_holdout_hash=holdout_hash,
            abstain_if_distance_gt=threshold,
            resume=args.resume,
        )
        return {"benchmark": benchmark_result.to_json()}
    if args.command == "run-pipeline":
        if args.dry_run:
            return {"dry_run": True, **_preparation_result(workflow)}
        provider = _configured_provider(registry, pipeline)
        queries, holdout_hash = _holdout_settings(args, pipeline)
        workflow_result = workflow.run(
            provider,
            queries,
            descriptor_batch_size=_positive_int(
                pipeline.get("descriptor_batch_size"),
                "descriptor_batch_size",
            ),
            index_backend=_required_text(pipeline.get("index_backend"), "index_backend"),
            index_version=_required_text(pipeline.get("index_version"), "index_version"),
            index_shard_size=_positive_int(pipeline.get("shard_size"), "shard_size"),
            expected_holdout_hash=holdout_hash,
            abstain_if_distance_gt=_abstention_threshold(args, pipeline),
            resume=args.resume,
        )
        return {
            "state": workflow_result.checkpoint.state.value,
            "accepted_assets": workflow_result.accepted_assets,
            "rejected_assets": workflow_result.rejected_assets,
            "generated_descriptors": workflow_result.generated_descriptors,
            "resumed_descriptors": workflow_result.resumed_descriptors,
            "index_assets": workflow_result.index_assets,
            "benchmark": workflow_result.benchmark.to_json(),
        }
    raise ValueError("unsupported corpus command")


def _preparation_result(workflow: CorpusWorkflow) -> dict[str, object]:
    preparation = workflow.prepare()
    return {
        "accepted_assets": len(preparation.corpus.assets),
        "rejected_assets": len(preparation.corpus.rejections),
        "leakage_findings": len(preparation.leakage_report.findings),
        "split_assignments": len(preparation.split_lock.assignments),
    }


def _abstention_threshold(
    args: argparse.Namespace,
    pipeline: dict[str, Any],
) -> float | None:
    value = getattr(args, "abstain_if_distance_gt", None)
    if value is None:
        value = pipeline.get("abstain_if_distance_gt")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("abstain_if_distance_gt must be numeric or null")
    selected = float(value)
    if not 0.0 <= selected <= 2.0:
        raise ValueError("abstain_if_distance_gt must be between zero and two")
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
