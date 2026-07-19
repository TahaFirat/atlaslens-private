from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from atlaslens_api.evaluation.adapter import (
    GlobalProviderEvaluationAdapter,
    InferenceProviderEvaluationAdapter,
)
from atlaslens_api.evaluation.comparison import (
    AblationMode,
    BenchmarkComparisonBuilder,
    BenchmarkComparisonWriter,
    PairedProviderComparisonBuilder,
    read_benchmark_summary,
)
from atlaslens_api.evaluation.manifest import EvaluationManifestError, EvaluationManifestLoader
from atlaslens_api.evaluation.models import (
    BenchmarkSummary,
    EvaluationProvider,
    ManifestValidationReport,
    ValidatedManifest,
)
from atlaslens_api.evaluation.runner import BenchmarkRunner
from atlaslens_api.gazetteer import (
    GazetteerManager,
    GazetteerMetadata,
    SQLiteGazetteerResolver,
)
from atlaslens_api.gazetteer.cli import default_cache_root as default_gazetteer_cache_root
from atlaslens_api.model_management.cli import default_cache_root
from atlaslens_api.providers.registry import build_global_provider_registry
from atlaslens_api.trained_artifacts.integration import build_custom_provider
from atlaslens_api.trained_artifacts.manager import TrainedArtifactManager


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "compare"):
        item = commands.add_parser(name)
        item.add_argument("--manifest", type=Path, required=True)
        item.add_argument("--asset-root", type=Path, required=True)
        item.add_argument("--allow-license", action="append", required=True)
    run = commands.choices["run"]
    run.add_argument(
        "--provider",
        choices=("geoclip", "atlaslens-custom-geolocation"),
        default="geoclip",
    )
    run.add_argument("--model-cache", type=Path, default=default_cache_root())
    run.add_argument("--custom-model-id", default="atlaslens-custom-geolocation")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--gazetteer-cache", type=Path, default=default_gazetteer_cache_root())
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--split", choices=("calibration", "validation", "test"), default="test")
    compare = commands.choices["compare"]
    compare.add_argument("--baseline", choices=("geoclip",))
    compare.add_argument("--candidate", choices=("phase5b-v1",))
    compare.add_argument("--providers")
    compare.add_argument("--geoclip-report", type=Path, required=True)
    compare.add_argument("--ocr-report", type=Path)
    compare.add_argument("--retrieval-report", type=Path)
    compare.add_argument("--ocr-retrieval-report", type=Path)
    compare.add_argument("--full-report", type=Path)
    compare.add_argument("--custom-report", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--split", choices=("calibration", "validation", "test"), default="test")
    for name in ("report", "info"):
        item = commands.add_parser(name)
        item.add_argument("--output", type=Path, required=True)
    return parser


def _load(args: argparse.Namespace) -> ValidatedManifest:
    licenses = frozenset(str(item) for item in args.allow_license)
    return EvaluationManifestLoader(allowed_licenses=licenses).load(args.manifest, args.asset_root)


def _select_split(manifest: ValidatedManifest, split: str) -> ValidatedManifest:
    assets = tuple(asset for asset in manifest.assets if asset.record.split == split)
    if not assets:
        raise EvaluationManifestError("selected evaluation split is empty")
    canonical = [asset.record.model_dump(mode="json") for asset in assets]
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = ManifestValidationReport(
        fingerprint=fingerprint,
        image_count=len(assets),
        split_counts={split: len(assets)},
        continent_counts=dict(Counter(asset.record.continent for asset in assets)),
        country_counts=dict(Counter(asset.record.country_code for asset in assets)),
        scene_counts=dict(Counter(asset.record.scene_category for asset in assets)),
        warnings=manifest.report.warnings,
    )
    return ValidatedManifest(assets=assets, report=report)


def _gazetteer(gazetteer_cache_root: Path) -> SQLiteGazetteerResolver | None:
    manager = GazetteerManager(gazetteer_cache_root)
    info = manager.info()
    if info.status == "ready" and info.schema_version is not None and info.license is not None:
        return SQLiteGazetteerResolver(
            manager.database_path,
            GazetteerMetadata(
                dataset=info.dataset,
                version=info.schema_version,
                source="https://download.geonames.org/export/dump/",
                license=info.license,
            ),
        )
    return None


def _real_provider(
    cache_root: Path,
    gazetteer_cache_root: Path,
    *,
    provider_name: str,
    custom_model_id: str,
    device: str,
) -> EvaluationProvider:
    gazetteer = _gazetteer(gazetteer_cache_root)
    if provider_name == "geoclip":
        registry = build_global_provider_registry(cache_root)
        return GlobalProviderEvaluationAdapter(
            registry.get("geoclip-global-v1"), gazetteer=gazetteer
        )
    if provider_name != "atlaslens-custom-geolocation" or device not in {"cpu", "cuda"}:
        raise ValueError("unsupported evaluation provider")
    custom = build_custom_provider(
        TrainedArtifactManager(cache_root),
        custom_model_id,
        enabled=True,
        device=cast(Literal["cpu", "cuda"], device),
        max_input_bytes=100 * 1024 * 1024,
    )
    return InferenceProviderEvaluationAdapter(custom, gazetteer=gazetteer)


def _paired_provider_comparison(args: argparse.Namespace) -> bool:
    if args.providers:
        if args.baseline is not None or args.candidate is not None:
            raise ValueError("providers cannot be combined with baseline/candidate")
        providers = tuple(item.strip() for item in str(args.providers).split(","))
        if providers != ("geoclip", "atlaslens-custom-geolocation"):
            raise ValueError("unsupported provider comparison")
        return True
    if args.baseline is None or args.candidate is None:
        raise ValueError("baseline and candidate are required")
    return False


def _read_report(output: Path) -> BenchmarkSummary:
    return read_benchmark_summary(output)


def run_benchmark_command(
    argv: Sequence[str], *, provider: EvaluationProvider | None = None
) -> dict[str, object]:
    args = build_benchmark_parser().parse_args(argv)
    if args.command == "validate":
        return {"status": "valid", "manifest": _load(args).report.model_dump(mode="json")}
    if args.command == "run":
        manifest = _select_split(_load(args), args.split)
        selected_provider = provider or _real_provider(
            args.model_cache,
            args.gazetteer_cache,
            provider_name=args.provider,
            custom_model_id=args.custom_model_id,
            device=args.device,
        )
        run = BenchmarkRunner().run(manifest, selected_provider, output_directory=args.output)
        return {
            "status": "completed",
            "summary": run.summary.model_dump(mode="json"),
            "reports": ["benchmark.json", "per-image.csv", "summary.md"],
        }
    if args.command == "compare":
        manifest = _select_split(_load(args), args.split)
        paired = _paired_provider_comparison(args)
        if paired:
            if args.custom_report is None or args.full_report is not None:
                raise ValueError("exactly one custom provider report is required")
            paired_comparison = PairedProviderComparisonBuilder().build(
                manifest.report,
                baseline_directory=args.geoclip_report,
                candidate_directory=args.custom_report,
            )
            BenchmarkComparisonWriter().write_provider_comparison(
                args.output, paired_comparison
            )
            return {
                "status": "completed",
                "comparison": paired_comparison.model_dump(mode="json"),
                "reports": ["comparison.json", "comparison.md"],
            }
        if args.custom_report is not None:
            raise ValueError("custom report requires providers mode")
        directories: dict[AblationMode, Path | None] = {
            "geoclip_only": args.geoclip_report,
            "ocr": args.ocr_report,
            "retrieval": args.retrieval_report,
            "ocr_retrieval": args.ocr_retrieval_report,
            "full": args.full_report,
        }
        comparison = BenchmarkComparisonBuilder().build(
            manifest.report,
            directories,
            baseline=cast(Literal["geoclip"], args.baseline),
            candidate=cast(Literal["phase5b-v1"], args.candidate),
        )
        BenchmarkComparisonWriter().write(args.output, comparison)
        return {
            "status": "completed",
            "comparison": comparison.model_dump(mode="json"),
            "reports": ["comparison.json", "comparison.md"],
        }
    summary = _read_report(args.output)
    if args.command == "report":
        return {"status": "completed", "summary": summary.model_dump(mode="json")}
    return {
        "status": "available",
        "provider_id": summary.provider_id,
        "model_revision": summary.model_revision,
        "manifest_fingerprint": summary.manifest_fingerprint,
        "image_count": summary.image_count,
        "reports": ["benchmark.json", "per-image.csv", "summary.md"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run_benchmark_command(sys.argv[1:] if argv is None else argv)
    except (EvaluationManifestError, OSError, ValueError, json.JSONDecodeError):
        print(
            json.dumps({"status": "error", "code": "benchmark_operation_failed"}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
