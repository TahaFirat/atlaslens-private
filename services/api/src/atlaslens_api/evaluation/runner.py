from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from atlaslens_api.evaluation.metrics import haversine_km, summarize
from atlaslens_api.evaluation.models import (
    BenchmarkSummary,
    CloseableEvaluationProvider,
    EvaluationProvider,
    PerImageResult,
    ProviderPrediction,
    ValidatedEvaluationAsset,
    ValidatedManifest,
)
from atlaslens_api.evaluation.reports import BenchmarkReportWriter


def _opaque_report_id(namespace: str, value: str) -> str:
    digest = hashlib.sha256(f"{namespace}\x00{value}".encode()).hexdigest()[:20]
    return f"{namespace}-{digest}"


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    summary: BenchmarkSummary
    results: tuple[PerImageResult, ...]
    report_paths: tuple[Path, Path, Path] | None = None


class BenchmarkRunner:
    def __init__(self, report_writer: BenchmarkReportWriter | None = None) -> None:
        self._report_writer = report_writer or BenchmarkReportWriter()

    def run(
        self,
        manifest: ValidatedManifest,
        provider: EvaluationProvider,
        *,
        output_directory: Path | None = None,
    ) -> BenchmarkRun:
        try:
            results = tuple(self._evaluate(asset, provider) for asset in manifest.assets)
        finally:
            if isinstance(provider, CloseableEvaluationProvider):
                provider.close()
        summary = summarize(
            results,
            provider_id=provider.provider_id,
            model_revision=provider.model_revision,
            manifest_fingerprint=manifest.report.fingerprint,
        )
        paths = (
            self._report_writer.write(output_directory, summary, results)
            if output_directory is not None
            else None
        )
        return BenchmarkRun(summary=summary, results=results, report_paths=paths)

    @staticmethod
    def _evaluate(
        asset: ValidatedEvaluationAsset, provider: EvaluationProvider
    ) -> PerImageResult:
        started = time.perf_counter()
        try:
            prediction = provider.predict(asset.path)
        except Exception:
            prediction = ProviderPrediction(
                failure_code="provider_exception",
                latency_ms=0,
                device="other",
            )
        measured_latency_ms = (time.perf_counter() - started) * 1000
        record = asset.record
        candidates = prediction.candidates
        top1 = candidates[0] if candidates else None
        errors = [
            haversine_km(record.true_latitude, record.true_longitude, item.latitude, item.longitude)
            for item in candidates[:5]
        ]
        top1_error = errors[0] if errors else None
        top5_error = min(errors) if errors else None
        uncertainty_contains = (
            top1_error <= top1.uncertainty_radius_km
            if top1 is not None
            and top1_error is not None
            and top1.uncertainty_radius_km is not None
            else None
        )
        return PerImageResult(
            image_asset_key=_opaque_report_id("asset", record.image_asset_key),
            source_record_id=_opaque_report_id("source", record.source_record_id),
            split=record.split,
            scene_category=record.scene_category,
            continent=record.continent,
            country_code=record.country_code,
            returned_candidates=len(candidates),
            abstained=prediction.abstained,
            failure_code=prediction.failure_code,
            device=prediction.device,
            latency_ms=measured_latency_ms,
            top1_raw_score=top1.raw_score if top1 else None,
            top1_error_km=top1_error,
            top5_oracle_error_km=top5_error,
            country_top1_correct=(
                top1.country_code == record.country_code
                if top1 is not None and top1.country_code is not None
                else False
                if top1 is None
                else None
            ),
            country_top5_correct=(
                any(item.country_code == record.country_code for item in candidates[:5])
                if any(item.country_code is not None for item in candidates[:5])
                else False
                if not candidates
                else None
            ),
            region_top1_correct=(
                (
                    top1.region == record.region
                    if top1 is not None and top1.region is not None
                    else False
                    if top1 is None
                    else None
                )
                if record.region is not None
                else None
            ),
            city_or_area_top1_correct=(
                (
                    top1.city_or_area == record.city_or_area
                    if top1 is not None and top1.city_or_area is not None
                    else False
                    if top1 is None
                    else None
                )
                if record.city_or_area is not None
                else None
            ),
            uncertainty_contains_truth=uncertainty_contains,
        )
