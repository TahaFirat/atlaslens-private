from __future__ import annotations

import json
import re
from pathlib import Path

from atlaslens_api.evaluation.metrics import summarize
from atlaslens_api.evaluation.models import BenchmarkSummary, PerImageResult, RatioMetric
from atlaslens_api.schemas import (
    EvaluationReportList,
    EvaluationReportSummary,
    RatioView,
)

_REPORT_ID = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
_SIMULATED_PROVIDER = re.compile(
    r"(^|[-_.])(mock|simulated|simulation|fixture|test)([-_.]|$)", re.IGNORECASE
)
_MAX_REPORT_BYTES = 20 * 1024 * 1024
_RECALL_KEYS = ("1", "25", "100", "200", "500", "750", "2500")


class EvaluationCatalogError(ValueError):
    pass


def _ratio(metric: RatioMetric) -> RatioView:
    return RatioView(
        numerator=metric.numerator,
        denominator=metric.denominator,
        value=metric.value,
    )


def _safe_identity(value: str) -> str:
    if (
        not value.strip()
        or len(value) > 160
        or any(character in value for character in ("/", "\\", "\r", "\n", "\x00"))
    ):
        raise EvaluationCatalogError("evaluation_report_identity_unsafe")
    return value


def _safe_group(value: str) -> str:
    if (
        not value.strip()
        or len(value) > 160
        or any(character in value for character in ("/", "\\", "\r", "\n", "\x00"))
    ):
        raise EvaluationCatalogError("evaluation_report_group_unsafe")
    return value


class EvaluationReportCatalog:
    def __init__(self, private_root: Path) -> None:
        expanded = private_root.expanduser()
        if expanded.is_symlink():
            raise EvaluationCatalogError("evaluation_catalog_root_unsafe")
        try:
            self._root = expanded.resolve(strict=True)
        except OSError as exc:
            raise EvaluationCatalogError("evaluation_catalog_unavailable") from exc
        if not self._root.is_dir():
            raise EvaluationCatalogError("evaluation_catalog_unavailable")

    def get(self, report_id: str) -> EvaluationReportSummary:
        if not _REPORT_ID.fullmatch(report_id):
            raise EvaluationCatalogError("evaluation_report_id_invalid")
        directory = self._root / report_id
        report_path = directory / "benchmark.json"
        if directory.is_symlink() or report_path.is_symlink():
            raise EvaluationCatalogError("evaluation_catalog_entry_unsafe")
        try:
            resolved = report_path.resolve(strict=True)
            resolved.relative_to(self._root)
        except (OSError, ValueError) as exc:
            raise EvaluationCatalogError("evaluation_report_unavailable") from exc
        if not resolved.is_file() or resolved.stat().st_size > _MAX_REPORT_BYTES:
            raise EvaluationCatalogError("evaluation_report_unavailable")
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationCatalogError("evaluation_report_invalid") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
            raise EvaluationCatalogError("evaluation_report_invalid")
        if payload.get("result_classification") == "simulated":
            raise EvaluationCatalogError("simulated_evaluation_report_rejected")
        try:
            summary = BenchmarkSummary.model_validate(payload["summary"])
            raw_results = payload.get("results")
            if not isinstance(raw_results, list):
                raise ValueError("results missing")
            results = tuple(PerImageResult.model_validate(item) for item in raw_results)
        except (TypeError, ValueError) as exc:
            raise EvaluationCatalogError("evaluation_report_invalid") from exc
        provider_id = _safe_identity(summary.provider_id)
        model_revision = _safe_identity(summary.model_revision)
        if _SIMULATED_PROVIDER.search(provider_id):
            raise EvaluationCatalogError("simulated_evaluation_report_rejected")
        if len(results) != summary.image_count:
            raise EvaluationCatalogError("evaluation_report_count_mismatch")
        recomputed = summarize(
            results,
            provider_id=summary.provider_id,
            model_revision=summary.model_revision,
            manifest_fingerprint=summary.manifest_fingerprint,
        )
        if recomputed != summary:
            raise EvaluationCatalogError("evaluation_report_summary_mismatch")

        limitations = [
            "prebuilt_benchmark_report",
            "evaluation_measurements_are_not_location_guarantees",
        ]
        raw_calibration = payload.get("calibration_state")
        if raw_calibration is None:
            calibration_state = "uncalibrated"
            limitations.append("calibration_state_not_recorded_assumed_uncalibrated")
        elif raw_calibration in {"uncalibrated", "preliminary", "calibrated"}:
            calibration_state = raw_calibration
        else:
            raise EvaluationCatalogError("evaluation_calibration_state_invalid")
        recall: dict[str, RatioView] = {}
        for key in _RECALL_KEYS:
            metric = summary.recall_top1.get(key)
            if metric is None:
                recall[key] = RatioView(numerator=0, denominator=0, value=None)
                limitations.append(f"recall_{key}km_unavailable")
            else:
                recall[key] = _ratio(metric)
        continents = summary.breakdowns.get("continent", {})
        scenes = summary.breakdowns.get("scene_category", {})
        return EvaluationReportSummary(
            report_id=report_id,
            provider_id=provider_id,
            model_revision=model_revision,
            evaluation_fingerprint=summary.manifest_fingerprint,
            image_count=summary.image_count,
            calibration_state=calibration_state,
            country_top1=_ratio(summary.country_top1),
            country_top5=_ratio(summary.country_top5),
            region_top1=_ratio(summary.region_top1),
            city_top1=_ratio(summary.city_or_area_top1),
            recall_top1=recall,
            mean_error_km=summary.top1_error.mean_km,
            median_error_km=summary.top1_error.median_km,
            p95_error_km=summary.top1_error.p95_km,
            abstention=_ratio(summary.abstention),
            provider_failure=_ratio(summary.provider_failure),
            latency_median_ms=summary.latency_ms.median_ms,
            latency_p95_ms=summary.latency_ms.p95_ms,
            uncertainty_coverage=_ratio(summary.uncertainty_coverage),
            geographic_distribution={
                _safe_group(key): value.image_count for key, value in sorted(continents.items())
            },
            scene_distribution={
                _safe_group(key): value.image_count for key, value in sorted(scenes.items())
            },
            exclusions=dict(sorted(summary.exclusions.items())),
            limitations=limitations,
        )

    def list(self) -> EvaluationReportList:
        reports = []
        for directory in sorted(self._root.iterdir(), key=lambda item: item.name):
            if directory.is_symlink():
                raise EvaluationCatalogError("evaluation_catalog_entry_unsafe")
            if directory.is_dir() and (directory / "benchmark.json").is_file():
                try:
                    reports.append(self.get(directory.name))
                except EvaluationCatalogError as exc:
                    if str(exc) == "simulated_evaluation_report_rejected":
                        continue
                    raise
        return EvaluationReportList(reports=reports)

    def get_report(self, report_id: str) -> EvaluationReportSummary:
        return self.get(report_id)

    def list_reports(self) -> EvaluationReportList:
        return self.list()
