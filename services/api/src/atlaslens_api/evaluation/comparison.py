from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from atlaslens_api.evaluation.metrics import summarize
from atlaslens_api.evaluation.models import (
    BenchmarkSummary,
    EvaluationModel,
    ManifestValidationReport,
    PerImageResult,
)

AblationMode = Literal[
    "geoclip_only",
    "ocr",
    "retrieval",
    "ocr_retrieval",
    "full",
]
AblationStatus = Literal["available", "unavailable"]
UnavailableReason = Literal["report_not_supplied"]

ABLATION_MODE_ORDER: tuple[AblationMode, ...] = (
    "geoclip_only",
    "ocr",
    "retrieval",
    "ocr_retrieval",
    "full",
)
ABLATION_MODE_LABELS: dict[AblationMode, str] = {
    "geoclip_only": "GeoCLIP-only",
    "ocr": "+OCR",
    "retrieval": "+retrieval",
    "ocr_retrieval": "+OCR+retrieval",
    "full": "full",
}
COMPARISON_LIMITATIONS = (
    "Modes consume precomputed benchmark reports; this command does not run providers.",
    "Mode assignments are operator-declared and retain each source report's provider identity.",
    "Unavailable modes contain no metrics, and no cross-mode deltas are inferred.",
    "Results are evaluation measurements, not calibration, causality, or location guarantees.",
)
PROVIDER_COMPARISON_LIMITATIONS = (
    "Both providers use the same validated manifest fingerprint and sample count.",
    "The report preserves measured summaries and does not infer calibration or causality.",
    "Failures, abstentions and unavailable label metrics remain in their source summaries.",
    "Results are evaluation measurements, not individual-location guarantees.",
)

_MAX_REPORT_BYTES = 20 * 1024 * 1024
_SIMULATED_PROVIDER = re.compile(
    r"(^|[-_.])(mock|simulated|simulation|fixture|test)([-_.]|$)", re.IGNORECASE
)


class AblationModeReport(EvaluationModel):
    mode: AblationMode
    label: str = Field(min_length=1, max_length=40)
    status: AblationStatus
    summary: BenchmarkSummary | None = None
    unavailable_reason: UnavailableReason | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> AblationModeReport:
        if self.label != ABLATION_MODE_LABELS[self.mode]:
            raise ValueError("ablation mode label does not match its mode")
        if self.status == "available":
            if self.summary is None or self.unavailable_reason is not None:
                raise ValueError("available ablation mode requires exactly one summary")
        elif self.summary is not None or self.unavailable_reason is None:
            raise ValueError("unavailable ablation mode cannot contain metrics")
        return self


class BenchmarkComparisonReport(EvaluationModel):
    schema_version: Literal["atlaslens-phase5b-comparison-v1"] = "atlaslens-phase5b-comparison-v1"
    baseline: Literal["geoclip"] = "geoclip"
    candidate: Literal["phase5b-v1"] = "phase5b-v1"
    manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_count: int = Field(gt=0)
    modes: tuple[AblationModeReport, ...]
    limitations: tuple[str, ...] = COMPARISON_LIMITATIONS

    @model_validator(mode="after")
    def validate_comparison(self) -> BenchmarkComparisonReport:
        if tuple(item.mode for item in self.modes) != ABLATION_MODE_ORDER:
            raise ValueError("ablation modes must be complete and ordered")
        if self.modes[0].status != "available":
            raise ValueError("GeoCLIP baseline report is required")
        if self.limitations != COMPARISON_LIMITATIONS:
            raise ValueError("comparison limitations cannot be removed")
        for item in self.modes:
            if item.summary is None:
                continue
            if item.summary.manifest_fingerprint != self.manifest_fingerprint:
                raise ValueError("benchmark report manifest fingerprint mismatch")
            if item.summary.image_count != self.image_count:
                raise ValueError("benchmark report image count mismatch")
        return self


class PairedProviderComparisonReport(EvaluationModel):
    schema_version: Literal["atlaslens-provider-comparison-v1"] = (
        "atlaslens-provider-comparison-v1"
    )
    manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_count: int = Field(gt=0)
    baseline: BenchmarkSummary
    candidate: BenchmarkSummary
    limitations: tuple[str, ...] = PROVIDER_COMPARISON_LIMITATIONS

    @model_validator(mode="after")
    def validate_pair(self) -> PairedProviderComparisonReport:
        if self.baseline.provider_id == self.candidate.provider_id:
            raise ValueError("provider comparison requires distinct providers")
        for summary in (self.baseline, self.candidate):
            if summary.manifest_fingerprint != self.manifest_fingerprint:
                raise ValueError("benchmark report manifest fingerprint mismatch")
            if summary.image_count != self.image_count:
                raise ValueError("benchmark report image count mismatch")
        if self.limitations != PROVIDER_COMPARISON_LIMITATIONS:
            raise ValueError("provider comparison limitations cannot be removed")
        return self


def read_benchmark_summary(output_directory: Path) -> BenchmarkSummary:
    requested = output_directory.expanduser()
    if requested.is_symlink():
        raise ValueError("benchmark output symlinks are not accepted")
    candidate = requested / "benchmark.json"
    if candidate.is_symlink():
        raise ValueError("benchmark report symlinks are not accepted")
    report = candidate.resolve(strict=True)
    if not report.is_file() or report.stat().st_size > _MAX_REPORT_BYTES:
        raise ValueError("benchmark report is unavailable")
    payload = json.loads(report.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("summary"), dict)
        or not isinstance(payload.get("results"), list)
    ):
        raise ValueError("benchmark report is invalid")
    summary = BenchmarkSummary.model_validate(payload["summary"])
    if _SIMULATED_PROVIDER.search(summary.provider_id):
        raise ValueError("simulated benchmark report is rejected")
    results = tuple(PerImageResult.model_validate(item) for item in payload["results"])
    if len(results) != summary.image_count:
        raise ValueError("benchmark report image count mismatch")
    recomputed = summarize(
        results,
        provider_id=summary.provider_id,
        model_revision=summary.model_revision,
        manifest_fingerprint=summary.manifest_fingerprint,
    )
    if recomputed != summary:
        raise ValueError("benchmark report summary mismatch")
    return summary


class BenchmarkComparisonBuilder:
    def build(
        self,
        manifest: ManifestValidationReport,
        report_directories: Mapping[AblationMode, Path | None],
        *,
        baseline: Literal["geoclip"] = "geoclip",
        candidate: Literal["phase5b-v1"] = "phase5b-v1",
    ) -> BenchmarkComparisonReport:
        unknown = set(report_directories) - set(ABLATION_MODE_ORDER)
        if unknown:
            raise ValueError("unknown ablation mode")
        if report_directories.get("geoclip_only") is None:
            raise ValueError("GeoCLIP baseline report is required")

        modes: list[AblationModeReport] = []
        for mode in ABLATION_MODE_ORDER:
            directory = report_directories.get(mode)
            if directory is None:
                modes.append(
                    AblationModeReport(
                        mode=mode,
                        label=ABLATION_MODE_LABELS[mode],
                        status="unavailable",
                        unavailable_reason="report_not_supplied",
                    )
                )
                continue
            summary = read_benchmark_summary(directory)
            if summary.manifest_fingerprint != manifest.fingerprint:
                raise ValueError("benchmark report manifest fingerprint mismatch")
            if summary.image_count != manifest.image_count:
                raise ValueError("benchmark report image count mismatch")
            modes.append(
                AblationModeReport(
                    mode=mode,
                    label=ABLATION_MODE_LABELS[mode],
                    status="available",
                    summary=summary,
                )
            )
        return BenchmarkComparisonReport(
            baseline=baseline,
            candidate=candidate,
            manifest_fingerprint=manifest.fingerprint,
            image_count=manifest.image_count,
            modes=tuple(modes),
        )


class PairedProviderComparisonBuilder:
    def build(
        self,
        manifest: ManifestValidationReport,
        *,
        baseline_directory: Path,
        candidate_directory: Path,
    ) -> PairedProviderComparisonReport:
        baseline = read_benchmark_summary(baseline_directory)
        candidate = read_benchmark_summary(candidate_directory)
        return PairedProviderComparisonReport(
            manifest_fingerprint=manifest.fingerprint,
            image_count=manifest.image_count,
            baseline=baseline,
            candidate=candidate,
        )


class BenchmarkComparisonWriter:
    def write(self, output_directory: Path, report: BenchmarkComparisonReport) -> tuple[Path, Path]:
        if output_directory.is_symlink():
            raise ValueError("comparison output directory is unsafe")
        output_directory.mkdir(parents=True, exist_ok=True)
        if output_directory.is_symlink() or not output_directory.is_dir():
            raise ValueError("comparison output directory is unsafe")
        json_path = output_directory / "comparison.json"
        markdown_path = output_directory / "comparison.md"
        self._atomic_text(
            json_path,
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        )
        self._atomic_text(markdown_path, self._markdown(report))
        return json_path, markdown_path

    def write_provider_comparison(
        self, output_directory: Path, report: PairedProviderComparisonReport
    ) -> tuple[Path, Path]:
        if output_directory.is_symlink():
            raise ValueError("comparison output directory is unsafe")
        output_directory.mkdir(parents=True, exist_ok=True)
        if output_directory.is_symlink() or not output_directory.is_dir():
            raise ValueError("comparison output directory is unsafe")
        json_path = output_directory / "comparison.json"
        markdown_path = output_directory / "comparison.md"
        self._atomic_text(
            json_path,
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        )
        self._atomic_text(markdown_path, self._provider_markdown(report))
        return json_path, markdown_path

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        if path.is_symlink():
            raise ValueError("comparison output file is unsafe")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _markdown(report: BenchmarkComparisonReport) -> str:
        def cell(value: object) -> str:
            return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

        def ratio(summary: BenchmarkSummary, name: str) -> str:
            metric = getattr(summary, name)
            value = "N/A" if metric.value is None else f"{metric.value:.6f}"
            return f"{value} ({metric.numerator}/{metric.denominator})"

        lines = [
            "# AtlasLens benchmark comparison",
            "",
            f"- Baseline: `{report.baseline}`",
            f"- Candidate: `{report.candidate}`",
            f"- Manifest fingerprint: `{report.manifest_fingerprint}`",
            f"- Image count: {report.image_count}",
            "",
            "| Mode | Status | Provider | Model revision | Candidate return | "
            "Country Top-1 | Median Top-1 error (km) |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for item in report.modes:
            summary = item.summary
            if summary is None:
                lines.append(f"| {cell(item.label)} | unavailable | N/A | N/A | N/A | N/A | N/A |")
                continue
            median_error = (
                "N/A"
                if summary.top1_error.median_km is None
                else f"{summary.top1_error.median_km:.6f}"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        cell(item.label),
                        "available",
                        cell(summary.provider_id),
                        cell(summary.model_revision),
                        ratio(summary, "candidate_return"),
                        ratio(summary, "country_top1"),
                        median_error,
                    )
                )
                + " |"
            )
        lines.extend(("", "## Limitations", ""))
        lines.extend(f"- {item}" for item in report.limitations)
        return "\n".join(lines)

    @staticmethod
    def _provider_markdown(report: PairedProviderComparisonReport) -> str:
        def ratio(summary: BenchmarkSummary, name: str) -> str:
            metric = getattr(summary, name)
            value = "N/A" if metric.value is None else f"{metric.value:.6f}"
            return f"{value} ({metric.numerator}/{metric.denominator})"

        lines = [
            "# AtlasLens paired provider benchmark comparison",
            "",
            f"- Manifest fingerprint: `{report.manifest_fingerprint}`",
            f"- Image count: {report.image_count}",
            "",
            "| Role | Provider | Model revision | Candidate return | Country Top-1 | "
            "Median Top-1 error (km) |",
            "|---|---|---|---:|---:|---:|",
        ]
        for role, summary in (("baseline", report.baseline), ("candidate", report.candidate)):
            median_error = (
                "N/A"
                if summary.top1_error.median_km is None
                else f"{summary.top1_error.median_km:.6f}"
            )
            lines.append(
                f"| {role} | {summary.provider_id} | {summary.model_revision} | "
                f"{ratio(summary, 'candidate_return')} | {ratio(summary, 'country_top1')} | "
                f"{median_error} |"
            )
        lines.extend(("", "## Limitations", ""))
        lines.extend(f"- {item}" for item in report.limitations)
        return "\n".join(lines)
