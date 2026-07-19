from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from atlaslens_api.evaluation.models import BenchmarkSummary, PerImageResult


class BenchmarkReportWriter:
    def write(
        self,
        output_directory: Path,
        summary: BenchmarkSummary,
        results: tuple[PerImageResult, ...],
    ) -> tuple[Path, Path, Path]:
        output_directory.mkdir(parents=True, exist_ok=True)
        if output_directory.is_symlink() or not output_directory.is_dir():
            raise ValueError("benchmark output directory is unsafe")
        json_path = output_directory / "benchmark.json"
        csv_path = output_directory / "per-image.csv"
        markdown_path = output_directory / "summary.md"
        self._atomic_text(
            json_path,
            json.dumps(
                {
                    "summary": summary.model_dump(mode="json"),
                    "results": [result.model_dump(mode="json") for result in results],
                },
                indent=2,
                sort_keys=True,
            ),
        )
        fields = list(PerImageResult.model_fields)
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for result in results:
                writer.writerow(result.model_dump(mode="json"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_csv, csv_path)
        self._atomic_text(markdown_path, self._markdown(summary))
        return json_path, csv_path, markdown_path

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _markdown(summary: BenchmarkSummary) -> str:
        def metric(name: str) -> str:
            item = getattr(summary, name)
            value = "N/A" if item.value is None else f"{item.value:.6f}"
            return f"{value} ({item.numerator}/{item.denominator})"

        return "\n".join(
            [
                "# AtlasLens benchmark summary",
                "",
                f"- Provider: `{summary.provider_id}`",
                f"- Model revision: `{summary.model_revision}`",
                f"- Manifest fingerprint: `{summary.manifest_fingerprint}`",
                f"- Image count: {summary.image_count}",
                f"- Candidate return: {metric('candidate_return')}",
                f"- Abstention: {metric('abstention')}",
                f"- Provider failure: {metric('provider_failure')}",
                f"- Country Top-1: {metric('country_top1')}",
                f"- Country Top-5: {metric('country_top5')}",
                "",
                "Scores are evaluation measurements, not guarantees for individual images.",
            ]
        )
