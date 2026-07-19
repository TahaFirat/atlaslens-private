from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from atlaslens_api.dataset_qa.models import DatasetQAError, DatasetQAPolicy, DatasetQARun
from atlaslens_api.schemas import DatasetQAReport


def safe_output_directory(output: Path, input_roots: tuple[Path, ...]) -> Path:
    expanded = output.expanduser()
    if expanded.is_symlink():
        raise DatasetQAError("qa_output_unsafe")
    resolved = expanded.resolve(strict=False)
    for raw_root in input_roots:
        root = raw_root.expanduser().resolve(strict=True)
        if resolved == root or root in resolved.parents:
            raise DatasetQAError("qa_output_inside_source")
    resolved.mkdir(parents=True, exist_ok=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise DatasetQAError("qa_output_unsafe")
    return resolved


class DatasetQAReportWriter:
    def __init__(self, policy: DatasetQAPolicy | None = None) -> None:
        self._policy = policy or DatasetQAPolicy()

    def write(self, run: DatasetQARun, output: Path) -> tuple[Path, ...]:
        report = run.report
        paths = tuple(output / name for name in report.outputs)
        for path in paths:
            if path.is_symlink():
                raise DatasetQAError("qa_output_unsafe")
        self._atomic_text(
            output / "report.json",
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        )
        self._issues_csv(output / "issues.csv", report)
        self._atomic_text(output / "summary.md", self._markdown(report))
        self._atomic_text(output / "report.html", self._html(report))
        if "contact-sheet.jpg" in report.outputs:
            self._contact_sheet(output / "contact-sheet.jpg", run.source_images)
        return paths

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
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
    def _issues_csv(path: Path, report: DatasetQAReport) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "code",
                        "severity",
                        "asset_key",
                        "field",
                        "message_key",
                        "safe_metrics",
                    ),
                )
                writer.writeheader()
                for issue in report.issues:
                    row = issue.model_dump(mode="json")
                    row["safe_metrics"] = json.dumps(
                        issue.safe_metrics, sort_keys=True, separators=(",", ":")
                    )
                    writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _markdown(report: DatasetQAReport) -> str:
        summary = report.summary
        lines = [
            "# AtlasLens dataset QA",
            "",
            f"- Report: `{summary.report_id}`",
            f"- Dataset fingerprint: `{summary.dataset_fingerprint}`",
            f"- Dataset type: `{summary.dataset_type}`",
            f"- Scanned images: {summary.scanned_images}",
            f"- Scanned masks: {summary.scanned_masks}",
            f"- Errors: {summary.error_count}",
            f"- Warnings: {summary.warning_count}",
            "",
            "| Check | Status |",
            "|---|---|",
        ]
        lines.extend(f"| {name} | {status} |" for name, status in sorted(report.checks.items()))
        lines.extend(("", "## Issues", "", "| Asset | Severity | Code |", "|---|---|---|"))
        lines.extend(
            f"| `{issue.asset_key}` | {issue.severity} | `{issue.code}` |"
            for issue in report.issues
        )
        lines.extend(("", "## Limitations", ""))
        lines.extend(f"- {item}" for item in report.limitations)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _html(report: DatasetQAReport) -> str:
        summary = report.summary
        rows = "".join(
            "<tr><td>"
            + html.escape(issue.asset_key)
            + "</td><td>"
            + html.escape(issue.severity)
            + "</td><td>"
            + html.escape(issue.code)
            + "</td></tr>"
            for issue in report.issues
        )
        checks = "".join(
            f"<li><code>{html.escape(name)}</code>: {html.escape(status)}</li>"
            for name, status in sorted(report.checks.items())
        )
        limitations = "".join(f"<li>{html.escape(item)}</li>" for item in report.limitations)
        return (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>AtlasLens dataset QA</title>"
            "<style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:2rem;}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:.4rem}"
            "code{overflow-wrap:anywhere}</style><body>"
            f"<h1>AtlasLens dataset QA</h1><p>Report <code>{html.escape(summary.report_id)}</code>"
            f" — {summary.scanned_images} images, {summary.error_count} errors, "
            f"{summary.warning_count} warnings.</p><h2>Checks</h2><ul>{checks}</ul>"
            f"<h2>Issues</h2><table><thead><tr><th>Asset</th><th>Severity</th>"
            f"<th>Code</th></tr></thead><tbody>{rows}</tbody></table>"
            f"<h2>Limitations</h2><ul>{limitations}</ul></body></html>"
        )

    def _contact_sheet(self, path: Path, sources: tuple[Path, ...]) -> None:
        thumb = self._policy.contact_sheet_thumbnail
        columns = min(8, max(1, math_ceil_sqrt(len(sources))))
        rows = max(1, (len(sources) + columns - 1) // columns)
        canvas = Image.new("RGB", (columns * thumb, rows * thumb), "white")
        for index, source_path in enumerate(sources):
            with Image.open(source_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                x = (index % columns) * thumb + (thumb - image.width) // 2
                y = (index // columns) * thumb + (thumb - image.height) // 2
                canvas.paste(image, (x, y))
                image.close()
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            canvas.save(temporary, "JPEG", quality=82, optimize=True, exif=b"")
            with temporary.open("rb+") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            canvas.close()
            temporary.unlink(missing_ok=True)


def math_ceil_sqrt(value: int) -> int:
    if value <= 1:
        return 1
    root = int(value**0.5)
    return root if root * root == value else root + 1
