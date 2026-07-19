from __future__ import annotations

import json
import re
from pathlib import Path

from atlaslens_api.dataset_qa.models import DatasetQAError
from atlaslens_api.schemas import DatasetQAReport, DatasetQAReportList

_REPORT_ID = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
_ASSET_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SAFE_CODE = re.compile(r"^[a-z0-9_.-]{1,160}$")
_MAX_REPORT_BYTES = 20 * 1024 * 1024


def _safe_value(value: str) -> bool:
    return len(value) <= 200 and not any(
        character in value for character in ("/", "\\", "\r", "\n", "\x00")
    )


class DatasetQAReportCatalog:
    def __init__(self, private_root: Path) -> None:
        expanded = private_root.expanduser()
        if expanded.is_symlink():
            raise DatasetQAError("qa_catalog_root_unsafe")
        try:
            self._root = expanded.resolve(strict=True)
        except OSError as exc:
            raise DatasetQAError("qa_catalog_unavailable") from exc
        if not self._root.is_dir():
            raise DatasetQAError("qa_catalog_unavailable")

    def get(self, report_id: str) -> DatasetQAReport:
        if not _REPORT_ID.fullmatch(report_id):
            raise DatasetQAError("qa_report_id_invalid")
        directory = self._root / report_id
        report_path = directory / "report.json"
        if directory.is_symlink() or report_path.is_symlink():
            raise DatasetQAError("qa_catalog_entry_unsafe")
        try:
            resolved = report_path.resolve(strict=True)
            resolved.relative_to(self._root)
        except (OSError, ValueError) as exc:
            raise DatasetQAError("qa_report_unavailable") from exc
        if not resolved.is_file() or resolved.stat().st_size > _MAX_REPORT_BYTES:
            raise DatasetQAError("qa_report_unavailable")
        try:
            report = DatasetQAReport.model_validate_json(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise DatasetQAError("qa_report_invalid") from exc
        if report.summary.report_id != report_id:
            raise DatasetQAError("qa_report_identity_mismatch")
        if any(Path(item).name != item for item in report.outputs):
            raise DatasetQAError("qa_report_outputs_unsafe")
        if any(
            not _ASSET_KEY.fullmatch(issue.asset_key)
            or any(
                not _SAFE_CODE.fullmatch(key) or not _safe_value(value)
                for key, value in issue.safe_metrics.items()
            )
            for issue in report.issues
        ):
            raise DatasetQAError("qa_report_issue_unsafe")
        if any(not _SAFE_CODE.fullmatch(item) for item in report.limitations):
            raise DatasetQAError("qa_report_limitations_unsafe")
        if any(
            not _SAFE_CODE.fullmatch(group) or any(not _safe_value(key) for key in values)
            for group, values in report.distributions.items()
        ):
            raise DatasetQAError("qa_report_distribution_unsafe")
        return report

    def list(self) -> DatasetQAReportList:
        reports = []
        for directory in sorted(self._root.iterdir(), key=lambda item: item.name):
            if directory.is_symlink():
                raise DatasetQAError("qa_catalog_entry_unsafe")
            if directory.is_dir() and (directory / "report.json").is_file():
                reports.append(self.get(directory.name).summary)
        return DatasetQAReportList(reports=reports)

    def get_report(self, report_id: str) -> DatasetQAReport:
        return self.get(report_id)

    def list_reports(self) -> DatasetQAReportList:
        return self.list()
