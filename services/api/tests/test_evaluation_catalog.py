from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from atlaslens_api.dataset_qa import DatasetQAError, DatasetQAReportCatalog
from atlaslens_api.evaluation.catalog import (
    EvaluationCatalogError,
    EvaluationReportCatalog,
)
from atlaslens_api.evaluation.models import (
    CandidatePrediction,
    EvaluationRecord,
    ManifestValidationReport,
    ProviderPrediction,
    ValidatedEvaluationAsset,
    ValidatedManifest,
)
from atlaslens_api.evaluation.runner import BenchmarkRunner


class Provider:
    model_revision = "revision-1"

    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id

    def predict(self, image_path: Path) -> ProviderPrediction:
        assert image_path.is_file()
        return ProviderPrediction(
            candidates=(
                CandidatePrediction(
                    rank=1,
                    latitude=10,
                    longitude=20,
                    raw_score=0.5,
                    score_type="uncalibrated_relative",
                    country_code="TR",
                    region="Region",
                    city_or_area="City",
                    uncertainty_radius_km=100,
                ),
            ),
            latency_ms=1,
            device="cpu",
        )


def manifest(tmp_path: Path) -> ValidatedManifest:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (24, 16), (20, 80, 140)).save(image)
    record = EvaluationRecord(
        image_asset_key="evaluation:one",
        local_reference=image.name,
        true_latitude=10,
        true_longitude=20,
        country_code="TR",
        region="Region",
        city_or_area="City",
        continent="Asia",
        source="test-only",
        source_record_id="record-one",
        license="CC0-1.0",
        attribution="Generated fixture",
        split="test",
        scene_category="urban",
        geographic_cell="cell-one",
        capture_family_id="family-one",
        content_sha256="a" * 64,
    )
    return ValidatedManifest(
        assets=(ValidatedEvaluationAsset(record=record, path=image),),
        report=ManifestValidationReport(
            fingerprint="b" * 64,
            image_count=1,
            split_counts={"test": 1},
            continent_counts={"Asia": 1},
            country_counts={"TR": 1},
            scene_counts={"urban": 1},
        ),
    )


def test_evaluation_catalog_validates_maps_recall_and_excludes_mock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    evaluated = manifest(tmp_path)
    BenchmarkRunner().run(
        evaluated,
        Provider("real-global-v1"),
        output_directory=root / "real-report",
    )
    BenchmarkRunner().run(
        evaluated,
        Provider("mock-global-v1"),
        output_directory=root / "mock-report",
    )
    catalog = EvaluationReportCatalog(root)

    report = catalog.get_report("real-report")
    assert tuple(report.recall_top1) == ("1", "25", "100", "200", "500", "750", "2500")
    assert report.recall_top1["100"].value is None
    assert report.recall_top1["100"].denominator == 0
    assert report.geographic_distribution == {"Asia": 1}
    assert report.scene_distribution == {"urban": 1}
    assert report.calibration_state == "uncalibrated"
    assert "recall_100km_unavailable" in report.limitations
    assert [item.report_id for item in catalog.list_reports().reports] == ["real-report"]
    with pytest.raises(EvaluationCatalogError, match="simulated"):
        catalog.get("mock-report")
    with pytest.raises(EvaluationCatalogError, match="id_invalid"):
        catalog.get("../real-report")


def test_evaluation_catalog_rejects_summary_tamper_and_path_leaks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports"
    report_dir = root / "tampered"
    BenchmarkRunner().run(
        manifest(tmp_path),
        Provider("real-global-v1"),
        output_directory=report_dir,
    )
    path = report_dir / "benchmark.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary"]["top1_error"]["median_km"] = 999.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationCatalogError, match="summary_mismatch"):
        EvaluationReportCatalog(root).get("tampered")


def test_report_catalogs_reject_directory_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")
    with pytest.raises(EvaluationCatalogError, match="root_unsafe"):
        EvaluationReportCatalog(link)
    with pytest.raises(DatasetQAError, match="root_unsafe"):
        DatasetQAReportCatalog(link)
