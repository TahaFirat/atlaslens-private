from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from atlaslens_api.evaluation import BenchmarkRunner, EvaluationManifestLoader
from atlaslens_api.evaluation.cli import main as benchmark_main
from atlaslens_api.evaluation.cli import run_benchmark_command
from atlaslens_api.evaluation.comparison import BenchmarkComparisonBuilder
from atlaslens_api.evaluation.models import (
    CandidatePrediction,
    EvaluationRecord,
    ProviderPrediction,
    ValidatedManifest,
)


class FixedProvider:
    model_revision = "test-revision"

    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id

    def predict(self, image_path: Path) -> ProviderPrediction:
        assert image_path.is_file()
        return ProviderPrediction(
            candidates=(
                CandidatePrediction(
                    rank=1,
                    latitude=41.0082,
                    longitude=28.9784,
                    raw_score=0.5,
                    score_type="uncalibrated_test_score",
                    country_code="TR",
                    region="Istanbul",
                    city_or_area="Istanbul",
                    uncertainty_radius_km=10,
                ),
            ),
            latency_ms=1,
            device="cpu",
        )


def benchmark_fixture(tmp_path: Path) -> tuple[Path, Path, ValidatedManifest, Path]:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    image_path = asset_root / "istanbul.jpg"
    Image.new("RGB", (24, 16), (50, 100, 150)).save(image_path, "JPEG")
    manifest_path = tmp_path / "manifest.csv"
    values = {
        "image_asset_key": "fixture:istanbul",
        "local_reference": image_path.name,
        "true_latitude": "41.0082",
        "true_longitude": "28.9784",
        "country_code": "TR",
        "region": "Istanbul",
        "city_or_area": "Istanbul",
        "continent": "Europe",
        "source": "licensed-test-source",
        "source_record_id": "istanbul-1",
        "license": "CC0-1.0",
        "attribution": "Generated test fixture",
        "split": "test",
        "scene_category": "urban",
        "geographic_cell": "fixture-cell",
        "capture_family_id": "fixture-family",
        "content_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "perceptual_hash": "f" * 16,
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(EvaluationRecord.model_fields))
        writer.writeheader()
        writer.writerow(values)
    manifest = EvaluationManifestLoader(allowed_licenses=frozenset({"CC0-1.0"})).load(
        manifest_path, asset_root
    )
    baseline = tmp_path / "geoclip-report"
    BenchmarkRunner().run(manifest, FixedProvider("geoclip-global-v1"), output_directory=baseline)
    return manifest_path, asset_root, manifest, baseline


def common_compare_arguments(
    manifest_path: Path, asset_root: Path, baseline: Path, output: Path
) -> list[str]:
    return [
        "compare",
        "--baseline",
        "geoclip",
        "--candidate",
        "phase5b-v1",
        "--manifest",
        str(manifest_path),
        "--asset-root",
        str(asset_root),
        "--allow-license",
        "CC0-1.0",
        "--geoclip-report",
        str(baseline),
        "--output",
        str(output),
    ]


def test_compare_writes_fixed_modes_and_explicit_unavailable_entries(
    tmp_path: Path,
) -> None:
    manifest_path, asset_root, manifest, baseline = benchmark_fixture(tmp_path)
    output = tmp_path / "comparison"
    result = run_benchmark_command(
        common_compare_arguments(manifest_path, asset_root, baseline, output)
    )

    assert result["status"] == "completed"
    comparison = result["comparison"]
    assert isinstance(comparison, dict)
    assert comparison["manifest_fingerprint"] == manifest.report.fingerprint
    assert [item["mode"] for item in comparison["modes"]] == [
        "geoclip_only",
        "ocr",
        "retrieval",
        "ocr_retrieval",
        "full",
    ]
    assert comparison["modes"][0]["status"] == "available"
    for item in comparison["modes"][1:]:
        assert item["status"] == "unavailable"
        assert item["summary"] is None
        assert item["unavailable_reason"] == "report_not_supplied"

    payload = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert payload == comparison
    serialized = json.dumps(payload)
    assert str(asset_root) not in serialized
    assert str(baseline) not in serialized
    markdown = (output / "comparison.md").read_text(encoding="utf-8")
    assert "GeoCLIP-only" in markdown
    assert "+OCR+retrieval | unavailable" in markdown
    assert "no cross-mode deltas are inferred" in markdown


def test_compare_accepts_five_validated_reports_for_the_same_manifest(
    tmp_path: Path,
) -> None:
    manifest_path, asset_root, manifest, baseline = benchmark_fixture(tmp_path)
    reports: dict[str, Path] = {}
    for mode in ("ocr", "retrieval", "ocr-retrieval", "full"):
        directory = tmp_path / f"{mode}-report"
        BenchmarkRunner().run(
            manifest,
            FixedProvider(f"phase5b-{mode}-v1"),
            output_directory=directory,
        )
        reports[mode] = directory

    arguments = common_compare_arguments(
        manifest_path, asset_root, baseline, tmp_path / "comparison"
    )
    arguments.extend(
        [
            "--ocr-report",
            str(reports["ocr"]),
            "--retrieval-report",
            str(reports["retrieval"]),
            "--ocr-retrieval-report",
            str(reports["ocr-retrieval"]),
            "--full-report",
            str(reports["full"]),
        ]
    )
    result = run_benchmark_command(arguments)
    comparison = result["comparison"]
    assert isinstance(comparison, dict)
    assert [item["status"] for item in comparison["modes"]] == ["available"] * 5
    assert {item["summary"]["manifest_fingerprint"] for item in comparison["modes"]} == {
        manifest.report.fingerprint
    }


def test_compare_accepts_explicit_geoclip_custom_provider_pair(tmp_path: Path) -> None:
    manifest_path, asset_root, manifest, baseline = benchmark_fixture(tmp_path)
    custom = tmp_path / "custom-report"
    BenchmarkRunner().run(
        manifest,
        FixedProvider("atlaslens-custom-v1"),
        output_directory=custom,
    )
    output = tmp_path / "provider-comparison"
    result = run_benchmark_command(
        [
            "compare",
            "--providers",
            "geoclip,atlaslens-custom-geolocation",
            "--manifest",
            str(manifest_path),
            "--asset-root",
            str(asset_root),
            "--allow-license",
            "CC0-1.0",
            "--geoclip-report",
            str(baseline),
            "--custom-report",
            str(custom),
            "--output",
            str(output),
        ]
    )
    comparison = result["comparison"]
    assert isinstance(comparison, dict)
    assert comparison["schema_version"] == "atlaslens-provider-comparison-v1"
    assert comparison["baseline"]["provider_id"] == "geoclip-global-v1"
    assert comparison["candidate"]["provider_id"] == "atlaslens-custom-v1"


@pytest.mark.parametrize("field", ["manifest_fingerprint", "image_count"])
def test_compare_rejects_incompatible_precomputed_report(tmp_path: Path, field: str) -> None:
    _, _, manifest, baseline = benchmark_fixture(tmp_path)
    incompatible = tmp_path / "incompatible-report"
    incompatible.mkdir()
    payload = json.loads((baseline / "benchmark.json").read_text(encoding="utf-8"))
    payload["summary"][field] = "f" * 64 if field == "manifest_fingerprint" else 2
    (incompatible / "benchmark.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="summary mismatch|image count|fingerprint"):
        BenchmarkComparisonBuilder().build(
            manifest.report,
            {"geoclip_only": baseline, "ocr": incompatible},
        )


def test_compare_rejects_supplied_missing_report_without_leaking_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path, asset_root, _, baseline = benchmark_fixture(tmp_path)
    missing = tmp_path / "private-missing-report"
    arguments = common_compare_arguments(
        manifest_path, asset_root, baseline, tmp_path / "comparison"
    )
    arguments.extend(["--ocr-report", str(missing)])

    assert benchmark_main(arguments) == 2
    captured = capsys.readouterr()
    assert "benchmark_operation_failed" in captured.err
    assert str(missing) not in captured.err
