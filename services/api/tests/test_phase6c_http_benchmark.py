from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from atlaslens_api.evaluation.isolation import (
    IsolatedPredictionRequest,
    IsolatedPredictionResponse,
)
from atlaslens_api.evaluation.models import (
    CandidatePrediction,
    EvaluationRecord,
    ProviderPrediction,
)
from atlaslens_api.evaluation.phase6c_benchmark import main
from atlaslens_api.evaluation.phase6c_http import Phase6CHTTPEvaluationProvider


def _average_hash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = list(
            image.convert("L").resize((8, 8), Image.Resampling.LANCZOS).getdata()
        )
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return f"{value:016x}"


def _manifest(tmp_path: Path, *, split: str = "validation") -> tuple[Path, Path, Path]:
    root = tmp_path / "private-assets"
    root.mkdir()
    image_path = root / "private-city-and-coordinate.jpg"
    Image.new("RGB", (24, 16), (20, 80, 140)).save(image_path, "JPEG")
    row = {
        "image_asset_key": "opaque-asset",
        "local_reference": image_path.name,
        "true_latitude": "38.72",
        "true_longitude": "35.48",
        "country_code": "TR",
        "region": "Private Region",
        "city_or_area": "Private City",
        "continent": "Asia",
        "source": "kartaview",
        "source_record_id": "opaque-source",
        "license": "CC BY-SA 4.0",
        "attribution": "Fixture attribution",
        "split": split,
        "scene_category": "urban",
        "geographic_cell": "opaque-cell",
        "capture_family_id": "opaque-family",
        "content_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "perceptual_hash": _average_hash(image_path),
    }
    manifest = tmp_path / "private-manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(EvaluationRecord.model_fields))
        writer.writeheader()
        writer.writerow(row)
    return manifest, root, image_path


class _FixedProvider:
    provider_id = "atlaslens-phase6c-http"
    model_revision = "phase6c-v1"

    def predict(self, image_path: Path) -> ProviderPrediction:
        assert image_path.is_file()
        return ProviderPrediction(
            candidates=(
                CandidatePrediction(
                    rank=1,
                    latitude=38.70,
                    longitude=35.50,
                    raw_score=0.7,
                    score_type="phase6c-v1:uncalibrated_relative_rank_not_probability",
                    country_code="TR",
                    region="Private Region",
                    city_or_area="Private City",
                    uncertainty_radius_km=20,
                ),
            ),
            latency_ms=10,
            device="other",
        )


class _CapturingBoundary:
    request: IsolatedPredictionRequest | None = None

    def predict(self, request: IsolatedPredictionRequest) -> IsolatedPredictionResponse:
        self.request = request
        return IsolatedPredictionResponse(
            request_id=request.request_id,
            provider_id="atlaslens-phase6c-http",
            model_revision="phase6c-v1",
            prediction=ProviderPrediction(
                abstained=True,
                latency_ms=5,
                device="other",
            ),
        )


def _arguments(manifest: Path, root: Path, output: Path) -> list[str]:
    return [
        "--manifest",
        str(manifest),
        "--asset-root",
        str(root),
        "--allowed-license",
        "CC BY-SA 4.0",
        "--output-directory",
        str(output),
        "--api-base-url",
        "http://127.0.0.1:8000",
        "--timeout-seconds",
        "120",
        "--request-timeout-seconds",
        "15",
        "--poll-interval-seconds",
        "0.25",
    ]


def test_provider_adapter_projects_only_image_path_to_worker(tmp_path: Path) -> None:
    image = tmp_path / "private-input.jpg"
    image.write_bytes(b"not-read-by-boundary")
    boundary = _CapturingBoundary()

    prediction = Phase6CHTTPEvaluationProvider(boundary).predict(image)

    assert prediction.abstained
    assert boundary.request is not None
    payload = boundary.request.model_dump(mode="json")
    assert set(payload) == {"protocol_version", "request_id", "image_path"}
    assert payload["image_path"] == str(image)
    encoded = json.dumps(payload)
    for forbidden in ("latitude", "longitude", "country", "city", "ocr"):
        assert forbidden not in encoded.casefold()


def test_cli_runs_development_validation_benchmark_with_aggregate_console_only(
    tmp_path: Path,
) -> None:
    manifest, root, image = _manifest(tmp_path)
    output = tmp_path / "reports"
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = main(
        _arguments(manifest, root, output),
        provider=_FixedProvider(),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    assert stderr.getvalue() == ""
    console = stdout.getvalue()
    payload = json.loads(console)
    assert payload == {
        "abstention_count": 0,
        "event": "phase6c_benchmark_completed",
        "image_count": 1,
        "provider_failure_count": 0,
        "report_count": 3,
        "successful_inference_count": 1,
    }
    for forbidden in (
        str(manifest),
        str(root),
        str(image),
        "Private City",
        "Private Region",
        "38.72",
        "35.48",
        "ocr",
    ):
        assert forbidden not in console
    assert {path.name for path in output.iterdir()} == {
        "benchmark.json",
        "per-image.csv",
        "summary.md",
    }
    report = (output / "benchmark.json").read_text(encoding="utf-8")
    assert str(manifest) not in report
    assert str(root) not in report
    assert str(image) not in report
    assert "opaque-asset" not in report
    assert "opaque-source" not in report


def test_cli_rejects_test_split_and_non_loopback_without_sensitive_output(
    tmp_path: Path,
) -> None:
    manifest, root, image = _manifest(tmp_path, split="test")
    stdout = io.StringIO()
    stderr = io.StringIO()
    arguments = _arguments(manifest, root, tmp_path / "reports")

    assert main(arguments, stdout=stdout, stderr=stderr) == 2
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "phase6c_benchmark_failed\n"
    assert str(image) not in stderr.getvalue()
    assert "Private City" not in stderr.getvalue()

    arguments[arguments.index("http://127.0.0.1:8000")] = "https://example.test"
    stderr = io.StringIO()
    assert main(arguments, stdout=io.StringIO(), stderr=stderr) == 2
    assert stderr.getvalue() == "phase6c_benchmark_failed\n"
    assert "example.test" not in stderr.getvalue()
