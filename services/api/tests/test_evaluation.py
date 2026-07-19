from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from atlaslens_api.evaluation import (
    BenchmarkRunner,
    CalibrationArtifact,
    CalibrationCompatibilityError,
    CandidatePrediction,
    EvaluationManifestLoader,
    InferenceProviderEvaluationAdapter,
    LogisticArtifactCalibrator,
    ProviderPrediction,
    UncalibratedCalibrator,
)
from atlaslens_api.evaluation.cli import (
    GlobalProviderEvaluationAdapter,
    run_benchmark_command,
)
from atlaslens_api.evaluation.cli import (
    main as benchmark_main,
)
from atlaslens_api.evaluation.manifest import EvaluationManifestError
from atlaslens_api.evaluation.models import EvaluationRecord
from atlaslens_api.gazetteer.models import ResolvedPlace
from atlaslens_api.inference.models import (
    InferenceCandidate,
    InferenceProvenance,
    InferenceProviderStatus,
    InferenceRequest,
    InferenceResult,
)
from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    GlobalProviderStatus,
    ProviderDescriptor,
    ProviderOutcome,
)


def create_images(root: Path, count: int = 4) -> list[Path]:
    root.mkdir()
    paths = []
    for index in range(count):
        path = root / f"asset-{index}.jpg"
        digest = hashlib.sha256(f"evaluation-pattern-{index}".encode()).digest()
        pixels = [
            255 if digest[bit // 8] & (1 << (bit % 8)) else 0
            for bit in range(64)
        ]
        image = Image.new("L", (8, 8))
        image.putdata(pixels)
        image.resize((24, 16), Image.Resampling.NEAREST).convert("RGB").save(
            path, "JPEG", quality=95
        )
        paths.append(path)
    return paths


def average_hash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = list(
            image.convert("L").resize((8, 8), Image.Resampling.LANCZOS).getdata()
        )
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return f"{value:016x}"


def row(path: Path, index: int, *, split: str = "test") -> dict[str, str]:
    return {
        "image_asset_key": f"asset:{index}",
        "local_reference": path.name,
        "true_latitude": str(index * 10),
        "true_longitude": str(index * 10),
        "country_code": ["AA", "BB", "CC", "DD"][index],
        "region": f"region-{index}",
        "city_or_area": f"city-{index}",
        "continent": ["Africa", "Asia", "Europe", "Oceania"][index],
        "source": "licensed-test-source",
        "source_record_id": f"record-{index}",
        "license": "CC0-1.0",
        "attribution": "Generated test fixture",
        "split": split,
        "scene_category": ["urban", "rural", "road", "night"][index],
        "geographic_cell": f"cell-{index}",
        "capture_family_id": f"family-{index}",
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "perceptual_hash": average_hash(path),
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(EvaluationRecord.model_fields))
        writer.writeheader()
        writer.writerows(rows)


def valid_manifest(tmp_path: Path):
    root = tmp_path / "assets"
    paths = create_images(root)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, [row(path, index) for index, path in enumerate(paths)])
    return loader().load(manifest_path, root), root, paths


def loader(**kwargs: object) -> EvaluationManifestLoader:
    return EvaluationManifestLoader(
        allowed_licenses=frozenset({"CC0-1.0"}),
        **kwargs,  # type: ignore[arg-type]
    )


def test_manifest_validation_and_distribution_report(tmp_path: Path) -> None:
    manifest, root, paths = valid_manifest(tmp_path)
    assert manifest.report.image_count == 4
    assert manifest.report.split_counts == {"test": 4}
    assert manifest.report.scene_counts == {"urban": 1, "rural": 1, "road": 1, "night": 1}
    assert len(manifest.report.fingerprint) == 64
    assert str(root) not in repr(manifest.assets[0])

    rows = [row(paths[0], 0, split="calibration"), row(paths[1], 1, split="test")]
    rows[1]["geographic_cell"] = rows[0]["geographic_cell"]
    path = tmp_path / "cell-warning.csv"
    write_manifest(path, rows)
    report = loader().load(path, root).report
    assert report.warnings == ("geographic_cell_cross_split:cell-0",)


@pytest.mark.parametrize("field", ["license", "source", "attribution"])
def test_manifest_rejects_missing_required_provenance(tmp_path: Path, field: str) -> None:
    root = tmp_path / "assets"
    paths = create_images(root, 1)
    data = row(paths[0], 0)
    data[field] = ""
    manifest = tmp_path / "invalid.csv"
    write_manifest(manifest, [data])
    with pytest.raises(EvaluationManifestError, match="invalid evaluation row"):
        loader().load(manifest, root)


def test_manifest_rejects_coordinates_paths_hashes_and_oversized_assets(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    paths = create_images(root, 1)
    data = row(paths[0], 0)
    manifest = tmp_path / "invalid.csv"

    data["true_latitude"] = "91"
    write_manifest(manifest, [data])
    with pytest.raises(EvaluationManifestError):
        loader().load(manifest, root)

    data = row(paths[0], 0)
    data["local_reference"] = "../outside.jpg"
    write_manifest(manifest, [data])
    with pytest.raises(EvaluationManifestError, match="unsafe evaluation path"):
        loader().load(manifest, root)

    data = row(paths[0], 0)
    data["content_sha256"] = "0" * 64
    write_manifest(manifest, [data])
    with pytest.raises(EvaluationManifestError, match="hash mismatch"):
        loader().load(manifest, root)

    data = row(paths[0], 0)
    write_manifest(manifest, [data])
    with pytest.raises(EvaluationManifestError, match="oversized"):
        loader(max_asset_bytes=1).load(manifest, root)


@pytest.mark.parametrize("leakage", ["hash", "capture_family", "perceptual_hash"])
def test_manifest_rejects_duplicate_and_cross_split_leakage(
    tmp_path: Path, leakage: str
) -> None:
    root = tmp_path / "assets"
    paths = create_images(root, 2)
    first = row(paths[0], 0, split="calibration")
    second = row(paths[1], 1, split="test")
    if leakage == "hash":
        second["local_reference"] = first["local_reference"]
        second["content_sha256"] = first["content_sha256"]
    elif leakage == "capture_family":
        second["capture_family_id"] = first["capture_family_id"]
    else:
        with Image.open(paths[0]) as image:
            image.save(paths[1], "JPEG", quality=80)
        second["content_sha256"] = hashlib.sha256(paths[1].read_bytes()).hexdigest()
        second["perceptual_hash"] = average_hash(paths[1])
    manifest = tmp_path / "leakage.csv"
    write_manifest(manifest, [first, second])
    with pytest.raises(EvaluationManifestError):
        loader().load(manifest, root)


def test_manifest_rejects_conflicting_source_record(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    paths = create_images(root, 2)
    rows = [row(paths[0], 0), row(paths[1], 1)]
    rows[1]["source_record_id"] = rows[0]["source_record_id"]
    manifest = tmp_path / "source-conflict.csv"
    write_manifest(manifest, rows)
    with pytest.raises(EvaluationManifestError, match="conflicting source"):
        loader().load(manifest, root)


class InjectedProvider:
    provider_id = "injected-evaluation-provider"
    model_revision = "test-revision"

    def predict(self, image_path: Path) -> ProviderPrediction:
        index = int(image_path.stem.split("-")[-1])
        if index == 2:
            return ProviderPrediction(abstained=True, latency_ms=30, device="cpu")
        if index == 3:
            raise RuntimeError("test provider failure")
        if index == 0:
            candidates = (
                CandidatePrediction(
                    rank=1,
                    latitude=0,
                    longitude=0,
                    raw_score=0.9,
                    score_type="relative",
                    country_code="AA",
                    region="region-0",
                    city_or_area="city-0",
                    uncertainty_radius_km=1,
                ),
            )
        else:
            candidates = (
                CandidatePrediction(
                    rank=1,
                    latitude=-40,
                    longitude=-40,
                    raw_score=0.6,
                    score_type="relative",
                    country_code="ZZ",
                    uncertainty_radius_km=25,
                ),
                CandidatePrediction(
                    rank=2,
                    latitude=10,
                    longitude=10,
                    raw_score=0.5,
                    score_type="relative",
                    country_code="BB",
                    region="region-1",
                    city_or_area="city-1",
                ),
            )
        return ProviderPrediction(candidates=candidates, latency_ms=10 + index, device="cuda")


def test_benchmark_metrics_denominators_reports_and_breakdowns(tmp_path: Path) -> None:
    manifest, root, _ = valid_manifest(tmp_path)
    output = tmp_path / "reports"
    run = BenchmarkRunner().run(manifest, InjectedProvider(), output_directory=output)
    summary = run.summary
    assert summary.image_count == 4
    assert summary.successful_inference.value == 0.75
    assert summary.candidate_return.value == 0.5
    assert summary.abstention.value == 0.25
    assert summary.provider_failure.value == 0.25
    assert summary.country_top1.value == 0.25
    assert summary.country_top5.value == 0.5
    assert summary.recall_top1["1"].value == 0.25
    assert summary.recall_top5_oracle["1"].value == 0.5
    assert summary.uncertainty_coverage.value == 0.5
    assert set(summary.breakdowns) == {"scene_category", "continent", "country"}
    assert summary.cpu_gpu_split == {"cuda": 2, "cpu": 1, "other": 1}
    assert run.report_paths is not None
    for path in run.report_paths:
        assert path.exists()
        assert str(root) not in path.read_text(encoding="utf-8")
    payload = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert payload["summary"]["provider_id"] == InjectedProvider.provider_id
    assert all("local_reference" not in item for item in payload["results"])


def artifact(state: str = "calibrated", revision: str = "revision-1") -> CalibrationArtifact:
    return CalibrationArtifact(
        calibration_state=state,
        provider_id="provider",
        model_revision=revision,
        event="within_200_km",
        method="logistic",
        feature_names=("score", "margin"),
        coefficients=(1.0, -0.5),
        intercept=0.1,
        fit_fingerprint="1" * 64,
        validation_fingerprint="2" * 64,
        test_fingerprint="3" * 64,
        fit_count=100,
        validation_count=50,
        test_count=50,
        created_at=datetime.now(UTC),
        metrics={
            "fit_event_rate": 0.5,
            "validation_event_rate": 0.5,
            "test_event_rate": 0.5,
            "validation_brier": 0.2,
            "validation_ece": 0.1,
            "test_brier": 0.2,
            "test_ece": 0.1,
        },
        applicability_limits=("model revision specific",),
    )


def calibrator(
    value: CalibrationArtifact, *, model_revision: str = "revision-1"
) -> LogisticArtifactCalibrator:
    return LogisticArtifactCalibrator(
        value,
        provider_id="provider",
        model_revision=model_revision,
        feature_names=("score", "margin"),
        fit_fingerprint="1" * 64,
        validation_fingerprint="2" * 64,
        test_fingerprint="3" * 64,
        fit_count=100,
        validation_count=50,
        test_count=50,
    )


def test_calibration_states_and_strict_compatibility() -> None:
    assert UncalibratedCalibrator().probability({"score": 1}) is None
    preliminary = calibrator(artifact("preliminary"))
    assert preliminary.probability({"score": 0.5, "margin": 0.2}) is None

    calibrated_artifact = artifact()
    probability = calibrator(calibrated_artifact).probability(
        {"score": 0.5, "margin": 0.2}
    )
    assert probability is not None and 0 < probability < 1
    with pytest.raises(CalibrationCompatibilityError):
        calibrator(calibrated_artifact, model_revision="other")
    with pytest.raises(CalibrationCompatibilityError):
        calibrator(calibrated_artifact).probability({"score": 0.5})


def test_calibration_rejects_split_fingerprint_reuse() -> None:
    values = artifact().model_dump()
    values["test_fingerprint"] = values["fit_fingerprint"]
    with pytest.raises(ValueError, match="distinct fingerprints"):
        CalibrationArtifact.model_validate(values)


def test_calibrated_artifact_requires_promotion_evidence() -> None:
    values = artifact().model_dump()
    values["test_count"] = 1
    with pytest.raises(ValueError, match="split is too small"):
        CalibrationArtifact.model_validate(values)
    values = artifact().model_dump()
    values["metrics"].pop("test_ece")
    with pytest.raises(ValueError, match="promotion metrics"):
        CalibrationArtifact.model_validate(values)


class TypedGlobalProvider:
    descriptor = ProviderDescriptor(
        id="typed-global",
        kind="global_geolocation",
        version="1",
        execution_boundary="local",
        criticality="optional",
        available=True,
        model_name="typed-model",
    )

    def status(self) -> GlobalProviderStatus:
        return GlobalProviderStatus(
            status="ready",
            installed=True,
            verified=True,
            model_name="typed-model",
            model_revision="revision-1",
            device="cpu",
            calibration_state="uncalibrated",
        )

    async def predict(self, handle, context):
        assert handle.key == "evaluation.asset"
        assert context.mode.value == "local_only"
        hypotheses = [
            GlobalPredictionHypothesis(
                rank=index,
                original_rank=index,
                latitude=10 + index,
                longitude=20 + index,
                raw_score=1 / index,
                score_type="uncalibrated_gallery_softmax",
                normalization_method="softmax_over_fixed_gallery",
                calibration_state="uncalibrated",
                limitations=["not calibrated"],
            )
            for index in range(1, 4)
        ]
        return ProviderOutcome.succeeded(
            GlobalPredictionResult(
                provider_id="typed-global",
                model_name="typed-model",
                model_revision="revision-1",
                implementation_revision="implementation-1",
                device="cpu",
                dtype="float32",
                inference_ms=5,
                hypotheses=hypotheses,
            )
        )


class FixedGazetteer:
    def resolve(self, latitude: float, longitude: float) -> ResolvedPlace:
        assert latitude > 0 and longitude > 0
        return ResolvedPlace(
            label="Test place",
            country_code="AA",
            country="Country",
            region="Region",
            city="City",
            distance_km=1,
            source="test gazetteer",
            dataset_version="1",
            license="CC0-1.0",
        )


def test_real_global_provider_adapter_preserves_typed_scores_and_labels(tmp_path: Path) -> None:
    image = tmp_path / f"{uuid4()}.jpg"
    image.write_bytes(b"test input is not decoded by typed provider")
    adapter = GlobalProviderEvaluationAdapter(TypedGlobalProvider(), gazetteer=FixedGazetteer())
    result = adapter.predict(image)
    assert adapter.provider_id == "typed-global"
    assert adapter.model_revision == "revision-1"
    assert len(result.candidates) == 3
    assert result.candidates[0].score_type == "uncalibrated_gallery_softmax"
    assert result.candidates[0].country_code == "AA"
    assert all(
        candidate.uncertainty_radius_km is not None
        and candidate.uncertainty_radius_km >= 750
        for candidate in result.candidates
    )
    assert result.device == "cpu"
    adapter.close()


class TypedInferenceProvider:
    provider_id = "atlaslens-custom-v1"
    mode = "shadow"
    classification = "real"

    def status(self) -> InferenceProviderStatus:
        return InferenceProviderStatus(
            provider_id=self.provider_id,
            provider_type="global_geolocation",
            provider_revision="custom-adapter-v1",
            mode="shadow",
            available=True,
            status="ready",
            classification="real",
            model_name="AtlasLens custom",
            model_revision="1.0.0",
            runtime_revision="runtime-v1",
            device="cpu",
            calibration_state="uncalibrated",
        )

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        assert request.image_handle is not None
        return InferenceResult(
            provider_id=self.provider_id,
            provider_revision="custom-adapter-v1",
            model_name="AtlasLens custom",
            model_revision="1.0.0",
            runtime_revision="runtime-v1",
            classification="real",
            status="succeeded",
            device="cpu",
            dtype="float32",
            runtime_ms=3,
            score_semantics="uncalibrated_bounded_score",
            normalization_method="custom-rgb-v1",
            calibration_state="uncalibrated",
            candidates=(
                InferenceCandidate(
                    rank=1,
                    original_rank=1,
                    latitude=38.72,
                    longitude=35.48,
                    raw_score=0.7,
                    limitations=("unverified_custom_prediction",),
                ),
            ),
            provenance=InferenceProvenance(
                provider_id=self.provider_id,
                provider_revision="custom-adapter-v1",
                model_revision="1.0.0",
                runtime_revision="runtime-v1",
                source_kind="verified_custom_artifact",
                artifact_digest="d" * 64,
            ),
        )


def test_custom_inference_adapter_is_benchmarkable_without_affecting_ranking(
    tmp_path: Path,
) -> None:
    image = tmp_path / f"{uuid4()}.jpg"
    Image.new("RGB", (32, 24), (20, 40, 60)).save(image, "JPEG")
    adapter = InferenceProviderEvaluationAdapter(TypedInferenceProvider())
    result = adapter.predict(image)
    assert adapter.provider_id == "atlaslens-custom-v1"
    assert adapter.model_revision == "1.0.0"
    assert result.candidates[0].score_type == "uncalibrated_bounded_score"
    assert result.candidates[0].uncertainty_radius_km == 750
    adapter.close()


def test_benchmark_cli_validate_run_report_and_info(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    paths = create_images(root)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, [row(path, index) for index, path in enumerate(paths)])
    common = [
        "--manifest",
        str(manifest_path),
        "--asset-root",
        str(root),
        "--allow-license",
        "CC0-1.0",
    ]
    validated = run_benchmark_command(["validate", *common])
    assert validated["status"] == "valid"

    output = tmp_path / "reports"
    completed = run_benchmark_command(
        ["run", *common, "--output", str(output)], provider=InjectedProvider()
    )
    assert completed["status"] == "completed"
    assert completed["reports"] == ["benchmark.json", "per-image.csv", "summary.md"]
    assert run_benchmark_command(["report", "--output", str(output)])["status"] == "completed"
    info = run_benchmark_command(["info", "--output", str(output)])
    assert info["status"] == "available"
    assert info["image_count"] == 4

    custom_output = tmp_path / "custom-reports"
    custom_completed = run_benchmark_command(
        [
            "run",
            *common,
            "--provider",
            "atlaslens-custom-geolocation",
            "--output",
            str(custom_output),
        ],
        provider=InjectedProvider(),
    )
    assert custom_completed["status"] == "completed"


def test_benchmark_cli_error_is_safe(capsys) -> None:
    assert benchmark_main(["report", "--output", "missing-output"]) == 2
    captured = capsys.readouterr()
    assert "benchmark_operation_failed" in captured.err
    assert "missing-output" not in captured.err
