from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from atlaslens_api.evaluation.holdout import (
    HoldoutManifestError,
    HoldoutManifestLoader,
)
from atlaslens_api.evaluation.isolation import (
    IsolatedPredictionRequest,
    IsolatedPredictionResponse,
    LeakageSafeEvaluationRunner,
    ProcessPredictionClient,
)
from atlaslens_api.evaluation.leakage import (
    LeakageAuditError,
    LeakageAuditPolicy,
    ReferenceLeakageAuditor,
    load_descriptor_artifact,
)
from atlaslens_api.evaluation.models import CandidatePrediction, ProviderPrediction
from atlaslens_api.retrieval.models import ManifestRecord


def _image(path: Path, variant: int = 0) -> None:
    image = Image.new("RGB", (320, 240), (15 + variant * 20, 30, 45))
    draw = ImageDraw.Draw(image)
    for index in range(12):
        x = 12 + index * 23
        draw.rectangle(
            (x, 10 + (index % 4) * 35, x + 14, 220 - (index % 3) * 17),
            fill=((index * 31 + variant * 47) % 255, 220 - index * 9, 30 + index * 13),
        )
    draw.ellipse((75 + variant * 5, 55, 245, 195), outline=(250, 240, 20), width=8)
    draw.line((0, 230 - variant * 10, 319, 25 + variant * 7), fill=(10, 255, 180), width=5)
    image.save(path, "PNG")


def _record(
    evaluation_id: str,
    image: Path,
    *,
    split: str = "final_holdout",
    source: str | None = None,
    source_image_id: str | None = None,
    capture_family_id: str | None = None,
) -> dict[str, object]:
    usage = {
        "development": "development_only",
        "validation": "validation_only",
        "final_holdout": "holdout_only",
    }[split]
    value: dict[str, object] = {
        "id": evaluation_id,
        "path": str(image),
        "country": "TR",
        "city": "Evaluation City",
        "split": split,
        "usage": usage,
        "allow_reference_index": False,
        "allow_training": False,
        "allow_prompt_ground_truth": False,
    }
    if source is not None:
        value["source"] = source
        value["source_image_id"] = source_image_id
    if capture_family_id is not None:
        value["capture_family_id"] = capture_family_id
    return value


def _manifest(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-geolocation-evaluation-v1",
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path


def _reference(
    path: Path,
    *,
    key: str,
    source: str = "licensed-test-source",
    source_record_id: str | None = None,
    capture_family_id: str | None = None,
) -> ManifestRecord:
    return ManifestRecord(
        image_path=path,
        latitude=0,
        longitude=0,
        country="TR",
        region=None,
        city=None,
        license="test-only",
        source=source,
        content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        asset_key=key,
        source_record_id=source_record_id or key,
        capture_family_id=capture_family_id,
    )


def test_publishable_tree_contains_no_private_evaluation_artifacts() -> None:
    repository = Path(__file__).resolve().parents[3]
    evaluation_root = repository / "evaluation" / "geolocation"

    assert not [item for item in evaluation_root.rglob("*") if item.is_file()]
    assert not list((repository / "evaluation").rglob("*.jpg"))
    assert not list((repository / "evaluation").rglob("*.jpeg"))

    production_root = repository / "services" / "api" / "src" / "atlaslens_api"
    production_files = [
        item
        for item in production_root.rglob("*.py")
        if "evaluation" not in item.relative_to(production_root).parts
    ]
    assert all(
        "turkey_holdout" not in item.read_text(encoding="utf-8")
        for item in production_files
    )

    for relative_path in (
        Path("docs/phase-6b-multimodel.md"),
        Path("docs/phase6c-reference-corpus.md"),
    ):
        text = (repository / relative_path).read_text(encoding="utf-8")
        assert "<PRIVATE_MEDIA_ROOT>" in text
        assert not re.search(r"(?i)[a-z]:[\\/]+users[\\/]+", text)


def test_holdout_loader_enforces_split_source_family_and_duplicate_separation(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first, 0)
    _image(second, 1)
    loader = HoldoutManifestLoader()

    exact_copy = tmp_path / "copy.png"
    exact_copy.write_bytes(first.read_bytes())
    duplicate = _manifest(
        tmp_path / "duplicate.json",
        [
            _record("dev-1", first, split="development"),
            _record("holdout-1", exact_copy, split="final_holdout"),
        ],
    )
    with pytest.raises(HoldoutManifestError, match="duplicate_content"):
        loader.load(duplicate)

    family = _manifest(
        tmp_path / "family.json",
        [
            _record(
                "dev-1",
                first,
                split="development",
                capture_family_id="capture-1",
            ),
            _record(
                "validation-1",
                second,
                split="validation",
                capture_family_id="capture-1",
            ),
        ],
    )
    with pytest.raises(HoldoutManifestError, match="capture_family_crosses_splits"):
        loader.load(family)

    source = _manifest(
        tmp_path / "source.json",
        [
            _record(
                "dev-1",
                first,
                split="development",
                source="mapillary",
                source_image_id="image-1",
            ),
            _record(
                "validation-1",
                second,
                split="validation",
                source="mapillary",
                source_image_id="image-1",
            ),
        ],
    )
    with pytest.raises(HoldoutManifestError, match="duplicate_source_image"):
        loader.load(source)


def test_holdout_loader_and_runner_apply_hard_sample_bounds(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first, 0)
    _image(second, 1)
    manifest = _manifest(
        tmp_path / "bounded.json",
        [_record("one", first), _record("two", second)],
    )
    with pytest.raises(HoldoutManifestError, match="count_invalid"):
        HoldoutManifestLoader(max_records=1).load_prediction_inputs(manifest)
    with pytest.raises(ValueError, match="max_samples"):
        LeakageSafeEvaluationRunner(max_samples=10_001)
    with pytest.raises(ValueError, match="uncertainty radius"):
        IsolatedPredictionResponse(
            request_id="prediction-1",
            provider_id="provider",
            model_revision="revision",
            prediction=ProviderPrediction(
                candidates=(
                    CandidatePrediction(
                        rank=1,
                        latitude=0,
                        longitude=0,
                        score_type="uncalibrated",
                    ),
                ),
                latency_ms=1,
                device="cpu",
            ),
        )


class _RecordingBoundary:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.payloads: list[dict[str, object]] = []

    def predict(self, request: IsolatedPredictionRequest) -> IsolatedPredictionResponse:
        self.events.append("predict")
        payload = request.model_dump(mode="json")
        self.payloads.append(payload)
        return IsolatedPredictionResponse(
            request_id=request.request_id,
            provider_id="test-real-boundary",
            model_revision="test-revision",
            prediction=ProviderPrediction(
                candidates=(
                    CandidatePrediction(
                        rank=1,
                        latitude=41,
                        longitude=29,
                        score_type="uncalibrated_test_output",
                        country_code="TR",
                        city_or_area="Different City",
                        uncertainty_radius_km=750,
                    ),
                ),
                latency_ms=12,
                device="cpu",
            ),
        )


class _OrderedLoader(HoldoutManifestLoader):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def load_prediction_inputs(self, manifest_path: Path):  # type: ignore[no-untyped-def]
        self.events.append("path_projection")
        return super().load_prediction_inputs(manifest_path)

    def load(self, manifest_path: Path):  # type: ignore[no-untyped-def]
        self.events.append("truth_load")
        return super().load(manifest_path)


def test_ground_truth_is_loaded_only_after_isolated_prediction(tmp_path: Path) -> None:
    image = tmp_path / "target.png"
    _image(image)
    manifest = _manifest(tmp_path / "manifest.json", [_record("holdout-1", image)])
    events: list[str] = []
    boundary = _RecordingBoundary(events)

    report = LeakageSafeEvaluationRunner(
        manifest_loader=_OrderedLoader(events)
    ).run(manifest, boundary)

    assert events == ["path_projection", "predict", "truth_load"]
    assert set(boundary.payloads[0]) == {"protocol_version", "request_id", "image_path"}
    encoded = json.dumps(boundary.payloads[0])
    assert "Evaluation City" not in encoded
    assert "country" not in encoded
    assert '"ground_truth":' not in encoded
    assert report.predictions_completed_before_scoring
    assert report.country_top1.value == 1
    assert report.city_top1.value == 0
    assert report.samples[0].ground_truth_city_rank is None
    assert not report.accuracy_claim_allowed


def test_process_prediction_client_sends_only_protocol_and_image_path(tmp_path: Path) -> None:
    capture = tmp_path / "request.json"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            (
                "import json, pathlib, sys",
                "request = json.load(sys.stdin)",
                "pathlib.Path(sys.argv[1]).write_text(json.dumps(request), encoding='utf-8')",
                "response = {'protocol_version': request['protocol_version'], "
                "'request_id': request['request_id'], 'provider_id': 'worker', "
                "'model_revision': '1', 'prediction': {'candidates': [], "
                "'abstained': True, 'failure_code': None, 'latency_ms': 1, "
                "'device': 'cpu'}}",
                "json.dump(response, sys.stdout)",
            )
        ),
        encoding="utf-8",
    )
    client = ProcessPredictionClient(command=(sys.executable, str(worker), str(capture)))
    response = client.predict(
        IsolatedPredictionRequest(request_id="holdout-1", image_path=str(tmp_path / "x.png"))
    )

    request_payload = json.loads(capture.read_text(encoding="utf-8"))
    assert set(request_payload) == {"protocol_version", "request_id", "image_path"}
    assert response.prediction.abstained


def test_leakage_audit_excludes_target_sha_crop_source_and_capture_family(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    exact = tmp_path / "exact.png"
    cropped = tmp_path / "cropped.png"
    rotated = tmp_path / "rotated.png"
    unrelated = tmp_path / "unrelated.png"
    _image(target, 0)
    exact.write_bytes(target.read_bytes())
    with Image.open(target) as source:
        source.crop((40, 30, 280, 210)).save(cropped, "PNG")
        source.rotate(90, expand=True).save(rotated, "PNG")
    _image(unrelated, 4)
    manifest = _manifest(
        tmp_path / "holdout.json",
        [
            _record(
                "holdout-1",
                target,
                source="mapillary",
                source_image_id="target-source",
                capture_family_id="target-capture",
            )
        ],
    )
    holdout = HoldoutManifestLoader().load(manifest).records[0]
    references = (
        _reference(exact, key="exact"),
        _reference(cropped, key="crop"),
        _reference(rotated, key="rotated"),
        _reference(
            unrelated,
            key="same-source",
            source="mapillary",
            source_record_id="target-source",
        ),
        _reference(
            unrelated,
            key="same-family",
            capture_family_id="target-capture",
        ),
    )

    report = ReferenceLeakageAuditor().audit(holdout, references)
    reasons = {item.reference_key: set(item.reasons) for item in report.exclusions}

    assert report.status == "failed"
    assert "target_sha256_match" in reasons["exact"]
    assert "target_crop_or_transform_near_duplicate" in reasons["crop"]
    assert "target_perceptual_near_duplicate" in reasons["rotated"]
    assert "target_source_image_match" in reasons["same-source"]
    assert "target_capture_family_match" in reasons["same-family"]
    assert report.checks["descriptor_similarity"] == "not_run_no_real_descriptor_artifact"


def test_leakage_audit_uses_only_supplied_descriptors_and_reports_missing_artifact(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    reference = tmp_path / "reference.png"
    _image(target, 0)
    _image(reference, 7)
    manifest = _manifest(tmp_path / "holdout.json", [_record("holdout-1", target)])
    holdout = HoldoutManifestLoader().load(manifest).records[0]
    item = _reference(reference, key="reference")

    without_descriptors = ReferenceLeakageAuditor().audit(holdout, (item,))
    assert without_descriptors.checks["descriptor_similarity"] == (
        "not_run_no_real_descriptor_artifact"
    )
    artifact_path = tmp_path / "real-descriptor-output.npz"
    np.savez(
        artifact_path,
        provider=np.asarray("real-descriptor-test-fixture"),
        version=np.asarray("fixture-revision"),
        content_sha256=np.asarray([holdout.content_sha256, item.content_hash]),
        vectors=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    artifact = load_descriptor_artifact(artifact_path)
    with_descriptors = ReferenceLeakageAuditor().audit(
        holdout,
        (item,),
        descriptor_artifact=artifact,
    )
    assert with_descriptors.status == "failed"
    assert "target_descriptor_near_duplicate" in with_descriptors.exclusions[0].reasons
    assert with_descriptors.descriptor_checked_count == 1


def test_leakage_audit_refuses_reference_count_over_policy(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(target, 0)
    _image(first, 1)
    _image(second, 2)
    manifest = _manifest(tmp_path / "holdout.json", [_record("holdout-1", target)])
    holdout = HoldoutManifestLoader().load(manifest).records[0]
    auditor = ReferenceLeakageAuditor(LeakageAuditPolicy(max_references=1))

    with pytest.raises(LeakageAuditError, match="count_limit"):
        auditor.audit(
            holdout,
            (_reference(first, key="one"), _reference(second, key="two")),
        )
