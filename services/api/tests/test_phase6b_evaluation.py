from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from atlaslens_api.phase6b.evaluation import (
    ABLATION_PROFILES,
    EvaluationError,
    Phase6BEvaluationHarness,
    Phase6BEvaluationObservation,
)


def _manifest(tmp_path: Path, count: int = 2) -> Path:
    records = []
    for index in range(count):
        image = tmp_path / f"image-{index}.jpg"
        Image.new("RGB", (8, 8), (index, 20, 30)).save(image)
        records.append(
            {
                "image": image.name,
                "latitude": 38.72 + index,
                "longitude": 35.48,
                "country": "TR",
                "city": "Kayseri",
                "license": "user-provided",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    return manifest


def _observation(profile: str, image: str, *, latitude: float) -> dict[str, object]:
    return {
        "profile": profile,
        "image": image,
        "predictions": [
            {
                "latitude": latitude,
                "longitude": 35.48,
                "country": "TR",
                "city": "Kayseri",
            }
        ],
        "providers": [
            {"provider": "geoclip", "status": "completed", "latency_ms": 12.0}
        ],
        "cloud": {
            "triggered": profile == "optional_cloud_assistance",
            "cache_hit": False,
            "estimated_cost_usd": (
                0.001 if profile == "optional_cloud_assistance" else 0.0
            ),
        },
    }


def test_phase6b_evaluation_reports_all_ablations_and_small_sample_gate(tmp_path: Path) -> None:
    harness = Phase6BEvaluationHarness()
    manifest, fingerprint = harness.load_manifest(_manifest(tmp_path))
    records = tuple(
        Phase6BEvaluationObservation.model_validate(
            _observation(profile, item.image, latitude=item.latitude)
        )
        for profile in ABLATION_PROFILES
        for item in manifest
    )

    report = harness.evaluate(manifest, records, manifest_fingerprint=fingerprint)

    assert report.sample_count == 2
    assert not report.improvement_claim_allowed
    assert all(profile.complete for profile in report.profiles)
    assert report.profiles[0].top1_error.median_km == pytest.approx(0)
    assert report.profiles[-1].openai_trigger_rate.value == 1
    assert report.profiles[-1].estimated_openai_cost_usd == pytest.approx(0.002)
    assert report.profiles[-1].fusion_delta_from_geoclip is not None


def test_phase6b_evaluation_marks_missing_profiles_incomplete(tmp_path: Path) -> None:
    harness = Phase6BEvaluationHarness()
    manifest, fingerprint = harness.load_manifest(_manifest(tmp_path, count=1))
    record = Phase6BEvaluationObservation.model_validate(
        _observation("geoclip_only", manifest[0].image, latitude=manifest[0].latitude)
    )

    report = harness.evaluate(manifest, (record,), manifest_fingerprint=fingerprint)

    assert report.profiles[0].complete
    assert report.profiles[1].observed_sample_count == 0
    assert report.profiles[1].prediction_return.denominator == 0
    assert not report.improvement_claim_allowed


def test_phase6b_evaluation_rejects_unsafe_manifest_and_unknown_results(
    tmp_path: Path,
) -> None:
    harness = Phase6BEvaluationHarness()
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(
        json.dumps(
            [
                {
                    "image": "../outside.jpg",
                    "latitude": 1,
                    "longitude": 2,
                    "country": "TR",
                    "city": None,
                    "license": "user-provided",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="image_path_unsafe"):
        harness.load_manifest(unsafe)

    manifest, fingerprint = harness.load_manifest(_manifest(tmp_path))
    unknown = Phase6BEvaluationObservation.model_validate(
        _observation("geoclip_only", "missing.jpg", latitude=0)
    )
    with pytest.raises(EvaluationError, match="not_in_manifest"):
        harness.evaluate(manifest, (unknown,), manifest_fingerprint=fingerprint)
