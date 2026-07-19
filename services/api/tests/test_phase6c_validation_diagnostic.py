from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from atlaslens_api.evaluation.models import EvaluationRecord
from atlaslens_api.evaluation.phase6c_validation import (
    Phase6CValidationConfig,
    build_validation_diagnostic,
    main,
)

_OLD_ID = "11111111-1111-4111-8111-111111111111"
_NEW_ID = "22222222-2222-4222-8222-222222222222"
_TRUTH_LATITUDE = 38.72
_TRUTH_LONGITUDE = 35.48


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


def _manifest(tmp_path: Path) -> tuple[Path, Path, tuple[str, str]]:
    root = tmp_path / "private-assets"
    root.mkdir()
    first = root / "first-private.jpg"
    second = root / "second-private.jpg"
    Image.new("RGB", (24, 16), (20, 80, 140)).save(first, "JPEG")
    Image.new("RGB", (24, 16), (120, 30, 70)).save(second, "JPEG")
    hashes = (
        hashlib.sha256(first.read_bytes()).hexdigest(),
        hashlib.sha256(second.read_bytes()).hexdigest(),
    )
    rows = []
    for index, (path, digest) in enumerate(zip((first, second), hashes, strict=True), 1):
        rows.append(
            {
                "image_asset_key": f"private-asset-{index}",
                "local_reference": path.name,
                "true_latitude": str(_TRUTH_LATITUDE + index - 1),
                "true_longitude": str(_TRUTH_LONGITUDE + index - 1),
                "country_code": "TR",
                "region": f"Private Region {index}",
                "city_or_area": f"Private City {index}",
                "continent": "Asia",
                "source": "kartaview",
                "source_record_id": f"private-source-{index}",
                "license": "CC BY-SA 4.0",
                "attribution": "Fixture attribution",
                "split": "validation",
                "scene_category": "urban",
                "geographic_cell": f"private-cell-{index}",
                "capture_family_id": f"private-family-{index}",
                "content_sha256": digest,
                "perceptual_hash": _average_hash(path),
            }
        )
    manifest = tmp_path / "private-manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(EvaluationRecord.model_fields))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, root, hashes


def _analysis(analysis_id: str, *, newest: bool) -> dict[str, Any]:
    latitude = _TRUTH_LATITUDE if newest else 0.0
    longitude = _TRUTH_LONGITUDE if newest else 0.0
    coordinate = {"latitude": latitude, "longitude": longitude}
    return {
        "id": analysis_id,
        "status": "completed",
        "pipeline_version": "phase6c-v1",
        "result_classification": "real",
        "candidates": [{"center": coordinate, "source": "private ignored"}],
        "model_predictions": {
            "opaque-osv-key": {
                "provider": "osv5m/baseline",
                "candidates": [coordinate],
            },
            "opaque-plonk-key": {
                "provider": "plonk/yfcc",
                "candidates": [coordinate],
            },
        },
        "ocr": {
            "place_evidence": [
                {
                    "center": coordinate,
                    "normalized_name": "SECRET RAW OCR TOKEN",
                }
            ]
        },
        "phase6c": {
            "pipeline_version": "phase6c-v1",
            "hierarchical_candidates": [coordinate],
            "megaloc_matches": [
                {
                    **coordinate,
                    "source_url": "https://secret.invalid/private-reference",
                }
            ],
            "fusion_candidates": [coordinate],
            "providers": [
                {
                    "provider_id": "megaloc",
                    "status": "completed",
                    "duration_ms": 20 if newest else 10,
                    "candidates_produced": 1,
                }
            ],
            "ablations": [
                {
                    "profile_id": "phase6c_final",
                    "candidates": [coordinate],
                },
                {
                    "profile_id": "without_retrieval",
                    "candidates": [],
                },
            ],
        },
    }


def _transport(
    first_hash: str,
    *,
    methods: list[str] | None = None,
    events: list[str] | None = None,
) -> httpx.MockTransport:
    history = {
        "items": [
            {
                "id": _OLD_ID,
                "created_at": "2026-07-14T10:00:00+00:00",
                "status": "completed",
                "result_classification": "real",
                "image_sha256": first_hash,
            },
            {
                "id": _NEW_ID,
                "created_at": "2026-07-14T11:00:00+00:00",
                "status": "completed",
                "result_classification": "real",
                "image_sha256": first_hash,
            },
        ],
        "total": 2,
        "limit": 100,
        "offset": 0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if methods is not None:
            methods.append(request.method)
        if request.url.path == "/api/v1/analyses":
            if events is not None:
                events.append("history")
            assert request.method == "GET"
            assert request.url.params["status"] == "completed"
            assert request.url.params["classification"] == "real"
            return httpx.Response(200, json=history)
        if request.url.path == f"/api/v1/analyses/{_OLD_ID}":
            if events is not None:
                events.append("full_analysis")
            return httpx.Response(200, json=_analysis(_OLD_ID, newest=False))
        if request.url.path == f"/api/v1/analyses/{_NEW_ID}":
            if events is not None:
                events.append("full_analysis")
            return httpx.Response(200, json=_analysis(_NEW_ID, newest=True))
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_scores_newest_completed_analysis_with_truth_loaded_after_gets(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest, root, hashes = _manifest(tmp_path)
    methods: list[str] = []
    events: list[str] = []

    from atlaslens_api.evaluation import phase6c_validation

    original_load = phase6c_validation.EvaluationManifestLoader.load

    def tracked_load(self: Any, manifest_path: Path, asset_root: Path) -> Any:
        events.append("truth_load")
        return original_load(self, manifest_path, asset_root)

    monkeypatch.setattr(phase6c_validation.EvaluationManifestLoader, "load", tracked_load)
    report = build_validation_diagnostic(
        config=Phase6CValidationConfig(api_base_url="http://127.0.0.1:8761"),
        manifest_path=manifest,
        asset_root=root,
        allowed_licenses=frozenset({"CC BY-SA 4.0"}),
        transport=_transport(hashes[0], methods=methods, events=events),
    )

    assert methods and set(methods) == {"GET"}
    assert events[-1] == "truth_load"
    assert report["sample_count"] == 2
    assert report["analysis_join"] == {
        "matched_completed_phase6c": 1,
        "unmatched": 1,
        "coverage": {"numerator": 1, "denominator": 2, "value": 0.5},
        "duplicate_analyses_resolved": 1,
    }
    assert report["components"]["hierarchical"]["recall_within_km"]["25"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert report["components"]["hierarchical"]["nearest_error_km"] == {
        "denominator": 1,
        "mean_km": 0.0,
        "median_km": 0.0,
        "p95_km": 0.0,
    }
    assert set(report["components"]) == {
        "hierarchical",
        "megaloc",
        "osv5m",
        "plonk",
        "ocr_place_evidence",
        "final_fusion",
        "public_candidates",
    }
    assert set(report["ablations"]) == {"phase6c_final", "without_retrieval"}
    assert report["providers"]["megaloc"]["status_counts"] == {"completed": 1}
    assert report["providers"]["megaloc"]["latency_ms"]["mean_ms"] == 20.0
    assert report["no_claim"] is True

    encoded = json.dumps(report, sort_keys=True)
    for forbidden in (
        str(manifest),
        str(root),
        hashes[0],
        hashes[1],
        _OLD_ID,
        _NEW_ID,
        "38.72",
        "35.48",
        "SECRET RAW OCR TOKEN",
        "secret.invalid",
        "private-source",
        "private-asset",
    ):
        assert forbidden not in encoded


def test_cli_writes_sanitized_report_and_uses_stable_path_free_errors(
    tmp_path: Path,
) -> None:
    manifest, root, hashes = _manifest(tmp_path)
    output = tmp_path / "reports" / "validation.json"
    stdout = io.StringIO()
    stderr = io.StringIO()
    arguments = [
        "--api-base-url",
        "http://127.0.0.1:8761",
        "--manifest",
        str(manifest),
        "--asset-root",
        str(root),
        "--allowed-license",
        "CC BY-SA 4.0",
        "--output",
        str(output),
    ]

    status = main(
        arguments,
        stdout=stdout,
        stderr=stderr,
        transport=_transport(hashes[0]),
    )

    assert status == 0
    assert stderr.getvalue() == ""
    assert json.loads(stdout.getvalue()) == {
        "event": "phase6c_validation_diagnostic_completed",
        "matched_completed_phase6c": 1,
        "no_claim": True,
        "sample_count": 2,
    }
    written = output.read_text(encoding="utf-8")
    assert '"no_claim": true' in written
    for forbidden in (str(manifest), str(root), hashes[0], _NEW_ID, "38.72", "35.48"):
        assert forbidden not in written
        assert forbidden not in stdout.getvalue()

    bad_stderr = io.StringIO()
    assert (
        main(
            [
                *arguments[:1],
                "https://private.invalid",
                *arguments[2:-1],
                str(tmp_path / "unused.json"),
            ],
            stdout=io.StringIO(),
            stderr=bad_stderr,
        )
        == 2
    )
    error = json.loads(bad_stderr.getvalue())
    assert error == {
        "event": "phase6c_validation_diagnostic_failed",
        "reason_code": "api_base_url_invalid",
    }
    assert "private.invalid" not in bad_stderr.getvalue()
    assert str(manifest) not in bad_stderr.getvalue()
