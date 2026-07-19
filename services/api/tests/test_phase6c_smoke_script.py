from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]


def _script() -> ModuleType:
    path = ROOT / "scripts" / "smoke_phase6c_http.py"
    spec = importlib.util.spec_from_file_location("phase6c_smoke_http", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_smoke_report_drops_coordinates_ocr_text_and_reference_ids() -> None:
    module = _script()
    result = {
        "id": "00000000-0000-0000-0000-000000000001",
        "status": "completed",
        "result_classification": "real",
        "pipeline_version": "phase6c-v1",
        "image": {
            "sha256": "a" * 64,
            "format": "jpeg",
            "width": 100,
            "height": 80,
            "exif_present": False,
            "orientation_normalized": True,
        },
        "ocr": {
            "provider": "rapidocr",
            "status": "completed",
            "detections": [{"redacted_text": "PRIVATE OCR TOKEN"}],
            "place_evidence": [],
        },
        "phase6c": {
            "fusion_version": "phase6c-v1",
            "cache_fingerprint": f"phase6c-v1:{'b' * 64}",
            "hierarchical_candidates": [
                {
                    "provider_rank": 1,
                    "latitude": 12.34,
                    "longitude": 56.78,
                    "search_level": "catalogue",
                    "nearest_name": "Named place",
                }
            ],
            "megaloc_matches": [
                {
                    "rank": 1,
                    "latitude": 12.34,
                    "longitude": 56.78,
                    "reference_id": "private-reference-id",
                    "source_sequence_id": "private-sequence-id",
                    "source_url": "https://example.test/private",
                    "source": "kartaview",
                    "attribution": "Reviewed attribution",
                }
            ],
            "fusion_candidates": [],
            "ablations": [
                {
                    "profile_id": "phase6c_final",
                    "candidate_count": 1,
                    "publication_candidate_count": 1,
                    "abstained": False,
                    "candidates": [
                        {
                            "rank": 1,
                            "latitude": 12.34,
                            "longitude": 56.78,
                            "relative_rank_score": 0.5,
                            "publication_eligible": True,
                            "evidence_kinds": ["megaloc_retrieval"],
                        }
                    ],
                }
            ],
            "providers": [],
            "leakage_audit": {"status": "passed"},
        },
        "candidates": [],
        "timings_ms": {},
    }

    report = module.build_safe_report(result)
    encoded = json.dumps(report)

    assert "PRIVATE OCR TOKEN" not in encoded
    assert "private-reference-id" not in encoded
    assert "private-sequence-id" not in encoded
    assert "https://example.test/private" not in encoded
    assert '"latitude"' not in encoded
    assert '"longitude"' not in encoded
    assert report["ocr"]["detection_count"] == 1
    assert report["phase6c"]["retrieval"]["independent_sequence_count"] == 1


def test_powershell_smoke_uses_real_loopback_http_and_disables_cloud() -> None:
    script = (ROOT / "scripts" / "smoke_phase6c.ps1").read_text(encoding="utf-8")

    assert "uvicorn" in script
    assert '"PHASE6C_ENABLED" = "true"' in script
    assert '"OPENAI_GEO_ENABLED" = "false"' in script
    assert "http://127.0.0.1:$Port" in script
    assert "verify_megaloc_reference_index.py" in script
    assert "verify_phase6c_workers.py" in script
    assert "TestClient" not in script


def test_http_smoke_cli_supports_retained_source_recomputation() -> None:
    script = (ROOT / "scripts" / "smoke_phase6c_http.py").read_text(encoding="utf-8")

    assert "--rerun-analysis-id" in script
    assert '"retained_source_recomputation"' in script
    assert "/rerun" in script
