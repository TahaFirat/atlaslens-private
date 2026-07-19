from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]


def _module() -> ModuleType:
    path = ROOT / "scripts" / "score_phase6c_holdout.py"
    spec = importlib.util.spec_from_file_location("score_phase6c_holdout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scorer_reports_component_recall_without_coordinates_or_ocr_text() -> None:
    module = _module()
    analysis = {
        "pipeline_version": "phase6c-v1",
        "timings_ms": {"total": 1234},
        "phase6c": {
            "fusion_version": "phase6c-v1",
            "hierarchical_candidates": [
                {
                    "provider_rank": 1,
                    "latitude": 39.0,
                    "longitude": 35.0,
                    "nearest_name": "Target City",
                }
            ],
            "megaloc_matches": [
                {
                    "rank": 1,
                    "latitude": 39.0,
                    "longitude": 35.0,
                    "province": "Target City",
                    "city": "Target City",
                }
            ],
            "fusion_candidates": [
                {
                    "final_rank": 1,
                    "latitude": 39.0,
                    "longitude": 35.0,
                    "members": [{"evidence_kind": "megaloc_retrieval"}],
                }
            ],
            "ablations": [],
            "leakage_audit": {"status": "passed"},
        },
        "model_predictions": {},
        "ocr": {
            "place_evidence": [
                {
                    "normalized_name": "PRIVATE OCR TOKEN",
                    "center": {"latitude": 39.0, "longitude": 35.0},
                }
            ]
        },
        "candidates": [
            {
                "rank": 1,
                "center": {"latitude": 39.0, "longitude": 35.0},
                "reverse_geocode": {"city": "Target City", "country_code": "TR"},
            }
        ],
    }

    score = module.score_completed_analysis(
        analysis,
        truth_city="Target City",
        truth_country="TR",
        truth_latitude=39.0,
        truth_longitude=35.0,
    )
    encoded = json.dumps(score)

    assert score["city_rank"] == 1
    assert score["named_candidate_entry"]["first_component"] == "hierarchical"
    assert score["components"]["retrieval"]["recall_within_km"]["25"] is True
    assert "PRIVATE OCR TOKEN" not in encoded
    assert '"latitude"' not in encoded
    assert '"longitude"' not in encoded
