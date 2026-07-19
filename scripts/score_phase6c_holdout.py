#!/usr/bin/env python3
"""Score a completed Phase 6C HTTP analysis after prediction, without rerunning it."""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from atlaslens_api.evaluation.holdout import HoldoutManifestError, HoldoutManifestLoader
from atlaslens_api.evaluation.metrics import haversine_km
from atlaslens_api.evaluation.phase6c_http import Phase6CHTTPWorkerConfig
from atlaslens_api.phase6c.catalogue import load_coordinate_catalogue


def _normalize(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return normalized or None


def _coordinate(item: dict[str, Any]) -> tuple[float, float] | None:
    center = item.get("center")
    source = center if isinstance(center, dict) else item
    latitude = source.get("latitude")
    longitude = source.get("longitude")
    if (
        isinstance(latitude, bool)
        or isinstance(longitude, bool)
        or not isinstance(latitude, int | float)
        or not isinstance(longitude, int | float)
    ):
        return None
    lat, lon = float(latitude), float(longitude)
    if not math.isfinite(lat) or not math.isfinite(lon):
        return None
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return None
    return lat, lon


def _rank(item: dict[str, Any], default: int) -> int:
    for name in ("rank", "provider_rank", "final_rank"):
        value = item.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return default


def _metrics(
    values: Iterable[dict[str, Any]],
    *,
    truth_latitude: float,
    truth_longitude: float,
) -> dict[str, Any]:
    rows: list[tuple[int, float]] = []
    for default_rank, item in enumerate(values, start=1):
        coordinate = _coordinate(item)
        if coordinate is None:
            continue
        rows.append(
            (
                _rank(item, default_rank),
                haversine_km(
                    truth_latitude,
                    truth_longitude,
                    coordinate[0],
                    coordinate[1],
                ),
            )
        )
    nearest = min(rows, key=lambda row: (row[1], row[0])) if rows else None
    return {
        "candidate_count": len(rows),
        "nearest_candidate_distance_km": round(nearest[1], 3) if nearest else None,
        "nearest_candidate_rank": nearest[0] if nearest else None,
        "recall_within_km": {
            str(radius): any(distance <= radius for _, distance in rows)
            for radius in (25, 100, 250, 750)
        },
    }


def _named_rank(
    values: Iterable[dict[str, Any]],
    expected: str,
    selector: Callable[[dict[str, Any]], object],
) -> int | None:
    normalized_expected = _normalize(expected)
    for default_rank, item in enumerate(values, start=1):
        if _normalize(selector(item)) == normalized_expected:
            return _rank(item, default_rank)
    return None


def score_completed_analysis(
    analysis: dict[str, Any],
    *,
    truth_city: str,
    truth_country: str,
    truth_latitude: float,
    truth_longitude: float,
) -> dict[str, Any]:
    """Return only aggregate post-prediction metrics; never retain exact coordinates."""

    phase6c = analysis.get("phase6c") or {}
    hierarchy = phase6c.get("hierarchical_candidates") or []
    retrieval = phase6c.get("megaloc_matches") or []
    fused = phase6c.get("fusion_candidates") or []
    public = analysis.get("candidates") or []
    model_predictions = analysis.get("model_predictions") or {}
    ocr = analysis.get("ocr") or {}
    ocr_places = ocr.get("place_evidence") or []

    def fusion_with(kind: str) -> list[dict[str, Any]]:
        return [
            item
            for item in fused
            if any(
                member.get("evidence_kind") == kind
                for member in (item.get("members") or [])
            )
        ]

    osv = [
        candidate
        for prediction in model_predictions.values()
        if str(prediction.get("provider", "")).startswith("osv5m")
        for candidate in (prediction.get("candidates") or [])
    ]
    plonk = [
        candidate
        for prediction in model_predictions.values()
        if str(prediction.get("provider", "")).startswith("plonk")
        for candidate in (prediction.get("candidates") or [])
    ]
    components = {
        "geoclip_original_fusion_modes": fusion_with("geoclip_original"),
        "hierarchical": hierarchy,
        "osv5m": osv,
        "plonk": plonk,
        "retrieval": retrieval,
        "ocr": ocr_places,
        "g3_verified_fusion_modes": fusion_with("g3_verification"),
        "final_fusion": fused,
        "public": public,
    }
    component_metrics = {
        name: _metrics(
            values,
            truth_latitude=truth_latitude,
            truth_longitude=truth_longitude,
        )
        for name, values in components.items()
    }

    hierarchy_city_rank = _named_rank(
        hierarchy, truth_city, lambda item: item.get("nearest_name")
    )
    retrieval_city_rank = _named_rank(
        retrieval, truth_city, lambda item: item.get("city")
    )
    retrieval_province_rank = _named_rank(
        retrieval, truth_city, lambda item: item.get("province")
    )
    ocr_city_rank = _named_rank(
        ocr_places, truth_city, lambda item: item.get("normalized_name")
    )
    public_city_rank = _named_rank(
        public,
        truth_city,
        lambda item: (item.get("reverse_geocode") or {}).get("city")
        or item.get("label"),
    )
    named_entry = next(
        (
            name
            for name, rank in (
                ("hierarchical", hierarchy_city_rank),
                ("retrieval", retrieval_city_rank),
                ("ocr", ocr_city_rank),
                ("public", public_city_rank),
            )
            if rank is not None
        ),
        None,
    )

    ablations: dict[str, Any] = {}
    for item in phase6c.get("ablations") or []:
        profile_id = item.get("profile_id")
        if not isinstance(profile_id, str):
            continue
        ablations[profile_id] = {
            **_metrics(
                item.get("candidates") or [],
                truth_latitude=truth_latitude,
                truth_longitude=truth_longitude,
            ),
            "publication_candidate_count": item.get("publication_candidate_count"),
            "abstained": item.get("abstained"),
            "abstention_reason": item.get("abstention_reason"),
        }

    top_public = public[0] if public else None
    top_reverse = (top_public or {}).get("reverse_geocode") or {}
    top_country = (top_public or {}).get("country_code") or top_reverse.get("country_code")
    cloud = analysis.get("cloud_assist") or {}
    return {
        "schema_version": "atlaslens-phase6c-holdout-score-v1",
        "predictions_completed_before_scoring": True,
        "pipeline_version": analysis.get("pipeline_version"),
        "fusion_version": phase6c.get("fusion_version"),
        "evaluation_truth": {
            "city": truth_city,
            "country": truth_country,
            "coordinate_semantics": (
                "post_prediction_versioned_catalogue_province_centroid_not_camera_coordinate"
            ),
        },
        "country_top1_correct": top_country == truth_country,
        "city_rank": public_city_rank,
        "named_candidate_entry": {
            "first_component": named_entry,
            "hierarchical_city_rank": hierarchy_city_rank,
            "retrieval_city_rank": retrieval_city_rank,
            "retrieval_province_rank": retrieval_province_rank,
            "ocr_city_rank": ocr_city_rank,
            "public_city_rank": public_city_rank,
        },
        "components": component_metrics,
        "ablations": ablations,
        "leakage_audit": phase6c.get("leakage_audit"),
        "openai": {
            "triggered": cloud.get("triggered", False),
            "status": cloud.get("status", "not_requested"),
            "estimated_cost_usd": cloud.get("estimated_cost_usd"),
        },
        "total_latency_ms": (analysis.get("timings_ms") or {}).get("total"),
        "accuracy_claim": None,
        "privacy": {
            "exact_coordinates_persisted": False,
            "raw_ocr_persisted": False,
            "image_path_persisted": False,
        },
    }


def _load_prediction_report(path: Path) -> UUID:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("prediction_report_invalid")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "atlaslens-phase6c-http-smoke-v1"
        or payload.get("status") != "completed"
        or payload.get("pipeline_version") != "phase6c-v1"
    ):
        raise ValueError("prediction_report_invalid")
    return UUID(str(payload.get("analysis_id")))


def _fetch_analysis(base_url: str, analysis_id: UUID) -> dict[str, Any]:
    Phase6CHTTPWorkerConfig(api_base_url=base_url)
    with (
        httpx.Client(base_url=base_url, trust_env=False, timeout=30) as client,
        client.stream("GET", f"/api/v1/analyses/{analysis_id}") as response,
    ):
        if response.status_code != 200:
            raise ValueError("analysis_unavailable")
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > 16 * 1024 * 1024:
                raise ValueError("analysis_response_too_large")
    payload = json.loads(body)
    if (
        not isinstance(payload, dict)
        or payload.get("id") != str(analysis_id)
        or payload.get("status") != "completed"
        or payload.get("pipeline_version") != "phase6c-v1"
        or not isinstance(payload.get("phase6c"), dict)
    ):
        raise ValueError("analysis_response_invalid")
    return payload


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("score_output_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8760")
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--holdout-id", required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        analysis_id = _load_prediction_report(args.prediction_report.resolve(strict=True))
        # The complete prediction is fetched before the truth-bearing manifest is loaded.
        analysis = _fetch_analysis(args.api_base_url, analysis_id)
        manifest = HoldoutManifestLoader(max_records=1).load(args.holdout_manifest)
        matches = [item for item in manifest.records if item.metadata.id == args.holdout_id]
        if len(matches) != 1 or matches[0].metadata.split != "final_holdout":
            raise ValueError("holdout_record_invalid")
        truth = matches[0].metadata
        catalogue = load_coordinate_catalogue(args.catalogue)
        anchors = [
            item
            for item in catalogue.records
            if item.kind == "turkiye_province"
            and item.country_code == truth.country
            and _normalize(item.name) == _normalize(truth.city)
        ]
        if len(anchors) != 1:
            raise ValueError("truth_catalogue_anchor_invalid")
        score = score_completed_analysis(
            analysis,
            truth_city=truth.city,
            truth_country=truth.country,
            truth_latitude=anchors[0].latitude,
            truth_longitude=anchors[0].longitude,
        )
        _write(args.output.resolve(), score)
    except (
        HoldoutManifestError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        httpx.HTTPError,
    ):
        print(
            json.dumps(
                {"event": "phase6c_holdout_scoring_failed", "reason_code": "scoring_failed"}
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "event": "phase6c_holdout_scored",
                "city_rank": score["city_rank"],
                "accuracy_claim": None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
