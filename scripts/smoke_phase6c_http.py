#!/usr/bin/env python3
"""Run one image through an already-running Phase 6C HTTP API and save a safe report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx


def _wait_for_api(client: httpx.Client, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = client.get("/api/v1/ready")
            if response.status_code == 200 and response.json().get("status") == "ready":
                return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("phase6c_api_not_ready")


def _wait_for_analysis(
    client: httpx.Client,
    analysis_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/analyses/{analysis_id}")
        response.raise_for_status()
        result = response.json()
        if result.get("status") in {"completed", "failed"}:
            return result
        time.sleep(0.25)
    raise RuntimeError("phase6c_analysis_timeout")


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    reverse = candidate.get("reverse_geocode") or {}
    assessment = candidate.get("phase5b_assessment") or {}
    return {
        "rank": candidate.get("rank"),
        "label": candidate.get("label"),
        "city": reverse.get("city"),
        "province": reverse.get("region"),
        "country": reverse.get("country"),
        "country_code": candidate.get("country_code") or reverse.get("country_code"),
        "radius_km": candidate.get("radius_km"),
        "confidence": candidate.get("confidence"),
        "confidence_kind": candidate.get("confidence_kind"),
        "verification_status": candidate.get("verification_status"),
        "source": candidate.get("source"),
        "relative_rank_score": assessment.get("relative_rank_score"),
        "supporting_sources": assessment.get("supporting_sources", [])[:24],
        "contradictions": assessment.get("contradictions", [])[:12],
    }


def _model_summary(prediction: dict[str, Any]) -> dict[str, Any]:
    candidates = prediction.get("candidates") or []
    return {
        "provider": prediction.get("provider"),
        "model_id": prediction.get("model_id"),
        "model_revision": prediction.get("model_revision"),
        "source_family": prediction.get("source_family"),
        "status": prediction.get("status"),
        "device": prediction.get("device"),
        "duration_ms": prediction.get("duration_ms"),
        "candidate_count": len(candidates),
        "reason_code": prediction.get("reason_code"),
        "warnings": prediction.get("warnings", [])[:20],
    }


def build_safe_report(result: dict[str, Any]) -> dict[str, Any]:
    """Project a full API result to an aggregate report with no coordinates or OCR text."""

    phase6c = result.get("phase6c") or {}
    hierarchy = phase6c.get("hierarchical_candidates") or []
    retrieval = phase6c.get("megaloc_matches") or []
    fused = phase6c.get("fusion_candidates") or []
    ocr = result.get("ocr") or {}
    scene = result.get("scene_analysis") or {}
    cloud = result.get("cloud_assist") or {}
    image = result.get("image") or {}

    report: dict[str, Any] = {
        "schema_version": "atlaslens-phase6c-http-smoke-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "analysis_id": result.get("id"),
        "status": result.get("status"),
        "result_classification": result.get("result_classification"),
        "pipeline_version": result.get("pipeline_version"),
        "input": {
            "sha256": image.get("sha256"),
            "format": image.get("format"),
            "width": image.get("width"),
            "height": image.get("height"),
            "exif_present": image.get("exif_present"),
            "orientation_normalized": image.get("orientation_normalized"),
        },
        "quality": result.get("quality"),
        "segmentation": {
            "status": scene.get("status", "unavailable"),
            "provider": scene.get("provider"),
            "device": scene.get("device"),
            "inference_ms": scene.get("inference_ms"),
            "semantic_label_names_available": scene.get(
                "semantic_label_names_available", False
            ),
            "dominant_classes": [
                {
                    "class_name": item.get("class_name"),
                    "pixel_ratio": item.get("pixel_ratio"),
                }
                for item in (scene.get("dominant_classes") or [])[:12]
            ],
            "warnings": scene.get("warnings", [])[:20],
        },
        "ocr": {
            "provider": ocr.get("provider"),
            "status": ocr.get("status", "unavailable"),
            "detection_count": len(ocr.get("detections") or []),
            "place_evidence_count": len(ocr.get("place_evidence") or []),
            "fallback_used": ocr.get("fallback_used", False),
            "reason_code": ocr.get("reason_code"),
            "raw_text_persisted": False,
        },
        "phase6b_models": {
            name: _model_summary(prediction)
            for name, prediction in (result.get("model_predictions") or {}).items()
        },
        "phase6b_fusion": {
            "version": (result.get("fusion") or {}).get("version"),
            "cluster_count": len((result.get("fusion") or {}).get("candidate_clusters") or []),
            "agreement": (result.get("fusion") or {}).get("agreement_summary"),
            "provider_failures": (result.get("fusion") or {}).get(
                "provider_failures", []
            )[:24],
        },
        "phase6c": {
            "fusion_version": phase6c.get("fusion_version"),
            "reference_index_version": phase6c.get("reference_index_version"),
            "cache_fingerprint": phase6c.get("cache_fingerprint"),
            "turkiye_signal_count": phase6c.get("turkiye_signal_count"),
            "turkiye_signal_groups": phase6c.get("turkiye_signal_groups", []),
            "providers": phase6c.get("providers", []),
            "hierarchical": {
                "candidate_count": len(hierarchy),
                "search_level_distribution": dict(
                    sorted(Counter(item.get("search_level") for item in hierarchy).items())
                ),
                "nearest_named_places": [
                    {
                        "rank": item.get("provider_rank"),
                        "name": item.get("nearest_name"),
                        "kind": item.get("nearest_kind"),
                        "distance_km": item.get("nearest_distance_km"),
                        "search_level": item.get("search_level"),
                        "raw_score": item.get("provider_score"),
                    }
                    for item in hierarchy[:16]
                ],
            },
            "retrieval": {
                "match_count": len(retrieval),
                "source_distribution": dict(
                    sorted(Counter(item.get("source") for item in retrieval).items())
                ),
                "independent_sequence_count": len(
                    {item.get("source_sequence_id") for item in retrieval}
                ),
                "matches": [
                    {
                        "rank": item.get("rank"),
                        "similarity": item.get("similarity"),
                        "similarity_semantics": item.get("similarity_semantics"),
                        "province": item.get("province"),
                        "city": item.get("city"),
                        "source": item.get("source"),
                        "attribution": item.get("attribution"),
                    }
                    for item in retrieval[:50]
                ],
            },
            "fusion": {
                "candidate_count": len(fused),
                "publication_candidate_count": phase6c.get(
                    "publication_candidate_count", 0
                ),
                "candidates": [
                    {
                        "rank": item.get("final_rank"),
                        "relative_rank_score": item.get("relative_rank_score"),
                        "score_semantics": item.get("score_semantics"),
                        "uncertainty_radius_km": item.get("uncertainty_radius_km"),
                        "publication_eligible": item.get("publication_eligible"),
                        "publication_basis": item.get("publication_basis"),
                        "provider_count": item.get("provider_count"),
                        "independent_source_family_count": item.get(
                            "independent_source_family_count"
                        ),
                        "members": [
                            {
                                "evidence_kind": member.get("evidence_kind"),
                                "provider": member.get("provider"),
                                "source_family": member.get("source_family"),
                                "correlation_group": member.get("correlation_group"),
                                "provider_rank": member.get("provider_rank"),
                                "raw_value": member.get("raw_value"),
                            }
                            for member in (item.get("members") or [])[:32]
                        ],
                        "contributions": item.get("contributions", [])[:16],
                    }
                    for item in fused[:16]
                ],
            },
            "leakage_audit": phase6c.get("leakage_audit"),
            "reference_attributions": phase6c.get("reference_attributions", [])[:20],
            "ablations": [
                {
                    "profile_id": item.get("profile_id"),
                    "candidate_count": item.get("candidate_count"),
                    "publication_candidate_count": item.get(
                        "publication_candidate_count"
                    ),
                    "abstained": item.get("abstained"),
                    "abstention_reason": item.get("abstention_reason"),
                    "included_evidence_kinds": item.get(
                        "included_evidence_kinds", []
                    ),
                    "excluded_evidence_kinds": item.get(
                        "excluded_evidence_kinds", []
                    ),
                    "candidates": [
                        {
                            "rank": candidate.get("rank"),
                            "relative_rank_score": candidate.get(
                                "relative_rank_score"
                            ),
                            "publication_eligible": candidate.get(
                                "publication_eligible"
                            ),
                            "evidence_kinds": candidate.get("evidence_kinds", []),
                        }
                        for candidate in (item.get("candidates") or [])[:5]
                    ],
                }
                for item in (phase6c.get("ablations") or [])[:16]
            ],
        },
        "public_candidates": [
            _candidate_summary(candidate) for candidate in (result.get("candidates") or [])[:20]
        ],
        "cloud_assist": {
            "allowed": cloud.get("allowed", False),
            "triggered": cloud.get("triggered", False),
            "status": cloud.get("status", "not_requested"),
            "model": cloud.get("model"),
            "reason": cloud.get("reason"),
            "cache_hit": cloud.get("cache_hit", False),
            "estimated_cost_usd": cloud.get("estimated_cost_usd"),
        },
        "warnings": result.get("warnings", [])[:40],
        "timings_ms": result.get("timings_ms", {}),
        "failure": result.get("failure"),
        "privacy": {
            "raw_image_persisted_in_report": False,
            "raw_ocr_persisted_in_report": False,
            "exact_coordinates_persisted_in_report": False,
            "original_filename_persisted_in_report": False,
            "evaluation_ground_truth_available_to_inference": False,
        },
        "accuracy_claim": None,
    }
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError("phase6c_smoke_report_already_exists")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8760")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--rerun-analysis-id")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--idempotency-key")
    parser.add_argument("--startup-timeout-seconds", type=float, default=90)
    parser.add_argument("--analysis-timeout-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    if args.idempotency_key is not None and (
        not 1 <= len(args.idempotency_key) <= 128
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
            for character in args.idempotency_key
        )
    ):
        parser.error("--idempotency-key contains unsupported characters")
    if args.rerun_analysis_id is not None and args.idempotency_key is not None:
        parser.error("--idempotency-key is only valid with --image")

    image: Path | None = None
    rerun_analysis_id: UUID | None = None
    if args.image is not None:
        image = args.image.expanduser().resolve(strict=True)
        if image.is_symlink() or not image.is_file() or image.stat().st_size < 1:
            parser.error("--image must be a non-empty regular file")
        if image.stat().st_size > 20 * 1024 * 1024:
            parser.error("--image exceeds the bounded smoke input size")
    else:
        try:
            rerun_analysis_id = UUID(args.rerun_analysis_id)
        except (TypeError, ValueError):
            parser.error("--rerun-analysis-id must be a UUID")

    try:
        with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
            _wait_for_api(client, args.startup_timeout_seconds)
            if image is not None:
                with image.open("rb") as stream:
                    response = client.post(
                        "/api/v1/analyses",
                        headers=(
                            {"Idempotency-Key": args.idempotency_key}
                            if args.idempotency_key is not None
                            else None
                        ),
                        files={"image": ("private-smoke.jpg", stream, "image/jpeg")},
                        data={
                            "analysis_mode": "local_only",
                            "cloud_processing_consent": "false",
                            "authorization_acknowledged": "true",
                            "allow_cloud_assist": "false",
                        },
                    )
            else:
                response = client.post(
                    f"/api/v1/analyses/{rerun_analysis_id}/rerun"
                )
            response.raise_for_status()
            result = _wait_for_analysis(
                client,
                response.json()["id"],
                args.analysis_timeout_seconds,
            )
        report = build_safe_report(result)
        report["run_mode"] = (
            "retained_source_recomputation" if rerun_analysis_id else "new_upload"
        )
        _write_report(args.output.resolve(), report)
    except httpx.HTTPStatusError as exc:
        reason_code = f"phase6c_http_status_{exc.response.status_code}"
    except httpx.HTTPError:
        reason_code = "phase6c_http_transport_error"
    except (OSError, ValueError):
        reason_code = "phase6c_http_response_invalid"
    except RuntimeError as exc:
        reason_code = (
            str(exc) if str(exc).startswith("phase6c_") else "phase6c_http_smoke_error"
        )
    else:
        print(
            json.dumps(
                {
                    "event": "phase6c_http_smoke_completed",
                    "status": report["status"],
                    "pipeline_version": report["pipeline_version"],
                    "output": str(args.output),
                },
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "completed" else 1

    print(
        json.dumps(
            {
                "event": "phase6c_http_smoke_failed",
                "reason_code": reason_code,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
