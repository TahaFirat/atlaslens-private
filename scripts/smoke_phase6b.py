#!/usr/bin/env python3
"""Run one real, local Phase 6B pipeline smoke analysis without downloading data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from atlaslens_api.config import Settings
from atlaslens_api.main import create_app
from fastapi.testclient import TestClient


def _wait(client: TestClient, analysis_id: str, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/analyses/{analysis_id}")
        response.raise_for_status()
        body = response.json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.1)
    raise TimeoutError("Phase 6B smoke analysis did not reach a terminal state")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--cloud-consent", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=360)
    args = parser.parse_args()
    if args.use_openai and not args.cloud_consent:
        parser.error("--use-openai requires the explicit --cloud-consent flag")
    image = args.image.resolve()
    if image.is_symlink() or not image.is_file() or image.stat().st_size <= 0:
        parser.error("--image must be a non-empty user-provided or reviewed local image")
    if image.stat().st_size > 20 * 1024 * 1024:
        parser.error("--image exceeds the 20 MiB smoke-test bound")

    root = args.project_root.resolve()
    runtime = root / ".local" / "phase6b-smoke"
    settings = Settings(
        database_url=f"sqlite:///{(runtime / 'atlaslens.sqlite3').as_posix()}",
        temp_storage_dir=runtime / "tmp",
        phase6b_enabled=True,
        global_model_enabled=True,
        segmentation_enabled=True,
        segmentation_model_dir=root / ".local" / "models" / "atlaslens-segformer-b2-v4",
        openai_geo_enabled=args.use_openai,
    )
    app = create_app(settings)
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(image.suffix.casefold(), "application/octet-stream")
    with TestClient(app, raise_server_exceptions=False) as client:
        capabilities = client.get("/api/v1/capabilities").json()
        if args.use_openai:
            cloud = capabilities["providers"]["openai_geo_review"]
            if not cloud["key_configured"] or not cloud["budget_available"]:
                raise RuntimeError(
                    "OpenAI smoke test refused: backend key or configured local "
                    "budget is unavailable"
                )
        with image.open("rb") as source:
            response = client.post(
                "/api/v1/analyses",
                files={"image": (f"reviewed-smoke{image.suffix.casefold()}", source, media_type)},
                data={
                    "analysis_mode": "cloud_assisted" if args.use_openai else "local_only",
                    "cloud_processing_consent": str(args.cloud_consent).lower(),
                    "authorization_acknowledged": "true",
                    "allow_cloud_assist": str(args.use_openai).lower(),
                },
            )
        response.raise_for_status()
        result = _wait(client, response.json()["id"], args.timeout_seconds)

    model_predictions = result.get("model_predictions") or {}
    fusion = result.get("fusion") or {}
    ocr = result.get("ocr") or {}
    cloud = result.get("cloud_assist") or {}
    scene = result.get("scene_analysis") or {}
    candidates = result.get("candidates") or []
    report = {
        "status": result["status"],
        "candidate_count": len(candidates),
        "reverse_geocoding": {
            "resolved_candidate_count": sum(
                bool(candidate.get("reverse_geocode")) for candidate in candidates
            )
        },
        "segmentation": {
            "status": scene.get("status", "unavailable"),
            "semantic_label_names_available": scene.get(
                "semantic_label_names_available", False
            ),
            "dominant_class_names": [
                item["class_name"] for item in scene.get("dominant_classes", [])[:10]
            ],
        },
        "models": {
            name: {
                "status": prediction["status"],
                "model_id": prediction["model_id"],
                "reason_code": prediction.get("reason_code"),
            }
            for name, prediction in model_predictions.items()
        },
        "fusion": {
            "version": fusion.get("version"),
            "cluster_count": len(fusion.get("candidate_clusters", [])),
            "agreement": fusion.get("agreement_summary"),
        },
        "ocr": {
            "provider": ocr.get("provider"),
            "status": ocr.get("status"),
            "fallback_used": ocr.get("fallback_used"),
            "detection_count": len(ocr.get("detections", [])),
        },
        "cloud_assist": {
            "allowed": cloud.get("allowed", False),
            "triggered": cloud.get("triggered", False),
            "status": cloud.get("status", "not_requested"),
            "model": cloud.get("model"),
            "cache_hit": cloud.get("cache_hit", False),
            "estimated_cost_usd": cloud.get("estimated_cost_usd"),
        },
        "accuracy_claim": None,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
