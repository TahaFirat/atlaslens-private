"""One-image, no-retention operator CLI for Phase 3C1 blind queries."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from .client import NVIDIA_VISION_MODEL
from .errors import NvidiaVisionError
from .models import NvidiaGeolocationReasoning
from .prompt import NVIDIA_GEO_PROMPT_VERSION
from .provider import NvidiaCloudAuthorization, NvidiaStructuredResponse
from .reasoning import NvidiaGeolocationReasoner, build_nvidia_geolocation_reasoner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas nvidia-vision", add_help=True)
    subcommands = parser.add_subparsers(dest="command", required=True)
    blind = subcommands.add_parser("blind-query")
    blind.add_argument("image", type=Path)
    blind.add_argument(
        "--analysis-mode",
        choices=("local_only", "cloud_assisted"),
        default="local_only",
    )
    blind.add_argument("--cloud-consent", action="store_true")
    blind.add_argument("--execute", action="store_true")
    return parser


def _media_type(payload: bytes) -> str | None:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return None


def _safe_report(
    response: NvidiaStructuredResponse[NvidiaGeolocationReasoning],
) -> dict[str, Any]:
    result = response.value
    return {
        "status": "abstained" if result.decision == "abstain" else "completed",
        "provider": "nvidia",
        "model": response.model,
        "prompt_version": NVIDIA_GEO_PROMPT_VERSION,
        "schema_version": result.schema_version,
        "provenance": "nvidia_hosted_qwen_image_only_phase3c1",
        "network_attempts": response.network_attempts,
        "status_poll_attempts": response.status_poll_attempts,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "confidence_semantics": "uncalibrated_model_self_assessment",
        "coordinates_omitted": True,
        "retained": False,
        "abstention_reason": result.abstention_reason,
        "uncertainty_summary": result.uncertainty_summary,
        "observed_clues": [
            {
                "clue_id": item.clue_id,
                "category": item.category,
                "strength": item.strength,
                "observation_omitted": True,
            }
            for item in result.observed_clues
        ],
        "hypotheses": [
            {
                "rank": rank,
                "label": item.label,
                "country_code": item.country_code,
                "granularity": item.granularity,
                "uncertainty_radius_km": item.uncertainty_radius_km,
                "confidence": item.confidence,
                "confidence_semantics": item.confidence_semantics,
                "supporting_clue_ids": list(item.supporting_clue_ids),
                "limitations": list(item.limitations),
            }
            for rank, item in enumerate(result.hypotheses, start=1)
        ],
    }


async def _run_blind_query(image_path: Path) -> dict[str, Any]:
    key = os.environ.get("NVIDIA_API_KEY")
    reasoner: NvidiaGeolocationReasoner | None = None
    payload = b""
    network_counts = {"all": 0, "polls": 0}

    async def count_request(request: httpx.Request) -> None:
        network_counts["all"] += 1
        if request.method == "GET":
            network_counts["polls"] += 1

    try:
        async with httpx.AsyncClient(
            event_hooks={"request": [count_request]},
            follow_redirects=False,
            trust_env=False,
        ) as http_client:
            reasoner = build_nvidia_geolocation_reasoner(
                enabled=True,
                api_key=key,
                model=os.environ.get("NVIDIA_VISION_MODEL", NVIDIA_VISION_MODEL),
                total_timeout_seconds=120,
                max_transport_retries=0,
                http_client=http_client,
            )
            if not reasoner.status().credentials_configured:
                return {
                    "status": "error",
                    "code": "nvidia_credentials_missing",
                    "network_attempts": 0,
                    "status_poll_attempts": 0,
                }
            payload = await asyncio.to_thread(image_path.read_bytes)
            media_type = _media_type(payload)
            if media_type is None:
                return {
                    "status": "error",
                    "code": "nvidia_image_format_unsupported",
                    "network_attempts": 0,
                    "status_poll_attempts": 0,
                }
            response = await reasoner.analyze(
                payload,
                declared_media_type=media_type,
                authorization=NvidiaCloudAuthorization(
                    mode="cloud_assisted",
                    cloud_consent=True,
                ),
            )
            return _safe_report(response)
    except NvidiaVisionError as exc:
        return {
            "status": "error",
            "code": exc.code,
            "network_attempts": network_counts["all"],
            "status_poll_attempts": network_counts["polls"],
        }
    except (OSError, ValueError):
        return {
            "status": "error",
            "code": "nvidia_blind_query_input_invalid",
            "network_attempts": network_counts["all"],
            "status_poll_attempts": network_counts["polls"],
        }
    finally:
        del payload
        if reasoner is not None:
            await reasoner.close()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    if arguments.command != "blind-query":
        print(json.dumps({"status": "error", "code": "command_required"}), file=sys.stderr)
        return 2
    if (
        arguments.analysis_mode != "cloud_assisted"
        or not arguments.cloud_consent
        or not arguments.execute
    ):
        print(
            json.dumps(
                {
                    "status": "refused",
                    "code": "nvidia_explicit_cloud_authorization_required",
                    "network_attempts": 0,
                },
                sort_keys=True,
            )
        )
        return 2
    report = asyncio.run(_run_blind_query(arguments.image))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("status") in {"completed", "abstained"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
