from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from atlaslens_api.gazetteer import CoordinateFallbackResolver, GazetteerResolver
from atlaslens_api.providers.base import (
    GlobalGeolocationProvider,
    InvocationContext,
    OutcomeStatus,
)
from atlaslens_api.schemas import AnalysisMode
from atlaslens_api.storage import LocalImageHandle

_ALLOWED_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}
_MAX_ACCEPTANCE_FILE_BYTES = 50 * 1024 * 1024


class AcceptanceError(ValueError):
    pass


def _images(directory: Path) -> tuple[Path, ...]:
    expanded = directory.expanduser()
    if expanded.is_symlink():
        raise AcceptanceError("acceptance_directory_symlink")
    root = expanded.resolve(strict=True)
    if not root.is_dir():
        raise AcceptanceError("acceptance_directory_invalid")
    selected: list[Path] = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if candidate.suffix.casefold() in _ALLOWED_SUFFIXES:
            selected.append(candidate)
    if not selected:
        raise AcceptanceError("acceptance_images_missing")
    return tuple(selected)


def _inspect_image(path: Path) -> tuple[str, bool, bool]:
    if path.stat().st_size > _MAX_ACCEPTANCE_FILE_BYTES:
        raise AcceptanceError("image_too_large")
    try:
        with Image.open(path) as image:
            image.load()
            image_format = (image.format or "unknown").lower()
            exif = image.getexif()
            exif_present = bool(exif)
            gps_present = bool(exif.get_ifd(0x8825)) if 0x8825 in exif else False
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise AcceptanceError("invalid_image") from exc
    if image_format not in {"jpeg", "png", "webp"}:
        raise AcceptanceError("unsupported_image_format")
    return image_format, exif_present, gps_present


async def _run_one(
    provider: GlobalGeolocationProvider,
    image: Path,
    safe_id: str,
    gazetteer: GazetteerResolver,
) -> dict[str, object]:
    started = time.monotonic()
    try:
        image_format, exif_present, gps_present = await asyncio.to_thread(
            _inspect_image, image
        )
    except AcceptanceError as exc:
        return {
            "image_safe_id": safe_id,
            "provider_ran": False,
            "candidate_count": 0,
            "runtime_ms": 0,
            "result_status": "invalid_image",
            "reason_code": str(exc),
        }
    context = InvocationContext(
        analysis_id=uuid4(),
        request_id=f"acceptance-{safe_id}",
        mode=AnalysisMode.LOCAL_ONLY,
        cloud_consent=False,
        deadline=datetime.now(UTC) + timedelta(minutes=3),
        cancellation=asyncio.Event(),
    )
    try:
        outcome = await provider.predict(
            LocalImageHandle(key=f"acceptance.{safe_id}", path=image), context
        )
    except Exception:
        return {
            "image_safe_id": safe_id,
            "format": image_format,
            "exif_present": exif_present,
            "gps_exif_present": gps_present,
            "provider_ran": True,
            "candidate_count": 0,
            "runtime_ms": round((time.monotonic() - started) * 1000),
            "result_status": "provider_failed",
            "reason_code": "unhandled_provider_failure",
        }
    elapsed_ms = round((time.monotonic() - started) * 1000)
    if outcome.status != OutcomeStatus.SUCCEEDED or outcome.value is None:
        reason = outcome.failure.code if outcome.failure is not None else outcome.status.value
        return {
            "image_safe_id": safe_id,
            "format": image_format,
            "exif_present": exif_present,
            "gps_exif_present": gps_present,
            "provider_ran": outcome.status != OutcomeStatus.SKIPPED,
            "candidate_count": 0,
            "runtime_ms": elapsed_ms,
            "result_status": outcome.status.value,
            "reason_code": reason,
        }
    hypotheses = outcome.value.hypotheses
    primary_label: str | None = None
    if hypotheses:
        first = hypotheses[0]
        try:
            primary_label = gazetteer.resolve(first.latitude, first.longitude).label
        except Exception:
            primary_label = CoordinateFallbackResolver().resolve(
                first.latitude, first.longitude
            ).label
    return {
        "image_safe_id": safe_id,
        "format": image_format,
        "exif_present": exif_present,
        "gps_exif_present": gps_present,
        "provider_ran": True,
        "candidate_count": len(hypotheses),
        "primary_broad_label": primary_label,
        "runtime_ms": elapsed_ms,
        "result_status": "candidates_returned" if hypotheses else "abstained",
        "reason_code": None if hypotheses else "empty_model_output",
    }


def _safe_output_directory(output: Path, input_directory: Path) -> Path:
    expanded = output.expanduser()
    if expanded.is_symlink():
        raise AcceptanceError("acceptance_output_symlink")
    resolved = expanded.resolve()
    input_root = input_directory.expanduser().resolve(strict=True)
    if resolved == input_root or input_root in resolved.parents:
        raise AcceptanceError("acceptance_output_inside_input")
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise AcceptanceError("acceptance_output_invalid")
    return resolved


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# AtlasLens local acceptance report",
        "",
        "Candidate presence confirms operational inference, not location accuracy.",
        "",
        f"- Status: {report['status']}",
        f"- Provider: {report['provider_id']}",
        f"- Valid images: {summary['valid_images']}",
        f"- Non-EXIF images: {summary['non_exif_images']}",
        f"- Images with candidates: {summary['images_with_candidates']}",
        f"- Non-empty candidate rate: {summary['non_empty_candidate_rate']:.4f}",
        "",
        "| Safe ID | EXIF | Provider ran | Candidates | Broad label | Runtime ms | Status |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    records = report["images"]
    assert isinstance(records, Sequence)
    for raw in records:
        assert isinstance(raw, dict)
        lines.append(
            f"| {raw['image_safe_id']} | {raw.get('exif_present', '-')} | "
            f"{raw['provider_ran']} | {raw['candidate_count']} | "
            f"{raw.get('primary_broad_label') or '-'} | {raw['runtime_ms']} | "
            f"{raw['result_status']} |"
        )
    return "\n".join(lines) + "\n"


async def run_acceptance(
    directory: Path,
    output: Path,
    provider: GlobalGeolocationProvider,
    *,
    gazetteer: GazetteerResolver | None = None,
) -> dict[str, object]:
    paths = await asyncio.to_thread(_images, directory)
    records = [
        await _run_one(
            provider,
            path,
            f"image-{index:03d}",
            gazetteer or CoordinateFallbackResolver(),
        )
        for index, path in enumerate(paths, start=1)
    ]
    valid = [item for item in records if item["result_status"] != "invalid_image"]
    non_exif = [item for item in valid if item.get("exif_present") is False]
    with_candidates = [
        item
        for item in non_exif
        if isinstance(item["candidate_count"], int) and item["candidate_count"] > 0
    ]
    rate = len(with_candidates) / len(non_exif) if non_exif else 0.0
    runtimes = [
        value
        for item in valid
        if isinstance((value := item["runtime_ms"]), int)
    ]
    ordered_runtimes = sorted(runtimes)
    runtime_p95 = (
        ordered_runtimes[max(0, math.ceil(0.95 * len(ordered_runtimes)) - 1)]
        if ordered_runtimes
        else None
    )
    status = "passed" if len(non_exif) >= 5 and len(with_candidates) >= 4 else "failed"
    provider_status = provider.status()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "scope": "direct_provider_smoke_acceptance",
        "provider_id": provider.descriptor.id,
        "provider": {
            "operational_status": provider_status.status,
            "model_name": provider_status.model_name,
            "model_revision": provider_status.model_revision,
            "device": provider_status.device,
            "calibration_state": provider_status.calibration_state,
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "privacy": {
            "external_transfer": False,
            "images_modified": False,
            "images_added_to_training": False,
            "filenames_recorded": False,
        },
        "interpretation": "candidate_presence_is_not_location_accuracy",
        "product_gate_observations": {
            "fastapi_pipeline_checked": False,
            "frontend_checked": False,
            "offline_restart_checked": False,
        },
        "summary": {
            "discovered_images": len(paths),
            "valid_images": len(valid),
            "non_exif_images": len(non_exif),
            "images_with_candidates": len(with_candidates),
            "non_empty_candidate_rate": rate,
            "median_runtime_ms": (
                statistics.median(runtimes) if runtimes else None
            ),
            "p95_runtime_ms": runtime_p95,
        },
        "images": records,
    }
    if not math.isfinite(rate):
        raise AcceptanceError("invalid_acceptance_rate")
    destination = await asyncio.to_thread(_safe_output_directory, output, directory)
    await asyncio.to_thread(
        _write_atomic,
        destination / "acceptance.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    await asyncio.to_thread(
        _write_atomic, destination / "acceptance.md", _markdown(report)
    )
    return report
