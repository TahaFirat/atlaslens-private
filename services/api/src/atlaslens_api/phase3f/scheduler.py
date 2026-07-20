"""Atomic Phase 3F acquisition supervisor checkpoint schema v4."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast
from uuid import uuid4

SCHEDULER_CHECKPOINT_SCHEMA: Final = "atlaslens-phase3f-acquisition-scheduler-v4"
MAX_FAILURE_LEDGER_ROWS: Final = 4_096


class SchedulerCheckpointError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SchedulerCheckpointError(code)


def _read(path: Path, *, max_bytes: int) -> dict[str, object]:
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= max_bytes,
        "SCHEDULER_SOURCE_CHECKPOINT_INVALID",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchedulerCheckpointError("SCHEDULER_SOURCE_CHECKPOINT_INVALID") from exc
    _require(isinstance(value, dict), "SCHEDULER_SOURCE_CHECKPOINT_INVALID")
    return cast(dict[str, object], value)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _atomic(path: Path, value: object) -> str:
    payload = _canonical(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_id(city: str, bbox: object, depth: object) -> str:
    return hashlib.sha256(_canonical([city, bbox, depth])).hexdigest()


def _metadata_projection(
    document: dict[str, object], *, city_order: Sequence[str] | None
) -> dict[str, object]:
    cells_value = document.get("cells_by_city")
    rows_value = document.get("rows_by_city")
    _require(
        isinstance(cells_value, dict) and isinstance(rows_value, dict),
        "SCHEDULER_METADATA_INVALID",
    )
    cells_by_city = cast(dict[str, object], cells_value)
    rows_by_city = cast(dict[str, object], rows_value)
    raw_city_index = document.get("city_index")
    raw_box_index = document.get("box_index")
    _require(
        isinstance(raw_city_index, int)
        and not isinstance(raw_city_index, bool)
        and isinstance(raw_box_index, int)
        and not isinstance(raw_box_index, bool),
        "SCHEDULER_METADATA_INVALID",
    )
    city_index = cast(int, raw_city_index)
    box_index = cast(int, raw_box_index)
    cities = list(city_order) if city_order is not None else list(cells_by_city)
    _require(set(cities) == set(cells_by_city), "SCHEDULER_METADATA_INVALID")
    _require(0 <= city_index <= len(cities), "SCHEDULER_METADATA_INVALID")
    pending: list[dict[str, object]] = []
    completed: list[str] = []
    active: dict[str, object] | None = None
    for selected_city_index, city in enumerate(cities):
        raw_cells = cells_by_city[city]
        _require(isinstance(raw_cells, list), "SCHEDULER_METADATA_INVALID")
        for selected_box_index, raw_cell in enumerate(cast(list[object], raw_cells)):
            _require(isinstance(raw_cell, dict), "SCHEDULER_METADATA_INVALID")
            cell = cast(dict[str, object], raw_cell)
            bbox = cell.get("bbox")
            depth = cell.get("depth")
            identity = _cell_id(city, bbox, depth)
            is_completed = selected_city_index < city_index or (
                selected_city_index == city_index and selected_box_index < box_index
            )
            if is_completed:
                completed.append(identity)
                continue
            row = {
                "cell_id": identity,
                "city": city,
                "bbox": bbox,
                "depth": depth,
                "cursor": (
                    document.get("next_url")
                    if selected_city_index == city_index and selected_box_index == box_index
                    else None
                ),
            }
            pending.append(row)
            if active is None:
                active = row
    first_seen: list[str] = []
    coverage: dict[str, object] = {}
    for city, raw_rows in rows_by_city.items():
        _require(isinstance(raw_rows, list), "SCHEDULER_METADATA_INVALID")
        ids: list[str] = []
        for raw_row in cast(list[object], raw_rows):
            _require(isinstance(raw_row, dict), "SCHEDULER_METADATA_INVALID")
            identifier = cast(dict[str, object], raw_row).get("mapillary_image_id")
            _require(
                isinstance(identifier, str) and bool(identifier),
                "SCHEDULER_METADATA_INVALID",
            )
            safe_identifier = cast(str, identifier)
            ids.append(safe_identifier)
            first_seen.append(safe_identifier)
        coverage[city] = {"metadata_rows": len(ids), "first_seen_unique": len(set(ids))}
    _require(len(first_seen) == len(set(first_seen)), "SCHEDULER_FIRST_SEEN_DUPLICATE")
    return {
        "pending_cell_queue": pending,
        "completed_cells": completed,
        "active_cell": active,
        "first_seen_image_ids": first_seen,
        "coverage_summary": coverage,
        "metadata_row_count": len(first_seen),
    }


def classify_failure(code: str) -> tuple[str, bool]:
    normalized = code.upper()
    if normalized in {
        "MAPILLARY_TOKEN_REJECTED",
        "MAPILLARY_PERMISSION_DENIED",
        "MAPILLARY_MEDIA_AUTHORIZATION_REJECTED",
    }:
        return "TERMINAL_AUTH", True
    if normalized == "MAPILLARY_API_BAD_REQUEST":
        return "TERMINAL_REQUEST_CONTRACT", True
    if normalized in {
        "MAPILLARY_API_RATE_LIMIT_RETRY_EXHAUSTED",
        "MAPILLARY_MEDIA_RATE_LIMIT_RETRY_EXHAUSTED",
    }:
        return "PAUSED_RATE_LIMIT", False
    if normalized in {
        "MAPILLARY_API_SERVER_RETRY_EXHAUSTED",
        "MAPILLARY_API_TIMEOUT",
        "MAPILLARY_API_TRANSPORT_RETRY_EXHAUSTED",
    }:
        return "PAUSED_PROVIDER_RETRY_EXHAUSTED", False
    if normalized in {
        "MAPILLARY_PAGING_LOOP_DETECTED",
        "MAPILLARY_PAGING_URL_INVALID",
        "MAPILLARY_PAGING_QUERY_DUPLICATE",
        "MAPILLARY_API_PAGE_INVALID",
    }:
        return "CELL_FAILED_PAGING", False
    if normalized in {
        "MAPILLARY_REQUEST_CAP_REACHED",
        "MAPILLARY_PAGE_CAP_REACHED",
        "LOCAL_ACQUISITION_DEADLINE_REACHED",
        "MEDIA_CAP_EXCEEDED",
    }:
        return f"PAUSED_{normalized}", False
    return "PAUSED_TYPED_FAILURE", False


def sync_scheduler_checkpoint(
    destination: Path,
    *,
    run_id: str,
    metadata_checkpoint: Path,
    client_counters: Path,
    acquisition_checkpoint: Path | None,
    request_cap: int,
    media_byte_cap: int,
    max_wall_seconds: int,
    status: str,
    failure_code: str | None = None,
    city_order: Sequence[str] | None = None,
) -> dict[str, object]:
    metadata = _read(metadata_checkpoint, max_bytes=64 * 1024 * 1024)
    counters = _read(client_counters, max_bytes=16 * 1024)
    _require(
        metadata.get("run_id") == run_id and counters.get("run_id") == run_id,
        "SCHEDULER_RUN_ID_MISMATCH",
    )
    projection = _metadata_projection(metadata, city_order=city_order)
    prior: dict[str, object] = {}
    if destination.exists():
        prior = _read(destination, max_bytes=64 * 1024 * 1024)
        _require(
            prior.get("schema") == SCHEDULER_CHECKPOINT_SCHEMA
            and prior.get("run_id") == run_id,
            "SCHEDULER_CHECKPOINT_INCOMPATIBLE",
        )
    request_count = counters.get("request_count")
    page_count = counters.get("page_count")
    rejected_count = counters.get("rejected_item_count")
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (request_count, page_count, rejected_count)
        ),
        "SCHEDULER_COUNTERS_INVALID",
    )
    media_bytes = 0
    media_completed = 0
    acquisition_sha: str | None = None
    if acquisition_checkpoint is not None and acquisition_checkpoint.exists():
        acquisition = _read(acquisition_checkpoint, max_bytes=64 * 1024 * 1024)
        _require(acquisition.get("run_id") == run_id, "SCHEDULER_RUN_ID_MISMATCH")
        rows = acquisition.get("completed")
        _require(isinstance(rows, list), "SCHEDULER_ACQUISITION_INVALID")
        safe_rows = cast(list[object], rows)
        media_completed = len(safe_rows)
        media_bytes = sum(
            cast(int, cast(dict[str, object], row).get("admitted_media_bytes", 0))
            for row in safe_rows
            if isinstance(row, dict)
        )
        acquisition_sha = _sha256(acquisition_checkpoint)
    metadata_sha = _sha256(metadata_checkpoint)
    transition_ordinal = cast(int, prior.get("transition_ordinal", 0))
    prior_transition = prior.get("last_successful_atomic_transition")
    prior_metadata_sha = (
        cast(dict[str, object], prior_transition).get("metadata_checkpoint_sha256")
        if isinstance(prior_transition, dict)
        else None
    )
    prior_acquisition_sha = (
        cast(dict[str, object], prior_transition).get("acquisition_checkpoint_sha256")
        if isinstance(prior_transition, dict)
        else None
    )
    if metadata_sha != prior_metadata_sha or acquisition_sha != prior_acquisition_sha:
        transition_ordinal += 1
    ledger = list(cast(list[object], prior.get("failure_ledger", [])))
    partial_cells = list(cast(list[object], prior.get("partial_cells", [])))
    failed_cells = list(cast(list[object], prior.get("failed_cells", [])))
    terminal = False
    if failure_code is not None:
        classified, terminal = classify_failure(failure_code)
        active = projection["active_cell"]
        active_cell_id = (
            cast(dict[str, object], active).get("cell_id")
            if isinstance(active, dict)
            else None
        )
        attempt_count = 1 + sum(
            isinstance(row, dict)
            and row.get("cell_id") == active_cell_id
            and row.get("code") == failure_code.upper()
            for row in ledger
        )
        failure = {
            "ordinal": len(ledger) + 1,
            "code": failure_code.upper(),
            "classification": classified,
            "cell_id": active_cell_id,
            "attempt_count_for_cell": attempt_count,
            "request_count": request_count,
            "retry_same_cursor_allowed": False,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        last_failure = ledger[-1] if ledger else None
        duplicate_failure = bool(
            isinstance(last_failure, dict)
            and all(
                last_failure.get(key) == failure.get(key)
                for key in ("code", "classification", "cell_id", "request_count")
            )
        )
        if not duplicate_failure:
            ledger.append(failure)
            if classified == "CELL_FAILED_PAGING" and active is not None:
                failed_cells.append(active)
            elif classified.startswith("PAUSED_PROVIDER") and active is not None:
                partial_cells.append(active)
        status = classified
    _require(len(ledger) <= MAX_FAILURE_LEDGER_ROWS, "SCHEDULER_FAILURE_LEDGER_CAP_REACHED")
    document = {
        "schema": SCHEDULER_CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "status": status,
        "terminal": terminal,
        "pending_cell_queue": projection["pending_cell_queue"],
        "completed_cells": projection["completed_cells"],
        "partial_cells": partial_cells[-MAX_FAILURE_LEDGER_ROWS:],
        "failed_cells": failed_cells[-MAX_FAILURE_LEDGER_ROWS:],
        "active_cell": projection["active_cell"],
        "first_seen_image_ids": projection["first_seen_image_ids"],
        "quotas": {
            "per_city_metadata_rows": 600,
            "global_metadata_rows": 9_600,
            "admitted_metadata_rows": projection["metadata_row_count"],
        },
        "failure_ledger": ledger,
        "budgets": {
            "request_cap": request_cap,
            "request_count": request_count,
            "page_count": page_count,
            "rejected_item_count": rejected_count,
            "media_byte_cap": media_byte_cap,
            "media_bytes": media_bytes,
            "media_completed": media_completed,
            "max_wall_seconds": max_wall_seconds,
        },
        "transition_ordinal": transition_ordinal,
        "last_successful_atomic_transition": {
            "ordinal": transition_ordinal,
            "metadata_checkpoint_sha256": metadata_sha,
            "acquisition_checkpoint_sha256": acquisition_sha,
            "metadata_row_count": projection["metadata_row_count"],
            "media_completed": media_completed,
            "recorded_at": datetime.now(UTC).isoformat(),
        },
        "coverage_summary": projection["coverage_summary"],
        "migrated_from_schema": metadata.get("schema"),
        "secrets_included": False,
        "signed_urls_included": False,
    }
    _atomic(destination, document)
    return document


__all__ = [
    "SCHEDULER_CHECKPOINT_SCHEMA",
    "SchedulerCheckpointError",
    "classify_failure",
    "sync_scheduler_checkpoint",
]
