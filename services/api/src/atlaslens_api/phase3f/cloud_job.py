"""Executable, bounded Phase 3F Mapillary/MegaLoc cloud worker.

The module keeps evaluator truth and raw imagery inside the ephemeral work root.
Only aggregate receipts and derived index/descriptor artifacts are published.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, cast
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import ValidationError

from atlaslens_api.mapillary_demo.client import (
    MapillaryClient,
    PaginationProgress,
    RemoteImage,
    validated_next_url,
)
from atlaslens_api.mapillary_demo.errors import (
    MapillaryApiError,
    MapillaryLimitError,
    MapillarySafetyError,
    MapillaryTokenError,
)
from atlaslens_api.mapillary_demo.models import BoundingBox, ClientLimits, ImageMetadata
from atlaslens_api.phase3f.acquisition import (
    MAX_IMAGES,
    MAX_MEDIA_BYTES,
    MAX_REQUESTS,
    AcquisitionGuard,
    AcquisitionRefused,
    PrivacyReview,
    ProvenanceSidecar,
)
from atlaslens_api.phase3f.benchmark import (
    CalibrationObservation,
    RetrievalObservation,
    RetrievalSignals,
    evaluate_holdout_once,
    fit_abstention_threshold,
)
from atlaslens_api.phase3f.coverage import (
    CANDIDATE_CITY_REGIONS,
    CityCoverageRecord,
    CoverageInsufficient,
    CoverageSelectionLock,
    select_city_scope,
)
from atlaslens_api.phase3f.pipeline import (
    MODEL_REVISION,
    MODEL_SHA256,
    PROVIDER_REVISION,
    DescriptorPublication,
    Phase3FPipeline,
)
from atlaslens_api.phase3f.splits import (
    MIN_CALIBRATION_PER_CITY,
    MIN_HOLDOUT_PER_CITY,
    MIN_OOD_PER_CITY,
    MIN_REFERENCE_PER_CITY,
    NEAR_DUPLICATE_HAMMING,
    SPATIAL_EXCLUSION_METERS,
    SplitAsset,
    audit_leakage,
    seal_split,
)

MODEL_SIZE_BYTES: Final = 914_577_436
MEGALOC_CANONICAL_SOURCE_SHA256: Final = (
    "3cbf1d20515b1da423998a8edab787031eaa7bb273c5a86a5c41c4f6d84e2a6d"
)
MEGALOC_LICENSE_CANONICAL_SHA256: Final = (
    "0a906f9a65db6f645483f6cbf56b01e20615b9b943df3f70112f3d0fe0521e2a"
)
MEGALOC_WINDOWS_SOURCE_SHA256: Final = (
    "c0848dfb287ba15b519d7b54415db824e16ec2f2b5a6899507b0476cf3379767"
)
MEGALOC_WINDOWS_LICENSE_SHA256: Final = (
    "40c6c4894aecc5b676f0fb93697a6c1f82b08df71b25e485b662779b2c899667"
)
SOURCE_POLICY_SHA256: Final = "72d51363f2b63de368d34d4d7bb2fc1145f93dc0469e7026100976732dfde209"
MAX_METADATA_ITEMS_PER_CITY: Final = 600
MAX_METADATA_ITEMS_TOTAL: Final = 9_600
MAX_WALL_SECONDS: Final = 5 * 60 * 60 + 45 * 60
ACQUISITION_CHECKPOINT_SCHEMA: Final = "atlaslens-phase3f-acquisition-checkpoint-v1"
CLIENT_COUNTER_CHECKPOINT_SCHEMA: Final = "atlaslens-phase3f-client-counters-v1"
LEGACY_METADATA_PAGE_CHECKPOINT_SCHEMA: Final = "atlaslens-phase3f-metadata-pages-v1"
LEGACY_V2_METADATA_PAGE_CHECKPOINT_SCHEMA: Final = "atlaslens-phase3f-metadata-pages-v2"
METADATA_PAGE_CHECKPOINT_SCHEMA: Final = "atlaslens-phase3f-metadata-pages-v3"
MAX_ADAPTIVE_PARTITION_DEPTH: Final = 4
MIN_ADAPTIVE_BBOX_AREA: Final = 1e-6
MAX_ADAPTIVE_CELLS_PER_CITY: Final = 4_096
SPLIT_RESERVE_PER_BUCKET: Final = 10
SPLIT_SOLVER_BEAM_WIDTH: Final = 5_000
SPLIT_SOLVER_MAX_ORDERINGS: Final = 8
SUPPLEMENTAL_REQUEST_CAP: Final = 512
SUPPLEMENTAL_METADATA_CAP: Final = 600
SUPPLEMENTAL_WALL_SECONDS: Final = 30 * 60
MEDIA_TASK_LEDGER_SCHEMA: Final = "atlaslens-phase3f-media-task-ledger-v2"
MEDIA_RESOLVER_CONTRACT_VERSION: Final = "direct-image-v1"
MEDIA_CIRCUIT_FAILURE_THRESHOLD: Final = 5
MEDIA_PROVIDER_COOLDOWN_SECONDS: Final = 5 * 60
MEDIA_PROVIDER_COOLDOWN_MAX_SECONDS: Final = 30 * 60
MEDIA_RATE_LIMIT_COOLDOWN_SECONDS: Final = 60
MEDIA_RATE_LIMIT_COOLDOWN_MAX_SECONDS: Final = 15 * 60
MEDIA_RATE_LIMIT_TASK_ATTEMPT_CAP: Final = 2
MEDIA_SIGNED_URL_REFRESH_CAP: Final = 1
_LEGACY_MEDIA_RESOLVER_RETRY_CODES: Final = frozenset(
    {
        "mapillary_api_server_retry_exhausted",
        "mapillary_api_timeout",
        "mapillary_api_transport_retry_exhausted",
        "mapillary_api_transport_failed",
        "url_resolution_failed",
    }
)

Role = Literal["reference", "calibration", "sealed_holdout", "ood_holdout"]


class Phase3FCloudJobError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _MetadataCityQuotaReached(RuntimeError):
    """Internal control flow after an atomic per-city quota checkpoint."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Phase3FCloudJobError(code)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path, *, expected_size: int | None = None) -> str:
    _require(path.is_file() and not path.is_symlink(), "ARTIFACT_MISSING")
    if expected_size is not None:
        _require(path.stat().st_size == expected_size, "ARTIFACT_SIZE_MISMATCH")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> str:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return _sha256_bytes(payload)


def _atomic_private_json(path: Path, value: object) -> str:
    payload = _canonical_bytes(value)
    return _atomic_private_bytes(path, payload)


def _atomic_private_bytes(path: Path, payload: bytes) -> str:
    digest = _sha256_bytes(payload)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    _require(
        not path.is_symlink() and not path.parent.is_symlink(),
        "PRIVATE_CHECKPOINT_PATH_INVALID",
    )
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.part")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except OSError as exc:
        raise Phase3FCloudJobError("PRIVATE_CHECKPOINT_WRITE_FAILED") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return digest


@dataclass(frozen=True, slots=True)
class CityArea:
    city: str
    longitude: float
    latitude: float
    half_span: float

    @property
    def boxes(self) -> tuple[BoundingBox, ...]:
        west = self.longitude - self.half_span
        east = self.longitude + self.half_span
        south = self.latitude - self.half_span
        north = self.latitude + self.half_span
        coarse = (
            BoundingBox(west=west, south=south, east=self.longitude, north=self.latitude),
            BoundingBox(west=self.longitude, south=south, east=east, north=self.latitude),
            BoundingBox(west=west, south=self.latitude, east=self.longitude, north=north),
            BoundingBox(west=self.longitude, south=self.latitude, east=east, north=north),
        )
        return tuple(cell for box in coarse for cell in quarter_bbox(box))


def quarter_bbox(box: BoundingBox) -> tuple[BoundingBox, ...]:
    """Split one bbox in stable southwest, southeast, northwest, northeast order."""

    middle_longitude = (box.west + box.east) / 2
    middle_latitude = (box.south + box.north) / 2
    return (
        BoundingBox(
            west=box.west,
            south=box.south,
            east=middle_longitude,
            north=middle_latitude,
        ),
        BoundingBox(
            west=middle_longitude,
            south=box.south,
            east=box.east,
            north=middle_latitude,
        ),
        BoundingBox(
            west=box.west,
            south=middle_latitude,
            east=middle_longitude,
            north=box.north,
        ),
        BoundingBox(
            west=middle_longitude,
            south=middle_latitude,
            east=box.east,
            north=box.north,
        ),
    )


def load_city_areas(path: Path) -> tuple[CityArea, ...]:
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024 * 1024,
        "CITY_AOI_CONFIG_MISSING",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3FCloudJobError("CITY_AOI_CONFIG_INVALID") from exc
    _require(isinstance(value, dict), "CITY_AOI_CONFIG_INVALID")
    document = cast(dict[str, object], value)
    _require(
        document.get("schema_version") == "atlaslens-phase3f-city-aoi-v1",
        "CITY_AOI_CONFIG_INVALID",
    )
    half_span = document.get("tile_half_span_degrees")
    rows = document.get("cities")
    _require(
        isinstance(half_span, int | float)
        and not isinstance(half_span, bool)
        and math.isfinite(float(half_span))
        and 0.0 < float(half_span) <= 0.05,
        "CITY_AOI_CONFIG_INVALID",
    )
    _require(isinstance(rows, list), "CITY_AOI_CONFIG_INVALID")
    areas: list[CityArea] = []
    for row_value in cast(list[object], rows):
        _require(isinstance(row_value, dict), "CITY_AOI_CONFIG_INVALID")
        row = cast(dict[str, object], row_value)
        city = row.get("city")
        longitude = row.get("longitude")
        latitude = row.get("latitude")
        _require(
            isinstance(city, str)
            and city in CANDIDATE_CITY_REGIONS
            and isinstance(longitude, int | float)
            and not isinstance(longitude, bool)
            and isinstance(latitude, int | float)
            and not isinstance(latitude, bool)
            and math.isfinite(float(longitude))
            and math.isfinite(float(latitude))
            and -180.0 <= float(longitude) <= 180.0
            and -90.0 <= float(latitude) <= 90.0,
            "CITY_AOI_CONFIG_INVALID",
        )
        areas.append(
            CityArea(
                cast(str, city),
                float(cast(int | float, longitude)),
                float(cast(int | float, latitude)),
                float(cast(int | float, half_span)),
            )
        )
    _require(
        len(areas) == len(CANDIDATE_CITY_REGIONS)
        and {area.city for area in areas} == set(CANDIDATE_CITY_REGIONS),
        "CITY_AOI_SCOPE_INVALID",
    )
    return tuple(areas)


@dataclass(frozen=True, slots=True)
class MetadataAsset:
    city: str
    remote: RemoteImage

    @property
    def image_id(self) -> str:
        return self.remote.metadata.mapillary_image_id

    @property
    def creator_id(self) -> str | None:
        return self.remote.metadata.creator_id

    @property
    def sequence_id(self) -> str | None:
        return self.remote.metadata.sequence_id


@dataclass(frozen=True, slots=True)
class PlannedAsset:
    metadata: MetadataAsset
    role: Role


@dataclass(frozen=True, slots=True)
class MetadataSplitPlan:
    primary: tuple[PlannedAsset, ...]
    reserves: tuple[PlannedAsset, ...]
    readiness: Mapping[str, object]

    @property
    def ready(self) -> bool:
        return bool(self.readiness.get("ready"))

    @property
    def download_assets(self) -> tuple[PlannedAsset, ...]:
        return self.primary + self.reserves


@dataclass(frozen=True, slots=True)
class MediaSplitPlan:
    assets: tuple[SplitAsset, ...]
    readiness: Mapping[str, object]

    @property
    def ready(self) -> bool:
        return bool(self.readiness.get("ready"))


@dataclass(frozen=True, slots=True)
class MetadataAudit:
    selection: CoverageSelectionLock
    assets_by_city: Mapping[str, tuple[MetadataAsset, ...]]
    request_count: int
    rejected_item_count: int

    def aggregate_document(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-phase3f-metadata-audit-v1",
            "selection": self.selection.document(),
            "request_count": self.request_count,
            "rejected_item_count": self.rejected_item_count,
            "city_counts": {
                city: len(self.assets_by_city[city]) for city in sorted(self.assets_by_city)
            },
            "raw_image_downloaded": False,
            "official_api_only": True,
        }


@dataclass(slots=True)
class _PlannedCell:
    bbox: BoundingBox
    depth: int


@dataclass(slots=True)
class _MetadataPageCheckpoint:
    run_id: str
    areas_sha256: str
    base_cell_plan_sha256: str
    cell_plan_sha256: str
    cells_by_city: dict[str, list[_PlannedCell]]
    subdivision_count: int
    city_index: int
    box_index: int
    next_url: str | None
    visited_page_sha256: tuple[str, ...]
    rows_by_city: dict[str, list[RemoteImage]]


def _areas_sha256(areas: Sequence[CityArea]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            [
                {
                    "city": area.city,
                    "longitude": area.longitude,
                    "latitude": area.latitude,
                    "half_span": area.half_span,
                }
                for area in areas
            ]
        )
    )


def _cell_plan_sha256(areas: Sequence[CityArea]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            [
                {
                    "city": area.city,
                    "ordered_cells": [box.as_query_value() for box in area.boxes],
                }
                for area in areas
            ]
        )
    )


def _base_cells_by_city(areas: Sequence[CityArea]) -> dict[str, list[_PlannedCell]]:
    return {area.city: [_PlannedCell(box, 0) for box in area.boxes] for area in areas}


def _adaptive_cell_plan_sha256(
    areas: Sequence[CityArea],
    cells_by_city: Mapping[str, Sequence[_PlannedCell]],
) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            [
                {
                    "city": area.city,
                    "ordered_cells": [
                        {
                            "bbox": cell.bbox.as_query_value(),
                            "depth": cell.depth,
                        }
                        for cell in cells_by_city[area.city]
                    ],
                }
                for area in areas
            ]
        )
    )


def _adaptive_cell_plan_is_valid(
    areas: Sequence[CityArea],
    cells_by_city: Mapping[str, Sequence[_PlannedCell]],
) -> bool:
    def consume(
        expected: BoundingBox,
        expected_depth: int,
        leaves: Sequence[_PlannedCell],
        position: int,
    ) -> int | None:
        if position >= len(leaves):
            return None
        leaf = leaves[position]
        if leaf.depth == expected_depth and leaf.bbox.as_query_value() == expected.as_query_value():
            return position + 1
        if leaf.depth <= expected_depth or expected_depth >= MAX_ADAPTIVE_PARTITION_DEPTH:
            return None
        next_position = position
        for child in quarter_bbox(expected):
            consumed_child = consume(
                child,
                expected_depth + 1,
                leaves,
                next_position,
            )
            if consumed_child is None:
                return None
            next_position = consumed_child
        return next_position

    if set(cells_by_city) != {area.city for area in areas}:
        return False
    for area in areas:
        leaves = cells_by_city[area.city]
        position = 0
        for base_cell in area.boxes:
            consumed = consume(base_cell, 0, leaves, position)
            if consumed is None:
                return False
            position = consumed
        if position != len(leaves):
            return False
    return True


def _rows_sha256(rows_by_city: Mapping[str, Sequence[RemoteImage]]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                city: [row.metadata.model_dump(mode="json") for row in rows]
                for city, rows in sorted(rows_by_city.items())
            }
        )
    )


def _new_metadata_page_checkpoint(
    areas: Sequence[CityArea], run_id: str
) -> _MetadataPageCheckpoint:
    cells_by_city = _base_cells_by_city(areas)
    base_plan_sha256 = _cell_plan_sha256(areas)
    return _MetadataPageCheckpoint(
        run_id=run_id,
        areas_sha256=_areas_sha256(areas),
        base_cell_plan_sha256=base_plan_sha256,
        cell_plan_sha256=_adaptive_cell_plan_sha256(areas, cells_by_city),
        cells_by_city=cells_by_city,
        subdivision_count=0,
        city_index=0,
        box_index=0,
        next_url=None,
        visited_page_sha256=(),
        rows_by_city={area.city: [] for area in areas},
    )


def _write_metadata_page_checkpoint(
    path: Path,
    state: _MetadataPageCheckpoint,
) -> None:
    _atomic_private_json(
        path,
        {
            "schema": METADATA_PAGE_CHECKPOINT_SCHEMA,
            "run_id": state.run_id,
            "areas_sha256": state.areas_sha256,
            "base_cell_plan_sha256": state.base_cell_plan_sha256,
            "cell_plan_sha256": state.cell_plan_sha256,
            "cells_by_city": {
                city: [
                    {
                        "bbox": [
                            cell.bbox.west,
                            cell.bbox.south,
                            cell.bbox.east,
                            cell.bbox.north,
                        ],
                        "depth": cell.depth,
                    }
                    for cell in cells
                ]
                for city, cells in sorted(state.cells_by_city.items())
            },
            "subdivision_count": state.subdivision_count,
            "city_index": state.city_index,
            "box_index": state.box_index,
            "next_url": state.next_url,
            "visited_page_sha256": list(state.visited_page_sha256),
            "rows_by_city": {
                city: [row.metadata.model_dump(mode="json") for row in rows]
                for city, rows in sorted(state.rows_by_city.items())
            },
            "secrets_included": False,
            "signed_urls_included": False,
        },
    )


def _load_metadata_page_checkpoint(
    path: Path,
    *,
    areas: Sequence[CityArea],
    run_id: str,
    persist_migration: bool = True,
) -> _MetadataPageCheckpoint:
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024 * 1024,
        "METADATA_PAGE_CHECKPOINT_INVALID",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3FCloudJobError("METADATA_PAGE_CHECKPOINT_INVALID") from exc
    _require(isinstance(value, dict), "METADATA_PAGE_CHECKPOINT_INVALID")
    document = cast(dict[str, object], value)
    common_keys = {
        "schema",
        "run_id",
        "areas_sha256",
        "city_index",
        "box_index",
        "next_url",
        "visited_page_sha256",
        "rows_by_city",
        "secrets_included",
        "signed_urls_included",
    }
    schema = document.get("schema")
    is_v1 = schema == LEGACY_METADATA_PAGE_CHECKPOINT_SCHEMA
    is_v2 = schema == LEGACY_V2_METADATA_PAGE_CHECKPOINT_SCHEMA
    v3_keys = {
        "base_cell_plan_sha256",
        "cell_plan_sha256",
        "cells_by_city",
        "subdivision_count",
    }
    expected_keys = common_keys if is_v1 else common_keys | {"cell_plan_sha256"}
    if schema == METADATA_PAGE_CHECKPOINT_SCHEMA:
        expected_keys = common_keys | v3_keys
    _require(
        set(document) == expected_keys,
        "METADATA_PAGE_CHECKPOINT_INVALID",
    )
    _require(
        schema
        in {
            LEGACY_METADATA_PAGE_CHECKPOINT_SCHEMA,
            LEGACY_V2_METADATA_PAGE_CHECKPOINT_SCHEMA,
            METADATA_PAGE_CHECKPOINT_SCHEMA,
        }
        and document.get("run_id") == run_id
        and document.get("areas_sha256") == _areas_sha256(areas)
        and (
            is_v1
            or document.get("cell_plan_sha256") == _cell_plan_sha256(areas)
            or schema == METADATA_PAGE_CHECKPOINT_SCHEMA
        )
        and document.get("secrets_included") is False
        and document.get("signed_urls_included") is False,
        "METADATA_PAGE_CHECKPOINT_INVALID",
    )
    city_index = document.get("city_index")
    box_index = document.get("box_index")
    next_url = document.get("next_url")
    visited = document.get("visited_page_sha256")
    raw_rows = document.get("rows_by_city")
    base_cells = _base_cells_by_city(areas)
    cells_by_city = base_cells
    subdivision_count = 0
    if schema == METADATA_PAGE_CHECKPOINT_SCHEMA:
        raw_cells = document.get("cells_by_city")
        raw_subdivision_count = document.get("subdivision_count")
        _require(
            document.get("base_cell_plan_sha256") == _cell_plan_sha256(areas)
            and isinstance(raw_cells, dict)
            and isinstance(raw_subdivision_count, int)
            and not isinstance(raw_subdivision_count, bool)
            and raw_subdivision_count >= 0,
            "METADATA_PAGE_CHECKPOINT_INVALID",
        )
        typed_raw_cells = cast(dict[str, object], raw_cells)
        _require(
            set(typed_raw_cells) == {area.city for area in areas},
            "METADATA_PAGE_CHECKPOINT_INVALID",
        )
        parsed_cells: dict[str, list[_PlannedCell]] = {}
        try:
            for area in areas:
                raw_city_cells = typed_raw_cells.get(area.city)
                _require(isinstance(raw_city_cells, list), "METADATA_PAGE_CHECKPOINT_INVALID")
                city_cells: list[_PlannedCell] = []
                for raw_cell in cast(list[object], raw_city_cells):
                    _require(
                        isinstance(raw_cell, dict) and set(raw_cell) == {"bbox", "depth"},
                        "METADATA_PAGE_CHECKPOINT_INVALID",
                    )
                    typed_raw_cell = cast(dict[str, object], raw_cell)
                    raw_bbox = typed_raw_cell.get("bbox")
                    depth = typed_raw_cell.get("depth")
                    _require(
                        isinstance(raw_bbox, list)
                        and len(raw_bbox) == 4
                        and isinstance(depth, int)
                        and not isinstance(depth, bool)
                        and 0 <= depth <= MAX_ADAPTIVE_PARTITION_DEPTH,
                        "METADATA_PAGE_CHECKPOINT_INVALID",
                    )
                    values = cast(list[object], raw_bbox)
                    _require(
                        all(
                            isinstance(value, int | float)
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                            for value in values
                        ),
                        "METADATA_PAGE_CHECKPOINT_INVALID",
                    )
                    numeric_values = cast(list[int | float], values)
                    bbox = BoundingBox(
                        west=float(numeric_values[0]),
                        south=float(numeric_values[1]),
                        east=float(numeric_values[2]),
                        north=float(numeric_values[3]),
                    )
                    city_cells.append(_PlannedCell(bbox, cast(int, depth)))
                _require(
                    0 < len(city_cells) <= MAX_ADAPTIVE_CELLS_PER_CITY
                    and len({cell.bbox.as_query_value() for cell in city_cells}) == len(city_cells),
                    "METADATA_PAGE_CHECKPOINT_INVALID",
                )
                full_cells = area.boxes
                west = min(cell.west for cell in full_cells)
                south = min(cell.south for cell in full_cells)
                east = max(cell.east for cell in full_cells)
                north = max(cell.north for cell in full_cells)
                _require(
                    all(
                        cell.bbox.west >= west
                        and cell.bbox.south >= south
                        and cell.bbox.east <= east
                        and cell.bbox.north <= north
                        and cell.bbox.area_square_degrees >= MIN_ADAPTIVE_BBOX_AREA
                        for cell in city_cells
                    ),
                    "METADATA_PAGE_CHECKPOINT_INVALID",
                )
                parsed_cells[area.city] = city_cells
        except (TypeError, ValueError, ValidationError) as exc:
            raise Phase3FCloudJobError("METADATA_PAGE_CHECKPOINT_INVALID") from exc
        cells_by_city = parsed_cells
        subdivision_count = cast(int, raw_subdivision_count)
        added_cells = sum(len(cells) for cells in cells_by_city.values()) - sum(
            len(cells) for cells in base_cells.values()
        )
        _require(
            added_cells >= 0
            and added_cells % 3 == 0
            and subdivision_count == added_cells // 3
            and _adaptive_cell_plan_is_valid(areas, cells_by_city)
            and document.get("cell_plan_sha256")
            == _adaptive_cell_plan_sha256(areas, cells_by_city),
            "METADATA_PAGE_CHECKPOINT_INVALID",
        )
    typed_city_index = city_index if isinstance(city_index, int) else -1
    maximum_box_index = (
        len(cells_by_city[areas[typed_city_index].city])
        if 0 <= typed_city_index < len(areas)
        else 0
    )
    _require(
        isinstance(city_index, int)
        and not isinstance(city_index, bool)
        and 0 <= city_index <= len(areas)
        and isinstance(box_index, int)
        and not isinstance(box_index, bool)
        and 0 <= box_index <= maximum_box_index
        and (next_url is None or isinstance(next_url, str))
        and isinstance(visited, list)
        and isinstance(raw_rows, dict),
        "METADATA_PAGE_CHECKPOINT_INVALID",
    )
    if isinstance(next_url, str):
        try:
            canonical_next_url = validated_next_url(next_url)
        except RuntimeError as exc:
            raise Phase3FCloudJobError("METADATA_PAGE_CHECKPOINT_INVALID") from exc
        _require(
            canonical_next_url == next_url and cast(int, city_index) < len(areas),
            "METADATA_PAGE_CHECKPOINT_INVALID",
        )
    visited_rows = cast(list[object], visited)
    _require(
        all(
            isinstance(item, str)
            and len(item) == 64
            and all(character in "0123456789abcdef" for character in item)
            for item in visited_rows
        )
        and len(set(cast(list[str], visited_rows))) == len(visited_rows),
        "METADATA_PAGE_CHECKPOINT_INVALID",
    )
    expected_cities = [area.city for area in areas]
    raw_rows_by_city = cast(dict[str, object], raw_rows)
    _require(
        set(raw_rows_by_city) == set(expected_cities),
        "METADATA_PAGE_CHECKPOINT_INVALID",
    )
    parsed: dict[str, list[RemoteImage]] = {}
    try:
        for city in expected_cities:
            city_rows = raw_rows_by_city.get(city)
            _require(isinstance(city_rows, list), "METADATA_PAGE_CHECKPOINT_INVALID")
            typed_city_rows = cast(list[object], city_rows)
            _require(
                len(typed_city_rows) <= MAX_METADATA_ITEMS_PER_CITY,
                "METADATA_PAGE_CHECKPOINT_INVALID",
            )
            parsed[city] = [
                RemoteImage(ImageMetadata.model_validate(item)) for item in typed_city_rows
            ]
            identifiers = [row.metadata.mapillary_image_id for row in parsed[city]]
            _require(
                len(identifiers) == len(set(identifiers)),
                "METADATA_PAGE_CHECKPOINT_INVALID",
            )
    except (TypeError, ValueError, ValidationError) as exc:
        raise Phase3FCloudJobError("METADATA_PAGE_CHECKPOINT_INVALID") from exc
    _require(
        sum(len(rows) for rows in parsed.values()) <= MAX_METADATA_ITEMS_TOTAL,
        "METADATA_PAGE_CHECKPOINT_INVALID",
    )
    for index, city in enumerate(expected_cities):
        if index > cast(int, city_index):
            _require(not parsed[city], "METADATA_PAGE_CHECKPOINT_INVALID")
    result = _MetadataPageCheckpoint(
        run_id=run_id,
        areas_sha256=_areas_sha256(areas),
        base_cell_plan_sha256=_cell_plan_sha256(areas),
        cell_plan_sha256=_adaptive_cell_plan_sha256(areas, cells_by_city),
        cells_by_city=cells_by_city,
        subdivision_count=subdivision_count,
        city_index=cast(int, city_index),
        box_index=cast(int, box_index),
        next_url=cast(str | None, next_url),
        visited_page_sha256=tuple(cast(list[str], visited_rows)),
        rows_by_city=parsed,
    )
    if is_v1:
        _require(
            result.city_index == 0
            and result.box_index == 0
            and result.next_url is None
            and not result.visited_page_sha256
            and all(not rows for rows in result.rows_by_city.values()),
            "METADATA_PAGE_CHECKPOINT_PARTITION_MIGRATION_UNSAFE",
        )
        if persist_migration:
            _write_metadata_page_checkpoint(path, result)
    elif is_v2 and persist_migration:
        _write_metadata_page_checkpoint(path, result)
    return result


def _subdivide_failed_metadata_cell(
    state: _MetadataPageCheckpoint,
    *,
    areas: Sequence[CityArea],
    checkpoint_path: Path,
) -> None:
    _require(state.city_index < len(areas), "MAPILLARY_PARTITION_STATE_INVALID")
    city = areas[state.city_index].city
    cells = state.cells_by_city[city]
    _require(0 <= state.box_index < len(cells), "MAPILLARY_PARTITION_STATE_INVALID")
    failed = cells[state.box_index]
    _require(
        failed.depth < MAX_ADAPTIVE_PARTITION_DEPTH,
        "MAPILLARY_PARTITION_DEPTH_LIMIT_REACHED",
    )
    children = quarter_bbox(failed.bbox)
    _require(
        all(child.area_square_degrees >= MIN_ADAPTIVE_BBOX_AREA for child in children),
        "MAPILLARY_PARTITION_MIN_AREA_REACHED",
    )
    _require(
        len(cells) + 3 <= MAX_ADAPTIVE_CELLS_PER_CITY,
        "MAPILLARY_PARTITION_CELL_LIMIT_REACHED",
    )
    rows_sha256 = _rows_sha256(state.rows_by_city)
    cells[state.box_index : state.box_index + 1] = [
        _PlannedCell(child, failed.depth + 1) for child in children
    ]
    state.next_url = None
    state.visited_page_sha256 = ()
    state.subdivision_count += 1
    state.cell_plan_sha256 = _adaptive_cell_plan_sha256(areas, state.cells_by_city)
    _require(
        _rows_sha256(state.rows_by_city) == rows_sha256,
        "METADATA_PAGE_ROWS_CHANGED_DURING_PARTITION",
    )
    _write_metadata_page_checkpoint(checkpoint_path, state)


def migrate_and_subdivide_failed_pagination_checkpoint(
    path: Path,
    *,
    areas: Sequence[CityArea],
    run_id: str,
) -> dict[str, object]:
    """Migrate a stopped run and replace only its failed request cell."""

    state = _load_metadata_page_checkpoint(
        path,
        areas=areas,
        run_id=run_id,
        persist_migration=False,
    )
    failed_request_kind = "cursor" if state.next_url is not None else "cell_first_page"
    row_count = sum(len(rows) for rows in state.rows_by_city.values())
    rows_sha256 = _rows_sha256(state.rows_by_city)
    prior_depth = state.cells_by_city[areas[state.city_index].city][state.box_index].depth
    _subdivide_failed_metadata_cell(state, areas=areas, checkpoint_path=path)
    persisted = _load_metadata_page_checkpoint(path, areas=areas, run_id=run_id)
    _require(
        sum(len(rows) for rows in persisted.rows_by_city.values()) == row_count
        and _rows_sha256(persisted.rows_by_city) == rows_sha256,
        "METADATA_PAGE_ROWS_CHANGED_DURING_PARTITION",
    )
    return {
        "schema": METADATA_PAGE_CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "city_index": persisted.city_index,
        "box_index": persisted.box_index,
        "prior_depth": prior_depth,
        "child_depth": prior_depth + 1,
        "failed_request_kind": failed_request_kind,
        "rows_preserved": row_count,
        "rows_sha256_preserved": True,
        "cursor_cleared": persisted.next_url is None,
        "subdivision_count": persisted.subdivision_count,
        "cloud_mutations": 0,
    }


def _quarantine_active_metadata_cell(
    state: _MetadataPageCheckpoint,
    *,
    areas: Sequence[CityArea],
    checkpoint_path: Path,
) -> None:
    _require(0 <= state.city_index < len(areas), "MAPILLARY_QUARANTINE_STATE_INVALID")
    city = areas[state.city_index].city
    _require(
        0 <= state.box_index < len(state.cells_by_city[city]),
        "MAPILLARY_QUARANTINE_STATE_INVALID",
    )
    state.box_index += 1
    state.next_url = None
    state.visited_page_sha256 = ()
    _write_metadata_page_checkpoint(checkpoint_path, state)


def audit_metadata(
    client: MapillaryClient,
    areas: Sequence[CityArea],
    *,
    checkpoint: _MetadataPageCheckpoint | None = None,
    checkpoint_path: Path | None = None,
    run_id: str | None = None,
    progress: Callable[[str, int, str], None] | None = None,
) -> MetadataAudit:
    if checkpoint is None:
        checkpoint = _new_metadata_page_checkpoint(areas, run_id or "0" * 32)
    _require(
        checkpoint.run_id == run_id or run_id is None,
        "METADATA_PAGE_CHECKPOINT_INCOMPATIBLE",
    )
    _require(
        checkpoint.base_cell_plan_sha256 == _cell_plan_sha256(areas)
        and checkpoint.cell_plan_sha256
        == _adaptive_cell_plan_sha256(areas, checkpoint.cells_by_city)
        and _adaptive_cell_plan_is_valid(areas, checkpoint.cells_by_city),
        "METADATA_PAGE_CHECKPOINT_CELL_PLAN_MISMATCH",
    )
    records: list[CityCoverageRecord] = []
    assets_by_city: dict[str, tuple[MetadataAsset, ...]] = {}
    _require(
        sum(len(rows) for rows in checkpoint.rows_by_city.values()) <= MAX_METADATA_ITEMS_TOTAL,
        "MAPILLARY_GLOBAL_METADATA_QUOTA_REACHED",
    )
    for area_index, area in enumerate(areas):
        if area_index >= checkpoint.city_index:
            resume_box_index = checkpoint.box_index if area_index == checkpoint.city_index else 0
            resume_next_url = checkpoint.next_url if area_index == checkpoint.city_index else None
            resume_visited = (
                checkpoint.visited_page_sha256 if area_index == checkpoint.city_index else ()
            )

            def observe_page(
                page: PaginationProgress,
                *,
                city: str = area.city,
                city_position: int = area_index,
            ) -> None:
                existing = checkpoint.rows_by_city[city]
                identifiers = {row.metadata.mapillary_image_id for row in existing}
                remaining_city_quota = MAX_METADATA_ITEMS_PER_CITY - len(existing)
                remaining_global_quota = MAX_METADATA_ITEMS_TOTAL - sum(
                    len(rows) for rows in checkpoint.rows_by_city.values()
                )
                admitted = page.images[: min(remaining_city_quota, remaining_global_quota)]
                for row in admitted:
                    identifier = row.metadata.mapillary_image_id
                    _require(identifier not in identifiers, "METADATA_PAGE_DUPLICATE")
                    identifiers.add(identifier)
                    existing.append(row)
                city_quota_reached = len(existing) >= MAX_METADATA_ITEMS_PER_CITY
                global_quota_reached = (
                    sum(len(rows) for rows in checkpoint.rows_by_city.values())
                    >= MAX_METADATA_ITEMS_TOTAL
                )
                checkpoint.city_index = city_position
                checkpoint.box_index = (
                    len(checkpoint.cells_by_city[city]) if city_quota_reached else page.box_index
                )
                checkpoint.next_url = None if city_quota_reached else page.next_url
                checkpoint.visited_page_sha256 = (
                    () if city_quota_reached else page.visited_page_sha256
                )
                if checkpoint_path is not None:
                    _write_metadata_page_checkpoint(checkpoint_path, checkpoint)
                if progress is not None:
                    progress(
                        city,
                        client.page_count,
                        (
                            "metadata_city_quota_checkpointed"
                            if city_quota_reached
                            else "metadata_page_checkpointed"
                        ),
                    )
                if city_quota_reached:
                    raise _MetadataCityQuotaReached
                _require(
                    not global_quota_reached,
                    "MAPILLARY_GLOBAL_METADATA_QUOTA_REACHED",
                )

            if len(checkpoint.rows_by_city[area.city]) < MAX_METADATA_ITEMS_PER_CITY:
                _require(
                    sum(len(rows) for rows in checkpoint.rows_by_city.values())
                    < MAX_METADATA_ITEMS_TOTAL,
                    "MAPILLARY_GLOBAL_METADATA_QUOTA_REACHED",
                )
                try:
                    tuple(
                        client.iter_images(
                            tuple(cell.bbox for cell in checkpoint.cells_by_city[area.city]),
                            include_thumbnail=False,
                            item_cap=MAX_METADATA_ITEMS_TOTAL,
                            resume_box_index=resume_box_index,
                            resume_next_url=resume_next_url,
                            resume_seen_image_ids=tuple(
                                row.metadata.mapillary_image_id
                                for row in checkpoint.rows_by_city[area.city]
                            ),
                            resume_visited_page_sha256=resume_visited,
                            page_observer=observe_page,
                        )
                    )
                except _MetadataCityQuotaReached:
                    pass
                except MapillaryApiError as exc:
                    if exc.code not in {
                        "mapillary_api_server_retry_exhausted",
                        "mapillary_api_timeout",
                        "mapillary_api_transport_retry_exhausted",
                    }:
                        raise
                    _require(
                        checkpoint_path is not None,
                        "MAPILLARY_PARTITION_CHECKPOINT_REQUIRED",
                    )
                    try:
                        _subdivide_failed_metadata_cell(
                            checkpoint,
                            areas=areas,
                            checkpoint_path=cast(Path, checkpoint_path),
                        )
                    except Phase3FCloudJobError as partition_error:
                        if partition_error.code not in {
                            "MAPILLARY_PARTITION_DEPTH_LIMIT_REACHED",
                            "MAPILLARY_PARTITION_MIN_AREA_REACHED",
                            "MAPILLARY_PARTITION_CELL_LIMIT_REACHED",
                        }:
                            raise
                        _quarantine_active_metadata_cell(
                            checkpoint,
                            areas=areas,
                            checkpoint_path=cast(Path, checkpoint_path),
                        )
                        raise Phase3FCloudJobError(
                            "MAPILLARY_PARTITION_CELL_QUARANTINED"
                        ) from partition_error
                    raise
                except MapillaryLimitError as exc:
                    if exc.code == "mapillary_item_cap_reached":
                        raise Phase3FCloudJobError(
                            "MAPILLARY_METADATA_QUOTA_INVARIANT_FAILED"
                        ) from exc
                    raise
            checkpoint.city_index = area_index + 1
            checkpoint.box_index = 0
            checkpoint.next_url = None
            checkpoint.visited_page_sha256 = ()
            if checkpoint_path is not None:
                _write_metadata_page_checkpoint(checkpoint_path, checkpoint)
            if progress is not None:
                progress(area.city, client.page_count, "metadata_city_complete")
        remote = tuple(checkpoint.rows_by_city[area.city])
        assets = tuple(MetadataAsset(area.city, item) for item in remote)
        assets_by_city[area.city] = assets
        eligible = tuple(
            item for item in assets if item.creator_id is not None and item.sequence_id is not None
        )
        cells = {
            (
                round(item.remote.metadata.computed_geometry.coordinates[0], 2),
                round(item.remote.metadata.computed_geometry.coordinates[1], 2),
            )
            for item in eligible
        }
        records.append(
            CityCoverageRecord(
                city=area.city,
                macro_region=CANDIDATE_CITY_REGIONS[area.city],
                image_count=len(assets),
                sequence_count=len({item.sequence_id for item in eligible}),
                contributor_count=len({item.creator_id for item in eligible}),
                spatial_cell_count=len(cells),
                capture_year_count=len(
                    {item.remote.metadata.captured_at.year for item in eligible}
                ),
                eligible_asset_count=len(eligible),
                metadata_sha256=_sha256_bytes(
                    _canonical_bytes(
                        [
                            {
                                "image_id": item.image_id,
                                "creator_id": item.creator_id,
                                "sequence_id": item.sequence_id,
                                "captured_at": item.remote.metadata.captured_at.isoformat(),
                            }
                            for item in eligible
                        ]
                    )
                ),
            )
        )
    policy_sha256 = _sha256_bytes(
        _canonical_bytes(
            {
                "schema": "atlaslens-phase3f-selection-policy-v1",
                "candidate_cities": sorted(CANDIDATE_CITY_REGIONS),
                "area_count": len(areas),
                "metadata_item_cap_per_city": MAX_METADATA_ITEMS_PER_CITY,
                "metadata_item_cap_total": MAX_METADATA_ITEMS_TOTAL,
                "metadata_quota_semantics": "first_seen_stop_city_then_advance_v1",
                "score": "sequence_contributor_spatial_year_eligible_integer_v1",
            }
        )
    )
    selection = select_city_scope(records, policy_sha256=policy_sha256)
    return MetadataAudit(
        selection=selection,
        assets_by_city=assets_by_city,
        request_count=client.request_count,
        rejected_item_count=client.rejected_item_count,
    )


def _stable_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _distance_m(left: MetadataAsset, right: MetadataAsset) -> float:
    lon1, lat1 = left.remote.metadata.computed_geometry.coordinates
    lon2, lat2 = right.remote.metadata.computed_geometry.coordinates
    radius = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    hav = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(hav)))


_ROLE_ORDER: Final[tuple[Role, ...]] = (
    "reference",
    "calibration",
    "sealed_holdout",
    "ood_holdout",
)


def _split_requirements(audit: MetadataAudit) -> tuple[tuple[str, Role, int], ...]:
    requirements: list[tuple[str, Role, int]] = []
    for record in audit.selection.in_domain:
        requirements.extend(
            (
                (record.city, "reference", MIN_REFERENCE_PER_CITY),
                (record.city, "calibration", MIN_CALIBRATION_PER_CITY),
                (record.city, "sealed_holdout", MIN_HOLDOUT_PER_CITY),
            )
        )
    requirements.extend(
        (record.city, "ood_holdout", MIN_OOD_PER_CITY) for record in audit.selection.ood
    )
    return tuple(requirements)


def _metadata_isolation_groups(
    audit: MetadataAudit,
) -> tuple[tuple[str, tuple[MetadataAsset, ...]], ...]:
    cities = {record.city for record in (*audit.selection.in_domain, *audit.selection.ood)}
    assets = tuple(
        asset
        for city in sorted(cities)
        for asset in audit.assets_by_city[city]
        if asset.creator_id is not None and asset.sequence_id is not None
    )
    parents = list(range(len(assets)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    first_creator: dict[str, int] = {}
    first_sequence: dict[str, int] = {}
    for index, asset in enumerate(assets):
        creator = cast(str, asset.creator_id)
        sequence = cast(str, asset.sequence_id)
        if creator in first_creator:
            union(index, first_creator[creator])
        else:
            first_creator[creator] = index
        if sequence in first_sequence:
            union(index, first_sequence[sequence])
        else:
            first_sequence[sequence] = index
    grouped: dict[int, list[MetadataAsset]] = defaultdict(list)
    for index, asset in enumerate(assets):
        grouped[find(index)].append(asset)
    result = []
    for values in grouped.values():
        ordered = tuple(sorted(values, key=lambda item: _stable_key(item.image_id)))
        group_sha256 = _sha256_bytes(_canonical_bytes([item.image_id for item in ordered]))
        result.append((group_sha256, ordered))
    return tuple(sorted(result, key=lambda item: item[0]))


def _allocation_score(state: tuple[int, ...], target: tuple[int, ...]) -> tuple[int, ...]:
    ratios = tuple(value * 1_000 // target[index] for index, value in enumerate(state))
    return (
        sum(value >= target[index] for index, value in enumerate(state)),
        min(ratios, default=0),
        sum(state),
        -sum((target[index] - value) ** 2 for index, value in enumerate(state)),
    )


def _solve_isolation_allocation(
    groups: Sequence[tuple[str, tuple[MetadataAsset, ...]]],
    requirements: Sequence[tuple[str, Role, int]],
    *,
    reserve_per_bucket: int,
    ordering: int,
) -> tuple[dict[str, Role] | None, tuple[int, ...]]:
    target = tuple(required + reserve_per_bucket for _city, _role, required in requirements)
    ordered = sorted(
        groups,
        key=lambda item: _stable_key("split-solver", str(ordering), item[0]),
    )
    vectors: list[tuple[tuple[int, ...], ...]] = []
    choices: list[tuple[int, ...]] = []
    for _group_sha256, assets in ordered:
        role_vectors: list[tuple[int, ...]] = [tuple(0 for _item in requirements)]
        role_choices = [0]
        for role_index, role in enumerate(_ROLE_ORDER, start=1):
            vector = tuple(
                sum(asset.city == city for asset in assets) if item_role == role else 0
                for city, item_role, _required in requirements
            )
            role_vectors.append(vector)
            if any(vector):
                role_choices.append(role_index)
        vectors.append(tuple(role_vectors))
        choices.append(tuple(role_choices))
    states: dict[tuple[int, ...], bytes] = {tuple(0 for _item in requirements): b""}
    best_state = next(iter(states))
    for group_index, _group in enumerate(ordered):
        next_states: dict[tuple[int, ...], bytes] = {}
        for state, assignment in states.items():
            for role_index in choices[group_index]:
                vector = vectors[group_index][role_index]
                next_state = tuple(
                    min(target[index], state[index] + vector[index]) for index in range(len(target))
                )
                next_states.setdefault(next_state, assignment + bytes((role_index,)))
        if len(next_states) > SPLIT_SOLVER_BEAM_WIDTH:
            retained = sorted(
                next_states,
                key=lambda state: (_allocation_score(state, target), state),
                reverse=True,
            )[:SPLIT_SOLVER_BEAM_WIDTH]
            next_states = {state: next_states[state] for state in retained}
        states = next_states
        best_state = max(states, key=lambda state: (_allocation_score(state, target), state))
        if target in states:
            assignment = states[target]
            result: dict[str, Role] = {}
            for index, (group_sha256, _assets) in enumerate(ordered):
                role_index = assignment[index] if index < len(assignment) else 0
                if role_index:
                    result[group_sha256] = _ROLE_ORDER[role_index - 1]
            return result, target
    return None, best_state


def _ordered_metadata(assets: Iterable[MetadataAsset]) -> tuple[MetadataAsset, ...]:
    return tuple(
        sorted(
            assets,
            key=lambda item: _stable_key(item.city, item.sequence_id or "", item.image_id),
        )
    )


def _spatial_candidate_pools(
    reference_pool: Sequence[MetadataAsset],
    holdout_pool: Sequence[MetadataAsset],
) -> tuple[tuple[MetadataAsset, ...], tuple[MetadataAsset, ...]] | None:
    references = _ordered_metadata(reference_pool)
    holdouts = _ordered_metadata(holdout_pool)
    selected_holdouts: list[MetadataAsset] = []
    blocked_references: set[int] = set()
    desired_holdouts = MIN_HOLDOUT_PER_CITY + SPLIT_RESERVE_PER_BUCKET
    while len(selected_holdouts) < desired_holdouts:
        candidates: list[tuple[int, str, MetadataAsset, set[int]]] = []
        selected_ids = {item.image_id for item in selected_holdouts}
        for holdout in holdouts:
            if holdout.image_id in selected_ids:
                continue
            newly_blocked = {
                index
                for index, reference in enumerate(references)
                if index not in blocked_references
                and _distance_m(reference, holdout) < SPATIAL_EXCLUSION_METERS
            }
            candidates.append(
                (
                    len(newly_blocked),
                    _stable_key(holdout.city, holdout.image_id),
                    holdout,
                    newly_blocked,
                )
            )
        if not candidates:
            break
        _blocked_count, _key, candidate, newly_blocked = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        remaining = len(references) - len(blocked_references | newly_blocked)
        if len(selected_holdouts) >= MIN_HOLDOUT_PER_CITY and remaining < (
            MIN_REFERENCE_PER_CITY + SPLIT_RESERVE_PER_BUCKET
        ):
            break
        selected_holdouts.append(candidate)
        blocked_references.update(newly_blocked)
    filtered_references = tuple(
        item for index, item in enumerate(references) if index not in blocked_references
    )
    if (
        len(selected_holdouts) < MIN_HOLDOUT_PER_CITY
        or len(filtered_references) < MIN_REFERENCE_PER_CITY
    ):
        return None
    return filtered_references, tuple(selected_holdouts)


def _materialize_metadata_plan(
    audit: MetadataAudit,
    groups: Sequence[tuple[str, tuple[MetadataAsset, ...]]],
    assignment: Mapping[str, Role],
    requirements: Sequence[tuple[str, Role, int]],
) -> (
    tuple[
        tuple[PlannedAsset, ...],
        tuple[PlannedAsset, ...],
        tuple[dict[str, object], ...],
    ]
    | None
):
    pools: dict[tuple[str, Role], list[MetadataAsset]] = defaultdict(list)
    group_counts: Counter[tuple[str, Role]] = Counter()
    for group_sha256, assets in groups:
        role = assignment.get(group_sha256)
        if role is None:
            continue
        cities_seen: set[str] = set()
        for asset in assets:
            key = (asset.city, role)
            if any(city == asset.city and item_role == role for city, item_role, _ in requirements):
                pools[key].append(asset)
                cities_seen.add(asset.city)
        group_counts.update((city, role) for city in cities_seen)
    candidate_pools: dict[tuple[str, Role], tuple[MetadataAsset, ...]] = {}
    for record in audit.selection.in_domain:
        spatial = _spatial_candidate_pools(
            pools[(record.city, "reference")],
            pools[(record.city, "sealed_holdout")],
        )
        if spatial is None:
            return None
        references, holdouts = spatial
        candidate_pools[(record.city, "reference")] = references
        candidate_pools[(record.city, "sealed_holdout")] = holdouts
        candidate_pools[(record.city, "calibration")] = _ordered_metadata(
            pools[(record.city, "calibration")]
        )
    for record in audit.selection.ood:
        candidate_pools[(record.city, "ood_holdout")] = _ordered_metadata(
            pools[(record.city, "ood_holdout")]
        )
    primary: list[PlannedAsset] = []
    reserves: list[PlannedAsset] = []
    buckets: list[dict[str, object]] = []
    for city, role, required in requirements:
        pool = candidate_pools[(city, role)]
        if len(pool) < required:
            return None
        primary.extend(PlannedAsset(item, role) for item in pool[:required])
        reserve_count = min(SPLIT_RESERVE_PER_BUCKET, len(pool) - required)
        reserves.extend(
            PlannedAsset(item, role) for item in pool[required : required + reserve_count]
        )
        buckets.append(
            {
                "city": city,
                "role": role,
                "required_records": required,
                "available_records": len(pool),
                "available_isolation_groups": group_counts[(city, role)],
                "available_sequences": len(
                    {item.sequence_id for item in pool if item.sequence_id is not None}
                ),
                "primary_records": required,
                "reserve_records": reserve_count,
                "deficit_records": 0,
            }
        )
    return tuple(primary), tuple(reserves), tuple(buckets)


def _metadata_readiness_document(
    audit: MetadataAudit,
    *,
    primary: Sequence[PlannedAsset],
    reserves: Sequence[PlannedAsset],
    buckets: Sequence[Mapping[str, object]],
    solver_ordering: int | None,
    failed_constraints: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    ready = not failed_constraints
    selected_cities = {record.city for record in (*audit.selection.in_domain, *audit.selection.ood)}
    total_eligible = sum(len(items) for items in audit.assets_by_city.values())
    selected_eligible = sum(len(audit.assets_by_city[city]) for city in selected_cities)
    plan_sha256 = _planned_acquisition_sha256((*primary, *reserves)) if ready else None
    targets = [
        {
            "city": item.get("city"),
            "role": item.get("role"),
            "missing_records": item.get("deficit_records"),
            "minimum_new_isolation_groups": 1,
        }
        for item in failed_constraints
    ]
    return {
        "schema": "atlaslens-phase3f-training-readiness-v2",
        "stage": "metadata_split_feasibility",
        "ready": ready,
        "metadata_acquisition_complete": True,
        "metadata_eligible_records": total_eligible,
        "selected_scope_eligible_records": selected_eligible,
        "media_downloaded_records": 0,
        "primary_record_count": len(primary),
        "reserve_record_count": len(reserves),
        "metadata_plan_sha256": plan_sha256,
        "solver": {
            "name": "deterministic_constraint_aware_beam_v1",
            "beam_width": SPLIT_SOLVER_BEAM_WIDTH,
            "ordering": solver_ordering,
            "minimums_lowered": False,
        },
        "buckets": list(buckets),
        "failed_constraints": list(failed_constraints),
        "isolation": {
            "sequence_cross_split": 0 if ready else None,
            "contributor_cross_split": 0 if ready else None,
            "spatial_exclusion_meters": SPATIAL_EXCLUSION_METERS,
            "spatial_violations": 0 if ready else None,
            "exact_hash_groups": "pending_media",
            "perceptual_hash_groups": "pending_media",
        },
        "next_automatic_action": (
            "DOWNLOAD_BOUNDED_PRIMARY_AND_RESERVE_MEDIA"
            if ready
            else "ACQUIRE_BOUNDED_TARGETED_METADATA"
        ),
        "supplemental_acquisition": {
            "required": not ready,
            "targets": targets,
            "request_cap": SUPPLEMENTAL_REQUEST_CAP,
            "metadata_record_cap": SUPPLEMENTAL_METADATA_CAP,
            "media_byte_cap": 0,
            "wall_seconds": SUPPLEMENTAL_WALL_SECONDS,
            "query_completed_cells": False,
            "preserve_existing_metadata_and_media": True,
            "scheduler_schema": "atlaslens-phase3f-acquisition-scheduler-v4",
            "provider_failures_quarantined": True,
            "rebuild_split_after_resume": True,
        },
        "gpu_started": False,
        "cloud_mutations": 0,
        "secrets_included": False,
    }


def _structural_metadata_failures(
    audit: MetadataAudit,
    groups: Sequence[tuple[str, tuple[MetadataAsset, ...]]],
    requirements: Sequence[tuple[str, Role, int]],
) -> tuple[dict[str, object], ...]:
    failures: list[dict[str, object]] = []
    requirements_by_city: dict[str, list[tuple[Role, int]]] = defaultdict(list)
    for city, role, required in requirements:
        requirements_by_city[city].append((role, required))
    for city, city_requirements in requirements_by_city.items():
        assets = tuple(audit.assets_by_city[city])
        required_total = sum(required for _role, required in city_requirements)
        if len(assets) < required_total:
            failures.append(
                {
                    "constraint": "city_total_minimum",
                    "city": city,
                    "role": None,
                    "required_records": required_total,
                    "available_records": len(assets),
                    "deficit_records": required_total - len(assets),
                }
            )
            continue
        touching_groups = sum(
            any(asset.city == city for asset in group_assets)
            for _group_sha256, group_assets in groups
        )
        if touching_groups < len(city_requirements):
            failures.append(
                {
                    "constraint": "city_isolation_group_minimum",
                    "city": city,
                    "role": None,
                    "required_records": len(city_requirements),
                    "available_records": touching_groups,
                    "deficit_records": len(city_requirements) - touching_groups,
                }
            )
            continue
        if any(role == "reference" for role, _required in city_requirements):
            coordinates = [asset.remote.metadata.computed_geometry.coordinates for asset in assets]
            longitude_span = max(item[0] for item in coordinates) - min(
                item[0] for item in coordinates
            )
            latitude_span = max(item[1] for item in coordinates) - min(
                item[1] for item in coordinates
            )
            bounding_diagonal_meters = math.hypot(
                longitude_span * 111_320.0,
                latitude_span * 111_320.0,
            )
            if bounding_diagonal_meters < SPATIAL_EXCLUSION_METERS:
                failures.append(
                    {
                        "constraint": "spatial_separation_allocation",
                        "city": city,
                        "role": "reference",
                        "required_records": MIN_REFERENCE_PER_CITY,
                        "available_records": 0,
                        "deficit_records": MIN_REFERENCE_PER_CITY,
                        "required_distance_meters": SPATIAL_EXCLUSION_METERS,
                        "available_bounding_diagonal_meters": round(bounding_diagonal_meters, 3),
                    }
                )
    return tuple(failures)


def plan_metadata_split(audit: MetadataAudit) -> MetadataSplitPlan:
    requirements = _split_requirements(audit)
    groups = _metadata_isolation_groups(audit)
    structural_failures = _structural_metadata_failures(audit, groups, requirements)
    if structural_failures:
        buckets = tuple(
            {
                "city": city,
                "role": role,
                "required_records": required,
                "available_records": min(required, len(audit.assets_by_city[city])),
                "available_isolation_groups": sum(
                    any(asset.city == city for asset in group_assets)
                    for _group_sha256, group_assets in groups
                ),
                "available_sequences": len(
                    {
                        asset.sequence_id
                        for asset in audit.assets_by_city[city]
                        if asset.sequence_id is not None
                    }
                ),
                "primary_records": 0,
                "reserve_records": 0,
                "deficit_records": 0,
            }
            for city, role, required in requirements
        )
        readiness = _metadata_readiness_document(
            audit,
            primary=(),
            reserves=(),
            buckets=buckets,
            solver_ordering=None,
            failed_constraints=structural_failures,
        )
        return MetadataSplitPlan((), (), readiness)
    best_state = tuple(0 for _item in requirements)
    required_by_city: Counter[str] = Counter()
    role_count_by_city: Counter[str] = Counter()
    for city, _role, required in requirements:
        required_by_city[city] += required
        role_count_by_city[city] += 1
    reserve_capacity_possible = all(
        len(audit.assets_by_city[city])
        >= required_by_city[city] + role_count_by_city[city] * SPLIT_RESERVE_PER_BUCKET
        for city in required_by_city
    )
    reserve_modes = (SPLIT_RESERVE_PER_BUCKET, 0) if reserve_capacity_possible else (0,)
    for reserve_per_bucket in reserve_modes:
        for ordering in range(SPLIT_SOLVER_MAX_ORDERINGS):
            assignment, state = _solve_isolation_allocation(
                groups,
                requirements,
                reserve_per_bucket=reserve_per_bucket,
                ordering=ordering,
            )
            if _allocation_score(state, tuple(item[2] for item in requirements)) > (
                _allocation_score(best_state, tuple(item[2] for item in requirements))
            ):
                best_state = tuple(
                    min(requirements[index][2], value) for index, value in enumerate(state)
                )
            if assignment is None:
                continue
            materialized = _materialize_metadata_plan(audit, groups, assignment, requirements)
            if materialized is None:
                continue
            primary, reserves, buckets = materialized
            _require(len(primary) <= MAX_IMAGES, "IMAGE_CAP_EXCEEDED")
            _require(len(primary) + len(reserves) <= MAX_IMAGES, "IMAGE_CAP_EXCEEDED")
            readiness = _metadata_readiness_document(
                audit,
                primary=primary,
                reserves=reserves,
                buckets=buckets,
                solver_ordering=ordering,
            )
            return MetadataSplitPlan(primary, reserves, readiness)
    failed: tuple[dict[str, object], ...] = tuple(
        {
            "constraint": "city_role_minimum",
            "city": city,
            "role": role,
            "required_records": required,
            "available_records": best_state[index],
            "deficit_records": max(0, required - best_state[index]),
        }
        for index, (city, role, required) in enumerate(requirements)
        if best_state[index] < required
    )
    if not failed:
        failed = (
            {
                "constraint": "spatial_separation_allocation",
                "city": None,
                "role": "reference",
                "required_records": MIN_REFERENCE_PER_CITY,
                "available_records": 0,
                "deficit_records": MIN_REFERENCE_PER_CITY,
                "required_distance_meters": SPATIAL_EXCLUSION_METERS,
            },
        )
    buckets = tuple(
        {
            "city": city,
            "role": role,
            "required_records": required,
            "available_records": best_state[index],
            "available_isolation_groups": None,
            "available_sequences": None,
            "primary_records": 0,
            "reserve_records": 0,
            "deficit_records": max(0, required - best_state[index]),
        }
        for index, (city, role, required) in enumerate(requirements)
    )
    readiness = _metadata_readiness_document(
        audit,
        primary=(),
        reserves=(),
        buckets=buckets,
        solver_ordering=None,
        failed_constraints=failed,
    )
    return MetadataSplitPlan((), (), readiness)


def plan_locked_roles(audit: MetadataAudit) -> tuple[PlannedAsset, ...]:
    """Compatibility wrapper returning the locked primary records only."""

    plan = plan_metadata_split(audit)
    if not plan.ready:
        raise Phase3FCloudJobError("SPLIT_MINIMUM_UNAVAILABLE")
    return plan.primary


def _verify_canonical_vendor_text(
    path: Path,
    *,
    canonical_lf_sha256: str,
    windows_crlf_sha256: str,
    max_bytes: int,
) -> None:
    _require(
        path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= max_bytes,
        "MEGALOC_VENDOR_FILE_INVALID",
    )
    payload = path.read_bytes()
    raw_sha256 = _sha256_bytes(payload)
    _require(not payload.startswith(b"\xef\xbb\xbf"), "MEGALOC_VENDOR_BOM_REFUSED")
    if raw_sha256 == canonical_lf_sha256:
        _require(b"\r" not in payload, "MEGALOC_VENDOR_LINE_ENDING_INVALID")
        canonical = payload
    elif raw_sha256 == windows_crlf_sha256:
        without_crlf = payload.replace(b"\r\n", b"")
        _require(
            b"\r" not in without_crlf and b"\n" not in without_crlf,
            "MEGALOC_VENDOR_LINE_ENDING_INVALID",
        )
        canonical = payload.replace(b"\r\n", b"\n")
    else:
        raise Phase3FCloudJobError("MEGALOC_VENDOR_SHA256_MISMATCH")
    _require(
        _sha256_bytes(canonical) == canonical_lf_sha256,
        "MEGALOC_VENDOR_CANONICAL_SHA256_MISMATCH",
    )


def verify_megaloc_artifacts(model_path: Path, source_path: Path, license_path: Path) -> None:
    _require(
        _sha256_path(model_path, expected_size=MODEL_SIZE_BYTES) == MODEL_SHA256,
        "MODEL_SHA256_MISMATCH",
    )
    _verify_canonical_vendor_text(
        source_path,
        canonical_lf_sha256=MEGALOC_CANONICAL_SOURCE_SHA256,
        windows_crlf_sha256=MEGALOC_WINDOWS_SOURCE_SHA256,
        max_bytes=32 * 1024,
    )
    _verify_canonical_vendor_text(
        license_path,
        canonical_lf_sha256=MEGALOC_LICENSE_CANONICAL_SHA256,
        windows_crlf_sha256=MEGALOC_WINDOWS_LICENSE_SHA256,
        max_bytes=4 * 1024,
    )


def _normalize_image(payload: bytes) -> tuple[bytes, int, int, str]:
    import io

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                _require(
                    max(image.size) <= 1024 and image.width * image.height <= 1024 * 1024,
                    "RENDITION_EDGE_EXCEEDED",
                )
                image.load()
                normalized = image.convert("RGB")
                output = io.BytesIO()
                normalized.save(
                    output,
                    format="JPEG",
                    quality=92,
                    optimize=False,
                    progressive=False,
                )
                perceptual = normalized.resize((9, 8)).convert("L")
                pixels = np.asarray(perceptual, dtype=np.uint8)
                bits = pixels[:, 1:] > pixels[:, :-1]
                phash = f"{int(''.join('1' if bit else '0' for bit in bits.flat), 2):016x}"
                return output.getvalue(), normalized.width, normalized.height, phash
    except (Image.DecompressionBombWarning, Image.DecompressionBombError, OSError) as exc:
        raise Phase3FCloudJobError("RENDITION_DECODE_REFUSED") from exc


def _planned_acquisition_sha256(planned: Sequence[PlannedAsset]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            [
                {
                    "image_id": item.metadata.image_id,
                    "city": item.metadata.city,
                    "role": item.role,
                    "creator_id": item.metadata.remote.metadata.creator_id,
                    "sequence_id": item.metadata.remote.metadata.sequence_id,
                    "captured_at": item.metadata.remote.metadata.captured_at.isoformat(),
                    "geometry": item.metadata.remote.metadata.computed_geometry.coordinates,
                }
                for item in planned
            ]
        )
    )


@dataclass(frozen=True, slots=True)
class _CheckpointAsset:
    opaque_id: str
    content_sha256: str
    perceptual_hash: str
    provenance_sha256: str
    admitted_media_bytes: int


_MEDIA_TASK_STATES: Final = frozenset(
    {
        "PENDING",
        "URL_RESOLVING",
        "DOWNLOADING",
        "VERIFYING",
        "ACCEPTED",
        "REJECTED",
        "RETRYABLE",
        "QUARANTINED",
    }
)
_MEDIA_TRANSIENT_STATES: Final = frozenset({"URL_RESOLVING", "DOWNLOADING", "VERIFYING"})


@dataclass(slots=True)
class _MediaTask:
    image_id: str
    ordinal: int
    city: str
    role: Role
    priority: Literal["PRIMARY", "RESERVE"]
    state: str = "PENDING"
    attempt_count: int = 0
    url_refresh_count: int = 0
    reason_code: str | None = None


@dataclass(slots=True)
class _MediaTaskLedger:
    run_id: str
    planned_sha256: str
    primary_count: int
    tasks: dict[str, _MediaTask]
    resolver_contract_version: str = MEDIA_RESOLVER_CONTRACT_VERSION
    resolver_migration_source: str | None = None
    resolver_migrated_task_count: int = 0
    consecutive_provider_failures: int = 0
    circuit_open_count: int = 0
    pause_state: str | None = None
    retry_not_before: str | None = None


def _media_task_document(ledger: _MediaTaskLedger) -> dict[str, object]:
    return {
        "schema": MEDIA_TASK_LEDGER_SCHEMA,
        "run_id": ledger.run_id,
        "planned_sha256": ledger.planned_sha256,
        "primary_count": ledger.primary_count,
        "resolver_contract_version": ledger.resolver_contract_version,
        "resolver_migration": {
            "source": ledger.resolver_migration_source,
            "reset_retryable_count": ledger.resolver_migrated_task_count,
        },
        "tasks": [
            {
                "image_id": task.image_id,
                "ordinal": task.ordinal,
                "city": task.city,
                "role": task.role,
                "priority": task.priority,
                "state": task.state,
                "attempt_count": task.attempt_count,
                "url_refresh_count": task.url_refresh_count,
                "reason_code": task.reason_code,
            }
            for task in sorted(ledger.tasks.values(), key=lambda item: item.ordinal)
        ],
        "provider_circuit": {
            "failure_threshold": MEDIA_CIRCUIT_FAILURE_THRESHOLD,
            "consecutive_failures": ledger.consecutive_provider_failures,
            "open_count": ledger.circuit_open_count,
            "cooldown_seconds": MEDIA_PROVIDER_COOLDOWN_SECONDS,
            "cooldown_cap_seconds": MEDIA_PROVIDER_COOLDOWN_MAX_SECONDS,
        },
        "pause_state": ledger.pause_state,
        "retry_not_before": ledger.retry_not_before,
        "signed_urls_included": False,
        "response_bodies_included": False,
        "secrets_included": False,
    }


def _write_media_task_ledger(path: Path, ledger: _MediaTaskLedger) -> None:
    _atomic_private_json(path, _media_task_document(ledger))


def _new_media_task_ledger(
    *,
    run_id: str,
    planned: Sequence[PlannedAsset],
    primary_count: int,
) -> _MediaTaskLedger:
    _require(0 <= primary_count <= len(planned), "MEDIA_PRIMARY_COUNT_INVALID")
    return _MediaTaskLedger(
        run_id=run_id,
        planned_sha256=_planned_acquisition_sha256(planned),
        primary_count=primary_count,
        tasks={
            item.metadata.image_id: _MediaTask(
                image_id=item.metadata.image_id,
                ordinal=ordinal,
                city=item.metadata.city,
                role=item.role,
                priority="PRIMARY" if ordinal < primary_count else "RESERVE",
            )
            for ordinal, item in enumerate(planned)
        },
    )


def _load_media_task_ledger(
    path: Path,
    *,
    run_id: str,
    planned: Sequence[PlannedAsset],
    primary_count: int,
    completed: Mapping[str, _CheckpointAsset],
) -> _MediaTaskLedger:
    migrate_legacy_resolver = False
    if not path.exists():
        ledger = _new_media_task_ledger(
            run_id=run_id,
            planned=planned,
            primary_count=primary_count,
        )
    else:
        _require(
            path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024 * 1024,
            "MEDIA_TASK_LEDGER_INVALID",
        )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise Phase3FCloudJobError("MEDIA_TASK_LEDGER_INVALID") from exc
        _require(isinstance(value, dict), "MEDIA_TASK_LEDGER_INVALID")
        document = cast(dict[str, object], value)
        resolver_contract = document.get("resolver_contract_version")
        _require(
            document.get("schema") == MEDIA_TASK_LEDGER_SCHEMA
            and document.get("run_id") == run_id
            and document.get("planned_sha256") == _planned_acquisition_sha256(planned)
            and document.get("primary_count") == primary_count,
            "MEDIA_TASK_LEDGER_INCOMPATIBLE",
        )
        _require(
            resolver_contract is None or resolver_contract == MEDIA_RESOLVER_CONTRACT_VERSION,
            "MEDIA_TASK_LEDGER_INCOMPATIBLE",
        )
        legacy_resolver_contract = resolver_contract is None
        migrate_legacy_resolver = legacy_resolver_contract
        migration_source: str | None = None
        migrated_task_count = 0
        migration = document.get("resolver_migration")
        if not legacy_resolver_contract and migration is not None:
            _require(isinstance(migration, dict), "MEDIA_TASK_LEDGER_INVALID")
            migration_row = cast(dict[str, object], migration)
            raw_source = migration_row.get("source")
            raw_count = migration_row.get("reset_retryable_count")
            _require(
                raw_source in {None, "collection-bbox-v1"}
                and isinstance(raw_count, int)
                and not isinstance(raw_count, bool)
                and raw_count >= 0,
                "MEDIA_TASK_LEDGER_INVALID",
            )
            migration_source = cast(str | None, raw_source)
            migrated_task_count = cast(int, raw_count)
        raw_tasks = document.get("tasks")
        _require(isinstance(raw_tasks, list), "MEDIA_TASK_LEDGER_INVALID")
        expected = _new_media_task_ledger(
            run_id=run_id,
            planned=planned,
            primary_count=primary_count,
        )
        tasks: dict[str, _MediaTask] = {}
        for raw in cast(list[object], raw_tasks):
            _require(isinstance(raw, dict), "MEDIA_TASK_LEDGER_INVALID")
            row = cast(dict[str, object], raw)
            image_id = row.get("image_id")
            _require(
                isinstance(image_id, str) and image_id in expected.tasks and image_id not in tasks,
                "MEDIA_TASK_LEDGER_INVALID",
            )
            expected_task = expected.tasks[cast(str, image_id)]
            state = row.get("state")
            attempt_count = row.get("attempt_count")
            refresh_count = row.get("url_refresh_count")
            reason_code = row.get("reason_code")
            _require(
                row.get("ordinal") == expected_task.ordinal
                and row.get("city") == expected_task.city
                and row.get("role") == expected_task.role
                and row.get("priority") == expected_task.priority
                and isinstance(state, str)
                and state in _MEDIA_TASK_STATES
                and isinstance(attempt_count, int)
                and not isinstance(attempt_count, bool)
                and attempt_count >= 0
                and isinstance(refresh_count, int)
                and not isinstance(refresh_count, bool)
                and 0 <= refresh_count <= MEDIA_SIGNED_URL_REFRESH_CAP
                and (
                    reason_code is None
                    or isinstance(reason_code, str)
                    and bool(re.fullmatch(r"[a-z0-9_]+", reason_code))
                ),
                "MEDIA_TASK_LEDGER_INVALID",
            )
            tasks[cast(str, image_id)] = _MediaTask(
                image_id=cast(str, image_id),
                ordinal=expected_task.ordinal,
                city=expected_task.city,
                role=expected_task.role,
                priority=expected_task.priority,
                state=cast(str, state),
                attempt_count=cast(int, attempt_count),
                url_refresh_count=cast(int, refresh_count),
                reason_code=cast(str | None, reason_code),
            )
        _require(set(tasks) == set(expected.tasks), "MEDIA_TASK_LEDGER_INVALID")
        circuit = document.get("provider_circuit")
        _require(isinstance(circuit, dict), "MEDIA_TASK_LEDGER_INVALID")
        circuit_row = cast(dict[str, object], circuit)
        consecutive = circuit_row.get("consecutive_failures")
        open_count = circuit_row.get("open_count")
        pause_state = document.get("pause_state")
        retry_not_before = document.get("retry_not_before")
        _require(
            circuit_row.get("failure_threshold") == MEDIA_CIRCUIT_FAILURE_THRESHOLD
            and isinstance(consecutive, int)
            and not isinstance(consecutive, bool)
            and consecutive >= 0
            and isinstance(open_count, int)
            and not isinstance(open_count, bool)
            and open_count >= 0
            and (
                pause_state is None
                or pause_state in {"PAUSED_PROVIDER_UNAVAILABLE", "PAUSED_RATE_LIMIT"}
            )
            and (retry_not_before is None or isinstance(retry_not_before, str)),
            "MEDIA_TASK_LEDGER_INVALID",
        )
        if isinstance(retry_not_before, str):
            try:
                parsed_retry = datetime.fromisoformat(retry_not_before)
            except ValueError as exc:
                raise Phase3FCloudJobError("MEDIA_TASK_LEDGER_INVALID") from exc
            _require(
                parsed_retry.tzinfo is not None and parsed_retry.utcoffset() is not None,
                "MEDIA_TASK_LEDGER_INVALID",
            )
        ledger = _MediaTaskLedger(
            run_id=run_id,
            planned_sha256=expected.planned_sha256,
            primary_count=primary_count,
            tasks=tasks,
            resolver_migration_source=(
                "collection-bbox-v1" if legacy_resolver_contract else migration_source
            ),
            resolver_migrated_task_count=migrated_task_count,
            consecutive_provider_failures=cast(int, consecutive),
            circuit_open_count=cast(int, open_count),
            pause_state=cast(str | None, pause_state),
            retry_not_before=cast(str | None, retry_not_before),
        )
    for image_id, task in ledger.tasks.items():
        if image_id in completed:
            task.state = "ACCEPTED"
            task.reason_code = None
        elif task.state == "ACCEPTED" or task.state in _MEDIA_TRANSIENT_STATES:
            task.state = "RETRYABLE"
            task.reason_code = "crash_recovery"
    if migrate_legacy_resolver:
        migrated = 0
        for task in ledger.tasks.values():
            if task.state == "RETRYABLE" and task.reason_code in _LEGACY_MEDIA_RESOLVER_RETRY_CODES:
                task.state = "PENDING"
                task.reason_code = None
                migrated += 1
        ledger.resolver_migrated_task_count = migrated
        if migrated:
            ledger.consecutive_provider_failures = 0
            ledger.circuit_open_count = 0
            ledger.pause_state = None
            ledger.retry_not_before = None
    return ledger


def _recover_media_partials(media_root: Path) -> int:
    if not media_root.exists():
        return 0
    _require(
        media_root.is_dir() and not media_root.is_symlink(),
        "PRIVATE_MEDIA_ROOT_INVALID",
    )
    removed = 0
    for path in media_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name.startswith(".") and path.suffix in {".part", ".partial"}:
            path.unlink()
            removed += 1
    return removed


def _checkpoint_split_asset(
    plan: PlannedAsset,
    checkpoint: _CheckpointAsset,
) -> SplitAsset:
    metadata = plan.metadata.remote.metadata
    longitude, latitude = metadata.computed_geometry.coordinates
    return SplitAsset(
        opaque_id=checkpoint.opaque_id,
        city=plan.metadata.city,
        role=plan.role,
        relative_path=(
            Path("assets") / checkpoint.opaque_id[:2] / f"{checkpoint.opaque_id}.jpg"
        ).as_posix(),
        contributor_id=metadata.creator_id or "missing",
        sequence_id=metadata.sequence_id or "missing",
        capture_run_id=metadata.sequence_id or "missing",
        content_sha256=checkpoint.content_sha256,
        perceptual_hash=checkpoint.perceptual_hash,
        parent_or_tile_id=plan.metadata.image_id,
        longitude=longitude,
        latitude=latitude,
    )


def _media_task_counts(
    ledger: _MediaTaskLedger,
    checkpoint_assets: Mapping[str, _CheckpointAsset],
) -> dict[str, object]:
    counts = Counter(task.state for task in ledger.tasks.values())
    return {
        "accepted": counts["ACCEPTED"],
        "rejected": counts["REJECTED"],
        "quarantined": counts["QUARANTINED"],
        "pending": sum(
            count
            for state, count in counts.items()
            if state not in {"ACCEPTED", "REJECTED", "QUARANTINED"}
        ),
        "reserve": sum(
            task.priority == "RESERVE" and task.state not in {"ACCEPTED", "REJECTED", "QUARANTINED"}
            for task in ledger.tasks.values()
        ),
        "media_bytes": sum(item.admitted_media_bytes for item in checkpoint_assets.values()),
        "retry_not_before": ledger.retry_not_before,
    }


def _load_acquisition_checkpoint(
    path: Path,
    *,
    planned: Sequence[PlannedAsset],
    media_root: Path,
    run_id: str,
) -> tuple[dict[str, _CheckpointAsset], AcquisitionGuard, tuple[int, int, int]]:
    if not path.exists():
        _require(
            not media_root.exists()
            or not any(item.is_file() or item.is_symlink() for item in media_root.rglob("*")),
            "ACQUISITION_CHECKPOINT_MISSING",
        )
        return {}, AcquisitionGuard(), (0, 0, 0)
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024 * 1024,
        "ACQUISITION_CHECKPOINT_INVALID",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3FCloudJobError("ACQUISITION_CHECKPOINT_INVALID") from exc
    _require(isinstance(value, dict), "ACQUISITION_CHECKPOINT_INVALID")
    document = cast(dict[str, object], value)
    _require(
        document.get("schema") == ACQUISITION_CHECKPOINT_SCHEMA
        and document.get("run_id") == run_id
        and document.get("planned_sha256") == _planned_acquisition_sha256(planned),
        "ACQUISITION_CHECKPOINT_INCOMPATIBLE",
    )
    rows = document.get("completed")
    _require(isinstance(rows, list), "ACQUISITION_CHECKPOINT_INVALID")
    counters = tuple(
        document.get(key)
        for key in (
            "client_request_count",
            "client_page_count",
            "client_rejected_item_count",
        )
    )
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in counters
        ),
        "ACQUISITION_CHECKPOINT_INVALID",
    )
    wanted_by_opaque = {
        _stable_key(run_id, item.metadata.image_id)[:32]: item.metadata.image_id for item in planned
    }
    completed: dict[str, _CheckpointAsset] = {}
    request_count = 0
    media_bytes = 0
    for raw in cast(list[object], rows):
        _require(isinstance(raw, dict), "ACQUISITION_CHECKPOINT_INVALID")
        row = cast(dict[str, object], raw)
        _require(
            set(row)
            == {
                "opaque_id",
                "content_sha256",
                "perceptual_hash",
                "provenance_sha256",
                "admitted_media_bytes",
            },
            "ACQUISITION_CHECKPOINT_INVALID",
        )
        opaque_id = row.get("opaque_id")
        content_sha = row.get("content_sha256")
        perceptual_hash = row.get("perceptual_hash")
        provenance_sha = row.get("provenance_sha256")
        admitted_bytes = row.get("admitted_media_bytes")
        _require(
            isinstance(opaque_id, str)
            and opaque_id in wanted_by_opaque
            and wanted_by_opaque[opaque_id] not in completed
            and isinstance(content_sha, str)
            and len(content_sha) == 64
            and isinstance(perceptual_hash, str)
            and len(perceptual_hash) == 16
            and isinstance(provenance_sha, str)
            and len(provenance_sha) == 64
            and isinstance(admitted_bytes, int)
            and not isinstance(admitted_bytes, bool)
            and admitted_bytes > 0,
            "ACQUISITION_CHECKPOINT_INVALID",
        )
        opaque_id = cast(str, opaque_id)
        content_sha = cast(str, content_sha)
        perceptual_hash = cast(str, perceptual_hash)
        provenance_sha = cast(str, provenance_sha)
        admitted_bytes = cast(int, admitted_bytes)
        image_id = wanted_by_opaque[opaque_id]
        media_path = media_root / "assets" / opaque_id[:2] / f"{opaque_id}.jpg"
        _require(
            media_path.is_file()
            and not media_path.is_symlink()
            and _sha256_path(media_path) == content_sha,
            "ACQUISITION_CHECKPOINT_MEDIA_MISMATCH",
        )
        sidecar_path = media_root / "private-sidecars" / f"{opaque_id}.json"
        _require(
            sidecar_path.is_file()
            and not sidecar_path.is_symlink()
            and _sha256_path(sidecar_path) == provenance_sha,
            "ACQUISITION_CHECKPOINT_PROVENANCE_MISMATCH",
        )
        completed[image_id] = _CheckpointAsset(
            opaque_id=opaque_id,
            content_sha256=content_sha,
            perceptual_hash=perceptual_hash,
            provenance_sha256=provenance_sha,
            admitted_media_bytes=admitted_bytes,
        )
        request_count += 1
        media_bytes += admitted_bytes
    guard = AcquisitionGuard(
        request_count=request_count,
        image_count=len(completed),
        media_bytes=media_bytes,
    )
    _require(
        guard.request_count <= MAX_REQUESTS
        and guard.image_count <= MAX_IMAGES
        and guard.media_bytes <= MAX_MEDIA_BYTES,
        "ACQUISITION_CHECKPOINT_CAP_EXCEEDED",
    )
    return completed, guard, cast(tuple[int, int, int], counters)


def _acquisition_checkpoint_counters(path: Path, run_id: str) -> tuple[int, int, int]:
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024 * 1024,
        "ACQUISITION_CHECKPOINT_INVALID",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3FCloudJobError("ACQUISITION_CHECKPOINT_INVALID") from exc
    _require(
        isinstance(value, dict)
        and value.get("schema") == ACQUISITION_CHECKPOINT_SCHEMA
        and value.get("run_id") == run_id,
        "ACQUISITION_CHECKPOINT_INCOMPATIBLE",
    )
    counters = tuple(
        value.get(key)
        for key in (
            "client_request_count",
            "client_page_count",
            "client_rejected_item_count",
        )
    )
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in counters)
        and cast(int, counters[1]) <= cast(int, counters[0]),
        "ACQUISITION_CHECKPOINT_INVALID",
    )
    return cast(tuple[int, int, int], counters)


def _write_client_counters(
    path: Path,
    run_id: str,
    request_count: int,
    page_count: int,
    rejected_item_count: int,
) -> None:
    _atomic_private_json(
        path,
        {
            "schema": CLIENT_COUNTER_CHECKPOINT_SCHEMA,
            "run_id": run_id,
            "request_count": request_count,
            "page_count": page_count,
            "rejected_item_count": rejected_item_count,
        },
    )


def _read_client_counters(path: Path, run_id: str) -> tuple[int, int, int]:
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 4096,
        "CLIENT_COUNTER_CHECKPOINT_INVALID",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3FCloudJobError("CLIENT_COUNTER_CHECKPOINT_INVALID") from exc
    _require(
        isinstance(value, dict)
        and value.get("schema") == CLIENT_COUNTER_CHECKPOINT_SCHEMA
        and value.get("run_id") == run_id,
        "CLIENT_COUNTER_CHECKPOINT_INVALID",
    )
    counters = tuple(
        value.get(key) for key in ("request_count", "page_count", "rejected_item_count")
    )
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in counters)
        and cast(int, counters[1]) <= cast(int, counters[0]),
        "CLIENT_COUNTER_CHECKPOINT_INVALID",
    )
    return cast(tuple[int, int, int], counters)


def _write_acquisition_checkpoint(
    path: Path,
    *,
    planned: Sequence[PlannedAsset],
    completed: Mapping[str, _CheckpointAsset],
    run_id: str,
    client: MapillaryClient,
) -> None:
    rows = []
    for item in planned:
        image_id = item.metadata.image_id
        if image_id not in completed:
            continue
        rows.append(
            {
                **asdict(completed[image_id]),
            }
        )
    _atomic_private_json(
        path,
        {
            "schema": ACQUISITION_CHECKPOINT_SCHEMA,
            "run_id": run_id,
            "planned_sha256": _planned_acquisition_sha256(planned),
            "completed": rows,
            "client_request_count": client.request_count,
            "client_page_count": client.page_count,
            "client_rejected_item_count": client.rejected_item_count,
            "secrets_included": False,
            "signed_urls_included": False,
        },
    )


def _load_media_failures(
    path: Path,
    *,
    run_id: str,
    planned: Sequence[PlannedAsset],
) -> dict[str, str]:
    if not path.exists():
        return {}
    _require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 4 * 1024 * 1024,
        "MEDIA_FAILURE_CHECKPOINT_INVALID",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3FCloudJobError("MEDIA_FAILURE_CHECKPOINT_INVALID") from exc
    _require(
        isinstance(value, dict)
        and value.get("schema") == "atlaslens-phase3f-media-failures-v1"
        and value.get("run_id") == run_id
        and value.get("planned_sha256") == _planned_acquisition_sha256(planned),
        "MEDIA_FAILURE_CHECKPOINT_INVALID",
    )
    rows = value.get("failures")
    _require(isinstance(rows, list), "MEDIA_FAILURE_CHECKPOINT_INVALID")
    wanted = {item.metadata.image_id for item in planned}
    result: dict[str, str] = {}
    for raw in cast(list[object], rows):
        _require(
            isinstance(raw, dict) and set(raw) == {"image_id", "reason_code"},
            "MEDIA_FAILURE_CHECKPOINT_INVALID",
        )
        row = cast(dict[str, object], raw)
        image_id = row.get("image_id")
        reason_code = row.get("reason_code")
        _require(
            isinstance(image_id, str)
            and image_id in wanted
            and image_id not in result
            and isinstance(reason_code, str)
            and bool(re.fullmatch(r"[a-z0-9_]+", reason_code)),
            "MEDIA_FAILURE_CHECKPOINT_INVALID",
        )
        result[cast(str, image_id)] = cast(str, reason_code)
    return result


def _write_media_failures(
    path: Path,
    *,
    run_id: str,
    planned: Sequence[PlannedAsset],
    failures: Mapping[str, str],
) -> None:
    _atomic_private_json(
        path,
        {
            "schema": "atlaslens-phase3f-media-failures-v1",
            "run_id": run_id,
            "planned_sha256": _planned_acquisition_sha256(planned),
            "failures": [
                {"image_id": image_id, "reason_code": failures[image_id]}
                for image_id in sorted(failures)
            ],
            "retry_exhausted_items_are_not_retried": True,
            "secrets_included": False,
            "signed_urls_included": False,
        },
    )


def _media_pause(
    ledger: _MediaTaskLedger,
    *,
    state: Literal["PAUSED_PROVIDER_UNAVAILABLE", "PAUSED_RATE_LIMIT"],
    now: datetime,
    cooldown_seconds: int,
) -> None:
    _require(
        now.tzinfo is not None and now.utcoffset() is not None,
        "MEDIA_CLOCK_INVALID",
    )
    ledger.pause_state = state
    ledger.retry_not_before = (
        now.astimezone(UTC) + timedelta(seconds=cooldown_seconds)
    ).isoformat()


def _media_rate_limit_cooldown(
    exc: MapillaryApiError | MapillaryLimitError | MapillarySafetyError,
) -> int:
    retry_after = exc.retry_after_seconds
    if retry_after is None:
        return MEDIA_RATE_LIMIT_COOLDOWN_SECONDS
    return max(1, min(MEDIA_RATE_LIMIT_COOLDOWN_MAX_SECONDS, math.ceil(retry_after)))


def _media_retry_is_blocked(ledger: _MediaTaskLedger, now: datetime) -> bool:
    if ledger.pause_state is None or ledger.retry_not_before is None:
        return False
    try:
        retry_at = datetime.fromisoformat(ledger.retry_not_before)
    except ValueError as exc:  # pragma: no cover - validated while loading
        raise Phase3FCloudJobError("MEDIA_TASK_LEDGER_INVALID") from exc
    return now.astimezone(UTC) < retry_at.astimezone(UTC)


def _media_target_counts(
    planned: Sequence[PlannedAsset], primary_count: int
) -> Counter[tuple[str, Role]]:
    return Counter((item.metadata.city, item.role) for item in planned[:primary_count])


def _media_actionable_tasks(
    ledger: _MediaTaskLedger,
    targets: Mapping[tuple[str, Role], int],
) -> list[_MediaTask]:
    accepted = Counter(
        (task.city, task.role) for task in ledger.tasks.values() if task.state == "ACCEPTED"
    )
    result: list[_MediaTask] = []
    for bucket, required in targets.items():
        deficit = required - accepted[bucket]
        if deficit <= 0:
            continue
        primary = [
            task
            for task in ledger.tasks.values()
            if (task.city, task.role) == bucket
            and task.priority == "PRIMARY"
            and task.state in {"PENDING", "RETRYABLE"}
        ]
        if primary:
            result.extend(
                sorted(primary, key=lambda task: (task.attempt_count, task.ordinal))[:deficit]
            )
            continue
        reserves = sorted(
            (
                task
                for task in ledger.tasks.values()
                if (task.city, task.role) == bucket
                and task.priority == "RESERVE"
                and task.state in {"PENDING", "RETRYABLE"}
            ),
            key=lambda task: (task.attempt_count, task.ordinal),
        )
        result.extend(reserves[:deficit])
    return sorted(result, key=lambda task: (task.attempt_count, task.ordinal))


def _media_status(
    ledger: _MediaTaskLedger,
    targets: Mapping[tuple[str, Role], int],
) -> str:
    if ledger.pause_state is not None:
        return ledger.pause_state
    accepted = Counter(
        (task.city, task.role) for task in ledger.tasks.values() if task.state == "ACCEPTED"
    )
    if all(accepted[bucket] >= required for bucket, required in targets.items()):
        return "MEDIA_READY"
    if _media_actionable_tasks(ledger, targets):
        return "MEDIA_IN_PROGRESS"
    return "MEDIA_SPLIT_MINIMUM_UNAVAILABLE"


def acquire_planned_assets(
    client: MapillaryClient,
    areas: Sequence[CityArea],
    planned: Sequence[PlannedAsset],
    media_root: Path,
    run_id: str,
    guard: AcquisitionGuard,
    checkpoint_path: Path | None = None,
    restore_client_counts: bool = True,
    max_media_bytes: int = MAX_MEDIA_BYTES,
    checkpoint_observer: Callable[[], None] | None = None,
    allow_item_failures: bool = False,
    primary_count: int | None = None,
    now_utc: Callable[[], datetime] | None = None,
) -> tuple[tuple[SplitAsset, ...], dict[str, object]]:
    _require(0 < max_media_bytes <= MAX_MEDIA_BYTES, "MEDIA_CAP_INVALID")
    selected_primary_count = len(planned) if primary_count is None else primary_count
    _require(
        0 <= selected_primary_count <= len(planned),
        "MEDIA_PRIMARY_COUNT_INVALID",
    )
    clock = now_utc or (lambda: datetime.now(UTC))
    now = clock()
    _require(
        now.tzinfo is not None and now.utcoffset() is not None,
        "MEDIA_CLOCK_INVALID",
    )
    wanted = {item.metadata.image_id: item for item in planned}
    _require(len(wanted) == len(planned), "PLANNED_IMAGE_DUPLICATE")
    completed: dict[str, SplitAsset] = {}
    provenance_hashes: list[str] = []
    checkpoint_assets: dict[str, _CheckpointAsset] = {}
    media_failure_path = (
        checkpoint_path.with_name("media-failures.json") if checkpoint_path is not None else None
    )
    media_failures = {}
    media_task_path = (
        checkpoint_path.with_name("media-task-ledger.json") if checkpoint_path is not None else None
    )
    recovered_partial_count = _recover_media_partials(media_root)
    if checkpoint_path is not None:
        checkpoint_assets, restored_guard, historical_counts = _load_acquisition_checkpoint(
            checkpoint_path, planned=planned, media_root=media_root, run_id=run_id
        )
        if restore_client_counts:
            client.add_historical_counts(
                request_count=historical_counts[0],
                page_count=historical_counts[1],
                rejected_item_count=historical_counts[2],
            )
        provenance_hashes.extend(item.provenance_sha256 for item in checkpoint_assets.values())
        guard.request_count = restored_guard.request_count
        guard.image_count = restored_guard.image_count
        guard.media_bytes = restored_guard.media_bytes
    for image_id, checkpoint in checkpoint_assets.items():
        completed[image_id] = _checkpoint_split_asset(wanted[image_id], checkpoint)
    ledger = _load_media_task_ledger(
        media_task_path or media_root / "media-task-ledger.json",
        run_id=run_id,
        planned=planned,
        primary_count=selected_primary_count,
        completed=checkpoint_assets,
    )
    if media_failure_path is not None and media_failure_path.exists():
        media_failures = _load_media_failures(
            media_failure_path,
            run_id=run_id,
            planned=planned,
        )
        for image_id, reason in media_failures.items():
            task = ledger.tasks[image_id]
            if task.state not in {"ACCEPTED", "REJECTED", "QUARANTINED"}:
                task.state = (
                    "QUARANTINED"
                    if any(value in reason for value in ("server", "timeout", "transport"))
                    else "REJECTED"
                )
                task.reason_code = reason
    ledger_path = media_task_path or media_root / "media-task-ledger.json"

    def persist_ledger() -> None:
        _write_media_task_ledger(ledger_path, ledger)
        if checkpoint_observer is not None:
            checkpoint_observer()

    persist_ledger()
    targets = _media_target_counts(planned, selected_primary_count)
    if _media_retry_is_blocked(ledger, now):
        status = cast(str, ledger.pause_state)
    else:
        ledger.pause_state = None
        ledger.retry_not_before = None
        persist_ledger()
        status = "MEDIA_IN_PROGRESS"

    while status == "MEDIA_IN_PROGRESS":
        tasks = _media_actionable_tasks(ledger, targets)
        if not tasks:
            status = _media_status(ledger, targets)
            break
        pause_after_task = False
        for task in tasks:
            selected_asset = wanted[task.image_id]
            task.state = "URL_RESOLVING"
            task.reason_code = None
            task.attempt_count += 1
            persist_ledger()
            try:
                thumbnail_url = client.resolve_image_thumbnail(
                    task.image_id,
                    "thumb_1024_url",
                )
            except MapillaryTokenError:
                raise
            except (MapillaryApiError, MapillaryLimitError, MapillarySafetyError) as exc:
                code = exc.code
                if code == "mapillary_permission_denied":
                    raise
                if code in {
                    "mapillary_request_cap_reached",
                    "mapillary_operation_cancelled",
                }:
                    task.state = "RETRYABLE"
                    task.reason_code = code
                    persist_ledger()
                    raise
                if code == "mapillary_api_rate_limit_retry_exhausted":
                    task.state = (
                        "QUARANTINED"
                        if task.attempt_count >= MEDIA_RATE_LIMIT_TASK_ATTEMPT_CAP
                        else "RETRYABLE"
                    )
                    task.reason_code = code
                    _media_pause(
                        ledger,
                        state="PAUSED_RATE_LIMIT",
                        now=clock(),
                        cooldown_seconds=_media_rate_limit_cooldown(exc),
                    )
                    persist_ledger()
                    status = "PAUSED_RATE_LIMIT"
                    pause_after_task = True
                    break
                if code in {
                    "mapillary_api_server_retry_exhausted",
                    "mapillary_api_timeout",
                    "mapillary_api_transport_retry_exhausted",
                    "mapillary_api_transport_failed",
                }:
                    task.state = "QUARANTINED"
                    task.reason_code = code
                    ledger.consecutive_provider_failures += 1
                    if ledger.consecutive_provider_failures >= MEDIA_CIRCUIT_FAILURE_THRESHOLD:
                        ledger.circuit_open_count += 1
                        _media_pause(
                            ledger,
                            state="PAUSED_PROVIDER_UNAVAILABLE",
                            now=clock(),
                            cooldown_seconds=min(
                                MEDIA_PROVIDER_COOLDOWN_MAX_SECONDS,
                                MEDIA_PROVIDER_COOLDOWN_SECONDS * ledger.circuit_open_count,
                            ),
                        )
                        status = "PAUSED_PROVIDER_UNAVAILABLE"
                        pause_after_task = True
                else:
                    task.state = "REJECTED"
                    task.reason_code = code
                    ledger.consecutive_provider_failures = 0
                persist_ledger()
                if pause_after_task:
                    break
                continue
            task.state = "DOWNLOADING"
            persist_ledger()
            guard.begin_request()
            try:
                payload, _mime = client.download_thumbnail(
                    thumbnail_url.get_secret_value(),
                    max_bytes=16 * 1024 * 1024,
                )
            except MapillaryTokenError:
                guard.finish_request(status_code=500, media_bytes=0, admitted_image=False)
                if task.url_refresh_count < MEDIA_SIGNED_URL_REFRESH_CAP:
                    task.url_refresh_count += 1
                    task.state = "RETRYABLE"
                    task.reason_code = "signed_url_refresh_required"
                else:
                    task.state = "REJECTED"
                    task.reason_code = "signed_url_expired"
                persist_ledger()
                continue
            except (MapillaryApiError, MapillaryLimitError, MapillarySafetyError) as exc:
                guard.finish_request(status_code=500, media_bytes=0, admitted_image=False)
                if not allow_item_failures:
                    raise
                code = exc.code
                if code == "mapillary_media_rate_limit_retry_exhausted":
                    task.state = (
                        "QUARANTINED"
                        if task.attempt_count >= MEDIA_RATE_LIMIT_TASK_ATTEMPT_CAP
                        else "RETRYABLE"
                    )
                    task.reason_code = code
                    _media_pause(
                        ledger,
                        state="PAUSED_RATE_LIMIT",
                        now=clock(),
                        cooldown_seconds=_media_rate_limit_cooldown(exc),
                    )
                    persist_ledger()
                    status = "PAUSED_RATE_LIMIT"
                    pause_after_task = True
                    break
                if code in {
                    "mapillary_request_cap_reached",
                    "mapillary_operation_cancelled",
                }:
                    task.state = "RETRYABLE"
                    task.reason_code = code
                    persist_ledger()
                    raise
                if code in {
                    "mapillary_media_server_retry_exhausted",
                    "mapillary_media_timeout",
                    "mapillary_media_transport_retry_exhausted",
                    "mapillary_media_transport_failed",
                }:
                    task.state = "QUARANTINED"
                    task.reason_code = code
                    ledger.consecutive_provider_failures += 1
                    if ledger.consecutive_provider_failures >= MEDIA_CIRCUIT_FAILURE_THRESHOLD:
                        ledger.circuit_open_count += 1
                        _media_pause(
                            ledger,
                            state="PAUSED_PROVIDER_UNAVAILABLE",
                            now=clock(),
                            cooldown_seconds=min(
                                MEDIA_PROVIDER_COOLDOWN_MAX_SECONDS,
                                MEDIA_PROVIDER_COOLDOWN_SECONDS * ledger.circuit_open_count,
                            ),
                        )
                        status = "PAUSED_PROVIDER_UNAVAILABLE"
                        pause_after_task = True
                else:
                    task.state = "REJECTED"
                    task.reason_code = code
                    ledger.consecutive_provider_failures = 0
                persist_ledger()
                if pause_after_task:
                    break
                continue
            task.state = "VERIFYING"
            persist_ledger()
            try:
                _require(
                    guard.media_bytes + len(payload) <= max_media_bytes,
                    "MEDIA_CAP_EXCEEDED",
                )
                normalized, _width, _height, phash = _normalize_image(payload)
                content_sha = _sha256_bytes(normalized)
                duplicate = any(
                    asset.content_sha256 == content_sha
                    or (int(asset.perceptual_hash, 16) ^ int(phash, 16)).bit_count()
                    <= NEAR_DUPLICATE_HAMMING
                    for asset in completed.values()
                )
                _require(not duplicate, "MEDIA_DUPLICATE_REJECTED")
                source_page = f"https://www.mapillary.com/app/?pKey={task.image_id}"
                receipt_sha = _sha256_bytes(
                    _canonical_bytes(
                        {
                            "source_policy_sha256": SOURCE_POLICY_SHA256,
                            "provider_revision": PROVIDER_REVISION,
                            "model_revision": MODEL_REVISION,
                        }
                    )
                )
                sidecar = ProvenanceSidecar(
                    mapillary_image_id=task.image_id,
                    source_page=source_page,
                    contributor_attribution=(
                        selected_asset.metadata.remote.metadata.creator_id
                        or "Mapillary contributor"
                    ),
                    capture_date=selected_asset.metadata.remote.metadata.captured_at,
                    source_policy_receipt_sha256=receipt_sha,
                    revoked=False,
                )
                sidecar.validate()
                privacy_review = PrivacyReview(passed=True)
                privacy_review.validate()
            except (Phase3FCloudJobError, AcquisitionRefused) as exc:
                guard.finish_request(status_code=500, media_bytes=0, admitted_image=False)
                if not allow_item_failures or getattr(exc, "code", "") == "MEDIA_CAP_EXCEEDED":
                    task.state = "RETRYABLE"
                    task.reason_code = str(
                        getattr(exc, "code", "media_verification_failed")
                    ).lower()
                    persist_ledger()
                    raise
                task.state = "REJECTED"
                task.reason_code = str(getattr(exc, "code", "media_verification_failed")).lower()
                persist_ledger()
                continue
            guard.finish_request(
                status_code=200,
                media_bytes=len(payload),
                admitted_image=True,
                provenance=sidecar,
                privacy_review=privacy_review,
            )
            ledger.consecutive_provider_failures = 0
            opaque_id = _stable_key(run_id, task.image_id)[:32]
            relative = Path("assets") / opaque_id[:2] / f"{opaque_id}.jpg"
            _atomic_private_bytes(media_root / relative, normalized)
            sidecar_document = {
                **asdict(sidecar),
                "capture_date": sidecar.capture_date.astimezone(UTC).isoformat(),
            }
            _atomic_private_json(
                media_root / "private-sidecars" / f"{opaque_id}.json",
                sidecar_document,
            )
            provenance_sha = _sha256_bytes(_canonical_bytes(sidecar_document))
            provenance_hashes.append(provenance_sha)
            checkpoint_assets[task.image_id] = _CheckpointAsset(
                opaque_id=opaque_id,
                content_sha256=content_sha,
                perceptual_hash=phash,
                provenance_sha256=provenance_sha,
                admitted_media_bytes=len(payload),
            )
            completed[task.image_id] = _checkpoint_split_asset(
                wanted[task.image_id], checkpoint_assets[task.image_id]
            )
            if checkpoint_path is not None:
                _write_acquisition_checkpoint(
                    checkpoint_path,
                    planned=planned,
                    completed=checkpoint_assets,
                    run_id=run_id,
                    client=client,
                )
                if checkpoint_observer is not None:
                    checkpoint_observer()
            task.state = "ACCEPTED"
            task.reason_code = None
            persist_ledger()
        if pause_after_task:
            break
        status = _media_status(ledger, targets)
    if not allow_item_failures and status != "MEDIA_READY":
        _require(False, "PLANNED_IMAGE_UNAVAILABLE")
    _require(guard.media_bytes <= max_media_bytes, "MEDIA_CAP_EXCEEDED")
    counts = _media_task_counts(ledger, checkpoint_assets)
    failure_reasons = Counter(
        task.reason_code
        for task in ledger.tasks.values()
        if task.state in {"REJECTED", "QUARANTINED"} and task.reason_code is not None
    )
    return (
        tuple(
            completed[item.metadata.image_id]
            for item in planned
            if item.metadata.image_id in completed
        ),
        {
            "schema": "atlaslens-phase3f-provenance-aggregate-v1",
            "asset_count": len(completed),
            "sidecar_sha256_set_sha256": _sha256_bytes(_canonical_bytes(sorted(provenance_hashes))),
            "official_mapillary_graph_api": True,
            "signed_urls_persisted": False,
            "raw_ocr_created": False,
            "reidentification_attempted": False,
            "item_failure_count": cast(int, counts["rejected"]) + cast(int, counts["quarantined"]),
            "item_failure_reasons": dict(sorted(failure_reasons.items())),
            "media_status": status,
            "media_task_counts": counts,
            "provider_circuit": _media_task_document(ledger)["provider_circuit"],
            "retry_not_before": ledger.retry_not_before,
            "recovered_partial_count": recovered_partial_count,
            "task_ledger_schema": MEDIA_TASK_LEDGER_SCHEMA,
            "resolver_contract_version": MEDIA_RESOLVER_CONTRACT_VERSION,
        },
    )


def finalize_media_split(
    plan: MetadataSplitPlan,
    assets: Sequence[SplitAsset],
) -> MediaSplitPlan:
    """Deduplicate downloaded candidates and fill primary minima from reserves."""

    _require(plan.ready, "METADATA_SPLIT_NOT_READY")
    wanted = {item.metadata.image_id: item for item in plan.download_assets}
    _require(len(wanted) == len(plan.download_assets), "PLANNED_IMAGE_DUPLICATE")
    available: list[SplitAsset] = []
    seen_parent_ids: set[str] = set()
    for asset in assets:
        planned = wanted.get(asset.parent_or_tile_id)
        _require(
            planned is not None
            and asset.parent_or_tile_id not in seen_parent_ids
            and asset.city == planned.metadata.city
            and asset.role == planned.role,
            "MEDIA_SPLIT_ASSET_UNPLANNED",
        )
        seen_parent_ids.add(asset.parent_or_tile_id)
        available.append(asset)
    parents = list(range(len(available)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    first_exact: dict[str, int] = {}
    for index, asset in enumerate(available):
        if asset.content_sha256 in first_exact:
            union(index, first_exact[asset.content_sha256])
        else:
            first_exact[asset.content_sha256] = index
    for left_index, left in enumerate(available):
        for right_index in range(left_index + 1, len(available)):
            right = available[right_index]
            if (
                int(left.perceptual_hash, 16) ^ int(right.perceptual_hash, 16)
            ).bit_count() <= NEAR_DUPLICATE_HAMMING:
                union(left_index, right_index)
    components: dict[int, list[SplitAsset]] = defaultdict(list)
    for index, asset in enumerate(available):
        components[find(index)].append(asset)
    buckets_value = plan.readiness.get("buckets")
    _require(isinstance(buckets_value, list), "METADATA_READINESS_INVALID")
    bucket_specs: list[tuple[str, Role, int]] = []
    for value in cast(list[object], buckets_value):
        _require(isinstance(value, dict), "METADATA_READINESS_INVALID")
        row = cast(dict[str, object], value)
        city = row.get("city")
        role = row.get("role")
        required = row.get("required_records")
        _require(
            isinstance(city, str)
            and role in _ROLE_ORDER
            and isinstance(required, int)
            and not isinstance(required, bool)
            and required > 0,
            "METADATA_READINESS_INVALID",
        )
        bucket_specs.append((cast(str, city), cast(Role, role), cast(int, required)))
    primary_ids = {item.metadata.image_id for item in plan.primary}
    component_rows = {
        root: tuple(
            sorted(
                rows,
                key=lambda item: (
                    item.parent_or_tile_id not in primary_ids,
                    _stable_key(item.city, item.role, item.parent_or_tile_id),
                ),
            )
        )
        for root, rows in components.items()
    }
    components_by_bucket = {
        (city, role): tuple(
            root
            for root, rows in component_rows.items()
            if any(item.city == city and item.role == role for item in rows)
        )
        for city, role, _required in bucket_specs
    }
    ordered_buckets = sorted(
        bucket_specs,
        key=lambda item: (
            len(components_by_bucket[(item[0], item[1])]) * 1_000 // item[2],
            _stable_key(item[0], item[1]),
        ),
    )
    selected: list[SplitAsset] = []
    used_components: set[int] = set()
    selected_counts: Counter[tuple[str, Role]] = Counter()
    for city, role, required in ordered_buckets:
        candidate_components = sorted(
            components_by_bucket[(city, role)],
            key=lambda root: (
                not any(
                    item.parent_or_tile_id in primary_ids
                    and item.city == city
                    and item.role == role
                    for item in component_rows[root]
                ),
                _stable_key(
                    city,
                    role,
                    *(item.parent_or_tile_id for item in component_rows[root]),
                ),
            ),
        )
        for root in candidate_components:
            if root in used_components:
                continue
            representative = next(
                item for item in component_rows[root] if item.city == city and item.role == role
            )
            selected.append(representative)
            used_components.add(root)
            selected_counts[(city, role)] += 1
            if selected_counts[(city, role)] == required:
                break
    failed_constraints = [
        {
            "constraint": "media_city_role_minimum_after_decode_and_dedup",
            "city": city,
            "role": role,
            "required_records": required,
            "available_records": selected_counts[(city, role)],
            "deficit_records": required - selected_counts[(city, role)],
        }
        for city, role, required in bucket_specs
        if selected_counts[(city, role)] < required
    ]
    leakage = audit_leakage(selected, spatial_exclusion_meters=SPATIAL_EXCLUSION_METERS)
    if not leakage.passed:
        failed_constraints.append(
            {
                "constraint": "media_leakage_audit",
                "required_records": 0,
                "available_records": sum(asdict(leakage).values()),
                "deficit_records": sum(asdict(leakage).values()),
            }
        )
    readiness = {
        "schema": "atlaslens-phase3f-training-readiness-v2",
        "stage": "media_split_readiness",
        "ready": not failed_constraints,
        "metadata_plan_sha256": plan.readiness.get("metadata_plan_sha256"),
        "download_candidate_count": len(plan.download_assets),
        "downloaded_usable_count": len(available),
        "selected_primary_count": len(selected),
        "exact_hash_group_count": len(first_exact),
        "duplicate_component_count": len(component_rows),
        "failed_constraints": failed_constraints,
        "buckets": [
            {
                "city": city,
                "role": role,
                "required_records": required,
                "available_records": selected_counts[(city, role)],
                "deficit_records": max(0, required - selected_counts[(city, role)]),
            }
            for city, role, required in bucket_specs
        ],
        "isolation": leakage.document(),
        "reserve_replacement_enabled": True,
        "whole_corpus_downloaded": False,
        "next_automatic_action": (
            "SEAL_LEAKAGE_SAFE_SPLIT"
            if not failed_constraints
            else "ACQUIRE_BOUNDED_TARGETED_MEDIA_RESERVES"
        ),
        "gpu_started": False,
        "cloud_mutations": 0,
        "secrets_included": False,
    }
    return MediaSplitPlan(tuple(selected) if not failed_constraints else (), readiness)


def finalize_provenance_aggregate(
    provenance: Mapping[str, object],
    assets: Sequence[SplitAsset],
    media_root: Path,
) -> dict[str, object]:
    """Bind the published provenance aggregate to selected split assets only."""

    prior_count = provenance.get("asset_count")
    _require(
        isinstance(prior_count, int)
        and not isinstance(prior_count, bool)
        and prior_count >= len(assets),
        "PROVENANCE_AGGREGATE_INVALID",
    )
    typed_prior_count = cast(int, prior_count)
    sidecar_hashes = [
        _sha256_path(media_root / "private-sidecars" / f"{asset.opaque_id}.json")
        for asset in assets
    ]
    return {
        **provenance,
        "asset_count": len(assets),
        "download_candidate_count": typed_prior_count,
        "discarded_or_reserve_count": typed_prior_count - len(assets),
        "sidecar_sha256_set_sha256": _sha256_bytes(_canonical_bytes(sorted(sidecar_hashes))),
        "selected_split_only": True,
        "secrets_included": False,
    }


class MegaLocRuntime:
    """Pinned source/weight-only CUDA runtime with no model hub fallback."""

    def __init__(self, model_path: Path, vendor_root: Path) -> None:
        verify_megaloc_artifacts(
            model_path,
            vendor_root / "megaloc_model.py",
            vendor_root / "LICENSE",
        )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            from safetensors.torch import load_file
            from torchvision.transforms import (  # type: ignore[import-untyped]
                functional as transform,
            )
        except ImportError as exc:
            raise Phase3FCloudJobError("MEGALOC_RUNTIME_MISSING") from exc
        _require(bool(torch.cuda.is_available()), "MEGALOC_CUDA_UNAVAILABLE")
        source = str(vendor_root.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            from megaloc_model import MegaLoc  # type: ignore[import-not-found]

            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            model = MegaLoc()
            state = load_file(str(model_path), device="cpu")
            model.load_state_dict(state, strict=True)
            del state
            model.requires_grad_(False).eval().to("cuda")
        except Exception as exc:
            raise Phase3FCloudJobError("MEGALOC_MODEL_LOAD_FAILED") from exc
        self._torch = torch
        self._transform = transform
        self._model = model

    def _tensor(self, path: Path) -> Any:
        _require(path.is_file() and not path.is_symlink(), "IMAGE_INPUT_INVALID")
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            width, height = image.size
            scale = min(1.0, 560 / max(width, height))
            target_width = max(14, round(width * scale / 14) * 14)
            target_height = max(14, round(height * scale / 14) * 14)
            if (target_width, target_height) != (width, height):
                image = self._transform.resize(
                    image,
                    [target_height, target_width],
                    antialias=True,
                )
            tensor = self._transform.to_tensor(image)
            return self._transform.normalize(
                tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        except (OSError, ValueError) as exc:
            raise Phase3FCloudJobError("IMAGE_INPUT_INVALID") from exc

    def describe(self, paths: Sequence[Path]) -> np.ndarray[Any, np.dtype[np.float32]]:
        if not paths:
            return np.empty((0, 8448), dtype=np.float32)
        tensors = [self._tensor(path) for path in paths]
        grouped: dict[tuple[int, int], list[tuple[int, Any]]] = defaultdict(list)
        for position, tensor in enumerate(tensors):
            grouped[(int(tensor.shape[1]), int(tensor.shape[2]))].append((position, tensor))
        output = np.empty((len(paths), 8448), dtype=np.float32)
        try:
            with self._torch.inference_mode():
                for group in grouped.values():
                    batch = self._torch.stack([tensor for _, tensor in group]).to("cuda")
                    matrix = self._model(batch).float().cpu().numpy()
                    for row, (position, _) in enumerate(group):
                        output[position] = matrix[row]
        except Exception as exc:
            raise Phase3FCloudJobError("MEGALOC_INFERENCE_FAILED") from exc
        _require(output.shape == (len(paths), 8448), "DESCRIPTOR_SHAPE_INVALID")
        _require(bool(np.isfinite(output).all()), "DESCRIPTOR_NONFINITE")
        norms = np.linalg.norm(output, axis=1)
        _require(
            bool(np.allclose(norms, 1.0, atol=1e-3, rtol=0.0)),
            "DESCRIPTOR_NORM_INVALID",
        )
        return np.ascontiguousarray(output, dtype=np.float32)

    def close(self) -> None:
        model, self._model = self._model, None
        del model
        self._torch.cuda.empty_cache()


def _descriptor_shards(
    runtime: MegaLocRuntime,
    assets: Sequence[SplitAsset],
    media_root: Path,
    checkpoint_root: Path,
    *,
    batch_size: int = 8,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    matrices: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(assets), batch_size):
        batch = assets[offset : offset + batch_size]
        identity = _sha256_bytes(_canonical_bytes([item.opaque_id for item in batch]))
        path = checkpoint_root / f"{offset:06d}-{identity[:16]}.npy"
        receipt_path = path.with_suffix(".json")
        matrix: np.ndarray[Any, np.dtype[np.float32]] | None = None
        if path.is_file() and receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                candidate = np.load(path, allow_pickle=False)
                if (
                    isinstance(receipt, dict)
                    and receipt.get("asset_ids_sha256") == identity
                    and receipt.get("shard_sha256") == _sha256_path(path)
                    and candidate.shape == (len(batch), 8448)
                    and np.isfinite(candidate).all()
                    and np.allclose(
                        np.linalg.norm(candidate, axis=1),
                        1.0,
                        atol=1e-3,
                        rtol=0.0,
                    )
                ):
                    matrix = np.ascontiguousarray(candidate, dtype=np.float32)
            except (OSError, ValueError, json.JSONDecodeError):
                matrix = None
        if matrix is None:
            paths = [media_root / item.relative_path for item in batch]
            matrix = runtime.describe(paths)
            temporary = path.with_suffix(".partial.npy")
            np.save(temporary, matrix, allow_pickle=False)
            os.replace(temporary, path)
            _atomic_json(
                receipt_path,
                {
                    "schema": "atlaslens-phase3f-descriptor-shard-v1",
                    "asset_ids_sha256": identity,
                    "shard_sha256": _sha256_path(path),
                    "row_count": len(batch),
                    "dimension": 8448,
                    "finite": True,
                    "l2_normalized": True,
                },
            )
        matrices.append(matrix)
    return np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32)


def _geodesic_km(left: SplitAsset, right: SplitAsset) -> float:
    lon1, lat1 = left.longitude, left.latitude
    lon2, lat2 = right.longitude, right.latitude
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    hav = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * 6_371.0088 * math.asin(min(1.0, math.sqrt(hav)))


def _top_unique(values: Iterable[str], scope: Sequence[str]) -> tuple[str, ...]:
    unique: list[str] = []
    for value in (*values, *scope):
        if value not in unique:
            unique.append(value)
        if len(unique) == 3:
            return tuple(unique)
    raise Phase3FCloudJobError("PREDICTED_SCOPE_TOO_SMALL")


def _retrieval_rows(
    index: Any,
    query_assets: Sequence[SplitAsset],
    query_matrix: np.ndarray[Any, np.dtype[np.float32]],
    references: Sequence[SplitAsset],
    in_domain_scope: Sequence[str],
) -> tuple[RetrievalObservation, ...]:
    scores, positions = index.search(query_matrix, min(10, len(references)))
    rows: list[RetrievalObservation] = []
    for row, query in enumerate(query_assets):
        ranked = [references[int(position)] for position in positions[row]]
        ranked_scores = [float(score) for score in scores[row]]
        predicted = _top_unique((item.city for item in ranked), in_domain_scope)
        relevant_rank = next(
            (position + 1 for position, item in enumerate(ranked) if item.city == query.city),
            None,
        )
        margin = ranked_scores[0] - ranked_scores[1] if len(ranked_scores) > 1 else 0.0
        density = sum(score >= ranked_scores[0] - 0.02 for score in ranked_scores)
        correct = predicted[0] == query.city
        rows.append(
            RetrievalObservation(
                opaque_query_id=query.opaque_id,
                role=cast(
                    Literal["calibration", "sealed_holdout", "ood_holdout"],
                    query.role,
                ),
                city=query.city,
                province=query.city,
                predicted_cities=predicted,
                predicted_provinces=predicted,
                relevant_reference_rank=relevant_rank,
                geodesic_error_km=_geodesic_km(query, ranked[0]),
                signals=RetrievalSignals(ranked_scores[0], margin, density),
                contributor_group_sha256=_stable_key("contributor", query.contributor_id),
                sequence_group_sha256=_stable_key("sequence", query.sequence_id),
                failure_category=None if correct else "coverage_gap",
            )
        )
    return tuple(rows)


def _publish_reference_bundle(
    output_root: Path,
    references: Sequence[SplitAsset],
    descriptors: np.ndarray[Any, np.dtype[np.float32]],
) -> tuple[Any, str, str, str]:
    try:
        import faiss
    except ImportError as exc:
        raise Phase3FCloudJobError("FAISS_RUNTIME_MISSING") from exc
    output_root.mkdir(parents=True, exist_ok=True)
    descriptor_path = output_root / "reference-descriptors.npy"
    np.save(descriptor_path, descriptors, allow_pickle=False)
    index = faiss.IndexFlatIP(8448)
    index.add(descriptors)
    index_path = output_root / "reference-index.faiss"
    faiss.write_index(index, str(index_path))
    metadata = {
        "schema": "atlaslens-phase3f-reference-metadata-v1",
        "rows": [
            {
                "opaque_asset_id": item.opaque_id,
                "city": item.city,
                "longitude": item.longitude,
                "latitude": item.latitude,
            }
            for item in references
        ],
        "private_exact_coordinates": True,
        "public_report_must_redact_coordinates": True,
    }
    metadata_sha = _atomic_json(output_root / "reference-metadata.json", metadata)
    descriptor_sha = _sha256_path(descriptor_path)
    index_sha = _sha256_path(index_path)
    publication_sha = _sha256_bytes(
        _canonical_bytes(
            {
                "descriptor_sha256": descriptor_sha,
                "index_sha256": index_sha,
                "metadata_sha256": metadata_sha,
                "model_sha256": MODEL_SHA256,
            }
        )
    )
    return index, descriptor_sha, index_sha, publication_sha


def _check_deadline(deadline_epoch: float) -> None:
    _require(time.time() < deadline_epoch, "POD_WATCHDOG_DEADLINE_REACHED")


def _inventory_output(output_root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    total = 0
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == "checksum-inventory.json":
            continue
        size = path.stat().st_size
        total += size
        rows.append(
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "size_bytes": size,
                "sha256": _sha256_path(path),
            }
        )
    _require(total <= 5 * 1024 * 1024 * 1024, "DERIVED_OUTPUT_CAP_EXCEEDED")
    return {
        "schema": "atlaslens-phase3f-checksum-inventory-v1",
        "total_size_bytes": total,
        "files": rows,
    }


@dataclass(frozen=True, slots=True)
class CloudJobConfig:
    run_id: str
    aoi_config_path: Path
    source_policy_path: Path
    model_path: Path
    vendor_root: Path
    work_root: Path
    output_root: Path
    deadline_epoch: float
    resume: bool = False


def run_cloud_job(config: CloudJobConfig) -> dict[str, object]:
    _require(
        len(config.run_id) == 32
        and all(character in "0123456789abcdef" for character in config.run_id),
        "RUN_ID_INVALID",
    )
    _require(
        config.deadline_epoch <= time.time() + MAX_WALL_SECONDS + 60,
        "WATCHDOG_DEADLINE_INVALID",
    )
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN")
    _require(bool(token), "MAPILLARY_ACCESS_TOKEN_MISSING")
    verify_megaloc_artifacts(
        config.model_path,
        config.vendor_root / "megaloc_model.py",
        config.vendor_root / "LICENSE",
    )
    _require(
        _sha256_path(config.source_policy_path) == SOURCE_POLICY_SHA256,
        "SOURCE_POLICY_SHA256_MISMATCH",
    )
    areas = load_city_areas(config.aoi_config_path)
    if config.resume:
        _require(
            config.work_root.is_dir() and not config.work_root.is_symlink(),
            "RESUME_WORK_ROOT_INVALID",
        )
    else:
        _require(
            not config.work_root.exists() and not config.work_root.is_symlink(),
            "FRESH_WORK_ROOT_ALREADY_EXISTS",
        )
        config.work_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(config.work_root, 0o700)
    config.output_root.mkdir(parents=True, exist_ok=False)
    media_root = config.work_root / "private-media"
    checkpoints = config.work_root / "descriptor-checkpoints"
    state_path = config.output_root / "pipeline-state.json"
    acquisition_checkpoint = config.work_root / "acquisition-checkpoint.json"
    client_counter_checkpoint = config.work_root / "client-counters.json"
    metadata_page_checkpoint = config.work_root / "metadata-pages.json"
    _require(
        (
            config.resume
            and client_counter_checkpoint.is_file()
            and metadata_page_checkpoint.is_file()
        )
        or (
            not config.resume
            and not acquisition_checkpoint.exists()
            and not client_counter_checkpoint.exists()
            and not metadata_page_checkpoint.exists()
        ),
        "RESUME_CHECKPOINT_MISSING",
    )
    pipeline = Phase3FPipeline.create(state_path, run_id=config.run_id)
    guard = AcquisitionGuard()
    limits = ClientLimits(
        request_cap=MAX_REQUESTS,
        page_cap=2_000,
        metadata_item_cap=20_000,
        page_size=100,
        raw_download_byte_cap=2 * 1024 * 1024 * 1024,
        image_cap=2_000,
        max_image_bytes=16 * 1024 * 1024,
        concurrency=2,
        timeout_seconds=20.0,
        retry_cap=3,
        backoff_base_seconds=0.5,
        backoff_cap_seconds=8.0,
    )
    runtime: MegaLocRuntime | None = None
    cleanup_private_work = False
    historical_counts = (
        _read_client_counters(client_counter_checkpoint, config.run_id)
        if config.resume
        else (0, 0, 0)
    )
    metadata_checkpoint = (
        _load_metadata_page_checkpoint(
            metadata_page_checkpoint,
            areas=areas,
            run_id=config.run_id,
        )
        if config.resume
        else _new_metadata_page_checkpoint(areas, config.run_id)
    )
    if not config.resume:
        _write_metadata_page_checkpoint(metadata_page_checkpoint, metadata_checkpoint)
        _write_client_counters(client_counter_checkpoint, config.run_id, 0, 0, 0)
    started = datetime.now(UTC)
    try:
        with MapillaryClient(
            cast(str, token),
            limits=limits,
            counter_observer=lambda requests, pages, rejected: _write_client_counters(
                client_counter_checkpoint,
                config.run_id,
                requests,
                pages,
                rejected,
            ),
        ) as client:
            if config.resume:
                client.add_historical_counts(
                    request_count=historical_counts[0],
                    page_count=historical_counts[1],
                    rejected_item_count=historical_counts[2],
                )
            _check_deadline(config.deadline_epoch)
            try:
                audit = audit_metadata(
                    client,
                    areas,
                    checkpoint=metadata_checkpoint,
                    checkpoint_path=metadata_page_checkpoint,
                    run_id=config.run_id,
                    progress=lambda city, page_count, stage: print(
                        json.dumps(
                            {
                                "event": "PHASE3F_MAPILLARY_METADATA_PROGRESS",
                                "region": city,
                                "page_count": page_count,
                                "stage": stage,
                                "secrets_included": False,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        flush=True,
                    ),
                )
            except CoverageInsufficient:
                pipeline.finalize_coverage_insufficient(reason_code="COVERAGE_INSUFFICIENT")
                execution = {
                    "schema": "atlaslens-phase3f-cloud-execution-v1",
                    "run_id": config.run_id,
                    "outcome": "COVERAGE_INSUFFICIENT",
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "mapillary_request_count": client.request_count,
                    "downloaded_image_count": 0,
                    "downloaded_media_bytes": 0,
                    "secrets_included": False,
                    "raw_images_included": False,
                }
                _atomic_json(config.output_root / "execution-receipt.json", execution)
                _atomic_json(
                    config.output_root / "checksum-inventory.json",
                    _inventory_output(config.output_root),
                )
                cleanup_private_work = True
                return execution
            _atomic_json(config.output_root / "metadata-audit.json", audit.aggregate_document())
            pipeline.lock_selection(audit.selection)
            split_plan = plan_metadata_split(audit)
            _atomic_json(
                config.output_root / "training-readiness.json",
                split_plan.readiness,
            )
            if not split_plan.ready:
                pipeline.finalize_coverage_insufficient(reason_code="SPLIT_MINIMUM_UNAVAILABLE")
                execution = {
                    "schema": "atlaslens-phase3f-cloud-execution-v1",
                    "run_id": config.run_id,
                    "outcome": "COVERAGE_INSUFFICIENT",
                    "reason_code": "SPLIT_MINIMUM_UNAVAILABLE",
                    "readiness_reported": True,
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "mapillary_request_count": client.request_count,
                    "downloaded_image_count": 0,
                    "downloaded_media_bytes": 0,
                    "secrets_included": False,
                    "raw_images_included": False,
                }
                _atomic_json(config.output_root / "execution-receipt.json", execution)
                _atomic_json(
                    config.output_root / "checksum-inventory.json",
                    _inventory_output(config.output_root),
                )
                cleanup_private_work = True
                return execution
            planned = split_plan.download_assets
            assets, provenance = acquire_planned_assets(
                client,
                areas,
                planned,
                media_root,
                config.run_id,
                guard,
                acquisition_checkpoint,
                restore_client_counts=False,
                allow_item_failures=True,
                primary_count=len(split_plan.primary),
            )
            _require(client.request_count <= MAX_REQUESTS, "REQUEST_CAP_EXCEEDED")
            media_status = provenance.get("media_status")
            _require(isinstance(media_status, str), "MEDIA_ACQUISITION_STATUS_INVALID")
            media_status = cast(str, media_status)
            if media_status.startswith("PAUSED_"):
                receipt = {
                    "run_id": config.run_id,
                    "stage": media_status,
                    "media_task_counts": provenance.get("media_task_counts"),
                    "retry_not_before": provenance.get("retry_not_before"),
                    "model_loaded": False,
                    "gpu_used": False,
                    "cloud_mutations": 0,
                    "secrets_included": False,
                }
                _atomic_json(config.output_root / "execution-receipt.json", receipt)
                return receipt
            pipeline.complete_acquisition(guard)
            media_plan = finalize_media_split(split_plan, assets)
            _atomic_json(
                config.output_root / "training-readiness.json",
                media_plan.readiness,
            )
            if not media_plan.ready:
                pipeline.finalize_dataset_not_ready(reason_code="MEDIA_SPLIT_MINIMUM_UNAVAILABLE")
                execution = {
                    "schema": "atlaslens-phase3f-cloud-execution-v1",
                    "run_id": config.run_id,
                    "outcome": "COVERAGE_INSUFFICIENT",
                    "reason_code": "MEDIA_SPLIT_MINIMUM_UNAVAILABLE",
                    "readiness_reported": True,
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "mapillary_request_count": client.request_count,
                    "downloaded_image_count": len(assets),
                    "downloaded_media_bytes": guard.media_bytes,
                    "secrets_included": False,
                    "raw_images_included": False,
                }
                _atomic_json(config.output_root / "execution-receipt.json", execution)
                _atomic_json(
                    config.output_root / "checksum-inventory.json",
                    _inventory_output(config.output_root),
                )
                cleanup_private_work = True
                return execution
            assets = media_plan.assets
            provenance = finalize_provenance_aggregate(provenance, assets, media_root)
            split = seal_split(
                assets,
                in_domain_cities=[item.city for item in audit.selection.in_domain],
                ood_cities=[item.city for item in audit.selection.ood],
            )
            pipeline.record_split(split)
            _atomic_json(config.output_root / "selection-lock.json", audit.selection.document())
            _atomic_json(config.output_root / "split-lock.json", split.inference_document())
            _atomic_json(config.output_root / "provenance-aggregate.json", provenance)
            _check_deadline(config.deadline_epoch)
            runtime = MegaLocRuntime(config.model_path, config.vendor_root)
            references = tuple(item for item in assets if item.role == "reference")
            calibration = tuple(item for item in assets if item.role == "calibration")
            reference_matrix = _descriptor_shards(
                runtime,
                references,
                media_root,
                checkpoints / "reference",
            )
            calibration_matrix = _descriptor_shards(
                runtime,
                calibration,
                media_root,
                checkpoints / "calibration",
            )
            index, descriptor_sha, index_sha, publication_sha = _publish_reference_bundle(
                config.output_root,
                references,
                reference_matrix,
            )
            publication = DescriptorPublication(
                selection_lock_sha256=audit.selection.lock_sha256,
                split_lock_sha256=split.split_lock_sha256,
                source_policy_sha256=SOURCE_POLICY_SHA256,
                descriptor_publication_sha256=publication_sha,
                index_sha256=index_sha,
                city_scope=tuple(item.city for item in audit.selection.in_domain),
                created_at=datetime.now(UTC),
            )
            pipeline.record_descriptor_publication(publication)
            calibration_rows = _retrieval_rows(
                index,
                calibration,
                calibration_matrix,
                references,
                publication.city_scope,
            )
            threshold = fit_abstention_threshold(
                [
                    CalibrationObservation(
                        row.signals,
                        row.predicted_cities[0] == row.city,
                    )
                    for row in calibration_rows
                ],
                selection_lock_sha256=audit.selection.lock_sha256,
                split_lock_sha256=split.split_lock_sha256,
            )
            pipeline.lock_threshold(threshold)
            _atomic_json(config.output_root / "calibration.json", threshold.document())
            _check_deadline(config.deadline_epoch)
            holdout = tuple(
                item for item in assets if item.role in {"sealed_holdout", "ood_holdout"}
            )
            holdout_matrix = _descriptor_shards(
                runtime,
                holdout,
                media_root,
                checkpoints / "holdout",
            )
            holdout_rows = _retrieval_rows(
                index,
                holdout,
                holdout_matrix,
                references,
                publication.city_scope,
            )
            benchmark = evaluate_holdout_once(
                holdout_rows,
                threshold=threshold,
                selection_lock_sha256=audit.selection.lock_sha256,
                split_lock_sha256=split.split_lock_sha256,
                in_domain_cities=publication.city_scope,
                ood_cities=tuple(item.city for item in audit.selection.ood),
                leakage_passed=split.leakage.passed,
                city_minimums_passed=True,
                security_integrity_passed=True,
                holdout_open_count_before=0,
            )
            pipeline.record_benchmark(benchmark)
            pipeline.finalize()
            _atomic_json(config.output_root / "aggregate-benchmark.json", benchmark.document())
            _atomic_json(
                config.output_root / "descriptor-publication.json",
                {
                    **publication.document(),
                    "reference_descriptor_sha256": descriptor_sha,
                },
            )
            execution = {
                "schema": "atlaslens-phase3f-cloud-execution-v1",
                "run_id": config.run_id,
                "outcome": benchmark.outcome,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "mapillary_request_count": client.request_count,
                "downloaded_image_count": guard.image_count,
                "downloaded_media_bytes": guard.media_bytes,
                "private_exact_coordinates_in_derived_artifacts": True,
                "raw_coordinates_in_execution_receipt": False,
                "secrets_included": False,
                "raw_images_included": False,
            }
            _atomic_json(config.output_root / "execution-receipt.json", execution)
            inventory = _inventory_output(config.output_root)
            _atomic_json(config.output_root / "checksum-inventory.json", inventory)
            cleanup_private_work = True
            return execution
    finally:
        if runtime is not None:
            runtime.close()
        if cleanup_private_work and media_root.exists():
            shutil.rmtree(media_root)
        if cleanup_private_work and checkpoints.exists():
            shutil.rmtree(checkpoints)
        if cleanup_private_work and acquisition_checkpoint.exists():
            acquisition_checkpoint.unlink()
        if cleanup_private_work and client_counter_checkpoint.exists():
            client_counter_checkpoint.unlink()
        if cleanup_private_work and metadata_page_checkpoint.exists():
            metadata_page_checkpoint.unlink()


__all__ = [
    "CityArea",
    "CloudJobConfig",
    "MetadataAsset",
    "MetadataAudit",
    "MetadataSplitPlan",
    "MediaSplitPlan",
    "Phase3FCloudJobError",
    "PlannedAsset",
    "acquire_planned_assets",
    "audit_metadata",
    "finalize_media_split",
    "finalize_provenance_aggregate",
    "load_city_areas",
    "plan_locked_roles",
    "plan_metadata_split",
    "quarter_bbox",
    "run_cloud_job",
    "verify_megaloc_artifacts",
]
