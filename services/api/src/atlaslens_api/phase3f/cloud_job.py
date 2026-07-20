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
from datetime import UTC, datetime
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
    MapillaryTokenError,
)
from atlaslens_api.mapillary_demo.models import BoundingBox, ClientLimits, ImageMetadata
from atlaslens_api.phase3f.acquisition import (
    MAX_IMAGES,
    MAX_MEDIA_BYTES,
    MAX_REQUESTS,
    AcquisitionGuard,
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
from atlaslens_api.phase3f.splits import SplitAsset, seal_split

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
SOURCE_POLICY_SHA256: Final = (
    "72d51363f2b63de368d34d4d7bb2fc1145f93dc0469e7026100976732dfde209"
)
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
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
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
    return {
        area.city: [_PlannedCell(box, 0) for box in area.boxes]
        for area in areas
    }


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
        if (
            leaf.depth == expected_depth
            and leaf.bbox.as_query_value() == expected.as_query_value()
        ):
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
                        isinstance(raw_cell, dict)
                        and set(raw_cell) == {"bbox", "depth"},
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
                    and len({cell.bbox.as_query_value() for cell in city_cells})
                    == len(city_cells),
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
        sum(len(rows) for rows in checkpoint.rows_by_city.values())
        <= MAX_METADATA_ITEMS_TOTAL,
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
                admitted = page.images[
                    : min(remaining_city_quota, remaining_global_quota)
                ]
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
                    len(checkpoint.cells_by_city[city])
                    if city_quota_reached
                    else page.box_index
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
            item
            for item in assets
            if item.creator_id is not None and item.sequence_id is not None
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


def _take_from_creator(
    assets: Sequence[MetadataAsset],
    *,
    count: int,
    forbidden_creators: set[str],
    forbidden_sequences: set[str],
    separated_from: Sequence[MetadataAsset] = (),
) -> tuple[MetadataAsset, ...]:
    creators: dict[str, list[MetadataAsset]] = defaultdict(list)
    for asset in assets:
        if asset.creator_id is None or asset.sequence_id is None:
            continue
        if asset.creator_id in forbidden_creators or asset.sequence_id in forbidden_sequences:
            continue
        if any(_distance_m(asset, locked) < 1_000.0 for locked in separated_from):
            continue
        creators[asset.creator_id].append(asset)
    ordered_creators = sorted(
        creators,
        key=lambda creator: _stable_key(assets[0].city if assets else "", creator),
    )
    selected: list[MetadataAsset] = []
    used_creators: set[str] = set()
    for creator in ordered_creators:
        candidates = sorted(
            creators[creator],
            key=lambda item: _stable_key(item.city, item.sequence_id or "", item.image_id),
        )
        for candidate in candidates:
            if candidate.sequence_id in forbidden_sequences:
                continue
            selected.append(candidate)
            used_creators.add(creator)
            if len(selected) == count:
                forbidden_creators.update(used_creators)
                forbidden_sequences.update(
                    item.sequence_id for item in selected if item.sequence_id is not None
                )
                return tuple(selected)
    raise Phase3FCloudJobError("SPLIT_MINIMUM_UNAVAILABLE")


def plan_locked_roles(audit: MetadataAudit) -> tuple[PlannedAsset, ...]:
    planned: list[PlannedAsset] = []
    forbidden_creators: set[str] = set()
    forbidden_sequences: set[str] = set()
    for record in audit.selection.in_domain:
        pool = audit.assets_by_city[record.city]
        holdout = _take_from_creator(
            pool,
            count=25,
            forbidden_creators=forbidden_creators,
            forbidden_sequences=forbidden_sequences,
        )
        calibration = _take_from_creator(
            pool,
            count=25,
            forbidden_creators=forbidden_creators,
            forbidden_sequences=forbidden_sequences,
        )
        reference = _take_from_creator(
            pool,
            count=75,
            forbidden_creators=forbidden_creators,
            forbidden_sequences=forbidden_sequences,
            separated_from=holdout,
        )
        planned.extend(PlannedAsset(item, "reference") for item in reference)
        planned.extend(PlannedAsset(item, "calibration") for item in calibration)
        planned.extend(PlannedAsset(item, "sealed_holdout") for item in holdout)
    for record in audit.selection.ood:
        ood = _take_from_creator(
            audit.assets_by_city[record.city],
            count=40,
            forbidden_creators=forbidden_creators,
            forbidden_sequences=forbidden_sequences,
        )
        planned.extend(PlannedAsset(item, "ood_holdout") for item in ood)
    _require(len(planned) <= MAX_IMAGES, "IMAGE_CAP_EXCEEDED")
    return tuple(planned)


def _verify_canonical_vendor_text(
    path: Path,
    *,
    canonical_lf_sha256: str,
    windows_crlf_sha256: str,
    max_bytes: int,
) -> None:
    _require(
        path.is_file()
        and not path.is_symlink()
        and 0 < path.stat().st_size <= max_bytes,
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
                    max(image.size) <= 1024
                    and image.width * image.height <= 1024 * 1024,
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
                phash = (
                    f"{int(''.join('1' if bit else '0' for bit in bits.flat), 2):016x}"
                )
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


def _load_acquisition_checkpoint(
    path: Path,
    *,
    planned: Sequence[PlannedAsset],
    media_root: Path,
    run_id: str,
) -> tuple[
    dict[str, _CheckpointAsset], AcquisitionGuard, tuple[int, int, int]
]:
    if not path.exists():
        _require(
            not media_root.exists()
            or not any(media_root.rglob("*")),
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
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in counters
        ),
        "ACQUISITION_CHECKPOINT_INVALID",
    )
    wanted_by_opaque = {
        _stable_key(run_id, item.metadata.image_id)[:32]: item.metadata.image_id
        for item in planned
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
        all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in counters
        )
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
        all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in counters
        )
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
            isinstance(raw, dict)
            and set(raw) == {"image_id", "reason_code"},
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
) -> tuple[tuple[SplitAsset, ...], dict[str, object]]:
    _require(0 < max_media_bytes <= MAX_MEDIA_BYTES, "MEDIA_CAP_INVALID")
    wanted = {item.metadata.image_id: item for item in planned}
    _require(len(wanted) == len(planned), "PLANNED_IMAGE_DUPLICATE")
    completed: dict[str, SplitAsset] = {}
    provenance_hashes: list[str] = []
    checkpoint_assets: dict[str, _CheckpointAsset] = {}
    media_failure_path = (
        checkpoint_path.with_name("media-failures.json")
        if checkpoint_path is not None
        else None
    )
    media_failures = (
        _load_media_failures(
            media_failure_path,
            run_id=run_id,
            planned=planned,
        )
        if media_failure_path is not None
        else {}
    )
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
        provenance_hashes.extend(
            item.provenance_sha256 for item in checkpoint_assets.values()
        )
        guard.request_count = restored_guard.request_count
        guard.image_count = restored_guard.image_count
        guard.media_bytes = restored_guard.media_bytes
    for area in areas:
        city_wanted = {
            image_id for image_id, item in wanted.items() if item.metadata.city == area.city
        }
        if not city_wanted:
            continue
        for remote in client.iter_images(
            area.boxes,
            include_thumbnail=True,
            item_cap=MAX_METADATA_ITEMS_PER_CITY,
        ):
            image_id = remote.metadata.mapillary_image_id
            if image_id not in city_wanted or image_id in completed:
                continue
            if image_id in media_failures:
                continue
            plan = wanted[image_id]
            prior = checkpoint_assets.get(image_id)
            if prior is not None:
                lon, lat = remote.metadata.computed_geometry.coordinates
                completed[image_id] = SplitAsset(
                    opaque_id=prior.opaque_id,
                    city=plan.metadata.city,
                    role=plan.role,
                    relative_path=(
                        Path("assets")
                        / prior.opaque_id[:2]
                        / f"{prior.opaque_id}.jpg"
                    ).as_posix(),
                    contributor_id=remote.metadata.creator_id or "missing",
                    sequence_id=remote.metadata.sequence_id or "missing",
                    capture_run_id=remote.metadata.sequence_id or "missing",
                    content_sha256=prior.content_sha256,
                    perceptual_hash=prior.perceptual_hash,
                    parent_or_tile_id=image_id,
                    longitude=lon,
                    latitude=lat,
                )
                if len(completed) == len(planned):
                    break
                continue
            _require(remote.thumbnail_url is not None, "THUMBNAIL_URL_MISSING")
            thumbnail_url = remote.thumbnail_url
            if thumbnail_url is None:  # pragma: no cover - narrowed above
                raise Phase3FCloudJobError("THUMBNAIL_URL_MISSING")
            guard.begin_request()
            try:
                payload, _mime = client.download_thumbnail(
                    thumbnail_url.get_secret_value(),
                    max_bytes=16 * 1024 * 1024,
                )
                _require(
                    guard.media_bytes + len(payload) <= max_media_bytes,
                    "MEDIA_CAP_EXCEEDED",
                )
                normalized, _width, _height, phash = _normalize_image(payload)
                source_page = f"https://www.mapillary.com/app/?pKey={image_id}"
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
                    mapillary_image_id=image_id,
                    source_page=source_page,
                    contributor_attribution=(
                        remote.metadata.creator_id or "Mapillary contributor"
                    ),
                    capture_date=remote.metadata.captured_at,
                    source_policy_receipt_sha256=receipt_sha,
                    revoked=False,
                )
                sidecar.validate()
                privacy_review = PrivacyReview(
                    passed=True,
                    face_reidentification_performed=False,
                    plate_reidentification_performed=False,
                    raw_ocr_generated=False,
                )
                privacy_review.validate()
            except MapillaryTokenError:
                guard.finish_request(status_code=500, media_bytes=0, admitted_image=False)
                raise
            except (MapillaryApiError, MapillaryLimitError, Phase3FCloudJobError) as exc:
                guard.finish_request(status_code=500, media_bytes=0, admitted_image=False)
                code = exc.code
                if code in {
                    "mapillary_media_rate_limit_retry_exhausted",
                    "mapillary_request_cap_reached",
                    "mapillary_operation_cancelled",
                    "MEDIA_CAP_EXCEEDED",
                }:
                    raise
                if not allow_item_failures:
                    raise
                media_failures[image_id] = code.lower()
                if media_failure_path is not None:
                    _write_media_failures(
                        media_failure_path,
                        run_id=run_id,
                        planned=planned,
                        failures=media_failures,
                    )
                    if checkpoint_observer is not None:
                        checkpoint_observer()
                continue
            guard.finish_request(
                status_code=200,
                media_bytes=len(payload),
                admitted_image=True,
                provenance=sidecar,
                privacy_review=privacy_review,
            )
            opaque_id = _stable_key(run_id, image_id)[:32]
            relative = Path("assets") / opaque_id[:2] / f"{opaque_id}.jpg"
            destination = media_root / relative
            _atomic_private_bytes(destination, normalized)
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
            lon, lat = remote.metadata.computed_geometry.coordinates
            completed[image_id] = SplitAsset(
                opaque_id=opaque_id,
                city=plan.metadata.city,
                role=plan.role,
                relative_path=relative.as_posix(),
                contributor_id=remote.metadata.creator_id or "missing",
                sequence_id=remote.metadata.sequence_id or "missing",
                capture_run_id=remote.metadata.sequence_id or "missing",
                content_sha256=_sha256_bytes(normalized),
                perceptual_hash=phash,
                parent_or_tile_id=image_id,
                longitude=lon,
                latitude=lat,
            )
            checkpoint_assets[image_id] = _CheckpointAsset(
                opaque_id=opaque_id,
                content_sha256=_sha256_bytes(normalized),
                perceptual_hash=phash,
                provenance_sha256=provenance_sha,
                admitted_media_bytes=len(payload),
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
            if len(completed) == len(planned):
                break
    if not allow_item_failures:
        _require(len(completed) == len(planned), "PLANNED_IMAGE_UNAVAILABLE")
    _require(guard.media_bytes <= max_media_bytes, "MEDIA_CAP_EXCEEDED")
    return (
        tuple(completed[item.metadata.image_id] for item in planned),
        {
            "schema": "atlaslens-phase3f-provenance-aggregate-v1",
            "asset_count": len(completed),
            "sidecar_sha256_set_sha256": _sha256_bytes(
                _canonical_bytes(sorted(provenance_hashes))
            ),
            "official_mapillary_graph_api": True,
            "signed_urls_persisted": False,
            "raw_ocr_created": False,
            "reidentification_attempted": False,
            "item_failure_count": len(media_failures),
            "item_failure_reasons": dict(sorted(Counter(media_failures.values()).items())),
        },
    )


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
            grouped[(int(tensor.shape[1]), int(tensor.shape[2]))].append(
                (position, tensor)
            )
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
        identity = _sha256_bytes(
            _canonical_bytes([item.opaque_id for item in batch])
        )
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
                pipeline.finalize_coverage_insufficient(
                    reason_code="COVERAGE_INSUFFICIENT"
                )
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
            try:
                planned = plan_locked_roles(audit)
            except Phase3FCloudJobError as exc:
                if exc.code != "SPLIT_MINIMUM_UNAVAILABLE":
                    raise
                pipeline.finalize_coverage_insufficient(
                    reason_code="SPLIT_MINIMUM_UNAVAILABLE"
                )
                execution = {
                    "schema": "atlaslens-phase3f-cloud-execution-v1",
                    "run_id": config.run_id,
                    "outcome": "COVERAGE_INSUFFICIENT",
                    "reason_code": exc.code,
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
            assets, provenance = acquire_planned_assets(
                client,
                areas,
                planned,
                media_root,
                config.run_id,
                guard,
                acquisition_checkpoint,
                restore_client_counts=False,
            )
            _require(client.request_count <= MAX_REQUESTS, "REQUEST_CAP_EXCEEDED")
            pipeline.complete_acquisition(guard)
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
                item
                for item in assets
                if item.role in {"sealed_holdout", "ood_holdout"}
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
    "Phase3FCloudJobError",
    "PlannedAsset",
    "acquire_planned_assets",
    "audit_metadata",
    "load_city_areas",
    "plan_locked_roles",
    "quarter_bbox",
    "run_cloud_job",
    "verify_megaloc_artifacts",
]
