"""Local-first Phase 3F acquisition and offline CUDA compute stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import stat
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import httpx

from atlaslens_api.mapillary_demo.client import MapillaryClient
from atlaslens_api.mapillary_demo.models import (
    MAPILLARY_API_BASE_URL,
    MAPILLARY_API_FIELDS,
    BoundingBox,
    ClientLimits,
)
from atlaslens_api.phase3f import cloud_job as worker
from atlaslens_api.phase3f.acquisition import MAX_REQUESTS, AcquisitionGuard
from atlaslens_api.phase3f.benchmark import (
    CalibrationObservation,
    evaluate_holdout_once,
    fit_abstention_threshold,
)
from atlaslens_api.phase3f.coverage import (
    CityCoverageRecord,
    CoverageInsufficient,
    CoverageSelectionLock,
)
from atlaslens_api.phase3f.pipeline import DescriptorPublication, Phase3FPipeline
from atlaslens_api.phase3f.scheduler import sync_scheduler_checkpoint
from atlaslens_api.phase3f.splits import SealedSplit, SplitAsset, seal_split

LOCAL_RUNTIME_CAP_BYTES: Final = 8 * 1024 * 1024 * 1024
LOCAL_MEDIA_CAP_BYTES: Final = 7 * 1024 * 1024 * 1024
LOCAL_MAX_WALL_SECONDS: Final = 12 * 60 * 60
LOCAL_STATE_SCHEMA: Final = "atlaslens-phase3f-local-first-state-v1"
LOCAL_CURRENT_SCHEMA: Final = "atlaslens-phase3f-local-first-current-v1"
SEALED_BUNDLE_SCHEMA: Final = "atlaslens-phase3f-sealed-acquisition-v1"
SEALED_ASSETS_SCHEMA: Final = "atlaslens-phase3f-sealed-assets-v1"
ACQUISITION_DIAGNOSTIC_SCHEMA: Final = "atlaslens-phase3f-acquisition-diagnostic-v1"
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUNPOD_SECRET_REFERENCE = re.compile(
    r"^\{\{\s*RUNPOD_SECRET_[A-Za-z0-9_.-]+\s*\}\}$"
)
_TERMINAL_ACQUISITION_ERRORS: Final = frozenset(
    {
        "MAPILLARY_PARTITION_DEPTH_LIMIT_REACHED",
        "MAPILLARY_PARTITION_MIN_AREA_REACHED",
        "MAPILLARY_PARTITION_CELL_LIMIT_REACHED",
        "MAPILLARY_GLOBAL_METADATA_QUOTA_REACHED",
        "MAPILLARY_METADATA_QUOTA_INVARIANT_FAILED",
    }
)
_AUTO_CONTINUE_METADATA_FAILURES: Final = frozenset(
    {
        "MAPILLARY_API_SERVER_RETRY_EXHAUSTED",
        "MAPILLARY_API_TIMEOUT",
        "MAPILLARY_API_TRANSPORT_RETRY_EXHAUSTED",
        "MAPILLARY_PAGING_CELL_QUARANTINED",
        "MAPILLARY_PARTITION_CELL_QUARANTINED",
    }
)
_QUARANTINE_METADATA_FAILURES: Final = frozenset(
    {
        "MAPILLARY_API_PAGE_INVALID",
        "MAPILLARY_PAGING_LOOP_DETECTED",
        "MAPILLARY_PAGING_URL_INVALID",
        "MAPILLARY_PAGING_QUERY_DUPLICATE",
    }
)
MAX_CONSECUTIVE_PROVIDER_FAILURES: Final = 8


class LocalFirstError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _acquisition_failure_stage(code: str) -> str:
    if code in _TERMINAL_ACQUISITION_ERRORS:
        return "ACQUISITION_FAILED_TERMINAL"
    return "ACQUISITION_FAILED_RESUMABLE"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise LocalFirstError(code)


def _is_reparse(path: Path) -> bool:
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _safe_existing_directory(path: Path, code: str) -> Path:
    _require(path.is_dir() and not _is_reparse(path), code)
    resolved = path.resolve()
    _require(resolved.is_dir() and not _is_reparse(resolved), code)
    return resolved


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _tree_size(path: Path, *, max_bytes: int | None = LOCAL_RUNTIME_CAP_BYTES) -> int:
    total = 0
    if not path.exists():
        return 0
    _require(path.is_dir() and not _is_reparse(path), "LOCAL_RUNTIME_PATH_INVALID")
    for item in path.rglob("*"):
        _require(not _is_reparse(item), "LOCAL_RUNTIME_REPARSE_REFUSED")
        if item.is_file():
            total += item.stat().st_size
            _require(
                max_bytes is None or total <= max_bytes,
                "LOCAL_RUNTIME_CAP_EXCEEDED",
            )
    return total


def _remove_owned_tree(path: Path, *, parent: Path, code: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    _require(
        path.parent.resolve() == parent.resolve()
        and path.is_dir()
        and not _is_reparse(path),
        code,
    )
    for item in path.rglob("*"):
        _require(not _is_reparse(item), code)
    shutil.rmtree(path)


def _atomic_private_json(path: Path, value: object) -> None:
    worker._atomic_private_json(path, value)  # noqa: SLF001


def _read_json(path: Path, *, max_bytes: int, code: str) -> dict[str, object]:
    _require(
        path.is_file() and not _is_reparse(path) and path.stat().st_size <= max_bytes,
        code,
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalFirstError(code) from exc
    _require(isinstance(value, dict), code)
    return {str(key): item for key, item in cast(dict[object, object], value).items()}


def _state_path(runtime_root: Path, run_id: str) -> Path:
    _require(bool(_RUN_ID.fullmatch(run_id)), "LOCAL_RUN_ID_INVALID")
    run_root = (runtime_root / run_id).resolve()
    _require(run_root.parent == runtime_root, "LOCAL_RUN_ROOT_INVALID")
    return run_root / "state.json"


def _write_state(
    runtime_root: Path,
    run_id: str,
    stage: str,
    **updates: object,
) -> None:
    path = _state_path(runtime_root, run_id)
    existing: dict[str, object] = {}
    if path.exists():
        existing = _read_json(path, max_bytes=1_048_576, code="LOCAL_STATE_INVALID")
        _require(existing.get("run_id") == run_id, "LOCAL_STATE_INVALID")
    document = {
        **existing,
        "schema": LOCAL_STATE_SCHEMA,
        "run_id": run_id,
        "stage": stage,
        "updated_at": datetime.now(UTC).isoformat(),
        "secrets_included": False,
        **updates,
    }
    _atomic_private_json(path, document)


def _current_run(runtime_root: Path) -> str:
    document = _read_json(
        runtime_root / "current.json",
        max_bytes=4096,
        code="LOCAL_CURRENT_RUN_MISSING",
    )
    run_id = document.get("run_id")
    _require(
        document.get("schema") == LOCAL_CURRENT_SCHEMA
        and isinstance(run_id, str)
        and bool(_RUN_ID.fullmatch(run_id)),
        "LOCAL_CURRENT_RUN_INVALID",
    )
    return cast(str, run_id)


def _new_run(runtime_root: Path) -> str:
    current = runtime_root / "current.json"
    _require(not current.exists() and not current.is_symlink(), "LOCAL_CURRENT_RUN_EXISTS")
    run_id = secrets.token_hex(16)
    run_root = runtime_root / run_id
    run_root.mkdir(mode=0o700, exist_ok=False)
    os.chmod(run_root, 0o700)
    _atomic_private_json(current, {"schema": LOCAL_CURRENT_SCHEMA, "run_id": run_id})
    _write_state(runtime_root, run_id, "ACQUISITION_CREATED")
    return run_id


def _mapillary_token() -> str:
    value = os.environ.get("MAPILLARY_ACCESS_TOKEN")
    if value is None or value == "":
        raise LocalFirstError("MAPILLARY_ACCESS_TOKEN_MISSING")
    if _RUNPOD_SECRET_REFERENCE.fullmatch(value):
        raise LocalFirstError("UNRESOLVED_RUNPOD_SECRET_REFERENCE")
    _require(value.startswith("MLY"), "MAPILLARY_ACCESS_TOKEN_INVALID_FORMAT")
    return value


def _inventory(root: Path, *, max_bytes: int) -> dict[str, object]:
    files: list[dict[str, object]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        _require(not _is_reparse(path), "SEALED_BUNDLE_REPARSE_REFUSED")
        if not path.is_file() or path.name == "checksum-inventory.json":
            continue
        size = path.stat().st_size
        total += size
        _require(total <= max_bytes, "SEALED_BUNDLE_SIZE_CAP_EXCEEDED")
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": worker._sha256_path(path),  # noqa: SLF001
            }
        )
    return {
        "schema": "atlaslens-phase3f-checksum-inventory-v1",
        "total_size_bytes": total,
        "files": files,
    }


def _selection_from_document(document: Mapping[str, object]) -> CoverageSelectionLock:
    def records(key: str) -> tuple[CityCoverageRecord, ...]:
        raw = document.get(key)
        _require(isinstance(raw, list), "SEALED_SELECTION_INVALID")
        result: list[CityCoverageRecord] = []
        for value in cast(list[object], raw):
            _require(isinstance(value, dict), "SEALED_SELECTION_INVALID")
            row = cast(dict[str, object], value)
            try:
                result.append(
                    CityCoverageRecord(
                        city=cast(str, row["city"]),
                        macro_region=cast(str, row["macro_region"]),
                        image_count=cast(int, row["image_count"]),
                        sequence_count=cast(int, row["sequence_count"]),
                        contributor_count=cast(int, row["contributor_count"]),
                        spatial_cell_count=cast(int, row["spatial_cell_count"]),
                        capture_year_count=cast(int, row["capture_year_count"]),
                        eligible_asset_count=cast(int, row["eligible_asset_count"]),
                        metadata_sha256=cast(str, row["metadata_sha256"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LocalFirstError("SEALED_SELECTION_INVALID") from exc
        return tuple(result)

    try:
        selection = CoverageSelectionLock(
            in_domain=records("in_domain"),
            ood=records("ood"),
            audited_city_count=cast(int, document["audited_city_count"]),
            policy_sha256=cast(str, document["policy_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalFirstError("SEALED_SELECTION_INVALID") from exc
    _require(selection.document() == dict(document), "SEALED_SELECTION_INVALID")
    return selection


def _assets_from_document(document: Mapping[str, object]) -> tuple[SplitAsset, ...]:
    _require(document.get("schema") == SEALED_ASSETS_SCHEMA, "SEALED_ASSETS_INVALID")
    rows = document.get("assets")
    _require(isinstance(rows, list), "SEALED_ASSETS_INVALID")
    assets: list[SplitAsset] = []
    try:
        for value in cast(list[object], rows):
            _require(isinstance(value, dict), "SEALED_ASSETS_INVALID")
            assets.append(SplitAsset(**cast(dict[str, Any], value)))
    except (TypeError, ValueError) as exc:
        raise LocalFirstError("SEALED_ASSETS_INVALID") from exc
    _require(len(assets) == len({item.opaque_id for item in assets}), "SEALED_ASSETS_INVALID")
    return tuple(assets)


def _validate_inventory(root: Path) -> dict[str, object]:
    inventory = _read_json(
        root / "checksum-inventory.json",
        max_bytes=32 * 1024 * 1024,
        code="SEALED_INVENTORY_INVALID",
    )
    rows = inventory.get("files")
    _require(
        inventory.get("schema") == "atlaslens-phase3f-checksum-inventory-v1"
        and isinstance(rows, list),
        "SEALED_INVENTORY_INVALID",
    )
    expected: set[str] = set()
    total = 0
    for value in cast(list[object], rows):
        _require(isinstance(value, dict), "SEALED_INVENTORY_INVALID")
        row = cast(dict[str, object], value)
        relative = row.get("relative_path")
        size = row.get("size_bytes")
        sha256 = row.get("sha256")
        _require(
            isinstance(relative, str)
            and relative not in expected
            and not relative.startswith("/")
            and ".." not in Path(relative).parts
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size >= 0
            and isinstance(sha256, str)
            and bool(_SHA256.fullmatch(sha256)),
            "SEALED_INVENTORY_INVALID",
        )
        safe_relative = cast(str, relative)
        safe_size = cast(int, size)
        safe_sha256 = cast(str, sha256)
        path = (root / safe_relative).resolve()
        _require(
            path.is_relative_to(root)
            and path.is_file()
            and not _is_reparse(path)
            and path.stat().st_size == safe_size
            and worker._sha256_path(path) == safe_sha256,  # noqa: SLF001
            "SEALED_INVENTORY_MISMATCH",
        )
        expected.add(safe_relative)
        total += safe_size
        _require(total <= LOCAL_RUNTIME_CAP_BYTES, "SEALED_BUNDLE_SIZE_CAP_EXCEEDED")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksum-inventory.json"
    }
    _require(actual == expected, "SEALED_INVENTORY_MISMATCH")
    _require(inventory.get("total_size_bytes") == total, "SEALED_INVENTORY_INVALID")
    return inventory


@dataclass(frozen=True, slots=True)
class VerifiedAcquisition:
    run_id: str
    root: Path
    selection: CoverageSelectionLock
    split: SealedSplit
    receipt: dict[str, object]
    provenance: dict[str, object]


def verify_sealed_acquisition(root: Path) -> VerifiedAcquisition:
    resolved = _safe_existing_directory(root, "SEALED_BUNDLE_MISSING")
    _validate_inventory(resolved)
    manifest = _read_json(
        resolved / "manifest.json",
        max_bytes=1_048_576,
        code="SEALED_MANIFEST_INVALID",
    )
    run_id = manifest.get("run_id")
    _require(
        manifest.get("schema") == SEALED_BUNDLE_SCHEMA
        and isinstance(run_id, str)
        and bool(_RUN_ID.fullmatch(run_id))
        and manifest.get("sealed") is True
        and manifest.get("acquisition_complete") is True
        and manifest.get("compute_started") is False
        and manifest.get("secrets_included") is False,
        "SEALED_MANIFEST_INVALID",
    )
    selection_document = _read_json(
        resolved / "selection-lock.json",
        max_bytes=4 * 1024 * 1024,
        code="SEALED_SELECTION_INVALID",
    )
    selection = _selection_from_document(selection_document)
    assets_document = _read_json(
        resolved / "sealed-assets.json",
        max_bytes=32 * 1024 * 1024,
        code="SEALED_ASSETS_INVALID",
    )
    assets = _assets_from_document(assets_document)
    split = seal_split(
        assets,
        in_domain_cities=[item.city for item in selection.in_domain],
        ood_cities=[item.city for item in selection.ood],
    )
    _require(
        manifest.get("selection_lock_sha256") == selection.lock_sha256
        and manifest.get("split_lock_sha256") == split.split_lock_sha256
        and manifest.get("asset_count") == len(assets),
        "SEALED_MANIFEST_MISMATCH",
    )
    for asset in assets:
        media = (resolved / asset.relative_path).resolve()
        _require(
            media.is_relative_to(resolved)
            and media.is_file()
            and not _is_reparse(media)
            and worker._sha256_path(media) == asset.content_sha256,  # noqa: SLF001
            "SEALED_MEDIA_MISMATCH",
        )
    receipt = _read_json(
        resolved / "acquisition-receipt.json",
        max_bytes=1_048_576,
        code="SEALED_RECEIPT_INVALID",
    )
    provenance = _read_json(
        resolved / "provenance-aggregate.json",
        max_bytes=1_048_576,
        code="SEALED_PROVENANCE_INVALID",
    )
    _require(
        receipt.get("run_id") == run_id
        and receipt.get("secrets_included") is False
        and receipt.get("acquisition_complete") is True,
        "SEALED_RECEIPT_INVALID",
    )
    return VerifiedAcquisition(
        cast(str, run_id),
        resolved,
        selection,
        split,
        receipt,
        provenance,
    )


def _seal_acquisition(
    *,
    run_id: str,
    media_root: Path,
    sealed_root: Path,
    selection: CoverageSelectionLock,
    split: SealedSplit,
    provenance: Mapping[str, object],
    client: MapillaryClient,
    guard: AcquisitionGuard,
) -> None:
    _require(not sealed_root.exists() and not sealed_root.is_symlink(), "SEALED_BUNDLE_EXISTS")
    _require(media_root.is_dir() and not _is_reparse(media_root), "PRIVATE_MEDIA_MISSING")
    assets_document = {
        "schema": SEALED_ASSETS_SCHEMA,
        "assets": [asdict(item) for item in split.assets],
        "private_exact_coordinates": True,
        "secrets_included": False,
    }
    receipt = {
        "schema": "atlaslens-phase3f-acquisition-receipt-v1",
        "run_id": run_id,
        "acquisition_complete": True,
        "mapillary_request_count": client.request_count,
        "mapillary_page_count": client.page_count,
        "rejected_item_count": client.rejected_item_count,
        "image_count": guard.image_count,
        "media_bytes": guard.media_bytes,
        "secrets_included": False,
        "signed_urls_included": False,
        "model_loaded": False,
        "gpu_used": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest = {
        "schema": SEALED_BUNDLE_SCHEMA,
        "run_id": run_id,
        "sealed": True,
        "acquisition_complete": True,
        "compute_started": False,
        "selection_lock_sha256": selection.lock_sha256,
        "split_lock_sha256": split.split_lock_sha256,
        "source_policy_sha256": worker.SOURCE_POLICY_SHA256,
        "asset_count": len(split.assets),
        "secrets_included": False,
        "signed_urls_included": False,
        "model_loaded": False,
    }
    _atomic_private_json(media_root / "selection-lock.json", selection.document())
    _atomic_private_json(media_root / "sealed-assets.json", assets_document)
    _atomic_private_json(media_root / "provenance-aggregate.json", dict(provenance))
    _atomic_private_json(media_root / "acquisition-receipt.json", receipt)
    _atomic_private_json(media_root / "manifest.json", manifest)
    _atomic_private_json(
        media_root / "checksum-inventory.json",
        _inventory(media_root, max_bytes=LOCAL_RUNTIME_CAP_BYTES),
    )
    os.replace(media_root, sealed_root)


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    runtime_root: Path
    aoi_config_path: Path
    source_policy_path: Path
    resume: bool = False
    max_wall_seconds: int = LOCAL_MAX_WALL_SECONDS


@dataclass(frozen=True, slots=True)
class DiagnoseAcquisitionConfig:
    runtime_root: Path
    aoi_config_path: Path
    request_limit: int = 10
    timeout_seconds: float = 20.0


@dataclass(frozen=True, slots=True)
class _ProbeSpec:
    name: str
    bbox: BoundingBox
    fields: tuple[str, ...]
    limit: int


def _response_class(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "success"
    if status_code == 400:
        return "bad_request"
    if status_code == 401:
        return "auth_invalid"
    if status_code == 403:
        return "permission_denied"
    if status_code == 429:
        return "rate_limited"
    if 500 <= status_code < 600:
        return "server_error"
    return "other_http"


def _safe_content_type(value: str | None) -> str | None:
    if value is None:
        return None
    media_type = value.split(";", 1)[0].strip().casefold()
    if re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", media_type):
        return media_type
    return "invalid"


def _safe_request_id(headers: httpx.Headers) -> dict[str, str] | None:
    for name in ("x-fb-request-id", "x-request-id", "x-trace-id"):
        value = headers.get(name)
        if value is not None and 0 < len(value) <= 512:
            return {
                "header_name": name,
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
    return None


def _probe_images(
    client: httpx.Client,
    token: str,
    spec: _ProbeSpec,
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    _require(1 <= spec.limit <= 100, "DIAGNOSTIC_PROBE_LIMIT_INVALID")
    _require(
        bool(spec.fields)
        and set(spec.fields).issubset(MAPILLARY_API_FIELDS)
        and "thumb_1024_url" not in spec.fields,
        "DIAGNOSTIC_PROBE_FIELDS_INVALID",
    )
    started = time.perf_counter()
    status_code: int | None = None
    response_class = "transport_error"
    content_type: str | None = None
    request_id: dict[str, str] | None = None
    try:
        with client.stream(
            "GET",
            f"{MAPILLARY_API_BASE_URL}/images",
            params={
                "bbox": spec.bbox.as_query_value(),
                "fields": ",".join(spec.fields),
                "limit": spec.limit,
            },
            headers={
                "Accept": "application/json",
                "Authorization": f"OAuth {token}",
                "User-Agent": "AtlasLens-Phase3F-Acquisition-Diagnostic/1",
            },
            timeout=timeout_seconds,
        ) as response:
            status_code = response.status_code
            response_class = _response_class(response.status_code)
            content_type = _safe_content_type(response.headers.get("Content-Type"))
            request_id = _safe_request_id(response.headers)
    except httpx.TimeoutException:
        response_class = "timeout"
    except (httpx.NetworkError, httpx.RemoteProtocolError):
        response_class = "transport_error"
    except httpx.HTTPError:
        response_class = "http_error"
    elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
    return {
        "name": spec.name,
        "status_code": status_code,
        "response_class": response_class,
        "content_type": content_type,
        "elapsed_ms": elapsed_ms,
        "request_id": request_id,
        "response_body_retained": False,
    }


def _is_success(result: Mapping[str, object]) -> bool:
    return result.get("response_class") == "success"


def _is_server_error(result: Mapping[str, object]) -> bool:
    return result.get("response_class") == "server_error"


def _diagnose_probe_results(results: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    baseline = results["baseline_auth"]
    exact = results["exact_failing_request"]
    repeat = results["exact_repeat"]
    if not _is_success(baseline):
        return {
            "code": "BASELINE_AUTH_OR_PROVIDER_UNAVAILABLE",
            "deterministic_cause": False,
            "request_change_authorized": False,
        }
    if _is_success(exact):
        return {
            "code": "HISTORICAL_FAILURE_NOT_REPRODUCED",
            "deterministic_cause": False,
            "request_change_authorized": False,
        }
    if not (_is_server_error(exact) and _is_server_error(repeat)):
        return {
            "code": "EXACT_FAILURE_NOT_STABLE_SERVER_ERROR",
            "deterministic_cause": False,
            "request_change_authorized": False,
        }
    reduced = (results["reduced_cell_first"], results["reduced_cell_last"])
    if all(_is_success(result) for result in reduced):
        return {
            "code": "BBOX_PARTITION_REQUIRED",
            "deterministic_cause": True,
            "request_change_authorized": True,
            "change": "deterministic_quarter_cell_partition",
        }
    confirmation = results["adaptive_confirmation"]
    for size in (50, 25, 1):
        candidate = results[f"exact_limit_{size}"]
        if (
            _is_success(candidate)
            and confirmation.get("confirmed_limit") == size
            and _is_success(confirmation)
        ):
            return {
                "code": "PAGE_LIMIT_REDUCTION_REQUIRED",
                "deterministic_cause": True,
                "request_change_authorized": True,
                "safe_page_size": size,
                "change": "bounded_page_size",
            }
    if (
        _is_success(results["exact_minimal_fields"])
        and confirmation.get("omitted_field") == "computed_compass_angle"
        and _is_success(confirmation)
    ):
        return {
            "code": "FIELD_COMPUTED_COMPASS_ANGLE_REJECTED",
            "deterministic_cause": True,
            "request_change_authorized": True,
            "field_to_remove": "computed_compass_angle",
            "change": "remove_single_rejected_field",
        }
    return {
        "code": "PROVIDER_GENERAL_5XX_NOT_REQUEST_SHAPE_ISOLATED",
        "deterministic_cause": False,
        "request_change_authorized": False,
    }


def diagnose_acquisition(config: DiagnoseAcquisitionConfig) -> dict[str, object]:
    _require(config.request_limit == 10, "DIAGNOSTIC_REQUEST_LIMIT_INVALID")
    _require(0 < config.timeout_seconds <= 30, "DIAGNOSTIC_TIMEOUT_INVALID")
    token = _mapillary_token()
    runtime_root = _safe_existing_directory(
        _absolute_path(config.runtime_root),
        "LOCAL_RUNTIME_ROOT_INVALID",
    )
    run_id = _current_run(runtime_root)
    run_root = runtime_root / run_id
    state = _read_json(
        _state_path(runtime_root, run_id),
        max_bytes=1_048_576,
        code="LOCAL_STATE_INVALID",
    )
    _require(
        state.get("stage") == "ACQUISITION_FAILED_RESUMABLE"
        and state.get("error_code") == "MAPILLARY_API_SERVER_RETRY_EXHAUSTED",
        "DIAGNOSTIC_RUN_STATE_INCOMPATIBLE",
    )
    areas = worker.load_city_areas(config.aoi_config_path)
    checkpoint = worker._load_metadata_page_checkpoint(  # noqa: SLF001
        run_root / "acquisition-work" / "metadata-pages.json",
        areas=areas,
        run_id=run_id,
    )
    _require(
        checkpoint.city_index < len(areas)
        and checkpoint.box_index < len(areas[checkpoint.city_index].boxes)
        and checkpoint.next_url is None,
        "DIAGNOSTIC_CHECKPOINT_SHAPE_UNSUPPORTED",
    )
    exact_box = areas[checkpoint.city_index].boxes[checkpoint.box_index]
    reduced_boxes = worker.quarter_bbox(exact_box)
    full_fields = MAPILLARY_API_FIELDS[:-1]
    baseline_box = BoundingBox(
        west=35.4400,
        south=38.7200,
        east=35.4410,
        north=38.7210,
    )
    initial_specs = (
        _ProbeSpec("baseline_auth", baseline_box, ("id",), 1),
        _ProbeSpec("exact_failing_request", exact_box, full_fields, 100),
        _ProbeSpec("exact_limit_1", exact_box, full_fields, 1),
        _ProbeSpec("exact_minimal_fields", exact_box, ("id",), 100),
        _ProbeSpec("reduced_cell_first", reduced_boxes[0], full_fields, 100),
        _ProbeSpec("reduced_cell_last", reduced_boxes[-1], full_fields, 100),
        _ProbeSpec("exact_limit_25", exact_box, full_fields, 25),
        _ProbeSpec("exact_limit_50", exact_box, full_fields, 50),
        _ProbeSpec("exact_repeat", exact_box, full_fields, 100),
    )
    _require(len(initial_specs) < config.request_limit, "DIAGNOSTIC_REQUEST_LIMIT_INVALID")
    results: list[dict[str, object]] = []
    with httpx.Client(follow_redirects=False, trust_env=False) as client:
        for spec in initial_specs:
            results.append(
                _probe_images(
                    client,
                    token,
                    spec,
                    timeout_seconds=config.timeout_seconds,
                )
            )
        indexed = {cast(str, result["name"]): result for result in results}
        confirmation_spec: _ProbeSpec
        confirmation_metadata: dict[str, object]
        confirmed_limit = next(
            (
                size
                for size in (50, 25, 1)
                if _is_success(indexed[f"exact_limit_{size}"])
            ),
            None,
        )
        if confirmed_limit is not None:
            confirmation_spec = _ProbeSpec(
                "adaptive_confirmation",
                exact_box,
                full_fields,
                confirmed_limit,
            )
            confirmation_metadata = {"confirmed_limit": confirmed_limit}
        elif _is_success(indexed["exact_minimal_fields"]):
            confirmation_spec = _ProbeSpec(
                "adaptive_confirmation",
                exact_box,
                tuple(field for field in full_fields if field != "computed_compass_angle"),
                100,
            )
            confirmation_metadata = {"omitted_field": "computed_compass_angle"}
        else:
            confirmation_spec = _ProbeSpec(
                "adaptive_confirmation",
                exact_box,
                full_fields,
                100,
            )
            confirmation_metadata = {"control_repeat": True}
        confirmation = _probe_images(
            client,
            token,
            confirmation_spec,
            timeout_seconds=config.timeout_seconds,
        )
        confirmation.update(confirmation_metadata)
        results.append(confirmation)
    _require(len(results) == config.request_limit, "DIAGNOSTIC_REQUEST_COUNT_INVALID")
    indexed_results = {cast(str, result["name"]): result for result in results}
    diagnosis = _diagnose_probe_results(indexed_results)
    receipt = {
        "schema": ACQUISITION_DIAGNOSTIC_SCHEMA,
        "run_id": run_id,
        "historical_error_code": state["error_code"],
        "exact_failing_request": {
            "endpoint": "/images",
            "parameter_names": ["bbox", "fields", "limit"],
            "checkpoint_city_index": checkpoint.city_index,
            "checkpoint_box_index": checkpoint.box_index,
            "next_url_present": False,
        },
        "probe_request_count": len(results),
        "request_limit": config.request_limit,
        "retry_attempts_per_request": 0,
        "image_download_requests": 0,
        "response_bodies_retained": False,
        "full_urls_retained": False,
        "authorization_headers_retained": False,
        "secrets_included": False,
        "probes": results,
        "diagnosis": diagnosis,
        "created_at": datetime.now(UTC).isoformat(),
    }
    receipt_path = run_root / "diagnostics" / "acquisition-diagnostic.json"
    _atomic_private_json(receipt_path, receipt)
    return {
        "run_id": run_id,
        "diagnosis": diagnosis,
        "probe_request_count": len(results),
        "image_download_requests": 0,
        "retry_attempts_per_request": 0,
        "exact_failing_request": receipt["exact_failing_request"],
        "probes": results,
        "receipt_path": str(receipt_path),
        "secrets_included": False,
    }


def run_acquisition(config: AcquisitionConfig) -> dict[str, object]:
    _require(0 < config.max_wall_seconds <= LOCAL_MAX_WALL_SECONDS, "LOCAL_WALL_LIMIT_INVALID")
    token = _mapillary_token()
    runtime_root = _absolute_path(config.runtime_root)
    _require(not runtime_root.is_symlink(), "LOCAL_RUNTIME_ROOT_INVALID")
    runtime_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime_root = _safe_existing_directory(runtime_root, "LOCAL_RUNTIME_ROOT_INVALID")
    os.chmod(runtime_root, 0o700)
    _tree_size(runtime_root)
    run_id = _current_run(runtime_root) if config.resume else _new_run(runtime_root)
    run_root = runtime_root / run_id
    work_root = run_root / "acquisition-work"
    sealed_root = run_root / "sealed-acquisition"
    _require(not sealed_root.exists(), "ACQUISITION_ALREADY_SEALED")
    if config.resume:
        _require(work_root.is_dir() and not _is_reparse(work_root), "ACQUISITION_WORK_MISSING")
    else:
        _require(not work_root.exists(), "ACQUISITION_WORK_EXISTS")
        work_root.mkdir(mode=0o700)
        os.chmod(work_root, 0o700)
    prior_state = (
        _read_json(
            _state_path(runtime_root, run_id),
            max_bytes=1_048_576,
            code="LOCAL_STATE_INVALID",
        )
        if config.resume
        else {}
    )
    resume_after_server_exhaustion = (
        prior_state.get("stage") == "ACQUISITION_FAILED_RESUMABLE"
        and prior_state.get("error_code") == "MAPILLARY_API_SERVER_RETRY_EXHAUSTED"
    )
    _write_state(runtime_root, run_id, "ACQUISITION_RUNNING", resume=config.resume)
    _require(
        worker._sha256_path(config.source_policy_path) == worker.SOURCE_POLICY_SHA256,  # noqa: SLF001
        "SOURCE_POLICY_SHA256_MISMATCH",
    )
    areas = worker.load_city_areas(config.aoi_config_path)
    metadata_path = work_root / "metadata-pages.json"
    counters_path = work_root / "client-counters.json"
    acquisition_path = work_root / "acquisition-checkpoint.json"
    scheduler_path = work_root / "acquisition-scheduler-v4.json"
    media_root = work_root / "private-media"
    if config.resume:
        historical = worker._read_client_counters(counters_path, run_id)  # noqa: SLF001
        checkpoint_schema = _read_json(
            metadata_path,
            max_bytes=16 * 1024 * 1024,
            code="METADATA_PAGE_CHECKPOINT_INVALID",
        ).get("schema")
        migrate_failed_v2_cell = (
            resume_after_server_exhaustion
            and checkpoint_schema == worker.LEGACY_V2_METADATA_PAGE_CHECKPOINT_SCHEMA
        )
        metadata = worker._load_metadata_page_checkpoint(  # noqa: SLF001
            metadata_path,
            areas=areas,
            run_id=run_id,
            persist_migration=not migrate_failed_v2_cell,
        )
        if migrate_failed_v2_cell:
            worker._subdivide_failed_metadata_cell(  # noqa: SLF001
                metadata,
                areas=areas,
                checkpoint_path=metadata_path,
            )
    else:
        historical = (0, 0, 0)
        metadata = worker._new_metadata_page_checkpoint(areas, run_id)  # noqa: SLF001
        worker._write_metadata_page_checkpoint(metadata_path, metadata)  # noqa: SLF001
        worker._write_client_counters(counters_path, run_id, 0, 0, 0)  # noqa: SLF001
    sync_scheduler_checkpoint(
        scheduler_path,
        run_id=run_id,
        metadata_checkpoint=metadata_path,
        client_counters=counters_path,
        acquisition_checkpoint=acquisition_path,
        request_cap=MAX_REQUESTS,
        media_byte_cap=LOCAL_MEDIA_CAP_BYTES,
        max_wall_seconds=config.max_wall_seconds,
        status="ACQUISITION_RUNNING",
        failure_code=(
            cast(str, prior_state["error_code"])
            if config.resume and isinstance(prior_state.get("error_code"), str)
            else None
        ),
        city_order=[area.city for area in areas],
    )
    deadline = time.time() + config.max_wall_seconds
    limits = ClientLimits(
        request_cap=MAX_REQUESTS,
        page_cap=2_000,
        metadata_item_cap=20_000,
        page_size=100,
        image_cap=2_000,
        max_image_bytes=16 * 1024 * 1024,
        concurrency=2,
        timeout_seconds=20.0,
        retry_cap=3,
        backoff_base_seconds=0.5,
        backoff_cap_seconds=8.0,
    )
    try:
        with MapillaryClient(
            token,
            limits=limits,
            counter_observer=lambda requests, pages, rejected: worker._write_client_counters(  # noqa: SLF001
                counters_path,
                run_id,
                requests,
                pages,
                rejected,
            ),
        ) as client:
            if config.resume:
                client.add_historical_counts(
                    request_count=historical[0],
                    page_count=historical[1],
                    rejected_item_count=historical[2],
                )

            consecutive_provider_failures = 0

            def record_progress(city: str, page_count: int, stage: str) -> None:
                nonlocal consecutive_provider_failures
                _require(time.time() < deadline, "LOCAL_ACQUISITION_DEADLINE_REACHED")
                consecutive_provider_failures = 0
                _write_state(
                    runtime_root,
                    run_id,
                    "ACQUISITION_RUNNING",
                    region=city,
                    page_count=page_count,
                    progress_stage=stage,
                )
                sync_scheduler_checkpoint(
                    scheduler_path,
                    run_id=run_id,
                    metadata_checkpoint=metadata_path,
                    client_counters=counters_path,
                    acquisition_checkpoint=acquisition_path,
                    request_cap=MAX_REQUESTS,
                    media_byte_cap=LOCAL_MEDIA_CAP_BYTES,
                    max_wall_seconds=config.max_wall_seconds,
                    status="ACQUISITION_RUNNING",
                    city_order=[area.city for area in areas],
                )

            def record_media_checkpoint() -> None:
                _require(time.time() < deadline, "LOCAL_ACQUISITION_DEADLINE_REACHED")
                sync_scheduler_checkpoint(
                    scheduler_path,
                    run_id=run_id,
                    metadata_checkpoint=metadata_path,
                    client_counters=counters_path,
                    acquisition_checkpoint=acquisition_path,
                    request_cap=MAX_REQUESTS,
                    media_byte_cap=LOCAL_MEDIA_CAP_BYTES,
                    max_wall_seconds=config.max_wall_seconds,
                    status="MEDIA_ACQUISITION_RUNNING",
                    city_order=[area.city for area in areas],
                )

            try:
                while True:
                    try:
                        audit = worker.audit_metadata(
                            client,
                            areas,
                            checkpoint=metadata,
                            checkpoint_path=metadata_path,
                            run_id=run_id,
                            progress=record_progress,
                        )
                        break
                    except BaseException as metadata_error:
                        raw_code = getattr(metadata_error, "code", None)
                        normalized_code = (
                            raw_code.upper()
                            if isinstance(raw_code, str)
                            else "METADATA_ACQUISITION_FAILED"
                        )
                        if normalized_code in _QUARANTINE_METADATA_FAILURES:
                            worker._quarantine_active_metadata_cell(  # noqa: SLF001
                                metadata,
                                areas=areas,
                                checkpoint_path=metadata_path,
                            )
                            normalized_code = "MAPILLARY_PAGING_CELL_QUARANTINED"
                        if normalized_code not in _AUTO_CONTINUE_METADATA_FAILURES:
                            raise
                        consecutive_provider_failures += 1
                        sync_scheduler_checkpoint(
                            scheduler_path,
                            run_id=run_id,
                            metadata_checkpoint=metadata_path,
                            client_counters=counters_path,
                            acquisition_checkpoint=acquisition_path,
                            request_cap=MAX_REQUESTS,
                            media_byte_cap=LOCAL_MEDIA_CAP_BYTES,
                            max_wall_seconds=config.max_wall_seconds,
                            status="ACQUISITION_RUNNING",
                            failure_code=normalized_code,
                            city_order=[area.city for area in areas],
                        )
                        if consecutive_provider_failures >= MAX_CONSECUTIVE_PROVIDER_FAILURES:
                            raise LocalFirstError(
                                "MAPILLARY_CONSECUTIVE_PROVIDER_FAILURE_LIMIT_REACHED"
                            ) from metadata_error
            except CoverageInsufficient as exc:
                _write_state(
                    runtime_root,
                    run_id,
                    "COVERAGE_INSUFFICIENT",
                    outcome=exc.code,
                )
                return {"run_id": run_id, "stage": "COVERAGE_INSUFFICIENT"}
            planned = worker.plan_locked_roles(audit)
            guard = AcquisitionGuard()
            assets, provenance = worker.acquire_planned_assets(
                client,
                areas,
                planned,
                media_root,
                run_id,
                guard,
                acquisition_path,
                restore_client_counts=False,
                max_media_bytes=LOCAL_MEDIA_CAP_BYTES,
                checkpoint_observer=record_media_checkpoint,
                allow_item_failures=True,
            )
            if len(assets) != len(planned):
                report = {
                    "schema": "atlaslens-phase3f-training-readiness-v1",
                    "outcome": "DATASET_NOT_READY_FOR_TRAINING",
                    "ready": False,
                    "planned_asset_count": len(planned),
                    "usable_asset_count": len(assets),
                    "item_failure_count": len(planned) - len(assets),
                    "gpu_started": False,
                    "cloud_mutations": 0,
                    "secrets_included": False,
                }
                _atomic_private_json(run_root / "training-readiness.json", report)
                _write_state(
                    runtime_root,
                    run_id,
                    "DATASET_NOT_READY_FOR_TRAINING",
                    asset_count=len(assets),
                    item_failure_count=len(planned) - len(assets),
                    model_loaded=False,
                    gpu_used=False,
                )
                return {
                    "run_id": run_id,
                    "stage": "DATASET_NOT_READY_FOR_TRAINING",
                    "asset_count": len(assets),
                    "secrets_included": False,
                }
            split = seal_split(
                assets,
                in_domain_cities=[item.city for item in audit.selection.in_domain],
                ood_cities=[item.city for item in audit.selection.ood],
            )
            _require(time.time() < deadline, "LOCAL_ACQUISITION_DEADLINE_REACHED")
            sync_scheduler_checkpoint(
                scheduler_path,
                run_id=run_id,
                metadata_checkpoint=metadata_path,
                client_counters=counters_path,
                acquisition_checkpoint=acquisition_path,
                request_cap=MAX_REQUESTS,
                media_byte_cap=LOCAL_MEDIA_CAP_BYTES,
                max_wall_seconds=config.max_wall_seconds,
                status="CORPUS_READY_TO_SEAL",
                city_order=[area.city for area in areas],
            )
            shutil.copy2(scheduler_path, media_root / "acquisition-scheduler-v4.json")
            _seal_acquisition(
                run_id=run_id,
                media_root=media_root,
                sealed_root=sealed_root,
                selection=audit.selection,
                split=split,
                provenance=provenance,
                client=client,
                guard=guard,
            )
            _remove_owned_tree(
                work_root,
                parent=run_root,
                code="ACQUISITION_WORK_CLEANUP_REFUSED",
            )
            _write_state(
                runtime_root,
                run_id,
                "ACQUISITION_SEALED",
                sealed_root=str(sealed_root),
                asset_count=len(split.assets),
                media_bytes=guard.media_bytes,
                model_loaded=False,
                gpu_used=False,
            )
            _tree_size(runtime_root)
            return {
                "run_id": run_id,
                "stage": "ACQUISITION_SEALED",
                "sealed_root": str(sealed_root),
                "asset_count": len(split.assets),
                "secrets_included": False,
            }
    except BaseException as exc:
        code = getattr(exc, "code", None)
        safe_code = (
            code.upper()
            if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_]+", code)
            else "LOCAL_ACQUISITION_FAILED"
        )
        _write_state(
            runtime_root,
            run_id,
            _acquisition_failure_stage(safe_code),
            error_code=safe_code,
            model_loaded=False,
            gpu_used=False,
        )
        if metadata_path.is_file() and counters_path.is_file():
            with suppress(BaseException):
                sync_scheduler_checkpoint(
                    scheduler_path,
                    run_id=run_id,
                    metadata_checkpoint=metadata_path,
                    client_counters=counters_path,
                    acquisition_checkpoint=acquisition_path,
                    request_cap=MAX_REQUESTS,
                    media_byte_cap=LOCAL_MEDIA_CAP_BYTES,
                    max_wall_seconds=config.max_wall_seconds,
                    status=_acquisition_failure_stage(safe_code),
                    failure_code=safe_code,
                    city_order=[area.city for area in areas],
                )
        raise


@contextmanager
def _network_disabled() -> Iterator[None]:
    original_create_connection = socket.create_connection
    original_socket = socket.socket

    class OfflineSocket(socket.socket):
        def connect(self, address: Any) -> None:  # noqa: ANN401
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                raise LocalFirstError("COMPUTE_NETWORK_DISABLED")
            super().connect(address)

        def connect_ex(self, address: Any) -> int:  # noqa: ANN401
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                raise LocalFirstError("COMPUTE_NETWORK_DISABLED")
            return super().connect_ex(address)

    def refused(*_args: object, **_kwargs: object) -> socket.socket:
        raise LocalFirstError("COMPUTE_NETWORK_DISABLED")

    socket_namespace = vars(socket)
    socket.create_connection = refused
    socket_namespace["socket"] = OfflineSocket
    try:
        yield
    finally:
        socket.create_connection = original_create_connection
        socket_namespace["socket"] = original_socket


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    runtime_root: Path
    model_path: Path
    vendor_root: Path
    max_wall_seconds: int = LOCAL_MAX_WALL_SECONDS


def run_compute(config: ComputeConfig) -> dict[str, object]:
    _require(0 < config.max_wall_seconds <= LOCAL_MAX_WALL_SECONDS, "LOCAL_WALL_LIMIT_INVALID")
    os.environ.pop("MAPILLARY_ACCESS_TOKEN", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    runtime_root = _safe_existing_directory(
        _absolute_path(config.runtime_root),
        "LOCAL_RUNTIME_ROOT_INVALID",
    )
    run_id = _current_run(runtime_root)
    run_root = runtime_root / run_id
    sealed = verify_sealed_acquisition(run_root / "sealed-acquisition")
    worker.verify_megaloc_artifacts(
        config.model_path,
        config.vendor_root / "megaloc_model.py",
        config.vendor_root / "LICENSE",
    )
    output_root = run_root / "compute-output"
    work_root = run_root / "compute-work"
    _require(not output_root.exists() and not output_root.is_symlink(), "COMPUTE_OUTPUT_EXISTS")
    _require(not work_root.exists() and not work_root.is_symlink(), "COMPUTE_WORK_EXISTS")
    output_root.mkdir(mode=0o700)
    work_root.mkdir(mode=0o700)
    _write_state(runtime_root, run_id, "COMPUTE_RUNNING", network_enabled=False)
    deadline = time.time() + config.max_wall_seconds
    pipeline = Phase3FPipeline.create(output_root / "pipeline-state.json", run_id=run_id)
    pipeline.lock_selection(sealed.selection)
    acquisition_guard = AcquisitionGuard(
        request_count=cast(int, sealed.receipt["mapillary_request_count"]),
        image_count=cast(int, sealed.receipt["image_count"]),
        media_bytes=cast(int, sealed.receipt["media_bytes"]),
    )
    pipeline.complete_acquisition(acquisition_guard)
    pipeline.record_split(sealed.split)
    worker._atomic_json(output_root / "selection-lock.json", sealed.selection.document())  # noqa: SLF001
    worker._atomic_json(output_root / "split-lock.json", sealed.split.inference_document())  # noqa: SLF001
    worker._atomic_json(output_root / "provenance-aggregate.json", sealed.provenance)  # noqa: SLF001
    runtime: worker.MegaLocRuntime | None = None
    try:
        with _network_disabled():
            runtime = worker.MegaLocRuntime(config.model_path, config.vendor_root)
            references = tuple(item for item in sealed.split.assets if item.role == "reference")
            calibration = tuple(
                item for item in sealed.split.assets if item.role == "calibration"
            )
            reference_matrix = worker._descriptor_shards(  # noqa: SLF001
                runtime,
                references,
                sealed.root,
                work_root / "reference",
            )
            calibration_matrix = worker._descriptor_shards(  # noqa: SLF001
                runtime,
                calibration,
                sealed.root,
                work_root / "calibration",
            )
            index, descriptor_sha, index_sha, publication_sha = worker._publish_reference_bundle(  # noqa: SLF001
                output_root,
                references,
                reference_matrix,
            )
            publication = DescriptorPublication(
                selection_lock_sha256=sealed.selection.lock_sha256,
                split_lock_sha256=sealed.split.split_lock_sha256,
                source_policy_sha256=worker.SOURCE_POLICY_SHA256,
                descriptor_publication_sha256=publication_sha,
                index_sha256=index_sha,
                city_scope=tuple(item.city for item in sealed.selection.in_domain),
                created_at=datetime.now(UTC),
            )
            pipeline.record_descriptor_publication(publication)
            calibration_rows = worker._retrieval_rows(  # noqa: SLF001
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
                selection_lock_sha256=sealed.selection.lock_sha256,
                split_lock_sha256=sealed.split.split_lock_sha256,
            )
            pipeline.lock_threshold(threshold)
            worker._atomic_json(output_root / "calibration.json", threshold.document())  # noqa: SLF001
            _require(time.time() < deadline, "LOCAL_COMPUTE_DEADLINE_REACHED")
            holdout = tuple(
                item
                for item in sealed.split.assets
                if item.role in {"sealed_holdout", "ood_holdout"}
            )
            holdout_matrix = worker._descriptor_shards(  # noqa: SLF001
                runtime,
                holdout,
                sealed.root,
                work_root / "holdout",
            )
            holdout_rows = worker._retrieval_rows(  # noqa: SLF001
                index,
                holdout,
                holdout_matrix,
                references,
                publication.city_scope,
            )
            benchmark = evaluate_holdout_once(
                holdout_rows,
                threshold=threshold,
                selection_lock_sha256=sealed.selection.lock_sha256,
                split_lock_sha256=sealed.split.split_lock_sha256,
                in_domain_cities=publication.city_scope,
                ood_cities=tuple(item.city for item in sealed.selection.ood),
                leakage_passed=sealed.split.leakage.passed,
                city_minimums_passed=True,
                security_integrity_passed=True,
                holdout_open_count_before=0,
            )
            pipeline.record_benchmark(benchmark)
            pipeline.finalize()
            worker._atomic_json(output_root / "aggregate-benchmark.json", benchmark.document())  # noqa: SLF001
            worker._atomic_json(  # noqa: SLF001
                output_root / "descriptor-publication.json",
                {**publication.document(), "reference_descriptor_sha256": descriptor_sha},
            )
            receipt = {
                "schema": "atlaslens-phase3f-local-compute-v1",
                "run_id": run_id,
                "outcome": benchmark.outcome,
                "network_calls": 0,
                "mapillary_requests": 0,
                "offline_model_loading": True,
                "secrets_included": False,
                "finished_at": datetime.now(UTC).isoformat(),
            }
            worker._atomic_json(output_root / "execution-receipt.json", receipt)  # noqa: SLF001
            worker._atomic_json(  # noqa: SLF001
                output_root / "checksum-inventory.json",
                worker._inventory_output(output_root),  # noqa: SLF001
            )
        _remove_owned_tree(
            work_root,
            parent=run_root,
            code="COMPUTE_WORK_CLEANUP_REFUSED",
        )
        _write_state(
            runtime_root,
            run_id,
            "COMPUTE_COMPLETE",
            output_root=str(output_root),
            network_calls=0,
            mapillary_requests=0,
            outcome=benchmark.outcome,
        )
        return receipt
    except BaseException as exc:
        code = getattr(exc, "code", None)
        safe_code = code if isinstance(code, str) else "LOCAL_COMPUTE_FAILED"
        _remove_owned_tree(
            work_root,
            parent=run_root,
            code="COMPUTE_WORK_CLEANUP_REFUSED",
        )
        _remove_owned_tree(
            output_root,
            parent=run_root,
            code="COMPUTE_OUTPUT_CLEANUP_REFUSED",
        )
        _write_state(
            runtime_root,
            run_id,
            "COMPUTE_FAILED",
            error_code=safe_code,
            network_calls=0,
            mapillary_requests=0,
        )
        raise
    finally:
        if runtime is not None:
            runtime.close()


def status(runtime_root: Path) -> dict[str, object]:
    root = _safe_existing_directory(
        _absolute_path(runtime_root),
        "LOCAL_RUNTIME_ROOT_INVALID",
    )
    run_id = _current_run(root)
    state = _read_json(
        _state_path(root, run_id),
        max_bytes=1_048_576,
        code="LOCAL_STATE_INVALID",
    )
    return {
        "run_id": run_id,
        "stage": state.get("stage"),
        "error_code": state.get("error_code"),
        "asset_count": state.get("asset_count"),
        "media_bytes": state.get("media_bytes"),
        "network_calls": state.get("network_calls"),
        "mapillary_requests": state.get("mapillary_requests"),
        "secrets_included": False,
    }


def cleanup(
    runtime_root: Path,
    *,
    execute: bool,
    confirm_run_id: str | None,
) -> dict[str, object]:
    root = _safe_existing_directory(
        _absolute_path(runtime_root),
        "LOCAL_RUNTIME_ROOT_INVALID",
    )
    run_id = _current_run(root)
    run_root = (root / run_id).resolve()
    _require(run_root.parent == root and run_root.is_dir(), "LOCAL_RUN_ROOT_INVALID")
    temp_root = root / "_tmp"
    bytes_to_remove = _tree_size(run_root, max_bytes=None) + _tree_size(
        temp_root,
        max_bytes=None,
    )
    result = {
        "run_id": run_id,
        "targets": [str(run_root), str(temp_root)],
        "bytes": bytes_to_remove,
        "execute": execute,
        "secrets_included": False,
    }
    if not execute:
        return result
    _require(confirm_run_id == run_id, "LOCAL_CLEANUP_CONFIRMATION_MISMATCH")
    _remove_owned_tree(
        run_root,
        parent=root,
        code="LOCAL_CLEANUP_REPARSE_REFUSED",
    )
    _remove_owned_tree(
        temp_root,
        parent=root,
        code="LOCAL_CLEANUP_REPARSE_REFUSED",
    )
    current = root / "current.json"
    _require(current.is_file() and not _is_reparse(current), "LOCAL_CURRENT_RUN_INVALID")
    current.unlink()
    return {**result, "removed": True}


__all__ = [
    "AcquisitionConfig",
    "ComputeConfig",
    "DiagnoseAcquisitionConfig",
    "LOCAL_RUNTIME_CAP_BYTES",
    "LocalFirstError",
    "VerifiedAcquisition",
    "cleanup",
    "diagnose_acquisition",
    "run_acquisition",
    "run_compute",
    "status",
    "verify_sealed_acquisition",
]
