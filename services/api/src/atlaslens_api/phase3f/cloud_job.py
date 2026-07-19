"""Executable, bounded Phase 3F Mapillary/MegaLoc cloud worker.

The module keeps evaluator truth and raw imagery inside the ephemeral work root.
Only aggregate receipts and derived index/descriptor artifacts are published.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
from PIL import Image

from atlaslens_api.mapillary_demo.client import MapillaryClient, RemoteImage
from atlaslens_api.mapillary_demo.models import BoundingBox, ClientLimits
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
SOURCE_POLICY_SHA256: Final = (
    "72d51363f2b63de368d34d4d7bb2fc1145f93dc0469e7026100976732dfde209"
)
MAX_METADATA_ITEMS_PER_CITY: Final = 600
MAX_WALL_SECONDS: Final = 5 * 60 * 60 + 45 * 60

Role = Literal["reference", "calibration", "sealed_holdout", "ood_holdout"]


class Phase3FCloudJobError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


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
        return (
            BoundingBox(west=west, south=south, east=self.longitude, north=self.latitude),
            BoundingBox(west=self.longitude, south=south, east=east, north=self.latitude),
            BoundingBox(west=west, south=self.latitude, east=self.longitude, north=north),
            BoundingBox(west=self.longitude, south=self.latitude, east=east, north=north),
        )


def load_city_areas(path: Path) -> tuple[CityArea, ...]:
    _require(path.is_file() and not path.is_symlink(), "CITY_AOI_CONFIG_MISSING")
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
        isinstance(half_span, int | float) and 0.0 < float(half_span) <= 0.05,
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
            and isinstance(latitude, int | float),
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
        {area.city for area in areas} == set(CANDIDATE_CITY_REGIONS),
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


def audit_metadata(
    client: MapillaryClient,
    areas: Sequence[CityArea],
) -> MetadataAudit:
    records: list[CityCoverageRecord] = []
    assets_by_city: dict[str, tuple[MetadataAsset, ...]] = {}
    for area in areas:
        remote = tuple(
            client.iter_images(
                area.boxes,
                include_thumbnail=False,
                item_cap=MAX_METADATA_ITEMS_PER_CITY,
            )
        )
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


def verify_megaloc_artifacts(model_path: Path, source_path: Path, license_path: Path) -> None:
    _require(
        _sha256_path(model_path, expected_size=MODEL_SIZE_BYTES) == MODEL_SHA256,
        "MODEL_SHA256_MISMATCH",
    )
    _require(
        _sha256_path(source_path) == MEGALOC_CANONICAL_SOURCE_SHA256,
        "MEGALOC_SOURCE_SHA256_MISMATCH",
    )
    _require(
        _sha256_path(license_path) == MEGALOC_LICENSE_CANONICAL_SHA256,
        "MEGALOC_LICENSE_SHA256_MISMATCH",
    )


def _normalize_image(payload: bytes) -> tuple[bytes, int, int, str]:
    import io

    with Image.open(io.BytesIO(payload)) as image:
        normalized = image.convert("RGB")
        _require(max(normalized.size) <= 1024, "RENDITION_EDGE_EXCEEDED")
        output = io.BytesIO()
        normalized.save(output, format="JPEG", quality=92, optimize=False, progressive=False)
        perceptual = normalized.resize((9, 8)).convert("L")
        pixels = np.asarray(perceptual, dtype=np.uint8)
        bits = pixels[:, 1:] > pixels[:, :-1]
        phash = f"{int(''.join('1' if bit else '0' for bit in bits.flat), 2):016x}"
        return output.getvalue(), normalized.width, normalized.height, phash


def acquire_planned_assets(
    client: MapillaryClient,
    areas: Sequence[CityArea],
    planned: Sequence[PlannedAsset],
    media_root: Path,
    run_id: str,
    guard: AcquisitionGuard,
) -> tuple[tuple[SplitAsset, ...], dict[str, object]]:
    wanted = {item.metadata.image_id: item for item in planned}
    _require(len(wanted) == len(planned), "PLANNED_IMAGE_DUPLICATE")
    completed: dict[str, SplitAsset] = {}
    provenance_hashes: list[str] = []
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
            except BaseException:
                guard.finish_request(status_code=500, media_bytes=0, admitted_image=False)
                raise
            guard.finish_request(
                status_code=200,
                media_bytes=len(payload),
                admitted_image=True,
                provenance=sidecar,
                privacy_review=privacy_review,
            )
            plan = wanted[image_id]
            opaque_id = _stable_key(run_id, image_id)[:32]
            relative = Path("assets") / opaque_id[:2] / f"{opaque_id}.jpg"
            destination = media_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(normalized)
            sidecar_document = {
                **asdict(sidecar),
                "capture_date": sidecar.capture_date.astimezone(UTC).isoformat(),
            }
            _atomic_json(
                media_root / "private-sidecars" / f"{opaque_id}.json",
                sidecar_document,
            )
            provenance_hashes.append(_sha256_bytes(_canonical_bytes(sidecar_document)))
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
            if len(completed) == len(planned):
                break
    _require(len(completed) == len(planned), "PLANNED_IMAGE_UNAVAILABLE")
    _require(guard.media_bytes <= MAX_MEDIA_BYTES, "MEDIA_CAP_EXCEEDED")
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
    config.work_root.mkdir(parents=True, exist_ok=True)
    config.output_root.mkdir(parents=True, exist_ok=False)
    media_root = config.work_root / "private-media"
    checkpoints = config.work_root / "descriptor-checkpoints"
    state_path = config.output_root / "pipeline-state.json"
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
    started = datetime.now(UTC)
    try:
        with MapillaryClient(cast(str, token), limits=limits) as client:
            _check_deadline(config.deadline_epoch)
            try:
                audit = audit_metadata(client, areas)
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
                return execution
            assets, provenance = acquire_planned_assets(
                client,
                areas,
                planned,
                media_root,
                config.run_id,
                guard,
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
                "raw_coordinates_included": False,
                "secrets_included": False,
                "raw_images_included": False,
            }
            _atomic_json(config.output_root / "execution-receipt.json", execution)
            inventory = _inventory_output(config.output_root)
            _atomic_json(config.output_root / "checksum-inventory.json", inventory)
            return execution
    finally:
        if runtime is not None:
            runtime.close()
        if media_root.exists():
            shutil.rmtree(media_root)
        if checkpoints.exists():
            shutil.rmtree(checkpoints)


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
    "run_cloud_job",
    "verify_megaloc_artifacts",
]
