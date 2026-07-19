from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib
import importlib.util
import io
import math
import os
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from atlaslens_api.model_management.errors import ModelManagementError
from atlaslens_api.model_management.service import ModelManagementService
from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    GlobalProviderStatus,
    InvocationContext,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.storage import LocalImageHandle

_LIMITATIONS = [
    "model_prediction_is_unverified",
    "gallery_softmax_is_not_calibrated_probability",
    "fixed_gallery_limits_geographic_resolution",
    "broad_global_model_not_street_level_proof",
]
_MAX_IMAGE_BYTES = 50 * 1024 * 1024
_MINIMUM_DISTINCT_RADIUS_KM = 0.001


class GeoClipOutputNormalizationError(ValueError):
    def __init__(self, subreason_code: str) -> None:
        super().__init__(subreason_code)
        self.subreason_code = subreason_code


@dataclass(frozen=True, slots=True)
class GeoClipNormalizationDiagnostics:
    input_candidate_count: int
    valid_candidate_count: int
    rejected_coordinate_non_finite: int
    rejected_coordinate_out_of_range: int
    rejected_score_non_finite: int
    rejected_score_out_of_range: int
    rejected_scalar_conversion: int
    configured_dedup_survivors: int
    minimum_diversity_backfill: int
    output_candidate_count: int


@dataclass(frozen=True, slots=True)
class NormalizedGeoClipOutput:
    result: GlobalPredictionResult
    diagnostics: GeoClipNormalizationDiagnostics


@dataclass(frozen=True, slots=True)
class _CandidateRow:
    original_rank: int
    latitude: float
    longitude: float
    raw_score: float


class GeoClipRuntime(Protocol):
    device: str
    dtype: str

    def predict(self, image_payload: bytes, top_k: int) -> tuple[Any, Any]: ...

    def encode_coordinates(self, coordinates: Sequence[tuple[float, float]]) -> Any: ...

    def score_coordinate_features(self, image_payload: bytes, location_features: Any) -> Any: ...


ModelLoader = Callable[[Path, Path, str], GeoClipRuntime]
DeviceSelector = Callable[[], str]
CudaCleanup = Callable[[], None]


class _OfficialGeoClipRuntime:
    def __init__(self, snapshot: Path, assets: Path, device: str) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
        image_module = importlib.import_module("geoclip.model.image_encoder")
        location_module = importlib.import_module("geoclip.model.location_encoder")

        original_clip = image_module.__dict__["CLIPModel"]
        original_processor = image_module.__dict__["AutoProcessor"]

        class LocalClip:
            @classmethod
            def from_pretrained(cls, _name: str) -> Any:
                return original_clip.from_pretrained(
                    snapshot, local_files_only=True, use_safetensors=True
                )

        class LocalProcessor:
            @classmethod
            def from_pretrained(cls, _name: str) -> Any:
                return original_processor.from_pretrained(
                    snapshot, local_files_only=True, use_fast=False
                )

        image_module.__dict__["CLIPModel"] = LocalClip
        image_module.__dict__["AutoProcessor"] = LocalProcessor
        try:
            self._image_encoder = image_module.ImageEncoder()
        finally:
            image_module.__dict__["CLIPModel"] = original_clip
            image_module.__dict__["AutoProcessor"] = original_processor

        self._location_encoder = location_module.LocationEncoder(from_pretrained=False)
        self._image_encoder.mlp.load_state_dict(
            torch.load(
                assets / "image_encoder_mlp_weights.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        self._location_encoder.load_state_dict(
            torch.load(
                assets / "location_encoder_weights.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        scale = torch.load(
            assets / "logit_scale_weights.pth", map_location="cpu", weights_only=True
        )
        self._logit_scale = scale.detach().to(device)
        self._image_encoder.eval().to(device)
        self._location_encoder.eval().to(device)
        self._torch = torch
        self._functional = functional
        self.device = device
        self.dtype = "float32"

        coordinates: list[tuple[float, float]] = []
        with (assets / "coordinates_100K.csv").open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                coordinates.append((float(row["LAT"]), float(row["LON"])))
        self._gps_cpu = torch.tensor(coordinates, dtype=torch.float32)
        encoded: list[Any] = []
        with torch.inference_mode():
            for chunk in self._gps_cpu.split(4096):
                features = self._location_encoder(chunk.to(device))
                encoded.append(functional.normalize(features, dim=1).cpu())
        self._location_features = torch.cat(encoded).to(device)

    def predict(self, image_payload: bytes, top_k: int) -> tuple[Any, Any]:
        prepared = self._prepare_image(image_payload)
        with self._torch.inference_mode():
            image_features = self._functional.normalize(
                self._image_encoder(prepared.to(self.device)), dim=1
            )
            logits = self._logit_scale.exp() * (image_features @ self._location_features.T)
            probabilities = logits.softmax(dim=-1).cpu()
            selected = self._torch.topk(probabilities, min(top_k, len(self._gps_cpu)), dim=1)
        return self._gps_cpu[selected.indices[0]], selected.values[0]

    def _prepare_image(self, image_payload: bytes) -> Any:
        image_class = importlib.import_module("PIL.Image")
        image_ops = importlib.import_module("PIL.ImageOps")
        with image_class.open(io.BytesIO(image_payload)) as image:
            oriented = image_ops.exif_transpose(image)
            return self._image_encoder.preprocess_image(oriented.convert("RGB"))

    def encode_coordinates(self, coordinates: Sequence[tuple[float, float]]) -> Any:
        gps_cpu = self._torch.tensor(coordinates, dtype=self._torch.float32)
        encoded: list[Any] = []
        with self._torch.inference_mode():
            for chunk in gps_cpu.split(4096):
                features = self._location_encoder(chunk.to(self.device))
                encoded.append(self._functional.normalize(features, dim=1).cpu())
        return self._torch.cat(encoded)

    def score_coordinate_features(self, image_payload: bytes, location_features: Any) -> Any:
        prepared = self._prepare_image(image_payload)
        with self._torch.inference_mode():
            image_features = self._functional.normalize(
                self._image_encoder(prepared.to(self.device)), dim=1
            )
            similarities = image_features @ location_features.to(self.device).T
        return similarities[0].float().cpu()


def _default_loader(snapshot: Path, assets: Path, device: str) -> GeoClipRuntime:
    return _OfficialGeoClipRuntime(snapshot, assets, device)


class GeoCLIPGlobalGeolocationProvider:
    def __init__(
        self,
        management: ModelManagementService,
        *,
        model_loader: ModelLoader = _default_loader,
        device_selector: DeviceSelector | None = None,
        timeout_seconds: float = 45.0,
        max_concurrency: int = 1,
        internal_top_k: int | None = None,
        cuda_cleanup: CudaCleanup | None = None,
    ) -> None:
        selected_top_k = internal_top_k or management.manifest.default_top_k
        if not 3 <= selected_top_k <= 100:
            raise ValueError("GeoCLIP internal Top-K must be between 3 and 100")
        self._management = management
        self._loader = model_loader
        self._device_selector = device_selector or _select_device
        self._timeout = timeout_seconds
        self._internal_top_k = selected_top_k
        self._raw_candidate_top_k = max(20, selected_top_k)
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._cuda_cleanup = cuda_cleanup or _clear_cuda_cache
        self._model: GeoClipRuntime | None = None
        self._load_task: asyncio.Task[GeoClipRuntime] | None = None
        self._detached_inference_tasks: set[asyncio.Task[tuple[Any, Any]]] = set()
        self._selected_device: str | None = None
        self._state = "ready" if management.is_installed() else "not_installed"
        self._reason: str | None = None
        self._last_output_subreason: str | None = None
        self._last_output_structure: dict[str, object] | None = None
        self._last_normalization: dict[str, int] | None = None
        self._load_duration_ms = 0
        self._coordinate_cache: OrderedDict[str, tuple[str, Any]] = OrderedDict()
        self._coordinate_cache_lock = asyncio.Lock()
        self._coordinate_cache_hits = 0
        self._coordinate_cache_misses = 0
        self.descriptor = ProviderDescriptor(
            id="geoclip-global-v1",
            kind="global_geolocation",
            version="1.0.0",
            execution_boundary="local",
            criticality="optional",
            available=management.is_installed(),
            unavailable_reason_code=None if management.is_installed() else "model_not_installed",
            model_name="GeoCLIP-1.2.0",
        )

    def status(self) -> GlobalProviderStatus:
        installed = self._management.is_installed()
        effective_state = self._state
        if not installed:
            effective_state = "not_installed"
        elif effective_state == "not_installed":
            effective_state = "ready"
        if installed and self._selected_device is None:
            try:
                self._selected_device = self._device_selector()
            except (ImportError, ModelManagementError):
                effective_state = "unavailable"
        return GlobalProviderStatus(
            status=effective_state,
            installed=installed,
            verified=self._model is not None,
            model_name="GeoCLIP",
            model_revision=self._management.manifest.model_version,
            device=(self._model.device if self._model is not None else self._selected_device),
            calibration_state="uncalibrated",
            reason_code=self._reason,
        )

    def diagnostics(self) -> dict[str, object]:
        installed = self._management.is_installed()
        return {
            "provider_id": self.descriptor.id,
            "model_name": "GeoCLIP",
            "model_revision": self._management.manifest.model_version,
            "implementation_revision": self._management.manifest.implementation_revision,
            "source": "https://github.com/VicenteVivan/geo-clip",
            "license": self._management.manifest.license,
            "weights_source": (
                "official pinned PyPI wheel plus pinned openai/clip-vit-large-patch14"
            ),
            "installation_state": "installed" if installed else "not_installed",
            "verification_state": (
                "verified_at_runtime_load"
                if self._model is not None
                else "not_checked_this_process"
                if installed
                else "not_installed"
            ),
            "status": self._state,
            "device": (self._model.device if self._model is not None else self._selected_device),
            "dtype": None if self._model is None else self._model.dtype,
            "cache_scope": "private_operator_configured_outside_git",
            "supported_input_formats": ("jpeg", "png", "webp"),
            "input_preprocessing": ("orientation_normalized_rgb_then_pinned_clip_processor"),
            "output_semantics": "top_k_fixed_gallery_uncalibrated_relative_scores",
            "internal_top_k": self._internal_top_k,
            "raw_candidate_top_k": self._raw_candidate_top_k,
            "load_duration_ms": self._load_duration_ms,
            "calibration_state": "uncalibrated",
            "external_transfer": False,
            "reason_code": self._reason,
            "output_subreason_code": self._last_output_subreason,
            "last_output_structure": self._last_output_structure,
            "last_normalization": self._last_normalization,
            "coordinate_embedding_cache_entries": len(self._coordinate_cache),
            "coordinate_embedding_cache_hits": self._coordinate_cache_hits,
            "coordinate_embedding_cache_misses": self._coordinate_cache_misses,
        }

    async def score_coordinates(
        self,
        image_bytes: bytes,
        coordinates: Sequence[tuple[float, float]],
        *,
        cache_key: str,
        cancellation: asyncio.Event,
    ) -> tuple[float, ...]:
        """Score bounded arbitrary coordinates with the real GeoCLIP encoders.

        Location features are safe, image-independent and cached by a digest-bound
        key. Image features remain ephemeral inside the model call.
        """

        if cancellation.is_set():
            raise asyncio.CancelledError
        if not image_bytes or len(image_bytes) > _MAX_IMAGE_BYTES:
            raise ValueError("image exceeds provider size bound")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,119}", cache_key):
            raise ValueError("coordinate cache key is invalid")
        normalized = tuple(self._validate_coordinate(item) for item in coordinates)
        if not 1 <= len(normalized) <= 20_000:
            raise ValueError("coordinate batch is outside the bounded range")
        started = time.monotonic()
        acquired = False
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout)
            acquired = True
            model = await asyncio.wait_for(self._load(), timeout=self._timeout)
            location_features = await self._cached_coordinate_features(
                model, normalized, cache_key=cache_key
            )
            elapsed = time.monotonic() - started
            raw_scores = await asyncio.wait_for(
                asyncio.to_thread(model.score_coordinate_features, image_bytes, location_features),
                timeout=max(0.001, self._timeout - elapsed),
            )
            if cancellation.is_set():
                raise asyncio.CancelledError
            scores = _score_rows(raw_scores)
            native = tuple(_native_float(item) for item in scores)
            if len(native) != len(normalized) or any(not math.isfinite(item) for item in native):
                raise GeoClipOutputNormalizationError("coordinate_scores_invalid")
            return native
        finally:
            if acquired:
                self._semaphore.release()

    @staticmethod
    def _validate_coordinate(value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError("coordinates must contain latitude and longitude")
        latitude, longitude = float(value[0]), float(value[1])
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError("coordinates must be finite")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("coordinates are outside WGS84 bounds")
        return latitude, longitude

    async def _cached_coordinate_features(
        self,
        model: GeoClipRuntime,
        coordinates: tuple[tuple[float, float], ...],
        *,
        cache_key: str,
    ) -> Any:
        digest = hashlib.sha256(
            "".join(f"{lat:.7f},{lon:.7f};" for lat, lon in coordinates).encode()
        ).hexdigest()
        async with self._coordinate_cache_lock:
            cached = self._coordinate_cache.get(cache_key)
            if cached is not None and cached[0] == digest:
                self._coordinate_cache.move_to_end(cache_key)
                self._coordinate_cache_hits += 1
                return cached[1]
            self._coordinate_cache_misses += 1
            features = await asyncio.to_thread(model.encode_coordinates, coordinates)
            self._coordinate_cache[cache_key] = (digest, features)
            self._coordinate_cache.move_to_end(cache_key)
            while len(self._coordinate_cache) > 32:
                self._coordinate_cache.popitem(last=False)
            return features

    async def _load(self) -> GeoClipRuntime:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:
                return self._model
            if self._load_task is None or self._load_task.done():
                self._load_task = asyncio.create_task(self._initialize())
            task = self._load_task
        return await asyncio.shield(task)

    async def _initialize(self) -> GeoClipRuntime:
        self._state = "loading"
        started = time.monotonic()
        try:
            await asyncio.to_thread(self._management.verify)
            device = self._selected_device
            if device is None:
                device = await asyncio.to_thread(self._device_selector)
                self._selected_device = device
            model = await asyncio.to_thread(
                self._loader,
                self._management.snapshot_directory(),
                self._management.geoclip_assets_directory(),
                device,
            )
        except ModelManagementError as exc:
            self._state = "failed"
            self._reason = exc.code
            raise
        except ImportError as exc:
            self._state = "failed"
            self._reason = "missing_dependency"
            raise ModelManagementError("missing_dependency") from exc
        except Exception as exc:
            self._state = "failed"
            if _is_out_of_memory(exc):
                self._reason = "cuda_out_of_memory"
                await asyncio.to_thread(self._cuda_cleanup)
                raise ModelManagementError("cuda_out_of_memory") from exc
            self._reason = "provider_initialization_failed"
            raise ModelManagementError("provider_initialization_failed") from exc
        self._load_duration_ms = round((time.monotonic() - started) * 1000)
        self._model = model
        self._state = "ready"
        self._reason = None
        return model

    def _detach_inference(self, task: asyncio.Task[tuple[Any, Any]]) -> None:
        self._detached_inference_tasks.add(task)
        task.add_done_callback(self._finish_detached_inference)

    def _finish_detached_inference(self, task: asyncio.Task[tuple[Any, Any]]) -> None:
        self._detached_inference_tasks.discard(task)
        self._semaphore.release()
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if _is_out_of_memory(exc):
                self._model = None
                self._state = "failed"
                self._reason = "cuda_out_of_memory"
                asyncio.create_task(asyncio.to_thread(self._cuda_cleanup))

    @staticmethod
    def _read_image_payload(path: Path) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise ValueError("image is unavailable")
        declared_size = path.stat().st_size
        if declared_size <= 0 or declared_size > _MAX_IMAGE_BYTES:
            raise ValueError("image exceeds provider size bound")
        with path.open("rb") as source:
            payload = source.read(_MAX_IMAGE_BYTES + 1)
        if len(payload) != declared_size or len(payload) > _MAX_IMAGE_BYTES:
            raise ValueError("image exceeds provider size bound")
        return payload

    async def predict(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[GlobalPredictionResult]:
        started = time.monotonic()
        if context.cancellation.is_set():
            return ProviderOutcome.skipped("unavailable")
        if not self._management.is_installed():
            return ProviderOutcome.failed(
                "model_not_installed", retryable=False, attempts=0, duration_ms=0
            )
        remaining = (context.deadline - datetime.now(UTC)).total_seconds()
        timeout = min(self._timeout, max(0.001, remaining))
        acquired = False
        detached = False
        inference_task: asyncio.Task[tuple[Any, Any]] | None = None
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            acquired = True
            model = await asyncio.wait_for(self._load(), timeout=timeout)
            image_payload = self._read_image_payload(handle.path)
            elapsed = time.monotonic() - started
            inference_task = asyncio.create_task(
                asyncio.to_thread(model.predict, image_payload, self._raw_candidate_top_k)
            )
            prediction = await asyncio.wait_for(
                asyncio.shield(inference_task),
                timeout=max(0.001, timeout - elapsed),
            )
            if context.cancellation.is_set():
                return ProviderOutcome.skipped("unavailable")
            self._last_output_structure = _safe_output_structure(prediction)
            self._last_output_subreason = None
            self._last_normalization = None
            normalized = _normalize_predictions(
                prediction,
                device=model.device,
                dtype=model.dtype,
                inference_ms=round((time.monotonic() - started) * 1000),
                deduplication_radius_km=self._management.manifest.deduplication_radius_km,
                top_k=self._internal_top_k,
            )
            self._last_normalization = {
                key: int(value) for key, value in asdict(normalized.diagnostics).items()
            }
            return ProviderOutcome.succeeded(normalized.result)
        except TimeoutError:
            if inference_task is not None and not inference_task.done():
                self._detach_inference(inference_task)
                detached = True
            return ProviderOutcome.failed(
                "inference_timeout",
                retryable=True,
                attempts=1,
                duration_ms=round((time.monotonic() - started) * 1000),
            )
        except asyncio.CancelledError:
            if inference_task is not None and not inference_task.done():
                self._detach_inference(inference_task)
                detached = True
            raise
        except GeoClipOutputNormalizationError as exc:
            self._last_output_subreason = exc.subreason_code
            return ProviderOutcome.failed(
                "invalid_model_output",
                retryable=False,
                attempts=1,
                duration_ms=round((time.monotonic() - started) * 1000),
                subreason_code=exc.subreason_code,
            )
        except ModelManagementError as exc:
            code = {
                "checksum_mismatch": "checksum_mismatch",
                "weights_incomplete": "weights_incomplete",
                "unsafe_installation_path": "weights_incomplete",
                "cuda_out_of_memory": "cuda_out_of_memory",
                "unsupported_device": "unsupported_device",
                "missing_dependency": "missing_dependency",
                "provider_initialization_failed": "internal_provider_error",
            }.get(exc.code, "internal_provider_error")
            return ProviderOutcome.failed(
                code,
                retryable=False,
                attempts=1,
                duration_ms=round((time.monotonic() - started) * 1000),
            )
        except Exception as exc:
            name = type(exc).__name__.lower()
            message = str(exc).lower()
            if _is_out_of_memory(exc):
                code = "cuda_out_of_memory"
                self._model = None
                self._state = "failed"
                self._reason = code
                await asyncio.to_thread(self._cuda_cleanup)
            elif "image" in name or "decode" in message:
                code = "image_decode_error"
            else:
                code = "invalid_model_output"
                self._last_output_subreason = "upstream_runtime_exception"
            return ProviderOutcome.failed(
                code,
                retryable=code == "cuda_out_of_memory",
                attempts=1,
                duration_ms=round((time.monotonic() - started) * 1000),
                subreason_code=(
                    self._last_output_subreason if code == "invalid_model_output" else None
                ),
            )
        finally:
            if acquired and not detached:
                self._semaphore.release()


def _select_device() -> str:
    if importlib.util.find_spec("torch") is None:
        raise ImportError("torch")
    torch = importlib.import_module("torch")
    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def select_device(requested: str) -> str:
    if requested == "auto":
        return _select_device()
    if requested == "cpu":
        return "cpu"
    if requested != "cuda":
        raise ModelManagementError("unsupported_device")
    if _select_device() != "cuda":
        raise ModelManagementError("unsupported_device")
    return "cuda"


def _clear_cuda_cache() -> None:
    try:
        torch = importlib.import_module("torch")
        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


def _is_out_of_memory(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "outofmemory" in name or "out of memory" in message


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _safe_value_structure(value: object) -> dict[str, object]:
    result: dict[str, object] = {"type": _type_name(value)}
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            result["shape"] = [int(item) for item in shape]
        except (TypeError, ValueError):
            result["shape"] = "unavailable"
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        result["dtype"] = str(dtype)[:80]
    device = getattr(value, "device", None)
    if device is not None:
        result["device"] = str(device)[:80]
    return result


def _safe_output_structure(prediction: object) -> dict[str, object]:
    structure: dict[str, object] = {"return_type": _type_name(prediction)}
    if isinstance(prediction, list | tuple):
        structure["tuple_length"] = len(prediction)
        if prediction:
            structure["gps"] = _safe_value_structure(prediction[0])
        if len(prediction) > 1:
            structure["scores"] = _safe_value_structure(prediction[1])
    return structure


def _materialize(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        to_float = getattr(value, "float", None)
        if callable(to_float):
            value = to_float()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    return value


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _coordinate_rows(value: Any) -> list[Sequence[Any]]:
    materialized = _materialize(value)
    if not _is_sequence(materialized):
        raise GeoClipOutputNormalizationError("gps_shape_invalid")
    values = list(materialized)
    if (
        len(values) == 1
        and _is_sequence(values[0])
        and (batch := list(values[0]))
        and all(_is_sequence(item) for item in batch)
    ):
        values = batch
    elif len(values) == 2 and all(not _is_sequence(item) for item in values):
        values = [values]
    if not values or any(
        not _is_sequence(item)
        or len(item) != 2
        or any(_is_sequence(component) for component in item)
        for item in values
    ):
        raise GeoClipOutputNormalizationError("gps_shape_invalid")
    return [item for item in values if isinstance(item, Sequence)]


def _score_rows(value: Any) -> list[Any]:
    materialized = _materialize(value)
    if not _is_sequence(materialized):
        return [materialized]
    values = list(materialized)
    if len(values) == 1 and _is_sequence(values[0]):
        values = list(values[0])
    if not values or any(_is_sequence(item) for item in values):
        raise GeoClipOutputNormalizationError("score_shape_invalid")
    return values


def _native_float(value: Any) -> float:
    materialized = _materialize(value)
    if _is_sequence(materialized):
        raise TypeError("scalar expected")
    return float(materialized)


def _invalid_candidate_subreason(
    *,
    candidate_count: int,
    coordinate_non_finite: int,
    coordinate_out_of_range: int,
    score_non_finite: int,
) -> str:
    if coordinate_non_finite == candidate_count:
        return "all_coordinates_non_finite"
    if coordinate_out_of_range == candidate_count:
        return "all_coordinates_out_of_range"
    if score_non_finite == candidate_count:
        return "all_scores_non_finite"
    return "schema_validation_failed"


def _normalize_predictions(
    prediction: Any,
    *,
    device: str,
    dtype: str,
    inference_ms: int,
    deduplication_radius_km: float,
    top_k: int,
    minimum_hypotheses: int = 3,
) -> NormalizedGeoClipOutput:
    if not isinstance(prediction, list | tuple):
        raise GeoClipOutputNormalizationError("unsupported_return_type")
    if len(prediction) != 2:
        raise GeoClipOutputNormalizationError("invalid_return_arity")
    if not 1 <= minimum_hypotheses <= top_k <= 100:
        raise GeoClipOutputNormalizationError("schema_validation_failed")
    coordinate_values = _coordinate_rows(prediction[0])
    score_values = _score_rows(prediction[1])
    if len(coordinate_values) != len(score_values):
        raise GeoClipOutputNormalizationError("candidate_count_mismatch")

    coordinate_non_finite = 0
    coordinate_out_of_range = 0
    score_non_finite = 0
    score_out_of_range = 0
    scalar_conversion = 0
    valid: list[_CandidateRow] = []
    for original_rank, (coordinate, score) in enumerate(
        zip(coordinate_values, score_values, strict=True), start=1
    ):
        try:
            latitude = _native_float(coordinate[0])
            longitude = _native_float(coordinate[1])
            raw_score = _native_float(score)
        except (TypeError, ValueError, OverflowError):
            scalar_conversion += 1
            continue
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            coordinate_non_finite += 1
            continue
        if not -90 <= latitude <= 90:
            coordinate_out_of_range += 1
            continue
        if longitude < -180 or longitude > 180:
            longitude = ((longitude + 180.0) % 360.0) - 180.0
        if not math.isfinite(raw_score):
            score_non_finite += 1
            continue
        if not 0 <= raw_score <= 1:
            score_out_of_range += 1
            continue
        valid.append(
            _CandidateRow(
                original_rank=original_rank,
                latitude=float(latitude),
                longitude=float(longitude),
                raw_score=float(raw_score),
            )
        )

    if len(valid) < minimum_hypotheses:
        subreason = (
            _invalid_candidate_subreason(
                candidate_count=len(coordinate_values),
                coordinate_non_finite=coordinate_non_finite,
                coordinate_out_of_range=coordinate_out_of_range,
                score_non_finite=score_non_finite,
            )
            if not valid
            else "insufficient_valid_hypotheses"
        )
        raise GeoClipOutputNormalizationError(subreason)

    diverse: list[_CandidateRow] = []
    for candidate in valid:
        if all(
            _haversine_km(
                candidate.latitude,
                candidate.longitude,
                item.latitude,
                item.longitude,
            )
            >= deduplication_radius_km
            for item in diverse
        ):
            diverse.append(candidate)
        if len(diverse) == top_k:
            break

    selected = list(diverse)
    selected_ranks = {item.original_rank for item in selected}
    for candidate in valid:
        if len(selected) == top_k:
            break
        if candidate.original_rank in selected_ranks:
            continue
        if all(
            _haversine_km(
                candidate.latitude,
                candidate.longitude,
                item.latitude,
                item.longitude,
            )
            >= _MINIMUM_DISTINCT_RADIUS_KM
            for item in selected
        ):
            selected.append(candidate)
            selected_ranks.add(candidate.original_rank)
    if len(selected) < minimum_hypotheses:
        raise GeoClipOutputNormalizationError("post_deduplication_insufficient")
    selected.sort(key=lambda item: item.original_rank)
    selected = selected[:top_k]

    try:
        hypotheses = [
            GlobalPredictionHypothesis(
                rank=rank,
                original_rank=item.original_rank,
                latitude=item.latitude,
                longitude=item.longitude,
                raw_score=item.raw_score,
                score_type="uncalibrated_gallery_softmax",
                normalization_method="softmax_over_fixed_gallery",
                calibration_state="uncalibrated",
                limitations=_LIMITATIONS,
            )
            for rank, item in enumerate(selected, start=1)
        ]
        result = GlobalPredictionResult(
            provider_id="geoclip-global-v1",
            model_name="GeoCLIP",
            model_revision="1.2.0",
            implementation_revision="7a1a23b49648a5872a771cfda28490a17ab17d15",
            device=str(device),
            dtype=str(dtype),
            inference_ms=int(inference_ms),
            hypotheses=hypotheses,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise GeoClipOutputNormalizationError("schema_validation_failed") from exc
    diagnostics = GeoClipNormalizationDiagnostics(
        input_candidate_count=len(coordinate_values),
        valid_candidate_count=len(valid),
        rejected_coordinate_non_finite=coordinate_non_finite,
        rejected_coordinate_out_of_range=coordinate_out_of_range,
        rejected_score_non_finite=score_non_finite,
        rejected_score_out_of_range=score_out_of_range,
        rejected_scalar_conversion=scalar_conversion,
        configured_dedup_survivors=len(diverse),
        minimum_diversity_backfill=max(0, len(selected) - len(diverse)),
        output_candidate_count=len(hypotheses),
    )
    return NormalizedGeoClipOutput(result=result, diagnostics=diagnostics)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 6371.0088 * 2 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1 - value)))
