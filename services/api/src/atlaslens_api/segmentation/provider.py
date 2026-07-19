from __future__ import annotations

import asyncio
import importlib
import importlib.util
import io
import json
import math
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from atlaslens_api.providers.base import (
    InvocationContext,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.segmentation.aggregation import (
    aggregate_segmentation,
    default_scene_group_config_path,
    load_scene_group_config,
)
from atlaslens_api.segmentation.models import (
    DeploymentMetadata,
    RawSegmentationPrediction,
    SceneGroupConfiguration,
    SegmentationProviderStatus,
    SegmentationResult,
)
from atlaslens_api.storage import LocalImageHandle

_PROVIDER_ID = "atlaslens-segformer-b2-v4"
_PROVIDER_VERSION = "phase6a-v1"
_METADATA_FILENAME = "deployment_metadata.json"
_MAX_METADATA_BYTES = 256 * 1024
_MAX_WEIGHT_INDEX_BYTES = 2 * 1024 * 1024


class SegmentationProviderError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SegmentationRuntime(Protocol):
    @property
    def device(self) -> Literal["cpu", "cuda"]: ...

    @property
    def num_labels(self) -> int: ...

    @property
    def id2label(self) -> Mapping[int, str]: ...

    @property
    def semantic_label_names_available(self) -> bool: ...

    def predict(self, image_payload: bytes) -> RawSegmentationPrediction: ...


class SceneSegmentationProvider(Protocol):
    descriptor: ProviderDescriptor

    def status(self) -> SegmentationProviderStatus: ...

    async def analyze(
        self,
        handle: LocalImageHandle,
        context: InvocationContext,
    ) -> ProviderOutcome[SegmentationResult]: ...


RuntimeLoader = Callable[[Path, str, DeploymentMetadata], SegmentationRuntime]
DeviceSelector = Callable[[str], str]
CudaCleanup = Callable[[], None]


class HuggingFaceSegmentationRuntime:
    """Local-only runtime over an already prepared Hugging Face directory."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        torch_module: Any,
        functional: Any,
        device: Literal["cpu", "cuda"],
        num_labels: int,
        id2label: Mapping[int, str],
        semantic_label_names_available: bool,
    ) -> None:
        if device not in {"cpu", "cuda"} or num_labels <= 0:
            raise SegmentationProviderError("invalid_runtime_configuration")
        if set(id2label) != set(range(num_labels)):
            raise SegmentationProviderError("invalid_label_mapping")
        model.eval()
        moved = model.to(device)
        self._model = model if moved is None else moved
        self._processor = processor
        self._torch = torch_module
        self._functional = functional
        self.device = device
        self.num_labels = num_labels
        self.id2label = dict(id2label)
        self.semantic_label_names_available = semantic_label_names_available

    def predict(self, image_payload: bytes) -> RawSegmentationPrediction:
        started = time.monotonic()
        try:
            with Image.open(io.BytesIO(image_payload)) as source:
                image = source.convert("RGB")
                width, height = image.size
                inputs = self._processor(images=image, return_tensors="pt")
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise SegmentationProviderError("image_decode_error") from exc
        if not isinstance(inputs, Mapping) or not inputs:
            raise SegmentationProviderError("invalid_processor_output")
        try:
            prepared_inputs = {str(name): tensor.to(self.device) for name, tensor in inputs.items()}
            with self._torch.inference_mode():
                if self.device == "cuda":
                    with self._torch.autocast(device_type="cuda", dtype=self._torch.float16):
                        output = self._model(**prepared_inputs)
                else:
                    output = self._model(**prepared_inputs)
                logits = output.logits
                shape = tuple(int(value) for value in logits.shape)
                if len(shape) != 4 or shape[0] != 1 or shape[1] != self.num_labels:
                    raise SegmentationProviderError("invalid_logits_shape")
                upsampled = self._functional.interpolate(
                    logits,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                class_ids = upsampled.argmax(dim=1)[0].detach().to("cpu")
                counts_value = self._torch.bincount(
                    class_ids.reshape(-1).to(dtype=self._torch.int64),
                    minlength=self.num_labels,
                ).tolist()
        except SegmentationProviderError:
            raise
        except Exception as exc:
            if _is_out_of_memory(exc):
                raise SegmentationProviderError("cuda_out_of_memory") from exc
            raise SegmentationProviderError("inference_failed") from exc
        try:
            counts = tuple(int(value) for value in counts_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SegmentationProviderError("invalid_pixel_counts") from exc
        return RawSegmentationPrediction(
            image_width=width,
            image_height=height,
            class_pixel_counts=counts,
            inference_ms=round((time.monotonic() - started) * 1000),
        )


def _default_runtime_loader(
    model_directory: Path,
    device: str,
    metadata: DeploymentMetadata,
) -> SegmentationRuntime:
    try:
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise SegmentationProviderError("missing_dependency") from exc
    try:
        processor = transformers.AutoImageProcessor.from_pretrained(
            model_directory,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=False,
        )
        model = transformers.AutoModelForSemanticSegmentation.from_pretrained(
            model_directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
    except Exception as exc:
        raise SegmentationProviderError("prepared_model_load_failed") from exc
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "segformer":
        raise SegmentationProviderError("model_family_mismatch")
    if int(getattr(config, "num_labels", -1)) != metadata.num_labels:
        raise SegmentationProviderError("classifier_dimension_mismatch")
    raw_labels = getattr(config, "id2label", None)
    if not isinstance(raw_labels, Mapping):
        raise SegmentationProviderError("invalid_label_mapping")
    try:
        id2label = {int(key): str(value).strip() for key, value in raw_labels.items()}
    except (TypeError, ValueError) as exc:
        raise SegmentationProviderError("invalid_label_mapping") from exc
    if set(id2label) != set(range(metadata.num_labels)) or any(
        not value for value in id2label.values()
    ):
        raise SegmentationProviderError("invalid_label_mapping")
    return HuggingFaceSegmentationRuntime(
        model=model,
        processor=processor,
        torch_module=torch,
        functional=functional,
        device=_validated_device(device),
        num_labels=metadata.num_labels,
        id2label=id2label,
        semantic_label_names_available=metadata.semantic_label_names_available,
    )


def select_segmentation_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise SegmentationProviderError("unsupported_device")
    if importlib.util.find_spec("torch") is None:
        raise SegmentationProviderError("missing_dependency")
    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
    except (ImportError, RuntimeError) as exc:
        raise SegmentationProviderError("missing_dependency") from exc
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise SegmentationProviderError("unsupported_device")
    return requested


def _validated_device(value: str) -> Literal["cpu", "cuda"]:
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        return "cuda"
    raise SegmentationProviderError("unsupported_device")


@dataclass(frozen=True, slots=True)
class _PreparedModel:
    installed: bool
    prepared: bool
    metadata: DeploymentMetadata | None
    reason_code: str | None


def _safe_regular_file(path: Path, *, maximum_bytes: int | None = None) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        size = path.stat().st_size
    except OSError:
        return False
    return size > 0 and (maximum_bytes is None or size <= maximum_bytes)


def _has_safe_weights(directory: Path) -> bool:
    weights = directory / "model.safetensors"
    if _safe_regular_file(weights):
        return True
    index = directory / "model.safetensors.index.json"
    if not _safe_regular_file(index, maximum_bytes=_MAX_WEIGHT_INDEX_BYTES):
        return False
    try:
        payload = json.loads(index.read_bytes())
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    filenames = set(weight_map.values())
    if any(
        not isinstance(filename, str)
        or PurePosixPath(filename).name != filename
        or "\\" in filename
        or not filename.endswith(".safetensors")
        for filename in filenames
    ):
        return False
    return all(_safe_regular_file(directory / str(filename)) for filename in filenames)


def _inspect_prepared_model(model_directory: Path) -> _PreparedModel:
    requested = model_directory.expanduser()
    try:
        installed = requested.is_dir() and not requested.is_symlink()
    except OSError:
        installed = False
    if not installed:
        return _PreparedModel(False, False, None, "model_not_installed")
    try:
        directory = requested.resolve(strict=True)
    except OSError:
        return _PreparedModel(True, False, None, "weights_incomplete")
    metadata_path = directory / _METADATA_FILENAME
    required = (directory / "config.json", directory / "preprocessor_config.json")
    if not _safe_regular_file(metadata_path, maximum_bytes=_MAX_METADATA_BYTES) or any(
        not _safe_regular_file(path, maximum_bytes=2 * 1024 * 1024) for path in required
    ):
        return _PreparedModel(True, False, None, "weights_incomplete")
    if not _has_safe_weights(directory):
        return _PreparedModel(True, False, None, "weights_incomplete")
    try:
        metadata = DeploymentMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError):
        return _PreparedModel(True, False, None, "deployment_metadata_invalid")
    return _PreparedModel(True, True, metadata, None)


class SegFormerSceneProvider:
    """Optional local semantic evidence provider; it never predicts geography."""

    def __init__(
        self,
        *,
        enabled: bool,
        model_directory: Path,
        requested_device: str = "auto",
        minimum_class_ratio: float = 0.001,
        maximum_dominant_classes: int = 20,
        timeout_seconds: float = 90.0,
        max_input_bytes: int = 20 * 1024 * 1024,
        max_decoded_pixels: int = 40_000_000,
        max_image_dimension: int = 16_384,
        scene_group_config: SceneGroupConfiguration | None = None,
        scene_group_config_path: Path | None = None,
        runtime_loader: RuntimeLoader = _default_runtime_loader,
        device_selector: DeviceSelector = select_segmentation_device,
        cuda_cleanup: CudaCleanup | None = None,
    ) -> None:
        if not 0 <= minimum_class_ratio <= 1 or not math.isfinite(minimum_class_ratio):
            raise ValueError("minimum segmentation class ratio must be finite and bounded")
        if not 1 <= maximum_dominant_classes <= 32:
            raise ValueError("dominant-class output bound must be between one and 32")
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("segmentation timeout must be positive and finite")
        if max_input_bytes <= 0 or max_decoded_pixels <= 0 or max_image_dimension <= 0:
            raise ValueError("segmentation image bounds must be positive")
        self._enabled = enabled
        self._model_directory = model_directory.expanduser()
        self._minimum_class_ratio = minimum_class_ratio
        self._maximum_dominant_classes = maximum_dominant_classes
        self._timeout = timeout_seconds
        self._max_input_bytes = max_input_bytes
        self._max_decoded_pixels = max_decoded_pixels
        self._max_image_dimension = max_image_dimension
        self._runtime_loader = runtime_loader
        self._cuda_cleanup = cuda_cleanup or _clear_cuda_cache
        self._load_lock = asyncio.Lock()
        self._inference_slot = asyncio.Semaphore(1)
        self._runtime: SegmentationRuntime | None = None
        self._load_task: asyncio.Task[SegmentationRuntime] | None = None
        self._detached_tasks: set[asyncio.Task[RawSegmentationPrediction]] = set()
        self._prepared = _inspect_prepared_model(self._model_directory)
        self._reason: str | None = self._prepared.reason_code
        self._scene_groups: SceneGroupConfiguration | None
        try:
            self._scene_groups = scene_group_config or load_scene_group_config(
                scene_group_config_path or default_scene_group_config_path()
            )
        except ValueError:
            self._scene_groups = None
            self._reason = "scene_group_config_invalid"
        self._device: Literal["cpu", "cuda"] | None = None
        self._state: Literal[
            "ready",
            "not_installed",
            "incomplete",
            "loading",
            "failed",
            "disabled",
            "unavailable",
        ]
        if not enabled:
            self._state = "disabled"
            self._reason = "disabled"
        elif not self._prepared.installed:
            self._state = "not_installed"
        elif not self._prepared.prepared or self._scene_groups is None:
            self._state = "incomplete"
        else:
            try:
                self._device = _validated_device(device_selector(requested_device))
            except SegmentationProviderError as exc:
                self._state = "unavailable"
                self._reason = exc.code
            except (ImportError, RuntimeError, ValueError):
                self._state = "unavailable"
                self._reason = "unsupported_device"
            else:
                self._state = "ready"
                self._reason = None
        self.descriptor = ProviderDescriptor(
            id=_PROVIDER_ID,
            kind="scene_segmentation",
            version=_PROVIDER_VERSION,
            execution_boundary="local",
            criticality="optional",
            available=self._state == "ready",
            unavailable_reason_code=self._reason,
            model_name="AtlasLens SegFormer-B2 v4",
        )

    def status(self) -> SegmentationProviderStatus:
        metadata = self._prepared.metadata
        usable = (
            self._enabled
            and self._prepared.prepared
            and self._scene_groups is not None
            and self._device is not None
            and self._state in {"ready", "loading"}
        )
        return SegmentationProviderStatus(
            provider_id=_PROVIDER_ID,
            status=self._state,
            enabled=self._enabled,
            installed=self._prepared.installed,
            prepared=self._prepared.prepared,
            loaded=self._runtime is not None,
            usable=usable,
            device=self._device,
            weight_source=metadata.weight_source if metadata is not None else None,
            num_labels=metadata.num_labels if metadata is not None else None,
            semantic_label_names_available=(
                metadata.semantic_label_names_available if metadata is not None else None
            ),
            checkpoint_sha256=(metadata.source_checkpoint_sha256 if metadata is not None else None),
            load_error=self._reason,
        )

    async def _load(self) -> SegmentationRuntime:
        if self._runtime is not None:
            return self._runtime
        async with self._load_lock:
            if self._runtime is not None:
                return self._runtime
            if self._load_task is None:
                self._load_task = asyncio.create_task(self._initialize())
            task = self._load_task
        return await asyncio.shield(task)

    async def _initialize(self) -> SegmentationRuntime:
        metadata = self._prepared.metadata
        if metadata is None or self._device is None:
            raise SegmentationProviderError(self._reason or "model_not_installed")
        self._state = "loading"
        try:
            runtime = await asyncio.to_thread(
                self._runtime_loader,
                self._model_directory.resolve(strict=True),
                self._device,
                metadata,
            )
            if (
                runtime.device != self._device
                or runtime.num_labels != metadata.num_labels
                or runtime.semantic_label_names_available != metadata.semantic_label_names_available
                or set(runtime.id2label) != set(range(metadata.num_labels))
            ):
                raise SegmentationProviderError("runtime_metadata_mismatch")
        except SegmentationProviderError as exc:
            self._state = "failed"
            self._reason = exc.code
            raise
        except Exception as exc:
            self._state = "failed"
            self._reason = "provider_initialization_failed"
            raise SegmentationProviderError("provider_initialization_failed") from exc
        self._runtime = runtime
        self._state = "ready"
        self._reason = None
        return runtime

    async def analyze(
        self,
        handle: LocalImageHandle,
        context: InvocationContext,
    ) -> ProviderOutcome[SegmentationResult]:
        started = time.monotonic()
        if not self._enabled:
            return ProviderOutcome.skipped("disabled")
        if not self._prepared.prepared or self._device is None or self._scene_groups is None:
            return ProviderOutcome.skipped(
                _public_failure_code(self._reason or "model_not_installed")
            )
        if context.cancellation.is_set():
            return ProviderOutcome.skipped("unavailable")
        try:
            payload, expected_width, expected_height = await asyncio.to_thread(
                self._read_bounded_image, handle.path
            )
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            OSError,
            SyntaxError,
            UnidentifiedImageError,
            ValueError,
        ):
            return ProviderOutcome.failed(
                "unsupported_input",
                retryable=False,
                attempts=1,
                duration_ms=_elapsed_ms(started),
                subreason_code="segmentation_image_bounds_or_decode_invalid",
            )

        remaining = (context.deadline - datetime.now(UTC)).total_seconds()
        timeout = min(self._timeout, max(0.001, remaining))
        acquired = False
        detached = False
        inference_task: asyncio.Task[RawSegmentationPrediction] | None = None
        try:
            await asyncio.wait_for(self._inference_slot.acquire(), timeout=timeout)
            acquired = True
            runtime = await asyncio.wait_for(self._load(), timeout=timeout)
            elapsed = time.monotonic() - started
            inference_task = asyncio.create_task(asyncio.to_thread(runtime.predict, payload))
            prediction = await asyncio.wait_for(
                asyncio.shield(inference_task),
                timeout=max(0.001, timeout - elapsed),
            )
            del payload
            if context.cancellation.is_set():
                return ProviderOutcome.skipped("unavailable")
            if (
                prediction.image_width != expected_width
                or prediction.image_height != expected_height
                or len(prediction.class_pixel_counts) != runtime.num_labels
            ):
                raise SegmentationProviderError("runtime_output_mismatch")
            result = aggregate_segmentation(
                prediction,
                provider_id=_PROVIDER_ID,
                device=runtime.device,
                id2label=runtime.id2label,
                semantic_label_names_available=runtime.semantic_label_names_available,
                minimum_class_ratio=self._minimum_class_ratio,
                scene_group_config=self._scene_groups,
                maximum_dominant_classes=self._maximum_dominant_classes,
            )
            return ProviderOutcome.succeeded(result)
        except TimeoutError:
            if inference_task is not None and not inference_task.done():
                self._detach_inference(inference_task)
                detached = True
            return ProviderOutcome.failed(
                "inference_timeout",
                retryable=True,
                attempts=1,
                duration_ms=_elapsed_ms(started),
            )
        except asyncio.CancelledError:
            if inference_task is not None and not inference_task.done():
                self._detach_inference(inference_task)
                detached = True
            raise
        except SegmentationProviderError as exc:
            if exc.code == "cuda_out_of_memory":
                self._runtime = None
                self._state = "failed"
                self._reason = exc.code
                await asyncio.to_thread(self._cuda_cleanup)
            return ProviderOutcome.failed(
                _public_failure_code(exc.code),
                retryable=exc.code == "cuda_out_of_memory",
                attempts=1,
                duration_ms=_elapsed_ms(started),
                subreason_code=_safe_subreason(exc.code),
            )
        except (TypeError, ValueError):
            return ProviderOutcome.failed(
                "invalid_model_output",
                retryable=False,
                attempts=1,
                duration_ms=_elapsed_ms(started),
                subreason_code="segmentation_aggregation_invalid",
            )
        except Exception as exc:
            if _is_out_of_memory(exc):
                self._runtime = None
                self._state = "failed"
                self._reason = "cuda_out_of_memory"
                await asyncio.to_thread(self._cuda_cleanup)
                code = "cuda_out_of_memory"
            else:
                code = "internal_provider_error"
            return ProviderOutcome.failed(
                code,
                retryable=code == "cuda_out_of_memory",
                attempts=1,
                duration_ms=_elapsed_ms(started),
            )
        finally:
            if acquired and not detached:
                self._inference_slot.release()

    def _detach_inference(self, task: asyncio.Task[RawSegmentationPrediction]) -> None:
        self._detached_tasks.add(task)
        task.add_done_callback(self._finish_detached_inference)

    def _finish_detached_inference(self, task: asyncio.Task[RawSegmentationPrediction]) -> None:
        self._detached_tasks.discard(task)
        self._inference_slot.release()
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if _is_out_of_memory(exc):
                self._runtime = None
                self._state = "failed"
                self._reason = "cuda_out_of_memory"
                asyncio.create_task(asyncio.to_thread(self._cuda_cleanup))

    def _read_bounded_image(self, path: Path) -> tuple[bytes, int, int]:
        if path.is_symlink() or not path.is_file():
            raise ValueError("segmentation image is unavailable")
        declared_size = path.stat().st_size
        if not 0 < declared_size <= self._max_input_bytes:
            raise ValueError("segmentation image byte bound exceeded")
        with path.open("rb") as source:
            payload = source.read(self._max_input_bytes + 1)
        if len(payload) != declared_size or len(payload) > self._max_input_bytes:
            raise ValueError("segmentation image changed while being read")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                frames = int(getattr(image, "n_frames", 1))
                if frames != 1 or bool(getattr(image, "is_animated", False)):
                    raise ValueError("animated images are unsupported")
                if (
                    width <= 0
                    or height <= 0
                    or max(width, height) > self._max_image_dimension
                    or width * height > self._max_decoded_pixels
                ):
                    raise ValueError("segmentation image dimension bound exceeded")
                image.verify()
        return payload, width, height


def _public_failure_code(code: str) -> str:
    return {
        "missing_dependency": "missing_dependency",
        "unsupported_device": "unsupported_device",
        "image_decode_error": "image_decode_error",
        "cuda_out_of_memory": "cuda_out_of_memory",
        "weights_incomplete": "weights_incomplete",
        "model_not_installed": "model_not_installed",
    }.get(code, "invalid_model_output")


def _safe_subreason(value: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in value.casefold())
    return ("_".join(part for part in safe.split("_") if part) or "segmentation_failed")[:120]


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def _is_out_of_memory(exc: BaseException) -> bool:
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    return "outofmemory" in name or "out of memory" in message


def _clear_cuda_cache() -> None:
    try:
        torch = importlib.import_module("torch")
        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return
