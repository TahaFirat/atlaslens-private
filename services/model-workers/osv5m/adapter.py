from __future__ import annotations

import gc
import json
import math
import os
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Mapping

from common.protocol import WorkerAdapterError


class ModelAdapterError(WorkerAdapterError):
    """Error safe to translate into the localhost worker protocol."""

    def __init__(self, code: str, message: str) -> None:
        del message
        if code == "model_not_loaded":
            status = 409
        elif code in {
            "device_mismatch",
            "invalid_device",
            "invalid_image",
            "invalid_parameters",
            "unsupported_model",
        }:
            status = 422
        else:
            status = 503
        super().__init__(code, status=status)


@dataclass(frozen=True, slots=True)
class OSV5MArtifacts:
    source_dir: Path
    model_dir: Path
    source_revision: str
    model_revision: str
    model_id: str = "osv5m/baseline"


def patch_offline_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy and validate the architecture used by the complete local checkpoint."""

    payload = json.loads(json.dumps(config))
    try:
        backbone = payload["model"]["backbone"]["instance"]
        target = backbone["_target_"]
    except (KeyError, TypeError) as exc:
        raise ModelAdapterError("invalid_model_config", "OSV-5M config is invalid") from exc
    if target != "models.networks.backbones.CLIP":
        raise ModelAdapterError(
            "unsupported_model_config",
            "OSV-5M backbone configuration is unsupported",
        )
    # The official checkpoint contains the complete 24-layer CLIP-L backbone.
    # Its source constructor nevertheless calls from_pretrained(path) before
    # loading that state dict. The adapter replaces the constructor in-memory
    # during model creation, so this value is never resolved or downloaded.
    backbone["path"] = "atlaslens-offline-complete-checkpoint"
    return payload


def adapt_osv5m_state_dict(value: object) -> dict[str, object]:
    """Adapt the pinned checkpoint to Transformers 5.13's CLIP key layout."""

    if not isinstance(value, Mapping) or not value:
        raise ModelAdapterError("invalid_model_weights", "OSV-5M weights are invalid")
    legacy_buffers = {
        f"{prefix}.{name}"
        for prefix in ("model.head", "head")
        for name in ("cell_center", "cell_size_up", "cell_size_down")
    }
    adapted: dict[str, object] = {}
    for key, tensor in value.items():
        if not isinstance(key, str):
            raise ModelAdapterError("invalid_model_weights", "OSV-5M weights are invalid")
        if key in legacy_buffers:
            continue
        current = key.replace("clip.vision_model.", "clip.")
        if current in adapted:
            raise ModelAdapterError("invalid_model_weights", "OSV-5M weight keys collide")
        adapted[current] = tensor
    if not adapted:
        raise ModelAdapterError("invalid_model_weights", "OSV-5M weights are invalid")
    return adapted


def normalize_radian_prediction(value: object) -> tuple[float, float]:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ModelAdapterError("invalid_model_output", "OSV-5M output shape is invalid")
    latitude_raw, longitude_raw = value
    if (
        isinstance(latitude_raw, bool)
        or isinstance(longitude_raw, bool)
        or not isinstance(latitude_raw, (int, float))
        or not isinstance(longitude_raw, (int, float))
    ):
        raise ModelAdapterError("invalid_model_output", "OSV-5M output is not numeric")
    latitude, longitude = float(latitude_raw), float(longitude_raw)
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ModelAdapterError("invalid_model_output", "OSV-5M output is not finite")
    if not -math.pi / 2 <= latitude <= math.pi / 2 or not -math.pi <= longitude <= math.pi:
        raise ModelAdapterError("invalid_model_output", "OSV-5M output is outside WGS84")
    return latitude, longitude


def _is_cuda_oom(exc: BaseException) -> bool:
    return "cuda" in str(exc).casefold() and "out of memory" in str(exc).casefold()


class OSV5MAdapter:
    provider = "osv5m"

    def __init__(
        self,
        artifacts: OSV5MArtifacts,
        *,
        max_input_bytes: int = 20 * 1024 * 1024,
        max_image_pixels: int = 40_000_000,
    ) -> None:
        if max_input_bytes <= 0 or max_image_pixels <= 0:
            raise ValueError("image bounds must be positive")
        self._artifacts = artifacts
        self._max_input_bytes = max_input_bytes
        self._max_image_pixels = max_image_pixels
        self._lock = threading.RLock()
        self._model: object | None = None
        self._torch: object | None = None
        self._device: str | None = None
        self._load_ms: int | None = None
        self._import_ok = False
        self._load_verified = False
        self._inference_verified = False
        self._last_error: str | None = None

    def health(self) -> dict[str, object]:
        with self._lock:
            import_ok = self._import_ok or self._probe_imports()
            return {
                "provider": self.provider,
                "model_id": self._artifacts.model_id,
                "model_revision": self._artifacts.model_revision,
                "source_revision": self._artifacts.source_revision,
                "import_ok": import_ok,
                "weights_available": self._weights_available(),
                "model_loaded": self._model is not None,
                "load_verified": self._load_verified,
                "real_inference_verified": self._inference_verified,
                "device": self._device,
                "load_ms": self._load_ms,
                "last_error": self._last_error,
            }

    def load(self, parameters: Mapping[str, object]) -> dict[str, object]:
        device = _string_parameter(parameters, "device", required=True)
        selected_device = _normalize_device(device)
        with self._lock:
            if self._model is not None and self._device == selected_device:
                return self.health()
            self.unload({})
            if not self._weights_available():
                self._last_error = "pretrained_weights_missing"
                raise ModelAdapterError(
                    "pretrained_weights_missing",
                    "OSV-5M inference artifacts are incomplete",
                )
            started = time.perf_counter()
            try:
                _force_offline_mode()
                self._prepare_import_path()
                import torch
                from models.huggingface import Geolocalizer

                self._import_ok = True
                _validate_device(torch, selected_device)
                config_path = self._artifacts.model_dir / "config.json"
                config = patch_offline_config(json.loads(config_path.read_text(encoding="utf-8")))
                with _working_directory(self._artifacts.source_dir):
                    with _offline_clip_constructor():
                        model = Geolocalizer(config)
                    state = torch.load(
                        self._artifacts.model_dir / "pytorch_model.bin",
                        map_location="cpu",
                        weights_only=True,
                        mmap=True,
                    )
                    adapted_state = adapt_osv5m_state_dict(state)
                    model.load_state_dict(adapted_state, strict=True)
                    del state, adapted_state
                model.requires_grad_(False).eval().to(selected_device)
                self._model = model
                self._torch = torch
                self._device = selected_device
                self._load_ms = _elapsed_ms(started)
                self._load_verified = True
                self._last_error = None
                return self.health()
            except ModelAdapterError as exc:
                self._last_error = exc.code
                self.unload({})
                raise
            except Exception as exc:
                code = "cuda_out_of_memory" if _is_cuda_oom(exc) else "model_load_failed"
                self._last_error = code
                self.unload({})
                raise ModelAdapterError(code, "OSV-5M model load failed") from exc

    def infer(
        self,
        image_bytes: bytes,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        with self._lock:
            if self._model is None or self._torch is None or self._device is None:
                raise ModelAdapterError("model_not_loaded", "OSV-5M model is not loaded")
            requested_device = _string_parameter(parameters, "device", required=False)
            if requested_device is not None and requested_device != self._device:
                raise ModelAdapterError("device_mismatch", "OSV-5M loaded device does not match")
            requested_model = _string_parameter(parameters, "model_id", required=False)
            if requested_model is not None and requested_model != self._artifacts.model_id:
                raise ModelAdapterError("unsupported_model", "OSV-5M model id is unsupported")
            image = _decode_image(
                image_bytes,
                max_input_bytes=self._max_input_bytes,
                max_pixels=self._max_image_pixels,
            )
            started = time.perf_counter()
            try:
                with self._torch.inference_mode():  # type: ignore[attr-defined]
                    tensor = self._model.transform(image).unsqueeze(0).to(self._device)  # type: ignore[attr-defined]
                    prediction = self._model(tensor)  # type: ignore[operator]
                latitude, longitude = normalize_radian_prediction(prediction)
            except ModelAdapterError:
                self._last_error = "invalid_model_output"
                raise
            except Exception as exc:
                code = "cuda_out_of_memory" if _is_cuda_oom(exc) else "inference_failed"
                self._last_error = code
                raise ModelAdapterError(code, "OSV-5M inference failed") from exc
            finally:
                image.close()
            inference_ms = _elapsed_ms(started)
            self._inference_verified = True
            self._last_error = None
            return {
                "provider": self.provider,
                "model_id": self._artifacts.model_id,
                "model_revision": self._artifacts.model_revision,
                "source_revision": self._artifacts.source_revision,
                "device": self._device,
                "coordinates_radians": [latitude, longitude],
                "coordinate_order": "latitude_longitude",
                "units": "radians",
                "load_ms": self._load_ms,
                "inference_ms": inference_ms,
            }

    def unload(self, parameters: Mapping[str, object]) -> dict[str, object]:
        del parameters
        with self._lock:
            model, torch_module = self._model, self._torch
            self._model = None
            self._torch = None
            self._device = None
            self._load_ms = None
            if model is not None:
                del model
            gc.collect()
            if torch_module is not None:
                try:
                    if torch_module.cuda.is_available():  # type: ignore[attr-defined]
                        torch_module.cuda.empty_cache()  # type: ignore[attr-defined]
                except RuntimeError:
                    pass
            return self.health()

    def _prepare_import_path(self) -> None:
        source = str(self._artifacts.source_dir.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)

    def _probe_imports(self) -> bool:
        try:
            _force_offline_mode()
            self._prepare_import_path()
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            from models.huggingface import Geolocalizer  # noqa: F401
        except Exception:
            return False
        self._import_ok = True
        return True

    def _weights_available(self) -> bool:
        source_files = (
            self._artifacts.source_dir / "models" / "huggingface.py",
            self._artifacts.source_dir / "utils" / "quadtree_10_1000.csv",
        )
        model_files = (
            self._artifacts.model_dir / "config.json",
            self._artifacts.model_dir / "pytorch_model.bin",
        )
        return all(path.is_file() for path in source_files + model_files)


def _force_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _normalize_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized not in {"cpu", "cuda"}:
        raise ModelAdapterError("invalid_device", "device must be cpu or cuda")
    return normalized


def _validate_device(torch_module: object, device: str) -> None:
    if device == "cuda" and not torch_module.cuda.is_available():  # type: ignore[attr-defined]
        raise ModelAdapterError("cuda_unavailable", "CUDA is unavailable in this worker")


def _string_parameter(
    parameters: Mapping[str, object],
    name: str,
    *,
    required: bool,
) -> str | None:
    value = parameters.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value or len(value) > 160:
        raise ModelAdapterError("invalid_parameters", "worker parameters are invalid")
    return value


def _decode_image(image_bytes: bytes, *, max_input_bytes: int, max_pixels: int) -> object:
    if type(image_bytes) is not bytes or not 0 < len(image_bytes) <= max_input_bytes:
        raise ModelAdapterError("invalid_image", "image payload is invalid")
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            source = Image.open(BytesIO(image_bytes))
            source.load()
        if source.width <= 0 or source.height <= 0 or source.width * source.height > max_pixels:
            source.close()
            raise ModelAdapterError("invalid_image", "decoded image dimensions are invalid")
        normalized = ImageOps.exif_transpose(source).convert("RGB")
        source.close()
        return normalized
    except ModelAdapterError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ModelAdapterError("invalid_image", "image could not be decoded") from exc


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _offline_clip_constructor() -> Iterator[None]:
    """Construct the exact CLIP-L/14 shape without resolving a Hub model."""

    import torch.nn as nn
    from transformers import CLIPVisionConfig, CLIPVisionModel

    from models.networks import backbones

    original = backbones.CLIP.__init__

    def initialize(instance: object, path: str) -> None:
        del path
        nn.Module.__init__(instance)
        instance.clip = CLIPVisionModel(  # type: ignore[attr-defined]
            CLIPVisionConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
                image_size=224,
                patch_size=14,
                hidden_act="quick_gelu",
                layer_norm_eps=1e-5,
                attention_dropout=0.0,
            )
        )

    backbones.CLIP.__init__ = initialize
    try:
        yield
    finally:
        backbones.CLIP.__init__ = original


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1_000))
