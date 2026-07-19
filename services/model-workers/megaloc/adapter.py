from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from common.protocol import WorkerAdapterError


class MegaLocAdapterError(WorkerAdapterError):
    def __init__(self, code: str) -> None:
        status = 409 if code == "model_not_loaded" else 422 if code.startswith("invalid_") else 503
        super().__init__(code, status=status)


@dataclass(frozen=True, slots=True)
class MegaLocArtifacts:
    source_dir: Path
    model_dir: Path
    source_revision: str
    model_revision: str
    weight_sha256: str
    model_id: str = "gberton/MegaLoc"

    @property
    def weights_path(self) -> Path:
        return self.model_dir / "model.safetensors"

    @property
    def receipt_path(self) -> Path:
        return self.model_dir / "receipt.json"


class MegaLocAdapter:
    provider = "megaloc"

    def __init__(
        self,
        artifacts: MegaLocArtifacts,
        *,
        max_input_bytes: int = 20 * 1024 * 1024,
        max_image_pixels: int = 40_000_000,
        maximum_edge: int = 560,
    ) -> None:
        if max_input_bytes <= 0 or max_image_pixels <= 0 or not 224 <= maximum_edge <= 1_120:
            raise ValueError("MegaLoc input bounds are invalid")
        self._artifacts = artifacts
        self._max_input_bytes = max_input_bytes
        self._max_image_pixels = max_image_pixels
        self._maximum_edge = maximum_edge
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._load_ms: int | None = None
        self._last_inference_ms: int | None = None
        self._import_ok = False
        self._load_verified = False
        self._inference_verified = False
        self._last_error: str | None = None

    def health(self) -> dict[str, object]:
        with self._lock:
            return {
                "provider": self.provider,
                "model_id": self._artifacts.model_id,
                "model_revision": self._artifacts.model_revision,
                "source_revision": self._artifacts.source_revision,
                "import_ok": self._import_ok or self._probe_imports(),
                "weights_available": self._weights_available(),
                "model_loaded": self._model is not None,
                "load_verified": self._load_verified,
                "real_inference_verified": self._inference_verified,
                "device": self._device,
                "load_ms": self._load_ms,
                "last_inference_ms": self._last_inference_ms,
                "descriptor_dimension": 8448,
                "last_error": self._last_error,
            }

    def load(self, parameters: Mapping[str, object]) -> dict[str, object]:
        selected_device = _normalize_device(_string_parameter(parameters, "device", required=True))
        with self._lock:
            if self._model is not None and self._device == selected_device:
                return self.health()
            self.unload({})
            if not self._weights_available():
                self._last_error = "pretrained_weights_missing"
                raise MegaLocAdapterError("pretrained_weights_missing")
            started = time.perf_counter()
            try:
                _force_offline_mode()
                self._prepare_import_path()
                import torch
                from megaloc_model import MegaLoc
                from safetensors.torch import load_file

                self._import_ok = True
                _validate_device(torch, selected_device)
                model = MegaLoc()
                state = load_file(str(self._artifacts.weights_path), device="cpu")
                model.load_state_dict(state, strict=True)
                del state
                model.requires_grad_(False).eval().to(selected_device)
                self._model = model
                self._torch = torch
                self._device = selected_device
                self._load_ms = _elapsed_ms(started)
                self._load_verified = True
                self._last_error = None
                return self.health()
            except MegaLocAdapterError:
                self.unload({})
                raise
            except Exception as exc:
                code = "cuda_out_of_memory" if _is_cuda_oom(exc) else "model_load_failed"
                self._last_error = code
                self.unload({})
                raise MegaLocAdapterError(code) from exc

    def infer(
        self,
        image_bytes: bytes,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        with self._lock:
            if self._model is None or self._torch is None or self._device is None:
                raise MegaLocAdapterError("model_not_loaded")
            requested_device = _string_parameter(parameters, "device", required=False)
            if requested_device is not None and requested_device != self._device:
                raise MegaLocAdapterError("invalid_device_mismatch")
            image = _decode_image(
                image_bytes,
                max_input_bytes=self._max_input_bytes,
                max_pixels=self._max_image_pixels,
            )
            started = time.perf_counter()
            try:
                tensor = _preprocess(image, maximum_edge=self._maximum_edge).unsqueeze(0)
                with self._torch.inference_mode():
                    descriptor = self._model(tensor.to(self._device))[0]
                descriptor = descriptor.float().cpu()
                norm = float(descriptor.norm(p=2).item())
                if descriptor.ndim != 1 or descriptor.numel() != 8448:
                    raise MegaLocAdapterError("invalid_model_output")
                if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
                    raise MegaLocAdapterError("invalid_model_output")
                values = descriptor.tolist()
                if any(not math.isfinite(float(item)) for item in values):
                    raise MegaLocAdapterError("invalid_model_output")
            except MegaLocAdapterError:
                self._last_error = "invalid_model_output"
                raise
            except Exception as exc:
                code = "cuda_out_of_memory" if _is_cuda_oom(exc) else "inference_failed"
                self._last_error = code
                raise MegaLocAdapterError(code) from exc
            finally:
                image.close()
            self._last_inference_ms = _elapsed_ms(started)
            self._inference_verified = True
            self._last_error = None
            return {
                "provider": self.provider,
                "model_id": self._artifacts.model_id,
                "model_revision": self._artifacts.model_revision,
                "source_revision": self._artifacts.source_revision,
                "device": self._device,
                "descriptor": values,
                "descriptor_dimension": 8448,
                "descriptor_normalization": "l2",
                "descriptor_semantics": "visual_place_descriptor_not_confidence",
                "inference_ms": self._last_inference_ms,
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
                    if torch_module.cuda.is_available():
                        torch_module.cuda.empty_cache()
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
            from megaloc_model import MegaLoc  # noqa: F401
            from safetensors.torch import load_file  # noqa: F401
        except Exception:
            return False
        self._import_ok = True
        return True

    def _weights_available(self) -> bool:
        required = (
            self._artifacts.source_dir / "megaloc_model.py",
            self._artifacts.source_dir / "LICENSE",
            self._artifacts.weights_path,
            self._artifacts.receipt_path,
        )
        if not all(path.is_file() and not path.is_symlink() for path in required):
            return False
        try:
            receipt = json.loads(self._artifacts.receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(
            receipt.get("source_revision") == self._artifacts.source_revision
            and receipt.get("model_revision") == self._artifacts.model_revision
            and receipt.get("weight_sha256") == self._artifacts.weight_sha256
            and receipt.get("weight_size") == self._artifacts.weights_path.stat().st_size
        )


def verify_weight(path: Path, expected_sha256: str) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise MegaLocAdapterError("checksum_mismatch")
    return actual


def _force_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _normalize_device(value: str | None) -> str:
    normalized = (value or "").strip().casefold()
    if normalized not in {"cpu", "cuda"}:
        raise MegaLocAdapterError("invalid_device")
    return normalized


def _validate_device(torch_module: Any, device: str) -> None:
    if device == "cuda" and not bool(torch_module.cuda.is_available()):
        raise MegaLocAdapterError("invalid_device")


def _string_parameter(
    parameters: Mapping[str, object], name: str, *, required: bool
) -> str | None:
    value = parameters.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value or len(value) > 160:
        raise MegaLocAdapterError("invalid_parameters")
    return value


def _decode_image(image_bytes: bytes, *, max_input_bytes: int, max_pixels: int) -> Any:
    if type(image_bytes) is not bytes or not 0 < len(image_bytes) <= max_input_bytes:
        raise MegaLocAdapterError("invalid_image")
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            source = Image.open(BytesIO(image_bytes))
            source.load()
        if source.width <= 0 or source.height <= 0 or source.width * source.height > max_pixels:
            source.close()
            raise MegaLocAdapterError("invalid_image")
        normalized = ImageOps.exif_transpose(source).convert("RGB")
        source.close()
        return normalized
    except MegaLocAdapterError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise MegaLocAdapterError("invalid_image") from exc


def _preprocess(image: Any, *, maximum_edge: int) -> Any:
    from torchvision.transforms import functional as transform

    width, height = image.size
    scale = min(1.0, maximum_edge / max(width, height))
    target_width = max(14, round(width * scale / 14) * 14)
    target_height = max(14, round(height * scale / 14) * 14)
    if (target_width, target_height) != (width, height):
        image = transform.resize(image, [target_height, target_width], antialias=True)
    tensor = transform.to_tensor(image)
    return transform.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".casefold()
    return "outofmemory" in text or "out of memory" in text


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1_000))
