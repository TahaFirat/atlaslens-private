from __future__ import annotations

import importlib
import io
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from atlaslens_api.model_management.errors import ModelManagementError
from atlaslens_api.model_management.service import EmbeddingModelManagementService
from atlaslens_api.retrieval.errors import ProviderUnavailableError
from atlaslens_api.retrieval.models import Embedding, EmbeddingSpec

SIGLIP2_PROVIDER_ID = "siglip2-b16-384"
_MAX_IMAGE_BYTES = 50 * 1024 * 1024
_MAX_IMAGE_PIXELS = 50_000_000

RuntimeLoader = Callable[[Path, str], tuple[Any, Any, Any]]


def select_embedding_device(requested: str = "auto") -> str:
    torch = importlib.import_module("torch")
    cuda = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda:
        raise ProviderUnavailableError("cuda_unavailable")
    if requested not in {"auto", "cpu", "cuda"}:
        raise ProviderUnavailableError("invalid_device")
    return "cuda" if requested == "cuda" or (requested == "auto" and cuda) else "cpu"


def _load_runtime(snapshot: Path, device: str) -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            snapshot, local_files_only=True, use_fast=False
        )
        model = transformers.AutoModel.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch.float32
        )
        model.eval()
        model.to(device)
    except Exception as exc:
        raise ProviderUnavailableError("model_load_failed") from exc
    return model, processor, torch


class Siglip2EmbeddingProvider:
    """Lazy offline image encoder backed by the verified pinned SigLIP2 snapshot."""

    def __init__(
        self,
        management: EmbeddingModelManagementService,
        *,
        requested_device: str = "auto",
        max_concurrency: int = 1,
        max_batch_size: int = 16,
        runtime_loader: RuntimeLoader = _load_runtime,
        device_selector: Callable[[str], str] = select_embedding_device,
    ) -> None:
        if max_concurrency < 1 or max_batch_size < 1:
            raise ValueError("embedding concurrency and batch size must be positive")
        self._management = management
        self._requested_device = requested_device
        self._max_batch_size = max_batch_size
        self._runtime_loader = runtime_loader
        self._device_selector = device_selector
        self._load_lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._runtime: tuple[Any, Any, Any] | None = None
        self._device: str | None = None
        manifest = management.manifest
        self._spec = EmbeddingSpec(
            provider=SIGLIP2_PROVIDER_ID,
            version=f"{manifest.revision}+{manifest.preprocessing_version}",
            dimension=manifest.embedding_dimension,
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    @property
    def available(self) -> bool:
        return self._management.is_installed()

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.available else "model_not_installed"

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    @property
    def device(self) -> str | None:
        return self._device

    def _ensure_runtime(self) -> tuple[Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        with self._load_lock:
            if self._runtime is not None:
                return self._runtime
            try:
                self._management.verify()
                snapshot = self._management.snapshot_directory()
            except ModelManagementError as exc:
                raise ProviderUnavailableError(exc.code) from exc
            device = self._device_selector(self._requested_device)
            self._runtime = self._runtime_loader(snapshot, device)
            self._device = device
            return self._runtime

    @staticmethod
    def _read_image(image_path: Path) -> Image.Image:
        try:
            resolved = image_path.expanduser().resolve(strict=True)
            if not resolved.is_file() or resolved.stat().st_size > _MAX_IMAGE_BYTES:
                raise ProviderUnavailableError("invalid_image")
            with Image.open(resolved) as source:
                width, height = source.size
                if width < 1 or height < 1 or width * height > _MAX_IMAGE_PIXELS:
                    raise ProviderUnavailableError("invalid_image")
                return ImageOps.exif_transpose(source).convert("RGB")
        except ProviderUnavailableError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ProviderUnavailableError("invalid_image") from exc

    @staticmethod
    def _read_image_bytes(payload: bytes) -> Image.Image:
        if not payload or len(payload) > _MAX_IMAGE_BYTES:
            raise ProviderUnavailableError("invalid_image")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                width, height = source.size
                if width < 1 or height < 1 or width * height > _MAX_IMAGE_PIXELS:
                    raise ProviderUnavailableError("invalid_image")
                return ImageOps.exif_transpose(source).convert("RGB")
        except ProviderUnavailableError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ProviderUnavailableError("invalid_image") from exc

    def embed(self, image_path: Path) -> Embedding:
        return self.embed_many([image_path])[0]

    def embed_bytes(self, payload: bytes) -> Embedding:
        """Encode an immutable payload so timed-out callers never retain an upload path."""

        with self._semaphore:
            image = self._read_image_bytes(payload)
            return self._encode_images([image])[0]

    def embed_many(self, image_paths: Sequence[Path]) -> list[Embedding]:
        if not image_paths:
            return []
        embeddings: list[Embedding] = []
        with self._semaphore:
            for start in range(0, len(image_paths), self._max_batch_size):
                paths = image_paths[start : start + self._max_batch_size]
                images = [self._read_image(path) for path in paths]
                embeddings.extend(self._encode_images(images))
        return embeddings

    def _encode_images(self, images: Sequence[Image.Image]) -> list[Embedding]:
        model, processor, torch = self._ensure_runtime()
        try:
            inputs = processor(images=images, return_tensors="pt")
            device = self._device or "cpu"
            moved = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = model.get_image_features(**moved)
            values = features.detach().float().cpu().numpy()
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError("embedding_inference_failed") from exc
        finally:
            for image in images:
                image.close()
        if values.ndim != 2 or values.shape != (len(images), self.spec.dimension):
            raise ProviderUnavailableError("invalid_embedding_output")
        return [Embedding(self.spec, np.asarray(row, dtype=np.float32)) for row in values]

    def diagnostics(self) -> dict[str, object]:
        return {
            "status": "ready" if self.available else "unavailable",
            "provider": self.spec.provider,
            "version": self.spec.version,
            "dimension": self.spec.dimension,
            "device": self.device,
            "loaded": self.loaded,
            "offline": True,
        }
