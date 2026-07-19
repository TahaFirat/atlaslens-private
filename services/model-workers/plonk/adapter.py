from __future__ import annotations

import gc
import importlib.metadata
import math
import os
import threading
import time
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Mapping, MutableMapping

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
            "invalid_sample_count",
            "model_mismatch",
            "unsupported_model",
        }:
            status = 422
        else:
            status = 503
        super().__init__(code, status=status)


@dataclass(frozen=True, slots=True)
class PlonkModelArtifact:
    model_id: str
    revision: str
    model_dir: Path
    conditioning: str


@dataclass(frozen=True, slots=True)
class PlonkArtifacts:
    models: Mapping[str, PlonkModelArtifact]
    source_revision: str
    streetclip_dir: Path
    dinov2_repo_dir: Path
    dinov2_weights_path: Path


def normalize_degree_samples(value: object, *, maximum: int) -> list[list[float]]:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, (list, tuple)):
        raise ModelAdapterError("invalid_model_output", "PLONK samples are not a sequence")
    samples: list[list[float]] = []
    for row in value[:maximum]:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ModelAdapterError("invalid_model_output", "PLONK sample shape is invalid")
        latitude_raw, longitude_raw = row
        if (
            isinstance(latitude_raw, bool)
            or isinstance(longitude_raw, bool)
            or not isinstance(latitude_raw, (int, float))
            or not isinstance(longitude_raw, (int, float))
        ):
            raise ModelAdapterError("invalid_model_output", "PLONK sample is not numeric")
        latitude, longitude = float(latitude_raw), float(longitude_raw)
        if (
            not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
        ):
            raise ModelAdapterError("invalid_model_output", "PLONK sample is outside WGS84")
        samples.append([latitude, longitude])
    if not samples:
        raise ModelAdapterError("invalid_model_output", "PLONK returned no samples")
    return samples


def _is_cuda_oom(exc: BaseException) -> bool:
    return "cuda" in str(exc).casefold() and "out of memory" in str(exc).casefold()


class PlonkAdapter:
    provider = "plonk"

    def __init__(
        self,
        artifacts: PlonkArtifacts,
        *,
        max_input_bytes: int = 20 * 1024 * 1024,
        max_image_pixels: int = 40_000_000,
        num_steps: int = 32,
        guidance_scale: float = 0.0,
        localizability_samples: int = 0,
    ) -> None:
        if max_input_bytes <= 0 or max_image_pixels <= 0:
            raise ValueError("image bounds must be positive")
        if not 1 <= num_steps <= 128:
            raise ValueError("num_steps must be between 1 and 128")
        if not math.isfinite(guidance_scale) or not 0 <= guidance_scale <= 10:
            raise ValueError("guidance_scale must be between 0 and 10")
        if not 0 <= localizability_samples <= 256:
            raise ValueError("localizability_samples must be between 0 and 256")
        self._artifacts = artifacts
        self._max_input_bytes = max_input_bytes
        self._max_image_pixels = max_image_pixels
        self._num_steps = num_steps
        self._guidance_scale = guidance_scale
        self._localizability_samples = localizability_samples
        self._lock = threading.RLock()
        self._pipeline: object | None = None
        self._torch: object | None = None
        self._model: PlonkModelArtifact | None = None
        self._device: str | None = None
        self._load_ms: int | None = None
        self._import_ok = False
        self._load_verified = False
        self._inference_verified: set[str] = set()
        self._last_error: str | None = None

    def health(self) -> dict[str, object]:
        with self._lock:
            import_ok = self._import_ok or self._probe_imports()
            prepared = {
                model_id: self._artifact_available(artifact)
                for model_id, artifact in self._artifacts.models.items()
            }
            return {
                "provider": self.provider,
                "source_revision": self._artifacts.source_revision,
                "package_version": _package_version(),
                "import_ok": import_ok,
                "weights_available": any(prepared.values()),
                "prepared_models": prepared,
                "model_loaded": self._pipeline is not None,
                "load_verified": self._load_verified,
                "loaded_model_id": self._model.model_id if self._model else None,
                "loaded_model_revision": self._model.revision if self._model else None,
                "model_revision": self._model.revision if self._model else "scene-routed",
                "inference_verified_models": sorted(self._inference_verified),
                "real_inference_verified": bool(self._inference_verified),
                "device": self._device,
                "load_ms": self._load_ms,
                "last_error": self._last_error,
            }

    def load(self, parameters: Mapping[str, object]) -> dict[str, object]:
        model_id = _string_parameter(parameters, "model_id", required=True)
        device = _string_parameter(parameters, "device", required=True)
        selected_device = _normalize_device(device)
        artifact = self._artifacts.models.get(model_id)
        if artifact is None:
            raise ModelAdapterError("unsupported_model", "PLONK model id is unsupported")
        with self._lock:
            if (
                self._pipeline is not None
                and self._model is not None
                and self._model.model_id == model_id
                and self._device == selected_device
            ):
                return self.health()
            self.unload({})
            if not self._artifact_available(artifact):
                self._last_error = "pretrained_weights_missing"
                raise ModelAdapterError(
                    "pretrained_weights_missing",
                    "PLONK inference artifacts are incomplete",
                )
            started = time.perf_counter()
            try:
                _force_offline_mode()
                import torch

                _validate_device(torch, selected_device)
                if _package_version() != "0.4":
                    raise ModelAdapterError(
                        "unsupported_package_version",
                        "PLONK package version is unsupported",
                    )
                pipeline = self._build_pipeline(artifact, selected_device, torch)
                self._pipeline = pipeline
                self._torch = torch
                self._model = artifact
                self._device = selected_device
                self._load_ms = _elapsed_ms(started)
                self._import_ok = True
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
                raise ModelAdapterError(code, "PLONK model load failed") from exc

    def infer(
        self,
        image_bytes: bytes,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        sample_count = _int_parameter(
            parameters,
            "sample_count",
            minimum=1,
            maximum=256,
        )
        if not 1 <= sample_count <= 256:
            raise ModelAdapterError("invalid_sample_count", "sample_count must be between 1 and 256")
        with self._lock:
            if (
                self._pipeline is None
                or self._torch is None
                or self._model is None
                or self._device is None
            ):
                raise ModelAdapterError("model_not_loaded", "PLONK model is not loaded")
            requested_device = _string_parameter(parameters, "device", required=False)
            if requested_device is not None and requested_device != self._device:
                raise ModelAdapterError("device_mismatch", "PLONK loaded device does not match")
            requested_model = _string_parameter(parameters, "model_id", required=False)
            if requested_model is not None and requested_model != self._model.model_id:
                raise ModelAdapterError("model_mismatch", "PLONK loaded model does not match")
            image = _decode_image(
                image_bytes,
                max_input_bytes=self._max_input_bytes,
                max_pixels=self._max_image_pixels,
            )
            started = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"`torch\.cuda\.amp\.autocast.*",
                        category=FutureWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r"CUDA is not available.*",
                        category=UserWarning,
                    )
                    with self._torch.inference_mode():  # type: ignore[attr-defined]
                        raw_samples = self._pipeline(  # type: ignore[operator]
                            image,
                            batch_size=sample_count,
                            cfg=self._guidance_scale,
                            num_steps=self._num_steps,
                        )
                samples = normalize_degree_samples(raw_samples, maximum=sample_count)
                localizability = self._compute_localizability(image)
            except ModelAdapterError:
                self._last_error = "invalid_model_output"
                raise
            except Exception as exc:
                code = "cuda_out_of_memory" if _is_cuda_oom(exc) else "inference_failed"
                self._last_error = code
                raise ModelAdapterError(code, "PLONK inference failed") from exc
            finally:
                image.close()
            inference_ms = _elapsed_ms(started)
            self._inference_verified.add(self._model.model_id)
            self._last_error = None
            return {
                "provider": self.provider,
                "model_id": self._model.model_id,
                "model_revision": self._model.revision,
                "source_revision": self._artifacts.source_revision,
                "device": self._device,
                "samples_degrees": samples,
                "coordinate_order": "latitude_longitude",
                "units": "degrees",
                "sample_count": len(samples),
                "localizability": localizability,
                "localizability_semantics": "raw_model_diagnostic_not_confidence",
                "load_ms": self._load_ms,
                "inference_ms": inference_ms,
            }

    def unload(self, parameters: Mapping[str, object]) -> dict[str, object]:
        del parameters
        with self._lock:
            pipeline, torch_module = self._pipeline, self._torch
            self._pipeline = None
            self._torch = None
            self._model = None
            self._device = None
            self._load_ms = None
            if pipeline is not None:
                del pipeline
            gc.collect()
            if torch_module is not None:
                try:
                    if torch_module.cuda.is_available():  # type: ignore[attr-defined]
                        torch_module.cuda.empty_cache()  # type: ignore[attr-defined]
                except RuntimeError:
                    pass
            return self.health()

    def _build_pipeline(self, artifact: PlonkModelArtifact, device: str, torch: object) -> object:
        from plonk import PlonkPipeline
        from plonk.models.postprocessing import CartesiantoGPS
        from plonk.models.preconditioning import DDPMPrecond
        from plonk.models.pretrained_models import Plonk
        from plonk.pipe import MODELS, scheduler_fn
        from plonk.utils.manifolds import Sphere

        if artifact.model_id not in MODELS:
            raise ModelAdapterError("unsupported_model", "PLONK model id is unsupported")
        conditioning = self._conditioning(artifact, device, torch)
        # Build the pinned diff-plonk 0.4 pipeline explicitly. Its public
        # constructor cannot consume local_dir snapshots because it indexes
        # MODELS with the path and would otherwise start auxiliary downloads.
        pipeline = PlonkPipeline.__new__(PlonkPipeline)
        pipeline.network = Plonk.from_pretrained(
            str(artifact.model_dir.resolve()),
            local_files_only=True,
            map_location="cpu",
        ).to(device)
        pipeline.network.requires_grad_(False).eval()
        pipeline.scheduler = scheduler_fn("sigmoid", -7, 3, 1.0)
        pipeline.cond_preprocessing = conditioning
        pipeline.postprocessing = CartesiantoGPS().to(device)
        pipeline.sampler = MODELS[artifact.model_id]["sampler"]
        pipeline.model_path = artifact.model_id
        pipeline.preconditioning = DDPMPrecond()
        pipeline.device = torch.device(device)  # type: ignore[attr-defined]
        pipeline.manifold = Sphere()
        pipeline.input_dim = 3
        return pipeline

    def _conditioning(self, artifact: PlonkModelArtifact, device: str, torch: object) -> object:
        if artifact.conditioning == "streetclip":
            return _LocalStreetClip(self._artifacts.streetclip_dir, device)
        if artifact.conditioning == "dinov2":
            return _LocalDinoV2(
                self._artifacts.dinov2_repo_dir,
                self._artifacts.dinov2_weights_path,
                device,
                torch,
            )
        raise ModelAdapterError("unsupported_conditioning", "PLONK conditioning is unsupported")

    def _compute_localizability(self, image: object) -> float | None:
        if self._localizability_samples == 0 or self._pipeline is None:
            return None
        value = self._pipeline.compute_localizability(  # type: ignore[attr-defined]
            image,
            number_monte_carlo_samples=self._localizability_samples,
        )
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelAdapterError("invalid_model_output", "PLONK localizability is invalid")
        result = float(value)
        if not math.isfinite(result):
            raise ModelAdapterError("invalid_model_output", "PLONK localizability is not finite")
        return result

    def _probe_imports(self) -> bool:
        try:
            _force_offline_mode()
            import plonk  # noqa: F401
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        self._import_ok = _package_version() == "0.4"
        return self._import_ok

    def _artifact_available(self, artifact: PlonkModelArtifact) -> bool:
        model_ready = all(
            (artifact.model_dir / name).is_file()
            for name in ("config.json", "model.safetensors")
        )
        if artifact.conditioning == "streetclip":
            conditioning_ready = (
                (self._artifacts.streetclip_dir / "config.json").is_file()
                and any(
                    (self._artifacts.streetclip_dir / name).is_file()
                    for name in ("model.safetensors", "pytorch_model.bin")
                )
                and any(
                    (self._artifacts.streetclip_dir / name).is_file()
                    for name in ("preprocessor_config.json", "processor_config.json")
                )
            )
        elif artifact.conditioning == "dinov2":
            conditioning_ready = (
                (self._artifacts.dinov2_repo_dir / "hubconf.py").is_file()
                and self._artifacts.dinov2_weights_path.is_file()
            )
        else:
            conditioning_ready = False
        return model_ready and conditioning_ready


class _LocalStreetClip:
    def __init__(self, model_dir: Path, device: str) -> None:
        try:
            from transformers import CLIPProcessor, CLIPVisionModel

            local = str(model_dir.resolve())
            self._model = CLIPVisionModel.from_pretrained(
                local,
                local_files_only=True,
            ).to(device)
            self._model.requires_grad_(False).eval()
            self._processor = CLIPProcessor.from_pretrained(local, local_files_only=True)
            self._device = device
        except Exception as exc:
            raise ModelAdapterError(
                "conditioning_load_failed",
                "StreetCLIP conditioning load failed",
            ) from exc

    def __call__(self, batch: MutableMapping[str, object]) -> MutableMapping[str, object]:
        inputs = self._processor(images=batch["img"], return_tensors="pt").to(self._device)
        outputs = self._model(**inputs)
        batch["emb"] = outputs.last_hidden_state[:, 0]
        return batch


class _LocalDinoV2:
    def __init__(
        self,
        repo_dir: Path,
        weights_path: Path,
        device: str,
        torch: object,
    ) -> None:
        try:
            from torchvision import transforms

            from plonk.utils.image_processing import CenterCrop

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"xFormers is not available.*",
                    category=UserWarning,
                )
                model = torch.hub.load(  # type: ignore[attr-defined]
                    str(repo_dir.resolve()),
                    "dinov2_vitl14_reg",
                    source="local",
                    pretrained=False,
                    verbose=False,
                )
            state = torch.load(  # type: ignore[attr-defined]
                str(weights_path.resolve()),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            state_dict = _unwrap_state_dict(state)
            model.load_state_dict(state_dict, strict=True)
            self._model = model.to(device).requires_grad_(False).eval()
            self._device = device
            self._transform = transforms.Compose(
                [
                    CenterCrop(ratio="1:1"),
                    transforms.Resize(
                        336,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise ModelAdapterError(
                "conditioning_load_failed",
                "DINOv2 conditioning load failed",
            ) from exc

    def __call__(self, batch: MutableMapping[str, object]) -> MutableMapping[str, object]:
        embeddings = []
        for image in batch["img"]:  # type: ignore[union-attr]
            if image.mode != "RGB":
                image = image.convert("RGB")
            tensor = self._transform(image).unsqueeze(0).to(self._device)
            embeddings.append(self._model(tensor).squeeze(0))
        import torch

        batch["emb"] = torch.stack(embeddings)
        return batch


def _unwrap_state_dict(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelAdapterError("invalid_conditioning_weights", "DINOv2 weights are invalid")
    candidate: object = value
    for key in ("state_dict", "model"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            candidate = nested
            break
    if not isinstance(candidate, Mapping) or not candidate:
        raise ModelAdapterError("invalid_conditioning_weights", "DINOv2 weights are invalid")
    if not all(isinstance(key, str) for key in candidate):
        raise ModelAdapterError("invalid_conditioning_weights", "DINOv2 weights are invalid")
    return candidate


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("diff-plonk")
    except importlib.metadata.PackageNotFoundError:
        return None


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


def _int_parameter(
    parameters: Mapping[str, object],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
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


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1_000))
