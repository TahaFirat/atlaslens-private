from __future__ import annotations

import gc
import importlib
import importlib.metadata
import io
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.protocol import JSONValue, WorkerAdapterError


PROVIDER = "paddleocr"
PROVIDER_REVISION = "3.7.0"
PADDLE_REVISION = "3.3.1"
DETECTION_MODEL_ID = "PP-OCRv5_server_det"
RECOGNITION_MODEL_ID = "latin_PP-OCRv5_mobile_rec"
MODEL_REVISION = f"{DETECTION_MODEL_ID}+{RECOGNITION_MODEL_ID}"

_MODEL_FILES = (
    ("inference.pdiparams",),
    ("inference.json", "inference.pdmodel", "model.pdmodel"),
    ("inference.yml", "infer_cfg.yml", "config.json"),
)
_MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_LINES = 128


class PaddleOCRAdapterError(WorkerAdapterError):
    """An adapter failure whose message is safe to cross the worker boundary."""

    def __init__(self, code: str) -> None:
        status = 422 if code in {
            "invalid_confidence",
            "invalid_image",
            "invalid_scales",
            "unsupported_device",
        } else 503
        super().__init__(code, status=status)


@dataclass(frozen=True, slots=True)
class PaddleOCRConfig:
    model_root: Path
    detection_model_dir: Path
    recognition_model_dir: Path
    private_home: Path
    device: str = "cpu"
    max_input_bytes: int = 20 * 1024 * 1024
    max_decoded_pixels: int = 16_000_000
    max_side: int = 4096

    @classmethod
    def from_environment(cls) -> PaddleOCRConfig:
        project_root = Path(__file__).resolve().parents[3]
        model_root = Path(
            os.environ.get(
                "ATLASLENS_PADDLEOCR_MODEL_ROOT",
                project_root / ".local" / "models" / "phase6b" / "paddleocr",
            )
        )
        detection_dir = Path(
            os.environ.get(
                "ATLASLENS_PADDLEOCR_DET_MODEL_DIR", model_root / "det"
            )
        )
        recognition_dir = Path(
            os.environ.get(
                "ATLASLENS_PADDLEOCR_REC_MODEL_DIR",
                model_root / "rec" / RECOGNITION_MODEL_ID,
            )
        )
        private_home = Path(
            os.environ.get(
                "ATLASLENS_PADDLEOCR_PRIVATE_HOME",
                project_root / ".local" / "workers" / "paddleocr" / "home",
            )
        )
        return cls(
            model_root=model_root,
            detection_model_dir=detection_dir,
            recognition_model_dir=recognition_dir,
            private_home=private_home,
        )


class PaddleOCRAdapter:
    """PaddleOCR 3.7 adapter that never resolves or downloads model names."""

    def __init__(
        self,
        config: PaddleOCRConfig | None = None,
        *,
        ocr_factory: Callable[..., object] | None = None,
        enforce_versions: bool = True,
    ) -> None:
        self._config = config or PaddleOCRConfig.from_environment()
        if self._config.device != "cpu":
            raise PaddleOCRAdapterError("unsupported_device")
        self._factory = ocr_factory
        self._enforce_versions = enforce_versions
        self._engine: object | None = None
        self._lock = threading.RLock()
        self._import_ok = False
        self._weights_available = self._model_artifacts_valid()
        self._model_loaded = False
        self._load_verified = False
        self._real_inference_verified = False
        self._last_error: str | None = None

    def health(self) -> Mapping[str, JSONValue]:
        with self._lock:
            self._weights_available = self._model_artifacts_valid()
            return {
                "process_running": True,
                "import_ok": self._import_ok,
                "weights_available": self._weights_available,
                "model_loaded": self._model_loaded,
                "load_verified": self._load_verified,
                "real_inference_verified": self._real_inference_verified,
                "device": self._config.device,
                "provider_revision": PROVIDER_REVISION,
                "model_revision": MODEL_REVISION,
                "last_error": self._last_error,
            }

    def load(self, parameters: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        with self._lock:
            self._validate_device(parameters)
            if self._model_loaded and self._engine is not None:
                return self._load_result(load_ms=0)
            self._last_error = None
            self._weights_available = self._model_artifacts_valid()
            if not self._weights_available:
                self._fail("pretrained_weights_missing")
            self._configure_private_environment()
            started = time.monotonic()
            try:
                if self._factory is None:
                    self._verify_versions()
                    module = importlib.import_module("paddleocr")
                    factory = getattr(module, "PaddleOCR")
                else:
                    factory = self._factory
                self._import_ok = True
                logging.getLogger("paddlex").setLevel(logging.ERROR)
            except PaddleOCRAdapterError:
                raise
            except Exception as exc:
                self._fail("provider_import_failed", cause=exc)
            try:
                self._engine = factory(
                    text_detection_model_name=DETECTION_MODEL_ID,
                    text_detection_model_dir=str(
                        self._config.detection_model_dir.resolve(strict=True)
                    ),
                    text_recognition_model_name=RECOGNITION_MODEL_ID,
                    text_recognition_model_dir=str(
                        self._config.recognition_model_dir.resolve(strict=True)
                    ),
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu",
                    enable_mkldnn=False,
                    cpu_threads=4,
                )
            except Exception as exc:
                self._engine = None
                self._fail("model_load_failed", cause=exc)
            self._model_loaded = True
            self._load_verified = True
            return self._load_result(
                load_ms=max(0, int((time.monotonic() - started) * 1000))
            )

    def infer(
        self, image_bytes: bytes, parameters: dict[str, JSONValue]
    ) -> Mapping[str, JSONValue]:
        with self._lock:
            self._validate_device(parameters)
            if not self._model_loaded or self._engine is None:
                self.load(parameters)
            engine = self._engine
            if engine is None:
                self._fail("model_not_loaded")
            try:
                scales = self._parse_scales(parameters.get("scales"))
                minimum_confidence = self._parse_confidence(
                    parameters.get("minimum_confidence", 0.0)
                )
            except PaddleOCRAdapterError as exc:
                self._last_error = exc.code
                raise
            image, width, height = self._decode_oriented_image(image_bytes)
            started = time.monotonic()
            lines: list[dict[str, JSONValue]] = []
            try:
                for scale in scales:
                    scaled = self._scaled_bgr_image(image, scale)
                    results = engine.predict(
                        scaled,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        text_rec_score_thresh=minimum_confidence,
                        return_word_box=False,
                    )
                    lines.extend(
                        self._normalize_results(
                            results,
                            scale=scale,
                            width=width,
                            height=height,
                            minimum_confidence=minimum_confidence,
                            remaining=_MAX_LINES - len(lines),
                        )
                    )
                    if len(lines) >= _MAX_LINES:
                        break
            except PaddleOCRAdapterError as exc:
                self._last_error = exc.code
                raise
            except Exception as exc:
                self._fail("inference_failed", cause=exc)
            finally:
                image.close()
            inference_ms = max(0, int((time.monotonic() - started) * 1000))
            self._real_inference_verified = True
            self._last_error = None
            return {
                "lines": lines,
                "device": self._config.device,
                "model_ids": [DETECTION_MODEL_ID, RECOGNITION_MODEL_ID],
                "model_revision": MODEL_REVISION,
                "inference_ms": inference_ms,
            }

    def unload(self, parameters: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        del parameters
        with self._lock:
            engine, self._engine = self._engine, None
            self._model_loaded = False
            if engine is not None:
                close = getattr(engine, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        self._last_error = "model_unload_failed"
                del engine
            gc.collect()
            return {
                "unloaded": True,
                "device": self._config.device,
                "model_revision": MODEL_REVISION,
            }

    def _configure_private_environment(self) -> None:
        home = self._config.private_home
        if home.is_symlink():
            self._fail("unsafe_private_home")
        try:
            home.mkdir(parents=True, exist_ok=True)
            directories = {
                "USERPROFILE": home,
                "HOME": home,
                "XDG_CACHE_HOME": home / ".cache",
                "PADDLE_PDX_CACHE_HOME": home / ".paddlex",
                "HF_HOME": home / ".cache" / "huggingface",
                "MODELSCOPE_CACHE": home / ".cache" / "modelscope",
                "TEMP": home / "tmp",
                "TMP": home / "tmp",
            }
            for path in directories.values():
                path.mkdir(parents=True, exist_ok=True)
            (home / ".cache" / "paddle" / "dataset").mkdir(
                parents=True, exist_ok=True
            )
        except OSError as exc:
            self._fail("private_cache_unavailable", cause=exc)
        for name, path in directories.items():
            os.environ[name] = str(path)
        os.environ.update(
            {
                "PADDLEOCR_DISABLE_AUTO_LOGGING_CONFIG": "1",
                "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "FLAGS_use_mkldnn": "0",
                "FLAGS_enable_pir_api": "0",
                "FLAGS_minloglevel": "2",
                "GLOG_minloglevel": "2",
            }
        )

    def _model_artifacts_valid(self) -> bool:
        try:
            root = self._config.model_root.resolve(strict=True)
            if not root.is_dir() or self._config.model_root.is_symlink():
                return False
            for requested in (
                self._config.detection_model_dir,
                self._config.recognition_model_dir,
            ):
                if requested.is_symlink():
                    return False
                model_dir = requested.resolve(strict=True)
                if not model_dir.is_dir() or root not in model_dir.parents:
                    return False
                if not self._model_directory_valid(model_dir):
                    return False
            return True
        except OSError:
            return False

    @staticmethod
    def _model_directory_valid(model_dir: Path) -> bool:
        total = 0
        for candidates in _MODEL_FILES:
            matches = [model_dir / name for name in candidates if (model_dir / name).is_file()]
            if not matches or any(match.is_symlink() for match in matches):
                return False
        for artifact in model_dir.iterdir():
            if artifact.is_symlink():
                return False
            if artifact.is_file():
                size = artifact.stat().st_size
                if size <= 0:
                    return False
                total += size
        return 0 < total <= _MAX_MODEL_BYTES

    def _verify_versions(self) -> None:
        if not self._enforce_versions:
            return
        try:
            paddleocr_version = importlib.metadata.version("paddleocr")
            paddle_version = importlib.metadata.version("paddlepaddle")
        except importlib.metadata.PackageNotFoundError as exc:
            self._fail("provider_import_failed", cause=exc)
        if paddleocr_version != PROVIDER_REVISION or paddle_version != PADDLE_REVISION:
            self._fail("provider_version_mismatch")

    def _decode_oriented_image(self, image_bytes: bytes) -> tuple[Any, int, int]:
        if not isinstance(image_bytes, bytes) or not (
            0 < len(image_bytes) <= self._config.max_input_bytes
        ):
            self._fail("invalid_image")
        try:
            from PIL import Image, ImageOps

            source = Image.open(io.BytesIO(image_bytes))
            source.load()
            width, height = source.size
            if (
                width <= 0
                or height <= 0
                or max(width, height) > self._config.max_side
                or width * height > self._config.max_decoded_pixels
            ):
                source.close()
                self._fail("invalid_image")
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            source.close()
            width, height = oriented.size
            if width * height > self._config.max_decoded_pixels:
                oriented.close()
                self._fail("invalid_image")
            return oriented, width, height
        except PaddleOCRAdapterError:
            raise
        except Exception as exc:
            self._fail("invalid_image", cause=exc)

    def _scaled_bgr_image(self, image: Any, scale: float) -> Any:
        try:
            import numpy as np
            from PIL import Image

            width, height = image.size
            scaled_width = max(1, round(width * scale))
            scaled_height = max(1, round(height * scale))
            if (
                max(scaled_width, scaled_height) > self._config.max_side
                or scaled_width * scaled_height > self._config.max_decoded_pixels
            ):
                self._fail("invalid_scales")
            scaled = (
                image
                if scale == 1.0
                else image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            )
            return np.asarray(scaled, dtype=np.uint8)[:, :, ::-1].copy()
        except PaddleOCRAdapterError:
            raise
        except Exception as exc:
            self._fail("invalid_image", cause=exc)

    def _normalize_results(
        self,
        results: object,
        *,
        scale: float,
        width: int,
        height: int,
        minimum_confidence: float,
        remaining: int,
    ) -> list[dict[str, JSONValue]]:
        if remaining <= 0 or isinstance(results, (str, bytes, Mapping)):
            self._fail("invalid_model_output")
        try:
            pages = list(results)  # type: ignore[arg-type]
        except TypeError as exc:
            self._fail("invalid_model_output", cause=exc)
        if len(pages) != 1:
            self._fail("invalid_model_output")
        page = pages[0]
        if not isinstance(page, Mapping):
            json_value = getattr(page, "json", None)
            if isinstance(json_value, Mapping):
                page = json_value.get("res", json_value)
        if not isinstance(page, Mapping):
            self._fail("invalid_model_output")
        texts = self._as_sequence(page.get("rec_texts"))
        scores = self._as_sequence(page.get("rec_scores"))
        polygons = self._as_sequence(page.get("rec_polys"))
        if not (len(texts) == len(scores) == len(polygons)):
            self._fail("invalid_model_output")
        lines: list[dict[str, JSONValue]] = []
        for text, score_value, polygon_value in zip(texts, scores, polygons, strict=True):
            if len(lines) >= remaining:
                break
            try:
                score = float(score_value)
            except (TypeError, ValueError):
                continue
            if (
                not isinstance(text, str)
                or not text.strip()
                or len(text) > 512
                or not math.isfinite(score)
                or not minimum_confidence <= score <= 1
            ):
                continue
            polygon = self._normalize_polygon(
                polygon_value, scale=scale, width=width, height=height
            )
            if polygon is None:
                continue
            lines.append(
                {
                    "text": text,
                    "confidence": score,
                    "polygon": polygon,
                    "crop_id": "standard" if scale == 1.0 else f"scale-{scale:g}",
                }
            )
        return lines

    @staticmethod
    def _as_sequence(value: object) -> list[object]:
        if value is None or isinstance(value, (str, bytes, Mapping)):
            raise PaddleOCRAdapterError("invalid_model_output")
        try:
            return list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise PaddleOCRAdapterError("invalid_model_output") from exc

    @staticmethod
    def _normalize_polygon(
        value: object, *, scale: float, width: int, height: int
    ) -> list[list[float]] | None:
        if isinstance(value, (str, bytes, Mapping)):
            return None
        try:
            points = list(value)  # type: ignore[arg-type]
        except TypeError:
            return None
        if len(points) != 4:
            return None
        polygon: list[list[float]] = []
        for point in points:
            try:
                x = float(point[0]) / scale
                y = float(point[1]) / scale
            except (TypeError, ValueError, IndexError):
                return None
            if (
                not math.isfinite(x)
                or not math.isfinite(y)
                or not 0 <= x <= width
                or not 0 <= y <= height
            ):
                return None
            polygon.append([x, y])
        return polygon

    @staticmethod
    def _parse_scales(value: object) -> tuple[float, ...]:
        if value is None:
            return (1.0,)
        if isinstance(value, (str, bytes, Mapping)):
            raise PaddleOCRAdapterError("invalid_scales")
        try:
            scales = tuple(float(item) for item in value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise PaddleOCRAdapterError("invalid_scales") from None
        if (
            not 1 <= len(scales) <= 3
            or any(not math.isfinite(item) or not 0.5 <= item <= 3.0 for item in scales)
        ):
            raise PaddleOCRAdapterError("invalid_scales")
        return scales

    @staticmethod
    def _parse_confidence(value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            raise PaddleOCRAdapterError("invalid_confidence") from None
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise PaddleOCRAdapterError("invalid_confidence")
        return confidence

    def _validate_device(self, parameters: Mapping[str, JSONValue]) -> None:
        requested = parameters.get("device", self._config.device)
        if requested != "cpu":
            self._fail("unsupported_device")

    def _load_result(self, *, load_ms: int) -> Mapping[str, JSONValue]:
        return {
            "loaded": True,
            "device": self._config.device,
            "model_ids": [DETECTION_MODEL_ID, RECOGNITION_MODEL_ID],
            "model_revision": MODEL_REVISION,
            "load_ms": load_ms,
        }

    def _fail(self, code: str, *, cause: BaseException | None = None) -> None:
        self._last_error = code
        error = PaddleOCRAdapterError(code)
        if cause is None:
            raise error
        raise error from cause
