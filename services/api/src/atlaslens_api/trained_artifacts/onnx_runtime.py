from __future__ import annotations

import importlib
import io
import threading
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps

from atlaslens_api.inference.providers import CustomModelArtifact
from atlaslens_api.trained_artifacts.manager import TrainedArtifactManager

_MAX_RUNTIME_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024


class OnnxCoordinateRuntime:
    """Lazy reviewed ONNX adapter using in-memory model bytes.

    Passing model bytes prevents ONNX external-data paths from reading arbitrary
    files. No custom-operator library or remote code is registered.
    """

    def __init__(
        self,
        manager: TrainedArtifactManager,
        model_id: str,
        *,
        device: Literal["cpu", "cuda"],
    ) -> None:
        self._manager = manager
        self._model_id = model_id
        self._device = device
        self._lock = threading.Lock()
        self._session: Any | None = None
        self._manifest: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        with self._lock:
            if self._session is not None and self._manifest is not None:
                return self._session, self._manifest
            verified = self._manager.verified_artifact_payload(
                self._model_id, maximum_bytes=_MAX_RUNTIME_ARTIFACT_BYTES
            )
            manifest = verified.manifest
            if (
                manifest.artifact_format != "onnx"
                or manifest.runtime_adapter != "onnx-coordinate-v1"
                or manifest.output.type != "top_k_coordinates"
                or manifest.artifact_size_bytes > _MAX_RUNTIME_ARTIFACT_BYTES
            ):
                raise RuntimeError("unsupported custom artifact runtime")
            runtime = importlib.import_module("onnxruntime")
            available = set(runtime.get_available_providers())
            provider = (
                "CUDAExecutionProvider" if self._device == "cuda" else "CPUExecutionProvider"
            )
            if provider not in available:
                raise RuntimeError("requested ONNX execution provider is unavailable")
            options = runtime.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            model_bytes = verified.artifact_bytes
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._device == "cuda"
                else ["CPUExecutionProvider"]
            )
            session = runtime.InferenceSession(
                model_bytes,
                sess_options=options,
                providers=providers,
            )
            inputs = session.get_inputs()
            if len(inputs) != 1 or inputs[0].name != manifest.input.tensor_name:
                raise RuntimeError("custom model input contract mismatch")
            output_names = {item.name for item in session.get_outputs()}
            expected_outputs = {
                manifest.output.coordinates_output,
                manifest.output.scores_output,
            }
            if None in expected_outputs or not expected_outputs <= output_names:
                raise RuntimeError("custom model output contract mismatch")
            self._session = session
            self._manifest = manifest
            return session, manifest

    def predict(
        self,
        *,
        artifact: CustomModelArtifact,
        image_bytes: bytes,
        width: int,
        height: int,
        top_k: int,
        device: Literal["cpu", "cuda"],
    ) -> Sequence[object]:
        if device != self._device:
            raise ValueError("custom runtime device contract mismatch")
        session, manifest = self._load()
        if artifact.artifact_digest != self._manager.info(self._model_id).artifact_identity:
            raise ValueError("custom artifact identity mismatch")
        with Image.open(io.BytesIO(image_bytes)) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            if oriented.size != (width, height):
                raise ValueError("custom runtime image dimensions changed")
            resampling = (
                Image.Resampling.BICUBIC
                if manifest.input.resize_method == "bicubic"
                else Image.Resampling.BILINEAR
            )
            prepared = oriented.resize(
                (manifest.input.width, manifest.input.height), resample=resampling
            )
            pixels = np.asarray(prepared, dtype=np.float32) / 255.0
        if manifest.input.normalization == "mean_std":
            if manifest.input.mean is None or manifest.input.std is None:
                raise ValueError("custom normalization contract is incomplete")
            mean = np.asarray(manifest.input.mean, dtype=np.float32)
            std = np.asarray(manifest.input.std, dtype=np.float32)
            pixels = (pixels - mean) / std
        tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None, ...])
        coordinates_name = manifest.output.coordinates_output
        scores_name = manifest.output.scores_output
        if coordinates_name is None or scores_name is None:
            raise ValueError("custom output tensor names are unavailable")
        coordinates_raw, scores_raw = session.run(
            [coordinates_name, scores_name],
            {manifest.input.tensor_name: tensor},
        )
        coordinates = np.asarray(coordinates_raw, dtype=np.float64)
        scores = np.asarray(scores_raw, dtype=np.float64)
        if coordinates.ndim == 3 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 2
            or scores.ndim != 1
            or coordinates.shape[0] != scores.shape[0]
            or not 1 <= coordinates.shape[0] <= 100
        ):
            raise ValueError("custom ONNX output shape is invalid")
        count = min(top_k, manifest.output.top_k, coordinates.shape[0])
        return [
            {
                "original_rank": index + 1,
                "latitude": float(coordinates[index, 0]),
                "longitude": float(coordinates[index, 1]),
                "raw_score": float(scores[index]),
                "limitations": [
                    "custom_model_prediction_is_unverified",
                    "raw_score_is_not_probability",
                ],
            }
            for index in range(count)
        ]
