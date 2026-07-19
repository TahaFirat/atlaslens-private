from __future__ import annotations

import importlib.util
from typing import Literal

from atlaslens_api.inference.providers import (
    CustomModelArtifact,
    CustomTrainedModelProvider,
)
from atlaslens_api.trained_artifacts.errors import TrainedArtifactError
from atlaslens_api.trained_artifacts.manager import TrainedArtifactManager
from atlaslens_api.trained_artifacts.onnx_runtime import OnnxCoordinateRuntime


def build_custom_provider(
    manager: TrainedArtifactManager,
    model_id: str,
    *,
    enabled: bool,
    device: Literal["cpu", "cuda"],
    max_input_bytes: int,
) -> CustomTrainedModelProvider:
    if not enabled:
        return CustomTrainedModelProvider(None, None, mode="disabled", device=device)
    try:
        registered = manager.registered_artifact(model_id)
    except TrainedArtifactError:
        return CustomTrainedModelProvider(None, None, mode="disabled", device=device)
    manifest = registered.manifest
    receipt = registered.receipt
    adapter_supported = (
        manifest.artifact_format == "onnx"
        and manifest.runtime_adapter == "onnx-coordinate-v1"
        and manifest.output.type == "top_k_coordinates"
    )
    artifact = CustomModelArtifact(
        provider_id=manifest.provider_id,
        artifact_id=manifest.model_id,
        artifact_digest=receipt.artifact_identity,
        model_name=manifest.model_id,
        model_revision=manifest.model_version,
        runtime_revision=manifest.implementation_revision,
        adapter_id=manifest.runtime_adapter,
        verified=receipt.verified,
        adapter_supported=adapter_supported,
        supported_devices=("cpu", "cuda"),
        max_input_bytes=max_input_bytes,
        output_dtype="float32",
        score_semantics=manifest.output.score_type,
        normalization_method=manifest.input.preprocessing_version,
        calibration_state="uncalibrated",
    )
    runtime = None
    if adapter_supported and importlib.util.find_spec("onnxruntime") is not None:
        runtime = OnnxCoordinateRuntime(manager, model_id, device=device)
    return CustomTrainedModelProvider(
        artifact,
        runtime,
        mode=receipt.mode.value,
        device=device,
    )
