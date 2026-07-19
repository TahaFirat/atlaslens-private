"""Explicit, auditable NVIDIA hosted vision-model profiles."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from .errors import NvidiaConfigurationError

NVIDIA_ACTIVE_VISION_MODEL: Final = "qwen/qwen3.5-122b-a10b"
NVIDIA_DEPRECATED_VISION_MODEL: Final = "qwen/qwen3.5-397b-a17b"


@dataclass(frozen=True, slots=True)
class NvidiaVisionCapabilities:
    """Capabilities AtlasLens is permitted to use for a hosted model."""

    image_input: bool
    structured_json: bool
    status_polling: bool


@dataclass(frozen=True, slots=True)
class NvidiaVisionOutputMode:
    """Exact prompt/output compatibility policy for one model profile."""

    name: Literal[
        "strict_json_object",
        "strict_json_object_or_single_json_fence",
    ]
    enable_thinking: bool
    allow_single_json_fence: bool
    use_reasoning_content: Literal[False] = False


@dataclass(frozen=True, slots=True)
class NvidiaVisionModelProfile:
    """Immutable request policy for one exact NVIDIA model identifier."""

    model_id: str
    temperature: float
    top_p: float
    top_k: int | None
    presence_penalty: float | None
    repetition_penalty: float | None
    seed: int | None
    max_output_tokens: int
    default_total_timeout_seconds: float
    maximum_total_timeout_seconds: float
    capabilities: NvidiaVisionCapabilities
    output_mode: NvidiaVisionOutputMode
    recommended: bool
    deprecated: bool


_VISION_CAPABILITIES = NvidiaVisionCapabilities(
    image_input=True,
    structured_json=True,
    status_polling=True,
)

_ACTIVE_OUTPUT_MODE = NvidiaVisionOutputMode(
    name="strict_json_object_or_single_json_fence",
    enable_thinking=False,
    allow_single_json_fence=True,
)

_DEPRECATED_OUTPUT_MODE = NvidiaVisionOutputMode(
    name="strict_json_object",
    enable_thinking=False,
    allow_single_json_fence=False,
)

NVIDIA_VISION_MODEL_PROFILES = MappingProxyType(
    {
        NVIDIA_ACTIVE_VISION_MODEL: NvidiaVisionModelProfile(
            model_id=NVIDIA_ACTIVE_VISION_MODEL,
            temperature=0.7,
            top_p=0.8,
            top_k=None,
            presence_penalty=None,
            repetition_penalty=None,
            seed=0,
            max_output_tokens=2_048,
            default_total_timeout_seconds=90.0,
            maximum_total_timeout_seconds=120.0,
            capabilities=_VISION_CAPABILITIES,
            output_mode=_ACTIVE_OUTPUT_MODE,
            recommended=True,
            deprecated=False,
        ),
        NVIDIA_DEPRECATED_VISION_MODEL: NvidiaVisionModelProfile(
            model_id=NVIDIA_DEPRECATED_VISION_MODEL,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            seed=None,
            max_output_tokens=2_048,
            default_total_timeout_seconds=90.0,
            maximum_total_timeout_seconds=120.0,
            capabilities=_VISION_CAPABILITIES,
            output_mode=_DEPRECATED_OUTPUT_MODE,
            recommended=False,
            deprecated=True,
        ),
    }
)


def get_nvidia_vision_model_profile(model_id: str) -> NvidiaVisionModelProfile:
    """Resolve an exact allowlisted model ID or fail closed."""

    try:
        return NVIDIA_VISION_MODEL_PROFILES[model_id]
    except KeyError:
        raise NvidiaConfigurationError("nvidia_vision_model_not_authorized") from None
