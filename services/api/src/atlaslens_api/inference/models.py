from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.providers.base import ExifResult
from atlaslens_api.schemas import AnalysisMode
from atlaslens_api.storage import LocalImageHandle

type InferenceProviderMode = Literal["disabled", "shadow", "candidate", "primary"]
type ResultClassification = Literal["real", "simulated"]
type InferenceOutcomeStatus = Literal["succeeded", "abstained", "skipped", "failed"]
type CalibrationState = Literal["uncalibrated", "preliminary", "calibrated"]
type ProviderOperationalStatus = Literal[
    "ready",
    "not_installed",
    "incomplete",
    "loading",
    "failed",
    "disabled",
    "unavailable",
]

_SAFE_TRACE_ID = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_SAFE_IDENTIFIER = r"^[a-z0-9][a-z0-9._-]*$"


class InferenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Private, bounded request passed only to local inference adapters."""

    image_handle: LocalImageHandle | None
    image_bytes: bytes | None
    width: int
    height: int
    exif: ExifResult | None
    analysis_mode: AnalysisMode
    top_k: int
    cancellation: asyncio.Event
    deadline: datetime
    trace_id: str
    max_input_bytes: int

    def __post_init__(self) -> None:
        if (self.image_handle is None) == (self.image_bytes is None):
            raise ValueError("exactly one private image input is required")
        if self.image_bytes is not None:
            if type(self.image_bytes) is not bytes:
                raise TypeError("inference bytes must be immutable bytes")
            if not self.image_bytes:
                raise ValueError("inference bytes must not be empty")
            if len(self.image_bytes) > self.max_input_bytes:
                raise ValueError("inference bytes exceed the request bound")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if self.max_input_bytes <= 0:
            raise ValueError("max_input_bytes must be positive")
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ValueError("deadline must be timezone-aware")
        if not _SAFE_TRACE_ID.fullmatch(self.trace_id):
            raise ValueError("trace_id is not a safe identifier")

    def __repr__(self) -> str:
        return (
            "InferenceRequest(image=<redacted>, exif=<redacted>, "
            f"dimensions={self.width}x{self.height}, mode={self.analysis_mode.value!r}, "
            f"top_k={self.top_k})"
        )


class InferenceCandidate(InferenceModel):
    rank: int = Field(ge=1, le=100)
    original_rank: int = Field(ge=1, le=100_000)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    raw_score: float
    limitations: tuple[str, ...] = Field(min_length=1, max_length=12)

    @field_validator("latitude", "longitude", "raw_score")
    @classmethod
    def finite_numbers(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("inference candidate values must be finite")
        return value


class InferenceFailure(InferenceModel):
    code: str = Field(min_length=1, max_length=120, pattern=_SAFE_IDENTIFIER)
    message_key: str = Field(min_length=1, max_length=160, pattern=r"^[a-z0-9._-]+$")
    retryable: bool
    subreason_code: str | None = Field(
        default=None, min_length=1, max_length=120, pattern=_SAFE_IDENTIFIER
    )


class InferenceProvenance(InferenceModel):
    provider_id: str = Field(min_length=1, max_length=80, pattern=_SAFE_IDENTIFIER)
    provider_revision: str = Field(min_length=1, max_length=120)
    model_revision: str = Field(min_length=1, max_length=120)
    runtime_revision: str = Field(min_length=1, max_length=120)
    execution_boundary: Literal["local"] = "local"
    source_kind: Literal[
        "verified_model",
        "verified_custom_artifact",
        "development_fixture",
        "provider_adapter",
        "custom_adapter",
    ]
    artifact_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class InferenceResult(InferenceModel):
    provider_id: str = Field(min_length=1, max_length=80, pattern=_SAFE_IDENTIFIER)
    provider_revision: str = Field(min_length=1, max_length=120)
    model_name: str = Field(min_length=1, max_length=120)
    model_revision: str = Field(min_length=1, max_length=120)
    runtime_revision: str = Field(min_length=1, max_length=120)
    classification: ResultClassification
    status: InferenceOutcomeStatus
    device: str | None = Field(default=None, max_length=80)
    dtype: str | None = Field(default=None, max_length=40)
    external_transfer: Literal[False] = False
    runtime_ms: int = Field(ge=0)
    score_semantics: str | None = Field(
        default=None, min_length=1, max_length=120, pattern=_SAFE_IDENTIFIER
    )
    normalization_method: str | None = Field(
        default=None, min_length=1, max_length=120, pattern=_SAFE_IDENTIFIER
    )
    calibration_state: CalibrationState
    candidates: tuple[InferenceCandidate, ...] = Field(default=(), max_length=100)
    warnings: tuple[str, ...] = Field(default=(), max_length=20)
    failure: InferenceFailure | None = None
    provenance: InferenceProvenance
    scenario_id: str | None = Field(
        default=None, min_length=1, max_length=80, pattern=r"^[a-z0-9_-]+$"
    )

    @model_validator(mode="after")
    def validate_result_semantics(self) -> InferenceResult:
        if self.status == "succeeded":
            if not self.candidates or self.failure is not None:
                raise ValueError("successful inference must contain candidates and no failure")
            if self.score_semantics is None or self.normalization_method is None:
                raise ValueError("successful inference must declare raw score semantics")
            ranks = [candidate.rank for candidate in self.candidates]
            if ranks != list(range(1, len(self.candidates) + 1)):
                raise ValueError("inference ranks must be contiguous")
        elif self.candidates:
            raise ValueError("non-successful inference cannot contain candidates")
        if self.status in {"failed", "skipped"} and self.failure is None:
            raise ValueError("failed or skipped inference must contain a safe failure")
        if self.status == "abstained" and self.failure is not None:
            raise ValueError("abstention is not a provider failure")
        if self.provenance.provider_id != self.provider_id:
            raise ValueError("inference provenance provider mismatch")
        if self.provenance.provider_revision != self.provider_revision:
            raise ValueError("inference provenance revision mismatch")
        if self.classification == "simulated":
            if self.provenance.source_kind != "development_fixture" or self.scenario_id is None:
                raise ValueError("simulated inference requires fixture provenance")
            if "warning.simulated_development_result" not in self.warnings:
                raise ValueError("simulated inference requires its warning marker")
        elif self.scenario_id is not None:
            raise ValueError("real inference cannot declare a simulation scenario")
        return self


class InferenceProviderStatus(InferenceModel):
    provider_id: str = Field(min_length=1, max_length=80, pattern=_SAFE_IDENTIFIER)
    provider_type: str = Field(min_length=1, max_length=80, pattern=_SAFE_IDENTIFIER)
    provider_revision: str = Field(min_length=1, max_length=120)
    mode: InferenceProviderMode
    available: bool
    status: ProviderOperationalStatus
    classification: ResultClassification
    model_name: str | None = Field(default=None, max_length=120)
    model_revision: str | None = Field(default=None, max_length=120)
    runtime_revision: str | None = Field(default=None, max_length=120)
    device: str | None = Field(default=None, max_length=80)
    calibration_state: CalibrationState | None = None
    reason_code: str | None = Field(
        default=None, min_length=1, max_length=120, pattern=_SAFE_IDENTIFIER
    )


class GeolocationInferenceProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def mode(self) -> InferenceProviderMode: ...

    @property
    def classification(self) -> ResultClassification: ...

    def status(self) -> InferenceProviderStatus: ...

    async def infer(self, request: InferenceRequest) -> InferenceResult: ...


def failure_result(
    status: InferenceProviderStatus,
    *,
    outcome_status: Literal["failed", "skipped"],
    code: str,
    retryable: bool,
    runtime_ms: int = 0,
    subreason_code: str | None = None,
) -> InferenceResult:
    return InferenceResult(
        provider_id=status.provider_id,
        provider_revision=status.provider_revision,
        model_name=status.model_name or "unavailable-model",
        model_revision=status.model_revision or "unavailable",
        runtime_revision=status.runtime_revision or "unavailable",
        classification=status.classification,
        status=outcome_status,
        device=status.device,
        runtime_ms=runtime_ms,
        calibration_state=status.calibration_state or "uncalibrated",
        failure=InferenceFailure(
            code=code,
            message_key=f"provider.inference.{code}",
            retryable=retryable,
            subreason_code=subreason_code,
        ),
        provenance=InferenceProvenance(
            provider_id=status.provider_id,
            provider_revision=status.provider_revision,
            model_revision=status.model_revision or "unavailable",
            runtime_revision=status.runtime_revision or "unavailable",
            source_kind=(
                "development_fixture"
                if status.classification == "simulated"
                else "provider_adapter"
            ),
        ),
        warnings=(
            ("warning.simulated_development_result",)
            if status.classification == "simulated"
            else ()
        ),
        scenario_id=("unavailable_simulation" if status.classification == "simulated" else None),
    )
