from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.image_processing import CloudSafeDerivative
from atlaslens_api.schemas import AnalysisMode, PlaceEvidenceSummary, QualitySummary
from atlaslens_api.storage import LocalImageHandle


class ProviderModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutcomeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    ABSTAINED = "abstained"
    SKIPPED = "skipped"
    FAILED = "failed"


class ProviderDescriptor(ProviderModel):
    id: str
    kind: str
    version: str
    execution_boundary: Literal["local", "cloud"]
    criticality: Literal["required", "optional"]
    available: bool
    unavailable_reason_code: str | None = None
    model_name: str | None = None


class ProviderFailure(ProviderModel):
    code: Literal[
        "timeout",
        "unavailable",
        "disabled",
        "missing_dependency",
        "missing_secret",
        "upstream_auth",
        "rate_limited",
        "transient_upstream",
        "invalid_output",
        "unsupported_input",
        "privacy_denied",
        "internal_provider_error",
        "model_not_installed",
        "weights_incomplete",
        "checksum_mismatch",
        "cuda_out_of_memory",
        "unsupported_device",
        "image_decode_error",
        "inference_timeout",
        "invalid_model_output",
    ]
    retryable: bool
    attempts: int = Field(ge=0, le=3)
    duration_ms: int = Field(ge=0)
    subreason_code: str | None = Field(
        default=None, min_length=1, max_length=120, pattern=r"^[a-z0-9_]+$"
    )


@dataclass(frozen=True, slots=True)
class ProviderOutcome[T]:
    status: OutcomeStatus
    value: T | None = None
    failure: ProviderFailure | None = None

    @classmethod
    def succeeded(cls, value: T) -> ProviderOutcome[T]:
        return cls(status=OutcomeStatus.SUCCEEDED, value=value)

    @classmethod
    def abstained(cls) -> ProviderOutcome[T]:
        return cls(status=OutcomeStatus.ABSTAINED)

    @classmethod
    def skipped(cls, code: str) -> ProviderOutcome[T]:
        return cls(
            status=OutcomeStatus.SKIPPED,
            failure=ProviderFailure(
                code=code,
                retryable=False,
                attempts=0,
                duration_ms=0,
            ),
        )

    @classmethod
    def failed(
        cls,
        code: str,
        *,
        retryable: bool,
        attempts: int,
        duration_ms: int,
        subreason_code: str | None = None,
    ) -> ProviderOutcome[T]:
        return cls(
            status=OutcomeStatus.FAILED,
            failure=ProviderFailure(
                code=code,
                retryable=retryable,
                attempts=attempts,
                duration_ms=duration_ms,
                subreason_code=subreason_code,
            ),
        )


@dataclass(frozen=True, slots=True)
class InvocationContext:
    analysis_id: UUID
    request_id: str
    mode: AnalysisMode
    cloud_consent: bool
    deadline: datetime
    cancellation: asyncio.Event

    def __repr__(self) -> str:
        return (
            "InvocationContext(analysis_id=<redacted>, request_id=<redacted>, "
            f"mode={self.mode.value!r}, cloud_consent={self.cloud_consent!r})"
        )


class ExifResult(ProviderModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    altitude_m: float | None = None
    captured_at: str | None = None
    orientation: int | None = None


class OCRPolygonPoint(ProviderModel):
    """One normalized image-space point; it never contains geographic coordinates."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class OCRBlock(ProviderModel):
    redacted_text: str = Field(min_length=1, max_length=240)
    normalized_text: str = Field(min_length=1, max_length=240, repr=False)
    script: Literal[
        "latin",
        "cyrillic",
        "arabic",
        "han",
        "japanese",
        "hangul",
        "devanagari",
        "mixed",
        "unknown",
    ]
    language_hints: list[str] = Field(default_factory=list, max_length=6)
    confidence: float = Field(ge=0, le=1)
    confidence_semantics: Literal["uncalibrated_ocr_engine_score"] = "uncalibrated_ocr_engine_score"
    bounding_polygon: tuple[OCRPolygonPoint, OCRPolygonPoint, OCRPolygonPoint, OCRPolygonPoint]
    provider: str = Field(min_length=1, max_length=80)
    profile: str = Field(min_length=1, max_length=80)
    sensitive_content: bool


class OCRResult(ProviderModel):
    redacted_snippets: list[str] = Field(max_length=5)
    blocks: list[OCRBlock] = Field(default_factory=list, max_length=32)
    place_matches: list[PlaceEvidenceSummary] = Field(default_factory=list, max_length=12)


class VisionHypothesis(ProviderModel):
    label: str = Field(min_length=1, max_length=120)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    granularity: Literal["city", "region", "country", "broad_area"]
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    visible_clues: list[str] = Field(min_length=1, max_length=6)


class VisionClueResult(ProviderModel):
    hypotheses: list[VisionHypothesis] = Field(max_length=5)


class GlobalPredictionHypothesis(ProviderModel):
    rank: int = Field(ge=1, le=100)
    original_rank: int = Field(ge=1, le=100_000)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    raw_score: float
    score_type: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    normalization_method: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    calibration_state: Literal["uncalibrated", "preliminary", "calibrated"]
    limitations: list[str] = Field(min_length=1, max_length=12)

    @field_validator("latitude", "longitude", "raw_score")
    @classmethod
    def finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("global prediction values must be finite")
        return value


class GlobalPredictionResult(ProviderModel):
    provider_id: str = Field(min_length=1, max_length=80)
    model_name: str = Field(min_length=1, max_length=120)
    model_revision: str = Field(min_length=1, max_length=120)
    implementation_revision: str = Field(min_length=1, max_length=120)
    device: str = Field(min_length=1, max_length=80)
    dtype: str = Field(min_length=1, max_length=40)
    external_transfer: Literal[False] = False
    inference_ms: int = Field(ge=0)
    hypotheses: list[GlobalPredictionHypothesis] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def stable_ranks(self) -> GlobalPredictionResult:
        if [item.rank for item in self.hypotheses] != list(range(1, len(self.hypotheses) + 1)):
            raise ValueError("global prediction ranks must be contiguous")
        return self


class GlobalProviderStatus(ProviderModel):
    status: Literal[
        "ready",
        "not_installed",
        "incomplete",
        "loading",
        "failed",
        "disabled",
        "unavailable",
    ]
    installed: bool
    verified: bool
    model_name: str
    model_revision: str
    device: str | None = None
    calibration_state: Literal["uncalibrated", "preliminary", "calibrated"]
    reason_code: str | None = None


class ExifProvider(Protocol):
    descriptor: ProviderDescriptor

    async def extract(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[ExifResult]: ...


class ImageQualityProvider(Protocol):
    descriptor: ProviderDescriptor

    async def analyze(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[QualitySummary]: ...


class OCRProvider(Protocol):
    descriptor: ProviderDescriptor

    async def extract(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[OCRResult]: ...


class VisionClueProvider(Protocol):
    descriptor: ProviderDescriptor

    async def extract(
        self, derivative: CloudSafeDerivative, context: InvocationContext
    ) -> ProviderOutcome[VisionClueResult]: ...


class GlobalGeolocationProvider(Protocol):
    descriptor: ProviderDescriptor

    def status(self) -> GlobalProviderStatus: ...

    async def predict(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[GlobalPredictionResult]: ...


class RetrievalProvider(Protocol):
    """Preserved image-facing seam; Phase 3 retrieval starts from an embedding."""

    async def search(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[object]: ...
