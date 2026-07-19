from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BoundedText = Annotated[str, Field(min_length=1, max_length=240)]
ShortCode = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z0-9._:-]+$")]
CandidateId = Annotated[
    str,
    Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]

_COORDINATE_PAIR = re.compile(
    r"(?<!\w)[+-]?\d{1,2}(?:\.\d+)?\s*[,;/]\s*[+-]?\d{1,3}(?:\.\d+)?(?!\w)"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class OpenAIReviewCandidate(_StrictModel):
    """Bounded candidate summary sent to the reviewer; coordinates are intentionally absent."""

    candidate_id: CandidateId
    rank: int = Field(ge=1, le=8)
    place_name: str | None = Field(default=None, min_length=1, max_length=160)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    source_families: tuple[ShortCode, ...] = Field(default=(), max_length=6)
    supports: tuple[BoundedText, ...] = Field(default=(), max_length=8)
    contradictions: tuple[BoundedText, ...] = Field(default=(), max_length=8)


class OpenAIGeoReviewEvidence(_StrictModel):
    candidates: tuple[OpenAIReviewCandidate, ...] = Field(default=(), max_length=8)
    ocr_evidence: tuple[BoundedText, ...] = Field(default=(), max_length=12)
    scene_evidence: tuple[BoundedText, ...] = Field(default=(), max_length=12)
    agreement: tuple[BoundedText, ...] = Field(default=(), max_length=12)
    contradictions: tuple[BoundedText, ...] = Field(default=(), max_length=12)
    image_quality: tuple[BoundedText, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def unique_candidates_and_ranks(self) -> OpenAIGeoReviewEvidence:
        identifiers = [item.candidate_id for item in self.candidates]
        ranks = [item.rank for item in self.candidates]
        if len(set(identifiers)) != len(identifiers) or len(set(ranks)) != len(ranks):
            raise ValueError("candidate identifiers and ranks must be unique")
        return self


class HardCaseSignals(_StrictModel):
    local_confidence_label: Literal["low", "medium", "high", "very_high"]
    top_candidate_margin: float | None = Field(default=None, ge=0, le=1)
    strong_model_disagreement: bool = False
    maximum_model_separation_km: float = Field(default=0, ge=0, le=40_100)
    strong_ocr_conflict: bool = False
    no_usable_candidate: bool = False
    candidate_dispersion_km: float = Field(default=0, ge=0, le=40_100)
    user_explicit_review: bool = False
    strong_ocr_and_independent_consensus: bool = False


class CandidateAdjustment(_StrictModel):
    candidate_id: CandidateId
    adjustment: float = Field(ge=-0.15, le=0.15)
    reason: BoundedText


class ObservedClue(_StrictModel):
    type: Literal["text", "road", "sign", "architecture", "terrain", "vehicle", "other"]
    observation: BoundedText
    supports_candidate_ids: tuple[CandidateId, ...] = Field(default=(), max_length=8)
    contradicts_candidate_ids: tuple[CandidateId, ...] = Field(default=(), max_length=8)


class OpenAIReviewStructuredOutput(_StrictModel):
    decision: Literal["support_candidate", "reject_all", "insufficient"]
    selected_candidate_ids: tuple[CandidateId, ...] = Field(default=(), max_length=8)
    candidate_adjustments: tuple[CandidateAdjustment, ...] = Field(default=(), max_length=8)
    observed_clues: tuple[ObservedClue, ...] = Field(default=(), max_length=12)
    suggested_place_query: str | None = Field(default=None, min_length=1, max_length=120)
    uncertainty_reason: BoundedText
    requires_high_detail: bool = False

    @field_validator("suggested_place_query")
    @classmethod
    def reject_coordinate_queries(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _COORDINATE_PAIR.search(value) or "\n" in value or "\r" in value:
            raise ValueError("suggested place queries cannot contain coordinates or control text")
        return value

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> OpenAIReviewStructuredOutput:
        selected = self.selected_candidate_ids
        adjustment_ids = [item.candidate_id for item in self.candidate_adjustments]
        if len(set(selected)) != len(selected) or len(set(adjustment_ids)) != len(adjustment_ids):
            raise ValueError("candidate references must be unique")
        if self.decision == "support_candidate" and not selected:
            raise ValueError("support_candidate requires a selected candidate")
        if self.decision == "reject_all" and selected:
            raise ValueError("reject_all cannot select a candidate")
        return self

    def validate_candidate_scope(
        self, candidate_ids: set[str], *, maximum_adjustment: float
    ) -> None:
        referenced = set(self.selected_candidate_ids)
        referenced.update(item.candidate_id for item in self.candidate_adjustments)
        for clue in self.observed_clues:
            referenced.update(clue.supports_candidate_ids)
            referenced.update(clue.contradicts_candidate_ids)
        if not referenced.issubset(candidate_ids):
            raise ValueError("review output referenced an unknown candidate")
        if any(abs(item.adjustment) > maximum_adjustment for item in self.candidate_adjustments):
            raise ValueError("candidate adjustment exceeds the configured bound")


class OpenAIReviewUsage(_StrictModel):
    input_tokens: int | None = Field(default=None, ge=0, le=10_000_000)
    cached_input_tokens: int | None = Field(default=None, ge=0, le=10_000_000)
    output_tokens: int | None = Field(default=None, ge=0, le=10_000_000)
    total_tokens: int | None = Field(default=None, ge=0, le=20_000_000)

    @model_validator(mode="after")
    def validate_usage_totals(self) -> OpenAIReviewUsage:
        if (
            self.input_tokens is not None
            and self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError("cached input cannot exceed input tokens")
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens < self.input_tokens + self.output_tokens
        ):
            raise ValueError("total tokens cannot be below input plus output")
        return self


class OpenAIGeoReviewResult(_StrictModel):
    status: Literal["completed", "skipped", "refused", "failed"]
    reason_code: ShortCode | None = None
    called: bool
    cache_hit: bool
    cache_key: str | None = Field(default=None, pattern=r"^v1:[0-9a-f]{64}$")
    provider: Literal["openai"] = "openai"
    model: str = Field(min_length=1, max_length=120)
    prompt_version: Literal["openai-geo-review-v1"] = "openai-geo-review-v1"
    trigger_reasons: tuple[ShortCode, ...] = Field(default=(), max_length=8)
    review: OpenAIReviewStructuredOutput | None = None
    usage: OpenAIReviewUsage | None = None
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0, decimal_places=8)
    cost_estimate_version: str | None = Field(default=None, min_length=1, max_length=80)
    cost_basis: Literal["usage_based", "reservation_upper_bound", "cache_hit"] | None = None
    suggested_place_verified: Literal[False] = False
    usage_recorded: bool = True
    warnings: tuple[ShortCode, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def validate_outcome(self) -> OpenAIGeoReviewResult:
        if self.cache_hit and self.called:
            raise ValueError("a cache hit cannot be a cloud call")
        if self.status == "completed" and self.review is None:
            raise ValueError("completed reviews require structured output")
        if self.status != "completed" and self.review is not None:
            raise ValueError("only completed reviews may contain structured output")
        if self.estimated_cost_usd is None and (
            self.cost_estimate_version is not None or self.cost_basis is not None
        ):
            raise ValueError("cost metadata requires an estimate")
        return self


class OpenAIGeoReviewCapability(_StrictModel):
    """Secret-free capability shape intended for the existing diagnostics adapter."""

    enabled: bool
    key_configured: bool
    model: str = Field(min_length=1, max_length=120)
    budget_available: bool


@dataclass(frozen=True, slots=True)
class OpenAIGeoReviewRequest:
    analysis_id: UUID
    image_bytes: bytes
    evidence: OpenAIGeoReviewEvidence
    signals: HardCaseSignals
    cloud_consent: bool
    image_valid: bool = True

    def __repr__(self) -> str:
        return (
            "OpenAIGeoReviewRequest(analysis_id=<redacted>, image_bytes=<redacted>, "
            f"cloud_consent={self.cloud_consent!r}, image_valid={self.image_valid!r})"
        )
