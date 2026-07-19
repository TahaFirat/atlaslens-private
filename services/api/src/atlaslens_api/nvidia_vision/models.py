"""Strict evidence models for bounded NVIDIA geolocation reasoning."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.providers.ocr import redact_sensitive_text

BoundedText = Annotated[str, Field(min_length=1, max_length=240)]
ClueId = Annotated[str, Field(pattern=r"^clue-[1-9][0-9]?$", min_length=6, max_length=7)]
HypothesisId = Annotated[
    str,
    Field(pattern=r"^hypothesis-[1-5]$", min_length=12, max_length=12),
]

NVIDIA_REASONING_SCHEMA_VERSION: Literal["phase3c1-nvidia-geolocation-v1"] = (
    "phase3c1-nvidia-geolocation-v1"
)
NVIDIA_CONFIDENCE_SEMANTICS: Literal["uncalibrated_model_self_assessment"] = (
    "uncalibrated_model_self_assessment"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(content=<redacted>)"

    def __str__(self) -> str:
        return self.__repr__()


def _safe_visible_text(value: str, *, maximum: int) -> str:
    redacted = redact_sensitive_text(value).text[:maximum]
    if not redacted:
        raise ValueError("visible clue text became empty after sanitization")
    return redacted


class NvidiaObservedClue(_StrictModel):
    """One concise observation, never a raw OCR transcript or hidden rationale."""

    clue_id: ClueId
    category: Literal[
        "language_or_script",
        "road_and_signage",
        "architecture",
        "terrain_and_climate",
        "utilities_and_infrastructure",
        "vehicles_and_traffic",
        "business_or_place_type",
        "other",
    ]
    observation: str = Field(min_length=1, max_length=180)
    strength: Literal["weak", "moderate", "strong"]

    @field_validator("observation")
    @classmethod
    def redact_observation(cls, value: str) -> str:
        return _safe_visible_text(value, maximum=180)


class NvidiaLocationHypothesis(_StrictModel):
    """One unverified WGS84 hypothesis with explicit uncertainty semantics."""

    hypothesis_id: HypothesisId
    label: str = Field(min_length=1, max_length=120)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_km: float = Field(ge=5, le=40_100)
    granularity: Literal["city", "region", "country", "broad_area"]
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    confidence: float = Field(ge=0, le=1)
    confidence_semantics: Literal["uncalibrated_model_self_assessment"] = (
        NVIDIA_CONFIDENCE_SEMANTICS
    )
    supporting_clue_ids: tuple[ClueId, ...] = Field(min_length=1, max_length=8)
    contradicting_clue_ids: tuple[ClueId, ...] = Field(default=(), max_length=8)
    limitations: tuple[BoundedText, ...] = Field(min_length=1, max_length=6)

    @field_validator("label")
    @classmethod
    def redact_label(cls, value: str) -> str:
        return _safe_visible_text(value, maximum=120)

    @field_validator("country_code", mode="before")
    @classmethod
    def normalize_country_code(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("limitations")
    @classmethod
    def redact_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_safe_visible_text(item, maximum=240) for item in value)

    @model_validator(mode="after")
    def unique_clue_references(self) -> NvidiaLocationHypothesis:
        supporting = set(self.supporting_clue_ids)
        contradicting = set(self.contradicting_clue_ids)
        if len(supporting) != len(self.supporting_clue_ids) or len(contradicting) != len(
            self.contradicting_clue_ids
        ):
            raise ValueError("hypothesis clue references must be unique")
        if supporting & contradicting:
            raise ValueError("one clue cannot both support and contradict a hypothesis")
        return self


class NvidiaGeolocationReasoning(_StrictModel):
    """Image-only evidence projection; it intentionally excludes chain of thought."""

    schema_version: Literal["phase3c1-nvidia-geolocation-v1"] = (
        NVIDIA_REASONING_SCHEMA_VERSION
    )
    decision: Literal["hypotheses", "abstain"]
    observed_clues: tuple[NvidiaObservedClue, ...] = Field(default=(), max_length=12)
    hypotheses: tuple[NvidiaLocationHypothesis, ...] = Field(default=(), max_length=5)
    abstention_reason: BoundedText | None = None
    uncertainty_summary: BoundedText

    @field_validator("abstention_reason", "uncertainty_summary")
    @classmethod
    def redact_summary(cls, value: str | None) -> str | None:
        return None if value is None else _safe_visible_text(value, maximum=240)

    @model_validator(mode="after")
    def validate_evidence_graph(self) -> NvidiaGeolocationReasoning:
        clue_ids = [item.clue_id for item in self.observed_clues]
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        if len(set(clue_ids)) != len(clue_ids) or len(set(hypothesis_ids)) != len(
            hypothesis_ids
        ):
            raise ValueError("clue and hypothesis identifiers must be unique")
        available_clues = set(clue_ids)
        referenced_clues = {
            clue_id
            for hypothesis in self.hypotheses
            for clue_id in (*hypothesis.supporting_clue_ids, *hypothesis.contradicting_clue_ids)
        }
        if not referenced_clues.issubset(available_clues):
            raise ValueError("hypothesis references an unknown clue")
        if self.decision == "abstain":
            if self.hypotheses or self.abstention_reason is None:
                raise ValueError("abstention requires a reason and no hypotheses")
        elif not self.hypotheses or self.abstention_reason is not None:
            raise ValueError("hypothesis decisions require hypotheses and no abstention reason")
        return self
