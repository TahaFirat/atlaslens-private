from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlaslens_api.phase6b.openai_assist.models import OpenAIReviewUsage


def default_openai_geo_config_path() -> Path:
    return Path(__file__).resolve().parents[6] / "config" / "cloud" / "openai-geo-review-v1.json"


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class OpenAIPricingConfig(_ConfigModel):
    version: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9._-]+$")
    model: str = Field(min_length=1, max_length=120)
    effective_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    source_url: str = Field(
        min_length=1, max_length=240, pattern=r"^https://developers\.openai\.com/"
    )
    input_per_million_usd: Decimal = Field(ge=0, decimal_places=8)
    cached_input_per_million_usd: Decimal = Field(ge=0, decimal_places=8)
    output_per_million_usd: Decimal = Field(ge=0, decimal_places=8)

    def estimate(self, usage: OpenAIReviewUsage | None) -> Decimal | None:
        if usage is None or usage.input_tokens is None or usage.output_tokens is None:
            return None
        cached = usage.cached_input_tokens or 0
        uncached = usage.input_tokens - cached
        estimate = (
            Decimal(uncached) * self.input_per_million_usd
            + Decimal(cached) * self.cached_input_per_million_usd
            + Decimal(usage.output_tokens) * self.output_per_million_usd
        ) / Decimal(1_000_000)
        return estimate.quantize(Decimal("0.00000001"))


class OpenAIGeoReviewConfig(_ConfigModel):
    version: Literal["openai-geo-review-v1"]
    enabled: bool = False
    model: Literal["gpt-5.6-luna"] = "gpt-5.6-luna"
    reasoning_effort: Literal["none", "low"] = "none"
    image_detail: Literal["low"] = "low"
    max_output_tokens: int = Field(default=500, ge=1, le=500)
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    maximum_image_edge: int = Field(default=768, ge=256, le=768)
    jpeg_quality: int = Field(default=75, ge=50, le=85)
    monthly_budget_usd: Decimal = Field(default=Decimal("4.50"), gt=0, decimal_places=2)
    daily_call_limit: int = Field(default=10, ge=1, le=100)
    per_analysis_call_limit: int = Field(default=1, ge=1, le=1)
    cache_ttl_days: int = Field(default=30, ge=1, le=365)
    allow_high_detail_retry: Literal[False] = False
    budget_reservation_usd: Decimal = Field(default=Decimal("0.15"), gt=0, decimal_places=4)
    conservative_input_token_reservation: int = Field(default=100_000, ge=10_000, le=1_000_000)
    maximum_candidate_adjustment: float = Field(default=0.15, gt=0, le=0.15)
    low_margin_threshold: float = Field(default=0.08, ge=0, le=1)
    large_separation_km: float = Field(default=750, gt=0, le=40_100)
    extreme_dispersion_km: float = Field(default=1_000, gt=0, le=40_100)
    pricing: OpenAIPricingConfig

    @model_validator(mode="after")
    def validate_model_and_budget(self) -> OpenAIGeoReviewConfig:
        if self.pricing.model != self.model:
            raise ValueError("pricing model must match the configured review model")
        if self.budget_reservation_usd > self.monthly_budget_usd:
            raise ValueError("one-call reservation cannot exceed the monthly budget")
        conservative_cost = (
            Decimal(self.conservative_input_token_reservation) * self.pricing.input_per_million_usd
            + Decimal(self.max_output_tokens) * self.pricing.output_per_million_usd
        ) / Decimal(1_000_000)
        if self.budget_reservation_usd < conservative_cost:
            raise ValueError("one-call reservation is below the configured token cost envelope")
        return self

    @classmethod
    def from_path(cls, path: Path) -> OpenAIGeoReviewConfig:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("OpenAI geo-review config is unreadable or invalid JSON") from exc
        return cls.model_validate(payload)
