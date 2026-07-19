from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConstraintModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class MapClue(ConstraintModel):
    clue: str = Field(min_length=1, max_length=160)
    map_feature: str = Field(min_length=1, max_length=80)
    expected_present: bool = True


class MapConstraintObservation(ConstraintModel):
    clue: str = Field(min_length=1, max_length=160)
    map_feature: str = Field(min_length=1, max_length=80)
    status: Literal["supported", "contradicted", "neutral", "unknown"]
    reliability: float = Field(ge=0, le=1)
    query_radius_km: float = Field(gt=0, le=100)
    provider: str = Field(min_length=1, max_length=80)
    limitation: str | None = Field(default=None, max_length=300)

    @field_validator("reliability", "query_radius_km")
    @classmethod
    def finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value
