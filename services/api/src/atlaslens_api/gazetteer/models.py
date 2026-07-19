from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GazetteerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class GazetteerMetadata(GazetteerModel):
    dataset: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=120)
    source: str = Field(min_length=1, max_length=500)
    license: str = Field(min_length=1, max_length=500)


class ResolvedPlace(GazetteerModel):
    label: str = Field(min_length=1, max_length=200)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    country: str | None = Field(default=None, max_length=120)
    region: str | None = Field(default=None, max_length=160)
    city: str | None = Field(default=None, max_length=160)
    distance_km: float = Field(ge=0)
    source: str = Field(min_length=1, max_length=500)
    dataset_version: str = Field(min_length=1, max_length=120)
    license: str = Field(min_length=1, max_length=500)

    @field_validator("distance_km")
    @classmethod
    def finite_distance(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("gazetteer distance must be finite")
        return value
