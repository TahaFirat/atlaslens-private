from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from atlaslens_api.place_evidence.normalization import Script


class PlaceEvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ForwardPlaceMatch(PlaceEvidenceModel):
    geoname_id: int = Field(gt=0)
    matched_entity: str = Field(min_length=1, max_length=200)
    normalized_name: str = Field(min_length=1, max_length=200)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    region: str | None = Field(default=None, max_length=160)
    city: str | None = Field(default=None, max_length=160)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    match_type: str = Field(min_length=1, max_length=40)
    text_similarity: float = Field(ge=0, le=1)
    ambiguity_count: int = Field(ge=1)
    population: int = Field(ge=0)
    script: Script
    language_hint: str | None = Field(default=None, max_length=20)


@dataclass(frozen=True, slots=True, repr=False)
class OCRTextObservation:
    text: str
    confidence: float
    script: Script
    language_hints: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return (
            "OCRTextObservation(text=<redacted>, "
            f"confidence={self.confidence!r}, script={self.script!r})"
        )

