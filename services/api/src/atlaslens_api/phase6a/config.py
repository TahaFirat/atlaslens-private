from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FEATURES = frozenset({"geoclip_cluster", "ocr_place_match", "language_consistency"})


def default_phase6a_config_path() -> Path:
    return Path(__file__).resolve().parents[5] / "config" / "reranking" / "phase6a-v1.json"


class Phase6AHybridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["phase6a-v1"]
    weights: dict[str, float]
    contradiction_penalty: float = Field(ge=0, le=1)
    minimum_ocr_evidence: float = Field(ge=0, le=1)
    strong_ocr_threshold: float = Field(ge=0, le=1)
    language_country_support: dict[str, tuple[str, ...]]
    max_candidates: int = Field(ge=1, le=8)
    model_only_radius_floor_km: float = Field(ge=750, le=5_000)
    ocr_supported_radius_floor_km: float = Field(gt=0, le=500)

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != FEATURES:
            raise ValueError("phase6a weights must contain every supported feature")
        if any(not math.isfinite(item) or item < 0 for item in value.values()):
            raise ValueError("phase6a weights must be finite and non-negative")
        if not math.isclose(sum(value.values()), 1.0, abs_tol=1e-9):
            raise ValueError("phase6a positive feature weights must sum to one")
        if value["geoclip_cluster"] <= max(value["ocr_place_match"], value["language_consistency"]):
            raise ValueError("GeoCLIP cluster evidence must remain dominant")
        return value

    @field_validator("language_country_support")
    @classmethod
    def validate_language_countries(
        cls, value: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        if any(
            not language
            or not countries
            or any(len(country) != 2 or country.upper() != country for country in countries)
            for language, countries in value.items()
        ):
            raise ValueError("language-country support must use bounded ISO-style codes")
        return value

    @model_validator(mode="after")
    def validate_ocr_thresholds(self) -> Phase6AHybridConfig:
        if self.strong_ocr_threshold < self.minimum_ocr_evidence:
            raise ValueError("strong OCR threshold must not be below the minimum")
        return self

    @classmethod
    def from_path(cls, path: Path) -> Phase6AHybridConfig:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("phase6a config is unreadable or invalid JSON") from exc
        return cls.model_validate(payload)
