from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlaslens_api.schemas import DatasetQAReport


class DatasetQAError(ValueError):
    pass


class DatasetQAPolicy(BaseModel):
    """Versioned diagnostic policy. It never authorizes source-data mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: Literal["atlaslens-dataset-qa-v1"] = "atlaslens-dataset-qa-v1"
    read_only: Literal[True] = True
    max_manifest_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_records: int = Field(default=100_000, gt=0, le=1_000_000)
    max_file_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    max_decoded_pixels: int = Field(default=40_000_000, gt=0)
    max_side: int = Field(default=16_384, gt=0)
    near_blank_stddev: float = Field(default=2.0, ge=0)
    blur_variance_threshold: float = Field(default=20.0, ge=0)
    underexposed_mean: float = Field(default=8.0, ge=0, le=255)
    overexposed_mean: float = Field(default=247.0, ge=0, le=255)
    perceptual_distance: int = Field(default=4, ge=0, le=16)
    exif_conflict_km: float = Field(default=1.0, gt=0)
    minimum_foreground_coverage: float = Field(default=0.0001, ge=0, le=1)
    maximum_foreground_coverage: float = Field(default=0.9999, ge=0, le=1)
    tiny_component_pixels: int = Field(default=16, ge=1)
    max_issues: int = Field(default=9_999, ge=1, le=9_999)
    contact_sheet_limit: int = Field(default=64, ge=1, le=100)
    contact_sheet_thumbnail: int = Field(default=160, ge=32, le=512)

    @model_validator(mode="after")
    def validate_thresholds(self) -> DatasetQAPolicy:
        if self.underexposed_mean >= self.overexposed_mean:
            raise ValueError("exposure thresholds must be ordered")
        if self.minimum_foreground_coverage >= self.maximum_foreground_coverage:
            raise ValueError("mask coverage thresholds must be ordered")
        return self


@dataclass(frozen=True, slots=True)
class DatasetQARun:
    report: DatasetQAReport
    source_images: tuple[Path, ...]


def utc_now() -> datetime:
    return datetime.now(UTC)
