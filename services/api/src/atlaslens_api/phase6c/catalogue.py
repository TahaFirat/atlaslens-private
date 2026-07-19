from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CatalogueFile(_FrozenModel):
    name: str = Field(min_length=1, max_length=180)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CatalogueSource(_FrozenModel):
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=160)
    license: Literal["public-domain", "CC-BY-4.0"]
    url: str = Field(pattern=r"^https://", max_length=500)
    files: tuple[CatalogueFile, ...] = Field(min_length=1, max_length=4)


class CoordinateCatalogueRecord(_FrozenModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$", max_length=80)
    kind: Literal["country", "populated_place", "turkiye_province"]
    name: str = Field(min_length=1, max_length=160)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    source_record_id: str = Field(min_length=1, max_length=80)
    admin1_code: str | None = Field(default=None, pattern=r"^[0-9A-Z.]{1,12}$")
    source_modified: str | None = Field(default=None, pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    population: int | None = Field(default=None, ge=0)
    rank: int | None = Field(default=None, ge=0, le=20)

    @field_validator("latitude", "longitude")
    @classmethod
    def finite_coordinate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("catalogue coordinates must be finite")
        return value

    @model_validator(mode="after")
    def coherent_kind(self) -> CoordinateCatalogueRecord:
        if self.kind == "turkiye_province":
            if self.country_code != "TR" or self.admin1_code is None:
                raise ValueError("Türkiye province records require TR and an admin code")
        elif self.admin1_code is not None:
            raise ValueError("only province records may carry an admin code")
        return self


class CoordinateCatalogue(_FrozenModel):
    schema_version: Literal["atlaslens-coordinate-catalogue-v1"]
    catalogue_version: str = Field(min_length=1, max_length=180)
    built_at: str = Field(min_length=20, max_length=40)
    sources: tuple[CatalogueSource, ...] = Field(min_length=2, max_length=4)
    records: tuple[CoordinateCatalogueRecord, ...] = Field(min_length=400, max_length=2_000)

    @model_validator(mode="after")
    def validate_coverage(self) -> CoordinateCatalogue:
        identifiers = [item.id for item in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("coordinate catalogue identifiers must be unique")
        provinces = [item for item in self.records if item.kind == "turkiye_province"]
        if len(provinces) != 81 or len({item.admin1_code for item in provinces}) != 81:
            raise ValueError("coordinate catalogue must cover all 81 Türkiye provinces")
        if not any(item.kind == "country" for item in self.records):
            raise ValueError("coordinate catalogue requires country coordinates")
        if not any(item.kind == "populated_place" for item in self.records):
            raise ValueError("coordinate catalogue requires populated places")
        return self

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()

    def records_of_kind(
        self, kind: Literal["country", "populated_place", "turkiye_province"]
    ) -> tuple[CoordinateCatalogueRecord, ...]:
        return tuple(item for item in self.records if item.kind == kind)


def default_coordinate_catalogue_path() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "assets"
        / "geolocation"
        / "coordinate_catalogue_v1.json"
    )


def load_coordinate_catalogue(path: Path | None = None) -> CoordinateCatalogue:
    selected = (path or default_coordinate_catalogue_path()).resolve()
    if selected.is_symlink() or not selected.is_file() or selected.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("coordinate catalogue is missing or unsafe")
    return CoordinateCatalogue.model_validate_json(selected.read_text(encoding="utf-8"))
