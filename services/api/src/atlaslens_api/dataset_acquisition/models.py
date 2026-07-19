from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SamplingCell(DatasetModel):
    cell_id: str = Field(min_length=1, max_length=100)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_m: int = Field(ge=10, le=10_000)
    target: int = Field(ge=1, le=500)
    country: str = Field(min_length=1, max_length=120)
    region: str | None = Field(default=None, max_length=160)
    continent: Literal[
        "Africa", "Asia", "Europe", "North America", "South America", "Oceania"
    ]


class CommonsReviewRecord(DatasetModel):
    approved: bool = False
    page_id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=500)
    source_record_id: str = Field(min_length=1, max_length=160)
    source_url: str = Field(pattern=r"^https://commons\.wikimedia\.org/", max_length=500)
    download_url: str = Field(pattern=r"^https://upload\.wikimedia\.org/", max_length=1000)
    source_sha1: str = Field(min_length=1, max_length=64)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    country: str = Field(min_length=1, max_length=120)
    region: str | None = Field(default=None, max_length=160)
    continent: Literal[
        "Africa", "Asia", "Europe", "North America", "South America", "Oceania"
    ]
    geographic_cell: str = Field(min_length=1, max_length=100)
    country_source: Literal["mediawiki_coordinates", "operator_sampling"]
    region_source: Literal["mediawiki_coordinates", "operator_sampling", "unavailable"]
    coordinate_kind: Literal["unknown", "camera_raw", "object", "manual"] = "unknown"
    coordinate_source: Literal["mediawiki_page_coordinates"] = (
        "mediawiki_page_coordinates"
    )
    license: str = Field(min_length=1, max_length=120)
    license_url: str = Field(min_length=1, max_length=500)
    attribution: str = Field(min_length=1, max_length=500)
    display_allowed: bool = False
    metadata_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class AcquisitionReceiptEntry(DatasetModel):
    asset_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    relative_path: str = Field(min_length=1, max_length=500)
    source_record_id: str = Field(min_length=1, max_length=160)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    perceptual_hash: str = Field(pattern=r"^[a-f0-9]{16}$")
    metadata_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class AcquisitionReceipt(DatasetModel):
    schema_version: Literal[1] = 1
    source: Literal["Wikimedia Commons"] = "Wikimedia Commons"
    entries: dict[str, AcquisitionReceiptEntry]


class DatasetValidationReport(DatasetModel):
    status: Literal["ready"] = "ready"
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    image_count: int = Field(ge=0)
    source_counts: dict[str, int]
    license_counts: dict[str, int]
    continent_counts: dict[str, int]
    country_counts: dict[str, int]
    geographic_cell_counts: dict[str, int]
    evaluation_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
