from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ManagementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelManifest(ManagementModel):
    model_id: Literal["geoclip"]
    model_name: str
    model_version: str
    implementation_revision: str
    license: str
    wheel_url: str
    wheel_filename: str
    wheel_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    clip_repo_id: str
    clip_revision: str
    clip_weights_filename: str
    clip_weights_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    required_clip_files: tuple[str, ...]
    score_type: Literal["uncalibrated_gallery_softmax"]
    normalization_method: Literal["softmax_over_fixed_gallery"]
    default_top_k: int = Field(ge=5, le=20)
    deduplication_radius_km: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def safe_artifact_paths(self) -> ModelManifest:
        if (
            PurePosixPath(self.wheel_filename).name != self.wheel_filename
            or "\\" in self.wheel_filename
        ):
            raise ValueError("model wheel filename must be a basename")
        if len(self.required_clip_files) != len(set(self.required_clip_files)):
            raise ValueError("required model files must be unique")
        for relative in self.required_clip_files:
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative:
                raise ValueError("required model file path is unsafe")
        if self.clip_weights_filename not in self.required_clip_files:
            raise ValueError("model weights must be a required file")
        return self


class InstallationReceipt(ManagementModel):
    schema_version: Literal[1] = 1
    model_id: Literal["geoclip"]
    model_version: str
    implementation_revision: str
    installed_at: datetime
    wheel_filename: str
    wheel_sha256: str
    clip_repo_id: str
    requested_clip_revision: str
    resolved_clip_revision: str
    clip_snapshot_directory: str
    clip_weights_filename: str
    clip_weights_sha256: str
    clip_file_sha256: dict[str, str]
    geoclip_assets_directory: str
    geoclip_asset_sha256: dict[str, str]


class ModelInfo(ManagementModel):
    model_id: str
    status: Literal["not_installed", "installed", "verified", "invalid"]
    model_version: str
    implementation_revision: str
    license: str
    score_type: str
    requested_clip_revision: str
    resolved_clip_revision: str | None = None
    storage_size_bytes: int = Field(ge=0)
    reason_code: str | None = None


class ModelTestResult(ManagementModel):
    status: Literal["passed", "failed"]
    hypothesis_count: int = Field(ge=0, le=20)
    duration_ms: int = Field(ge=0)
    reason_code: str | None = None
    subreason_code: str | None = Field(
        default=None, pattern=r"^[a-z0-9_]+$", max_length=120
    )
    provider_id: str | None = None
    model_revision: str | None = None
    implementation_revision: str | None = None
    device: str | None = None
    score_type: str | None = None
    calibration_state: str | None = None
    coordinate_validity: Literal["valid_wgs84"] | None = None


class ModelSmokeObservation(ManagementModel):
    hypothesis_count: int = Field(ge=0, le=20)
    provider_id: str
    model_revision: str
    implementation_revision: str
    device: str
    score_type: Literal["uncalibrated_gallery_softmax"]
    calibration_state: Literal["uncalibrated"]
    coordinate_validity: Literal["valid_wgs84"]


class ModelBenchmarkResult(ManagementModel):
    status: Literal["completed", "failed"]
    attempted: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    durations_ms: tuple[int, ...]
    runs: int = Field(ge=1)
    warmup_runs: int = Field(ge=0)
    cold_load_ms: int = Field(ge=0)
    warm_median_ms: float = Field(ge=0)
    warm_p95_ms: float = Field(ge=0)
    device: str
    peak_gpu_allocated_bytes: int = Field(ge=0)
    model_revision: str
    note: Literal["diagnostic_timings_not_a_performance_claim"] = (
        "diagnostic_timings_not_a_performance_claim"
    )


class EmbeddingModelManifest(ManagementModel):
    model_id: Literal["siglip2-b16-384"]
    model_name: str = Field(min_length=1, max_length=120)
    model_version: str = Field(min_length=1, max_length=120)
    repo_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    license: str = Field(min_length=1, max_length=80)
    weights_filename: str = Field(min_length=1, max_length=120)
    weights_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    required_files: tuple[str, ...]
    embedding_dimension: int = Field(gt=0, le=4096)
    image_size: int = Field(gt=0, le=4096)
    preprocessing_version: str = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def safe_snapshot_paths(self) -> EmbeddingModelManifest:
        if len(self.required_files) != len(set(self.required_files)):
            raise ValueError("required model files must be unique")
        for relative in self.required_files:
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative:
                raise ValueError("required model file path is unsafe")
        if self.weights_filename not in self.required_files:
            raise ValueError("model weights must be a required file")
        return self


class EmbeddingInstallationReceipt(ManagementModel):
    schema_version: Literal[1] = 1
    model_id: Literal["siglip2-b16-384"]
    model_version: str
    repo_id: str
    requested_revision: str
    resolved_revision: str
    installed_at: datetime
    snapshot_directory: Literal["snapshot"] = "snapshot"
    weights_filename: str
    weights_sha256: str
    file_sha256: dict[str, str]
    embedding_dimension: int
    preprocessing_version: str


class EmbeddingModelInfo(ManagementModel):
    model_id: str
    status: Literal["not_installed", "installed", "verified", "invalid"]
    model_version: str
    revision: str
    resolved_revision: str | None = None
    license: str
    embedding_dimension: int
    preprocessing_version: str
    storage_size_bytes: int = Field(ge=0)
    reason_code: str | None = None
