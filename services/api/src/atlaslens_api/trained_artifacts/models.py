from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ArtifactModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, allow_inf_nan=False
    )


class DeploymentMode(StrEnum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    CANDIDATE = "candidate"
    PRIMARY = "primary"


class ArtifactInput(ArtifactModel):
    tensor_name: str = Field(min_length=1, max_length=120)
    width: int = Field(gt=0, le=4096)
    height: int = Field(gt=0, le=4096)
    color_space: Literal["RGB"] = "RGB"
    layout: Literal["NCHW"] = "NCHW"
    dtype: Literal["float32"] = "float32"
    resize_method: Literal["bilinear", "bicubic"]
    normalization: Literal["zero_to_one", "mean_std"]
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    preprocessing_version: str = Field(
        min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$"
    )

    @model_validator(mode="after")
    def valid_normalization(self) -> ArtifactInput:
        if self.normalization == "mean_std":
            if self.mean is None or self.std is None or any(value <= 0 for value in self.std):
                raise ValueError("mean/std normalization requires three positive std values")
        elif self.mean is not None or self.std is not None:
            raise ValueError("zero-to-one normalization must not define mean/std")
        return self


class ArtifactOutput(ArtifactModel):
    type: Literal["top_k_coordinates", "geographic_cells", "embedding"]
    coordinate_order: Literal["lat_lon"] = "lat_lon"
    top_k: int = Field(ge=1, le=20)
    score_type: str = Field(
        min_length=1, max_length=80, pattern=r"^[a-z0-9._-]+$"
    )
    output_schema_version: str = Field(
        min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$"
    )
    coordinates_output: str | None = Field(default=None, max_length=120)
    scores_output: str | None = Field(default=None, max_length=120)
    cells_output: str | None = Field(default=None, max_length=120)
    embedding_dimension: int | None = Field(default=None, gt=0, le=16_384)
    labels_file: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def valid_output_contract(self) -> ArtifactOutput:
        if self.type == "top_k_coordinates" and (
            self.coordinates_output is None or self.scores_output is None
        ):
            raise ValueError("coordinate output requires coordinate and score tensor names")
        if self.type == "geographic_cells" and (
            self.cells_output is None or self.labels_file is None
        ):
            raise ValueError("geographic-cell output requires logits and labels")
        if self.type == "embedding" and self.embedding_dimension is None:
            raise ValueError("embedding output requires a dimension")
        if self.labels_file is not None:
            _safe_relative_file(self.labels_file)
        return self


class TrainingLineage(ArtifactModel):
    dataset_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    code_revision: str = Field(min_length=7, max_length=120)
    completed_at: datetime
    capture_family_fingerprint: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )


class EvaluationGate(ArtifactModel):
    report_required_before_promotion: Literal[True] = True
    minimum_sample_count: int = Field(ge=30)
    primary_minimum_sample_count: int = Field(ge=30)
    maximum_country_top1_regression: float = Field(ge=0, le=1)
    maximum_recall_200km_regression: float = Field(ge=0, le=1)
    maximum_median_error_increase: float = Field(ge=0)
    maximum_latency_p95_ms: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def primary_count_is_not_weaker(self) -> EvaluationGate:
        if self.primary_minimum_sample_count < self.minimum_sample_count:
            raise ValueError("primary sample gate cannot be weaker")
        return self


class ArtifactLicense(ArtifactModel):
    name: str = Field(min_length=1, max_length=200)
    status: Literal["approved", "pending", "rejected"]
    commercial_use: Literal["allowed", "restricted", "unknown"]
    review_reference: str = Field(min_length=1, max_length=200)


def _safe_relative_file(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or path.name != value:
        raise ValueError("artifact file must be a safe basename")


class TrainedArtifactManifest(ArtifactModel):
    schema_version: Literal["atlaslens-trained-artifact-v1"]
    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,119}$")
    provider_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,119}$")
    model_version: str = Field(min_length=1, max_length=120)
    implementation_revision: str = Field(min_length=1, max_length=120)
    task: Literal["global_geolocation", "image_embedding"]
    artifact_format: Literal["onnx", "safetensors"]
    runtime_adapter: str = Field(min_length=1, max_length=160)
    artifact_file: str = Field(min_length=1, max_length=160)
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_size_bytes: int = Field(gt=0, le=8 * 1024 * 1024 * 1024)
    input: ArtifactInput
    output: ArtifactOutput
    training: TrainingLineage
    evaluation: EvaluationGate
    license: ArtifactLicense

    @field_validator("artifact_file")
    @classmethod
    def safe_artifact_file(cls, value: str) -> str:
        _safe_relative_file(value)
        return value

    @model_validator(mode="after")
    def safe_runtime_contract(self) -> TrainedArtifactManifest:
        suffix = PurePosixPath(self.artifact_file).suffix.casefold()
        if self.artifact_format == "onnx" and (
            self.runtime_adapter != "onnx-coordinate-v1"
            or self.output.type != "top_k_coordinates"
            or suffix != ".onnx"
        ):
            raise ValueError("ONNX artifacts require the reviewed coordinate adapter")
        if self.artifact_format == "safetensors" and not self.runtime_adapter.startswith(
            "repository-architecture:"
        ):
            raise ValueError("safetensors requires a repository-known architecture")
        if self.artifact_format == "safetensors" and suffix != ".safetensors":
            raise ValueError("safetensors artifact extension is invalid")
        if self.task == "global_geolocation" and self.output.type == "embedding":
            raise ValueError("a global provider cannot expose an embedding-only output")
        if self.task == "global_geolocation" and not self.output.score_type.startswith(
            "uncalibrated_"
        ):
            raise ValueError("global model scores must be explicitly uncalibrated")
        if self.task == "image_embedding" and self.output.type != "embedding":
            raise ValueError("an embedding provider must expose an embedding")
        return self


class RegisteredArtifactReceipt(ArtifactModel):
    schema_version: Literal[1] = 1
    model_id: str
    provider_id: str
    model_version: str
    artifact_file: str
    artifact_sha256: str
    artifact_size_bytes: int
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_identity: str = Field(pattern=r"^[a-f0-9]{64}$")
    registered_at: datetime
    verified_at: datetime | None = None
    verified: bool = False
    mode: DeploymentMode = DeploymentMode.DISABLED


class PromotionReport(ArtifactModel):
    schema_version: Literal[1] = 1
    model_id: str
    model_version: str
    artifact_identity: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    paired_sample_set_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sample_count: int = Field(gt=0)
    baseline_provider_id: str = Field(min_length=1, max_length=120)
    candidate_provider_id: str = Field(min_length=1, max_length=120)
    country_top1_delta: float = Field(ge=-1, le=1)
    recall_200km_delta: float = Field(ge=-1, le=1)
    median_error_increase_km: float
    latency_p95_ms: float = Field(ge=0)
    safety_checks_passed: bool
    output_schema_compatible: bool
    lineage_verified: bool
    license_approved: bool
    subgroup_regressions_passed: bool
    runtime_isolation_verified: bool
    operator_approved: bool
    limitations: tuple[str, ...] = ()


class PromotionReceipt(ArtifactModel):
    schema_version: Literal[1] = 1
    model_id: str
    model_version: str
    artifact_identity: str
    from_mode: DeploymentMode
    to_mode: DeploymentMode
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    promoted_at: datetime


class ArtifactInfo(ArtifactModel):
    model_id: str
    provider_id: str | None = None
    model_version: str | None = None
    artifact_identity: str | None = None
    mode: DeploymentMode = DeploymentMode.DISABLED
    verified: bool = False
    status: Literal["not_registered", "registered", "verified", "invalid", "disabled"]
    reason_code: str | None = None
