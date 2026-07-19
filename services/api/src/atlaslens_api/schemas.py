from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AnalysisMode(StrEnum):
    LOCAL_ONLY = "local_only"
    CLOUD_ASSISTED = "cloud_assisted"


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadinessResponse(StrictModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, Literal["ok", "error"]]


class ProviderCapability(StrictModel):
    provider_id: str
    enabled: bool
    available: bool
    execution_boundary: Literal["local", "cloud"]
    reason_code: str | None = None
    installed: bool | None = None
    verified: bool | None = None
    operational_status: (
        Literal[
            "ready",
            "not_ready",
            "not_installed",
            "dependencies_installed",
            "weights_prepared",
            "worker_unreachable",
            "model_load_failed",
            "inference_not_verified",
            "incomplete",
            "loading",
            "failed",
            "disabled",
            "unavailable",
        ]
        | None
    ) = None
    model_name: str | None = None
    model_revision: str | None = None
    device: str | None = None
    calibration_state: Literal["uncalibrated", "preliminary", "calibrated"] | None = None
    provider_type: str | None = Field(default=None, max_length=80)
    offline: bool | None = None
    license: str | None = Field(default=None, max_length=200)
    limitation: str | None = Field(default=None, max_length=300)
    weights_available: bool | None = None
    usable: bool | None = None
    execution_mode: Literal["in_process", "isolated_worker"] | None = None
    source_revision: str | None = Field(default=None, max_length=160)
    load_error: str | None = Field(default=None, max_length=160)
    key_configured: bool | None = None
    budget_available: bool | None = None


class RetentionPolicy(StrictModel):
    keep_uploads: bool
    ttl_seconds: int = Field(ge=0)
    originals_deleted_after_analysis: bool


class CapabilitiesResponse(StrictModel):
    supported_formats: list[Literal["jpeg", "png", "webp"]]
    max_upload_bytes: int = Field(gt=0)
    max_decoded_pixels: int = Field(gt=0)
    enabled_analysis_modes: list[AnalysisMode]
    providers: dict[str, ProviderCapability]
    retention: RetentionPolicy
    version: str


class ProviderStatusItem(StrictModel):
    provider_id: str = Field(min_length=1, max_length=80)
    provider_type: str = Field(min_length=1, max_length=80)
    mode: Literal["disabled", "shadow", "candidate", "primary"]
    available: bool
    status: Literal[
        "ready",
        "not_ready",
        "not_installed",
        "dependencies_installed",
        "weights_prepared",
        "worker_unreachable",
        "model_load_failed",
        "inference_not_verified",
        "incomplete",
        "loading",
        "failed",
        "disabled",
        "unavailable",
    ]
    classification: Literal["real", "simulated"]
    model_name: str | None = Field(default=None, max_length=120)
    model_revision: str | None = Field(default=None, max_length=120)
    device: str | None = Field(default=None, max_length=80)
    calibration_state: Literal["uncalibrated", "preliminary", "calibrated"] | None = None
    reason_code: str | None = Field(default=None, max_length=120)
    weights_available: bool | None = None
    usable: bool | None = None
    execution_mode: Literal["in_process", "isolated_worker"] | None = None
    source_revision: str | None = Field(default=None, max_length=160)
    load_error: str | None = Field(default=None, max_length=160)
    key_configured: bool | None = None
    budget_available: bool | None = None


class ProviderStatusResponse(StrictModel):
    providers: list[ProviderStatusItem] = Field(max_length=32)


class SystemIntelligenceModelCard(StrictModel):
    model_id: str = Field(min_length=1, max_length=160)
    display_name: str = Field(min_length=1, max_length=160)
    runtime_model_id: str | None = Field(default=None, max_length=160)
    repository_url: str | None = Field(default=None, max_length=500)
    purpose: str = Field(min_length=1, max_length=300)
    enabled: bool
    available: bool
    status: Literal[
        "ready",
        "not_installed",
        "dependencies_installed",
        "weights_prepared",
        "worker_unreachable",
        "model_load_failed",
        "inference_not_verified",
        "incomplete",
        "loading",
        "failed",
        "disabled",
        "unavailable",
    ]
    installed: bool | None = None
    weights_available: bool | None = None
    worker_reachable: bool | None = None
    model_loaded: bool | None = None
    load_verified: bool | None = None
    real_inference_verified: bool | None = None
    device: str | None = Field(default=None, max_length=80)
    execution_mode: Literal[
        "in_process",
        "isolated_worker",
        "isolated_process",
        "cloud",
        "not_integrated",
    ]
    source_revision: str | None = Field(default=None, max_length=160)
    model_revision: str | None = Field(default=None, max_length=160)
    license: str | None = Field(default=None, max_length=200)
    last_success_at: datetime | None = None
    last_latency_ms: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(
        default=None,
        max_length=120,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    current_participation: Literal[
        "primary",
        "candidate",
        "fallback",
        "descriptive",
        "verifier",
        "optional_review",
        "shadow",
        "disabled",
        "not_integrated",
    ]


class SystemIntelligenceLeakageAudit(StrictModel):
    status: Literal["passed"] = "passed"
    audit_version: Literal["atlaslens-leakage-audit-v1"] = (
        "atlaslens-leakage-audit-v1"
    )
    audit_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    checked_reference_count: int = Field(ge=0)
    descriptor_checked_count: int = Field(ge=0)
    excluded_reference_count: int = Field(ge=0)
    pre_index_excluded_reference_count: int | None = Field(default=None, ge=0)


class SystemIntelligenceReferenceIndexCard(StrictModel):
    index_id: Literal["turkiye_megaloc_reference_index"] = (
        "turkiye_megaloc_reference_index"
    )
    enabled: bool
    status: Literal["ready", "empty", "unavailable", "invalid", "disabled"]
    reason_code: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_]*$",
    )
    index_version: str | None = Field(default=None, max_length=160)
    descriptor_version: str | None = Field(default=None, max_length=160)
    count: int = Field(ge=0)
    sequences: int = Field(ge=0)
    countries: int | None = Field(default=None, ge=0)
    provinces: int | None = Field(default=None, ge=0)
    images_per_province: dict[str, int] | None = Field(default=None, max_length=1_000)
    source_distribution: dict[str, int] | None = Field(default=None, max_length=32)
    built_at: datetime | None = None
    disk_usage_bytes: int = Field(ge=0)
    leakage_status: Literal["passed", "failed", "not_run", "unavailable"]
    leakage_audit: SystemIntelligenceLeakageAudit | None = None
    duplicates: int | None = Field(default=None, ge=0)
    excluded: int | None = Field(default=None, ge=0)
    attributions: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list,
        max_length=100,
    )
    health: Literal["healthy", "degraded", "unavailable", "invalid", "disabled"]


class SystemIntelligenceResponse(StrictModel):
    active_pipeline_version: Literal["legacy-v1", "phase6c-v1"]
    models: list[SystemIntelligenceModelCard] = Field(min_length=10, max_length=16)
    reference_index: SystemIntelligenceReferenceIndexCard

    @model_validator(mode="after")
    def require_unique_models(self) -> SystemIntelligenceResponse:
        identifiers = [item.model_id for item in self.models]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("system intelligence model IDs must be unique")
        return self


class ModelStatusItem(StrictModel):
    model_id: str = Field(min_length=1, max_length=120)
    model_version: str | None = Field(default=None, max_length=120)
    artifact_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    mode: Literal["disabled", "shadow", "candidate", "primary"]
    verified: bool
    status: Literal["not_registered", "registered", "verified", "invalid", "disabled"]
    reason_code: str | None = Field(default=None, max_length=120)


class ModelStatusResponse(StrictModel):
    models: list[ModelStatusItem] = Field(max_length=32)


class AnalysisAccepted(StrictModel):
    id: UUID
    status: Literal["queued"] = "queued"
    status_url: str
    events_url: str
    delete_url: str


class Progress(StrictModel):
    stage: str
    percent: int = Field(ge=0, le=100)
    message_key: str


class ImageSummary(StrictModel):
    format: Literal["jpeg", "png", "webp"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    megapixels: float = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    orientation_normalized: bool
    exif_present: bool


class QualitySummary(StrictModel):
    blur_score: float = Field(ge=0, le=1)
    brightness_score: float = Field(ge=0, le=1)
    contrast_score: float = Field(ge=0, le=1)
    resolution_score: float = Field(ge=0, le=1)
    warnings: list[str]


class Provenance(StrictModel):
    provider_id: str
    provider_kind: str
    provider_version: str
    execution_boundary: Literal["local", "cloud"]
    model_name: str | None = None
    output_schema_version: str


class RetrievalCoverageContext(StrictModel):
    """Coverage and score semantics carried without inference-time context."""

    analysis_scope: Literal[
        "generic_upload", "ankara_reference_pilot", "coarse_provider_routed"
    ]
    coverage_status: Literal[
        "pilot_eligible", "insufficient", "provider_unavailable"
    ]
    coverage_label: str = Field(min_length=1, max_length=160)
    retrieval_provider: str = Field(min_length=1, max_length=160)
    retrieval_scope: str = Field(min_length=1, max_length=160)
    result_semantics: str = Field(min_length=1, max_length=240)
    abstained: bool
    abstention_reason: str | None = Field(default=None, min_length=1, max_length=120)
    similarity_semantics: Literal["cosine_similarity_not_confidence"] = (
        "cosine_similarity_not_confidence"
    )
    supported_region: str = Field(min_length=1, max_length=240)
    evidence_version: str = Field(min_length=1, max_length=160)
    benchmark_version: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_abstention(self) -> RetrievalCoverageContext:
        if self.abstained != (self.abstention_reason is not None):
            raise ValueError("retrieval abstention reason is inconsistent")
        if self.analysis_scope == "generic_upload" and self.coverage_status != "insufficient":
            raise ValueError("generic upload cannot claim limited pilot coverage")
        return self


class Evidence(StrictModel):
    id: str
    type: str
    label: str
    display_value: str | None
    confidence: Annotated[float, Field(ge=0, le=1)] | None
    confidence_basis: str
    source: str
    sensitive: bool
    provenance: Provenance
    retrieval_context: RetrievalCoverageContext | None = None


class GeoPoint(StrictModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class GeoJsonPoint(StrictModel):
    type: Literal["Point"] = "Point"
    coordinates: tuple[
        Annotated[float, Field(ge=-180, le=180)],
        Annotated[float, Field(ge=-90, le=90)],
    ]


class GeoJsonPolygon(StrictModel):
    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[tuple[float, float]]]

    @model_validator(mode="after")
    def validate_polygon(self) -> GeoJsonPolygon:
        if not self.coordinates or any(len(ring) < 4 for ring in self.coordinates):
            raise ValueError("polygon requires at least one ring with four positions")
        for ring in self.coordinates:
            for longitude, latitude in ring:
                if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                    raise ValueError("polygon coordinate out of bounds")
        return self


class Phase4ScoreContribution(StrictModel):
    feature: str = Field(min_length=1, max_length=80)
    raw_value: float
    weight: float
    contribution: float
    reason_code: str = Field(min_length=1, max_length=120)


class MapConstraintSummary(StrictModel):
    clue: str = Field(min_length=1, max_length=120)
    map_feature: str = Field(min_length=1, max_length=120)
    status: Literal["supported", "contradicted", "neutral", "unknown"]
    reliability: float = Field(ge=0, le=1)
    query_radius_km: float = Field(gt=0)
    provider: str = Field(min_length=1, max_length=80)
    limitation: str | None = Field(default=None, max_length=300)


class GeometryVerificationSummary(StrictModel):
    reference_id: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=80)
    status: Literal["supported", "inconclusive", "contradicted", "unavailable", "failed"]
    query_keypoints: int = Field(ge=0)
    reference_keypoints: int = Field(ge=0)
    raw_matches: int = Field(ge=0)
    filtered_matches: int = Field(ge=0)
    inliers: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0, le=1)
    query_coverage: float = Field(ge=0, le=1)
    reference_coverage: float = Field(ge=0, le=1)
    residual_error_px: float | None = Field(default=None, ge=0)
    robust_model_type: Literal["homography", "fundamental_matrix"] | None = None
    limitations: list[str] = Field(default_factory=list, max_length=12)
    runtime_ms: int = Field(ge=0)


class ReferenceAttribution(StrictModel):
    reference_id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=500)
    license: str = Field(min_length=1, max_length=500)
    attribution: str = Field(min_length=1, max_length=500)
    display_allowed: bool = False


class Phase4Assessment(StrictModel):
    classification: Literal[
        "model_only",
        "retrieval_only",
        "map_supported",
        "geometry_supported",
        "multi_source_supported",
        "contradicted",
        "abstained",
    ]
    relative_rank_score: float = Field(ge=0, le=1)
    score_semantics: Literal["uncalibrated_relative_rank"] = "uncalibrated_relative_rank"
    reranker_version: str = Field(min_length=1, max_length=80)
    score_breakdown: list[Phase4ScoreContribution] = Field(max_length=32)
    source_diversity: int = Field(ge=0)
    contributing_retrieval_hit_ids: list[str] = Field(max_length=100)
    map_observations: list[MapConstraintSummary] = Field(default_factory=list, max_length=32)
    geometry_results: list[GeometryVerificationSummary] = Field(default_factory=list, max_length=16)
    contradictions: list[str] = Field(default_factory=list, max_length=24)
    reference_attributions: list[ReferenceAttribution] = Field(default_factory=list, max_length=16)
    limitations: list[str] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def validate_provenance_ids(self) -> Phase4Assessment:
        if len(self.contributing_retrieval_hit_ids) != len(
            set(self.contributing_retrieval_hit_ids)
        ):
            raise ValueError("contributing retrieval hit ids must be unique")
        reference_ids = [item.reference_id for item in self.reference_attributions]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference attribution ids must be unique")
        return self


class Phase5BScoreContribution(StrictModel):
    feature: str = Field(min_length=1, max_length=80)
    raw_value: float
    weight: float
    contribution: float
    reason_code: str = Field(min_length=1, max_length=120)


class PlaceEvidenceSummary(StrictModel):
    matched_entity: str = Field(min_length=1, max_length=200)
    normalized_name: str = Field(min_length=1, max_length=200)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    region: str | None = Field(default=None, max_length=160)
    center: GeoPoint
    match_type: Literal[
        "country",
        "region",
        "city",
        "district",
        "road",
        "airport",
        "station",
        "public_landmark",
        "public_institution",
        "domain_suffix",
    ]
    text_similarity: float = Field(ge=0, le=1)
    ambiguity_count: int = Field(ge=1)
    evidence_strength: float = Field(ge=0, le=1)
    source: str = Field(min_length=1, max_length=500)
    dataset_version: str = Field(min_length=1, max_length=120)
    license: str = Field(min_length=1, max_length=500)


class RetrievalMatchSummary(StrictModel):
    reference_id: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=500)
    distance: float = Field(ge=0, le=2)
    relative_similarity: float = Field(ge=-1, le=1)
    center: GeoPoint
    geographic_cluster: str = Field(min_length=1, max_length=100)
    license: str = Field(min_length=1, max_length=500)
    attribution: str = Field(min_length=1, max_length=500)
    display_allowed: bool = False


class Phase5BAssessment(StrictModel):
    classification: Literal[
        "model_only",
        "place_supported",
        "retrieval_supported",
        "map_supported",
        "multi_source_supported",
        "contradicted",
    ]
    relative_rank_score: float = Field(ge=0, le=1)
    score_semantics: Literal["uncalibrated_relative_rank"] = "uncalibrated_relative_rank"
    reranker_version: Literal[
        "phase5b-v1", "phase6a-v1", "phase6b-v1", "phase6c-v1"
    ] = "phase5b-v1"
    score_breakdown: list[Phase5BScoreContribution] = Field(max_length=48)
    provider_diversity: int = Field(ge=0)
    source_diversity: int = Field(ge=0)
    place_matches: list[PlaceEvidenceSummary] = Field(default_factory=list, max_length=12)
    retrieval_matches: list[RetrievalMatchSummary] = Field(default_factory=list, max_length=16)
    map_observations: list[MapConstraintSummary] = Field(default_factory=list, max_length=32)
    supports: list[str] = Field(default_factory=list, max_length=24)
    contradictions: list[str] = Field(default_factory=list, max_length=24)
    movement_reasons: list[str] = Field(default_factory=list, max_length=24)
    limitations: list[str] = Field(default_factory=list, max_length=24)


class ProviderRunDiagnostic(StrictModel):
    provider_id: str = Field(min_length=1, max_length=80)
    provider_type: str = Field(min_length=1, max_length=80)
    status: Literal["succeeded", "abstained", "skipped", "failed"]
    duration_ms: int = Field(ge=0)
    reason_code: str | None = Field(default=None, max_length=120)
    device: str | None = Field(default=None, max_length=80)
    offline: bool


class ReferenceIndexDiagnostic(StrictModel):
    status: Literal["ready", "unavailable", "disabled", "incompatible"]
    index_id: str | None = Field(default=None, max_length=128)
    embedding_provider: str | None = Field(default=None, max_length=80)
    embedding_version: str | None = Field(default=None, max_length=120)
    dimension: int | None = Field(default=None, gt=0)
    image_count: int = Field(ge=0)
    checksum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class Phase5BDiagnostics(StrictModel):
    reranker_version: Literal["phase5b-v1", "phase6a-v1", "phase6b-v1"] = "phase5b-v1"
    providers: list[ProviderRunDiagnostic] = Field(max_length=16)
    reference_index: ReferenceIndexDiagnostic | None = None
    partial_failures: list[str] = Field(default_factory=list, max_length=24)


class PlaceLabelDiagnostics(StrictModel):
    country: str | None = Field(default=None, max_length=120)
    region: str | None = Field(default=None, max_length=160)
    city: str | None = Field(default=None, max_length=160)
    distance_to_place_km: float = Field(ge=0)
    source: str = Field(min_length=1, max_length=500)
    dataset_version: str = Field(min_length=1, max_length=120)
    license: str = Field(min_length=1, max_length=500)


class ModelPredictionDiagnostics(StrictModel):
    provider_id: str = Field(min_length=1, max_length=80)
    model_name: str = Field(min_length=1, max_length=120)
    model_revision: str = Field(min_length=1, max_length=120)
    implementation_revision: str = Field(min_length=1, max_length=120)
    device: str = Field(min_length=1, max_length=80)
    dtype: str = Field(min_length=1, max_length=40)
    raw_score: float = Field(ge=0, le=1)
    score_type: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    normalization_method: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    calibration_state: Literal["uncalibrated", "preliminary", "calibrated"]
    original_rank: int = Field(ge=1, le=100_000)
    inference_ms: int = Field(ge=0)
    external_transfer: Literal[False] = False
    limitations: list[str] = Field(min_length=1, max_length=12)
    place_label: PlaceLabelDiagnostics | None = None


class SceneClassSummary(StrictModel):
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1, max_length=160)
    pixel_ratio: float = Field(ge=0, le=1)
    percentage: float = Field(ge=0, le=100)


class SceneGroupSummary(StrictModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    pixel_ratio: float = Field(ge=0, le=1)
    percentage: float = Field(ge=0, le=100)


class SceneTagSummary(StrictModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    strength: float = Field(ge=0, le=1)
    strength_semantics: Literal["deterministic_heuristic_not_probability"] = (
        "deterministic_heuristic_not_probability"
    )
    reason: str = Field(min_length=1, max_length=240)


class SceneSegmentationSummary(StrictModel):
    status: Literal["completed"] = "completed"
    provider: str = Field(min_length=1, max_length=80)
    device: str = Field(min_length=1, max_length=80)
    inference_ms: float = Field(ge=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    semantic_label_names_available: bool
    dominant_classes: list[SceneClassSummary] = Field(max_length=24)
    scene_groups: list[SceneGroupSummary] = Field(default_factory=list, max_length=16)
    scene_tags: list[SceneTagSummary] = Field(default_factory=list, max_length=16)
    warnings: list[str] = Field(default_factory=list, max_length=20)


class Phase6BGeographicCandidateSummary(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=128)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    raw_score: float | None = None
    provider_rank: int = Field(ge=1, le=100)
    sample_support: int = Field(ge=1, le=4096)
    metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, max_length=16
    )


class Phase6BProviderPredictionSummary(StrictModel):
    provider: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    source_family: Literal[
        "mp16_family",
        "osv5m_family",
        "yfcc_family",
        "inat_family",
        "textual_evidence_family",
        "cloud_reasoning_family",
    ]
    status: Literal["completed", "skipped", "disabled", "failed", "timeout"]
    device: Literal["cuda", "cpu"]
    duration_ms: int = Field(ge=0)
    score_semantics: Literal["similarity", "direct_regression", "sample_density"]
    candidates: list[Phase6BGeographicCandidateSummary] = Field(
        default_factory=list, max_length=100
    )
    warnings: list[str] = Field(default_factory=list, max_length=20)
    reason_code: str | None = Field(default=None, max_length=120)
    diagnostics: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, max_length=20
    )


class Phase6BFusionMemberSummary(StrictModel):
    provider: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=160)
    candidate_id: str = Field(min_length=1, max_length=128)
    source_family: Literal[
        "mp16_family",
        "osv5m_family",
        "yfcc_family",
        "inat_family",
        "textual_evidence_family",
        "cloud_reasoning_family",
    ]
    provider_rank: int = Field(ge=1, le=100)
    sample_support: int = Field(ge=1, le=4096)


class Phase6BFusionContributionSummary(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    raw_value: float
    weight: float
    contribution: float
    reason: str = Field(min_length=1, max_length=240)


class Phase6BFusedCandidateSummary(StrictModel):
    cluster_id: str = Field(min_length=1, max_length=128)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0)
    relative_rank_score: float = Field(ge=0, le=1)
    score_semantics: Literal["uncalibrated_relative_rank_not_probability"] = (
        "uncalibrated_relative_rank_not_probability"
    )
    provider_count: int = Field(ge=1, le=16)
    independent_family_count: int = Field(ge=1, le=6)
    same_family_duplicate_support: int = Field(ge=0, le=16)
    spread_km: float = Field(ge=0)
    members: list[Phase6BFusionMemberSummary] = Field(min_length=1, max_length=100)
    contributions: list[Phase6BFusionContributionSummary] = Field(min_length=1, max_length=20)
    ocr_agreement: bool
    ocr_contradiction: bool


class Phase6BAgreementSummary(StrictModel):
    provider_count: int = Field(ge=0, le=16)
    independent_family_count: int = Field(ge=0, le=6)
    same_family_duplicate_support: int = Field(ge=0, le=16)
    geographic_disagreement: bool
    ocr_agreement: bool
    ocr_contradiction: bool


class Phase6BFusionSummary(StrictModel):
    version: Literal["phase6b-v1"] = "phase6b-v1"
    source_families: list[str] = Field(default_factory=list, max_length=8)
    agreement_summary: Phase6BAgreementSummary
    candidate_clusters: list[Phase6BFusedCandidateSummary] = Field(
        default_factory=list, max_length=20
    )
    provider_failures: list[str] = Field(default_factory=list, max_length=24)
    warnings: list[str] = Field(default_factory=list, max_length=20)


class Phase6BOCRDetectionSummary(StrictModel):
    redacted_text: str = Field(min_length=1, max_length=240)
    script: str = Field(min_length=1, max_length=40)
    confidence: float = Field(ge=0, le=1)
    provider: str = Field(min_length=1, max_length=80)
    profile: str = Field(min_length=1, max_length=80)


class Phase6BOCRSummary(StrictModel):
    provider: str = Field(min_length=1, max_length=80)
    status: Literal["completed", "abstained", "skipped", "failed"]
    detections: list[Phase6BOCRDetectionSummary] = Field(default_factory=list, max_length=32)
    place_evidence: list[PlaceEvidenceSummary] = Field(default_factory=list, max_length=12)
    fallback_used: bool = False
    reason_code: str | None = Field(default=None, max_length=120)


class Phase6BCloudCandidateAdjustment(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=80)
    adjustment: float = Field(ge=-0.15, le=0.15)
    reason: str = Field(min_length=1, max_length=240)


class Phase6BCloudObservedClue(StrictModel):
    type: Literal["text", "road", "sign", "architecture", "terrain", "vehicle", "other"]
    observation: str = Field(min_length=1, max_length=240)
    supports_candidate_ids: list[str] = Field(default_factory=list, max_length=8)
    contradicts_candidate_ids: list[str] = Field(default_factory=list, max_length=8)


class Phase6BCloudReviewSummary(StrictModel):
    decision: Literal["support_candidate", "reject_all", "insufficient"]
    selected_candidate_ids: list[str] = Field(default_factory=list, max_length=8)
    candidate_adjustments: list[Phase6BCloudCandidateAdjustment] = Field(
        default_factory=list, max_length=8
    )
    observed_clues: list[Phase6BCloudObservedClue] = Field(default_factory=list, max_length=12)
    suggested_place_query: str | None = Field(default=None, max_length=120)
    uncertainty_reason: str = Field(min_length=1, max_length=240)
    requires_high_detail: bool = False


class Phase6BCloudBudgetSummary(StrictModel):
    calls_today: int = Field(ge=0)
    estimated_month_spend_usd: float = Field(ge=0)
    configured_monthly_budget_usd: float = Field(gt=0)
    remaining_budget_usd: float = Field(ge=0)


class Phase6BCloudAssistSummary(StrictModel):
    allowed: bool
    triggered: bool
    status: Literal["completed", "skipped", "refused", "failed"]
    provider: Literal["openai"] = "openai"
    model: str = Field(min_length=1, max_length=120)
    prompt_version: Literal["openai-geo-review-v1"] = "openai-geo-review-v1"
    reason: str | None = Field(default=None, max_length=120)
    trigger_reasons: list[str] = Field(default_factory=list, max_length=8)
    cache_hit: bool = False
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    cost_estimate_version: str | None = Field(default=None, max_length=80)
    budget: Phase6BCloudBudgetSummary | None = None
    review: Phase6BCloudReviewSummary | None = None
    warnings: list[str] = Field(default_factory=list, max_length=4)
    cache_key: str | None = Field(default=None, pattern=r"^v1:[a-f0-9]{64}$", exclude=True)


class Phase6CHierarchicalCandidateSummary(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=128)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    search_level: Literal[
        "global", "catalogue", "regional_refinement", "turkiye_refinement"
    ]
    provider_rank: int = Field(ge=1, le=64)
    provider_score: float
    score_semantics: Literal["raw_cosine_similarity_not_confidence"] = (
        "raw_cosine_similarity_not_confidence"
    )
    grid_resolution_km: float = Field(gt=0)
    diversity_cluster: str = Field(pattern=r"^mode-[0-9]{2}$")
    nearest_name: str | None = Field(default=None, max_length=160)
    nearest_kind: str | None = Field(default=None, max_length=40)
    nearest_distance_km: float | None = Field(default=None, ge=0)


class Phase6CMegaLocMatchSummary(StrictModel):
    reference_id: str = Field(min_length=1, max_length=160)
    rank: int = Field(ge=1, le=200)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    similarity: float = Field(ge=-1, le=1)
    similarity_semantics: Literal["cosine_similarity_not_confidence"] = (
        "cosine_similarity_not_confidence"
    )
    confidence: None = None
    uncertainty_radius_m: float = Field(gt=0)
    source: Literal["mapillary", "kartaview", "manual"]
    source_family: str = Field(min_length=1, max_length=160)
    source_sequence_id: str = Field(min_length=1, max_length=160)
    source_url: str = Field(pattern=r"^https://", max_length=800)
    captured_at: datetime | None = None
    province: str = Field(min_length=1, max_length=160)
    city: str | None = Field(default=None, max_length=160)
    license: str = Field(min_length=1, max_length=500)
    attribution: str = Field(min_length=1, max_length=500)


class Phase6CG3ScoreSummary(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=160)
    rank: int = Field(ge=1, le=256)
    raw_score: float
    score_semantics: Literal["raw_g3_similarity_not_confidence"] = (
        "raw_g3_similarity_not_confidence"
    )


class Phase6CProviderRunSummary(StrictModel):
    provider_id: str = Field(min_length=1, max_length=120)
    status: Literal[
        "completed", "abstained", "skipped", "disabled", "unavailable", "failed", "timeout"
    ]
    source_revision: str | None = Field(default=None, max_length=160)
    model_revision: str | None = Field(default=None, max_length=160)
    duration_ms: int = Field(ge=0)
    candidates_produced: int = Field(ge=0, le=1_000)
    reason_code: str | None = Field(default=None, max_length=120)


class Phase6CFusionMemberSummary(StrictModel):
    evidence_kind: Literal[
        "geoclip_original",
        "geoclip_hierarchical",
        "osv_direct_regression",
        "plonk_samples",
        "megaloc_retrieval",
        "g3_verification",
        "ocr_place_match",
        "openai_review",
    ]
    provider: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    candidate_id: str = Field(min_length=1, max_length=128)
    source_family: str = Field(min_length=1, max_length=80)
    correlation_group: str = Field(min_length=1, max_length=120)
    score_semantics: str = Field(min_length=1, max_length=120)
    raw_value: float | None = None
    provider_rank: int = Field(ge=1, le=256)
    sample_support: int = Field(ge=1, le=100_000)
    provenance: str = Field(min_length=1, max_length=240)


class Phase6CFusionContributionSummary(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=120)
    raw_value: float
    normalized_value: float = Field(ge=0, le=1)
    weight: float
    contribution: float
    reason: str = Field(min_length=1, max_length=300)
    independent: bool
    correlation_group: str = Field(min_length=1, max_length=120)


class Phase6CFusedCandidateSummary(StrictModel):
    cluster_id: str = Field(min_length=1, max_length=128)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_km: float = Field(gt=0)
    relative_rank_score: float = Field(ge=0, le=1)
    score_semantics: Literal["uncalibrated_relative_rank_not_probability"] = (
        "uncalibrated_relative_rank_not_probability"
    )
    confidence: None = None
    calibrated: Literal[False] = False
    source_family_count: int = Field(ge=1, le=16)
    independent_source_family_count: int = Field(ge=1, le=16)
    correlated_source_family_count: int = Field(ge=0, le=16)
    provider_count: int = Field(ge=1, le=32)
    spread_km: float = Field(ge=0)
    publication_eligible: bool
    publication_basis: Literal[
        "at_least_two_independent_source_families",
        "candidate_recall_only_insufficient_independence",
    ]
    pre_diversity_rank: int = Field(ge=1)
    final_rank: int = Field(ge=1, le=50)
    rank_movement: int
    movement_reasons: list[str] = Field(min_length=1, max_length=4)
    members: list[Phase6CFusionMemberSummary] = Field(min_length=1, max_length=256)
    contributions: list[Phase6CFusionContributionSummary] = Field(
        min_length=1, max_length=16
    )


class Phase6CLeakageAuditSummary(StrictModel):
    status: Literal["passed", "failed", "not_run", "incomplete"]
    audit_version: str = Field(min_length=1, max_length=160)
    report_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    references_checked: int = Field(ge=0, le=100_000)
    references_excluded: int = Field(ge=0, le=100_000)
    reason_code: str | None = Field(default=None, max_length=120)


class Phase6CAblationCandidateSummary(StrictModel):
    rank: int = Field(ge=1, le=5)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_km: float = Field(gt=0)
    relative_rank_score: float = Field(ge=0, le=1)
    score_semantics: Literal["uncalibrated_relative_rank_not_probability"] = (
        "uncalibrated_relative_rank_not_probability"
    )
    publication_eligible: bool
    evidence_kinds: list[str] = Field(min_length=1, max_length=8)


class Phase6CAblationSummary(StrictModel):
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{0,79}$")
    candidate_count: int = Field(ge=0, le=50)
    publication_candidate_count: int = Field(ge=0, le=50)
    abstained: bool
    abstention_reason: str | None = Field(default=None, max_length=120)
    included_evidence_kinds: list[str] = Field(max_length=8)
    excluded_evidence_kinds: list[str] = Field(max_length=8)
    candidates: list[Phase6CAblationCandidateSummary] = Field(
        default_factory=list, max_length=5
    )


class Phase6CAnalysisSummary(StrictModel):
    schema_version: Literal["atlaslens-phase6c-analysis-v1"] = (
        "atlaslens-phase6c-analysis-v1"
    )
    pipeline_version: Literal["phase6c-v1"] = "phase6c-v1"
    fusion_version: Literal["phase6c-v1"] = "phase6c-v1"
    reference_index_version: str | None = Field(default=None, max_length=160)
    evaluation_run_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    )
    hierarchical_candidates: list[Phase6CHierarchicalCandidateSummary] = Field(
        default_factory=list, max_length=64
    )
    megaloc_matches: list[Phase6CMegaLocMatchSummary] = Field(
        default_factory=list, max_length=200
    )
    g3_scores: list[Phase6CG3ScoreSummary] = Field(default_factory=list, max_length=96)
    providers: list[Phase6CProviderRunSummary] = Field(default_factory=list, max_length=20)
    fusion_candidates: list[Phase6CFusedCandidateSummary] = Field(
        default_factory=list, max_length=50
    )
    publication_candidate_count: int = Field(default=0, ge=0, le=50)
    turkiye_signal_count: int = Field(default=0, ge=0, le=8)
    turkiye_signal_groups: list[str] = Field(default_factory=list, max_length=8)
    leakage_audit: Phase6CLeakageAuditSummary
    reference_attributions: list[str] = Field(default_factory=list, max_length=20)
    ablations: list[Phase6CAblationSummary] = Field(default_factory=list, max_length=16)
    cache_fingerprint: str = Field(pattern=r"^phase6c-v1:[a-f0-9]{64}$")


class GeoClipClusterSummary(StrictModel):
    cluster_id: str = Field(min_length=1, max_length=128)
    source: Literal["geoclip"] = "geoclip"
    member_count: int = Field(gt=0, le=100)
    member_ranks: list[int] = Field(min_length=1, max_length=100)
    max_raw_similarity: float = Field(ge=0, le=1)
    mean_raw_similarity: float = Field(ge=0, le=1)
    raw_score_type: str = Field(min_length=1, max_length=120)
    cluster_support: float = Field(ge=0, le=1)
    score_semantics: Literal["uncalibrated_relative_rank"] = "uncalibrated_relative_rank"

    @model_validator(mode="after")
    def validate_member_ranks(self) -> GeoClipClusterSummary:
        if len(self.member_ranks) != self.member_count:
            raise ValueError("cluster member count and ranks must agree")
        if any(rank < 1 for rank in self.member_ranks):
            raise ValueError("cluster member ranks must be positive")
        if len(set(self.member_ranks)) != len(self.member_ranks):
            raise ValueError("cluster member ranks must be unique")
        return self


class UncalibratedConfidenceSummary(StrictModel):
    label: Literal["low", "medium", "high", "very_high"]
    score: None = None
    calibrated: Literal[False] = False
    basis: list[str] = Field(min_length=1, max_length=16)


class ReverseGeocodeSummary(StrictModel):
    country: str | None = Field(default=None, max_length=120)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    region: str | None = Field(default=None, max_length=160)
    city: str | None = Field(default=None, max_length=160)
    district: str | None = Field(default=None, max_length=160)
    display_name: str = Field(min_length=1, max_length=200)
    provider: str = Field(min_length=1, max_length=500)
    dataset_version: str = Field(min_length=1, max_length=120)
    license: str = Field(min_length=1, max_length=500)


class Candidate(StrictModel):
    id: str
    rank: int = Field(ge=1)
    center: GeoPoint
    geometry: GeoJsonPoint | GeoJsonPolygon
    radius_km: float = Field(gt=0)
    uncertainty_basis: str
    confidence: Annotated[float, Field(ge=0, le=1)] | None
    confidence_kind: Literal["source_reliability", "uncalibrated_score", "calibrated_probability"]
    confidence_basis: str
    granularity: Literal["exact_metadata", "city", "region", "country", "broad_area"]
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    label: str | None = None
    source: str
    evidence_ids: list[str] = Field(min_length=1)
    evidence_summary: str
    provenance: list[Provenance] = Field(min_length=1)
    verification_status: Literal[
        "metadata_only", "unverified_model", "corroborated", "geometrically_verified"
    ]
    verified: bool
    phase4_assessment: Phase4Assessment | None = None
    phase5b_assessment: Phase5BAssessment | None = None
    model_prediction: ModelPredictionDiagnostics | None = None
    geoclip_cluster: GeoClipClusterSummary | None = None
    confidence_assessment: UncalibratedConfidenceSummary | None = None
    reverse_geocode: ReverseGeocodeSummary | None = None

    @model_validator(mode="after")
    def validate_phase_one_semantics(self) -> Candidate:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique")
        if self.verification_status == "unverified_model":
            if self.granularity == "exact_metadata":
                raise ValueError("unverified model candidates cannot be exact")
            if (self.confidence is not None and self.confidence > 0.40) or self.radius_km < 25:
                raise ValueError("unverified model confidence/radius policy violated")
            if self.verified:
                raise ValueError("unverified model candidate cannot be verified")
        if self.model_prediction is not None:
            if self.confidence is not None:
                raise ValueError("uncalibrated model candidates must not expose confidence")
            if self.confidence_kind != "uncalibrated_score":
                raise ValueError("model prediction confidence semantics are invalid")
            if self.verification_status != "unverified_model" or self.verified:
                raise ValueError("global model prediction must remain unverified")
            if self.radius_km < 750:
                raise ValueError("uncalibrated global model radius floor violated")
        return self


class Abstention(StrictModel):
    abstained: Literal[True] = True
    reason_code: str
    message_key: str


class FailureSummary(StrictModel):
    code: str
    message_key: str
    retryable: bool


class SimulationSummary(StrictModel):
    scenario_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_-]+$")
    warning_key: Literal["warning.simulated_development_result"] = (
        "warning.simulated_development_result"
    )
    watermark: Literal["SIMULATED DEVELOPMENT RESULT"] = "SIMULATED DEVELOPMENT RESULT"


class ProviderComparisonSummary(StrictModel):
    provider_id: str = Field(min_length=1, max_length=80)
    mode: Literal["shadow", "candidate"]
    status: Literal["succeeded", "abstained", "skipped", "failed"]
    runtime_ms: int = Field(ge=0)
    distance_to_primary_km: float | None = Field(default=None, ge=0)
    candidate_overlap: int | None = Field(default=None, ge=0, le=100)
    ranking_impact: Literal["none", "eligible_not_applied"] = "none"
    failure_code: str | None = Field(default=None, max_length=120)


class Analysis(StrictModel):
    id: UUID
    status: AnalysisStatus
    analysis_mode: AnalysisMode
    created_at: datetime
    expires_at: datetime | None
    progress: Progress
    image: ImageSummary | None = None
    quality: QualitySummary | None = None
    evidence: list[Evidence]
    candidates: list[Candidate]
    abstention: Abstention | None = None
    warnings: list[str]
    timings_ms: dict[str, int]
    fusion_policy_version: str
    pipeline_version: str = Field(default="legacy-v1", min_length=1, max_length=64)
    phase5b_diagnostics: Phase5BDiagnostics | None = None
    scene_analysis: SceneSegmentationSummary | None = None
    model_predictions: dict[str, Phase6BProviderPredictionSummary] | None = Field(
        default=None, max_length=8
    )
    fusion: Phase6BFusionSummary | None = None
    ocr: Phase6BOCRSummary | None = None
    cloud_assist: Phase6BCloudAssistSummary | None = None
    phase6c: Phase6CAnalysisSummary | None = None
    result_classification: Literal["real", "simulated"] = "real"
    simulation: SimulationSummary | None = None
    provider_comparisons: list[ProviderComparisonSummary] = Field(
        default_factory=list, max_length=8
    )
    failure: FailureSummary | None = None

    @model_validator(mode="after")
    def validate_result_integrity(self) -> Analysis:
        evidence_ids = {item.id for item in self.evidence}
        if any(not set(candidate.evidence_ids) <= evidence_ids for candidate in self.candidates):
            raise ValueError("candidate references unknown evidence")
        expected_ranks = list(range(1, len(self.candidates) + 1))
        if [candidate.rank for candidate in self.candidates] != expected_ranks:
            raise ValueError("candidate ranks must be contiguous and sorted")
        if self.status == AnalysisStatus.COMPLETED:
            if self.candidates and self.abstention is not None:
                raise ValueError("completed result cannot both locate and abstain")
            if not self.candidates and self.abstention is None:
                raise ValueError("completed result without candidates must abstain")
        if (self.result_classification == "simulated") != (self.simulation is not None):
            raise ValueError("simulated analysis classification is inconsistent")
        return self


class AnalysisHistoryItem(StrictModel):
    id: UUID
    created_at: datetime
    expires_at: datetime | None
    status: AnalysisStatus
    analysis_mode: AnalysisMode
    result_classification: Literal["real", "simulated"]
    image_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)
    provider_ids: list[str] = Field(max_length=32)
    primary_label: str | None = Field(default=None, max_length=200)
    candidate_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    runtime_ms: int | None = Field(default=None, ge=0)
    warning_count: int = Field(ge=0)
    source_retained: bool
    deletion_state: Literal["active"] = "active"


class AnalysisHistoryPage(StrictModel):
    items: list[AnalysisHistoryItem]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class RatioView(StrictModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None = Field(default=None, ge=0, le=1)


class EvaluationReportSummary(StrictModel):
    report_id: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_.-]+$")
    provider_id: str = Field(min_length=1, max_length=120)
    model_revision: str = Field(min_length=1, max_length=160)
    evaluation_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    image_count: int = Field(gt=0)
    calibration_state: Literal["uncalibrated", "preliminary", "calibrated"]
    country_top1: RatioView
    country_top5: RatioView
    region_top1: RatioView
    city_top1: RatioView
    recall_top1: dict[str, RatioView]
    mean_error_km: float | None = Field(default=None, ge=0)
    median_error_km: float | None = Field(default=None, ge=0)
    p95_error_km: float | None = Field(default=None, ge=0)
    abstention: RatioView
    provider_failure: RatioView
    latency_median_ms: float | None = Field(default=None, ge=0)
    latency_p95_ms: float | None = Field(default=None, ge=0)
    uncertainty_coverage: RatioView
    geographic_distribution: dict[str, int]
    scene_distribution: dict[str, int]
    exclusions: dict[str, int]
    limitations: list[str] = Field(max_length=24)


class EvaluationReportList(StrictModel):
    reports: list[EvaluationReportSummary]


class DatasetQAIssueView(StrictModel):
    code: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9_.-]+$")
    severity: Literal["error", "warning", "info"]
    asset_key: str = Field(min_length=1, max_length=160)
    field: str | None = Field(default=None, max_length=80)
    message_key: str = Field(min_length=1, max_length=160)
    safe_metrics: dict[str, str] = Field(default_factory=dict)


class DatasetQAReportSummary(StrictModel):
    report_id: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_.-]+$")
    schema_version: Literal[1] = 1
    dataset_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_type: Literal["geolocation", "segmentation", "mixed"]
    created_at: datetime
    scanned_images: int = Field(ge=0)
    scanned_masks: int = Field(ge=0)
    error_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)


class DatasetQAReport(StrictModel):
    summary: DatasetQAReportSummary
    checks: dict[str, Literal["passed", "failed", "warning", "unavailable"]]
    distributions: dict[str, dict[str, int]]
    issues: list[DatasetQAIssueView] = Field(max_length=10_000)
    outputs: list[str] = Field(max_length=8)
    limitations: list[str] = Field(max_length=24)


class DatasetQAReportList(StrictModel):
    reports: list[DatasetQAReportSummary]


class AnalysisEvent(StrictModel):
    event_id: str
    event_type: Literal["progress", "heartbeat", "completed", "failed", "deleted"]
    analysis_id: UUID
    occurred_at: datetime
    status: AnalysisStatus
    progress: Progress | None = None


class DeleteResponse(StrictModel):
    id: UUID
    deleted: Literal[True] = True


class ProblemDetails(StrictModel):
    type: str
    title: str
    status: int = Field(ge=400, le=599)
    code: str
    message_key: str
    request_id: str
    retry_after_seconds: int | None = Field(default=None, ge=0)
