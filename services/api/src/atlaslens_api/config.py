from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_private_data_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "AtlasLens"
    return Path.home() / ".atlaslens"


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Validated server configuration with privacy-preserving defaults."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = "sqlite:///./data/atlaslens.db"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    app_version: str = "0.1.0"
    app_env: Literal["production", "development", "test"] = "production"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    max_decoded_pixels: int = Field(default=40_000_000, ge=1)
    max_image_dimension: int = Field(default=16_384, ge=1)
    retention_ttl_seconds: int = Field(default=3600, ge=0)
    keep_uploads: bool = False
    temp_storage_dir: Path = Path("./data/tmp")

    ocr_enabled: bool = False
    ocr_provider: Literal["rapidocr", "tesseract"] = "rapidocr"
    tesseract_cmd: str | None = None
    rapidocr_enabled: bool = True
    rapidocr_device: Literal["cpu", "cuda"] = "cpu"
    rapidocr_timeout_seconds: float = Field(default=30.0, gt=0, le=60)
    rapidocr_model_root: Path = Field(
        default_factory=lambda: _default_private_data_root() / "models" / "rapidocr-3.9.1"
    )
    openai_api_key: SecretStr | None = None
    openai_vision_model: str = "gpt-5.6-sol"
    openai_vision_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    cloud_vision_provider: Literal["openai", "nvidia"] = "openai"
    nvidia_vision_enabled: bool = False
    nvidia_api_key: SecretStr | None = None
    nvidia_vision_model: Literal[
        "qwen/qwen3.5-122b-a10b", "qwen/qwen3.5-397b-a17b"
    ] = "qwen/qwen3.5-122b-a10b"
    nvidia_vision_timeout_seconds: float = Field(default=90.0, gt=0, le=120)
    nvidia_vision_maximum_image_edge: int = Field(default=1_280, ge=256, le=1_280)
    nvidia_vision_jpeg_quality: int = Field(default=82, ge=50, le=90)

    global_model_enabled: bool = True
    global_model_device: Literal["auto", "cpu", "cuda"] = "auto"
    global_model_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    global_model_max_concurrency: int = Field(default=1, ge=1, le=4)
    geoclip_internal_top_k: int = Field(default=50, ge=5, le=100)
    enable_mock_inference: bool = False
    # Mock inference is disabled by default; a scenario must be selected explicitly.
    mock_scenario: str = "development_demo"
    mock_inference_fixture: str | None = None
    mock_inference_mode: Literal["shadow", "candidate", "primary"] = "primary"
    custom_model_enabled: bool = True
    custom_model_id: str = "atlaslens-custom-geolocation"
    custom_model_device: Literal["cpu", "cuda"] = "cpu"
    model_cache_dir: Path = Field(default_factory=lambda: _default_private_data_root() / "models")
    evaluation_report_dir: Path = Field(
        default_factory=lambda: _default_private_data_root() / "reports" / "evaluations"
    )
    dataset_qa_report_dir: Path = Field(
        default_factory=lambda: _default_private_data_root() / "reports" / "dataset-qa"
    )
    operator_api_enabled: bool = False
    gazetteer_cache_dir: Path = Field(
        default_factory=lambda: _default_private_data_root() / "gazetteer"
    )

    phase6a_enabled: bool = True
    segmentation_enabled: bool = False
    segmentation_checkpoint: Path = Path("./last_checkpoint.pt")
    segmentation_model_dir: Path = Field(
        default_factory=lambda: _default_private_data_root()
        / "models"
        / "atlaslens-segformer-b2-v4"
    )
    segmentation_device: Literal["auto", "cpu", "cuda"] = "auto"
    segmentation_min_class_ratio: float = Field(default=0.001, gt=0, le=0.1)
    segmentation_max_dominant_classes: int = Field(default=12, ge=1, le=24)
    segmentation_timeout_seconds: float = Field(default=90.0, gt=0, le=300)
    geoclip_cluster_radius_km: float = Field(default=40.0, gt=0, le=2_000)
    reverse_geocode_top_k: int = Field(default=10, ge=1, le=50)
    reverse_geocode_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    # Phase 6B extends Phase 6A without replacing its providers. Optional heavy
    # models use isolated workers and are never installed or downloaded at startup.
    phase6b_enabled: bool = True
    phase6b_worker_root: Path = Field(
        default_factory=lambda: _default_project_root() / ".local" / "workers"
    )
    phase6b_model_root: Path = Field(
        default_factory=lambda: _default_project_root() / ".local" / "models" / "phase6b"
    )
    phase6b_fusion_config: Path = Field(
        default_factory=lambda: _default_project_root() / "config" / "reranking" / "phase6b-v1.json"
    )
    osv5m_enabled: bool = True
    osv5m_worker_enabled: bool = True
    osv5m_worker_host: Literal["127.0.0.1"] = "127.0.0.1"
    osv5m_worker_port: int = Field(default=8791, ge=1, le=65535)
    osv5m_worker_python: Path = Field(
        default_factory=lambda: _default_project_root()
        / ".local"
        / "workers"
        / "osv5m"
        / ".venv"
        / "Scripts"
        / "python.exe"
    )
    osv5m_model_id: str = "osv5m/baseline"
    osv5m_model_revision: str = "71548b90ac4a1aa7c37839841f411a06da82b1a6"
    osv5m_device: Literal["auto", "cpu", "cuda"] = "cpu"
    osv5m_timeout_seconds: float = Field(default=60.0, gt=0, le=180)
    plonk_enabled: bool = True
    plonk_worker_enabled: bool = True
    plonk_worker_host: Literal["127.0.0.1"] = "127.0.0.1"
    plonk_worker_port: int = Field(default=8792, ge=1, le=65535)
    plonk_worker_python: Path = Field(
        default_factory=lambda: _default_project_root()
        / ".local"
        / "workers"
        / "plonk"
        / ".venv"
        / "Scripts"
        / "python.exe"
    )
    plonk_osv_model_id: str = "nicolas-dufour/PLONK_OSV_5M"
    plonk_yfcc_model_id: str = "nicolas-dufour/PLONK_YFCC"
    plonk_inat_model_id: str = "nicolas-dufour/PLONK_iNaturalist"
    plonk_device: Literal["auto", "cpu", "cuda"] = "cpu"
    plonk_sample_count: int = Field(default=32, ge=4, le=256)
    plonk_timeout_seconds: float = Field(default=90.0, gt=0, le=180)
    paddleocr_enabled: bool = True
    paddleocr_worker_enabled: bool = True
    paddleocr_worker_host: Literal["127.0.0.1"] = "127.0.0.1"
    paddleocr_worker_port: int = Field(default=8793, ge=1, le=65535)
    paddleocr_worker_python: Path = Field(
        default_factory=lambda: _default_project_root()
        / ".local"
        / "workers"
        / "paddleocr"
        / ".venv"
        / "Scripts"
        / "python.exe"
    )
    paddleocr_device: Literal["cpu"] = "cpu"
    paddleocr_timeout_seconds: float = Field(default=45.0, gt=0, le=60)
    paddleocr_fallback_to_rapidocr: bool = True
    ocr_provider_priority: Literal["paddleocr,rapidocr"] = "paddleocr,rapidocr"
    phase6b_worker_startup_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    gpu_max_heavy_concurrency: int = Field(default=1, ge=1, le=2)
    gpu_max_resident_models: int = Field(default=1, ge=1, le=2)

    openai_geo_enabled: bool = False
    openai_geo_model: Literal["gpt-5.6-luna"] = "gpt-5.6-luna"
    openai_geo_reasoning_effort: Literal["none", "low"] = "none"
    openai_geo_image_detail: Literal["low"] = "low"
    openai_geo_max_output_tokens: int = Field(default=500, ge=1, le=500)
    openai_geo_monthly_budget_usd: float = Field(default=4.50, gt=0, le=100)
    openai_geo_daily_call_limit: int = Field(default=10, ge=1, le=100)
    openai_geo_per_analysis_call_limit: Literal[1] = 1
    openai_geo_allow_high_detail_retry: Literal[False] = False
    openai_geo_cache_ttl_days: int = Field(default=30, ge=1, le=365)
    openai_geo_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    openai_geo_maximum_image_edge: int = Field(default=768, ge=256, le=768)
    openai_geo_ledger_path: Path = Field(
        default_factory=lambda: _default_private_data_root() / "openai-geo-usage.sqlite3"
    )

    # Phase 6C is an explicit additive activation. Model and index setup remain
    # operator actions; application startup never downloads either.
    phase6c_enabled: bool = False
    phase6c_pipeline_version: Literal["phase6c-v1"] = "phase6c-v1"
    phase6c_fusion_config: Path = Field(
        default_factory=lambda: _default_project_root() / "config" / "reranking" / "phase6c-v1.json"
    )
    phase6c_ocr_config: Path = Field(
        default_factory=lambda: _default_project_root() / "config" / "ocr" / "phase6c-v1.json"
    )
    geoclip_hierarchical_enabled: bool = True
    geoclip_global_grid_enabled: bool = True
    geoclip_turkiye_refinement_enabled: bool = True
    geoclip_grid_points: int = Field(default=2_048, ge=256, le=16_384)
    geoclip_grid_max_candidates: int = Field(default=32, ge=5, le=64)
    geoclip_grid_refinement_levels_km: str = "900,300,90"
    megaloc_enabled: bool = True
    megaloc_worker_enabled: bool = True
    megaloc_worker_host: Literal["127.0.0.1"] = "127.0.0.1"
    megaloc_worker_port: int = Field(default=8794, ge=1, le=65535)
    megaloc_worker_python: Path = Field(
        default_factory=lambda: _default_project_root()
        / ".local"
        / "workers"
        / "megaloc"
        / ".venv"
        / "Scripts"
        / "python.exe"
    )
    megaloc_model_root: Path = Field(
        default_factory=lambda: _default_project_root()
        / ".local"
        / "models"
        / "phase6c"
        / "megaloc"
    )
    megaloc_device: Literal["auto", "cpu", "cuda"] = "auto"
    megaloc_timeout_seconds: float = Field(default=60.0, gt=0, le=180)
    megaloc_top_k: int = Field(default=50, ge=1, le=200)
    # Phase 3B2 production registration is fail-closed until an operator supplies
    # an explicit offline adapter configuration and its approval gates pass.
    megaloc_phase3b2_config_path: str = ""
    reference_index_enabled: bool = True
    reference_index_path: Path = Field(
        default_factory=lambda: _default_project_root()
        / ".local"
        / "indexes"
        / "turkiye-megaloc"
    )
    reference_index_max_images: int = Field(default=5_000, ge=1, le=100_000)
    reference_index_max_disk_gb: float = Field(default=2.0, gt=0, le=100)
    reference_index_max_per_province: int = Field(default=100, ge=1, le=10_000)
    reference_index_max_per_sequence: int = Field(default=10, ge=1, le=1_000)
    # Phase 3B1 reference-corpus indexes are built offline and remain isolated
    # from the existing Phase 6C/MegaLoc path. Activation is always explicit.
    turkiye_reference_index_enabled: bool = False
    turkiye_reference_index_path: Path = Field(
        default_factory=lambda: _default_project_root()
        / ".local"
        / "indexes"
        / "turkiye-corpus-v1"
    )
    turkiye_reference_source_policy_path: Path = Field(
        default_factory=lambda: _default_project_root()
        / "config"
        / "corpus"
        / "source-policy-v1.json"
    )
    turkiye_reference_descriptor_id: str = ""
    turkiye_reference_descriptor_version: str = ""
    turkiye_reference_descriptor_dimension: int = Field(default=0, ge=0, le=65_536)
    turkiye_reference_descriptor_artifact_sha256: str = ""
    turkiye_reference_descriptor_preprocessing_version: str = ""
    # Phase 3B3 is a private localhost-only presentation seam. It remains
    # disabled unless the operator explicitly selects a checksum-bound bundle.
    atlaslens_turkiye_demo_enabled: bool = False
    atlaslens_turkiye_demo_bundle_path: Path = Path(
        r"C:\AtlasLensPilot\mapillary-demo\published\demo-bundle"
    )
    atlaslens_turkiye_demo_expected_publication_sha256: str = ""
    atlaslens_turkiye_demo_expected_source_policy_sha256: str = ""
    atlaslens_turkiye_demo_expected_selection_lock_sha256: str = ""
    atlaslens_turkiye_demo_top_k: int = Field(default=5, ge=1, le=10)
    atlaslens_turkiye_demo_uncertainty_radius_m: float = Field(
        default=1_000.0, gt=0.0, le=100_000.0
    )
    mapillary_enabled: bool = False
    mapillary_access_token: SecretStr | None = None
    mapillary_max_images: int = Field(default=2_500, ge=1, le=50_000)
    kartaview_enabled: bool = True
    kartaview_max_images: int = Field(default=2_500, ge=1, le=50_000)
    g3_enabled: bool = False
    g3_device: Literal["auto", "cpu", "cuda"] = "auto"
    g3_max_candidates: int = Field(default=96, ge=1, le=256)
    g3_timeout_seconds: float = Field(default=60.0, gt=0, le=180)
    geolocation_improvement_max_iterations: int = Field(default=8, ge=1, le=20)

    phase5b_enabled: bool = True
    retrieval_enabled: bool = True
    retrieval_index_dir: Path = Field(
        default_factory=lambda: _default_private_data_root()
        / "reference-indexes"
        / "phase5b-commons"
    )
    retrieval_embedding_provider: Literal["disabled", "siglip2", "siglip2-b16-384", "clip"] = (
        "siglip2-b16-384"
    )
    retrieval_device: Literal["auto", "cpu", "cuda"] = "auto"
    retrieval_top_k: int = Field(default=25, ge=1, le=100)
    retrieval_timeout_seconds: float = Field(default=20.0, gt=0, le=120)

    map_evidence_enabled: bool = False
    overpass_endpoint: str = "https://overpass-api.de/api/interpreter"
    overpass_user_agent: str = "AtlasLens/0.1 map-constraints (operator-contact-required)"
    overpass_timeout_seconds: float = Field(default=5.0, gt=0, le=30)

    analysis_rate_limit_per_minute: int = Field(default=12, ge=1)
    cloud_rate_limit_per_minute: int = Field(default=3, ge=1)
    max_queued_jobs: int = Field(default=8, ge=1, le=128)
    max_sse_connections_per_client: int = Field(default=4, ge=1, le=32)
    sse_heartbeat_seconds: float = Field(default=15.0, gt=0, le=60)
    max_sse_lifetime_seconds: float = Field(default=600.0, gt=0, le=3600)
    log_level: str = "INFO"

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, value: str) -> str:
        origins = [origin.strip() for origin in value.split(",") if origin.strip()]
        if not origins or "*" in origins:
            raise ValueError("ALLOWED_ORIGINS must contain explicit origins")
        if any(not origin.startswith(("http://", "https://")) for origin in origins):
            raise ValueError("ALLOWED_ORIGINS entries must be HTTP(S) origins")
        return ",".join(origins)

    @field_validator("mock_scenario")
    @classmethod
    def validate_mock_scenario(cls, value: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
        if not value or any(character not in allowed for character in value):
            raise ValueError("MOCK_SCENARIO must be a safe fixture identifier")
        return value

    @field_validator("mock_inference_fixture")
    @classmethod
    def validate_mock_fixture_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        fixture = Path(value)
        if fixture.is_absolute() or fixture.suffix.lower() != ".json" or ".." in fixture.parts:
            raise ValueError("MOCK_INFERENCE_FIXTURE must be a contained JSON filename")
        return value

    @field_validator("geoclip_grid_refinement_levels_km")
    @classmethod
    def validate_geoclip_refinement_levels(cls, value: str) -> str:
        try:
            levels = tuple(float(item.strip()) for item in value.split(","))
        except ValueError as exc:
            raise ValueError("GeoCLIP refinement levels must be comma-separated numbers") from exc
        if not 1 <= len(levels) <= 4 or any(level <= 0 or level > 5_000 for level in levels):
            raise ValueError("GeoCLIP refinement levels are outside the bounded range")
        if tuple(sorted(levels, reverse=True)) != levels:
            raise ValueError("GeoCLIP refinement levels must be coarse-to-fine")
        return ",".join(f"{level:g}" for level in levels)

    @model_validator(mode="after")
    def forbid_production_mock(self) -> Settings:
        if self.app_env == "production" and self.enable_mock_inference:
            raise ValueError("mock inference is forbidden in production")
        if self.app_env == "production" and self.operator_api_enabled:
            raise ValueError("unauthenticated operator API is forbidden in production")
        if self.atlaslens_turkiye_demo_enabled:
            if self.app_env == "production":
                raise ValueError("private Mapillary demo is forbidden in production")
            if self.api_host not in {"127.0.0.1", "localhost"}:
                raise ValueError("private Mapillary demo must bind to loopback")
            if self.phase6c_enabled:
                raise ValueError("private Mapillary demo must remain isolated from Phase 6C")
        leaked_frontend_keys = [
            name
            for name, value in os.environ.items()
            if name.upper().startswith("VITE_")
            and any(provider in name.upper() for provider in ("OPENAI", "NVIDIA"))
            and any(
                marker in name.upper()
                for marker in ("KEY", "SECRET", "TOKEN", "CREDENTIAL")
            )
            and value.strip()
        ]
        if leaked_frontend_keys:
            raise ValueError("Cloud API keys are forbidden in frontend-prefixed variables")
        return self

    @property
    def mock_fixture_name(self) -> str:
        return self.mock_inference_fixture or f"{self.mock_scenario}.json"

    @property
    def cors_origins(self) -> list[str]:
        return self.allowed_origins.split(",")

    @property
    def openai_key_value(self) -> str | None:
        if self.openai_api_key is None:
            return None
        value = self.openai_api_key.get_secret_value().strip()
        return value or None

    @property
    def nvidia_key_value(self) -> str | None:
        if self.nvidia_api_key is None:
            return None
        value = self.nvidia_api_key.get_secret_value().strip()
        return value or None

    @property
    def mapillary_token_value(self) -> str | None:
        if self.mapillary_access_token is None:
            return None
        value = self.mapillary_access_token.get_secret_value().strip()
        return value or None

    @property
    def geoclip_refinement_levels_km(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.geoclip_grid_refinement_levels_km.split(","))
