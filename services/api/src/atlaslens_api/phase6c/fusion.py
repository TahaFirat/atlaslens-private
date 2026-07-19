from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.reranking.geo import geodesic_km, percentile, spherical_center

type Phase6CEvidenceKind = Literal[
    "geoclip_original",
    "geoclip_hierarchical",
    "osv_direct_regression",
    "plonk_samples",
    "megaloc_retrieval",
    "g3_verification",
    "ocr_place_match",
    "openai_review",
]
type Phase6CSourceFamily = Literal[
    "geoclip_mp16_family",
    "osv5m_family",
    "plonk_osv_family",
    "plonk_yfcc_family",
    "plonk_inat_family",
    "megaloc_retrieval_family",
    "g3_family",
    "ocr_text_family",
    "openai_review_family",
]
type Phase6CEvidenceStatus = Literal[
    "completed", "abstained", "failed", "timeout", "skipped", "disabled"
]
type Phase6CScoreSemantics = Literal[
    "raw_cosine_similarity_not_confidence",
    "direct_regression_no_comparable_score",
    "sample_density_not_confidence",
    "descriptor_similarity_not_confidence",
    "verification_similarity_not_confidence",
    "place_match_strength_not_confidence",
    "bounded_review_rank_not_confidence",
]

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,159}$")
_ALL_EVIDENCE_KINDS: tuple[Phase6CEvidenceKind, ...] = (
    "geoclip_original",
    "geoclip_hierarchical",
    "osv_direct_regression",
    "plonk_samples",
    "megaloc_retrieval",
    "g3_verification",
    "ocr_place_match",
    "openai_review",
)
_EXPECTED_CORRELATION_GROUPS: Mapping[Phase6CSourceFamily, str] = {
    "geoclip_mp16_family": "geoclip_mp16_training",
    "osv5m_family": "osv5m_training",
    "plonk_osv_family": "osv5m_training",
    "plonk_yfcc_family": "plonk_yfcc_training",
    "plonk_inat_family": "plonk_inat_training",
    "megaloc_retrieval_family": "megaloc_descriptor_retrieval",
    # G3 reuses GeoCLIP's MP-16 location representation. Keep the provider family
    # distinct for provenance, but never count it as an independent training signal.
    "g3_family": "geoclip_mp16_training",
    "ocr_text_family": "ocr_text",
    "openai_review_family": "openai_review",
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _normalize_longitude(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("longitude must be finite")
    return ((value + 180.0) % 360.0) - 180.0


class Phase6CEvidenceCandidate(_FrozenModel):
    """One provider-owned hypothesis with its raw semantics kept intact."""

    candidate_id: str = Field(min_length=1, max_length=128, pattern=_SAFE_ID.pattern)
    latitude: float = Field(ge=-90, le=90)
    longitude: float
    raw_value: float | None = None
    provider_rank: int = Field(ge=1, le=256)
    sample_support: int = Field(default=1, ge=1, le=100_000)
    uncertainty_radius_km: float = Field(gt=0, le=20_050)
    provenance: str = Field(min_length=1, max_length=240)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    mode_id: str | None = Field(default=None, min_length=1, max_length=160)
    retrieval_cluster_id: str | None = Field(default=None, min_length=1, max_length=160)
    verification_target_ids: tuple[str, ...] = Field(default=(), max_length=32)
    confidence: None = None
    calibrated: Literal[False] = False
    confidence_semantics: Literal["uncalibrated_provider_value_not_probability"] = (
        "uncalibrated_provider_value_not_probability"
    )

    @field_validator("latitude", "raw_value", "uncertainty_radius_km")
    @classmethod
    def finite_numbers(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("candidate numbers must be finite")
        return value

    @field_validator("longitude", mode="before")
    @classmethod
    def canonical_longitude(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("longitude must be numeric")
        return _normalize_longitude(float(value))

    @field_validator("mode_id", "retrieval_cluster_id")
    @classmethod
    def safe_optional_ids(cls, value: str | None) -> str | None:
        if value is not None and not _SAFE_ID.fullmatch(value):
            raise ValueError("candidate mode identifiers must be safe")
        return value

    @field_validator("verification_target_ids")
    @classmethod
    def safe_target_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not _SAFE_ID.fullmatch(item) for item in value):
            raise ValueError("verification target identifiers must be unique and safe")
        return value


class _EvidenceBatch(_FrozenModel):
    provider: str = Field(min_length=1, max_length=80, pattern=_SAFE_ID.pattern)
    model_id: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    status: Phase6CEvidenceStatus = "completed"
    duration_ms: int = Field(default=0, ge=0)
    candidates: tuple[Phase6CEvidenceCandidate, ...] = Field(default=(), max_length=256)
    reason_code: str | None = Field(
        default=None, min_length=1, max_length=120, pattern=_SAFE_ID.pattern
    )

    @model_validator(mode="after")
    def coherent_outcome(self) -> _EvidenceBatch:
        if self.status == "completed":
            if not self.candidates or self.reason_code is not None:
                raise ValueError("completed evidence requires candidates and no failure reason")
            ranks = [candidate.provider_rank for candidate in self.candidates]
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError("provider ranks must be contiguous")
            if len({candidate.candidate_id for candidate in self.candidates}) != len(
                self.candidates
            ):
                raise ValueError("candidate identifiers must be unique within evidence")
        elif self.candidates or self.reason_code is None:
            raise ValueError("non-completed evidence requires a reason and no candidates")
        return self

    def _require_raw_values(self) -> None:
        if self.status == "completed" and any(
            candidate.raw_value is None for candidate in self.candidates
        ):
            raise ValueError("this evidence type requires a finite raw provider value")


class GeoCLIPOriginalEvidence(_EvidenceBatch):
    evidence_kind: Literal["geoclip_original"] = "geoclip_original"
    source_family: Literal["geoclip_mp16_family"] = "geoclip_mp16_family"
    correlation_group: Literal["geoclip_mp16_training"] = "geoclip_mp16_training"
    score_semantics: Literal["raw_cosine_similarity_not_confidence"] = (
        "raw_cosine_similarity_not_confidence"
    )

    @model_validator(mode="after")
    def raw_scores_present(self) -> GeoCLIPOriginalEvidence:
        self._require_raw_values()
        return self


class GeoCLIPHierarchicalEvidence(_EvidenceBatch):
    evidence_kind: Literal["geoclip_hierarchical"] = "geoclip_hierarchical"
    source_family: Literal["geoclip_mp16_family"] = "geoclip_mp16_family"
    correlation_group: Literal["geoclip_mp16_training"] = "geoclip_mp16_training"
    score_semantics: Literal["raw_cosine_similarity_not_confidence"] = (
        "raw_cosine_similarity_not_confidence"
    )

    @model_validator(mode="after")
    def modes_and_raw_scores_present(self) -> GeoCLIPHierarchicalEvidence:
        self._require_raw_values()
        if self.status == "completed" and any(
            candidate.mode_id is None for candidate in self.candidates
        ):
            raise ValueError("hierarchical evidence requires a search mode identifier")
        return self


class OSVDirectRegressionEvidence(_EvidenceBatch):
    evidence_kind: Literal["osv_direct_regression"] = "osv_direct_regression"
    source_family: Literal["osv5m_family"] = "osv5m_family"
    correlation_group: Literal["osv5m_training"] = "osv5m_training"
    score_semantics: Literal["direct_regression_no_comparable_score"] = (
        "direct_regression_no_comparable_score"
    )

    @model_validator(mode="after")
    def one_scoreless_regression(self) -> OSVDirectRegressionEvidence:
        if self.status == "completed" and (
            len(self.candidates) != 1 or self.candidates[0].raw_value is not None
        ):
            raise ValueError("OSV direct regression requires one scoreless coordinate")
        return self


class PlonkOSVSampleEvidence(_EvidenceBatch):
    evidence_kind: Literal["plonk_samples"] = "plonk_samples"
    variant: Literal["osv"] = "osv"
    source_family: Literal["plonk_osv_family"] = "plonk_osv_family"
    correlation_group: Literal["osv5m_training"] = "osv5m_training"
    score_semantics: Literal["sample_density_not_confidence"] = (
        "sample_density_not_confidence"
    )

    @model_validator(mode="after")
    def raw_scores_present(self) -> PlonkOSVSampleEvidence:
        self._require_raw_values()
        return self


class PlonkYFCCSampleEvidence(_EvidenceBatch):
    evidence_kind: Literal["plonk_samples"] = "plonk_samples"
    variant: Literal["yfcc"] = "yfcc"
    source_family: Literal["plonk_yfcc_family"] = "plonk_yfcc_family"
    correlation_group: Literal["plonk_yfcc_training"] = "plonk_yfcc_training"
    score_semantics: Literal["sample_density_not_confidence"] = (
        "sample_density_not_confidence"
    )

    @model_validator(mode="after")
    def raw_scores_present(self) -> PlonkYFCCSampleEvidence:
        self._require_raw_values()
        return self


class PlonkINatSampleEvidence(_EvidenceBatch):
    evidence_kind: Literal["plonk_samples"] = "plonk_samples"
    variant: Literal["inat"] = "inat"
    source_family: Literal["plonk_inat_family"] = "plonk_inat_family"
    correlation_group: Literal["plonk_inat_training"] = "plonk_inat_training"
    score_semantics: Literal["sample_density_not_confidence"] = (
        "sample_density_not_confidence"
    )

    @model_validator(mode="after")
    def raw_scores_present(self) -> PlonkINatSampleEvidence:
        self._require_raw_values()
        return self


class MegaLocRetrievalEvidence(_EvidenceBatch):
    evidence_kind: Literal["megaloc_retrieval"] = "megaloc_retrieval"
    source_family: Literal["megaloc_retrieval_family"] = "megaloc_retrieval_family"
    correlation_group: Literal["megaloc_descriptor_retrieval"] = (
        "megaloc_descriptor_retrieval"
    )
    score_semantics: Literal["descriptor_similarity_not_confidence"] = (
        "descriptor_similarity_not_confidence"
    )

    @model_validator(mode="after")
    def clusters_and_raw_scores_present(self) -> MegaLocRetrievalEvidence:
        self._require_raw_values()
        if self.status == "completed" and any(
            candidate.retrieval_cluster_id is None for candidate in self.candidates
        ):
            raise ValueError("MegaLoc evidence requires retrieval cluster identifiers")
        return self


class G3VerificationEvidence(_EvidenceBatch):
    evidence_kind: Literal["g3_verification"] = "g3_verification"
    source_family: Literal["g3_family"] = "g3_family"
    correlation_group: Literal["geoclip_mp16_training"] = "geoclip_mp16_training"
    score_semantics: Literal["verification_similarity_not_confidence"] = (
        "verification_similarity_not_confidence"
    )

    @model_validator(mode="after")
    def targets_and_raw_scores_present(self) -> G3VerificationEvidence:
        self._require_raw_values()
        if self.status == "completed" and any(
            not candidate.verification_target_ids for candidate in self.candidates
        ):
            raise ValueError("G3 verification must name existing candidate identifiers")
        return self


class OCRPlaceMatchEvidence(_EvidenceBatch):
    evidence_kind: Literal["ocr_place_match"] = "ocr_place_match"
    source_family: Literal["ocr_text_family"] = "ocr_text_family"
    correlation_group: Literal["ocr_text"] = "ocr_text"
    score_semantics: Literal["place_match_strength_not_confidence"] = (
        "place_match_strength_not_confidence"
    )

    @model_validator(mode="after")
    def raw_scores_present(self) -> OCRPlaceMatchEvidence:
        self._require_raw_values()
        return self


class OpenAIReviewEvidence(_EvidenceBatch):
    evidence_kind: Literal["openai_review"] = "openai_review"
    source_family: Literal["openai_review_family"] = "openai_review_family"
    correlation_group: Literal["openai_review"] = "openai_review"
    score_semantics: Literal["bounded_review_rank_not_confidence"] = (
        "bounded_review_rank_not_confidence"
    )

    @model_validator(mode="after")
    def raw_scores_present(self) -> OpenAIReviewEvidence:
        self._require_raw_values()
        return self


type Phase6CEvidence = (
    GeoCLIPOriginalEvidence
    | GeoCLIPHierarchicalEvidence
    | OSVDirectRegressionEvidence
    | PlonkOSVSampleEvidence
    | PlonkYFCCSampleEvidence
    | PlonkINatSampleEvidence
    | MegaLocRetrievalEvidence
    | G3VerificationEvidence
    | OCRPlaceMatchEvidence
    | OpenAIReviewEvidence
)


class Phase6CFusionWeights(_FrozenModel):
    within_provider_rank: float = Field(default=0.42, ge=0, le=1)
    sample_density: float = Field(default=0.08, ge=0, le=1)
    provider_diversity: float = Field(default=0.08, ge=0, le=1)
    independent_family_corroboration: float = Field(default=0.20, ge=0, le=1)
    correlated_source_support: float = Field(default=0.04, ge=0, le=0.2)
    geographic_spread_penalty: float = Field(default=0.12, ge=0, le=1)


def _default_ablation_profiles() -> dict[str, tuple[Phase6CEvidenceKind, ...]]:
    local_without_ocr: tuple[Phase6CEvidenceKind, ...] = (
        "geoclip_original",
        "geoclip_hierarchical",
        "osv_direct_regression",
        "plonk_samples",
        "megaloc_retrieval",
    )
    return {
        "geoclip_original_only": ("geoclip_original",),
        "geoclip_original_hierarchical": (
            "geoclip_original",
            "geoclip_hierarchical",
        ),
        "geoclip_osv5m": ("geoclip_original", "osv_direct_regression"),
        "geoclip_plonk": ("geoclip_original", "plonk_samples"),
        "geoclip_megaloc": ("geoclip_original", "megaloc_retrieval"),
        "all_local_without_ocr": local_without_ocr,
        "all_local_with_ocr": (*local_without_ocr, "ocr_place_match"),
        "all_local_with_g3": (*local_without_ocr, "ocr_place_match", "g3_verification"),
        "optional_openai_assist": _ALL_EVIDENCE_KINDS,
        "phase6c_final": _ALL_EVIDENCE_KINDS,
    }


class Phase6CAblationSelection(_FrozenModel):
    profile_id: str = Field(default="phase6c_final", pattern=_SAFE_ID.pattern)
    allowed_evidence_kinds: tuple[Phase6CEvidenceKind, ...] | None = None
    excluded_source_families: tuple[Phase6CSourceFamily, ...] = ()

    @model_validator(mode="after")
    def unique_filters(self) -> Phase6CAblationSelection:
        if self.allowed_evidence_kinds is not None:
            if not self.allowed_evidence_kinds:
                raise ValueError("allowed evidence kinds cannot be empty")
            if len(set(self.allowed_evidence_kinds)) != len(self.allowed_evidence_kinds):
                raise ValueError("allowed evidence kinds must be unique")
        if len(set(self.excluded_source_families)) != len(self.excluded_source_families):
            raise ValueError("excluded source families must be unique")
        return self


class Phase6CFusionConfig(_FrozenModel):
    version: Literal["phase6c-v1"] = "phase6c-v1"
    cluster_radius_km: float = Field(default=75.0, gt=0, le=500)
    minimum_uncertainty_radius_km: float = Field(default=5.0, gt=0, le=500)
    diversity_radius_km: float = Field(default=300.0, gt=0, le=5_000)
    maximum_candidates_per_diversity_region: int = Field(default=3, ge=1, le=10)
    max_candidates: int = Field(default=16, ge=1, le=50)
    maximum_provider_diversity: int = Field(default=8, ge=2, le=32)
    maximum_independent_families: int = Field(default=6, ge=2, le=12)
    minimum_independent_families_for_publication: int = Field(default=2, ge=2, le=6)
    weights: Phase6CFusionWeights = Field(default_factory=Phase6CFusionWeights)
    source_correlation_groups: dict[Phase6CSourceFamily, str] = Field(
        default_factory=lambda: dict(_EXPECTED_CORRELATION_GROUPS), max_length=16
    )
    ablation_profiles: dict[str, tuple[Phase6CEvidenceKind, ...]] = Field(
        default_factory=_default_ablation_profiles, max_length=24
    )
    external_baseline_profiles: tuple[str, ...] = ("phase6b_baseline",)

    @field_validator("source_correlation_groups")
    @classmethod
    def fixed_correlation_safety_map(
        cls, value: dict[Phase6CSourceFamily, str]
    ) -> dict[Phase6CSourceFamily, str]:
        if value != dict(_EXPECTED_CORRELATION_GROUPS):
            raise ValueError("Phase 6C source correlation groups are a fixed safety boundary")
        return value

    @field_validator("ablation_profiles")
    @classmethod
    def valid_ablation_profiles(
        cls, value: dict[str, tuple[Phase6CEvidenceKind, ...]]
    ) -> dict[str, tuple[Phase6CEvidenceKind, ...]]:
        if "phase6c_final" not in value or set(value["phase6c_final"]) != set(
            _ALL_EVIDENCE_KINDS
        ):
            raise ValueError("phase6c_final must retain every Phase 6C evidence kind")
        for name, kinds in value.items():
            if not _SAFE_ID.fullmatch(name) or not kinds or len(set(kinds)) != len(kinds):
                raise ValueError(
                    "ablation profile names and evidence kinds must be safe and unique"
                )
        return value

    @model_validator(mode="after")
    def coherent_limits(self) -> Phase6CFusionConfig:
        if (
            self.minimum_independent_families_for_publication
            > self.maximum_independent_families
        ):
            raise ValueError("publication family threshold exceeds the scoring bound")
        if set(self.ablation_profiles).intersection(self.external_baseline_profiles):
            raise ValueError("external baselines cannot masquerade as Phase 6C ablations")
        return self

    def ablation(self, profile_id: str) -> Phase6CAblationSelection:
        if profile_id in self.external_baseline_profiles:
            raise ValueError("external baselines must run their original fusion policy")
        try:
            kinds = self.ablation_profiles[profile_id]
        except KeyError as exc:
            raise ValueError("unknown Phase 6C ablation profile") from exc
        return Phase6CAblationSelection(
            profile_id=profile_id,
            allowed_evidence_kinds=kinds,
        )


def load_phase6c_fusion_config(path: Path) -> Phase6CFusionConfig:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("Phase 6C fusion config is missing or unsafe")
    return Phase6CFusionConfig.model_validate_json(path.read_text(encoding="utf-8"))


class Phase6CFusionMember(_FrozenModel):
    evidence_kind: Phase6CEvidenceKind
    provider: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    candidate_id: str = Field(min_length=1, max_length=128)
    source_family: Phase6CSourceFamily
    correlation_group: str = Field(min_length=1, max_length=120, pattern=_SAFE_ID.pattern)
    score_semantics: Phase6CScoreSemantics
    raw_value: float | None
    provider_rank: int = Field(ge=1, le=256)
    sample_support: int = Field(ge=1, le=100_000)
    provenance: str = Field(min_length=1, max_length=240)


class Phase6CContribution(_FrozenModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    source: str = Field(min_length=1, max_length=120, pattern=_SAFE_ID.pattern)
    raw_value: float
    normalized_value: float = Field(ge=0, le=1)
    weight: float
    contribution: float
    reason: str = Field(min_length=1, max_length=300)
    independent: bool
    correlation_group: str = Field(min_length=1, max_length=120, pattern=_SAFE_ID.pattern)

    @field_validator("raw_value", "normalized_value", "weight", "contribution")
    @classmethod
    def finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fusion contribution values must be finite")
        return value


class Phase6CFusedCandidate(_FrozenModel):
    cluster_id: str = Field(pattern=r"^phase6c-[a-f0-9]{24}$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_km: float = Field(gt=0, le=20_050)
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
    pre_diversity_rank: int = Field(ge=1, le=10_000)
    final_rank: int = Field(ge=1, le=50)
    rank_movement: int = Field(ge=-10_000, le=10_000)
    movement_reasons: tuple[str, ...] = Field(min_length=1, max_length=4)
    members: tuple[Phase6CFusionMember, ...] = Field(min_length=1, max_length=256)
    contributions: tuple[Phase6CContribution, ...] = Field(min_length=1, max_length=16)


class Phase6CProviderFailure(_FrozenModel):
    evidence_kind: Phase6CEvidenceKind
    provider: str = Field(min_length=1, max_length=80)
    status: Literal["abstained", "failed", "timeout", "skipped", "disabled"]
    reason_code: str = Field(min_length=1, max_length=120, pattern=_SAFE_ID.pattern)


class Phase6CAblationMetadata(_FrozenModel):
    profile_id: str = Field(pattern=_SAFE_ID.pattern)
    input_evidence_count: int = Field(ge=0, le=1_000)
    included_evidence_count: int = Field(ge=0, le=1_000)
    excluded_evidence_count: int = Field(ge=0, le=1_000)
    included_evidence_kinds: tuple[Phase6CEvidenceKind, ...]
    excluded_evidence_kinds: tuple[Phase6CEvidenceKind, ...]
    excluded_source_families: tuple[Phase6CSourceFamily, ...]


class Phase6CFusionResult(_FrozenModel):
    version: Literal["phase6c-v1"] = "phase6c-v1"
    candidates: tuple[Phase6CFusedCandidate, ...] = Field(default=(), max_length=50)
    provider_failures: tuple[Phase6CProviderFailure, ...] = Field(default=(), max_length=64)
    abstained: bool
    abstention_reason: str | None = Field(
        default=None, min_length=1, max_length=120, pattern=_SAFE_ID.pattern
    )
    publication_candidate_count: int = Field(ge=0, le=50)
    input_cluster_count: int = Field(ge=0, le=10_000)
    suppressed_cluster_count: int = Field(ge=0, le=10_000)
    ablation: Phase6CAblationMetadata
    warnings: tuple[str, ...] = (
        "warning.confidence_not_calibrated",
        "warning.candidate_recall_precedes_publication",
    )

    @model_validator(mode="after")
    def coherent_result(self) -> Phase6CFusionResult:
        if self.abstained:
            if self.candidates or self.abstention_reason is None:
                raise ValueError("abstention requires no candidates and an explicit reason")
        elif not self.candidates or self.abstention_reason is not None:
            raise ValueError("non-abstained fusion requires candidates and no abstention reason")
        if self.publication_candidate_count != sum(
            candidate.publication_eligible for candidate in self.candidates
        ):
            raise ValueError("publication candidate count is inconsistent")
        return self


@dataclass(frozen=True, slots=True)
class _Member:
    evidence_key: str
    evidence_kind: Phase6CEvidenceKind
    provider: str
    model_id: str
    model_revision: str
    source_family: Phase6CSourceFamily
    correlation_group: str
    score_semantics: Phase6CScoreSemantics
    candidate: Phase6CEvidenceCandidate


@dataclass(frozen=True, slots=True)
class _ScoredCluster:
    cluster_id: str
    latitude: float
    longitude: float
    uncertainty_radius_km: float
    relative_rank_score: float
    source_family_count: int
    independent_source_family_count: int
    correlated_source_family_count: int
    provider_count: int
    spread_km: float
    publication_eligible: bool
    members: tuple[Phase6CFusionMember, ...]
    contributions: tuple[Phase6CContribution, ...]
    diversity_tags: frozenset[str]


class Phase6CGeographicFusionEngine:
    """Recall-first, diversity-preserving fusion without cross-provider score mixing."""

    def __init__(self, config: Phase6CFusionConfig | None = None) -> None:
        self._config = config or Phase6CFusionConfig()

    @property
    def ablation_profile_ids(self) -> tuple[str, ...]:
        """Return the bounded configured profile order without exposing mutable config."""

        return tuple(self._config.ablation_profiles)

    def fuse_profile(
        self,
        evidence: Sequence[Phase6CEvidence],
        profile_id: str,
    ) -> Phase6CFusionResult:
        return self.fuse(evidence, ablation=self._config.ablation(profile_id))

    def fuse(
        self,
        evidence: Sequence[Phase6CEvidence],
        *,
        ablation: Phase6CAblationSelection | None = None,
    ) -> Phase6CFusionResult:
        selection = ablation or self._config.ablation("phase6c_final")
        included, ablation_metadata = self._apply_ablation(evidence, selection)
        failures: list[Phase6CProviderFailure] = []
        members: list[_Member] = []
        maximum_support: dict[str, int] = {}
        for batch in included:
            if batch.status != "completed":
                failures.append(
                    Phase6CProviderFailure(
                        evidence_kind=batch.evidence_kind,
                        provider=batch.provider,
                        status=batch.status,
                        reason_code=batch.reason_code or "unknown_provider_failure",
                    )
                )
                continue
            evidence_key = f"{batch.evidence_kind}:{batch.provider}:{batch.model_id}"
            maximum_support[evidence_key] = max(
                candidate.sample_support for candidate in batch.candidates
            )
            for candidate in batch.candidates:
                members.append(
                    _Member(
                        evidence_key=evidence_key,
                        evidence_kind=batch.evidence_kind,
                        provider=batch.provider,
                        model_id=batch.model_id,
                        model_revision=batch.model_revision,
                        source_family=batch.source_family,
                        correlation_group=batch.correlation_group,
                        score_semantics=batch.score_semantics,
                        candidate=candidate,
                    )
                )
        if not members:
            reason = (
                "ablation_excluded_all_evidence"
                if evidence and not included
                else "no_completed_evidence"
            )
            return Phase6CFusionResult(
                candidates=(),
                provider_failures=tuple(failures),
                abstained=True,
                abstention_reason=reason,
                publication_candidate_count=0,
                input_cluster_count=0,
                suppressed_cluster_count=0,
                ablation=ablation_metadata,
            )
        clusters = self._cluster(members)
        scored = [
            self._score_cluster(cluster, maximum_support=maximum_support) for cluster in clusters
        ]
        scored.sort(
            key=lambda item: (
                -item.relative_rank_score,
                -item.independent_source_family_count,
                -item.provider_count,
                item.cluster_id,
            )
        )
        selected = self._select_diverse(scored)
        candidates: list[Phase6CFusedCandidate] = []
        for final_rank, (pre_rank, item, selection_reason) in enumerate(selected, start=1):
            movement = pre_rank - final_rank
            movement_reason = (
                "rank_promoted_by_diversity"
                if movement > 0
                else "rank_deferred_by_diversity"
                if movement < 0
                else "base_rank_retained"
            )
            candidates.append(
                Phase6CFusedCandidate(
                    cluster_id=item.cluster_id,
                    latitude=item.latitude,
                    longitude=item.longitude,
                    uncertainty_radius_km=item.uncertainty_radius_km,
                    relative_rank_score=item.relative_rank_score,
                    source_family_count=item.source_family_count,
                    independent_source_family_count=item.independent_source_family_count,
                    correlated_source_family_count=item.correlated_source_family_count,
                    provider_count=item.provider_count,
                    spread_km=item.spread_km,
                    publication_eligible=item.publication_eligible,
                    publication_basis=(
                        "at_least_two_independent_source_families"
                        if item.publication_eligible
                        else "candidate_recall_only_insufficient_independence"
                    ),
                    pre_diversity_rank=pre_rank,
                    final_rank=final_rank,
                    rank_movement=movement,
                    movement_reasons=(selection_reason, movement_reason),
                    members=item.members,
                    contributions=item.contributions,
                )
            )
        return Phase6CFusionResult(
            candidates=tuple(candidates),
            provider_failures=tuple(failures),
            abstained=False,
            abstention_reason=None,
            publication_candidate_count=sum(item.publication_eligible for item in candidates),
            input_cluster_count=len(scored),
            suppressed_cluster_count=len(scored) - len(candidates),
            ablation=ablation_metadata,
        )

    def _apply_ablation(
        self,
        evidence: Sequence[Phase6CEvidence],
        selection: Phase6CAblationSelection,
    ) -> tuple[tuple[Phase6CEvidence, ...], Phase6CAblationMetadata]:
        allowed = (
            set(selection.allowed_evidence_kinds)
            if selection.allowed_evidence_kinds is not None
            else set(_ALL_EVIDENCE_KINDS)
        )
        excluded_families = set(selection.excluded_source_families)
        included = tuple(
            batch
            for batch in evidence
            if batch.evidence_kind in allowed and batch.source_family not in excluded_families
        )
        included_ids = {id(batch) for batch in included}
        excluded = tuple(batch for batch in evidence if id(batch) not in included_ids)
        return included, Phase6CAblationMetadata(
            profile_id=selection.profile_id,
            input_evidence_count=len(evidence),
            included_evidence_count=len(included),
            excluded_evidence_count=len(excluded),
            included_evidence_kinds=tuple(sorted({item.evidence_kind for item in included})),
            excluded_evidence_kinds=tuple(sorted({item.evidence_kind for item in excluded})),
            excluded_source_families=selection.excluded_source_families,
        )

    def _cluster(self, members: Sequence[_Member]) -> tuple[tuple[_Member, ...], ...]:
        """Complete-link clustering avoids transitive chains swallowing distant modes."""

        ordered = sorted(
            members,
            key=lambda item: (
                item.candidate.latitude,
                item.candidate.longitude,
                item.provider,
                item.candidate.provider_rank,
                item.candidate.candidate_id,
            ),
        )
        clusters: list[list[_Member]] = []
        for member in ordered:
            point = (member.candidate.latitude, member.candidate.longitude)
            compatible: list[tuple[float, int]] = []
            for index, cluster in enumerate(clusters):
                distances = [
                    geodesic_km(
                        point,
                        (item.candidate.latitude, item.candidate.longitude),
                    )
                    for item in cluster
                ]
                maximum_distance = max(distances)
                if maximum_distance <= self._config.cluster_radius_km:
                    compatible.append((maximum_distance, index))
            if compatible:
                _, cluster_index = min(compatible)
                clusters[cluster_index].append(member)
            else:
                clusters.append([member])
        return tuple(
            tuple(
                sorted(
                    cluster,
                    key=lambda item: (
                        item.provider,
                        item.candidate.provider_rank,
                        item.candidate.candidate_id,
                    ),
                )
            )
            for cluster in clusters
        )

    def _score_cluster(
        self,
        members: tuple[_Member, ...],
        *,
        maximum_support: Mapping[str, int],
    ) -> _ScoredCluster:
        points = [(item.candidate.latitude, item.candidate.longitude) for item in members]
        center = spherical_center(points)
        distances = [geodesic_km(center, point) for point in points]
        spread = percentile(distances, 0.80)
        provider_keys = sorted({item.evidence_key for item in members})
        providers = sorted({item.provider for item in members})
        families = sorted({item.source_family for item in members})
        correlation_groups = sorted({item.correlation_group for item in members})
        correlated_families = max(0, len(families) - len(correlation_groups))

        best_rank: dict[str, int] = {}
        best_density: dict[str, float] = {}
        for item in members:
            rank = item.candidate.provider_rank
            best_rank[item.evidence_key] = min(rank, best_rank.get(item.evidence_key, rank))
            density = item.candidate.sample_support / maximum_support[item.evidence_key]
            best_density[item.evidence_key] = max(
                density, best_density.get(item.evidence_key, 0.0)
            )
        rank_raw = sum(1.0 / rank for rank in best_rank.values()) / len(best_rank)
        density_raw = sum(best_density.values()) / len(best_density)
        provider_raw = float(len(provider_keys))
        provider_normalized = min(
            1.0, provider_raw / self._config.maximum_provider_diversity
        )
        independent_raw = float(max(0, len(correlation_groups) - 1))
        independent_normalized = min(
            1.0,
            independent_raw / max(1, self._config.maximum_independent_families - 1),
        )
        correlated_raw = float(correlated_families)
        correlated_normalized = min(1.0, correlated_raw / 3.0)
        spread_normalized = min(1.0, spread / self._config.cluster_radius_km)
        weights = self._config.weights
        contributions = (
            self._contribution(
                name="within_provider_rank",
                source="fusion.within_provider_rank",
                raw=rank_raw,
                normalized=rank_raw,
                weight=weights.within_provider_rank,
                reason="Reciprocal ranks are normalized only within each originating provider.",
                independent=False,
                correlation_group="within_source_ranking",
            ),
            self._contribution(
                name="sample_density",
                source="fusion.sample_density",
                raw=density_raw,
                normalized=density_raw,
                weight=weights.sample_density,
                reason="Sample support is normalized only inside its originating evidence batch.",
                independent=False,
                correlation_group="within_source_support",
            ),
            self._contribution(
                name="provider_diversity",
                source="fusion.provider_diversity",
                raw=provider_raw,
                normalized=provider_normalized,
                weight=weights.provider_diversity,
                reason=f"{len(providers)} named providers contribute to this geographic cluster.",
                independent=False,
                correlation_group="provider_identity",
            ),
            self._contribution(
                name="independent_family_corroboration",
                source="fusion.independent_family_corroboration",
                raw=independent_raw,
                normalized=independent_normalized,
                weight=weights.independent_family_corroboration,
                reason=(
                    f"{len(correlation_groups)} independent correlation groups support nearby "
                    "coordinates; the first group is the candidate source, not corroboration."
                ),
                independent=True,
                correlation_group="independent_cross_group_support",
            ),
            self._contribution(
                name="correlated_source_support",
                source="fusion.correlated_source_support",
                raw=correlated_raw,
                normalized=correlated_normalized,
                weight=weights.correlated_source_support,
                reason=(
                    f"{correlated_families} additional source families share an existing "
                    "correlation group, so their contribution is separately bounded."
                ),
                independent=False,
                correlation_group="correlated_cross_source_support",
            ),
            self._contribution(
                name="geographic_spread_penalty",
                source="fusion.geographic_spread",
                raw=spread,
                normalized=spread_normalized,
                weight=-weights.geographic_spread_penalty,
                reason="The geodesic 80th-percentile member spread reduces relative rank.",
                independent=False,
                correlation_group="geographic_cluster_geometry",
            ),
        )
        score = min(1.0, max(0.0, sum(item.contribution for item in contributions)))
        stable_payload = "|".join(
            f"{item.evidence_key}:{item.candidate.candidate_id}" for item in members
        )
        stable = hashlib.sha256(stable_payload.encode()).hexdigest()[:24]
        uncertainty = max(
            self._config.minimum_uncertainty_radius_km,
            spread * 2.0,
            max(item.candidate.uncertainty_radius_km for item in members),
        )
        publication_eligible = (
            len(correlation_groups)
            >= self._config.minimum_independent_families_for_publication
            and len(families) >= self._config.minimum_independent_families_for_publication
        )
        diversity_tags: set[str] = {f"family:{family}" for family in families}
        for item in members:
            candidate = item.candidate
            if candidate.country_code is not None:
                diversity_tags.add(f"country:{candidate.country_code}")
            if candidate.mode_id is not None:
                diversity_tags.add(f"mode:{candidate.mode_id}")
            if candidate.retrieval_cluster_id is not None:
                diversity_tags.add(f"retrieval:{candidate.retrieval_cluster_id}")
        return _ScoredCluster(
            cluster_id=f"phase6c-{stable}",
            latitude=center[0],
            longitude=center[1],
            uncertainty_radius_km=uncertainty,
            relative_rank_score=score,
            source_family_count=len(families),
            independent_source_family_count=len(correlation_groups),
            correlated_source_family_count=correlated_families,
            provider_count=len(providers),
            spread_km=spread,
            publication_eligible=publication_eligible,
            members=tuple(
                Phase6CFusionMember(
                    evidence_kind=item.evidence_kind,
                    provider=item.provider,
                    model_id=item.model_id,
                    model_revision=item.model_revision,
                    candidate_id=item.candidate.candidate_id,
                    source_family=item.source_family,
                    correlation_group=item.correlation_group,
                    score_semantics=item.score_semantics,
                    raw_value=item.candidate.raw_value,
                    provider_rank=item.candidate.provider_rank,
                    sample_support=item.candidate.sample_support,
                    provenance=item.candidate.provenance,
                )
                for item in members
            ),
            contributions=contributions,
            diversity_tags=frozenset(diversity_tags),
        )

    def _select_diverse(
        self, scored: Sequence[_ScoredCluster]
    ) -> tuple[tuple[int, _ScoredCluster, str], ...]:
        if not scored:
            return ()
        remaining: list[tuple[int, _ScoredCluster]] = list(enumerate(scored, start=1))
        first_rank, first = remaining.pop(0)
        selected: list[tuple[int, _ScoredCluster, str]] = [
            (first_rank, first, "highest_base_rank_retained")
        ]
        seen_tags = set(first.diversity_tags)
        while remaining and len(selected) < self._config.max_candidates:
            eligible: list[
                tuple[
                    tuple[int, int, float, float, int, str],
                    int,
                    int,
                    _ScoredCluster,
                    str,
                ]
            ] = []
            for remaining_index, (pre_rank, item) in enumerate(remaining):
                distances = [
                    geodesic_km(
                        (item.latitude, item.longitude),
                        (selected_item.latitude, selected_item.longitude),
                    )
                    for _, selected_item, _ in selected
                ]
                nearby = sum(
                    distance < self._config.diversity_radius_km for distance in distances
                )
                if nearby >= self._config.maximum_candidates_per_diversity_region:
                    continue
                minimum_distance = min(distances)
                geographically_new = minimum_distance >= self._config.diversity_radius_km
                new_tag_count = len(item.diversity_tags - seen_tags)
                reason = (
                    "geographic_and_source_diversity_preserved"
                    if geographically_new and new_tag_count
                    else "geographic_diversity_preserved"
                    if geographically_new
                    else "source_or_mode_diversity_preserved"
                    if new_tag_count
                    else "bounded_region_backfill"
                )
                priority = (
                    int(geographically_new),
                    min(new_tag_count, 8),
                    minimum_distance,
                    item.relative_rank_score,
                    -pre_rank,
                    item.cluster_id,
                )
                eligible.append(
                    (priority, remaining_index, pre_rank, item, reason)
                )
            if not eligible:
                break
            _, remaining_index, pre_rank, item, reason = max(eligible)
            remaining.pop(remaining_index)
            selected.append((pre_rank, item, reason))
            seen_tags.update(item.diversity_tags)
        return tuple(selected)

    @staticmethod
    def _contribution(
        *,
        name: str,
        source: str,
        raw: float,
        normalized: float,
        weight: float,
        reason: str,
        independent: bool,
        correlation_group: str,
    ) -> Phase6CContribution:
        return Phase6CContribution(
            name=name,
            source=source,
            raw_value=raw,
            normalized_value=normalized,
            weight=weight,
            contribution=normalized * weight,
            reason=reason,
            independent=independent,
            correlation_group=correlation_group,
        )
