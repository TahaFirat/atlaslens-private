from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from atlaslens_api.providers.base import ExifResult, ProviderDescriptor, VisionClueResult
from atlaslens_api.schemas import (
    Abstention,
    Candidate,
    Evidence,
    GeoClipClusterSummary,
    GeoJsonPoint,
    GeoPoint,
    ModelPredictionDiagnostics,
    Phase4Assessment,
    Phase5BAssessment,
    Provenance,
    ReverseGeocodeSummary,
    UncalibratedConfidenceSummary,
)

FUSION_POLICY_VERSION = "phase1-fusion-v1"


class CandidateDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    center: GeoPoint
    radius_km: float = Field(gt=0)
    uncertainty_basis: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_kind: str
    confidence_basis: str
    granularity: str
    country_code: str | None = None
    label: str | None = None
    source: str
    evidence_ids: list[str] = Field(min_length=1)
    evidence_summary: str
    provenance: list[Provenance] = Field(min_length=1)
    verification_status: str
    verified: bool
    phase4_assessment: Phase4Assessment | None = None
    phase5b_assessment: Phase5BAssessment | None = None
    model_prediction: ModelPredictionDiagnostics | None = None
    geoclip_cluster: GeoClipClusterSummary | None = None
    confidence_assessment: UncalibratedConfidenceSummary | None = None
    reverse_geocode: ReverseGeocodeSummary | None = None


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    provider_id: str
    candidates: tuple[CandidateDraft, ...]


class CandidateProvider(Protocol):
    descriptor: ProviderDescriptor


def provenance_for(descriptor: ProviderDescriptor) -> Provenance:
    return Provenance(
        provider_id=descriptor.id,
        provider_kind=descriptor.kind,
        provider_version=descriptor.version,
        execution_boundary=descriptor.execution_boundary,
        model_name=descriptor.model_name,
        output_schema_version="phase1-provider-output-v1",
    )


class ExifCandidateProvider:
    descriptor = ProviderDescriptor(
        id="exif-candidate-synthesis",
        kind="candidate_synthesis",
        version="1.0.0",
        execution_boundary="local",
        criticality="optional",
        available=True,
    )

    def propose(
        self, result: ExifResult, evidence_id: str, exif_provenance: Provenance
    ) -> CandidateBatch:
        stable = hashlib.sha256(
            f"{result.latitude:.7f},{result.longitude:.7f}".encode()
        ).hexdigest()[:16]
        candidate = CandidateDraft(
            id=f"candidate-exif-{stable}",
            center=GeoPoint(latitude=result.latitude, longitude=result.longitude),
            radius_km=1.0,
            uncertainty_basis="phase1.conservative_exif_radius_floor",
            confidence=0.95,
            confidence_kind="source_reliability",
            confidence_basis="phase1.embedded_gps_source_support_not_scene_verification",
            granularity="exact_metadata",
            source=self.descriptor.id,
            evidence_ids=[evidence_id],
            evidence_summary="evidence.embedded_gps_metadata",
            provenance=[exif_provenance, provenance_for(self.descriptor)],
            verification_status="metadata_only",
            verified=False,
        )
        return CandidateBatch(provider_id=self.descriptor.id, candidates=(candidate,))


class VisionCandidateProvider:
    descriptor = ProviderDescriptor(
        id="vision-candidate-synthesis",
        kind="candidate_synthesis",
        version="1.0.0",
        execution_boundary="local",
        criticality="optional",
        available=True,
    )

    def propose(
        self,
        result: VisionClueResult,
        evidence_ids: list[str],
        vision_provenance: Provenance,
    ) -> CandidateBatch:
        candidates: list[CandidateDraft] = []
        for index, hypothesis in enumerate(result.hypotheses[:5]):
            stable = hashlib.sha256(
                (
                    f"{hypothesis.latitude:.5f},{hypothesis.longitude:.5f},"
                    f"{hypothesis.label.casefold()}"
                ).encode()
            ).hexdigest()[:16]
            candidates.append(
                CandidateDraft(
                    id=f"candidate-vision-{stable}",
                    center=GeoPoint(
                        latitude=hypothesis.latitude,
                        longitude=hypothesis.longitude,
                    ),
                    radius_km=max(hypothesis.radius_km, 25.0),
                    uncertainty_basis="phase1.unverified_visual_hypothesis_radius",
                    confidence=min(hypothesis.confidence, 0.40),
                    confidence_kind="uncalibrated_score",
                    confidence_basis="phase1.capped_unverified_visual_source_support",
                    granularity=hypothesis.granularity,
                    country_code=hypothesis.country_code,
                    label=hypothesis.label,
                    source=self.descriptor.id,
                    evidence_ids=[evidence_ids[index]],
                    evidence_summary="evidence.visible_geographic_clues",
                    provenance=[vision_provenance, provenance_for(self.descriptor)],
                    verification_status="unverified_model",
                    verified=False,
                )
            )
        return CandidateBatch(provider_id=self.descriptor.id, candidates=tuple(candidates))


@dataclass(frozen=True, slots=True)
class FusionResult:
    candidates: list[Candidate]
    abstention: Abstention | None


def haversine_km(left: GeoPoint, right: GeoPoint) -> float:
    radius = 6371.0088
    lat1 = math.radians(left.latitude)
    lat2 = math.radians(right.latitude)
    delta_latitude = lat2 - lat1
    delta_longitude = math.radians(((right.longitude - left.longitude + 180.0) % 360.0) - 180.0)
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(haversine)))


class DeterministicCandidateFusionService:
    policy_version = FUSION_POLICY_VERSION

    def fuse(self, evidence: list[Evidence], batches: list[CandidateBatch]) -> FusionResult:
        evidence_ids = {item.id for item in evidence}
        valid: list[CandidateDraft] = []
        for batch in batches:
            for draft in batch.candidates:
                if not set(draft.evidence_ids) <= evidence_ids:
                    continue
                try:
                    self._to_candidate(draft, rank=1)
                except ValueError:
                    continue
                valid.append(draft)

        valid.sort(key=self._sort_key)
        deduplicated: list[CandidateDraft] = []
        for draft in valid:
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(deduplicated)
                    if self._comparable_duplicate(existing, draft)
                ),
                None,
            )
            if duplicate_index is None:
                deduplicated.append(draft)
            else:
                deduplicated[duplicate_index] = self._merge(deduplicated[duplicate_index], draft)

        candidates = [
            self._to_candidate(draft, rank=index)
            for index, draft in enumerate(deduplicated, start=1)
        ]
        if candidates:
            return FusionResult(candidates=candidates, abstention=None)
        return FusionResult(
            candidates=[],
            abstention=Abstention(
                reason_code="insufficient_geographic_evidence",
                message_key="abstention.insufficient_geographic_evidence",
            ),
        )

    @staticmethod
    def _sort_key(draft: CandidateDraft) -> tuple[int, float, str]:
        priority = 0 if draft.verification_status == "metadata_only" else 1
        score = draft.confidence
        if score is None and draft.phase5b_assessment is not None:
            score = draft.phase5b_assessment.relative_rank_score
        if score is None and draft.model_prediction is not None:
            score = draft.model_prediction.raw_score
        return priority, -(score or 0.0), draft.id

    @staticmethod
    def _comparable_duplicate(left: CandidateDraft, right: CandidateDraft) -> bool:
        if left.geoclip_cluster is not None or right.geoclip_cluster is not None:
            return (
                left.geoclip_cluster is not None
                and right.geoclip_cluster is not None
                and left.geoclip_cluster.cluster_id == right.geoclip_cluster.cluster_id
            )
        if left.model_prediction is not None or right.model_prediction is not None:
            return False
        if (left.verification_status == "metadata_only") != (
            right.verification_status == "metadata_only"
        ):
            return False
        threshold = max(1.0, min(left.radius_km, right.radius_km) * 0.10)
        return haversine_km(left.center, right.center) <= threshold

    @staticmethod
    def _merge(preferred: CandidateDraft, duplicate: CandidateDraft) -> CandidateDraft:
        evidence_ids = list(dict.fromkeys([*preferred.evidence_ids, *duplicate.evidence_ids]))
        provenance: list[Provenance] = []
        seen: set[str] = set()
        for item in [*preferred.provenance, *duplicate.provenance]:
            key = item.model_dump_json()
            if key not in seen:
                seen.add(key)
                provenance.append(item)
        summaries = sorted({preferred.evidence_summary, duplicate.evidence_summary})
        return preferred.model_copy(
            update={
                "evidence_ids": evidence_ids,
                "provenance": provenance,
                "evidence_summary": "; ".join(summaries),
            }
        )

    @staticmethod
    def _to_candidate(draft: CandidateDraft, rank: int) -> Candidate:
        return Candidate(
            id=draft.id,
            rank=rank,
            center=draft.center,
            geometry=GeoJsonPoint(coordinates=(draft.center.longitude, draft.center.latitude)),
            radius_km=draft.radius_km,
            uncertainty_basis=draft.uncertainty_basis,
            confidence=draft.confidence,
            confidence_kind=draft.confidence_kind,
            confidence_basis=draft.confidence_basis,
            granularity=draft.granularity,
            country_code=draft.country_code,
            label=draft.label,
            source=draft.source,
            evidence_ids=draft.evidence_ids,
            evidence_summary=draft.evidence_summary,
            provenance=draft.provenance,
            verification_status=draft.verification_status,
            verified=draft.verified,
            phase4_assessment=draft.phase4_assessment,
            phase5b_assessment=draft.phase5b_assessment,
            model_prediction=draft.model_prediction,
            geoclip_cluster=draft.geoclip_cluster,
            confidence_assessment=draft.confidence_assessment,
            reverse_geocode=draft.reverse_geocode,
        )
