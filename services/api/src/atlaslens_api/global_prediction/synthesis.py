from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import median

from atlaslens_api.fusion import CandidateBatch, CandidateDraft, haversine_km, provenance_for
from atlaslens_api.gazetteer import CoordinateFallbackResolver, GazetteerResolver
from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    ProviderDescriptor,
)
from atlaslens_api.reranking.model_prediction import ModelPredictionReranker
from atlaslens_api.schemas import (
    Evidence,
    GeoPoint,
    ModelPredictionDiagnostics,
    Phase4Assessment,
    Phase4ScoreContribution,
    PlaceLabelDiagnostics,
)


@dataclass(frozen=True, slots=True)
class GlobalPredictionSynthesis:
    evidence: tuple[Evidence, ...]
    batch: CandidateBatch
    warnings: tuple[str, ...]


class GlobalPredictionCandidateProvider:
    descriptor = ProviderDescriptor(
        id="global-prediction-candidate-synthesis",
        kind="candidate_synthesis",
        version="phase5-v1",
        execution_boundary="local",
        criticality="optional",
        available=True,
    )

    def __init__(
        self,
        *,
        gazetteer: GazetteerResolver | None = None,
        deduplication_radius_km: float = 0.001,
        reranker: ModelPredictionReranker | None = None,
    ) -> None:
        if deduplication_radius_km <= 0 or not math.isfinite(deduplication_radius_km):
            raise ValueError("deduplication radius must be positive and finite")
        self._gazetteer = gazetteer or CoordinateFallbackResolver()
        self._deduplication_radius_km = deduplication_radius_km
        self._reranker = reranker or ModelPredictionReranker()

    def synthesize(
        self, result: GlobalPredictionResult, provider_descriptor: ProviderDescriptor
    ) -> GlobalPredictionSynthesis:
        if result.provider_id != provider_descriptor.id:
            raise ValueError("global prediction provider provenance mismatch")
        hypotheses = self._deduplicate(result.hypotheses)
        ranked_hypotheses = self._reranker.rerank(hypotheses)
        if not ranked_hypotheses:
            raise ValueError("global prediction contains no usable hypothesis")
        candidates: list[CandidateDraft] = []
        evidence: list[Evidence] = []
        warnings: list[str] = []
        provider_provenance = provenance_for(provider_descriptor)
        synthesis_provenance = provenance_for(self.descriptor)
        for ranked in ranked_hypotheses:
            hypothesis = ranked.hypothesis
            if hypothesis.raw_score > 1.0:
                raise ValueError("model relative score must be at most one")
            point = GeoPoint(latitude=hypothesis.latitude, longitude=hypothesis.longitude)
            other_distances = [
                haversine_km(
                    point,
                    GeoPoint(latitude=other.latitude, longitude=other.longitude),
                )
                for other in hypotheses
                if other is not hypothesis
            ]
            dispersion = median(other_distances) if other_distances else 0.0
            radius_km = max(750.0, dispersion)
            try:
                place = self._gazetteer.resolve(hypothesis.latitude, hypothesis.longitude)
            except Exception:
                place = CoordinateFallbackResolver().resolve(
                    hypothesis.latitude, hypothesis.longitude
                )
                warnings.append("provider.gazetteer.safe_fallback")
            stable_payload = (
                f"{result.provider_id}:{result.model_revision}:"
                f"{hypothesis.latitude:.6f}:{hypothesis.longitude:.6f}:"
                f"{hypothesis.original_rank}"
            )
            stable = hashlib.sha256(stable_payload.encode()).hexdigest()[:20]
            evidence_id = f"evidence-global-prediction-{stable}"
            evidence.append(
                Evidence(
                    id=evidence_id,
                    type="global_model_prediction",
                    label="evidence.global_model_prediction",
                    display_value="evidence.global_model_prediction_present",
                    confidence=None,
                    confidence_basis="phase5.uncalibrated_model_score_not_confidence",
                    source=result.provider_id,
                    sensitive=False,
                    provenance=provider_provenance,
                )
            )
            limitations = list(
                dict.fromkeys(
                    [
                        *hypothesis.limitations,
                        "phase5.model_prediction_is_unverified",
                        "phase5.model_score_is_not_geographic_probability",
                        "phase5.minimum_model_only_radius_750km",
                    ]
                )
            )[:12]
            diagnostics = ModelPredictionDiagnostics(
                provider_id=result.provider_id,
                model_name=result.model_name,
                model_revision=result.model_revision,
                implementation_revision=result.implementation_revision,
                device=result.device,
                dtype=result.dtype,
                raw_score=hypothesis.raw_score,
                score_type=hypothesis.score_type,
                normalization_method=hypothesis.normalization_method,
                calibration_state=hypothesis.calibration_state,
                original_rank=hypothesis.original_rank,
                inference_ms=result.inference_ms,
                external_transfer=result.external_transfer,
                limitations=limitations,
                place_label=PlaceLabelDiagnostics(
                    country=place.country,
                    region=place.region,
                    city=place.city,
                    distance_to_place_km=place.distance_km,
                    source=place.source,
                    dataset_version=place.dataset_version,
                    license=place.license,
                ),
            )
            phase4 = Phase4Assessment(
                classification="model_only",
                relative_rank_score=ranked.relative_rank_score,
                reranker_version=self._reranker.version,
                score_breakdown=[
                    Phase4ScoreContribution(
                        feature="global_model_raw_relative_score",
                        raw_value=hypothesis.raw_score,
                        weight=1.0,
                        contribution=hypothesis.raw_score,
                        reason_code="phase5.raw_uncalibrated_model_rank_feature",
                    )
                ],
                source_diversity=1,
                contributing_retrieval_hit_ids=[],
                limitations=limitations,
            )
            candidates.append(
                CandidateDraft(
                    id=f"candidate-global-{stable}",
                    center=point,
                    radius_km=radius_km,
                    uncertainty_basis="phase5.topk_geodesic_dispersion_with_750km_floor",
                    confidence=None,
                    confidence_kind="uncalibrated_score",
                    confidence_basis="phase5.no_confidence_before_calibration",
                    granularity="broad_area",
                    country_code=place.country_code,
                    label=place.label,
                    source=result.provider_id,
                    evidence_ids=[evidence_id],
                    evidence_summary="evidence.global_model_prediction_unverified",
                    provenance=[provider_provenance, synthesis_provenance],
                    verification_status="unverified_model",
                    verified=False,
                    phase4_assessment=phase4,
                    model_prediction=diagnostics,
                )
            )
        return GlobalPredictionSynthesis(
            evidence=tuple(evidence),
            batch=CandidateBatch(provider_id=result.provider_id, candidates=tuple(candidates)),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _deduplicate(
        self, hypotheses: list[GlobalPredictionHypothesis]
    ) -> list[GlobalPredictionHypothesis]:
        ordered = sorted(
            hypotheses,
            key=lambda item: (
                item.rank,
                item.original_rank,
                -item.raw_score,
                item.latitude,
                item.longitude,
            ),
        )
        kept: list[GlobalPredictionHypothesis] = []
        for hypothesis in ordered:
            point = GeoPoint(latitude=hypothesis.latitude, longitude=hypothesis.longitude)
            if any(
                haversine_km(
                    point,
                    GeoPoint(latitude=item.latitude, longitude=item.longitude),
                )
                <= self._deduplication_radius_km
                for item in kept
            ):
                continue
            kept.append(hypothesis)
        return kept
