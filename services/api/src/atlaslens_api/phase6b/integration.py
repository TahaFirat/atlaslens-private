from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from atlaslens_api.fusion import CandidateBatch, CandidateDraft
from atlaslens_api.inference.models import InferenceResult
from atlaslens_api.phase6b.ensemble import Phase6BLocalModelEnsemble
from atlaslens_api.phase6b.fusion import Phase6BFusedCandidate, Phase6BFusionResult
from atlaslens_api.phase6b.models import GeographicProviderResult
from atlaslens_api.phase6b.openai_assist import (
    HardCaseSignals,
    OpenAIGeoReviewConfig,
    OpenAIGeoReviewEvidence,
    OpenAIGeoReviewProvider,
    OpenAIGeoReviewRequest,
    OpenAIReviewCandidate,
    OpenAIReviewStructuredOutput,
)
from atlaslens_api.providers.base import OCRResult, OutcomeStatus, ProviderOutcome
from atlaslens_api.reranking.geo import geodesic_km
from atlaslens_api.schemas import (
    Evidence,
    Phase5BAssessment,
    Phase5BScoreContribution,
    Phase6BAgreementSummary,
    Phase6BCloudAssistSummary,
    Phase6BCloudBudgetSummary,
    Phase6BCloudReviewSummary,
    Phase6BFusedCandidateSummary,
    Phase6BFusionSummary,
    Phase6BOCRDetectionSummary,
    Phase6BOCRSummary,
    Phase6BProviderPredictionSummary,
    Provenance,
    QualitySummary,
    SceneSegmentationSummary,
    UncalibratedConfidenceSummary,
)


@dataclass(frozen=True, slots=True)
class Phase6BPipelineResult:
    model_predictions: dict[str, Phase6BProviderPredictionSummary]
    fusion: Phase6BFusionSummary
    ocr: Phase6BOCRSummary
    cloud_assist: Phase6BCloudAssistSummary
    evidence: tuple[Evidence, ...]
    candidate_batch: CandidateBatch | None
    warnings: tuple[str, ...]
    timings_ms: dict[str, int]


class Phase6BPipelineExtension:
    """Translate the isolated Phase 6B providers into the preserved public pipeline."""

    provider_id = "phase6b-multimodel-fusion"

    def __init__(
        self,
        *,
        local_ensemble: Phase6BLocalModelEnsemble,
        openai_provider: OpenAIGeoReviewProvider | None,
        openai_config: OpenAIGeoReviewConfig,
    ) -> None:
        self._local = local_ensemble
        self._openai = openai_provider
        self._openai_config = openai_config

    async def run(
        self,
        image_path: Path,
        *,
        analysis_id: UUID,
        geoclip: InferenceResult,
        scene: SceneSegmentationSummary | None,
        ocr_outcome: ProviderOutcome[OCRResult] | None,
        ocr_provider_id: str,
        quality: QualitySummary | None,
        allow_cloud_assist: bool,
        cloud_consent: bool,
        cancellation: asyncio.Event,
    ) -> Phase6BPipelineResult:
        image_bytes = await asyncio.to_thread(image_path.read_bytes)
        ocr_result = (
            ocr_outcome.value
            if ocr_outcome is not None
            and ocr_outcome.status == OutcomeStatus.SUCCEEDED
            and ocr_outcome.value is not None
            else None
        )
        local = await self._local.run(
            image_bytes,
            geoclip=geoclip,
            scene=scene,
            ocr=ocr_result,
            cancellation=cancellation,
        )
        predictions = {
            self._prediction_key(item): Phase6BProviderPredictionSummary.model_validate(
                item.model_dump(mode="json")
            )
            for item in local.model_predictions
        }
        fusion_summary = self._fusion_summary(local.fusion, local.model_predictions)
        ocr_summary = self._ocr_summary(ocr_outcome, ocr_provider_id)
        evidence, batch = self._candidate_batch(local.fusion, local.model_predictions, ocr_result)
        cloud_summary, batch = await self._cloud_review(
            analysis_id=analysis_id,
            image_bytes=image_bytes,
            fusion=local.fusion,
            predictions=local.model_predictions,
            batch=batch,
            scene=scene,
            ocr=ocr_result,
            quality=quality,
            allow=allow_cloud_assist,
            consent=cloud_consent,
            cancellation=cancellation,
        )
        warnings = tuple(
            dict.fromkeys(
                warning for prediction in local.model_predictions for warning in prediction.warnings
            )
        )
        return Phase6BPipelineResult(
            model_predictions=predictions,
            fusion=fusion_summary,
            ocr=ocr_summary,
            cloud_assist=cloud_summary,
            evidence=evidence,
            candidate_batch=batch,
            warnings=warnings,
            timings_ms={
                f"phase6b_{self._prediction_key(item)}": item.duration_ms
                for item in local.model_predictions
            },
        )

    @staticmethod
    def _prediction_key(result: GeographicProviderResult) -> str:
        if result.source_family == "mp16_family":
            return "geoclip"
        if result.provider.startswith("osv5m"):
            return "osv5m"
        return "plonk"

    @staticmethod
    def _fusion_summary(
        result: Phase6BFusionResult,
        predictions: tuple[GeographicProviderResult, ...],
    ) -> Phase6BFusionSummary:
        clusters = [
            Phase6BFusedCandidateSummary.model_validate(item.model_dump(mode="json"))
            for item in result.candidates
        ]
        top = clusters[0] if clusters else None
        completed = [item for item in predictions if item.status == "completed"]
        source_families = sorted({item.source_family for item in completed})
        geographic_disagreement = len(completed) >= 2 and not any(
            item.provider_count >= 2 for item in clusters
        )
        return Phase6BFusionSummary(
            source_families=source_families,
            agreement_summary=Phase6BAgreementSummary(
                provider_count=top.provider_count if top is not None else 0,
                independent_family_count=(top.independent_family_count if top is not None else 0),
                same_family_duplicate_support=(
                    top.same_family_duplicate_support if top is not None else 0
                ),
                geographic_disagreement=geographic_disagreement,
                ocr_agreement=top.ocr_agreement if top is not None else False,
                ocr_contradiction=top.ocr_contradiction if top is not None else False,
            ),
            candidate_clusters=clusters,
            provider_failures=list(result.provider_failures),
            warnings=list(result.warnings),
        )

    @staticmethod
    def _ocr_summary(
        outcome: ProviderOutcome[OCRResult] | None,
        provider_id: str,
    ) -> Phase6BOCRSummary:
        if outcome is None:
            return Phase6BOCRSummary(
                provider=provider_id,
                status="skipped",
                reason_code="provider_result_unavailable",
            )
        status_map = {
            OutcomeStatus.SUCCEEDED: "completed",
            OutcomeStatus.ABSTAINED: "abstained",
            OutcomeStatus.SKIPPED: "skipped",
            OutcomeStatus.FAILED: "failed",
        }
        value = outcome.value
        blocks = value.blocks if value is not None else []
        actual_provider = blocks[0].provider if blocks else provider_id
        failure_reason = outcome.failure.code if outcome.failure is not None else None
        return Phase6BOCRSummary(
            provider=actual_provider,
            status=status_map[outcome.status],
            detections=[
                Phase6BOCRDetectionSummary(
                    redacted_text=item.redacted_text,
                    script=item.script,
                    confidence=item.confidence,
                    provider=item.provider,
                    profile=item.profile,
                )
                for item in blocks
                if not item.sensitive_content
            ],
            place_evidence=list(value.place_matches) if value is not None else [],
            fallback_used="rapidocr" in actual_provider.casefold(),
            reason_code=failure_reason,
        )

    def _candidate_batch(
        self,
        fusion: Phase6BFusionResult,
        predictions: tuple[GeographicProviderResult, ...],
        ocr: OCRResult | None,
    ) -> tuple[tuple[Evidence, ...], CandidateBatch | None]:
        completed = {item.provider: item for item in predictions if item.status == "completed"}
        provider_evidence: dict[str, Evidence] = {}
        for provider, result in completed.items():
            provider_evidence[provider] = Evidence(
                id=f"evidence-phase6b-{self._prediction_key(result)}",
                type="geographic_model",
                label="evidence.phase6b_geographic_model",
                display_value=(
                    f"{result.model_id}: {len(result.candidates)} bounded candidate(s); "
                    f"score semantics={result.score_semantics}"
                )[:500],
                confidence=None,
                confidence_basis="phase6b.raw_provider_output_not_confidence",
                source=provider,
                sensitive=False,
                provenance=self._provenance(result),
            )
        drafts: list[CandidateDraft] = []
        for cluster in fusion.candidates:
            member_providers = list(dict.fromkeys(item.provider for item in cluster.members))
            evidence_ids = [
                provider_evidence[item].id for item in member_providers if item in provider_evidence
            ]
            if not evidence_ids:
                continue
            label = self._confidence_label(cluster.independent_family_count, cluster)
            supports = [f"{item.provider}:{item.source_family}" for item in cluster.members[:12]]
            contradictions = (
                ["phase6b.strong_ocr_geographic_contradiction"] if cluster.ocr_contradiction else []
            )
            assessment = Phase5BAssessment(
                classification=(
                    "multi_source_supported"
                    if cluster.independent_family_count >= 2
                    else "model_only"
                ),
                relative_rank_score=cluster.relative_rank_score,
                reranker_version="phase6b-v1",
                score_breakdown=[
                    Phase5BScoreContribution(
                        feature=item.name,
                        raw_value=item.raw_value,
                        weight=item.weight,
                        contribution=item.contribution,
                        reason_code=f"phase6b.{item.name}",
                    )
                    for item in cluster.contributions
                ],
                provider_diversity=cluster.provider_count,
                source_diversity=cluster.independent_family_count,
                place_matches=list(ocr.place_matches) if ocr is not None else [],
                supports=supports,
                contradictions=contradictions,
                limitations=[
                    "phase6b.confidence_not_calibrated",
                    "phase6b.no_geometric_verification",
                ],
            )
            provenances = [
                self._provenance(completed[provider])
                for provider in member_providers
                if provider in completed
            ]
            provenances.append(
                Provenance(
                    provider_id=self.provider_id,
                    provider_kind="multi_model_fusion",
                    provider_version="phase6b-v1",
                    execution_boundary="local",
                    model_name=None,
                    output_schema_version="phase6b-fusion-v1",
                )
            )
            drafts.append(
                CandidateDraft(
                    id=f"candidate-{cluster.cluster_id}",
                    center={"latitude": cluster.latitude, "longitude": cluster.longitude},
                    radius_km=max(750.0, cluster.radius_km),
                    uncertainty_basis="phase6b.uncalibrated_multimodel_radius_floor",
                    confidence=None,
                    confidence_kind="uncalibrated_score",
                    confidence_basis="phase6b.relative_rank_not_probability",
                    granularity="broad_area",
                    source=self.provider_id,
                    evidence_ids=evidence_ids,
                    evidence_summary="evidence.phase6b_multimodel_geographic_agreement",
                    provenance=provenances,
                    verification_status="unverified_model",
                    verified=False,
                    phase5b_assessment=assessment,
                    confidence_assessment=UncalibratedConfidenceSummary(
                        label=label,
                        basis=self._confidence_basis(cluster),
                    ),
                )
            )
        batch = (
            CandidateBatch(provider_id=self.provider_id, candidates=tuple(drafts))
            if drafts
            else None
        )
        return tuple(provider_evidence.values()), batch

    @staticmethod
    def _provenance(result: GeographicProviderResult) -> Provenance:
        return Provenance(
            provider_id=result.provider,
            provider_kind="global_geolocation",
            provider_version=result.model_revision,
            execution_boundary="local",
            model_name=result.model_id,
            output_schema_version="phase6b-provider-output-v1",
        )

    @staticmethod
    def _confidence_label(
        independent_families: int,
        cluster: Phase6BFusedCandidate,
    ) -> Literal["low", "medium", "high"]:
        if independent_families >= 3 and not cluster.ocr_contradiction:
            return "high"
        if independent_families >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _confidence_basis(cluster: Phase6BFusedCandidate) -> list[str]:
        basis = [
            f"phase6b.independent_families.{cluster.independent_family_count}",
            f"phase6b.providers.{cluster.provider_count}",
        ]
        if cluster.same_family_duplicate_support:
            basis.append("phase6b.same_family_support_discounted")
        if cluster.ocr_agreement:
            basis.append("phase6b.ocr_agreement")
        if cluster.ocr_contradiction:
            basis.append("phase6b.ocr_contradiction")
        return basis

    async def _cloud_review(
        self,
        *,
        analysis_id: UUID,
        image_bytes: bytes,
        fusion: Phase6BFusionResult,
        predictions: tuple[GeographicProviderResult, ...],
        batch: CandidateBatch | None,
        scene: SceneSegmentationSummary | None,
        ocr: OCRResult | None,
        quality: QualitySummary | None,
        allow: bool,
        consent: bool,
        cancellation: asyncio.Event,
    ) -> tuple[Phase6BCloudAssistSummary, CandidateBatch | None]:
        if self._openai is None:
            return (
                Phase6BCloudAssistSummary(
                    allowed=allow and consent,
                    triggered=False,
                    status="skipped",
                    model=self._openai_config.model,
                    reason="provider_disabled",
                ),
                batch,
            )
        drafts = list(batch.candidates) if batch is not None else []
        clusters_by_id = {f"candidate-{item.cluster_id}": item for item in fusion.candidates}
        candidates = tuple(
            OpenAIReviewCandidate(
                candidate_id=draft.id,
                rank=index,
                source_families=tuple(
                    sorted({member.source_family for member in clusters_by_id[draft.id].members})
                ),
                supports=tuple(
                    f"{member.provider} rank {member.provider_rank}"
                    for member in clusters_by_id[draft.id].members[:8]
                ),
                contradictions=(
                    ("Strong OCR place evidence conflicts with this cluster.",)
                    if clusters_by_id[draft.id].ocr_contradiction
                    else ()
                ),
            )
            for index, draft in enumerate(drafts[:8], start=1)
        )
        top = fusion.candidates[0] if fusion.candidates else None
        completed = [item for item in predictions if item.status == "completed"]
        margin = (
            fusion.candidates[0].relative_rank_score - fusion.candidates[1].relative_rank_score
            if len(fusion.candidates) >= 2
            else None
        )
        evidence = OpenAIGeoReviewEvidence(
            candidates=candidates,
            ocr_evidence=tuple(ocr.redacted_snippets[:12]) if ocr is not None else (),
            scene_evidence=tuple(
                [
                    *(item.class_name for item in scene.dominant_classes[:8]),
                    *(item.name for item in scene.scene_groups[:4]),
                ]
            )
            if scene is not None
            else (),
            agreement=(
                (
                    f"Top cluster has {top.provider_count} providers and "
                    f"{top.independent_family_count} independent source families."
                ),
            )
            if top is not None
            else (),
            contradictions=(
                ("Top cluster conflicts with strong OCR place evidence.",)
                if top is not None and top.ocr_contradiction
                else ()
            ),
            image_quality=tuple(quality.warnings[:8]) if quality is not None else (),
        )
        signals = HardCaseSignals(
            local_confidence_label=(
                self._confidence_label(top.independent_family_count, top)
                if top is not None
                else "low"
            ),
            top_candidate_margin=margin,
            strong_model_disagreement=(
                len(completed) >= 2
                and not any(item.provider_count >= 2 for item in fusion.candidates)
            ),
            maximum_model_separation_km=self._maximum_separation(predictions),
            strong_ocr_conflict=bool(top and top.ocr_contradiction),
            no_usable_candidate=not drafts,
            candidate_dispersion_km=top.spread_km if top is not None else 0,
            user_explicit_review=allow,
            strong_ocr_and_independent_consensus=bool(
                top
                and top.ocr_agreement
                and top.independent_family_count >= 2
                and not top.ocr_contradiction
            ),
        )
        result = await self._openai.review(
            OpenAIGeoReviewRequest(
                analysis_id=analysis_id,
                image_bytes=image_bytes,
                evidence=evidence,
                signals=signals,
                cloud_consent=allow and consent,
            ),
            cancellation=cancellation,
        )
        try:
            usage = self._openai.usage_summary()
            budget = Phase6BCloudBudgetSummary(
                calls_today=usage.cloud_calls_today,
                estimated_month_spend_usd=float(usage.estimated_spend_this_month_usd),
                configured_monthly_budget_usd=float(usage.monthly_budget_usd),
                remaining_budget_usd=float(usage.remaining_configured_budget_usd),
            )
        except (OSError, ValueError):
            budget = None
        review = (
            Phase6BCloudReviewSummary.model_validate(result.review.model_dump(mode="json"))
            if result.review is not None
            else None
        )
        summary = Phase6BCloudAssistSummary(
            allowed=allow and consent,
            triggered=result.called or result.cache_hit,
            status=result.status,
            model=result.model,
            prompt_version=result.prompt_version,
            reason=result.reason_code,
            trigger_reasons=list(result.trigger_reasons),
            cache_hit=result.cache_hit,
            estimated_cost_usd=(
                float(result.estimated_cost_usd) if result.estimated_cost_usd is not None else None
            ),
            cost_estimate_version=result.cost_estimate_version,
            budget=budget,
            review=review,
            warnings=list(result.warnings),
            cache_key=result.cache_key,
        )
        return summary, self._apply_cloud_adjustments(batch, result.review)

    @staticmethod
    def _apply_cloud_adjustments(
        batch: CandidateBatch | None, review: OpenAIReviewStructuredOutput | None
    ) -> CandidateBatch | None:
        if batch is None or review is None:
            return batch
        adjustments = {item.candidate_id: item.adjustment for item in review.candidate_adjustments}
        if review.decision == "reject_all":
            for candidate in batch.candidates:
                adjustments.setdefault(candidate.id, -0.15)
        updated: list[CandidateDraft] = []
        for candidate in batch.candidates:
            adjustment = adjustments.get(candidate.id)
            assessment = candidate.phase5b_assessment
            if adjustment is None or assessment is None:
                updated.append(candidate)
                continue
            score = min(1.0, max(0.0, assessment.relative_rank_score + adjustment))
            updated_assessment = assessment.model_copy(
                update={
                    "relative_rank_score": score,
                    "score_breakdown": [
                        *assessment.score_breakdown,
                        Phase5BScoreContribution(
                            feature="openai_bounded_adjustment",
                            raw_value=adjustment,
                            weight=1.0,
                            contribution=adjustment,
                            reason_code="phase6b.openai_bounded_adjustment",
                        ),
                    ],
                }
            )
            confidence = candidate.confidence_assessment
            if review.decision == "reject_all" and confidence is not None:
                confidence = confidence.model_copy(
                    update={
                        "label": "low",
                        "basis": [*confidence.basis, "phase6b.openai_rejected_all_candidates"],
                    }
                )
            updated.append(
                candidate.model_copy(
                    update={
                        "phase5b_assessment": updated_assessment,
                        "confidence_assessment": confidence,
                    }
                )
            )
        return CandidateBatch(provider_id=batch.provider_id, candidates=tuple(updated))

    @staticmethod
    def _maximum_separation(predictions: tuple[GeographicProviderResult, ...]) -> float:
        points = [
            (candidate.latitude, candidate.longitude)
            for result in predictions
            if result.status == "completed"
            for candidate in result.candidates[:5]
        ]
        maximum = 0.0
        for left, point in enumerate(points):
            for other in points[left + 1 :]:
                maximum = max(maximum, geodesic_km(point, other))
        return maximum
