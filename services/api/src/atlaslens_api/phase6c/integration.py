from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from atlaslens_api.fusion import CandidateBatch, CandidateDraft
from atlaslens_api.inference.models import InferenceResult
from atlaslens_api.phase6b.integration import Phase6BPipelineResult
from atlaslens_api.phase6c.fusion import (
    G3VerificationEvidence,
    GeoCLIPHierarchicalEvidence,
    GeoCLIPOriginalEvidence,
    MegaLocRetrievalEvidence,
    OCRPlaceMatchEvidence,
    OpenAIReviewEvidence,
    OSVDirectRegressionEvidence,
    Phase6CEvidence,
    Phase6CEvidenceCandidate,
    Phase6CFusionResult,
    Phase6CGeographicFusionEngine,
    PlonkINatSampleEvidence,
    PlonkOSVSampleEvidence,
    PlonkYFCCSampleEvidence,
)
from atlaslens_api.phase6c.hierarchical import (
    GeoCLIPHierarchicalSearchProvider,
    HierarchicalSearchResult,
)
from atlaslens_api.phase6c.megaloc import (
    MEGALOC_MODEL_ID,
    MEGALOC_MODEL_REVISION,
    MEGALOC_SOURCE_REVISION,
    MegaLocProviderCapability,
    MegaLocRetrievalProvider,
    MegaLocRetrievalResult,
)
from atlaslens_api.phase6c.ocr import (
    Phase6COCRConfig,
    build_place_token_evidence,
    to_phase6c_fusion_evidence,
)
from atlaslens_api.phase6c.reference_index import ReferenceSearchHit
from atlaslens_api.providers.base import (
    OCRResult,
    OutcomeStatus,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.schemas import (
    Evidence,
    GeoPoint,
    Phase5BAssessment,
    Phase5BScoreContribution,
    Phase6CAblationCandidateSummary,
    Phase6CAblationSummary,
    Phase6CAnalysisSummary,
    Phase6CFusedCandidateSummary,
    Phase6CHierarchicalCandidateSummary,
    Phase6CLeakageAuditSummary,
    Phase6CMegaLocMatchSummary,
    Phase6CProviderRunSummary,
    Provenance,
    UncalibratedConfidenceSummary,
)

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_TURKIYE_BOUNDS = (35.5, 42.2, 25.5, 45.0)


@dataclass(frozen=True, slots=True)
class Phase6CCacheVersions:
    pipeline_version: str
    provider_versions: tuple[str, ...]
    model_revisions: tuple[str, ...]
    fusion_config_version: str
    fusion_config_sha256: str
    reference_index_version: str | None
    ocr_version: str
    openai_prompt_version: str | None
    reference_index_leakage_attestation: str = "not_run"

    def fingerprint(self, image_sha256: str) -> str:
        if _SHA256.fullmatch(image_sha256) is None:
            raise ValueError("Phase 6C cache fingerprint requires an image SHA-256")
        payload = {
            "image_sha256": image_sha256,
            "pipeline_version": self.pipeline_version,
            "provider_versions": sorted(set(self.provider_versions)),
            "model_revisions": sorted(set(self.model_revisions)),
            "fusion_config_version": self.fusion_config_version,
            "fusion_config_sha256": self.fusion_config_sha256,
            "reference_index_version": self.reference_index_version,
            "ocr_version": self.ocr_version,
            "openai_prompt_version": self.openai_prompt_version,
            "reference_index_leakage_attestation": (
                self.reference_index_leakage_attestation
            ),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"phase6c-v1:{digest}"


@dataclass(frozen=True, slots=True)
class Phase6CPipelineResult:
    summary: Phase6CAnalysisSummary
    evidence: tuple[Evidence, ...]
    candidate_batch: CandidateBatch | None
    warnings: tuple[str, ...]
    timings_ms: dict[str, int]

    @property
    def replacement_eligible(self) -> bool:
        return self.candidate_batch is not None and self.summary.publication_candidate_count > 0


class Phase6CPipelineExtension:
    """Recall-first Phase 6C bridge over the preserved HTTP analysis pipeline."""

    provider_id = "phase6c-candidate-recall-fusion"

    def __init__(
        self,
        *,
        hierarchical: GeoCLIPHierarchicalSearchProvider | None,
        megaloc: MegaLocRetrievalProvider | None,
        fusion: Phase6CGeographicFusionEngine,
        cache_versions: Phase6CCacheVersions,
        ocr_descriptor: ProviderDescriptor,
        ocr_config: Phase6COCRConfig | None = None,
        leakage_audit: Phase6CLeakageAuditSummary | None = None,
        turkiye_refinement_enabled: bool = True,
    ) -> None:
        if cache_versions.pipeline_version != "phase6c-v1":
            raise ValueError("Phase 6C integration requires the phase6c-v1 pipeline")
        self._hierarchical = hierarchical
        self._megaloc = megaloc
        self._fusion = fusion
        self._cache_versions = cache_versions
        self._ocr_descriptor = ocr_descriptor
        self._ocr_config = ocr_config or Phase6COCRConfig()
        self._leakage_audit = leakage_audit or Phase6CLeakageAuditSummary(
            status="not_run",
            audit_version="atlaslens-leakage-audit-v1",
            references_checked=0,
            references_excluded=0,
            reason_code="reference_index_leakage_not_attested",
        )
        self._turkiye_refinement_enabled = turkiye_refinement_enabled

    @property
    def hierarchical_available(self) -> bool:
        return self._hierarchical is not None

    def cache_fingerprint(self, image_sha256: str) -> str:
        return self._cache_versions.fingerprint(image_sha256)

    async def megaloc_status(self) -> MegaLocProviderCapability | None:
        return await self._megaloc.status() if self._megaloc is not None else None

    async def close(self) -> None:
        if self._megaloc is not None:
            await self._megaloc.close()

    async def run(
        self,
        image_path: Path,
        *,
        image_sha256: str,
        geoclip: InferenceResult | None,
        phase6b: Phase6BPipelineResult | None,
        ocr_outcome: ProviderOutcome[OCRResult] | None,
        cancellation: asyncio.Event,
    ) -> Phase6CPipelineResult:
        if cancellation.is_set():
            raise asyncio.CancelledError
        image_bytes = await asyncio.to_thread(image_path.read_bytes)
        if not image_bytes:
            raise ValueError("Phase 6C requires non-empty private image bytes")

        batches: list[Phase6CEvidence] = [self._original_geoclip(geoclip)]
        batches.extend(self._phase6b_evidence(phase6b))
        ocr_evidence = self._ocr_evidence(ocr_outcome)
        batches.append(ocr_evidence)

        megaloc_result = await self._run_megaloc(image_bytes, cancellation)
        megaloc_evidence = self._megaloc_evidence(megaloc_result)
        batches.append(megaloc_evidence)

        signal_groups = derive_turkiye_signal_groups(batches)
        hierarchical_evidence, hierarchical_result = await self._run_hierarchical(
            image_bytes,
            signal_count=len(signal_groups) if self._turkiye_refinement_enabled else 0,
            cancellation=cancellation,
        )
        batches.append(hierarchical_evidence)
        batches.append(
            G3VerificationEvidence(
                provider="g3",
                model_id="Jia-py/G3-checkpoint",
                model_revision="not_integrated",
                status="disabled",
                reason_code="g3_not_integrated",
            )
        )
        batches.append(self._openai_evidence(phase6b))

        fusion_started = time.monotonic()
        fused = self._fusion.fuse(batches)
        ablations = self._ablation_summaries(batches, fused)
        fusion_ms = _elapsed_ms(fusion_started)
        public_evidence, evidence_ids = self._public_evidence(batches)
        candidate_batch = self._candidate_batch(fused, evidence_ids)
        providers = self._provider_runs(batches)
        if not any(item.provider_id == "plonk" for item in providers):
            providers.append(
                Phase6CProviderRunSummary(
                    provider_id="plonk",
                    status="unavailable",
                    duration_ms=0,
                    candidates_produced=0,
                    reason_code="phase6b_provider_unavailable",
                )
            )
        if not any(item.provider_id.startswith("osv5m") for item in providers):
            providers.append(
                Phase6CProviderRunSummary(
                    provider_id="osv5m",
                    status="unavailable",
                    duration_ms=0,
                    candidates_produced=0,
                    reason_code="phase6b_provider_unavailable",
                )
            )

        summary = Phase6CAnalysisSummary(
            reference_index_version=megaloc_result.index_version,
            hierarchical_candidates=[
                Phase6CHierarchicalCandidateSummary(
                    candidate_id=item.candidate_id,
                    latitude=item.latitude,
                    longitude=item.longitude,
                    search_level=item.search_level,
                    provider_rank=item.provider_rank,
                    provider_score=item.provider_score,
                    grid_resolution_km=item.grid_resolution_km,
                    diversity_cluster=item.diversity_cluster,
                    nearest_name=item.nearest_name,
                    nearest_kind=item.nearest_kind,
                    nearest_distance_km=item.nearest_distance_km,
                )
                for item in (hierarchical_result.candidates if hierarchical_result else ())
            ],
            megaloc_matches=[self._megaloc_match(item) for item in megaloc_result.matches],
            providers=providers[:20],
            fusion_candidates=[
                Phase6CFusedCandidateSummary.model_validate(item.model_dump(mode="json"))
                for item in fused.candidates
            ],
            publication_candidate_count=fused.publication_candidate_count,
            turkiye_signal_count=len(signal_groups),
            turkiye_signal_groups=list(signal_groups),
            leakage_audit=self._leakage_audit,
            ablations=ablations,
            reference_attributions=list(
                dict.fromkeys(
                    f"{item.reference.source}: {item.reference.attribution}"[:500]
                    for item in megaloc_result.matches
                )
            )[:20],
            cache_fingerprint=self.cache_fingerprint(image_sha256),
        )
        warnings = tuple(
            dict.fromkeys(
                [
                    *fused.warnings,
                    *megaloc_result.warnings,
                    *(hierarchical_result.warnings if hierarchical_result else ()),
                    "warning.phase6c.g3_not_integrated",
                ]
            )
        )
        return Phase6CPipelineResult(
            summary=summary,
            evidence=public_evidence,
            candidate_batch=candidate_batch,
            warnings=warnings,
            timings_ms={
                "phase6c_hierarchical": (
                    hierarchical_result.duration_ms if hierarchical_result is not None else 0
                ),
                "phase6c_megaloc": megaloc_result.duration_ms,
                "phase6c_fusion": fusion_ms,
            },
        )

    def _ablation_summaries(
        self,
        batches: Sequence[Phase6CEvidence],
        final_result: Phase6CFusionResult,
    ) -> list[Phase6CAblationSummary]:
        summaries: list[Phase6CAblationSummary] = []
        for profile_id in self._fusion.ablation_profile_ids:
            result = (
                final_result
                if profile_id == "phase6c_final"
                else self._fusion.fuse_profile(batches, profile_id)
            )
            summaries.append(
                Phase6CAblationSummary(
                    profile_id=profile_id,
                    candidate_count=len(result.candidates),
                    publication_candidate_count=result.publication_candidate_count,
                    abstained=result.abstained,
                    abstention_reason=result.abstention_reason,
                    included_evidence_kinds=list(result.ablation.included_evidence_kinds),
                    excluded_evidence_kinds=list(result.ablation.excluded_evidence_kinds),
                    candidates=[
                        Phase6CAblationCandidateSummary(
                            rank=item.final_rank,
                            latitude=item.latitude,
                            longitude=item.longitude,
                            uncertainty_radius_km=item.uncertainty_radius_km,
                            relative_rank_score=item.relative_rank_score,
                            publication_eligible=item.publication_eligible,
                            evidence_kinds=list(
                                dict.fromkeys(member.evidence_kind for member in item.members)
                            ),
                        )
                        for item in result.candidates[:5]
                    ],
                )
            )
        return summaries

    @staticmethod
    def _original_geoclip(result: InferenceResult | None) -> GeoCLIPOriginalEvidence:
        if result is None:
            return GeoCLIPOriginalEvidence(
                provider="geoclip-global-v1",
                model_id="GeoCLIP",
                model_revision="unavailable",
                status="skipped",
                reason_code="geoclip_original_unavailable",
            )
        if result.status != "succeeded":
            reason = result.failure.code if result.failure is not None else "provider_abstained"
            status: Literal["abstained", "failed", "skipped"] = (
                "failed"
                if result.status == "failed"
                else "skipped"
                if result.status == "skipped"
                else "abstained"
            )
            return GeoCLIPOriginalEvidence(
                provider=result.provider_id,
                model_id=result.model_name,
                model_revision=result.model_revision,
                status=status,
                duration_ms=result.runtime_ms,
                reason_code=_safe_code(reason),
            )
        candidates = tuple(
            Phase6CEvidenceCandidate(
                candidate_id=(
                    "geoclip-original-"
                    + _stable_id(
                        result.provider_id,
                        str(item.original_rank),
                        str(item.latitude),
                        str(item.longitude),
                    )
                ),
                latitude=item.latitude,
                longitude=item.longitude,
                raw_value=item.raw_score,
                provider_rank=item.rank,
                uncertainty_radius_km=750.0,
                provenance=f"{result.provider_id}:{result.model_revision}:gallery_rank_{item.original_rank}",
            )
            for item in result.candidates
        )
        return GeoCLIPOriginalEvidence(
            provider=result.provider_id,
            model_id=result.model_name,
            model_revision=result.model_revision,
            duration_ms=result.runtime_ms,
            candidates=candidates,
        )

    @staticmethod
    def _phase6b_evidence(
        phase6b: Phase6BPipelineResult | None,
    ) -> list[Phase6CEvidence]:
        if phase6b is None:
            return []
        evidence: list[Phase6CEvidence] = []
        for result in phase6b.model_predictions.values():
            if result.source_family == "mp16_family":
                continue
            status = result.status
            reason = _safe_code(result.reason_code or status)
            if result.provider.startswith("osv5m"):
                candidates: tuple[Phase6CEvidenceCandidate, ...] = ()
                if status == "completed" and len(result.candidates) == 1:
                    item = result.candidates[0]
                    candidates = (
                        Phase6CEvidenceCandidate(
                            candidate_id=item.candidate_id,
                            latitude=item.latitude,
                            longitude=item.longitude,
                            raw_value=None,
                            provider_rank=1,
                            sample_support=item.sample_support,
                            uncertainty_radius_km=750.0,
                            provenance=f"{result.provider}:{result.model_revision}:direct_regression",
                        ),
                    )
                elif status == "completed":
                    status, reason = "failed", "invalid_direct_regression_output"
                evidence.append(
                    OSVDirectRegressionEvidence(
                        provider=result.provider,
                        model_id=result.model_id,
                        model_revision=result.model_revision,
                        status=status,
                        duration_ms=result.duration_ms,
                        candidates=candidates,
                        reason_code=None if status == "completed" else reason,
                    )
                )
                continue
            if not result.provider.startswith("plonk"):
                continue
            candidate_rows: tuple[Phase6CEvidenceCandidate, ...] = ()
            if status == "completed" and all(
                item.raw_score is not None for item in result.candidates
            ):
                candidate_rows = tuple(
                    Phase6CEvidenceCandidate(
                        candidate_id=item.candidate_id,
                        latitude=item.latitude,
                        longitude=item.longitude,
                        raw_value=item.raw_score,
                        provider_rank=item.provider_rank,
                        sample_support=item.sample_support,
                        uncertainty_radius_km=max(
                            25.0, _numeric_metadata(item.metadata.get("dispersion_km"))
                        ),
                        provenance=f"{result.provider}:{result.model_revision}:sample_cluster",
                    )
                    for item in result.candidates
                )
            elif status == "completed":
                status, reason = "failed", "invalid_sample_density_output"
            if result.source_family == "osv5m_family":
                evidence.append(
                    PlonkOSVSampleEvidence(
                        provider=result.provider,
                        model_id=result.model_id,
                        model_revision=result.model_revision,
                        status=status,
                        duration_ms=result.duration_ms,
                        candidates=candidate_rows,
                        reason_code=None if status == "completed" else reason,
                    )
                )
            elif result.source_family == "inat_family":
                evidence.append(
                    PlonkINatSampleEvidence(
                        provider=result.provider,
                        model_id=result.model_id,
                        model_revision=result.model_revision,
                        status=status,
                        duration_ms=result.duration_ms,
                        candidates=candidate_rows,
                        reason_code=None if status == "completed" else reason,
                    )
                )
            else:
                evidence.append(
                    PlonkYFCCSampleEvidence(
                        provider=result.provider,
                        model_id=result.model_id,
                        model_revision=result.model_revision,
                        status=status,
                        duration_ms=result.duration_ms,
                        candidates=candidate_rows,
                        reason_code=None if status == "completed" else reason,
                    )
                )
        return evidence

    def _ocr_evidence(self, outcome: ProviderOutcome[OCRResult] | None) -> OCRPlaceMatchEvidence:
        provider = _safe_code(self._ocr_descriptor.id)
        if outcome is None:
            return OCRPlaceMatchEvidence(
                provider=provider,
                model_id=self._ocr_descriptor.model_name or "local-ocr",
                model_revision=self._ocr_descriptor.version,
                status="skipped",
                reason_code="ocr_result_unavailable",
            )
        value = outcome.value
        if outcome.status == OutcomeStatus.SUCCEEDED and value is not None:
            if value.blocks:
                provider = _safe_code(value.blocks[0].provider)
            token_evidence = build_place_token_evidence(value, self._ocr_config)
            return to_phase6c_fusion_evidence(
                token_evidence,
                provider=provider,
                model_id=self._ocr_descriptor.model_name or "local-ocr",
                model_revision=self._ocr_descriptor.version,
                duration_ms=_outcome_duration(outcome),
            )
        status: Literal["abstained", "failed", "skipped"] = (
            "failed"
            if outcome.status == OutcomeStatus.FAILED
            else "skipped"
            if outcome.status == OutcomeStatus.SKIPPED
            else "abstained"
        )
        return OCRPlaceMatchEvidence(
            provider=provider,
            model_id=self._ocr_descriptor.model_name or "local-ocr",
            model_revision=self._ocr_descriptor.version,
            status=status,
            duration_ms=_outcome_duration(outcome),
            reason_code=_safe_code(
                outcome.failure.code if outcome.failure is not None else outcome.status.value
            ),
        )

    async def _run_megaloc(
        self, image_bytes: bytes, cancellation: asyncio.Event
    ) -> MegaLocRetrievalResult:
        if self._megaloc is None:
            return MegaLocRetrievalResult(
                status="skipped",
                reason_code="provider_unavailable",
                model_id=MEGALOC_MODEL_ID,
                model_revision=MEGALOC_MODEL_REVISION,
                source_revision=MEGALOC_SOURCE_REVISION,
                query_descriptor_version="megaloc-7cb9f797-max560-imagenet-v1",
                device="cpu",
                duration_ms=0,
            )
        return await self._megaloc.retrieve(image_bytes, cancellation=cancellation)

    @staticmethod
    def _megaloc_evidence(result: MegaLocRetrievalResult) -> MegaLocRetrievalEvidence:
        candidates = tuple(
            Phase6CEvidenceCandidate(
                candidate_id=item.cluster_id,
                latitude=item.latitude,
                longitude=item.longitude,
                raw_value=item.best_similarity,
                provider_rank=rank,
                sample_support=item.independent_sequence_support,
                uncertainty_radius_km=item.uncertainty_radius_km,
                provenance=f"reference_index:{result.index_version or 'unavailable'}:cluster",
                retrieval_cluster_id=item.cluster_id,
            )
            for rank, item in enumerate(result.clusters, start=1)
        )
        return MegaLocRetrievalEvidence(
            provider=result.provider,
            model_id=result.model_id,
            model_revision=result.model_revision,
            status=result.status,
            duration_ms=result.duration_ms,
            candidates=candidates,
            reason_code=result.reason_code,
        )

    async def _run_hierarchical(
        self,
        image_bytes: bytes,
        *,
        signal_count: int,
        cancellation: asyncio.Event,
    ) -> tuple[GeoCLIPHierarchicalEvidence, HierarchicalSearchResult | None]:
        if self._hierarchical is None:
            return (
                GeoCLIPHierarchicalEvidence(
                    provider="geoclip_hierarchical_search",
                    model_id="GeoCLIP-1.2.0",
                    model_revision="geoclip-hierarchical-v1",
                    status="disabled",
                    reason_code="hierarchical_provider_unavailable",
                ),
                None,
            )
        try:
            result = await self._hierarchical.search(
                image_bytes,
                turkiye_signal_count=signal_count,
                cancellation=cancellation,
            )
        except TimeoutError:
            return (
                GeoCLIPHierarchicalEvidence(
                    provider="geoclip_hierarchical_search",
                    model_id="GeoCLIP-1.2.0",
                    model_revision="geoclip-hierarchical-v1",
                    status="timeout",
                    reason_code="hierarchical_search_timeout",
                ),
                None,
            )
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, ValueError):
            return (
                GeoCLIPHierarchicalEvidence(
                    provider="geoclip_hierarchical_search",
                    model_id="GeoCLIP-1.2.0",
                    model_revision="geoclip-hierarchical-v1",
                    status="failed",
                    reason_code="hierarchical_search_failed",
                ),
                None,
            )
        candidates = tuple(
            Phase6CEvidenceCandidate(
                candidate_id=item.candidate_id,
                latitude=item.latitude,
                longitude=item.longitude,
                raw_value=item.provider_score,
                provider_rank=item.provider_rank,
                uncertainty_radius_km=max(5.0, item.grid_resolution_km / 2.0),
                provenance=f"{result.catalogue_version}:{item.search_level}:coordinate_search",
                mode_id=f"{item.search_level}:{item.diversity_cluster}",
            )
            for item in result.candidates
        )
        return (
            GeoCLIPHierarchicalEvidence(
                provider=result.provider,
                model_id=result.model_id,
                model_revision=result.provider_revision,
                duration_ms=result.duration_ms,
                candidates=candidates,
            ),
            result,
        )

    @staticmethod
    def _openai_evidence(
        phase6b: Phase6BPipelineResult | None,
    ) -> OpenAIReviewEvidence:
        if phase6b is None or not phase6b.cloud_assist.triggered:
            return OpenAIReviewEvidence(
                provider="openai_geo_review",
                model_id=(phase6b.cloud_assist.model if phase6b is not None else "disabled"),
                model_revision=(
                    phase6b.cloud_assist.prompt_version
                    if phase6b is not None
                    else "openai-geo-review-v1"
                ),
                status="skipped",
                reason_code="not_triggered",
            )
        review = phase6b.cloud_assist.review
        batch = phase6b.candidate_batch
        if review is None or batch is None or not review.selected_candidate_ids:
            return OpenAIReviewEvidence(
                provider="openai_geo_review",
                model_id=phase6b.cloud_assist.model,
                model_revision=phase6b.cloud_assist.prompt_version,
                status="abstained",
                reason_code="review_no_supported_candidate",
            )
        by_id = {item.id: item for item in batch.candidates}
        selected = [by_id[item] for item in review.selected_candidate_ids if item in by_id]
        if not selected:
            return OpenAIReviewEvidence(
                provider="openai_geo_review",
                model_id=phase6b.cloud_assist.model,
                model_revision=phase6b.cloud_assist.prompt_version,
                status="abstained",
                reason_code="review_candidate_unavailable",
            )
        candidates = tuple(
            Phase6CEvidenceCandidate(
                candidate_id=f"openai-review-{_stable_id(item.id)}",
                latitude=item.center.latitude,
                longitude=item.center.longitude,
                raw_value=1.0 - ((rank - 1) / max(1, len(selected))),
                provider_rank=rank,
                uncertainty_radius_km=item.radius_km,
                provenance=f"{phase6b.cloud_assist.prompt_version}:bounded_existing_candidate_review",
            )
            for rank, item in enumerate(selected, start=1)
        )
        return OpenAIReviewEvidence(
            provider="openai_geo_review",
            model_id=phase6b.cloud_assist.model,
            model_revision=phase6b.cloud_assist.prompt_version,
            candidates=candidates,
        )

    @staticmethod
    def _public_evidence(
        batches: Sequence[Phase6CEvidence],
    ) -> tuple[tuple[Evidence, ...], dict[str, str]]:
        evidence: list[Evidence] = []
        ids: dict[str, str] = {}
        for batch in batches:
            if batch.status != "completed":
                continue
            evidence_id = f"evidence-phase6c-{batch.evidence_kind}-{_stable_id(batch.provider)}"
            ids[batch.provider] = evidence_id
            evidence.append(
                Evidence(
                    id=evidence_id,
                    type="geographic_model",
                    label="evidence.phase6c_candidate_recall",
                    display_value=(
                        f"{batch.model_id}: {len(batch.candidates)} bounded candidate(s); "
                        f"score semantics={batch.score_semantics}"
                    )[:500],
                    confidence=None,
                    confidence_basis="phase6c.raw_provider_output_not_confidence",
                    source=batch.provider,
                    sensitive=False,
                    provenance=Provenance(
                        provider_id=batch.provider,
                        provider_kind="phase6c_geolocation_evidence",
                        provider_version=batch.model_revision,
                        execution_boundary=(
                            "cloud" if batch.evidence_kind == "openai_review" else "local"
                        ),
                        model_name=batch.model_id,
                        output_schema_version="phase6c-provider-evidence-v1",
                    ),
                )
            )
        return tuple(evidence), ids

    def _candidate_batch(
        self, fused: Phase6CFusionResult, evidence_ids: dict[str, str]
    ) -> CandidateBatch | None:
        drafts: list[CandidateDraft] = []
        for item in fused.candidates:
            if not item.publication_eligible:
                continue
            member_evidence = list(
                dict.fromkeys(
                    evidence_ids[member.provider]
                    for member in item.members
                    if member.provider in evidence_ids
                )
            )
            if not member_evidence:
                continue
            provenances = list(
                {
                    (member.provider, member.model_revision): Provenance(
                        provider_id=member.provider,
                        provider_kind="phase6c_geolocation_evidence",
                        provider_version=member.model_revision,
                        execution_boundary=(
                            "cloud" if member.evidence_kind == "openai_review" else "local"
                        ),
                        model_name=member.model_id,
                        output_schema_version="phase6c-provider-evidence-v1",
                    )
                    for member in item.members
                }.values()
            )
            provenances.append(
                Provenance(
                    provider_id=self.provider_id,
                    provider_kind="multi_model_fusion",
                    provider_version="phase6c-v1",
                    execution_boundary="local",
                    model_name=None,
                    output_schema_version="phase6c-fusion-v1",
                )
            )
            supports = [
                (f"{member.provider}:{member.source_family}:{member.correlation_group}")[:240]
                for member in item.members[:24]
            ]
            assessment = Phase5BAssessment(
                classification="multi_source_supported",
                relative_rank_score=item.relative_rank_score,
                reranker_version="phase6c-v1",
                score_breakdown=[
                    Phase5BScoreContribution(
                        feature=entry.name,
                        raw_value=entry.raw_value,
                        weight=entry.weight,
                        contribution=entry.contribution,
                        reason_code=f"phase6c.{entry.name}",
                    )
                    for entry in item.contributions
                ],
                provider_diversity=item.provider_count,
                source_diversity=item.independent_source_family_count,
                supports=supports,
                movement_reasons=list(item.movement_reasons),
                limitations=[
                    "phase6c.confidence_not_calibrated",
                    "phase6c.candidate_recall_not_location_proof",
                    "phase6c.g3_not_integrated",
                ],
            )
            label: Literal["medium", "high", "very_high"] = (
                "very_high"
                if item.independent_source_family_count >= 4
                else "high"
                if item.independent_source_family_count >= 3
                else "medium"
            )
            drafts.append(
                CandidateDraft(
                    id=f"candidate-{item.cluster_id}",
                    center=GeoPoint(latitude=item.latitude, longitude=item.longitude),
                    radius_km=max(0.001, item.uncertainty_radius_km),
                    uncertainty_basis="phase6c.uncalibrated_multisource_geodesic_spread",
                    confidence=None,
                    confidence_kind="uncalibrated_score",
                    confidence_basis="phase6c.relative_rank_not_probability",
                    granularity="broad_area",
                    source=self.provider_id,
                    evidence_ids=member_evidence,
                    evidence_summary="evidence.phase6c_independent_source_agreement",
                    provenance=provenances,
                    verification_status="unverified_multisource",
                    verified=False,
                    phase5b_assessment=assessment,
                    confidence_assessment=UncalibratedConfidenceSummary(
                        label=label,
                        basis=[
                            f"phase6c.independent_groups.{item.independent_source_family_count}",
                            f"phase6c.providers.{item.provider_count}",
                            "phase6c.confidence_unavailable",
                        ],
                    ),
                )
            )
        return (
            CandidateBatch(provider_id=self.provider_id, candidates=tuple(drafts))
            if drafts
            else None
        )

    @staticmethod
    def _provider_runs(
        batches: Sequence[Phase6CEvidence],
    ) -> list[Phase6CProviderRunSummary]:
        source_revisions = {
            "geoclip_original": "7a1a23b49648a5872a771cfda28490a17ab17d15",
            "geoclip_hierarchical": "geoclip-hierarchical-v1",
            "osv_direct_regression": "4e6075387ecde4255410785ffb83830c9aa099f6",
            "plonk_samples": "76d46410910c9dfec9e19ed371450ebc7051cdf3",
            "megaloc_retrieval": MEGALOC_SOURCE_REVISION,
            "g3_verification": "b4e3acf7c0ac51221f21b7877fefb4826715c9e2",
            "ocr_place_match": "local-ocr-provider-boundary-v1",
            "openai_review": "openai-geo-review-v1",
        }
        return [
            Phase6CProviderRunSummary(
                provider_id=batch.provider,
                status=batch.status,
                source_revision=source_revisions[batch.evidence_kind],
                model_revision=batch.model_revision,
                duration_ms=batch.duration_ms,
                candidates_produced=len(batch.candidates),
                reason_code=batch.reason_code,
            )
            for batch in batches
        ]

    @staticmethod
    def _megaloc_match(item: ReferenceSearchHit) -> Phase6CMegaLocMatchSummary:
        # Kept as a narrow runtime boundary so reference objects never leak paths.
        reference = item.reference
        return Phase6CMegaLocMatchSummary(
            reference_id=item.reference_id,
            rank=item.rank,
            latitude=reference.latitude,
            longitude=reference.longitude,
            similarity=item.similarity,
            uncertainty_radius_m=item.uncertainty_radius_m,
            source=reference.source,
            source_family=reference.source_family,
            source_sequence_id=reference.source_sequence_id,
            source_url=reference.source_url,
            captured_at=reference.captured_at,
            province=reference.province,
            city=reference.city,
            license=reference.license,
            attribution=reference.attribution,
        )


def derive_turkiye_signal_groups(
    evidence: Sequence[Phase6CEvidence],
) -> tuple[str, ...]:
    """Count generic independent evidence groups indicating the Türkiye region."""

    groups = {
        batch.correlation_group
        for batch in evidence
        if batch.status == "completed"
        and batch.evidence_kind != "geoclip_hierarchical"
        and any(
            candidate.country_code == "TR"
            or _coordinate_in_turkiye_region(candidate.latitude, candidate.longitude)
            for candidate in batch.candidates
        )
    }
    return tuple(sorted(groups))[:8]


def _coordinate_in_turkiye_region(latitude: float, longitude: float) -> bool:
    south, north, west, east = _TURKIYE_BOUNDS
    return south <= latitude <= north and west <= longitude <= east


def _numeric_metadata(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else 0.0


def _stable_id(*values: str) -> str:
    return hashlib.sha256("|".join(values).encode()).hexdigest()[:24]


def _safe_code(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9._:-]+", "_", value.casefold()).strip("_.:-")
    return (cleaned or "unavailable")[:120]


def _outcome_duration[T](outcome: ProviderOutcome[T]) -> int:
    return outcome.failure.duration_ms if outcome.failure is not None else 0


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
