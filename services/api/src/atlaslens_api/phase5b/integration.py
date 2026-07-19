from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from atlaslens_api.fusion import CandidateDraft
from atlaslens_api.global_prediction.synthesis import GlobalPredictionSynthesis
from atlaslens_api.phase5b.models import EvidenceHypothesis
from atlaslens_api.providers.base import (
    GlobalPredictionResult,
    OCRResult,
    OutcomeStatus,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.schemas import (
    Evidence,
    PlaceEvidenceSummary,
    Provenance,
    ProviderRunDiagnostic,
)

_PLACE_RADIUS_KM = {
    "country": 1_500.0,
    "domain_suffix": 1_500.0,
    "region": 350.0,
    "city": 75.0,
    "district": 50.0,
    "road": 75.0,
    "airport": 50.0,
    "station": 50.0,
    "public_landmark": 25.0,
    "public_institution": 50.0,
}
_SAFE_REASON = re.compile(r"[^a-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class Phase5BSourceAdaptation:
    hypotheses: tuple[EvidenceHypothesis, ...]
    evidence: tuple[Evidence, ...]
    providers: tuple[ProviderRunDiagnostic, ...]
    partial_failures: tuple[str, ...]


class Phase5BSourceAdapter:
    """Translate existing provider outputs into source-specific Phase 5B evidence.

    The adapter never copies OCR snippets or blocks. Only canonical, attributed
    public-place entities already returned by the local place resolver cross this
    boundary.
    """

    version = "phase5b-source-adapter-v1"

    def adapt(
        self,
        *,
        global_descriptor: ProviderDescriptor,
        ocr_descriptor: ProviderDescriptor,
        global_synthesis: GlobalPredictionSynthesis | None = None,
        ocr_result: OCRResult | None = None,
        global_outcome: ProviderOutcome[GlobalPredictionResult] | None = None,
        ocr_outcome: ProviderOutcome[OCRResult] | None = None,
        global_duration_ms: int | None = None,
        ocr_duration_ms: int | None = None,
        ocr_device: str | None = None,
    ) -> Phase5BSourceAdaptation:
        effective_ocr = self._effective_ocr_result(ocr_result, ocr_outcome)
        global_usable = self._source_usable(global_synthesis is not None, global_outcome)
        ocr_usable = self._source_usable(effective_ocr is not None, ocr_outcome)

        global_diagnostic = self._global_diagnostic(
            global_descriptor,
            global_synthesis=global_synthesis,
            global_outcome=global_outcome,
            usable=global_usable,
            duration_ms=global_duration_ms,
        )
        ocr_diagnostic = self._diagnostic(
            ocr_descriptor,
            provider_type="ocr_place",
            outcome=ocr_outcome,
            value_present=effective_ocr is not None,
            usable=ocr_usable,
            duration_ms=ocr_duration_ms,
            device=ocr_device,
        )

        evidence: list[Evidence] = []
        hypotheses: list[EvidenceHypothesis] = []
        partial_failures: list[str] = []
        if global_usable and global_synthesis is not None:
            global_hypotheses, global_evidence, rejected = self._adapt_global(
                global_synthesis, expected_provider_id=global_descriptor.id
            )
            hypotheses.extend(global_hypotheses)
            evidence.extend(global_evidence)
            if rejected:
                partial_failures.append("phase5b.global.invalid_candidate_draft")
        elif global_diagnostic.status in {"failed", "skipped"}:
            partial_failures.append(
                self._partial_failure("global", global_diagnostic.reason_code)
            )

        if ocr_usable and effective_ocr is not None:
            place_hypotheses, place_evidence = self._adapt_places(
                effective_ocr.place_matches, ocr_descriptor
            )
            hypotheses.extend(place_hypotheses)
            evidence.extend(place_evidence)
        elif ocr_diagnostic.status in {"failed", "skipped"}:
            partial_failures.append(self._partial_failure("ocr", ocr_diagnostic.reason_code))

        evidence_by_id = {item.id: item for item in evidence}
        hypothesis_by_id = {item.id: item for item in hypotheses}
        return Phase5BSourceAdaptation(
            hypotheses=tuple(
                sorted(
                    hypothesis_by_id.values(),
                    key=lambda item: (
                        item.original_rank,
                        item.source_kind,
                        item.provider_id,
                        item.id,
                    ),
                )
            ),
            evidence=tuple(sorted(evidence_by_id.values(), key=lambda item: item.id)),
            providers=(global_diagnostic, ocr_diagnostic),
            partial_failures=tuple(dict.fromkeys(partial_failures)),
        )

    @staticmethod
    def _effective_ocr_result(
        result: OCRResult | None, outcome: ProviderOutcome[OCRResult] | None
    ) -> OCRResult | None:
        if outcome is None:
            return result
        if outcome.status != OutcomeStatus.SUCCEEDED:
            return None
        return outcome.value or result

    @staticmethod
    def _source_usable[T](
        value_present: bool, outcome: ProviderOutcome[T] | None
    ) -> bool:
        if outcome is None:
            return value_present
        return outcome.status == OutcomeStatus.SUCCEEDED and value_present

    def _global_diagnostic(
        self,
        descriptor: ProviderDescriptor,
        *,
        global_synthesis: GlobalPredictionSynthesis | None,
        global_outcome: ProviderOutcome[GlobalPredictionResult] | None,
        usable: bool,
        duration_ms: int | None,
    ) -> ProviderRunDiagnostic:
        value = global_outcome.value if global_outcome is not None else None
        inferred_duration = (
            duration_ms
            if duration_ms is not None
            else value.inference_ms
            if value is not None
            else 0
        )
        device = value.device if value is not None else self._global_device(global_synthesis)
        return self._diagnostic(
            descriptor,
            provider_type="global_geolocation",
            outcome=global_outcome,
            value_present=global_synthesis is not None,
            usable=usable,
            duration_ms=inferred_duration,
            device=device,
        )

    @staticmethod
    def _global_device(synthesis: GlobalPredictionSynthesis | None) -> str | None:
        if synthesis is None:
            return None
        for draft in synthesis.batch.candidates:
            if draft.model_prediction is not None:
                return draft.model_prediction.device
        return None

    def _diagnostic[T](
        self,
        descriptor: ProviderDescriptor,
        *,
        provider_type: str,
        outcome: ProviderOutcome[T] | None,
        value_present: bool,
        usable: bool,
        duration_ms: int | None,
        device: str | None,
    ) -> ProviderRunDiagnostic:
        if usable:
            status = "succeeded"
            reason = None
        elif outcome is not None and outcome.status != OutcomeStatus.SUCCEEDED:
            status = outcome.status.value
            reason = outcome.failure.code if outcome.failure is not None else outcome.status.value
        elif outcome is not None and outcome.status == OutcomeStatus.SUCCEEDED:
            status = "failed"
            reason = "invalid_source_state"
        elif not value_present:
            status = "skipped"
            reason = descriptor.unavailable_reason_code or "unavailable"
        else:
            status = "failed"
            reason = "invalid_source_state"
        failure_duration = (
            outcome.failure.duration_ms
            if outcome is not None and outcome.failure is not None
            else 0
        )
        return ProviderRunDiagnostic(
            provider_id=descriptor.id,
            provider_type=provider_type,
            status=status,
            duration_ms=max(0, duration_ms if duration_ms is not None else failure_duration),
            reason_code=self._safe_reason(reason) if reason is not None else None,
            device=device,
            offline=descriptor.execution_boundary == "local",
        )

    def _adapt_global(
        self, synthesis: GlobalPredictionSynthesis, *, expected_provider_id: str
    ) -> tuple[list[EvidenceHypothesis], list[Evidence], int]:
        evidence_by_id = {item.id: item for item in synthesis.evidence}
        hypotheses: list[EvidenceHypothesis] = []
        used_evidence: dict[str, Evidence] = {}
        rejected = 0
        drafts = sorted(
            synthesis.batch.candidates,
            key=lambda item: (
                item.model_prediction.original_rank
                if item.model_prediction is not None
                else 1_000_000,
                item.id,
            ),
        )[:10]
        for draft in drafts:
            hypothesis = self._global_hypothesis(
                draft, evidence_by_id, expected_provider_id=expected_provider_id
            )
            if hypothesis is None:
                rejected += 1
                continue
            hypotheses.append(hypothesis)
            for evidence_id in hypothesis.evidence_ids:
                used_evidence[evidence_id] = evidence_by_id[evidence_id]
        return hypotheses, list(used_evidence.values()), rejected

    def _global_hypothesis(
        self,
        draft: CandidateDraft,
        evidence_by_id: dict[str, Evidence],
        *,
        expected_provider_id: str,
    ) -> EvidenceHypothesis | None:
        prediction = draft.model_prediction
        if (
            prediction is None
            or prediction.provider_id != expected_provider_id
            or not draft.evidence_ids
            or not draft.provenance
        ):
            return None
        if any(evidence_id not in evidence_by_id for evidence_id in draft.evidence_ids):
            return None
        radius = min(20_050.0, max(750.0, draft.radius_km))
        stable = self._stable_hash(
            "global",
            prediction.provider_id,
            prediction.model_revision,
            f"{draft.center.latitude:.7f}",
            f"{draft.center.longitude:.7f}",
            str(prediction.original_rank),
        )
        source_revision = self._stable_hash(
            prediction.provider_id, prediction.model_revision
        )[:20]
        return EvidenceHypothesis(
            id=f"phase5b-global-{stable[:24]}",
            provider_id=prediction.provider_id,
            source_id=f"geoclip-gallery-{source_revision}",
            source_kind="global_model",
            latitude=draft.center.latitude,
            longitude=draft.center.longitude,
            raw_score=prediction.raw_score,
            score_semantics=prediction.score_type,
            uncertainty_radius_km=radius,
            original_rank=prediction.original_rank,
            evidence_ids=tuple(draft.evidence_ids),
            provenance=tuple(draft.provenance),
            model_prediction=prediction,
        )

    def _adapt_places(
        self,
        place_matches: list[PlaceEvidenceSummary],
        descriptor: ProviderDescriptor,
    ) -> tuple[list[EvidenceHypothesis], list[Evidence]]:
        unique: dict[tuple[object, ...], PlaceEvidenceSummary] = {}
        for match in place_matches[:12]:
            key = (
                match.match_type,
                match.matched_entity.casefold(),
                match.country_code,
                round(match.center.latitude, 7),
                round(match.center.longitude, 7),
                match.dataset_version,
            )
            current = unique.get(key)
            if current is None or self._place_sort_key(match) < self._place_sort_key(current):
                unique[key] = match
        ordered = sorted(unique.values(), key=self._place_sort_key)
        hypotheses: list[EvidenceHypothesis] = []
        evidence: list[Evidence] = []
        provider_provenance = Provenance(
            provider_id=descriptor.id,
            provider_kind=descriptor.kind,
            provider_version=descriptor.version,
            execution_boundary=descriptor.execution_boundary,
            model_name=descriptor.model_name,
            output_schema_version="phase5b-ocr-place-v1",
        )
        adapter_provenance = Provenance(
            provider_id="phase5b-source-adapter",
            provider_kind="evidence_adapter",
            provider_version=self.version,
            execution_boundary="local",
            output_schema_version="phase5b-source-adapter-v1",
        )
        for rank, match in enumerate(ordered, 1):
            stable = self._stable_hash(
                "place",
                descriptor.id,
                match.source,
                match.dataset_version,
                match.match_type,
                match.matched_entity,
                match.country_code or "",
                f"{match.center.latitude:.7f}",
                f"{match.center.longitude:.7f}",
            )
            evidence_id = f"evidence-ocr-place-{stable[:24]}"
            evidence.append(
                Evidence(
                    id=evidence_id,
                    type="ocr_place_match",
                    label="evidence.ocr_public_place_match",
                    display_value=match.matched_entity,
                    confidence=match.evidence_strength,
                    confidence_basis=(
                        "phase5b.ocr_place_relative_evidence_strength_not_probability"
                    ),
                    source=descriptor.id,
                    sensitive=False,
                    provenance=provider_provenance,
                )
            )
            ambiguity_factor = min(4.0, math.sqrt(max(1, match.ambiguity_count)))
            radius = min(
                20_050.0,
                _PLACE_RADIUS_KM[match.match_type] * ambiguity_factor,
            )
            source_id = self._stable_hash(match.source, match.dataset_version)[:20]
            hypotheses.append(
                EvidenceHypothesis(
                    id=f"phase5b-place-{stable[:24]}",
                    provider_id=descriptor.id,
                    source_id=f"gazetteer-{source_id}",
                    source_kind="ocr_place",
                    latitude=match.center.latitude,
                    longitude=match.center.longitude,
                    raw_score=match.evidence_strength,
                    score_semantics="uncalibrated_ocr_place_evidence_strength",
                    uncertainty_radius_km=radius,
                    original_rank=rank,
                    evidence_ids=(evidence_id,),
                    provenance=(provider_provenance, adapter_provenance),
                    place_matches=(match,),
                    supports=("phase5b.support.ocr_place",),
                )
            )
        return hypotheses, evidence

    @staticmethod
    def _place_sort_key(match: PlaceEvidenceSummary) -> tuple[object, ...]:
        return (
            -match.evidence_strength,
            -match.text_similarity,
            match.ambiguity_count,
            match.match_type,
            match.matched_entity.casefold(),
            match.country_code or "",
            match.center.latitude,
            match.center.longitude,
        )

    @staticmethod
    def _stable_hash(*parts: str) -> str:
        payload = "\x1f".join(parts).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _safe_reason(cls, reason: str) -> str:
        safe = _SAFE_REASON.sub("_", reason.casefold()).strip("_.-")
        return (safe or "unavailable")[:120]

    @classmethod
    def _partial_failure(cls, source: str, reason: str | None) -> str:
        return f"phase5b.{source}.{cls._safe_reason(reason or 'unavailable')}"
