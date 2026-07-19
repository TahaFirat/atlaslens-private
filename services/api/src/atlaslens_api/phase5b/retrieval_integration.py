from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from atlaslens_api.phase5b.models import EvidenceHypothesis
from atlaslens_api.providers.base import OutcomeStatus, ProviderDescriptor, ProviderOutcome
from atlaslens_api.retrieval.models import RetrievalHit
from atlaslens_api.schemas import (
    Evidence,
    GeoPoint,
    Provenance,
    ProviderRunDiagnostic,
    ReferenceIndexDiagnostic,
    RetrievalMatchSummary,
)


@dataclass(frozen=True, slots=True)
class Phase5BRetrievalAdaptation:
    hypotheses: tuple[EvidenceHypothesis, ...]
    evidence: tuple[Evidence, ...]
    provider: ProviderRunDiagnostic
    reference_index: ReferenceIndexDiagnostic
    partial_failures: tuple[str, ...]


class Phase5BRetrievalAdapter:
    version = "phase5b-retrieval-adapter-v1"

    def adapt(
        self,
        *,
        descriptor: ProviderDescriptor,
        outcome: ProviderOutcome[object] | None,
        hits: tuple[RetrievalHit, ...],
        duration_ms: int,
        diagnostics: dict[str, Any] | None,
    ) -> Phase5BRetrievalAdaptation:
        usable = (
            descriptor.available
            and outcome is not None
            and outcome.status == OutcomeStatus.SUCCEEDED
        )
        if usable and hits:
            status = "succeeded"
            reason = None
        elif usable:
            status = "abstained"
            reason = "no_matches"
        elif outcome is not None:
            status = outcome.status.value
            reason = outcome.failure.code if outcome.failure is not None else outcome.status.value
        else:
            status = "skipped"
            reason = descriptor.unavailable_reason_code or "unavailable"
        provider = ProviderRunDiagnostic(
            provider_id=descriptor.id,
            provider_type="visual_retrieval",
            status=status,
            duration_ms=max(0, duration_ms),
            reason_code=reason,
            device=None,
            offline=True,
        )
        reference_index = self._reference_index(diagnostics)
        if not usable:
            return Phase5BRetrievalAdaptation(
                hypotheses=(),
                evidence=(),
                provider=provider,
                reference_index=reference_index,
                partial_failures=(f"phase5b.retrieval.{reason or 'unavailable'}",),
            )

        provenance = Provenance(
            provider_id=descriptor.id,
            provider_kind=descriptor.kind,
            provider_version=descriptor.version,
            execution_boundary="local",
            model_name=descriptor.model_name,
            output_schema_version="phase5b-retrieval-v1",
        )
        adapter_provenance = Provenance(
            provider_id="phase5b-retrieval-adapter",
            provider_kind="evidence_adapter",
            provider_version=self.version,
            execution_boundary="local",
            output_schema_version="phase5b-retrieval-adapter-v1",
        )
        hypotheses: list[EvidenceHypothesis] = []
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for rank, hit in enumerate(hits[:100], 1):
            metadata = hit.metadata
            reference_id = metadata.asset_key or f"reference-{metadata.image_id}"
            stable = hashlib.sha256(
                f"{descriptor.id}\x1f{metadata.image_id}\x1f{metadata.hash}".encode()
            ).hexdigest()
            if stable in seen:
                continue
            seen.add(stable)
            evidence_id = f"evidence-retrieval-{stable[:24]}"
            similarity = max(-1.0, min(1.0, 1.0 - hit.distance))
            geographic_cluster = metadata.geographic_cell or (
                f"cell-{metadata.latitude:+.2f}-{metadata.longitude:+.2f}"
            )
            summary = RetrievalMatchSummary(
                reference_id=reference_id,
                provider=hit.provider.provider,
                source=metadata.source,
                distance=hit.distance,
                relative_similarity=similarity,
                center=GeoPoint(
                    latitude=metadata.latitude, longitude=metadata.longitude
                ),
                geographic_cluster=geographic_cluster,
                license=metadata.license,
                attribution=metadata.attribution,
                display_allowed=metadata.display_allowed,
            )
            evidence.append(
                Evidence(
                    id=evidence_id,
                    type="licensed_visual_retrieval",
                    label="evidence.licensed_visual_match",
                    display_value="evidence.licensed_visual_match_present",
                    confidence=None,
                    confidence_basis="phase5b.retrieval_similarity_is_not_probability",
                    source=descriptor.id,
                    sensitive=False,
                    provenance=provenance,
                )
            )
            coordinate_floor = (
                metadata.coordinate_uncertainty_m / 1000.0
                if metadata.coordinate_uncertainty_m is not None
                else 0.0
            )
            source_id = hashlib.sha256(metadata.source.encode()).hexdigest()[:20]
            hypotheses.append(
                EvidenceHypothesis(
                    id=f"phase5b-retrieval-{stable[:24]}",
                    provider_id=descriptor.id,
                    source_id=f"reference-source-{source_id}",
                    source_kind="visual_retrieval",
                    latitude=metadata.latitude,
                    longitude=metadata.longitude,
                    raw_score=max(0.0, similarity),
                    score_semantics="cosine_relative_similarity_not_probability",
                    uncertainty_radius_km=max(5.0, coordinate_floor),
                    original_rank=rank,
                    capture_family_id=metadata.capture_family_id,
                    content_hash=metadata.hash,
                    evidence_ids=(evidence_id,),
                    provenance=(provenance, adapter_provenance),
                    retrieval_matches=(summary,),
                    supports=("phase5b.support.licensed_visual_retrieval",),
                )
            )
        return Phase5BRetrievalAdaptation(
            hypotheses=tuple(hypotheses),
            evidence=tuple(evidence),
            provider=provider,
            reference_index=reference_index,
            partial_failures=(),
        )

    @staticmethod
    def _reference_index(diagnostics: dict[str, Any] | None) -> ReferenceIndexDiagnostic:
        if diagnostics is None:
            return ReferenceIndexDiagnostic(status="unavailable", image_count=0)
        raw_status = diagnostics.get("status")
        status = (
            "ready"
            if raw_status == "ready"
            else "disabled"
            if raw_status == "disabled"
            else "unavailable"
        )
        checksum = diagnostics.get("checksum")
        if not isinstance(checksum, str) or len(checksum) != 64:
            checksum = None
        index_id = diagnostics.get("index_id")
        return ReferenceIndexDiagnostic(
            status=status,
            index_id=index_id if isinstance(index_id, str) else None,
            embedding_provider=(
                value
                if isinstance((value := diagnostics.get("embedding_provider")), str)
                else None
            ),
            embedding_version=(
                value
                if isinstance((value := diagnostics.get("embedding_version")), str)
                else None
            ),
            dimension=(
                value
                if isinstance((value := diagnostics.get("dimension")), int) and value > 0
                else None
            ),
            image_count=(
                value
                if isinstance((value := diagnostics.get("index_size")), int) and value >= 0
                else 0
            ),
            checksum=checksum,
        )
