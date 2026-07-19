from __future__ import annotations

from atlaslens_api.repository import StoredAnalysis
from atlaslens_api.schemas import AnalysisHistoryItem


def history_item(stored: StoredAnalysis) -> AnalysisHistoryItem:
    analysis = stored.analysis
    provider_ids = sorted(
        {
            provenance.provider_id
            for candidate in analysis.candidates
            for provenance in candidate.provenance
        }
        | {evidence.provenance.provider_id for evidence in analysis.evidence}
    )
    primary = analysis.candidates[0] if analysis.candidates else None
    return AnalysisHistoryItem(
        id=analysis.id,
        created_at=analysis.created_at,
        expires_at=analysis.expires_at,
        status=analysis.status,
        analysis_mode=analysis.analysis_mode,
        result_classification=analysis.result_classification,
        image_sha256=analysis.image.sha256 if analysis.image is not None else None,
        image_width=analysis.image.width if analysis.image is not None else None,
        image_height=analysis.image.height if analysis.image is not None else None,
        provider_ids=provider_ids,
        primary_label=primary.label if primary is not None else None,
        candidate_count=len(analysis.candidates),
        evidence_count=len(analysis.evidence),
        runtime_ms=analysis.timings_ms.get("total"),
        warning_count=len(analysis.warnings),
        source_retained=stored.storage_key is not None,
    )
