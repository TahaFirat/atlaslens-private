from __future__ import annotations

import hashlib
from typing import Literal

from atlaslens_api.inference.models import InferenceResult
from atlaslens_api.phase6b.models import GeographicCandidate, GeographicProviderResult


def normalize_geoclip_result(result: InferenceResult) -> GeographicProviderResult:
    """Adapt the preserved canonical GeoCLIP result without changing its provider."""

    device: Literal["cuda", "cpu"] = (
        "cuda" if (result.device or "").casefold().startswith("cuda") else "cpu"
    )
    if result.status != "succeeded":
        reason = result.failure.code if result.failure is not None else "provider_abstained"
        status: Literal["skipped", "disabled", "failed"] = (
            "disabled"
            if result.status == "skipped" and reason == "disabled"
            else "skipped"
            if result.status in {"skipped", "abstained"}
            else "failed"
        )
        return GeographicProviderResult(
            provider=result.provider_id,
            model_id=result.model_name,
            model_revision=result.model_revision,
            source_family="mp16_family",
            status=status,
            device=device,
            duration_ms=result.runtime_ms,
            score_semantics="similarity",
            warnings=result.warnings,
            reason_code=reason,
            diagnostics={
                "provider_revision": result.provider_revision,
                "runtime_revision": result.runtime_revision,
                "calibration_state": result.calibration_state,
            },
        )
    candidates: list[GeographicCandidate] = []
    for candidate in result.candidates:
        stable = hashlib.sha256(
            (
                f"{result.provider_id}|{candidate.original_rank}|"
                f"{candidate.latitude:.7f}|{candidate.longitude:.7f}"
            ).encode()
        ).hexdigest()[:24]
        candidates.append(
            GeographicCandidate(
                candidate_id=f"geoclip-{stable}",
                latitude=candidate.latitude,
                longitude=candidate.longitude,
                raw_score=candidate.raw_score,
                provider_rank=candidate.rank,
                sample_support=1,
                metadata={
                    "original_rank": candidate.original_rank,
                    "limitations": ",".join(candidate.limitations)[:240],
                },
            )
        )
    return GeographicProviderResult(
        provider=result.provider_id,
        model_id=result.model_name,
        model_revision=result.model_revision,
        source_family="mp16_family",
        status="completed",
        device=device,
        duration_ms=result.runtime_ms,
        score_semantics="similarity",
        candidates=tuple(candidates),
        warnings=result.warnings,
        diagnostics={
            "provider_revision": result.provider_revision,
            "runtime_revision": result.runtime_revision,
            "normalization_method": result.normalization_method,
            "calibration_state": result.calibration_state,
        },
    )
