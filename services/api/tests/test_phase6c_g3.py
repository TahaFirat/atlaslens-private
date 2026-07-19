from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from atlaslens_api.phase6c.g3 import (
    G3_MAX_CANDIDATES,
    G3CandidateInput,
    G3CandidateVerifier,
    G3VerificationConfig,
    G3WorkerCandidate,
    G3WorkerCapability,
    G3WorkerResponse,
    G3WorkerScore,
)


def candidate(
    candidate_id: str,
    latitude: float,
    longitude: float,
    *,
    radius: float = 12.0,
    provenance: str = "unit-test-upstream-provider",
) -> G3CandidateInput:
    return G3CandidateInput(
        candidate_id=candidate_id,
        latitude=latitude,
        longitude=longitude,
        uncertainty_radius_km=radius,
        provenance=provenance,
    )


def ready_capability(**updates: bool) -> G3WorkerCapability:
    values = {
        "cuda_available": True,
        "artifacts_available": True,
        "load_verified": True,
        "real_inference_verified": True,
        "ready": True,
    }
    values.update(updates)
    return G3WorkerCapability(**values)


class StubWorker:
    def __init__(
        self,
        *,
        capability: G3WorkerCapability | None = None,
        scores: dict[str, float] | None = None,
        delay: float = 0,
        fail: bool = False,
        mismatched_id: bool = False,
        nonfinite_score: bool = False,
    ) -> None:
        self.worker_capability = capability or ready_capability()
        self.scores = scores or {}
        self.delay = delay
        self.fail = fail
        self.mismatched_id = mismatched_id
        self.nonfinite_score = nonfinite_score
        self.calls: list[tuple[G3WorkerCandidate, ...]] = []
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def capability(
        self,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> G3WorkerCapability:
        return self.worker_capability

    async def score_candidates(
        self,
        image_bytes: bytes,
        candidates: tuple[G3WorkerCandidate, ...],
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> G3WorkerResponse:
        assert image_bytes == b"private-image"
        self.calls.append(candidates)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("private worker detail must not escape")
            if self.nonfinite_score:
                return cast(
                    G3WorkerResponse,
                    {
                        "provider": "g3",
                        "scores": [
                            {"candidate_id": candidates[0].candidate_id, "raw_score": float("nan")}
                        ],
                        "inference_ms": 1,
                    },
                )
            scores = tuple(
                G3WorkerScore(
                    candidate_id=(
                        "unexpected-candidate"
                        if self.mismatched_id and index == 0
                        else item.candidate_id
                    ),
                    raw_score=self.scores.get(item.candidate_id, float(-index)),
                )
                for index, item in enumerate(candidates)
            )
            return G3WorkerResponse(scores=scores, inference_ms=2)
        finally:
            self.active -= 1

    async def close(self) -> None:
        self.closed = True


def verifier(
    worker: StubWorker | None,
    **updates: object,
) -> G3CandidateVerifier:
    values: dict[str, object] = {
        "enabled": True,
        "worker": worker,
        "cuda_available": True,
    }
    values.update(updates)
    return G3CandidateVerifier(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_real_candidate_input_is_rescored_and_converted_to_fusion_evidence() -> None:
    inputs = (
        candidate("geoclip:mode-1", 48.8566, 2.3522, provenance="geoclip-global-v1"),
        candidate("megaloc:cluster-2", 51.5074, -0.1278, provenance="megaloc-index-v1"),
        candidate("osv:direct-3", 41.0082, 28.9784, provenance="osv5m-baseline"),
    )
    worker = StubWorker(
        scores={
            "geoclip:mode-1": 0.2,
            "megaloc:cluster-2": 2.4,
            "osv:direct-3": -0.3,
        }
    )
    result = await verifier(worker).verify(
        b"private-image",
        inputs,
        trigger_reasons=("provider_disagreement",),
    )

    assert result.status == "completed"
    assert [item.candidate_id for item in result.candidates] == [
        "megaloc:cluster-2",
        "geoclip:mode-1",
        "osv:direct-3",
    ]
    assert [item.raw_score for item in result.candidates] == [2.4, 0.2, -0.3]
    assert result.score_semantics == "raw_g3_similarity_not_confidence"
    assert result.confidence is None and result.calibrated is False
    evidence = result.to_evidence()
    assert evidence.status == "completed"
    assert evidence.candidates[0].candidate_id == "megaloc:cluster-2"
    assert evidence.candidates[0].verification_target_ids == ("megaloc:cluster-2",)
    assert evidence.candidates[0].provenance == "megaloc-index-v1"
    assert evidence.candidates[0].uncertainty_radius_km == 12
    assert evidence.candidates[0].confidence is None
    assert evidence.score_semantics == "verification_similarity_not_confidence"


@pytest.mark.asyncio
async def test_bounded_list_and_coordinate_validation_reject_bad_input() -> None:
    worker = StubWorker()
    too_many = tuple(
        candidate(f"candidate-{index}", 0, 0) for index in range(G3_MAX_CANDIDATES + 1)
    )
    with pytest.raises(ValueError, match="at most 96"):
        await verifier(worker).verify(
            b"private-image",
            too_many,
            trigger_reasons=("diagnostics",),
        )
    with pytest.raises(ValidationError):
        candidate("bad-latitude", 91, 0)
    with pytest.raises(ValidationError):
        candidate("bad-longitude", 0, float("nan"))
    with pytest.raises(ValidationError):
        candidate("bad-radius", 0, 0, radius=0)
    assert worker.calls == []


@pytest.mark.asyncio
async def test_geodesic_dedup_preserves_targets_without_generating_coordinates() -> None:
    inputs = (
        candidate("source-a", 48.85660, 2.35220, radius=3, provenance="source-a-proof"),
        candidate("source-b", 48.85662, 2.35222, radius=5, provenance="source-b-proof"),
        candidate("source-c", 51.50740, -0.12780, radius=7, provenance="source-c-proof"),
    )
    worker = StubWorker(scores={"source-a": 0.8, "source-c": 0.4})
    result = await verifier(worker).verify(
        b"private-image",
        inputs,
        trigger_reasons=("weak_margin",),
    )

    assert result.status == "completed"
    assert len(worker.calls[0]) == 2
    assert result.worker_candidate_count == 2
    assert result.candidates[0].verification_target_ids == ("source-a", "source-b")
    assert result.candidates[0].uncertainty_radius_km >= 5
    input_coordinates = {(item.latitude, item.longitude) for item in inputs}
    output_coordinates = {(item.latitude, item.longitude) for item in result.candidates}
    worker_coordinates = {(item.latitude, item.longitude) for item in worker.calls[0]}
    assert output_coordinates <= input_coordinates
    assert worker_coordinates <= input_coordinates
    assert {
        target for item in result.candidates for target in item.verification_target_ids
    } == {item.candidate_id for item in inputs}


@pytest.mark.asyncio
async def test_optional_runtime_reports_disabled_and_unavailable_states_honestly() -> None:
    one = (candidate("candidate-one", 48.8566, 2.3522),)
    disabled = await verifier(None, enabled=False, cuda_available=False).verify(
        b"private-image", one, trigger_reasons=("diagnostics",)
    )
    missing = await verifier(None).verify(
        b"private-image", one, trigger_reasons=("diagnostics",)
    )
    no_cuda_worker = StubWorker()
    no_cuda = await verifier(no_cuda_worker, cuda_available=False).verify(
        b"private-image", one, trigger_reasons=("diagnostics",)
    )
    unattested_worker = StubWorker(
        capability=ready_capability(real_inference_verified=False, ready=False)
    )
    unattested = await verifier(unattested_worker).verify(
        b"private-image", one, trigger_reasons=("diagnostics",)
    )
    untriggered = await verifier(StubWorker()).verify(
        b"private-image", one, trigger_reasons=()
    )

    assert (disabled.status, disabled.reason_code) == ("disabled", "disabled")
    assert (missing.status, missing.reason_code) == (
        "unavailable",
        "isolated_worker_not_installed",
    )
    assert (no_cuda.status, no_cuda.reason_code) == ("unavailable", "cuda_required")
    assert no_cuda_worker.calls == []
    assert (unattested.status, unattested.reason_code) == (
        "unavailable",
        "real_inference_not_verified",
    )
    assert (untriggered.status, untriggered.reason_code) == (
        "skipped",
        "conditional_trigger_not_met",
    )
    assert missing.to_evidence().status == "skipped"


@pytest.mark.asyncio
async def test_worker_failure_timeout_and_malformed_output_return_no_fake_scores() -> None:
    one = (candidate("candidate-one", 48.8566, 2.3522),)
    failed = await verifier(StubWorker(fail=True)).verify(
        b"private-image", one, trigger_reasons=("diagnostics",)
    )
    timed_out = await verifier(
        StubWorker(delay=0.05),
        config=G3VerificationConfig(timeout_seconds=0.01),
    ).verify(b"private-image", one, trigger_reasons=("diagnostics",))
    mismatched = await verifier(StubWorker(mismatched_id=True)).verify(
        b"private-image", one, trigger_reasons=("diagnostics",)
    )
    nonfinite = await verifier(StubWorker(nonfinite_score=True)).verify(
        b"private-image", one, trigger_reasons=("diagnostics",)
    )

    assert (failed.status, failed.reason_code, failed.candidates) == (
        "failed",
        "inference_failed",
        (),
    )
    assert timed_out.status == "timeout"
    assert timed_out.reason_code in {"capability_timeout", "inference_timeout"}
    assert timed_out.candidates == ()
    assert (mismatched.status, mismatched.reason_code, mismatched.candidates) == (
        "failed",
        "candidate_id_mismatch",
        (),
    )
    assert (nonfinite.status, nonfinite.reason_code, nonfinite.candidates) == (
        "failed",
        "invalid_worker_output",
        (),
    )


@pytest.mark.asyncio
async def test_timeout_cancellation_and_semaphore_bound_optional_worker() -> None:
    one = (candidate("candidate-one", 48.8566, 2.3522),)
    worker = StubWorker(delay=0.03)
    instance = verifier(
        worker,
        config=G3VerificationConfig(max_concurrent_verifications=1),
    )
    left, right = await asyncio.gather(
        instance.verify(b"private-image", one, trigger_reasons=("diagnostics",)),
        instance.verify(b"private-image", one, trigger_reasons=("diagnostics",)),
    )
    assert left.status == right.status == "completed"
    assert worker.max_active == 1

    blocked = asyncio.create_task(
        instance.verify(b"private-image", one, trigger_reasons=("diagnostics",))
    )
    await asyncio.sleep(0.005)
    cancellation = asyncio.Event()
    waiting = asyncio.create_task(
        instance.verify(
            b"private-image",
            one,
            trigger_reasons=("diagnostics",),
            cancellation=cancellation,
        )
    )
    await asyncio.sleep(0.005)
    cancellation.set()
    cancelled = await waiting
    completed = await blocked
    assert (cancelled.status, cancelled.reason_code) == ("skipped", "cancelled")
    assert completed.status == "completed"


@pytest.mark.asyncio
async def test_status_requires_pinned_cuda_real_inference_attestation() -> None:
    unavailable = await verifier(None).status()
    pending = await verifier(
        StubWorker(capability=ready_capability(load_verified=False, ready=False))
    ).status()
    ready = await verifier(StubWorker()).status()
    assert (unavailable.state, unavailable.available) == ("unavailable", False)
    assert pending.reason_code == "model_load_not_verified"
    assert (ready.state, ready.available) == ("ready", True)
    assert ready.required_device == "cuda"
    assert ready.independence_semantics == "verification_only_correlated_visual_gps_evidence"


def test_production_seam_has_no_target_logic_or_coordinate_generation_api() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "atlaslens_api"
        / "phase6c"
        / "g3.py"
    )
    source = path.read_text(encoding="utf-8").casefold()
    assert not any(term in source for term in ("kayseri", "erciyes", "talas", "4cedcf22c000"))
    assert "generated_coordinate" not in source
    worker_fields = set(G3WorkerResponse.model_fields)
    assert "latitude" not in worker_fields and "longitude" not in worker_fields
