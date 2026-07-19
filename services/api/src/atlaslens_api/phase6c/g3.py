from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import asynccontextmanager
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.phase6c.fusion import (
    G3VerificationEvidence,
    Phase6CEvidenceCandidate,
)
from atlaslens_api.reranking.geo import geodesic_km

G3_MODEL_ID: Literal["Jia-py/G3-checkpoint"] = "Jia-py/G3-checkpoint"
G3_SOURCE_REVISION: Literal["b4e3acf7c0ac51221f21b7877fefb4826715c9e2"] = (
    "b4e3acf7c0ac51221f21b7877fefb4826715c9e2"
)
G3_MODEL_REVISION: Literal["12d886fc2a1e59b3b52821acee193084420409cc"] = (
    "12d886fc2a1e59b3b52821acee193084420409cc"
)
G3_RUNTIME_REVISION: Literal[
    "python3.9-torch2.1.1-torchvision0.16.1-cuda12"
] = "python3.9-torch2.1.1-torchvision0.16.1-cuda12"
G3_MAX_CANDIDATES = 96

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class G3CandidateInput(_FrozenModel):
    """A validated candidate owned by an upstream provider, never by G3."""

    candidate_id: str = Field(min_length=1, max_length=128, pattern=_SAFE_ID.pattern)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_km: float = Field(gt=0, le=20_050)
    provenance: str = Field(min_length=1, max_length=240)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    mode_id: str | None = Field(default=None, min_length=1, max_length=160)
    retrieval_cluster_id: str | None = Field(default=None, min_length=1, max_length=160)

    @field_validator("latitude", "longitude", "uncertainty_radius_km")
    @classmethod
    def finite_numbers(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("candidate coordinates and radius must be finite")
        return value


class G3WorkerCandidate(_FrozenModel):
    """Minimal bounded worker input; the worker cannot augment this coordinate set."""

    candidate_id: str = Field(min_length=1, max_length=128, pattern=_SAFE_ID.pattern)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    @field_validator("latitude", "longitude")
    @classmethod
    def finite_coordinates(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("G3 worker coordinates must be finite")
        return value


class G3WorkerScore(_FrozenModel):
    candidate_id: str = Field(min_length=1, max_length=128, pattern=_SAFE_ID.pattern)
    raw_score: float

    @field_validator("raw_score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("G3 scores must be finite")
        return value


class G3WorkerResponse(_FrozenModel):
    provider: Literal["g3"] = "g3"
    model_id: Literal["Jia-py/G3-checkpoint"] = G3_MODEL_ID
    source_revision: Literal["b4e3acf7c0ac51221f21b7877fefb4826715c9e2"] = (
        G3_SOURCE_REVISION
    )
    model_revision: Literal["12d886fc2a1e59b3b52821acee193084420409cc"] = (
        G3_MODEL_REVISION
    )
    runtime_revision: Literal[
        "python3.9-torch2.1.1-torchvision0.16.1-cuda12"
    ] = G3_RUNTIME_REVISION
    device: Literal["cuda"] = "cuda"
    scores: tuple[G3WorkerScore, ...] = Field(min_length=1, max_length=G3_MAX_CANDIDATES)
    inference_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def unique_candidate_ids(self) -> G3WorkerResponse:
        identifiers = [item.candidate_id for item in self.scores]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("G3 worker candidate identifiers must be unique")
        return self


class G3WorkerCapability(_FrozenModel):
    provider: Literal["g3"] = "g3"
    model_id: Literal["Jia-py/G3-checkpoint"] = G3_MODEL_ID
    source_revision: Literal["b4e3acf7c0ac51221f21b7877fefb4826715c9e2"] = (
        G3_SOURCE_REVISION
    )
    model_revision: Literal["12d886fc2a1e59b3b52821acee193084420409cc"] = (
        G3_MODEL_REVISION
    )
    runtime_revision: Literal[
        "python3.9-torch2.1.1-torchvision0.16.1-cuda12"
    ] = G3_RUNTIME_REVISION
    device: Literal["cuda"] = "cuda"
    cuda_available: bool
    artifacts_available: bool
    load_verified: bool
    real_inference_verified: bool
    ready: bool

    @model_validator(mode="after")
    def attested_readiness(self) -> G3WorkerCapability:
        attested = (
            self.cuda_available
            and self.artifacts_available
            and self.load_verified
            and self.real_inference_verified
        )
        if self.ready != attested:
            raise ValueError("G3 readiness requires CUDA, artifacts, load and real inference proof")
        return self


class G3Worker(Protocol):
    """Optional isolated CUDA worker; no implementation or artifact download is implied."""

    async def capability(
        self,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> G3WorkerCapability: ...

    async def score_candidates(
        self,
        image_bytes: bytes,
        candidates: tuple[G3WorkerCandidate, ...],
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> G3WorkerResponse: ...

    async def close(self) -> None: ...


type G3TriggerReason = Literal["weak_margin", "provider_disagreement", "diagnostics"]


class G3VerificationConfig(_FrozenModel):
    timeout_seconds: float = Field(default=45.0, gt=0, le=180)
    status_timeout_seconds: float = Field(default=2.0, gt=0, le=10)
    max_input_bytes: int = Field(default=20 * 1024 * 1024, gt=0, le=32 * 1024 * 1024)
    max_candidates: int = Field(default=G3_MAX_CANDIDATES, ge=1, le=G3_MAX_CANDIDATES)
    geodesic_dedup_radius_km: float = Field(default=1.0, ge=0, le=25)
    max_concurrent_verifications: int = Field(default=1, ge=1, le=2)


type G3ProviderState = Literal["disabled", "unavailable", "ready"]


class G3ProviderCapability(_FrozenModel):
    provider: Literal["g3_verifier"] = "g3_verifier"
    state: G3ProviderState
    available: bool
    reason_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[a-z0-9][a-z0-9._-]+$",
    )
    model_id: Literal["Jia-py/G3-checkpoint"] = G3_MODEL_ID
    source_revision: Literal["b4e3acf7c0ac51221f21b7877fefb4826715c9e2"] = (
        G3_SOURCE_REVISION
    )
    model_revision: Literal["12d886fc2a1e59b3b52821acee193084420409cc"] = (
        G3_MODEL_REVISION
    )
    runtime_revision: Literal[
        "python3.9-torch2.1.1-torchvision0.16.1-cuda12"
    ] = G3_RUNTIME_REVISION
    required_device: Literal["cuda"] = "cuda"
    load_verified: bool = False
    real_inference_verified: bool = False
    confidence: None = None
    confidence_semantics: Literal["uncalibrated_unavailable"] = "uncalibrated_unavailable"
    independence_semantics: Literal["verification_only_correlated_visual_gps_evidence"] = (
        "verification_only_correlated_visual_gps_evidence"
    )

    @model_validator(mode="after")
    def coherent_state(self) -> G3ProviderCapability:
        if self.available != (self.state == "ready"):
            raise ValueError("only a ready G3 verifier may be available")
        if self.available and (
            self.reason_code is not None
            or not self.load_verified
            or not self.real_inference_verified
        ):
            raise ValueError("ready G3 verifier requires load and real-inference proof")
        if not self.available and self.reason_code is None:
            raise ValueError("unavailable G3 verifier requires a safe reason")
        return self


class G3VerifiedCandidate(_FrozenModel):
    candidate_id: str = Field(min_length=1, max_length=128, pattern=_SAFE_ID.pattern)
    verification_target_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_km: float = Field(gt=0, le=20_050)
    provenance: str = Field(min_length=1, max_length=240)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    mode_id: str | None = Field(default=None, min_length=1, max_length=160)
    retrieval_cluster_id: str | None = Field(default=None, min_length=1, max_length=160)
    rank: int = Field(ge=1, le=G3_MAX_CANDIDATES)
    raw_score: float
    score_semantics: Literal["raw_g3_similarity_not_confidence"] = (
        "raw_g3_similarity_not_confidence"
    )
    confidence: None = None
    calibrated: Literal[False] = False

    @field_validator("latitude", "longitude", "uncertainty_radius_km", "raw_score")
    @classmethod
    def finite_numbers(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("G3 verification numbers must be finite")
        return value

    @model_validator(mode="after")
    def representative_is_a_target(self) -> G3VerifiedCandidate:
        if self.candidate_id not in self.verification_target_ids:
            raise ValueError("G3 representative must be an existing verification target")
        if len(set(self.verification_target_ids)) != len(self.verification_target_ids):
            raise ValueError("G3 verification target identifiers must be unique")
        return self


type G3VerificationState = Literal[
    "completed", "skipped", "disabled", "unavailable", "failed", "timeout"
]


class G3VerificationResult(_FrozenModel):
    provider: Literal["g3_verifier"] = "g3_verifier"
    status: G3VerificationState
    reason_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[a-z0-9][a-z0-9._-]+$",
    )
    model_id: Literal["Jia-py/G3-checkpoint"] = G3_MODEL_ID
    source_revision: Literal["b4e3acf7c0ac51221f21b7877fefb4826715c9e2"] = (
        G3_SOURCE_REVISION
    )
    model_revision: Literal["12d886fc2a1e59b3b52821acee193084420409cc"] = (
        G3_MODEL_REVISION
    )
    runtime_revision: Literal[
        "python3.9-torch2.1.1-torchvision0.16.1-cuda12"
    ] = G3_RUNTIME_REVISION
    required_device: Literal["cuda"] = "cuda"
    duration_ms: int = Field(ge=0)
    trigger_reasons: tuple[G3TriggerReason, ...] = Field(default=(), max_length=3)
    input_candidate_count: int = Field(ge=0, le=G3_MAX_CANDIDATES)
    worker_candidate_count: int = Field(ge=0, le=G3_MAX_CANDIDATES)
    candidates: tuple[G3VerifiedCandidate, ...] = Field(
        default=(), max_length=G3_MAX_CANDIDATES
    )
    score_semantics: Literal["raw_g3_similarity_not_confidence"] = (
        "raw_g3_similarity_not_confidence"
    )
    confidence: None = None
    calibrated: Literal[False] = False
    independence_semantics: Literal["verification_only_correlated_visual_gps_evidence"] = (
        "verification_only_correlated_visual_gps_evidence"
    )

    @model_validator(mode="after")
    def coherent_result(self) -> G3VerificationResult:
        if len(set(self.trigger_reasons)) != len(self.trigger_reasons):
            raise ValueError("G3 trigger reasons must be unique")
        if self.status == "completed":
            if not self.candidates or self.reason_code is not None:
                raise ValueError("completed G3 verification requires scored candidates")
            if [item.rank for item in self.candidates] != list(
                range(1, len(self.candidates) + 1)
            ):
                raise ValueError("G3 verification ranks must be contiguous")
            if self.worker_candidate_count != len(self.candidates):
                raise ValueError("G3 worker and output candidate counts must match")
            covered = {
                target
                for candidate in self.candidates
                for target in candidate.verification_target_ids
            }
            if len(covered) != self.input_candidate_count:
                raise ValueError("G3 verification must account for every input candidate")
        elif self.candidates or self.reason_code is None:
            raise ValueError("non-completed G3 verification requires a reason and no scores")
        return self

    def to_evidence(self) -> G3VerificationEvidence:
        status_map: dict[G3VerificationState, Literal[
            "completed", "failed", "timeout", "skipped", "disabled"
        ]] = {
            "completed": "completed",
            "skipped": "skipped",
            "disabled": "disabled",
            "unavailable": "skipped",
            "failed": "failed",
            "timeout": "timeout",
        }
        evidence_candidates = tuple(
            Phase6CEvidenceCandidate(
                candidate_id=item.candidate_id,
                latitude=item.latitude,
                longitude=item.longitude,
                raw_value=item.raw_score,
                provider_rank=item.rank,
                sample_support=len(item.verification_target_ids),
                uncertainty_radius_km=item.uncertainty_radius_km,
                provenance=item.provenance,
                country_code=item.country_code,
                mode_id=item.mode_id,
                retrieval_cluster_id=item.retrieval_cluster_id,
                verification_target_ids=item.verification_target_ids,
            )
            for item in self.candidates
        )
        return G3VerificationEvidence(
            provider="g3",
            model_id=self.model_id,
            model_revision=self.model_revision,
            status=status_map[self.status],
            duration_ms=self.duration_ms,
            candidates=evidence_candidates,
            reason_code=self.reason_code,
        )


class G3CandidateVerifier:
    """Conditionally rescore only supplied coordinates with an optional CUDA worker."""

    def __init__(
        self,
        *,
        enabled: bool,
        worker: G3Worker | None,
        cuda_available: bool,
        config: G3VerificationConfig | None = None,
    ) -> None:
        self._enabled = enabled
        self._worker = worker
        self._cuda_available = cuda_available
        self._config = config or G3VerificationConfig()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_verifications)

    async def status(
        self,
        *,
        cancellation: asyncio.Event | None = None,
    ) -> G3ProviderCapability:
        if not self._enabled:
            return self._capability("disabled", "disabled")
        if not self._cuda_available:
            return self._capability("unavailable", "cuda_required")
        if self._worker is None:
            return self._capability("unavailable", "isolated_worker_not_installed")
        stopped = cancellation or asyncio.Event()
        if stopped.is_set():
            return self._capability("unavailable", "cancelled")
        try:
            async with asyncio.timeout(self._config.status_timeout_seconds):
                health = G3WorkerCapability.model_validate(
                    await _await_or_cancel(
                        self._worker.capability(
                            timeout_seconds=self._config.status_timeout_seconds,
                            cancellation=stopped,
                        ),
                        stopped,
                    )
                )
        except TimeoutError:
            return self._capability("unavailable", "worker_timeout")
        except _G3Cancelled:
            return self._capability("unavailable", "cancelled")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - public capability exposes only safe states
            return self._capability("unavailable", "worker_unavailable")
        if not health.ready:
            return self._capability(
                "unavailable",
                _health_reason(health),
                health=health,
            )
        return self._capability("ready", None, health=health)

    async def verify(
        self,
        image_bytes: bytes,
        candidates: Sequence[G3CandidateInput],
        *,
        trigger_reasons: Sequence[G3TriggerReason],
        cancellation: asyncio.Event | None = None,
    ) -> G3VerificationResult:
        started = time.monotonic()
        validated = _validate_candidates(candidates, maximum=self._config.max_candidates)
        triggers = _validate_triggers(trigger_reasons)
        if not self._enabled:
            return self._empty_result("disabled", "disabled", started, validated, triggers)
        if not validated:
            return self._empty_result("skipped", "no_candidates", started, validated, triggers)
        if not triggers:
            return self._empty_result(
                "skipped", "conditional_trigger_not_met", started, validated, triggers
            )
        if not self._cuda_available:
            return self._empty_result(
                "unavailable", "cuda_required", started, validated, triggers
            )
        if self._worker is None:
            return self._empty_result(
                "unavailable",
                "isolated_worker_not_installed",
                started,
                validated,
                triggers,
            )
        if type(image_bytes) is not bytes or not (
            0 < len(image_bytes) <= self._config.max_input_bytes
        ):
            return self._empty_result(
                "failed", "unsupported_input", started, validated, triggers
            )
        stopped = cancellation or asyncio.Event()
        if stopped.is_set():
            return self._empty_result("skipped", "cancelled", started, validated, triggers)

        groups = _deduplicate_candidates(
            validated,
            radius_km=self._config.geodesic_dedup_radius_km,
        )
        worker_candidates = tuple(
            G3WorkerCandidate(
                candidate_id=group[0].candidate_id,
                latitude=group[0].latitude,
                longitude=group[0].longitude,
            )
            for group in groups
        )
        worker = self._worker
        stage = "capability"
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                async with _bounded_semaphore(self._semaphore, stopped):
                    health = G3WorkerCapability.model_validate(
                        await _await_or_cancel(
                            worker.capability(
                                timeout_seconds=self._config.timeout_seconds,
                                cancellation=stopped,
                            ),
                            stopped,
                        )
                    )
                    if not health.ready:
                        return self._empty_result(
                            "unavailable",
                            _health_reason(health),
                            started,
                            validated,
                            triggers,
                        )
                    stage = "inference"
                    response = G3WorkerResponse.model_validate(
                        await _await_or_cancel(
                            worker.score_candidates(
                                image_bytes,
                                worker_candidates,
                                timeout_seconds=self._config.timeout_seconds,
                                cancellation=stopped,
                            ),
                            stopped,
                        )
                    )
        except TimeoutError:
            return self._empty_result(
                "timeout", f"{stage}_timeout", started, validated, triggers
            )
        except _G3Cancelled:
            return self._empty_result("skipped", "cancelled", started, validated, triggers)
        except asyncio.CancelledError:
            raise
        except (ValueError, TypeError):
            return self._empty_result(
                "failed", "invalid_worker_output", started, validated, triggers
            )
        except Exception:  # noqa: BLE001 - provider boundary emits only safe reason codes
            status: Literal["unavailable", "failed"] = (
                "unavailable" if stage == "capability" else "failed"
            )
            reason = "worker_unavailable" if stage == "capability" else "inference_failed"
            return self._empty_result(status, reason, started, validated, triggers)

        expected_ids = {item.candidate_id for item in worker_candidates}
        returned_ids = {item.candidate_id for item in response.scores}
        if expected_ids != returned_ids or len(response.scores) != len(worker_candidates):
            return self._empty_result(
                "failed", "candidate_id_mismatch", started, validated, triggers
            )
        groups_by_id = {group[0].candidate_id: group for group in groups}
        ordered_scores = sorted(
            response.scores,
            key=lambda item: (-item.raw_score, item.candidate_id),
        )
        scored = tuple(
            _verified_candidate(
                groups_by_id[item.candidate_id],
                raw_score=item.raw_score,
                rank=rank,
            )
            for rank, item in enumerate(ordered_scores, start=1)
        )
        return G3VerificationResult(
            status="completed",
            duration_ms=_elapsed_ms(started),
            trigger_reasons=triggers,
            input_candidate_count=len(validated),
            worker_candidate_count=len(worker_candidates),
            candidates=scored,
        )

    async def close(self) -> None:
        if self._worker is not None:
            await self._worker.close()

    def _capability(
        self,
        state: G3ProviderState,
        reason_code: str | None,
        *,
        health: G3WorkerCapability | None = None,
    ) -> G3ProviderCapability:
        return G3ProviderCapability(
            state=state,
            available=state == "ready",
            reason_code=reason_code,
            load_verified=health.load_verified if health is not None else False,
            real_inference_verified=(
                health.real_inference_verified if health is not None else False
            ),
        )

    def _empty_result(
        self,
        status: Literal["skipped", "disabled", "unavailable", "failed", "timeout"],
        reason_code: str,
        started: float,
        candidates: tuple[G3CandidateInput, ...],
        trigger_reasons: tuple[G3TriggerReason, ...],
    ) -> G3VerificationResult:
        return G3VerificationResult(
            status=status,
            reason_code=reason_code,
            duration_ms=_elapsed_ms(started),
            trigger_reasons=trigger_reasons,
            input_candidate_count=len(candidates),
            worker_candidate_count=0,
        )


class _G3Cancelled(Exception):
    pass


async def _await_or_cancel[T](
    awaitable: Awaitable[T],
    cancellation: asyncio.Event,
) -> T:
    operation = asyncio.ensure_future(awaitable)
    cancelled = asyncio.create_task(cancellation.wait())
    try:
        done, _ = await asyncio.wait(
            {operation, cancelled},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancelled in done and cancellation.is_set():
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            raise _G3Cancelled
        return await operation
    finally:
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)


@asynccontextmanager
async def _bounded_semaphore(
    semaphore: asyncio.Semaphore,
    cancellation: asyncio.Event,
) -> AsyncIterator[None]:
    acquired = False
    try:
        await _await_or_cancel(semaphore.acquire(), cancellation)
        acquired = True
        if cancellation.is_set():
            raise _G3Cancelled
        yield
    finally:
        if acquired:
            semaphore.release()


def _validate_candidates(
    candidates: Sequence[G3CandidateInput],
    *,
    maximum: int,
) -> tuple[G3CandidateInput, ...]:
    if not isinstance(candidates, Sequence) or isinstance(candidates, str | bytes):
        raise ValueError("G3 candidates must be a bounded sequence")
    if len(candidates) > maximum:
        raise ValueError(f"G3 accepts at most {maximum} existing candidates")
    validated = tuple(G3CandidateInput.model_validate(item) for item in candidates)
    identifiers = [item.candidate_id for item in validated]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("G3 candidate identifiers must be unique")
    return validated


def _validate_triggers(
    values: Sequence[G3TriggerReason],
) -> tuple[G3TriggerReason, ...]:
    allowed = {"weak_margin", "provider_disagreement", "diagnostics"}
    if len(values) > len(allowed) or any(value not in allowed for value in values):
        raise ValueError("unsupported G3 trigger reason")
    triggers = tuple(values)
    if len(set(triggers)) != len(triggers):
        raise ValueError("G3 trigger reasons must be unique")
    return triggers


def _deduplicate_candidates(
    candidates: tuple[G3CandidateInput, ...],
    *,
    radius_km: float,
) -> tuple[tuple[G3CandidateInput, ...], ...]:
    groups: list[list[G3CandidateInput]] = []
    for candidate in candidates:
        point = (candidate.latitude, candidate.longitude)
        for group in groups:
            representative = group[0]
            if len(group) < 32 and geodesic_km(
                point,
                (representative.latitude, representative.longitude),
            ) <= radius_km:
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    return tuple(tuple(group) for group in groups)


def _verified_candidate(
    group: tuple[G3CandidateInput, ...],
    *,
    raw_score: float,
    rank: int,
) -> G3VerifiedCandidate:
    representative = group[0]
    radius = max(
        geodesic_km(
            (representative.latitude, representative.longitude),
            (item.latitude, item.longitude),
        )
        + item.uncertainty_radius_km
        for item in group
    )
    return G3VerifiedCandidate(
        candidate_id=representative.candidate_id,
        verification_target_ids=tuple(item.candidate_id for item in group),
        latitude=representative.latitude,
        longitude=representative.longitude,
        uncertainty_radius_km=radius,
        provenance=representative.provenance,
        country_code=representative.country_code,
        mode_id=representative.mode_id,
        retrieval_cluster_id=representative.retrieval_cluster_id,
        rank=rank,
        raw_score=raw_score,
    )


def _health_reason(health: G3WorkerCapability) -> str:
    if not health.cuda_available:
        return "cuda_required"
    if not health.artifacts_available:
        return "model_artifact_not_verified"
    if not health.load_verified:
        return "model_load_not_verified"
    if not health.real_inference_verified:
        return "real_inference_not_verified"
    return "worker_unavailable"


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
