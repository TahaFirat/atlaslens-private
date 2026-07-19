from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from pydantic import Field, ValidationError, field_validator, model_validator

from atlaslens_api.inference.models import (
    CalibrationState,
    InferenceCandidate,
    InferenceFailure,
    InferenceModel,
    InferenceProvenance,
    InferenceProviderMode,
    InferenceProviderStatus,
    InferenceRequest,
    InferenceResult,
    failure_result,
)
from atlaslens_api.providers.base import (
    GlobalGeolocationProvider,
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    InvocationContext,
    OutcomeStatus,
    ProviderDescriptor,
    ProviderOutcome,
)

_DEVELOPMENT_PROVIDER_ID = "development-mock-geolocation"
_CUSTOM_PROVIDER_ID = "custom-trained-geolocation"
_FIXTURE_SCHEMA = "atlaslens-development-inference-fixture-v1"
_MAX_FIXTURE_BYTES = 256 * 1024


class GeoCLIPInferenceAdapter:
    """Canonical adapter over the preserved GlobalGeolocationProvider contract."""

    classification: Literal["real"] = "real"

    def __init__(
        self,
        provider: GlobalGeolocationProvider,
        *,
        mode: InferenceProviderMode = "primary",
    ) -> None:
        self._provider = provider
        self._mode = mode

    @property
    def provider_id(self) -> str:
        return self._provider.descriptor.id

    @property
    def mode(self) -> InferenceProviderMode:
        return self._mode

    def status(self) -> InferenceProviderStatus:
        legacy = self._provider.status()
        descriptor = self._provider.descriptor
        reason_code: str | None
        if self._mode == "disabled":
            operational_status = "disabled"
            available = False
            reason_code = "disabled"
        else:
            operational_status = legacy.status
            available = legacy.status == "ready"
            reason_code = legacy.reason_code or {
                "not_installed": "model_not_installed",
                "disabled": "disabled",
                "unavailable": "unavailable",
                "incomplete": "weights_incomplete",
                "failed": "internal_provider_error",
            }.get(legacy.status)
        return InferenceProviderStatus(
            provider_id=descriptor.id,
            provider_type=descriptor.kind,
            provider_revision=descriptor.version,
            mode=self._mode,
            available=available,
            status=operational_status,
            classification="real",
            model_name=legacy.model_name,
            model_revision=legacy.model_revision,
            runtime_revision=descriptor.version,
            device=legacy.device,
            calibration_state=legacy.calibration_state,
            reason_code=reason_code,
        )

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        provider_status = self.status()
        if self._mode == "disabled":
            return failure_result(
                provider_status,
                outcome_status="skipped",
                code="disabled",
                retryable=False,
            )
        if request.image_handle is None:
            return failure_result(
                provider_status,
                outcome_status="failed",
                code="unsupported_input",
                retryable=False,
            )
        if request.cancellation.is_set():
            raise asyncio.CancelledError
        context = InvocationContext(
            analysis_id=UUID(int=0),
            request_id=request.trace_id,
            mode=request.analysis_mode,
            cloud_consent=False,
            deadline=request.deadline,
            cancellation=request.cancellation,
        )
        started = time.monotonic()
        try:
            outcome = await self._provider.predict(request.image_handle, context)
        except Exception:
            return failure_result(
                provider_status,
                outcome_status="failed",
                code="internal_provider_error",
                retryable=False,
                runtime_ms=_elapsed_ms(started),
            )
        if outcome.status != OutcomeStatus.SUCCEEDED or outcome.value is None:
            if outcome.status == OutcomeStatus.ABSTAINED:
                return _abstained_result(provider_status, runtime_ms=_elapsed_ms(started))
            legacy_failure = outcome.failure
            return failure_result(
                provider_status,
                outcome_status=("skipped" if outcome.status == OutcomeStatus.SKIPPED else "failed"),
                code=legacy_failure.code if legacy_failure is not None else "unavailable",
                retryable=legacy_failure.retryable if legacy_failure is not None else False,
                runtime_ms=(
                    legacy_failure.duration_ms
                    if legacy_failure is not None
                    else _elapsed_ms(started)
                ),
                subreason_code=(
                    legacy_failure.subreason_code if legacy_failure is not None else None
                ),
            )
        value = outcome.value
        descriptor = self._provider.descriptor
        if value.provider_id != descriptor.id:
            return failure_result(
                provider_status,
                outcome_status="failed",
                code="invalid_model_output",
                retryable=False,
                runtime_ms=value.inference_ms,
                subreason_code="provider_id_mismatch",
            )
        hypotheses = value.hypotheses[: request.top_k]
        candidates = tuple(
            InferenceCandidate(
                rank=index,
                original_rank=hypothesis.original_rank,
                latitude=hypothesis.latitude,
                longitude=hypothesis.longitude,
                raw_score=hypothesis.raw_score,
                limitations=tuple(hypothesis.limitations),
            )
            for index, hypothesis in enumerate(hypotheses, start=1)
        )
        first = hypotheses[0]
        return InferenceResult(
            provider_id=value.provider_id,
            provider_revision=descriptor.version,
            model_name=value.model_name,
            model_revision=value.model_revision,
            runtime_revision=value.implementation_revision,
            classification="real",
            status="succeeded",
            device=value.device,
            dtype=value.dtype,
            runtime_ms=value.inference_ms,
            score_semantics=first.score_type,
            normalization_method=first.normalization_method,
            calibration_state=first.calibration_state,
            candidates=candidates,
            warnings=("warning.global_prediction_uncalibrated",),
            provenance=InferenceProvenance(
                provider_id=value.provider_id,
                provider_revision=descriptor.version,
                model_revision=value.model_revision,
                runtime_revision=value.implementation_revision,
                source_kind="verified_model",
            ),
        )


class _DevelopmentFixture(InferenceModel):
    schema_version: Literal["atlaslens-development-inference-fixture-v1"]
    scenario_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_-]+$")
    provider_id: Literal["development-mock-geolocation"]
    model_name: str = Field(min_length=1, max_length=120)
    model_revision: str = Field(min_length=1, max_length=120)
    runtime_revision: str = Field(min_length=1, max_length=120)
    score_semantics: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    normalization_method: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    calibration_state: CalibrationState
    candidates: tuple[InferenceCandidate, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def stable_ranks(self) -> _DevelopmentFixture:
        if [candidate.rank for candidate in self.candidates] != list(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("development fixture ranks must be contiguous")
        return self


class DevelopmentMockProvider:
    """Explicit development-only provider backed by one contained JSON fixture."""

    provider_id = _DEVELOPMENT_PROVIDER_ID
    classification: Literal["simulated"] = "simulated"

    def __init__(
        self,
        fixture_name: str,
        *,
        environment: Literal["production", "development", "test"],
        mode: InferenceProviderMode = "disabled",
        fixture_root: Path | None = None,
    ) -> None:
        if environment not in {"production", "development", "test"}:
            raise ValueError("unknown application environment")
        if environment == "production" and mode != "disabled":
            raise ValueError("simulated inference is forbidden in production")
        self._mode = mode
        self._fixture = self._load_fixture(fixture_name, fixture_root=fixture_root)

    @property
    def mode(self) -> InferenceProviderMode:
        return self._mode

    @staticmethod
    def _load_fixture(fixture_name: str, *, fixture_root: Path | None) -> _DevelopmentFixture:
        relative = Path(fixture_name)
        if relative.is_absolute() or relative.suffix.lower() != ".json" or ".." in relative.parts:
            raise ValueError("invalid development fixture name")
        default_root = Path(__file__).resolve().parents[3] / "dev_fixtures" / "inference"
        try:
            root = (fixture_root or default_root).resolve(strict=True)
            unresolved = root / relative
            if unresolved.is_symlink():
                raise ValueError("development fixture symlinks are forbidden")
            fixture_path = unresolved.resolve(strict=True)
        except OSError as exc:
            raise ValueError("development fixture is unavailable") from exc
        if not fixture_path.is_relative_to(root) or not fixture_path.is_file():
            raise ValueError("development fixture escaped its root")
        try:
            payload = fixture_path.read_bytes()
        except OSError as exc:
            raise ValueError("development fixture is unreadable") from exc
        if not payload or len(payload) > _MAX_FIXTURE_BYTES:
            raise ValueError("development fixture size is invalid")
        try:
            decoded = json.loads(payload)
            return _DevelopmentFixture.model_validate(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            raise ValueError("development fixture validation failed") from exc

    def status(self) -> InferenceProviderStatus:
        disabled = self._mode == "disabled"
        return InferenceProviderStatus(
            provider_id=self.provider_id,
            provider_type="global_geolocation",
            provider_revision="development-mock-adapter-v1",
            mode=self._mode,
            available=not disabled,
            status="disabled" if disabled else "ready",
            classification="simulated",
            model_name=self._fixture.model_name,
            model_revision=self._fixture.model_revision,
            runtime_revision=self._fixture.runtime_revision,
            device="fixture",
            calibration_state=self._fixture.calibration_state,
            reason_code="disabled" if disabled else None,
        )

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        provider_status = self.status()
        if self._mode == "disabled":
            return failure_result(
                provider_status,
                outcome_status="skipped",
                code="disabled",
                retryable=False,
            )
        if request.cancellation.is_set():
            raise asyncio.CancelledError
        if request.deadline <= datetime.now(UTC):
            return failure_result(
                provider_status,
                outcome_status="failed",
                code="timeout",
                retryable=True,
            )
        candidates = tuple(
            candidate.model_copy(update={"rank": index})
            for index, candidate in enumerate(self._fixture.candidates[: request.top_k], start=1)
        )
        return InferenceResult(
            provider_id=self.provider_id,
            provider_revision="development-mock-adapter-v1",
            model_name=self._fixture.model_name,
            model_revision=self._fixture.model_revision,
            runtime_revision=self._fixture.runtime_revision,
            classification="simulated",
            status="succeeded",
            device="fixture",
            dtype="fixture",
            runtime_ms=0,
            score_semantics=self._fixture.score_semantics,
            normalization_method=self._fixture.normalization_method,
            calibration_state=self._fixture.calibration_state,
            candidates=candidates,
            warnings=("warning.simulated_development_result",),
            provenance=InferenceProvenance(
                provider_id=self.provider_id,
                provider_revision="development-mock-adapter-v1",
                model_revision=self._fixture.model_revision,
                runtime_revision=self._fixture.runtime_revision,
                source_kind="development_fixture",
            ),
            scenario_id=self._fixture.scenario_id,
        )


class CustomModelArtifact(InferenceModel):
    provider_id: str = Field(
        default=_CUSTOM_PROVIDER_ID,
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    artifact_id: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    artifact_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_name: str = Field(min_length=1, max_length=120)
    model_revision: str = Field(min_length=1, max_length=120)
    runtime_revision: str = Field(min_length=1, max_length=120)
    adapter_id: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    verified: bool
    adapter_supported: bool
    supported_devices: tuple[Literal["cpu", "cuda"], ...] = Field(min_length=1, max_length=2)
    max_input_bytes: int = Field(ge=1, le=100 * 1024 * 1024)
    output_dtype: str = Field(min_length=1, max_length=40)
    score_semantics: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    normalization_method: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9._-]+$")
    calibration_state: CalibrationState

    @field_validator("supported_devices")
    @classmethod
    def unique_devices(
        cls, value: tuple[Literal["cpu", "cuda"], ...]
    ) -> tuple[Literal["cpu", "cuda"], ...]:
        if len(value) != len(set(value)):
            raise ValueError("supported devices must be unique")
        return value


class CustomRuntimeCandidate(InferenceModel):
    original_rank: int = Field(ge=1, le=100_000)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    raw_score: float = Field(ge=0, le=1)
    limitations: tuple[str, ...] = Field(min_length=1, max_length=12)

    @field_validator("latitude", "longitude", "raw_score")
    @classmethod
    def finite_numbers(cls, value: float) -> float:
        if not (-float("inf") < value < float("inf")):
            raise ValueError("custom runtime candidate values must be finite")
        return value


class CustomInferenceRuntime(Protocol):
    def predict(
        self,
        *,
        artifact: CustomModelArtifact,
        image_bytes: bytes,
        width: int,
        height: int,
        top_k: int,
        device: Literal["cpu", "cuda"],
    ) -> Sequence[object]: ...


class CustomTrainedModelProvider:
    """Fail-closed adapter for an injected, separately verified custom artifact."""

    provider_id = _CUSTOM_PROVIDER_ID
    classification: Literal["real"] = "real"

    def __init__(
        self,
        artifact: CustomModelArtifact | None,
        runtime: CustomInferenceRuntime | None,
        *,
        mode: InferenceProviderMode = "disabled",
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        self.provider_id = artifact.provider_id if artifact is not None else _CUSTOM_PROVIDER_ID
        self._artifact = artifact
        self._runtime = runtime
        self._mode = mode
        self._device = device
        self._semaphore = asyncio.Semaphore(1)
        self._detached_runtime_tasks: set[asyncio.Task[Sequence[object]]] = set()

    @property
    def mode(self) -> InferenceProviderMode:
        return self._mode

    def status(self) -> InferenceProviderStatus:
        artifact = self._artifact
        reason_code: str | None = None
        operational_status: Literal["ready", "not_installed", "failed", "disabled", "unavailable"]
        if self._mode == "disabled":
            operational_status = "disabled"
            reason_code = "disabled"
        elif artifact is None:
            operational_status = "not_installed"
            reason_code = "model_not_installed"
        elif not artifact.verified:
            operational_status = "failed"
            reason_code = "artifact_unverified"
        elif not artifact.adapter_supported:
            operational_status = "unavailable"
            reason_code = "unsupported_adapter"
        elif self._runtime is None:
            operational_status = "unavailable"
            reason_code = "runtime_unavailable"
        elif self._device not in artifact.supported_devices:
            operational_status = "unavailable"
            reason_code = "unsupported_device"
        else:
            operational_status = "ready"
        return InferenceProviderStatus(
            provider_id=self.provider_id,
            provider_type="global_geolocation",
            provider_revision="custom-trained-adapter-v1",
            mode=self._mode,
            available=operational_status == "ready",
            status=operational_status,
            classification="real",
            model_name=artifact.model_name if artifact is not None else None,
            model_revision=artifact.model_revision if artifact is not None else None,
            runtime_revision=artifact.runtime_revision if artifact is not None else None,
            device=self._device,
            calibration_state=(
                artifact.calibration_state if artifact is not None else "uncalibrated"
            ),
            reason_code=reason_code,
        )

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        provider_status = self.status()
        if not provider_status.available:
            return self._failure(
                provider_status,
                outcome_status="skipped",
                code=provider_status.reason_code or "unavailable",
                retryable=False,
            )
        artifact = self._artifact
        runtime = self._runtime
        if artifact is None or runtime is None:
            return self._failure(
                provider_status,
                outcome_status="skipped",
                code="unavailable",
                retryable=False,
            )
        if request.cancellation.is_set():
            raise asyncio.CancelledError
        if request.deadline <= datetime.now(UTC):
            return self._failure(
                provider_status,
                outcome_status="failed",
                code="timeout",
                retryable=True,
            )
        started = time.monotonic()
        try:
            payload = await self._immutable_payload(request, artifact.max_input_bytes)
        except (OSError, ValueError):
            return self._failure(
                provider_status,
                outcome_status="failed",
                code="unsupported_input",
                retryable=False,
                runtime_ms=_elapsed_ms(started),
            )
        try:
            raw_candidates = await self._invoke_runtime(
                runtime,
                artifact=artifact,
                payload=payload,
                request=request,
            )
            candidates = self._validated_candidates(raw_candidates, top_k=request.top_k)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return self._failure(
                provider_status,
                outcome_status="failed",
                code="timeout",
                retryable=True,
                runtime_ms=_elapsed_ms(started),
            )
        except (OSError, RuntimeError):
            return self._failure(
                provider_status,
                outcome_status="failed",
                code="internal_provider_error",
                retryable=False,
                runtime_ms=_elapsed_ms(started),
            )
        except (TypeError, ValueError, ValidationError):
            return self._failure(
                provider_status,
                outcome_status="failed",
                code="invalid_model_output",
                retryable=False,
                runtime_ms=_elapsed_ms(started),
            )
        except Exception:
            return self._failure(
                provider_status,
                outcome_status="failed",
                code="internal_provider_error",
                retryable=False,
                runtime_ms=_elapsed_ms(started),
            )
        return InferenceResult(
            provider_id=self.provider_id,
            provider_revision="custom-trained-adapter-v1",
            model_name=artifact.model_name,
            model_revision=artifact.model_revision,
            runtime_revision=artifact.runtime_revision,
            classification="real",
            status="succeeded",
            device=self._device,
            dtype=artifact.output_dtype,
            runtime_ms=_elapsed_ms(started),
            score_semantics=artifact.score_semantics,
            normalization_method=artifact.normalization_method,
            calibration_state=artifact.calibration_state,
            candidates=candidates,
            provenance=InferenceProvenance(
                provider_id=self.provider_id,
                provider_revision="custom-trained-adapter-v1",
                model_revision=artifact.model_revision,
                runtime_revision=artifact.runtime_revision,
                source_kind="verified_custom_artifact",
                artifact_digest=artifact.artifact_digest,
            ),
        )

    def _detach_runtime_task(self, task: asyncio.Task[Sequence[object]]) -> None:
        self._detached_runtime_tasks.add(task)
        task.add_done_callback(self._finish_detached_runtime_task)

    def _finish_detached_runtime_task(self, task: asyncio.Task[Sequence[object]]) -> None:
        self._detached_runtime_tasks.discard(task)
        self._semaphore.release()
        try:
            task.result()
        except (asyncio.CancelledError, Exception):
            return

    async def _invoke_runtime(
        self,
        runtime: CustomInferenceRuntime,
        *,
        artifact: CustomModelArtifact,
        payload: bytes,
        request: InferenceRequest,
    ) -> Sequence[object]:
        remaining = (request.deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise TimeoutError
        acquired = False
        detached = False
        task: asyncio.Task[Sequence[object]] | None = None
        started = time.monotonic()
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=remaining)
            acquired = True
            remaining -= time.monotonic() - started
            if remaining <= 0:
                raise TimeoutError
            task = asyncio.create_task(
                asyncio.to_thread(
                    runtime.predict,
                    artifact=artifact,
                    image_bytes=payload,
                    width=request.width,
                    height=request.height,
                    top_k=request.top_k,
                    device=self._device,
                )
            )
            return await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except (TimeoutError, asyncio.CancelledError):
            if task is not None and not task.done():
                self._detach_runtime_task(task)
                detached = True
            raise
        finally:
            if acquired and not detached:
                self._semaphore.release()

    @staticmethod
    async def _immutable_payload(request: InferenceRequest, limit: int) -> bytes:
        if request.image_bytes is not None:
            if len(request.image_bytes) > limit:
                raise ValueError("custom model input exceeds artifact bound")
            return request.image_bytes
        handle = request.image_handle
        if handle is None:
            raise ValueError("custom model input is unavailable")

        def read_bounded() -> bytes:
            with handle.path.open("rb") as source:
                payload = source.read(limit + 1)
            if not payload or len(payload) > limit:
                raise ValueError("custom model input size is invalid")
            return bytes(payload)

        return await asyncio.to_thread(read_bounded)

    @staticmethod
    def _validated_candidates(
        values: Sequence[object], *, top_k: int
    ) -> tuple[InferenceCandidate, ...]:
        if isinstance(values, str | bytes | bytearray) or not 1 <= len(values) <= 100:
            raise ValueError("custom runtime candidate count is invalid")
        parsed = tuple(CustomRuntimeCandidate.model_validate(value) for value in values)
        selected = parsed[:top_k]
        original_ranks = [item.original_rank for item in selected]
        if len(original_ranks) != len(set(original_ranks)):
            raise ValueError("custom runtime candidate ranks must be unique")
        return tuple(
            InferenceCandidate(
                rank=index,
                original_rank=item.original_rank,
                latitude=item.latitude,
                longitude=item.longitude,
                raw_score=item.raw_score,
                limitations=item.limitations,
            )
            for index, item in enumerate(selected, start=1)
        )

    @staticmethod
    def _failure(
        status: InferenceProviderStatus,
        *,
        outcome_status: Literal["failed", "skipped"],
        code: str,
        retryable: bool,
        runtime_ms: int = 0,
    ) -> InferenceResult:
        artifact = status.model_revision or "unavailable"
        return InferenceResult(
            provider_id=status.provider_id,
            provider_revision=status.provider_revision,
            model_name=status.model_name or "unavailable-model",
            model_revision=artifact,
            runtime_revision=status.runtime_revision or "unavailable",
            classification="real",
            status=outcome_status,
            device=status.device,
            runtime_ms=runtime_ms,
            calibration_state=status.calibration_state or "uncalibrated",
            failure=InferenceFailure(
                code=code,
                message_key=f"provider.inference.{code}",
                retryable=retryable,
            ),
            provenance=InferenceProvenance(
                provider_id=status.provider_id,
                provider_revision=status.provider_revision,
                model_revision=artifact,
                runtime_revision=status.runtime_revision or "unavailable",
                source_kind="verified_custom_artifact",
            ),
        )


def descriptor_for_inference(result: InferenceResult) -> ProviderDescriptor:
    return ProviderDescriptor(
        id=result.provider_id,
        kind="global_geolocation",
        version=result.provider_revision,
        execution_boundary="local",
        criticality="optional",
        available=result.status == "succeeded",
        unavailable_reason_code=(result.failure.code if result.failure is not None else None),
        model_name=result.model_name,
    )


def legacy_outcome_for_inference(
    result: InferenceResult,
) -> ProviderOutcome[GlobalPredictionResult]:
    if result.status == "abstained":
        return ProviderOutcome.abstained()
    if result.status != "succeeded":
        failure_code = result.failure.code if result.failure is not None else "unavailable"
        mapped_code = _legacy_failure_code(failure_code)
        if result.status == "skipped":
            return ProviderOutcome.skipped(mapped_code)
        return ProviderOutcome.failed(
            mapped_code,
            retryable=result.failure.retryable if result.failure is not None else False,
            attempts=1,
            duration_ms=result.runtime_ms,
            subreason_code=(result.failure.subreason_code if result.failure is not None else None),
        )
    if (
        result.score_semantics is None
        or result.normalization_method is None
        or any(not 0 <= candidate.raw_score <= 1 for candidate in result.candidates)
    ):
        return ProviderOutcome.failed(
            "invalid_model_output",
            retryable=False,
            attempts=1,
            duration_ms=result.runtime_ms,
            subreason_code="unsupported_score_semantics",
        )
    hypotheses = [
        GlobalPredictionHypothesis(
            rank=index,
            original_rank=candidate.original_rank,
            latitude=candidate.latitude,
            longitude=candidate.longitude,
            raw_score=candidate.raw_score,
            score_type=result.score_semantics,
            normalization_method=result.normalization_method,
            calibration_state=result.calibration_state,
            limitations=list(candidate.limitations),
        )
        # Preserve the full bounded internal candidate set for downstream
        # clustering. Public synthesis remains separately capped at five.
        for index, candidate in enumerate(result.candidates, start=1)
    ]
    return ProviderOutcome.succeeded(
        GlobalPredictionResult(
            provider_id=result.provider_id,
            model_name=result.model_name,
            model_revision=result.model_revision,
            implementation_revision=result.runtime_revision,
            device=result.device or "unknown",
            dtype=result.dtype or "unknown",
            inference_ms=result.runtime_ms,
            hypotheses=hypotheses,
        )
    )


def _legacy_failure_code(code: str) -> str:
    if code in {"timeout", "inference_timeout"}:
        return "inference_timeout"
    if code in {"invalid_output", "invalid_model_output"}:
        return "invalid_model_output"
    if code == "unsupported_input":
        return "unsupported_input"
    if code == "disabled":
        return "disabled"
    if code == "model_not_installed":
        return "model_not_installed"
    if code == "unsupported_device":
        return "unsupported_device"
    if code in {
        "artifact_unverified",
        "unsupported_adapter",
        "runtime_unavailable",
        "unavailable",
    }:
        return "unavailable"
    return "internal_provider_error"


def _abstained_result(status: InferenceProviderStatus, *, runtime_ms: int) -> InferenceResult:
    return InferenceResult(
        provider_id=status.provider_id,
        provider_revision=status.provider_revision,
        model_name=status.model_name or "unavailable-model",
        model_revision=status.model_revision or "unavailable",
        runtime_revision=status.runtime_revision or "unavailable",
        classification=status.classification,
        status="abstained",
        device=status.device,
        runtime_ms=runtime_ms,
        calibration_state=status.calibration_state or "uncalibrated",
        warnings=(
            ("warning.simulated_development_result",)
            if status.classification == "simulated"
            else ()
        ),
        provenance=InferenceProvenance(
            provider_id=status.provider_id,
            provider_revision=status.provider_revision,
            model_revision=status.model_revision or "unavailable",
            runtime_revision=status.runtime_revision or "unavailable",
            source_kind=(
                "development_fixture"
                if status.classification == "simulated"
                else "provider_adapter"
            ),
        ),
        scenario_id=("abstained_simulation" if status.classification == "simulated" else None),
    )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
