from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from atlaslens_api.config import Settings
from atlaslens_api.inference import (
    CustomModelArtifact,
    CustomTrainedModelProvider,
    DevelopmentMockProvider,
    GeoCLIPInferenceAdapter,
    InferenceCancelledError,
    InferenceCandidate,
    InferenceFailure,
    InferenceProvenance,
    InferenceProviderStatus,
    InferenceRequest,
    InferenceResult,
    ProviderEnsembleCoordinator,
    legacy_outcome_for_inference,
)
from atlaslens_api.providers.base import (
    GlobalPredictionHypothesis,
    GlobalPredictionResult,
    GlobalProviderStatus,
    InvocationContext,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.schemas import AnalysisMode
from atlaslens_api.storage import LocalImageHandle
from conftest import image_bytes, upload, wait_for_terminal


def _request(
    *,
    cancellation: asyncio.Event | None = None,
    deadline: datetime | None = None,
    image_handle: LocalImageHandle | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        image_handle=image_handle,
        image_bytes=None if image_handle is not None else b"immutable-image-payload",
        width=64,
        height=48,
        exif=None,
        analysis_mode=AnalysisMode.LOCAL_ONLY,
        top_k=5,
        cancellation=cancellation or asyncio.Event(),
        deadline=deadline or datetime.now(UTC) + timedelta(seconds=2),
        trace_id="trace-test-0001",
        max_input_bytes=1024,
    )


def _status(
    provider_id: str,
    mode: Literal["disabled", "shadow", "candidate", "primary"],
    *,
    classification: Literal["real", "simulated"] = "real",
) -> InferenceProviderStatus:
    return InferenceProviderStatus(
        provider_id=provider_id,
        provider_type="global_geolocation",
        provider_revision="provider-v1",
        mode=mode,
        available=mode != "disabled",
        status="disabled" if mode == "disabled" else "ready",
        classification=classification,
        model_name="test-model",
        model_revision="model-v1",
        runtime_revision="runtime-v1",
        device="cpu",
        calibration_state="uncalibrated",
        reason_code="disabled" if mode == "disabled" else None,
    )


def _result(
    provider_id: str,
    *,
    latitude: float = 40.0,
    longitude: float = 30.0,
) -> InferenceResult:
    return InferenceResult(
        provider_id=provider_id,
        provider_revision="provider-v1",
        model_name="test-model",
        model_revision="model-v1",
        runtime_revision="runtime-v1",
        classification="real",
        status="succeeded",
        device="cpu",
        dtype="float32",
        runtime_ms=1,
        score_semantics="uncalibrated_gallery_softmax",
        normalization_method="softmax_over_fixed_gallery",
        calibration_state="uncalibrated",
        candidates=(
            InferenceCandidate(
                rank=1,
                original_rank=1,
                latitude=latitude,
                longitude=longitude,
                raw_score=1.0,
                limitations=("test_only",),
            ),
        ),
        provenance=InferenceProvenance(
            provider_id=provider_id,
            provider_revision="provider-v1",
            model_revision="model-v1",
            runtime_revision="runtime-v1",
            source_kind="verified_model",
        ),
    )


def test_legacy_outcome_preserves_full_internal_candidate_set() -> None:
    candidates = tuple(
        InferenceCandidate(
            rank=rank,
            original_rank=rank,
            latitude=-60.0 + rank,
            longitude=-120.0 + rank,
            raw_score=rank / 50.0,
            limitations=("test_only",),
        )
        for rank in range(1, 51)
    )
    result = _result("geoclip-global-v1").model_copy(update={"candidates": candidates})

    outcome = legacy_outcome_for_inference(result)

    assert outcome.status.value == "succeeded"
    assert outcome.value is not None
    assert len(outcome.value.hypotheses) == 50
    assert [item.rank for item in outcome.value.hypotheses] == list(range(1, 51))
    assert outcome.value.hypotheses[-1].original_rank == 50


def _abstained_result(provider_id: str) -> InferenceResult:
    return InferenceResult(
        provider_id=provider_id,
        provider_revision="provider-v1",
        model_name="test-model",
        model_revision="model-v1",
        runtime_revision="runtime-v1",
        classification="real",
        status="abstained",
        device="cpu",
        runtime_ms=1,
        calibration_state="uncalibrated",
        provenance=InferenceProvenance(
            provider_id=provider_id,
            provider_revision="provider-v1",
            model_revision="model-v1",
            runtime_revision="runtime-v1",
            source_kind="verified_model",
        ),
    )


def _failed_result(provider_id: str) -> InferenceResult:
    return InferenceResult(
        provider_id=provider_id,
        provider_revision="provider-v1",
        model_name="test-model",
        model_revision="model-v1",
        runtime_revision="runtime-v1",
        classification="real",
        status="failed",
        device="cpu",
        runtime_ms=1,
        calibration_state="uncalibrated",
        failure=InferenceFailure(
            code="unavailable",
            message_key="provider.inference.unavailable",
            retryable=False,
        ),
        provenance=InferenceProvenance(
            provider_id=provider_id,
            provider_revision="provider-v1",
            model_revision="model-v1",
            runtime_revision="runtime-v1",
            source_kind="verified_model",
        ),
    )


class StaticProvider:
    classification: Literal["real"] = "real"

    def __init__(
        self,
        provider_id: str,
        mode: Literal["disabled", "shadow", "candidate", "primary"],
        result: InferenceResult,
    ) -> None:
        self.provider_id = provider_id
        self.mode = mode
        self._result = result

    def status(self) -> InferenceProviderStatus:
        return _status(self.provider_id, self.mode)

    async def infer(self, _: InferenceRequest) -> InferenceResult:
        return self._result


class UnavailableProvider(StaticProvider):
    def __init__(self) -> None:
        super().__init__("unavailable-real", "candidate", _result("unavailable-real"))
        self.invoked = False

    def status(self) -> InferenceProviderStatus:
        return _status(self.provider_id, self.mode).model_copy(
            update={
                "available": False,
                "status": "not_installed",
                "reason_code": "model_not_installed",
            }
        )

    async def infer(self, _: InferenceRequest) -> InferenceResult:
        self.invoked = True
        raise AssertionError("unavailable provider must not be invoked")


def _artifact(
    *,
    verified: bool = True,
    adapter_supported: bool = True,
) -> CustomModelArtifact:
    return CustomModelArtifact(
        artifact_id="custom-model-v1",
        artifact_digest="a" * 64,
        model_name="custom-coordinate-model",
        model_revision="model-v1",
        runtime_revision="runtime-v1",
        adapter_id="reviewed-runtime-v1",
        verified=verified,
        adapter_supported=adapter_supported,
        supported_devices=("cpu", "cuda"),
        max_input_bytes=1024,
        output_dtype="float32",
        score_semantics="uncalibrated_gallery_softmax",
        normalization_method="softmax_over_fixed_gallery",
        calibration_state="uncalibrated",
    )


class CapturingRuntime:
    def __init__(self, output: list[object] | None = None) -> None:
        self.output = output or [
            {
                "original_rank": 1,
                "latitude": 38.72,
                "longitude": 35.48,
                "raw_score": 0.7,
                "limitations": ["uncalibrated_custom_output"],
            }
        ]
        self.devices: list[str] = []
        self.payload_types: list[type[object]] = []

    def predict(self, **kwargs: Any) -> list[object]:
        self.devices.append(kwargs["device"])
        self.payload_types.append(type(kwargs["image_bytes"]))
        return self.output


def test_mock_is_disabled_by_default_and_active_mock_is_refused_in_production() -> None:
    settings = Settings(_env_file=None)
    assert settings.enable_mock_inference is False
    disabled = DevelopmentMockProvider(
        "kayseri-development.json", environment="production", mode="disabled"
    )
    assert disabled.status().status == "disabled"

    with pytest.raises(ValidationError, match="mock inference is forbidden"):
        Settings(_env_file=None, app_env="production", enable_mock_inference=True)
    with pytest.raises(ValueError, match="forbidden in production"):
        DevelopmentMockProvider(
            "kayseri-development.json", environment="production", mode="candidate"
        )


@pytest.mark.asyncio
async def test_explicit_development_mock_is_deterministic_and_watermarked() -> None:
    provider = DevelopmentMockProvider(
        "kayseri-development.json", environment="development", mode="candidate"
    )
    first = await provider.infer(_request())
    second = await provider.infer(_request())

    assert first.candidates == second.candidates
    assert first.classification == "simulated"
    assert first.scenario_id == "kayseri_development"
    assert "warning.simulated_development_result" in first.warnings
    assert first.provenance.source_kind == "development_fixture"


def test_development_mock_rejects_fixture_root_escape() -> None:
    with pytest.raises(ValueError, match="invalid development fixture name"):
        DevelopmentMockProvider("../kayseri-development.json", environment="test", mode="candidate")


class LegacyGeoProvider:
    descriptor = ProviderDescriptor(
        id="legacy-geoclip",
        kind="global_geolocation",
        version="legacy-provider-v1",
        execution_boundary="local",
        criticality="optional",
        available=True,
        model_name="GeoCLIP",
    )

    def __init__(self) -> None:
        self.seen_handle: LocalImageHandle | None = None
        self.seen_context: InvocationContext | None = None

    def status(self) -> GlobalProviderStatus:
        return GlobalProviderStatus(
            status="ready",
            installed=True,
            verified=True,
            model_name="GeoCLIP",
            model_revision="model-v1",
            device="cpu",
            calibration_state="uncalibrated",
        )

    async def predict(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[GlobalPredictionResult]:
        self.seen_handle = handle
        self.seen_context = context
        return ProviderOutcome.succeeded(
            GlobalPredictionResult(
                provider_id=self.descriptor.id,
                model_name="GeoCLIP",
                model_revision="model-v1",
                implementation_revision="runtime-v1",
                device="cpu",
                dtype="float32",
                inference_ms=1,
                hypotheses=[
                    GlobalPredictionHypothesis(
                        rank=1,
                        original_rank=1,
                        latitude=40.0,
                        longitude=30.0,
                        raw_score=1.0,
                        score_type="uncalibrated_gallery_softmax",
                        normalization_method="softmax_over_fixed_gallery",
                        calibration_state="uncalibrated",
                        limitations=["uncalibrated_model_output"],
                    )
                ],
            )
        )


@pytest.mark.asyncio
async def test_geoclip_adapter_preserves_the_existing_provider_contract(tmp_path: Path) -> None:
    path = tmp_path / "normalized.png"
    path.write_bytes(b"normalized")
    handle = LocalImageHandle(key="a" * 32 + ".png", path=path)
    legacy = LegacyGeoProvider()
    adapter = GeoCLIPInferenceAdapter(legacy)

    result = await adapter.infer(_request(image_handle=handle))

    assert result.status == "succeeded"
    assert result.provider_id == legacy.descriptor.id
    assert result.runtime_revision == "runtime-v1"
    assert legacy.seen_handle is handle
    assert legacy.seen_context is not None
    assert legacy.seen_context.request_id == "trace-test-0001"


@pytest.mark.asyncio
async def test_geoclip_adapter_preserves_not_installed_reason_without_invocation() -> None:
    legacy = LegacyGeoProvider()
    legacy.status = lambda: GlobalProviderStatus(  # type: ignore[method-assign]
        status="not_installed",
        installed=False,
        verified=False,
        model_name="GeoCLIP",
        model_revision="model-v1",
        calibration_state="uncalibrated",
    )
    adapter = GeoCLIPInferenceAdapter(legacy)
    coordinated = await ProviderEnsembleCoordinator((adapter,)).infer(_request())

    assert legacy.seen_handle is None
    assert coordinated.decision_result is not None
    assert coordinated.decision_result.failure is not None
    assert coordinated.decision_result.failure.code == "model_not_installed"
    assert f"provider.inference.{adapter.provider_id}.model_not_installed" in (coordinated.warnings)


@pytest.mark.asyncio
async def test_custom_provider_missing_unverified_and_unsupported_artifacts_fail_closed() -> None:
    runtime = CapturingRuntime()
    providers = (
        CustomTrainedModelProvider(None, runtime, mode="candidate"),
        CustomTrainedModelProvider(_artifact(verified=False), runtime, mode="candidate"),
        CustomTrainedModelProvider(_artifact(adapter_supported=False), runtime, mode="candidate"),
    )

    results = [await provider.infer(_request()) for provider in providers]

    assert [result.status for result in results] == ["skipped", "skipped", "skipped"]
    assert [result.failure.code for result in results if result.failure is not None] == [
        "model_not_installed",
        "artifact_unverified",
        "unsupported_adapter",
    ]
    assert runtime.devices == []


@pytest.mark.asyncio
async def test_custom_provider_rejects_invalid_runtime_output() -> None:
    runtime = CapturingRuntime(output=[{"latitude": "not-a-number"}])
    provider = CustomTrainedModelProvider(_artifact(), runtime, mode="candidate", device="cpu")

    result = await provider.infer(_request())

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "invalid_model_output"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.asyncio
async def test_custom_provider_has_injected_cpu_and_cuda_runtime_seams(
    device: Literal["cpu", "cuda"],
) -> None:
    runtime = CapturingRuntime()
    provider = CustomTrainedModelProvider(_artifact(), runtime, mode="candidate", device=device)

    result = await provider.infer(_request())

    assert result.status == "succeeded"
    assert runtime.devices == [device]
    assert runtime.payload_types == [bytes]


@pytest.mark.asyncio
async def test_shadow_and_candidate_results_do_not_change_primary_ranking() -> None:
    primary = StaticProvider("primary-real", "primary", _result("primary-real"))
    shadow = StaticProvider("shadow-real", "shadow", _result("shadow-real", latitude=-20.0))
    candidate = StaticProvider(
        "candidate-real", "candidate", _result("candidate-real", latitude=10.0)
    )
    coordinator = ProviderEnsembleCoordinator((primary, shadow, candidate))

    coordinated = await coordinator.infer(_request())

    assert coordinated.selected is primary._result
    assert [comparison.provider_id for comparison in coordinated.comparisons] == [
        "shadow-real",
        "candidate-real",
    ]
    assert coordinated.comparisons[0].ranking_impact == "none"
    assert coordinated.comparisons[1].ranking_impact == "eligible_not_applied"


@pytest.mark.asyncio
async def test_simulated_candidate_is_excluded_when_real_provider_succeeds() -> None:
    real = StaticProvider("primary-real", "primary", _result("primary-real"))
    simulated = DevelopmentMockProvider(
        "kayseri-development.json", environment="test", mode="candidate"
    )
    coordinator = ProviderEnsembleCoordinator((real, simulated))

    coordinated = await coordinator.infer(_request())

    assert coordinated.selected is real._result
    assert coordinated.comparisons[0].status == "skipped"
    assert coordinated.comparisons[0].failure_code == "simulated_excluded_from_real_result"
    assert any(
        execution.result.classification == "simulated" for execution in coordinated.executions
    )


@pytest.mark.asyncio
async def test_partial_shadow_failure_does_not_fail_primary_result() -> None:
    primary = StaticProvider("primary-real", "primary", _result("primary-real"))
    shadow = StaticProvider("shadow-real", "shadow", _failed_result("shadow-real"))
    coordinator = ProviderEnsembleCoordinator((primary, shadow))

    coordinated = await coordinator.infer(_request())

    assert coordinated.selected is primary._result
    assert coordinated.comparisons[0].status == "failed"
    assert "provider.inference.shadow-real.unavailable" in coordinated.warnings


@pytest.mark.asyncio
async def test_unavailable_provider_is_reported_without_invocation() -> None:
    provider = UnavailableProvider()
    coordinator = ProviderEnsembleCoordinator((provider,))

    coordinated = await coordinator.infer(_request())

    assert provider.invoked is False
    assert coordinated.selected is None
    assert coordinated.decision_result is not None
    assert coordinated.decision_result.status == "skipped"
    assert coordinated.decision_result.failure is not None
    assert coordinated.decision_result.failure.code == "model_not_installed"


class BarrierState:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.started = 0
        self.ready = asyncio.Event()


class BarrierProvider(StaticProvider):
    def __init__(self, provider_id: str, mode: Literal["primary", "shadow"], state: BarrierState):
        super().__init__(provider_id, mode, _result(provider_id))
        self._state = state

    async def infer(self, _: InferenceRequest) -> InferenceResult:
        async with self._state.lock:
            self._state.started += 1
            if self._state.started == 2:
                self._state.ready.set()
        await self._state.ready.wait()
        return self._result


@pytest.mark.asyncio
async def test_coordinator_starts_independent_providers_concurrently() -> None:
    state = BarrierState()
    coordinator = ProviderEnsembleCoordinator(
        (
            BarrierProvider("primary-real", "primary", state),
            BarrierProvider("shadow-real", "shadow", state),
        )
    )

    coordinated = await coordinator.infer(_request())

    assert state.started == 2
    assert coordinated.selected is not None
    assert coordinated.selected.provider_id == "primary-real"


class BlockingProvider(StaticProvider):
    def __init__(self) -> None:
        super().__init__("blocking-real", "primary", _result("blocking-real"))
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def infer(self, _: InferenceRequest) -> InferenceResult:
        self.started.set()
        await self.release.wait()
        return self._result


@pytest.mark.asyncio
async def test_coordinator_propagates_request_cancellation() -> None:
    provider = BlockingProvider()
    cancellation = asyncio.Event()
    coordinator = ProviderEnsembleCoordinator((provider,))
    task = asyncio.create_task(coordinator.infer(_request(cancellation=cancellation)))
    await provider.started.wait()

    cancellation.set()

    with pytest.raises(InferenceCancelledError):
        await task


@pytest.mark.asyncio
async def test_coordinator_turns_deadline_into_safe_timeout() -> None:
    provider = BlockingProvider()
    coordinator = ProviderEnsembleCoordinator((provider,))

    coordinated = await coordinator.infer(
        _request(deadline=datetime.now(UTC) + timedelta(milliseconds=20))
    )

    assert coordinated.selected is None
    assert coordinated.decision_result is not None
    assert coordinated.decision_result.status == "failed"
    assert coordinated.decision_result.failure is not None
    assert coordinated.decision_result.failure.code == "timeout"


@pytest.mark.asyncio
async def test_coordinator_preserves_provider_abstention() -> None:
    provider = StaticProvider("abstaining-real", "primary", _abstained_result("abstaining-real"))
    coordinator = ProviderEnsembleCoordinator((provider,))

    coordinated = await coordinator.infer(_request())

    assert coordinated.selected is None
    assert coordinated.decision_result is not None
    assert coordinated.decision_result.status == "abstained"


def test_mock_pipeline_marks_analysis_as_simulated(client_factory: Any) -> None:
    client = client_factory(
        app_env="test",
        enable_mock_inference=True,
        mock_inference_fixture="kayseri-development.json",
        phase5b_enabled=False,
        retrieval_enabled=False,
    )
    accepted = upload(client, image_bytes())
    assert accepted.status_code == 202
    analysis = wait_for_terminal(client, accepted.json()["id"])

    assert analysis["status"] == "completed"
    assert analysis["result_classification"] == "simulated"
    assert analysis["simulation"] == {
        "scenario_id": "kayseri_development",
        "warning_key": "warning.simulated_development_result",
        "watermark": "SIMULATED DEVELOPMENT RESULT",
    }
    assert "warning.simulated_development_result" in analysis["warnings"]
    assert analysis["candidates"]
