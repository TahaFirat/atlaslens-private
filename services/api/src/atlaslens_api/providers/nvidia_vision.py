"""AtlasLens VisionClueProvider adapter for the optional NVIDIA Phase 3C1 service."""

from __future__ import annotations

import asyncio
import base64
import binascii
import time

import httpx
from pydantic import SecretStr

from atlaslens_api.image_processing import CloudSafeDerivative
from atlaslens_api.nvidia_vision.client import NVIDIA_VISION_MODEL
from atlaslens_api.nvidia_vision.errors import (
    NvidiaAuthenticationError,
    NvidiaCancelledError,
    NvidiaCircuitOpenError,
    NvidiaConfigurationError,
    NvidiaModelUnavailableError,
    NvidiaPreprocessingError,
    NvidiaProviderUnavailableError,
    NvidiaRateLimitError,
    NvidiaResponseError,
    NvidiaTimeoutError,
    NvidiaTransportError,
)
from atlaslens_api.nvidia_vision.provider import NvidiaCloudAuthorization
from atlaslens_api.nvidia_vision.reasoning import (
    NvidiaGeolocationReasoner,
    build_nvidia_geolocation_reasoner,
)
from atlaslens_api.providers.base import (
    InvocationContext,
    ProviderDescriptor,
    ProviderOutcome,
    VisionClueResult,
    VisionHypothesis,
)
from atlaslens_api.schemas import AnalysisMode

_JPEG_DATA_URL_PREFIX = "data:image/jpeg;base64,"
_MAX_EXISTING_DERIVATIVE_BYTES = 4 * 1024 * 1024


def _decode_existing_derivative(derivative: CloudSafeDerivative) -> bytes:
    if (
        not derivative.data_url.startswith(_JPEG_DATA_URL_PREFIX)
        or not 0 < derivative.byte_count <= _MAX_EXISTING_DERIVATIVE_BYTES
    ):
        raise NvidiaPreprocessingError("nvidia_cloud_derivative_invalid")
    encoded = derivative.data_url[len(_JPEG_DATA_URL_PREFIX) :]
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise NvidiaPreprocessingError("nvidia_cloud_derivative_invalid") from None
    if len(payload) != derivative.byte_count:
        raise NvidiaPreprocessingError("nvidia_cloud_derivative_invalid")
    return payload


class NvidiaVisionClueProvider:
    """Disabled-by-default NVIDIA adapter preserving the existing provider contract."""

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: SecretStr | str | None,
        model: str = NVIDIA_VISION_MODEL,
        timeout_seconds: float = 90,
        maximum_edge: int = 1_280,
        jpeg_quality: int = 82,
        http_client: httpx.AsyncClient | None = None,
        reasoner: NvidiaGeolocationReasoner | None = None,
    ) -> None:
        self._reasoner = reasoner or build_nvidia_geolocation_reasoner(
            enabled=enabled,
            api_key=api_key,
            model=model,
            total_timeout_seconds=timeout_seconds,
            max_transport_retries=0,
            maximum_edge=maximum_edge,
            jpeg_quality=jpeg_quality,
            http_client=http_client,
        )
        status = self._reasoner.status()
        available = enabled and status.credentials_configured
        self.descriptor = ProviderDescriptor(
            id="nvidia-qwen-vision-reasoning",
            kind="vision_language",
            version="phase3c1-v1",
            execution_boundary="cloud",
            criticality="optional",
            available=available,
            unavailable_reason_code=(
                None if available else "disabled" if not enabled else "missing_secret"
            ),
            model_name=self._reasoner.model,
        )

    def __repr__(self) -> str:
        return (
            "NvidiaVisionClueProvider("
            f"available={self.descriptor.available!r}, model={self.descriptor.model_name!r}, "
            "api_key=<redacted>)"
        )

    async def close(self) -> None:
        await self._reasoner.close()

    async def extract(
        self,
        derivative: CloudSafeDerivative,
        context: InvocationContext,
    ) -> ProviderOutcome[VisionClueResult]:
        if context.mode != AnalysisMode.CLOUD_ASSISTED or not context.cloud_consent:
            return ProviderOutcome.skipped("privacy_denied")
        if not self.descriptor.available:
            code = "disabled" if self.descriptor.unavailable_reason_code == "disabled" else (
                "missing_secret"
            )
            return ProviderOutcome.skipped(code)
        if context.cancellation.is_set():
            raise asyncio.CancelledError

        started = time.monotonic()
        try:
            payload = _decode_existing_derivative(derivative)
            try:
                response = await self._reasoner.analyze(
                    payload,
                    declared_media_type="image/jpeg",
                    authorization=NvidiaCloudAuthorization(
                        mode=context.mode.value,
                        cloud_consent=context.cloud_consent,
                    ),
                    cancellation=context.cancellation,
                )
            finally:
                del payload
        except NvidiaCancelledError:
            raise asyncio.CancelledError from None
        except NvidiaPreprocessingError as exc:
            return self._failure("unsupported_input", exc.code, started, retryable=False)
        except NvidiaAuthenticationError as exc:
            return self._failure("upstream_auth", exc.code, started, retryable=False)
        except NvidiaRateLimitError as exc:
            return self._failure("rate_limited", exc.code, started, retryable=True)
        except NvidiaModelUnavailableError as exc:
            return self._failure("unavailable", exc.code, started, retryable=False)
        except NvidiaTimeoutError as exc:
            return self._failure("timeout", exc.code, started, retryable=True)
        except NvidiaCircuitOpenError as exc:
            return self._failure("transient_upstream", exc.code, started, retryable=True)
        except NvidiaResponseError as exc:
            return self._failure("invalid_output", exc.code, started, retryable=False)
        except NvidiaProviderUnavailableError as exc:
            return self._failure("unavailable", exc.code, started, retryable=False)
        except NvidiaConfigurationError as exc:
            return self._failure(
                "internal_provider_error", exc.code, started, retryable=False
            )
        except NvidiaTransportError as exc:
            return self._failure("transient_upstream", exc.code, started, retryable=True)

        result = response.value
        if result.decision == "abstain":
            return ProviderOutcome.abstained()
        clues = {item.clue_id: item.observation for item in result.observed_clues}
        hypotheses = [
            VisionHypothesis(
                label=item.label,
                latitude=item.latitude,
                longitude=item.longitude,
                radius_km=max(item.uncertainty_radius_km, 25.0),
                confidence=min(item.confidence, 0.35),
                granularity=item.granularity,
                country_code=item.country_code,
                visible_clues=[clues[clue_id] for clue_id in item.supporting_clue_ids[:6]],
            )
            for item in result.hypotheses
        ]
        return ProviderOutcome.succeeded(VisionClueResult(hypotheses=hypotheses))

    @staticmethod
    def _failure(
        code: str,
        subreason: str,
        started: float,
        *,
        retryable: bool,
    ) -> ProviderOutcome[VisionClueResult]:
        return ProviderOutcome.failed(
            code,
            retryable=retryable,
            attempts=1,
            duration_ms=max(0, int((time.monotonic() - started) * 1_000)),
            subreason_code=subreason,
        )
