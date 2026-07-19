"""Target-independent, consent-gated NVIDIA geolocation reasoning service."""

from __future__ import annotations

import asyncio

import httpx
from pydantic import SecretStr, ValidationError

from .client import NVIDIA_BASE_URL, NVIDIA_VISION_MODEL, NvidiaVisionClientConfig
from .errors import NvidiaConfigurationError
from .models import NvidiaGeolocationReasoning
from .preprocessing import CropKind, prepare_nvidia_images
from .prompt import NVIDIA_GEO_SYSTEM_PROMPT, build_nvidia_geo_user_prompt
from .provider import (
    NvidiaCloudAuthorization,
    NvidiaProviderStatus,
    NvidiaStructuredResponse,
    NvidiaVisionProvider,
)


class NvidiaGeolocationReasoner:
    """Prepare one private image in memory and perform one schema-validated call."""

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: SecretStr | str | None,
        model: str = NVIDIA_VISION_MODEL,
        total_timeout_seconds: float = 90,
        max_transport_retries: int = 0,
        maximum_edge: int = 1_280,
        jpeg_quality: int = 82,
        crop_kinds: tuple[CropKind, ...] = ("full", "centre", "upper", "lower"),
        http_client: httpx.AsyncClient | None = None,
        provider: NvidiaVisionProvider | None = None,
    ) -> None:
        self._maximum_edge = maximum_edge
        self._jpeg_quality = jpeg_quality
        self._crop_kinds = crop_kinds
        self._provider = provider or NvidiaVisionProvider(
            NvidiaVisionClientConfig(
                model=model,
                total_timeout_seconds=total_timeout_seconds,
                max_transport_retries=max_transport_retries,
            ),
            enabled=enabled,
            api_key=api_key,
            http_client=http_client,
        )

    def __repr__(self) -> str:
        return (
            "NvidiaGeolocationReasoner(image_policy=<bounded>, "
            f"provider={self._provider!r})"
        )

    @property
    def model(self) -> str:
        return self._provider.config.model

    def status(self) -> NvidiaProviderStatus:
        return self._provider.status()

    async def close(self) -> None:
        await self._provider.aclose()

    async def analyze(
        self,
        payload: bytes,
        *,
        declared_media_type: str | None,
        authorization: NvidiaCloudAuthorization,
        cancellation: asyncio.Event | None = None,
    ) -> NvidiaStructuredResponse[NvidiaGeolocationReasoning]:
        prepared = await asyncio.to_thread(
            prepare_nvidia_images,
            payload,
            declared_media_type=declared_media_type,
            maximum_edge=self._maximum_edge,
            jpeg_quality=self._jpeg_quality,
            crop_kinds=self._crop_kinds,
        )
        try:
            return await self._provider.complete_json(
                system_prompt=NVIDIA_GEO_SYSTEM_PROMPT,
                user_prompt=build_nvidia_geo_user_prompt(),
                authorization=authorization,
                images=prepared,
                response_model=NvidiaGeolocationReasoning,
                cancellation=cancellation,
            )
        finally:
            del prepared


def build_nvidia_geolocation_reasoner(
    *,
    enabled: bool,
    api_key: SecretStr | str | None,
    base_url: str = NVIDIA_BASE_URL,
    model: str = NVIDIA_VISION_MODEL,
    total_timeout_seconds: float = 90,
    max_transport_retries: int = 0,
    maximum_edge: int = 1_280,
    jpeg_quality: int = 82,
    crop_kinds: tuple[CropKind, ...] = ("full", "centre", "upper", "lower"),
    http_client: httpx.AsyncClient | None = None,
) -> NvidiaGeolocationReasoner:
    try:
        config = NvidiaVisionClientConfig(
            base_url=base_url,
            model=model,
            total_timeout_seconds=total_timeout_seconds,
            max_transport_retries=max_transport_retries,
        )
    except ValidationError as exc:
        raise NvidiaConfigurationError("nvidia_configuration_invalid") from exc
    provider = NvidiaVisionProvider(
        config,
        enabled=enabled,
        api_key=api_key,
        http_client=http_client,
    )
    return NvidiaGeolocationReasoner(
        enabled=enabled,
        api_key=api_key,
        maximum_edge=maximum_edge,
        jpeg_quality=jpeg_quality,
        crop_kinds=crop_kinds,
        provider=provider,
    )
