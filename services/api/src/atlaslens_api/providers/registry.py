from __future__ import annotations

from pathlib import Path

from atlaslens_api.model_management.manifest import default_manifest_path, load_manifest
from atlaslens_api.model_management.service import ModelManagementService

from .base import GlobalGeolocationProvider
from .geoclip import GeoCLIPGlobalGeolocationProvider


class GlobalProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, GlobalGeolocationProvider] = {}

    def register(self, provider: GlobalGeolocationProvider) -> None:
        if provider.descriptor.id in self._providers:
            raise ValueError("duplicate global provider")
        self._providers[provider.descriptor.id] = provider

    def get(self, provider_id: str) -> GlobalGeolocationProvider:
        return self._providers[provider_id]

    def statuses(self) -> dict[str, object]:
        return {provider_id: provider.status() for provider_id, provider in self._providers.items()}


def build_global_provider_registry(cache_root: Path) -> GlobalProviderRegistry:
    management = ModelManagementService(cache_root, load_manifest(default_manifest_path()))
    registry = GlobalProviderRegistry()
    registry.register(GeoCLIPGlobalGeolocationProvider(management))
    return registry
