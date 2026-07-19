from __future__ import annotations

from pathlib import Path

from atlaslens_api.model_management.manifest import (
    load_embedding_manifest,
    siglip2_manifest_path,
)
from atlaslens_api.model_management.paths import default_cache_root
from atlaslens_api.model_management.service import EmbeddingModelManagementService
from atlaslens_api.retrieval.errors import ProviderUnavailableError
from atlaslens_api.retrieval.models import Embedding, EmbeddingSpec
from atlaslens_api.retrieval.protocols import EmbeddingProvider
from atlaslens_api.retrieval.siglip2 import Siglip2EmbeddingProvider


class UnavailableEmbeddingProvider:
    def __init__(self, spec: EmbeddingSpec, reason: str) -> None:
        self._spec = spec
        self._reason = reason

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    @property
    def available(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    def embed(self, image_path: Path) -> Embedding:
        del image_path
        raise ProviderUnavailableError(self._reason)


def production_embedding_provider(
    name: str,
    *,
    cache_root: Path | None = None,
    requested_device: str = "auto",
) -> EmbeddingProvider:
    if name in {"siglip2", "siglip2-b16-384"}:
        management = EmbeddingModelManagementService(
            cache_root or default_cache_root(),
            load_embedding_manifest(siglip2_manifest_path()),
        )
        return Siglip2EmbeddingProvider(
            management, requested_device=requested_device
        )
    providers = {
        "disabled": UnavailableEmbeddingProvider(
            EmbeddingSpec(provider="disabled", version="1", dimension=1),
            "embedding provider is disabled",
        ),
        "clip": UnavailableEmbeddingProvider(
            EmbeddingSpec(provider="clip", version="future", dimension=768),
            "CLIP is a future provider and has not been installed or approved",
        ),
    }
    try:
        return providers[name]
    except KeyError as exc:
        raise ProviderUnavailableError("unknown embedding provider") from exc
