"""Rights-gated offline MegaLoc descriptor provider."""

from atlaslens_api.megaloc_adapter.errors import MegaLocAdapterError
from atlaslens_api.megaloc_adapter.models import (
    MEGALOC_DESCRIPTOR_DIMENSION,
    MEGALOC_PREPROCESSING_ID,
    MegaLocAdapterConfig,
    MegaLocApproval,
    MegaLocReadiness,
    MegaLocSmokeReceipt,
    load_megaloc_config,
    write_megaloc_smoke_receipt,
)
from atlaslens_api.megaloc_adapter.provider import (
    MegaLocBackend,
    MegaLocBackendFactory,
    MegaLocDescriptorProvider,
    inspect_megaloc_approval,
)

__all__ = [
    "MEGALOC_DESCRIPTOR_DIMENSION",
    "MEGALOC_PREPROCESSING_ID",
    "MegaLocAdapterConfig",
    "MegaLocAdapterError",
    "MegaLocApproval",
    "MegaLocBackend",
    "MegaLocBackendFactory",
    "MegaLocDescriptorProvider",
    "MegaLocReadiness",
    "MegaLocSmokeReceipt",
    "inspect_megaloc_approval",
    "load_megaloc_config",
    "write_megaloc_smoke_receipt",
]
