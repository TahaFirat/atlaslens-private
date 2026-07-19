"""Phase 3B rights-gated descriptor, index, and benchmark components."""

from atlaslens_api.corpus_index.descriptors import (
    DescriptorBuildResult,
    DescriptorDataset,
    build_descriptors,
    build_production_descriptors,
)
from atlaslens_api.corpus_index.index import PublishedCorpusIndex, build_index
from atlaslens_api.corpus_index.models import (
    AssetProvenance,
    DescriptorAsset,
    DescriptorRecord,
    DescriptorSpec,
    ProviderApproval,
    SearchHit,
)
from atlaslens_api.corpus_index.providers import (
    DescriptorProvider,
    ProductionDescriptorRegistry,
)

__all__ = [
    "AssetProvenance",
    "DescriptorAsset",
    "DescriptorBuildResult",
    "DescriptorDataset",
    "DescriptorProvider",
    "DescriptorRecord",
    "DescriptorSpec",
    "ProductionDescriptorRegistry",
    "PublishedCorpusIndex",
    "ProviderApproval",
    "SearchHit",
    "build_descriptors",
    "build_index",
    "build_production_descriptors",
]
