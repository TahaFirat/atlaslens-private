"""Phase 3 licensed-image retrieval boundaries and local FAISS implementation."""

from atlaslens_api.retrieval.faiss_index import (
    FaissImageIndex,
    MilvusImageIndex,
    QdrantImageIndex,
)
from atlaslens_api.retrieval.manifest import MANIFEST_COLUMNS, validate_manifest
from atlaslens_api.retrieval.metadata import SQLAlchemyImageMetadataRepository
from atlaslens_api.retrieval.models import (
    Embedding,
    EmbeddingSpec,
    ImageMetadata,
    ImportSummary,
    ProviderDiagnostics,
    RetrievalHit,
)
from atlaslens_api.retrieval.providers import (
    UnavailableEmbeddingProvider,
    production_embedding_provider,
)
from atlaslens_api.retrieval.service import (
    FaissNearestNeighborEngine,
    ManifestDatasetImporter,
    RetrievalQueryService,
    open_retrieval_query_service,
)
from atlaslens_api.retrieval.siglip2 import Siglip2EmbeddingProvider

__all__ = [
    "MANIFEST_COLUMNS",
    "Embedding",
    "EmbeddingSpec",
    "FaissImageIndex",
    "FaissNearestNeighborEngine",
    "ImageMetadata",
    "ImportSummary",
    "ManifestDatasetImporter",
    "MilvusImageIndex",
    "ProviderDiagnostics",
    "QdrantImageIndex",
    "RetrievalQueryService",
    "RetrievalHit",
    "SQLAlchemyImageMetadataRepository",
    "UnavailableEmbeddingProvider",
    "Siglip2EmbeddingProvider",
    "open_retrieval_query_service",
    "production_embedding_provider",
    "validate_manifest",
]
