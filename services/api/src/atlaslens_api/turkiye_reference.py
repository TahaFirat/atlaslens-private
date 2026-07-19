"""Disabled-by-default API seam for a verified Turkiye corpus index."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from atlaslens_api.corpus_index.errors import ArtifactIntegrityError, IndexCompatibilityError
from atlaslens_api.corpus_index.index import PublishedCorpusIndex
from atlaslens_api.corpus_index.models import DescriptorSpec, FloatVector, SearchHit

TurkiyeReferenceState = Literal["disabled", "not_ready", "ready"]


@dataclass(frozen=True, slots=True)
class TurkiyeReferenceStatus:
    """Safe capability projection with no private filesystem details."""

    state: TurkiyeReferenceState
    enabled: bool
    available: bool
    reason_code: str | None
    backend: str | None = None
    descriptor_provider: str | None = None
    descriptor_version: str | None = None
    dimension: int | None = None
    asset_count: int = 0


class TurkiyeReferenceUnavailable(RuntimeError):
    """Raised when retrieval is attempted before every compatibility gate passes."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class TurkiyeReferenceIndexProvider:
    """Open and query only a checksum-verified, policy-bound published index.

    The seam consumes an explicit query vector. It never selects or substitutes a
    descriptor model and is deliberately not attached to AtlasLens fusion/ranking.
    """

    provider_id = "turkiye-reference-index-v1"

    def __init__(
        self,
        *,
        enabled: bool,
        index_path: Path,
        source_policy_path: Path,
        descriptor_provider: str,
        descriptor_version: str,
        descriptor_dimension: int,
        descriptor_artifact_sha256: str = "",
        descriptor_preprocessing_version: str = "",
        descriptor_artifact_approved: bool = False,
    ) -> None:
        self._enabled = enabled
        self._index_path = index_path
        self._source_policy_path = source_policy_path
        self._descriptor_provider = descriptor_provider.strip()
        self._descriptor_version = descriptor_version.strip()
        self._descriptor_dimension = descriptor_dimension
        self._descriptor_artifact_sha256 = descriptor_artifact_sha256.strip()
        self._descriptor_preprocessing_version = descriptor_preprocessing_version.strip()
        self._descriptor_artifact_approved = descriptor_artifact_approved

    def status(self) -> TurkiyeReferenceStatus:
        """Revalidate publication, checksums, policy and descriptor compatibility."""
        status, _ = self._inspect()
        return status

    def search_vector(
        self,
        query: Sequence[float] | FloatVector,
        *,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Return ranked distances and governed reference provenance, never confidence."""
        status, index = self._inspect()
        if index is None:
            raise TurkiyeReferenceUnavailable(status.reason_code or "index_not_ready")
        vector = np.asarray(query, dtype=np.float32)
        return index.search(vector, top_k=top_k)

    def _inspect(self) -> tuple[TurkiyeReferenceStatus, PublishedCorpusIndex | None]:
        if not self._enabled:
            return self._status("disabled", "explicitly_disabled"), None
        try:
            expected_spec = self._expected_spec()
        except ValueError:
            return self._status("not_ready", "descriptor_configuration_invalid"), None
        if self._descriptor_provider.startswith("atlaslens-test-only-"):
            return self._status("not_ready", "test_descriptor_provider_forbidden"), None
        if self._descriptor_provider == "megaloc" and (
            not self._descriptor_artifact_approved
            or not self._descriptor_artifact_sha256
            or not self._descriptor_preprocessing_version
        ):
            return self._status("not_ready", "descriptor_artifact_not_approved"), None
        if expected_spec is None:
            return self._status("not_ready", "descriptor_configuration_missing"), None
        source_policy_hash = _safe_policy_hash(self._source_policy_path)
        if source_policy_hash is None:
            return self._status("not_ready", "source_policy_unavailable"), None
        if self._index_path.is_symlink() or not self._index_path.is_dir():
            return self._status("not_ready", "index_artifacts_missing"), None
        try:
            index = PublishedCorpusIndex.open(
                self._index_path,
                expected_spec=expected_spec,
                expected_source_policy_hash=source_policy_hash,
            )
        except (ArtifactIntegrityError, IndexCompatibilityError, OSError, ValueError):
            return self._status("not_ready", "index_artifacts_incompatible"), None
        return (
            TurkiyeReferenceStatus(
                state="ready",
                enabled=True,
                available=True,
                reason_code=None,
                backend=index.backend,
                descriptor_provider=index.spec.provider_id,
                descriptor_version=index.spec.version,
                dimension=index.spec.dimension,
                asset_count=index.size,
            ),
            index,
        )

    def _expected_spec(self) -> DescriptorSpec | None:
        if (
            not self._descriptor_provider
            or not self._descriptor_version
            or self._descriptor_dimension <= 0
        ):
            return None
        return DescriptorSpec(
            provider_id=self._descriptor_provider,
            version=self._descriptor_version,
            dimension=self._descriptor_dimension,
            metric="cosine",
            artifact_sha256=self._descriptor_artifact_sha256 or None,
            preprocessing_version=self._descriptor_preprocessing_version or None,
        )

    def _status(self, state: TurkiyeReferenceState, reason_code: str) -> TurkiyeReferenceStatus:
        return TurkiyeReferenceStatus(
            state=state,
            enabled=self._enabled,
            available=False,
            reason_code=reason_code,
            descriptor_provider=self._descriptor_provider or None,
            descriptor_version=self._descriptor_version or None,
            dimension=self._descriptor_dimension or None,
        )


def _safe_policy_hash(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(payload).hexdigest()
