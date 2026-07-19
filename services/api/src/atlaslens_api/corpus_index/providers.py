"""Descriptor provider protocol and explicit production-only registration."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, final

import numpy as np

from atlaslens_api.corpus_index.errors import (
    ProviderNotConfiguredError,
    ProviderRegistrationError,
)
from atlaslens_api.corpus_index.models import (
    DescriptorSpec,
    FloatMatrix,
    ProviderApproval,
    RuntimeKind,
)


class DescriptorProvider(Protocol):
    """Model-independent batch descriptor boundary."""

    @property
    def spec(self) -> DescriptorSpec: ...

    @property
    def runtime_kind(self) -> RuntimeKind: ...

    def describe_batch(self, locators: Sequence[Path]) -> FloatMatrix: ...


class ProductionDescriptorRegistry:
    """Registry with no fallback: only an explicitly approved adapter can activate."""

    def __init__(self) -> None:
        self._providers: dict[str, DescriptorProvider] = {}
        self._active_identity: str | None = None

    def register(self, provider: DescriptorProvider, approval: ProviderApproval) -> None:
        if provider.runtime_kind != "production":
            raise ProviderRegistrationError("test-only descriptor providers cannot be registered")
        spec = provider.spec
        if (
            approval.provider_id != spec.provider_id
            or approval.version != spec.version
            or approval.dimension != spec.dimension
            or (
                spec.artifact_sha256 is not None
                and approval.artifact_sha256 != spec.artifact_sha256
            )
        ):
            raise ProviderRegistrationError("provider approval does not match the adapter spec")
        if not approval.rights_approved:
            raise ProviderRegistrationError("descriptor provider rights are not approved")
        if spec.identity in self._providers:
            raise ProviderRegistrationError("descriptor provider is already registered")
        self._providers[spec.identity] = provider

    def activate(self, identity: str) -> None:
        if identity not in self._providers:
            raise ProviderNotConfiguredError("approved descriptor provider is not registered")
        self._active_identity = identity

    def active(self) -> DescriptorProvider:
        if self._active_identity is None:
            raise ProviderNotConfiguredError("no approved production descriptor provider is active")
        provider = self._providers.get(self._active_identity)
        if provider is None or provider.runtime_kind != "production":
            raise ProviderNotConfiguredError("active production descriptor provider is unavailable")
        return provider


@final
class TestOnlyDeterministicDescriptorProvider:
    """Synthetic-fixture helper; structurally ineligible for production registration."""

    __slots__ = ("_spec",)

    def __init__(self, *, dimension: int = 8, version: str = "synthetic-v1") -> None:
        self._spec = DescriptorSpec(
            provider_id="atlaslens-test-only-deterministic",
            version=version,
            dimension=dimension,
        )

    @property
    def spec(self) -> DescriptorSpec:
        return self._spec

    @property
    def runtime_kind(self) -> RuntimeKind:
        return "test_only"

    def describe_batch(self, locators: Sequence[Path]) -> FloatMatrix:
        rows: list[list[float]] = []
        for locator in locators:
            if not locator.is_file() or locator.is_symlink():
                raise ValueError("synthetic descriptor fixture must be a regular file")
            seed = locator.read_bytes()
            values = bytearray()
            counter = 0
            while len(values) < self._spec.dimension:
                values.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
                counter += 1
            rows.append([float(value) + 1.0 for value in values[: self._spec.dimension]])
        return np.asarray(rows, dtype=np.float32).reshape(len(rows), self._spec.dimension)
