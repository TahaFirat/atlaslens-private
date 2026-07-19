"""Fail-closed errors for the Phase 3B corpus index pipeline."""

from __future__ import annotations


class CorpusIndexError(RuntimeError):
    """Base class for corpus index failures safe to surface to an operator."""


class ProviderNotConfiguredError(CorpusIndexError):
    """Raised when production descriptor execution has no approved provider."""


class ProviderRegistrationError(CorpusIndexError):
    """Raised when a descriptor adapter is not eligible for production registration."""


class DescriptorValidationError(CorpusIndexError):
    """Raised when a provider returns an invalid descriptor."""


class CheckpointCompatibilityError(CorpusIndexError):
    """Raised when a descriptor checkpoint cannot be resumed safely."""


class ArtifactIntegrityError(CorpusIndexError):
    """Raised when a published artifact is missing, corrupt, or internally inconsistent."""


class IndexCompatibilityError(CorpusIndexError):
    """Raised when an index does not match the requested provider or policy version."""


class HoldoutIntegrityError(CorpusIndexError):
    """Raised when locked benchmark inputs do not match their recorded hash."""


class BenchmarkLeakageError(CorpusIndexError):
    """Raised before scoring when reference/holdout leakage is detected."""
