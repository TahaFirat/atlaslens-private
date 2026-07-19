"""Sanitized NVIDIA vision-provider errors.

Only stable reason codes are exposed.  Upstream response bodies, request payloads,
credentials and local image identities must never be attached to these exceptions.
"""

from __future__ import annotations


class NvidiaVisionError(Exception):
    """Base class whose string and repr forms contain only a stable reason code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r})"


class NvidiaConfigurationError(NvidiaVisionError):
    """The provider configuration violates a fail-closed safety rule."""


class NvidiaProviderUnavailableError(NvidiaVisionError):
    """The optional provider is disabled or lacks usable credentials."""


class NvidiaPreprocessingError(NvidiaVisionError):
    """An image could not be converted into a bounded metadata-free derivative."""


class NvidiaTransportError(NvidiaVisionError):
    """A sanitized HTTP transport failure."""


class NvidiaAuthenticationError(NvidiaTransportError):
    """The upstream rejected authentication without exposing its response body."""


class NvidiaModelUnavailableError(NvidiaTransportError):
    """The configured endpoint or exact model is not available."""


class NvidiaRateLimitError(NvidiaTransportError):
    """The bounded retry policy exhausted an upstream rate limit."""

    def __init__(self, code: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(code)
        self.retry_after_seconds = retry_after_seconds

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code!r}, "
            f"retry_after_seconds={self.retry_after_seconds!r})"
        )


class NvidiaResponseError(NvidiaVisionError):
    """The upstream envelope was malformed or exceeded a safety limit."""


class NvidiaSchemaError(NvidiaResponseError):
    """The model content did not satisfy the caller-provided Pydantic schema."""


class NvidiaCircuitOpenError(NvidiaTransportError):
    """Network use is blocked by the local circuit breaker."""


class NvidiaTimeoutError(NvidiaTransportError):
    """The total bounded operation deadline elapsed."""


class NvidiaCancelledError(NvidiaVisionError):
    """The owning AtlasLens analysis cancelled the cloud operation."""
