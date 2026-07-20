"""Token-safe errors for the private Mapillary pilot."""

from __future__ import annotations


class MapillaryDemoError(RuntimeError):
    """An operator-safe failure that exposes only a stable code."""

    def __init__(
        self,
        code: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        if not code or not code.replace("_", "").isalnum():
            code = "mapillary_demo_failed"
        self.code = code.lower()
        self.retry_after_seconds = (
            retry_after_seconds
            if retry_after_seconds is not None and retry_after_seconds >= 0
            else None
        )
        super().__init__(self.code)


class MapillaryTokenError(MapillaryDemoError):
    """The runtime token is missing, a placeholder, or rejected."""


class MapillaryApiError(MapillaryDemoError):
    """The official Graph API returned an unusable response."""


class MapillaryLimitError(MapillaryDemoError):
    """A finite pilot safety limit was reached."""


class MapillarySafetyError(MapillaryDemoError):
    """A filesystem, URL, attribution, or integrity gate failed."""


__all__ = [
    "MapillaryApiError",
    "MapillaryDemoError",
    "MapillaryLimitError",
    "MapillarySafetyError",
    "MapillaryTokenError",
]
