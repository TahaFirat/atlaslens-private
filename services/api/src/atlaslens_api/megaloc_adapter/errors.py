"""Stable, path-free failures for the offline MegaLoc corpus adapter."""

from __future__ import annotations


class MegaLocAdapterError(RuntimeError):
    """Fail-closed adapter error whose message is a non-sensitive reason code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)
