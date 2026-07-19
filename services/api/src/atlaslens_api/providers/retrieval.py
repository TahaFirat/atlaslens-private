from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable

from atlaslens_api.providers.base import (
    InvocationContext,
    ProviderDescriptor,
    ProviderOutcome,
)
from atlaslens_api.retrieval.errors import RetrievalError
from atlaslens_api.retrieval.models import RetrievalHit
from atlaslens_api.retrieval.service import RetrievalQueryService
from atlaslens_api.storage import LocalImageHandle


class FaissRetrievalProvider:
    """Bounded local retrieval adapter for the preserved image-facing provider seam."""

    def __init__(
        self,
        service: RetrievalQueryService | None,
        *,
        enabled: bool,
        top_k: int = 25,
        timeout_seconds: float = 20.0,
        max_input_bytes: int = 20 * 1024 * 1024,
        unavailable_reason: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= top_k <= 100:
            raise ValueError("retrieval Top-K must be between 1 and 100")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("retrieval timeout must be positive and finite")
        if max_input_bytes <= 0:
            raise ValueError("retrieval input bound must be positive")
        reason = "disabled" if not enabled else unavailable_reason
        if enabled and service is None and reason is None:
            reason = "index_unavailable"
        self._service = service
        self._top_k = top_k
        self._timeout = timeout_seconds
        self._max_input_bytes = max_input_bytes
        self._clock = clock
        self.descriptor = ProviderDescriptor(
            id="faiss-siglip2-local-retrieval",
            kind="image_retrieval",
            version="phase5b-v1",
            execution_boundary="local",
            criticality="optional",
            available=reason is None and service is not None,
            unavailable_reason_code=reason,
            model_name="SigLIP2 B/16-384 + FAISS",
        )

    async def search(
        self, handle: LocalImageHandle, context: InvocationContext
    ) -> ProviderOutcome[object]:
        if not self.descriptor.available or self._service is None:
            return ProviderOutcome.skipped("unavailable")
        if context.cancellation.is_set():
            return ProviderOutcome.skipped("unavailable")
        started = self._clock()
        try:
            size = handle.path.stat().st_size
            if not 0 < size <= self._max_input_bytes:
                raise ValueError("retrieval input exceeds its byte bound")
            payload = handle.path.read_bytes()
            remaining = (context.deadline.timestamp() - time.time())
            timeout = min(self._timeout, max(0.05, remaining))
            hits = await asyncio.wait_for(
                asyncio.to_thread(
                    self._service.retrieve_image_bytes, payload, self._top_k
                ),
                timeout=timeout,
            )
            del payload
        except TimeoutError:
            return ProviderOutcome.failed(
                "inference_timeout",
                retryable=False,
                attempts=1,
                duration_ms=self._duration_ms(started),
                subreason_code="retrieval_timeout",
            )
        except (OSError, RetrievalError, ValueError):
            return ProviderOutcome.failed(
                "internal_provider_error",
                retryable=False,
                attempts=1,
                duration_ms=self._duration_ms(started),
                subreason_code="retrieval_query_failed",
            )
        return ProviderOutcome.succeeded(list(hits))

    def diagnostics(self) -> dict[str, object]:
        if self._service is None:
            return {
                "status": "disabled"
                if self.descriptor.unavailable_reason_code == "disabled"
                else "unavailable",
                "index_size": 0,
                "embedding_provider": None,
                "embedding_version": None,
                "dimension": None,
                "provider_available": False,
            }
        return self._service.diagnostics()

    def close(self) -> None:
        if self._service is not None:
            self._service.close()

    def _duration_ms(self, started: float) -> int:
        return max(0, int((self._clock() - started) * 1000))


def validated_retrieval_hits(value: object, *, limit: int = 100) -> tuple[RetrievalHit, ...]:
    if not isinstance(value, list) or len(value) > limit:
        return ()
    if not all(isinstance(item, RetrievalHit) for item in value):
        return ()
    return tuple(value)
