from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Protocol

from atlaslens_api.repository import AnalysisRepository
from atlaslens_api.storage import StorageBackend

_logger = logging.getLogger("atlaslens_api.cleanup")


class RetentionCleanupService(Protocol):
    async def startup_cleanup(self) -> None: ...

    async def run_once(self) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class DefaultRetentionCleanupService:
    def __init__(
        self,
        repository: AnalysisRepository,
        storage: StorageBackend,
        *,
        ttl_seconds: int,
        interval_seconds: float = 60.0,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._ttl_seconds = ttl_seconds
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None

    async def startup_cleanup(self) -> None:
        interrupted = await self._repository.fail_incomplete()
        for stored in interrupted:
            await self._storage.delete(stored.storage_key)
            await self._repository.clear_storage_key(stored.analysis.id)
        await self.run_once()

    async def run_once(self) -> None:
        now = datetime.now(UTC)
        for stored in await self._repository.list_expired(now):
            await self._storage.delete(stored.storage_key)
            await self._repository.delete(stored.analysis.id)
        orphan_age = max(self._ttl_seconds, 60)
        await self._storage.cleanup_orphans(now - timedelta(seconds=orphan_age))

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._periodic(), name="atlaslens-retention-cleanup")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _periodic(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                _logger.warning(
                    "retention_cleanup_failed",
                    extra={"event": "retention_cleanup_failed"},
                )
