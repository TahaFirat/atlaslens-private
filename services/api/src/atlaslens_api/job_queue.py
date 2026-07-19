from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

_logger = logging.getLogger("atlaslens_api.job_queue")


class JobQueueFullError(RuntimeError):
    pass


JobHandler = Callable[[asyncio.Event], Awaitable[None]]


class JobQueue(Protocol):
    async def start(self) -> None: ...

    async def submit(self, job_id: UUID, handler: JobHandler) -> None: ...

    async def cancel(self, job_id: UUID) -> None: ...

    async def shutdown(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _QueuedJob:
    id: UUID
    handler: JobHandler
    cancellation: asyncio.Event


class InProcessJobQueue:
    """Bounded, non-durable Phase 1 queue behind the replaceable JobQueue API."""

    def __init__(self, max_jobs: int, workers: int = 2) -> None:
        self._queue: asyncio.Queue[_QueuedJob] = asyncio.Queue(maxsize=max_jobs)
        self._worker_count = min(workers, max_jobs)
        self._workers: list[asyncio.Task[None]] = []
        self._cancellations: dict[UUID, asyncio.Event] = {}

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._worker(), name=f"atlaslens-job-worker-{index}")
            for index in range(self._worker_count)
        ]

    async def submit(self, job_id: UUID, handler: JobHandler) -> None:
        cancellation = asyncio.Event()
        job = _QueuedJob(id=job_id, handler=handler, cancellation=cancellation)
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise JobQueueFullError from exc
        self._cancellations[job_id] = cancellation

    async def cancel(self, job_id: UUID) -> None:
        cancellation = self._cancellations.get(job_id)
        if cancellation is not None:
            cancellation.set()

    async def shutdown(self) -> None:
        for cancellation in self._cancellations.values():
            cancellation.set()
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._cancellations.clear()

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await job.handler(job.cancellation)
            except asyncio.CancelledError:
                raise
            except Exception:
                _logger.error(
                    "job_worker_guard_triggered",
                    extra={"event": "job_worker_guard_triggered"},
                )
            finally:
                self._cancellations.pop(job.id, None)
                self._queue.task_done()
