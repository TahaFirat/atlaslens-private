from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from uuid import UUID

from atlaslens_api.schemas import AnalysisEvent, AnalysisStatus, Progress

TERMINAL_EVENT_TYPES = {"completed", "failed", "deleted"}


class AnalysisEventBroker:
    def __init__(self, history_limit: int = 64) -> None:
        self._history_limit = history_limit
        self._history: dict[UUID, list[AnalysisEvent]] = defaultdict(list)
        self._subscribers: dict[UUID, set[asyncio.Queue[AnalysisEvent]]] = defaultdict(set)
        self._counters: dict[UUID, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def publish(
        self,
        analysis_id: UUID,
        event_type: str,
        status: AnalysisStatus,
        progress: Progress | None,
    ) -> AnalysisEvent:
        async with self._lock:
            self._counters[analysis_id] += 1
            event = AnalysisEvent(
                event_id=str(self._counters[analysis_id]),
                event_type=event_type,
                analysis_id=analysis_id,
                occurred_at=datetime.now(UTC),
                status=status,
                progress=progress,
            )
            history = self._history[analysis_id]
            history.append(event)
            if len(history) > self._history_limit:
                del history[: len(history) - self._history_limit]
            for queue in tuple(self._subscribers[analysis_id]):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # A slow client can reconnect and replay bounded history.
                    continue
            return event

    async def subscribe(
        self, analysis_id: UUID
    ) -> tuple[list[AnalysisEvent], asyncio.Queue[AnalysisEvent]]:
        queue: asyncio.Queue[AnalysisEvent] = asyncio.Queue(maxsize=32)
        async with self._lock:
            history = list(self._history.get(analysis_id, []))
            self._subscribers[analysis_id].add(queue)
        return history, queue

    async def unsubscribe(self, analysis_id: UUID, queue: asyncio.Queue[AnalysisEvent]) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(analysis_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    self._subscribers.pop(analysis_id, None)

    async def clear(self, analysis_id: UUID) -> None:
        async with self._lock:
            self._history.pop(analysis_id, None)
            self._counters.pop(analysis_id, None)

    @staticmethod
    def heartbeat(analysis_id: UUID, status: AnalysisStatus) -> AnalysisEvent:
        return AnalysisEvent(
            event_id=f"heartbeat-{int(datetime.now(UTC).timestamp())}",
            event_type="heartbeat",
            analysis_id=analysis_id,
            occurred_at=datetime.now(UTC),
            status=status,
        )
