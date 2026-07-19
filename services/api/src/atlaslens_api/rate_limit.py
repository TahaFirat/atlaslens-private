from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    def __init__(self, period_seconds: float = 60.0) -> None:
        self._period = period_seconds
        self._entries: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str, limit: int) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self._period
        async with self._lock:
            entries = self._entries[key]
            while entries and entries[0] <= cutoff:
                entries.popleft()
            if len(entries) >= limit:
                retry_after = max(1, int(self._period - (now - entries[0])) + 1)
                return False, retry_after
            entries.append(now)
            return True, 0


class ConnectionLimiter:
    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def acquire(self, key: str, limit: int) -> bool:
        async with self._lock:
            if self._counts[key] >= limit:
                return False
            self._counts[key] += 1
            return True

    async def release(self, key: str) -> None:
        async with self._lock:
            if self._counts.get(key, 0) <= 1:
                self._counts.pop(key, None)
            else:
                self._counts[key] -= 1
