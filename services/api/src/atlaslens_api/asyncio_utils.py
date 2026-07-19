from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


def run_isolated[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a CLI coroutine without replacing the process-global event loop."""

    with asyncio.Runner(loop_factory=asyncio.new_event_loop) as runner:
        return runner.run(coroutine)
