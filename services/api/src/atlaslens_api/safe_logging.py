from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

_SECRET = re.compile(r"\b(?:sk|key)-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE)
_ALLOWED_EXTRA = {
    "event",
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "error_code",
}


def _safe_text(value: object) -> str:
    text = str(value)
    return _SECRET.sub("[redacted-secret]", text)[:256]


class SafeJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": _safe_text(getattr(record, "event", record.msg)),
        }
        for field in _ALLOWED_EXTRA - {"event"}:
            if hasattr(record, field):
                payload[field] = _safe_text(getattr(record, field))
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(SafeJsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
