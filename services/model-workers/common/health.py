from __future__ import annotations

import re
from typing import Mapping

from common.protocol import JSONValue, PROTOCOL_VERSION, is_safe_code

_PROVIDER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
_DEVICE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")


def build_health_payload(
    raw: Mapping[str, JSONValue],
    *,
    provider: str,
    provider_revision: str,
    model_revision: str,
) -> dict[str, JSONValue]:
    """Validate adapter health and add process-owned immutable metadata."""

    if _PROVIDER.fullmatch(provider) is None:
        raise ValueError("worker provider id is invalid")
    resolved_model_revision = _bounded_text(raw.get("model_revision"), model_revision, 160)
    device = _bounded_text(raw.get("device"), "unknown", 32)
    if _DEVICE.fullmatch(device) is None:
        device = "unknown"
    last_error = raw.get("last_error")
    return {
        "schema_version": PROTOCOL_VERSION,
        "provider": provider,
        "provider_revision": _bounded_text(provider_revision, "unknown", 160),
        "model_revision": resolved_model_revision,
        "device": device,
        "process_running": True,
        "import_ok": raw.get("import_ok") is True,
        "weights_available": raw.get("weights_available") is True,
        "model_loaded": raw.get("model_loaded") is True,
        "load_verified": (
            raw.get("load_verified") is True or raw.get("model_loaded") is True
        ),
        "real_inference_verified": raw.get("real_inference_verified") is True,
        "last_error": last_error if is_safe_code(last_error) else None,
    }


def _bounded_text(value: object, fallback: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        return fallback
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return fallback
    return value
