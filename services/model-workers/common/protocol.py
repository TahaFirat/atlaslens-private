from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Protocol, TypeAlias

PROTOCOL_VERSION = "atlaslens-worker-v1"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,119}$")
_PARAMETER_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class WorkerAdapterError(RuntimeError):
    """Adapter failure carrying only a bounded code safe for the RPC boundary."""

    def __init__(self, code: str, *, status: int = 503) -> None:
        if not is_safe_code(code):
            raise ValueError("worker error code is invalid")
        if status not in {400, 409, 413, 422, 500, 503}:
            raise ValueError("worker error status is invalid")
        super().__init__(code)
        self.code = code
        self.status = status


class WorkerAdapter(Protocol):
    """Synchronous model boundary owned by an isolated worker process."""

    def health(self) -> Mapping[str, JSONValue]: ...

    def load(self, parameters: dict[str, JSONValue]) -> Mapping[str, JSONValue]: ...

    def infer(
        self,
        image_bytes: bytes,
        parameters: dict[str, JSONValue],
    ) -> Mapping[str, JSONValue]: ...

    def unload(self, parameters: dict[str, JSONValue]) -> Mapping[str, JSONValue]: ...


@dataclass(frozen=True)
class WorkerRequest:
    request_id: str
    parameters: dict[str, JSONValue]
    image_base64: str | None


class WorkerProtocolError(ValueError):
    def __init__(self, code: str, *, status: int = 400) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def is_safe_code(value: object) -> bool:
    return isinstance(value, str) and _SAFE_CODE.fullmatch(value) is not None


def safe_adapter_error(exc: BaseException) -> tuple[str, int]:
    code = getattr(exc, "code", None)
    if not is_safe_code(code):
        return "worker_operation_failed", 500
    status = getattr(exc, "status", 503)
    if not isinstance(status, int) or status not in {400, 409, 413, 422, 500, 503}:
        status = 503
    return code, status


def parse_request(body: bytes, *, require_image: bool) -> WorkerRequest:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError("invalid_json") from exc
    if not isinstance(payload, dict):
        raise WorkerProtocolError("invalid_request_envelope")
    allowed = {"schema_version", "request_id", "parameters", "image_base64"}
    if set(payload) - allowed:
        raise WorkerProtocolError("unknown_request_field")
    if payload.get("schema_version") != PROTOCOL_VERSION:
        raise WorkerProtocolError("unsupported_protocol_version")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
        raise WorkerProtocolError("invalid_request_id")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict) or len(parameters) > 20:
        raise WorkerProtocolError("invalid_parameters")
    normalized: dict[str, JSONValue] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or _PARAMETER_KEY.fullmatch(key) is None:
            raise WorkerProtocolError("invalid_parameter_key")
        _validate_json_value(value, depth=0)
        normalized[key] = value
    encoded = payload.get("image_base64")
    if require_image and not isinstance(encoded, str):
        raise WorkerProtocolError("image_required")
    if not require_image and encoded is not None:
        raise WorkerProtocolError("image_not_allowed")
    return WorkerRequest(
        request_id=request_id,
        parameters=normalized,
        image_base64=encoded if isinstance(encoded, str) else None,
    )


def encode_json(payload: Mapping[str, JSONValue], *, max_bytes: int) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkerProtocolError("invalid_adapter_output", status=500) from exc
    if len(encoded) > max_bytes:
        raise WorkerProtocolError("response_too_large", status=500)
    return encoded


def _validate_json_value(value: object, *, depth: int) -> None:
    if depth > 4:
        raise WorkerProtocolError("parameter_nesting_too_deep")
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise WorkerProtocolError("non_finite_parameter")
        return
    if isinstance(value, str):
        if len(value) > 240 or any(ord(character) < 32 for character in value):
            raise WorkerProtocolError("invalid_parameter_string")
        return
    if isinstance(value, list):
        if len(value) > 256:
            raise WorkerProtocolError("parameter_list_too_large")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 32:
            raise WorkerProtocolError("parameter_object_too_large")
        for key, item in value.items():
            if not isinstance(key, str) or _PARAMETER_KEY.fullmatch(key) is None:
                raise WorkerProtocolError("invalid_parameter_key")
            _validate_json_value(item, depth=depth + 1)
        return
    raise WorkerProtocolError("invalid_parameter_type")
