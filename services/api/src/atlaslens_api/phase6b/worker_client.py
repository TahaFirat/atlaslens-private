from __future__ import annotations

import asyncio
import base64
import contextlib
import http.client
import json
import math
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from atlaslens_api.phase6b.ocr import RawPaddleOCRLine
from atlaslens_api.phase6b.plonk import PlonkWorkerOutput

_PROTOCOL_VERSION = "atlaslens-worker-v1"
_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,119}$")
_SAFE_PROVIDER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
_SAFE_PROFILE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
_CONNECTION_ERRORS = (OSError, TimeoutError, http.client.HTTPException)


class _WorkerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class WorkerHealth(_WorkerModel):
    schema_version: Literal["atlaslens-worker-v1"]
    provider: str = Field(min_length=1, max_length=80)
    provider_revision: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    device: str = Field(min_length=1, max_length=32)
    process_running: bool
    import_ok: bool
    weights_available: bool
    model_loaded: bool
    load_verified: bool
    real_inference_verified: bool
    last_error: str | None = Field(default=None, max_length=120)

    @property
    def ready(self) -> bool:
        return all(
            (
                self.process_running,
                self.import_ok,
                self.weights_available,
                self.load_verified,
                self.real_inference_verified,
            )
        )


class WorkerRPCResponse(_WorkerModel):
    schema_version: Literal["atlaslens-worker-v1"]
    ok: Literal[True]
    request_id: str = Field(min_length=8, max_length=128)
    provider: str = Field(min_length=1, max_length=80)
    provider_revision: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    device: str = Field(min_length=1, max_length=32)
    duration_ms: int = Field(ge=0)
    result: dict[str, Any]


class WorkerClientError(RuntimeError):
    """Local RPC failure carrying no URL, path, stack, or provider payload."""

    def __init__(self, code: str) -> None:
        safe = code if _SAFE_CODE.fullmatch(code) is not None else "worker_client_failed"
        # Preserve the scheduler's existing one-time CUDA OOM fallback detection
        # without exposing native exception text from the isolated process.
        super().__init__("cuda out of memory" if safe == "cuda_out_of_memory" else safe)
        self.code = safe


class _ConnectionOpenFailure(OSError):
    pass


ConnectionFactory = Callable[[str, int, float], http.client.HTTPConnection]


def _default_connection_factory(
    host: str, port: int, timeout_seconds: float
) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(host=host, port=port, timeout=timeout_seconds)


class LocalWorkerHTTPClient:
    """Bounded no-proxy HTTP client for one 127.0.0.1 AtlasLens worker."""

    def __init__(
        self,
        *,
        provider: str,
        host: str,
        port: int,
        provider_revision: str | None = None,
        model_revisions: Sequence[str] = (),
        timeout_seconds: float = 60.0,
        max_image_bytes: int = 20 * 1024 * 1024,
        max_request_bytes: int = 30 * 1024 * 1024,
        max_response_bytes: int = 2 * 1024 * 1024,
        connection_factory: ConnectionFactory = _default_connection_factory,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("model-worker clients may connect only to 127.0.0.1")
        if _SAFE_PROVIDER.fullmatch(provider) is None:
            raise ValueError("worker provider id is invalid")
        if not 1 <= port <= 65_535:
            raise ValueError("worker port is invalid")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600:
            raise ValueError("worker timeout is invalid")
        if not 1 <= max_image_bytes <= max_request_bytes <= 64 * 1024 * 1024:
            raise ValueError("worker request bounds are invalid")
        if not 1 <= max_response_bytes <= 8 * 1024 * 1024:
            raise ValueError("worker response bound is invalid")
        revisions = frozenset(model_revisions)
        if any(not revision or len(revision) > 160 for revision in revisions):
            raise ValueError("worker model revisions are invalid")
        if provider_revision is not None and (
            not provider_revision or len(provider_revision) > 160
        ):
            raise ValueError("worker provider revision is invalid")
        self._provider = provider
        self._host = host
        self._port = port
        self._provider_revision = provider_revision
        self._model_revisions = revisions
        self._timeout = timeout_seconds
        self._max_image_bytes = max_image_bytes
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._connection_factory = connection_factory
        self._active: set[http.client.HTTPConnection] = set()
        self._active_lock = threading.Lock()
        self._closed = False

    async def health(
        self,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> WorkerHealth:
        payload = await self._request(
            "GET",
            "/health",
            body=None,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )
        return self._parse_health(payload)

    def health_sync(self, *, timeout_seconds: float = 1.0) -> WorkerHealth:
        """Probe worker health during synchronous application composition."""

        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= self._timeout:
            raise ValueError("health timeout exceeds configured worker timeout")
        payload = self._request_sync(
            "GET",
            "/health",
            None,
            time.monotonic() + timeout_seconds,
        )
        return self._parse_health(payload)

    async def load(
        self,
        parameters: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> WorkerRPCResponse:
        return await self._post(
            "/v1/load",
            parameters,
            image_bytes=None,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )

    async def infer(
        self,
        image_bytes: bytes,
        parameters: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> WorkerRPCResponse:
        if type(image_bytes) is not bytes or not 0 < len(image_bytes) <= self._max_image_bytes:
            raise WorkerClientError("invalid_image_payload")
        return await self._post(
            "/v1/infer",
            parameters,
            image_bytes=image_bytes,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )

    async def unload(
        self,
        parameters: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> WorkerRPCResponse:
        return await self._post(
            "/v1/unload",
            parameters,
            image_bytes=None,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await self.unload({}, timeout_seconds=min(5.0, self._timeout))
        except WorkerClientError:
            pass
        finally:
            self._closed = True
            self._abort_active_connections()

    async def _post(
        self,
        path: str,
        parameters: Mapping[str, Any],
        *,
        image_bytes: bytes | None,
        timeout_seconds: float | None,
        cancellation: asyncio.Event | None,
    ) -> WorkerRPCResponse:
        request_id = f"worker-{secrets.token_hex(16)}"
        envelope: dict[str, Any] = {
            "schema_version": _PROTOCOL_VERSION,
            "request_id": request_id,
            "parameters": dict(parameters),
        }
        if image_bytes is not None:
            envelope["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
        try:
            body = json.dumps(
                envelope,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise WorkerClientError("invalid_worker_parameters") from exc
        if len(body) > self._max_request_bytes:
            raise WorkerClientError("worker_request_too_large")
        payload = await self._request(
            "POST",
            path,
            body=body,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )
        try:
            response = WorkerRPCResponse.model_validate(payload)
        except ValueError as exc:
            raise WorkerClientError("invalid_worker_response") from exc
        if response.request_id != request_id:
            raise WorkerClientError("worker_request_id_mismatch")
        self._validate_identity(
            response.provider,
            response.provider_revision,
            response.model_revision,
            validate_model_revision=True,
        )
        return response

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        timeout_seconds: float | None,
        cancellation: asyncio.Event | None,
    ) -> dict[str, Any]:
        if self._closed:
            raise WorkerClientError("worker_client_closed")
        timeout = self._timeout if timeout_seconds is None else timeout_seconds
        if not math.isfinite(timeout) or not 0 < timeout <= self._timeout:
            raise ValueError("request timeout exceeds configured worker timeout")
        if cancellation is not None and cancellation.is_set():
            raise WorkerClientError("cancelled")
        deadline = time.monotonic() + timeout
        operation = asyncio.create_task(
            asyncio.to_thread(self._request_sync, method, path, body, deadline)
        )
        cancellation_task = (
            asyncio.create_task(cancellation.wait()) if cancellation is not None else None
        )
        waiters: set[asyncio.Task[Any]] = {operation}
        if cancellation_task is not None:
            waiters.add(cancellation_task)
        try:
            done, _ = await asyncio.wait(
                waiters,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_task is not None and cancellation_task in done:
                self._abort_active_connections()
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                raise WorkerClientError("cancelled")
            if operation not in done:
                self._abort_active_connections()
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                raise WorkerClientError("worker_timeout")
            return await operation
        except asyncio.CancelledError:
            self._abort_active_connections()
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            raise
        finally:
            if cancellation_task is not None:
                cancellation_task.cancel()
                await asyncio.gather(cancellation_task, return_exceptions=True)

    def _request_sync(
        self,
        method: str,
        path: str,
        body: bytes | None,
        deadline: float,
    ) -> dict[str, Any]:
        connection: http.client.HTTPConnection | None = None
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerClientError("worker_timeout")
            candidate = self._connection_factory(self._host, self._port, remaining)
            self._register(candidate)
            try:
                candidate.connect()
            except _CONNECTION_ERRORS as exc:
                self._discard(candidate)
                with contextlib.suppress(OSError):
                    candidate.close()
                if attempt == 0:
                    continue
                raise WorkerClientError("worker_unreachable") from exc
            connection = candidate
            break
        if connection is None:
            raise WorkerClientError("worker_unreachable")
        try:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Connection": "close",
                "User-Agent": "AtlasLens-local-worker-client/1",
            }
            if body is not None:
                headers["Content-Type"] = "application/json"
                headers["Content-Length"] = str(len(body))
            try:
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                content_length = response.getheader("Content-Length")
                if content_length is not None:
                    try:
                        declared = int(content_length)
                    except ValueError as exc:
                        raise WorkerClientError("invalid_worker_response") from exc
                    if not 0 <= declared <= self._max_response_bytes:
                        raise WorkerClientError("worker_response_too_large")
                content_type = response.getheader("Content-Type", "")
                if not content_type.casefold().startswith("application/json"):
                    raise WorkerClientError("invalid_worker_response")
                raw = response.read(self._max_response_bytes + 1)
            except WorkerClientError:
                raise
            except _CONNECTION_ERRORS as exc:
                code = (
                    "worker_timeout"
                    if time.monotonic() >= deadline
                    else "worker_transport_failed"
                )
                raise WorkerClientError(code) from exc
            if len(raw) > self._max_response_bytes:
                raise WorkerClientError("worker_response_too_large")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WorkerClientError("invalid_worker_response") from exc
            if not isinstance(payload, dict):
                raise WorkerClientError("invalid_worker_response")
            if response.status < 200 or response.status >= 300 or payload.get("ok") is False:
                raise WorkerClientError(self._error_code(payload))
            return payload
        finally:
            self._discard(connection)
            with contextlib.suppress(OSError):
                connection.close()

    def _validate_identity(
        self,
        provider: str,
        provider_revision: str,
        model_revision: str,
        *,
        validate_model_revision: bool,
    ) -> None:
        if provider != self._provider:
            raise WorkerClientError("worker_identity_mismatch")
        if self._provider_revision is not None and provider_revision != self._provider_revision:
            raise WorkerClientError("worker_revision_mismatch")
        if (
            validate_model_revision
            and self._model_revisions
            and model_revision not in self._model_revisions
        ):
            raise WorkerClientError("worker_model_revision_mismatch")

    def _parse_health(self, payload: Mapping[str, Any]) -> WorkerHealth:
        try:
            health = WorkerHealth.model_validate(payload)
        except ValueError as exc:
            raise WorkerClientError("invalid_worker_health") from exc
        self._validate_identity(
            health.provider,
            health.provider_revision,
            health.model_revision,
            validate_model_revision=False,
        )
        if not health.process_running:
            raise WorkerClientError("invalid_worker_health")
        return health

    @staticmethod
    def _error_code(payload: Mapping[str, Any]) -> str:
        error = payload.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        return code if isinstance(code, str) and _SAFE_CODE.fullmatch(code) else "worker_http_error"

    def _register(self, connection: http.client.HTTPConnection) -> None:
        with self._active_lock:
            self._active.add(connection)

    def _discard(self, connection: http.client.HTTPConnection) -> None:
        with self._active_lock:
            self._active.discard(connection)

    def _abort_active_connections(self) -> None:
        with self._active_lock:
            connections = tuple(self._active)
            self._active.clear()
        for connection in connections:
            with contextlib.suppress(OSError):
                connection.close()


class OSV5MHTTPWorkerClient:
    """HTTP implementation of the existing OSV5MWorker protocol."""

    def __init__(self, client: LocalWorkerHTTPClient, *, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    async def health(self) -> WorkerHealth:
        return await self._client.health()

    async def load(self, device: Literal["cuda", "cpu"]) -> None:
        await self._client.load({"device": device, "model_id": self._model_id})

    async def predict_radians(
        self,
        image_bytes: bytes,
        *,
        device: Literal["cuda", "cpu"],
    ) -> object:
        response = await self._client.infer(
            image_bytes,
            {"device": device, "model_id": self._model_id},
        )
        return _finite_pair(response.result.get("coordinates_radians"), "invalid_osv5m_response")

    async def unload(self, device: Literal["cuda", "cpu"]) -> None:
        await self._client.unload({"device": device, "model_id": self._model_id})

    async def close(self) -> None:
        await self._client.close()


class PlonkHTTPWorkerClient:
    """HTTP implementation of the existing PlonkWorker protocol."""

    def __init__(self, client: LocalWorkerHTTPClient) -> None:
        self._client = client

    async def health(self) -> WorkerHealth:
        return await self._client.health()

    async def load(self, model_id: str, device: Literal["cuda", "cpu"]) -> None:
        await self._client.load({"device": device, "model_id": model_id})

    async def sample(
        self,
        image_bytes: bytes,
        *,
        model_id: str,
        sample_count: int,
        device: Literal["cuda", "cpu"],
    ) -> PlonkWorkerOutput:
        response = await self._client.infer(
            image_bytes,
            {
                "device": device,
                "model_id": model_id,
                "sample_count": sample_count,
            },
        )
        raw_samples = response.result.get("samples_degrees")
        if not isinstance(raw_samples, list) or not 1 <= len(raw_samples) <= 256:
            raise WorkerClientError("invalid_plonk_response")
        samples = tuple(
            _finite_pair(value, "invalid_plonk_response") for value in raw_samples
        )
        localizability_raw = response.result.get("localizability")
        if localizability_raw is None:
            localizability = None
        elif (
            isinstance(localizability_raw, bool)
            or not isinstance(localizability_raw, int | float)
            or not math.isfinite(float(localizability_raw))
        ):
            raise WorkerClientError("invalid_plonk_response")
        else:
            localizability = float(localizability_raw)
        return PlonkWorkerOutput(samples_degrees=samples, localizability=localizability)

    async def unload(self, model_id: str, device: Literal["cuda", "cpu"]) -> None:
        await self._client.unload({"device": device, "model_id": model_id})

    async def close(self) -> None:
        await self._client.close()


class PaddleOCRHTTPWorkerClient:
    """HTTP implementation of the existing PaddleOCRWorker protocol."""

    def __init__(self, client: LocalWorkerHTTPClient, *, device: Literal["cpu"] = "cpu") -> None:
        self._client = client
        self._device = device

    async def health(self) -> WorkerHealth:
        return await self._client.health()

    async def load(self) -> None:
        await self._client.load({"device": self._device})

    async def infer(
        self,
        image_bytes: bytes,
        *,
        scales: tuple[float, ...],
        timeout_seconds: float,
        cancellation: asyncio.Event,
    ) -> tuple[RawPaddleOCRLine, ...]:
        response = await self._client.infer(
            image_bytes,
            {"device": self._device, "scales": list(scales)},
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )
        raw_lines = response.result.get("lines")
        if not isinstance(raw_lines, list) or len(raw_lines) > 128:
            raise WorkerClientError("invalid_paddleocr_response")
        return tuple(_paddle_line(value) for value in raw_lines)

    async def close(self) -> None:
        await self._client.close()


def _finite_pair(value: object, code: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise WorkerClientError(code)
    left, right = value
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, int | float)
        or not isinstance(right, int | float)
    ):
        raise WorkerClientError(code)
    first, second = float(left), float(right)
    if not math.isfinite(first) or not math.isfinite(second):
        raise WorkerClientError(code)
    return first, second


def _paddle_line(value: object) -> RawPaddleOCRLine:
    if not isinstance(value, dict):
        raise WorkerClientError("invalid_paddleocr_response")
    text = value.get("text")
    confidence = value.get("confidence")
    polygon = value.get("polygon")
    crop_id = value.get("crop_id", "standard")
    if (
        not isinstance(text, str)
        or len(text) > 512
        or isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not math.isfinite(float(confidence))
        or not isinstance(polygon, list | tuple)
        or len(polygon) != 4
        or not isinstance(crop_id, str)
        or _SAFE_PROFILE.fullmatch(crop_id) is None
    ):
        raise WorkerClientError("invalid_paddleocr_response")
    points = tuple(_finite_pair(point, "invalid_paddleocr_response") for point in polygon)
    return RawPaddleOCRLine(
        text=text,
        confidence=float(confidence),
        polygon=(points[0], points[1], points[2], points[3]),
        crop_id=crop_id,
    )
