from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.parse import urlsplit

from common.health import build_health_payload
from common.image_io import decode_bounded_image
from common.protocol import (
    JSONValue,
    PROTOCOL_VERSION,
    WorkerAdapter,
    WorkerProtocolError,
    encode_json,
    parse_request,
    safe_adapter_error,
)


class WorkerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        adapter: WorkerAdapter,
        *,
        provider: str,
        provider_revision: str,
        model_revision: str,
        max_request_bytes: int,
        max_image_bytes: int,
        max_response_bytes: int,
        max_decoded_pixels: int,
        max_image_dimension: int,
        request_timeout_seconds: float,
    ) -> None:
        self.adapter = adapter
        self.provider = provider
        self.provider_revision = provider_revision
        self.model_revision = model_revision
        self.max_request_bytes = max_request_bytes
        self.max_image_bytes = max_image_bytes
        self.max_response_bytes = max_response_bytes
        self.max_decoded_pixels = max_decoded_pixels
        self.max_image_dimension = max_image_dimension
        self.request_timeout_seconds = request_timeout_seconds
        self.adapter_lock = threading.Lock()
        super().__init__(server_address, WorkerRequestHandler)


class WorkerRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: WorkerHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlsplit(self.path)
        if parsed.path != "/health" or parsed.query or parsed.fragment:
            self._send_error("route_not_found", status=404)
            return
        try:
            raw = self.server.adapter.health()
            payload = build_health_payload(
                raw,
                provider=self.server.provider,
                provider_revision=self.server.provider_revision,
                model_revision=self.server.model_revision,
            )
            self._send(payload, status=200)
        except Exception:  # noqa: BLE001 - public boundary returns only a safe code
            self._send_error("health_unavailable", status=503)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlsplit(self.path)
        operations = {
            "/v1/load": "load",
            "/v1/infer": "infer",
            "/v1/unload": "unload",
        }
        operation = operations.get(parsed.path)
        if operation is None or parsed.query or parsed.fragment:
            self._send_error("route_not_found", status=404)
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._send_error("chunked_requests_not_supported", status=400)
            return
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            self._send_error("json_content_type_required", status=415)
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_error("content_length_required", status=411)
            return
        if not 0 < length <= self.server.max_request_bytes:
            self._send_error("request_too_large", status=413)
            return
        self.connection.settimeout(self.server.request_timeout_seconds)
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise WorkerProtocolError("incomplete_request")
            request = parse_request(body, require_image=operation == "infer")
            image_bytes = (
                decode_bounded_image(
                    cast(str, request.image_base64),
                    max_image_bytes=self.server.max_image_bytes,
                    max_decoded_pixels=self.server.max_decoded_pixels,
                    max_dimension=self.server.max_image_dimension,
                )
                if operation == "infer"
                else None
            )
        except WorkerProtocolError as exc:
            self._send_error(exc.code, status=exc.status)
            return
        except (OSError, TimeoutError):
            self._send_error("request_read_failed", status=400)
            return

        started = time.monotonic()
        try:
            with self.server.adapter_lock:
                if operation == "load":
                    result = self.server.adapter.load(request.parameters)
                elif operation == "infer":
                    result = self.server.adapter.infer(cast(bytes, image_bytes), request.parameters)
                else:
                    result = self.server.adapter.unload(request.parameters)
            if not isinstance(result, Mapping):
                raise WorkerProtocolError("invalid_adapter_output", status=500)
            health = build_health_payload(
                self.server.adapter.health(),
                provider=self.server.provider,
                provider_revision=self.server.provider_revision,
                model_revision=self.server.model_revision,
            )
            response: dict[str, JSONValue] = {
                "schema_version": PROTOCOL_VERSION,
                "ok": True,
                "request_id": request.request_id,
                "provider": self.server.provider,
                "provider_revision": self.server.provider_revision,
                "model_revision": cast(str, health["model_revision"]),
                "device": cast(str, health["device"]),
                "duration_ms": max(0, int((time.monotonic() - started) * 1000)),
                "result": dict(result),
            }
            self._send(response, status=200)
        except WorkerProtocolError as exc:
            self._send_error(exc.code, status=exc.status)
        except Exception as exc:  # noqa: BLE001 - never expose stack or native details
            code, status = safe_adapter_error(exc)
            self._send_error(code, status=status)

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send_error(self, code: str, *, status: int) -> None:
        payload: dict[str, JSONValue] = {
            "schema_version": PROTOCOL_VERSION,
            "ok": False,
            "error": {"code": code},
        }
        self._send(payload, status=status)

    def _send(self, payload: Mapping[str, JSONValue], *, status: int) -> None:
        try:
            body = encode_json(payload, max_bytes=self.server.max_response_bytes)
        except WorkerProtocolError:
            body = (
                b'{"schema_version":"atlaslens-worker-v1","ok":false,'
                b'"error":{"code":"response_encoding_failed"}}'
            )
            status = 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        with contextlib.suppress(OSError):
            self.wfile.write(body)
        self.close_connection = True


def create_worker_server(
    adapter: WorkerAdapter,
    *,
    provider: str,
    provider_revision: str,
    model_revision: str,
    host: str = "127.0.0.1",
    port: int,
    max_request_bytes: int = 30 * 1024 * 1024,
    max_image_bytes: int = 20 * 1024 * 1024,
    max_response_bytes: int = 2 * 1024 * 1024,
    max_decoded_pixels: int = 40_000_000,
    max_image_dimension: int = 16_384,
    request_timeout_seconds: float = 180.0,
) -> WorkerHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("model workers may bind only to 127.0.0.1")
    if not 0 <= port <= 65_535:
        raise ValueError("worker port is invalid")
    if not 1 <= max_image_bytes <= max_request_bytes <= 64 * 1024 * 1024:
        raise ValueError("worker request bounds are invalid")
    if not 1 <= max_response_bytes <= 8 * 1024 * 1024:
        raise ValueError("worker response bound is invalid")
    if not 1 <= max_decoded_pixels <= 100_000_000 or not 1 <= max_image_dimension <= 32_768:
        raise ValueError("worker image bounds are invalid")
    if not 0 < request_timeout_seconds <= 600:
        raise ValueError("worker request timeout is invalid")
    return WorkerHTTPServer(
        (host, port),
        adapter,
        provider=provider,
        provider_revision=provider_revision,
        model_revision=model_revision,
        max_request_bytes=max_request_bytes,
        max_image_bytes=max_image_bytes,
        max_response_bytes=max_response_bytes,
        max_decoded_pixels=max_decoded_pixels,
        max_image_dimension=max_image_dimension,
        request_timeout_seconds=request_timeout_seconds,
    )


def run_worker(
    adapter: WorkerAdapter,
    *,
    provider: str,
    provider_revision: str,
    model_revision: str,
    host: str = "127.0.0.1",
    port: int,
    max_request_bytes: int = 30 * 1024 * 1024,
    max_image_bytes: int = 20 * 1024 * 1024,
    max_response_bytes: int = 2 * 1024 * 1024,
    max_decoded_pixels: int = 40_000_000,
    max_image_dimension: int = 16_384,
    request_timeout_seconds: float = 180.0,
) -> None:
    server = create_worker_server(
        adapter,
        provider=provider,
        provider_revision=provider_revision,
        model_revision=model_revision,
        host=host,
        port=port,
        max_request_bytes=max_request_bytes,
        max_image_bytes=max_image_bytes,
        max_response_bytes=max_response_bytes,
        max_decoded_pixels=max_decoded_pixels,
        max_image_dimension=max_image_dimension,
        request_timeout_seconds=request_timeout_seconds,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            adapter.unload({})
        except Exception:  # noqa: BLE001 - best-effort private cleanup
            pass
        server.server_close()
