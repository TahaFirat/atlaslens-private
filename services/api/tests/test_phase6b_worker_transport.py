from __future__ import annotations

import asyncio
import http.client
import io
import json
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

MODEL_WORKERS_ROOT = Path(__file__).resolve().parents[2] / "model-workers"
sys.path.insert(0, str(MODEL_WORKERS_ROOT))

from common.protocol import JSONValue  # noqa: E402
from common.server import WorkerHTTPServer, create_worker_server  # noqa: E402

from atlaslens_api.phase6b.worker_client import (  # noqa: E402
    LocalWorkerHTTPClient,
    OSV5MHTTPWorkerClient,
    PaddleOCRHTTPWorkerClient,
    PlonkHTTPWorkerClient,
    WorkerClientError,
)


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 16), "white").save(output, format="PNG")
    return output.getvalue()


class StubAdapter:
    def __init__(self, result: Mapping[str, JSONValue], *, delay: float = 0.0) -> None:
        self.result = dict(result)
        self.delay = delay
        self.loaded = False
        self.verified = False
        self.device = "cpu"
        self.load_calls = 0
        self.infer_calls = 0
        self.unload_calls = 0

    def health(self) -> Mapping[str, JSONValue]:
        return {
            "import_ok": True,
            "weights_available": True,
            "model_loaded": self.loaded,
            "real_inference_verified": self.verified,
            "device": self.device,
            "model_revision": "model-revision",
            "last_error": None,
        }

    def load(self, parameters: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        self.load_calls += 1
        self.device = str(parameters.get("device", "cpu"))
        self.loaded = True
        return {"loaded": True}

    def infer(
        self,
        image_bytes: bytes,
        parameters: dict[str, JSONValue],
    ) -> Mapping[str, JSONValue]:
        assert image_bytes.startswith(b"\x89PNG")
        assert self.loaded
        self.infer_calls += 1
        if self.delay:
            time.sleep(self.delay)
        self.verified = True
        return self.result

    def unload(self, parameters: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        self.unload_calls += 1
        self.loaded = False
        return {"unloaded": True}


@contextmanager
def worker_server(
    adapter: StubAdapter,
    *,
    provider: str,
    max_request_bytes: int = 30 * 1024 * 1024,
    max_response_bytes: int = 2 * 1024 * 1024,
) -> Iterator[WorkerHTTPServer]:
    server = create_worker_server(
        adapter,
        provider=provider,
        provider_revision="provider-revision",
        model_revision="model-revision",
        port=0,
        max_request_bytes=max_request_bytes,
        max_image_bytes=min(20 * 1024 * 1024, max_request_bytes),
        max_response_bytes=max_response_bytes,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def transport(server: WorkerHTTPServer, provider: str, **updates: Any) -> LocalWorkerHTTPClient:
    values: dict[str, Any] = {
        "provider": provider,
        "host": "127.0.0.1",
        "port": server.server_address[1],
        "provider_revision": "provider-revision",
        "model_revisions": ("model-revision",),
        "timeout_seconds": 2.0,
    }
    values.update(updates)
    return LocalWorkerHTTPClient(**values)


@pytest.mark.asyncio
async def test_osv_http_worker_round_trip_and_strict_health() -> None:
    adapter = StubAdapter({"coordinates_radians": [0.5, -0.25]})
    with worker_server(adapter, provider="osv5m-baseline") as server:
        worker = OSV5MHTTPWorkerClient(
            transport(server, "osv5m-baseline"),
            model_id="osv5m/baseline",
        )
        before = await worker.health()
        assert before.ready is False
        await worker.load("cpu")
        coordinate = await worker.predict_radians(png_bytes(), device="cpu")
        after = await worker.health()
        assert coordinate == (0.5, -0.25)
        assert after.ready is True
        assert after.provider_revision == "provider-revision"
        assert after.model_revision == "model-revision"
        assert after.device == "cpu"
        await worker.unload("cpu")
        await worker.close()
    assert adapter.load_calls == 1
    assert adapter.infer_calls == 1
    assert adapter.unload_calls == 2


@pytest.mark.asyncio
async def test_plonk_and_paddle_clients_validate_typed_results() -> None:
    plonk_adapter = StubAdapter(
        {
            "samples_degrees": [[41.0, 29.0], [40.9, 29.1]],
            "localizability": 0.4,
        }
    )
    with worker_server(plonk_adapter, provider="plonk") as server:
        worker = PlonkHTTPWorkerClient(transport(server, "plonk"))
        await worker.load("nicolas-dufour/PLONK_YFCC", "cpu")
        result = await worker.sample(
            png_bytes(),
            model_id="nicolas-dufour/PLONK_YFCC",
            sample_count=2,
            device="cpu",
        )
        assert result.samples_degrees == ((41.0, 29.0), (40.9, 29.1))
        assert result.localizability == 0.4
        await worker.close()

    paddle_adapter = StubAdapter(
        {
            "lines": [
                {
                    "text": "KAYSERİ",
                    "confidence": 0.91,
                    "polygon": [[1, 1], [20, 1], [20, 10], [1, 10]],
                    "crop_id": "standard",
                }
            ]
        }
    )
    with worker_server(paddle_adapter, provider="paddleocr") as server:
        worker = PaddleOCRHTTPWorkerClient(transport(server, "paddleocr"))
        await worker.load()
        lines = await worker.infer(
            png_bytes(),
            scales=(1.0, 1.5),
            timeout_seconds=1.0,
            cancellation=asyncio.Event(),
        )
        assert lines[0].text == "KAYSERİ"
        assert lines[0].confidence == pytest.approx(0.91)
        await worker.close()


@pytest.mark.asyncio
async def test_invalid_image_and_oversized_response_are_safe() -> None:
    adapter = StubAdapter({"coordinates_radians": [0.0, 0.0]})
    with worker_server(adapter, provider="osv5m-baseline") as server:
        client = transport(server, "osv5m-baseline")
        await client.load({"device": "cpu"})
        with pytest.raises(WorkerClientError, match="unsupported_or_invalid_image"):
            await client.infer(b"not-an-image", {"device": "cpu"})
        assert adapter.infer_calls == 0
        await client.close()

    large = StubAdapter({"payload": "x" * 5_000})
    with worker_server(
        large,
        provider="osv5m-baseline",
        max_response_bytes=1_024,
    ) as server:
        client = transport(server, "osv5m-baseline")
        await client.load({"device": "cpu"})
        with pytest.raises(WorkerClientError, match="response_encoding_failed"):
            await client.infer(png_bytes(), {"device": "cpu"})
        await client.close()


@pytest.mark.asyncio
async def test_connection_is_retried_once_before_any_request_is_sent() -> None:
    adapter = StubAdapter({"coordinates_radians": [0.0, 0.0]})
    with worker_server(adapter, provider="osv5m-baseline") as server:
        attempts = 0

        def factory(host: str, port: int, timeout: float) -> http.client.HTTPConnection:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return RefusingConnection(host, port, timeout=timeout)
            return http.client.HTTPConnection(host, port, timeout=timeout)

        client = transport(
            server,
            "osv5m-baseline",
            connection_factory=factory,
        )
        health = await client.health()
        assert health.process_running is True
        assert attempts == 2
        await client.close()


class RefusingConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        raise ConnectionRefusedError


@pytest.mark.asyncio
async def test_cancellation_aborts_a_live_inference_request() -> None:
    adapter = StubAdapter({"coordinates_radians": [0.0, 0.0]}, delay=0.3)
    with worker_server(adapter, provider="osv5m-baseline") as server:
        client = transport(server, "osv5m-baseline")
        await client.load({"device": "cpu"})
        cancellation = asyncio.Event()
        task = asyncio.create_task(
            client.infer(
                png_bytes(),
                {"device": "cpu"},
                cancellation=cancellation,
            )
        )
        await asyncio.sleep(0.03)
        cancellation.set()
        with pytest.raises(WorkerClientError, match="cancelled"):
            await task
        await client.close()


def test_server_refuses_public_bind_and_oversized_http_body() -> None:
    adapter = StubAdapter({"coordinates_radians": [0.0, 0.0]})
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_worker_server(
            adapter,
            provider="osv5m-baseline",
            provider_revision="provider-revision",
            model_revision="model-revision",
            host="0.0.0.0",
            port=0,
        )
    with worker_server(
        adapter,
        provider="osv5m-baseline",
        max_request_bytes=1_024,
    ) as server:
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=1
        )
        connection.request(
            "POST",
            "/v1/infer",
            body=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": "2048",
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        assert response.status == 413
        assert payload["error"]["code"] == "request_too_large"
