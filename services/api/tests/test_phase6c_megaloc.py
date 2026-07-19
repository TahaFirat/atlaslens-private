from __future__ import annotations

import asyncio
import hashlib
import io
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

MODEL_WORKERS_ROOT = Path(__file__).resolve().parents[2] / "model-workers"
sys.path.insert(0, str(MODEL_WORKERS_ROOT))

from common.protocol import JSONValue  # noqa: E402
from common.server import WorkerHTTPServer, create_worker_server  # noqa: E402

from atlaslens_api.phase6b.scheduler import HeavyModelScheduler  # noqa: E402
from atlaslens_api.phase6b.worker_client import (  # noqa: E402
    LocalWorkerHTTPClient,
    WorkerClientError,
    WorkerHealth,
)
from atlaslens_api.phase6c.megaloc import (  # noqa: E402
    MEGALOC_DESCRIPTOR_DIMENSION,
    MEGALOC_DESCRIPTOR_VERSION,
    MEGALOC_MODEL_ID,
    MEGALOC_MODEL_REVISION,
    MEGALOC_SOURCE_REVISION,
    MegaLocHTTPWorkerClient,
    MegaLocRetrievalConfig,
    MegaLocRetrievalProvider,
    create_megaloc_http_worker_client,
    validate_megaloc_descriptor,
)
from atlaslens_api.phase6c.reference_index import (  # noqa: E402
    ReferenceRecord,
    ReferenceSearchHit,
)


def unit_descriptor() -> tuple[float, ...]:
    return (1.0,) + (0.0,) * (MEGALOC_DESCRIPTOR_DIMENSION - 1)


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 16), "white").save(output, format="PNG")
    return output.getvalue()


def health(
    *,
    import_ok: bool = True,
    weights_available: bool = True,
    load_verified: bool = True,
    inference_verified: bool = True,
    last_error: str | None = None,
) -> WorkerHealth:
    return WorkerHealth(
        schema_version="atlaslens-worker-v1",
        provider="megaloc",
        provider_revision=MEGALOC_SOURCE_REVISION,
        model_revision=MEGALOC_MODEL_REVISION,
        device="cpu",
        process_running=True,
        import_ok=import_ok,
        weights_available=weights_available,
        model_loaded=load_verified,
        load_verified=load_verified,
        real_inference_verified=inference_verified,
        last_error=last_error,
    )


class StubWorker:
    def __init__(
        self,
        descriptor: tuple[float, ...] = unit_descriptor(),
        *,
        worker_health: WorkerHealth | None = None,
        delay: float = 0.0,
        health_error: WorkerClientError | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.worker_health = worker_health or health()
        self.delay = delay
        self.health_error = health_error
        self.calls: list[tuple[str, str]] = []
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def health(
        self,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> WorkerHealth:
        assert timeout_seconds is not None
        assert cancellation is None
        if self.health_error is not None:
            raise self.health_error
        return self.worker_health

    async def load(
        self,
        device: str,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> None:
        assert timeout_seconds is not None
        self.calls.append(("load", device))

    async def describe(
        self,
        image_bytes: bytes,
        *,
        device: str,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> tuple[float, ...]:
        assert image_bytes == b"private-query"
        assert timeout_seconds is not None
        self.calls.append(("describe", device))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.descriptor
        finally:
            self.active -= 1

    async def unload(
        self,
        device: str,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> None:
        self.calls.append(("unload", device))

    async def close(self) -> None:
        self.closed = True


def reference_hit(
    reference_id: str,
    *,
    rank: int,
    similarity: float,
    latitude: float,
    longitude: float,
    sequence: str,
    source: str = "mapillary",
) -> ReferenceSearchHit:
    digest = hashlib.sha256(reference_id.encode()).hexdigest()
    record = ReferenceRecord(
        reference_id=reference_id,
        source=source,
        source_family=f"{source}_imagery",
        source_image_id=f"image-{reference_id}",
        source_sequence_id=sequence,
        source_url=f"https://example.com/{source}/{reference_id}",
        latitude=latitude,
        longitude=longitude,
        coordinate_uncertainty_m=8.0,
        heading_degrees=None,
        captured_at=datetime(2025, 1, 2, tzinfo=UTC),
        country="TÃ¼rkiye",
        province="Test province",
        city="Test city",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="Public imagery contributor",
        asset_key=f"references/{reference_id}.jpg",
        sha256=digest,
        perceptual_hash=digest[:16],
        descriptor_version=MEGALOC_DESCRIPTOR_VERSION,
    )
    return ReferenceSearchHit(
        reference_id=reference_id,
        rank=rank,
        similarity=similarity,
        distance=1.0 - similarity,
        uncertainty_radius_m=record.coordinate_uncertainty_m,
        reference=record,
    )


class StubIndex:
    def __init__(
        self,
        hits: tuple[ReferenceSearchHit, ...],
        *,
        available: bool = True,
        descriptor_version: str = MEGALOC_DESCRIPTOR_VERSION,
    ) -> None:
        self.hits = hits
        self.available = available
        self.descriptor_version = descriptor_version
        self.index_version = "turkiye-megaloc-test-v1"
        self.search_calls = 0

    @property
    def size(self) -> int:
        return len(self.hits) if self.available else 0

    def search(
        self,
        query_descriptor: object,
        *,
        top_k: int,
        max_per_sequence: int = 2,
    ) -> tuple[ReferenceSearchHit, ...]:
        assert validate_megaloc_descriptor(query_descriptor) == unit_descriptor()
        assert 1 <= top_k <= 1_000
        assert max_per_sequence == 1
        self.search_calls += 1
        return self.hits


def provider(
    worker: StubWorker | None,
    index: StubIndex | None,
    **updates: object,
) -> MegaLocRetrievalProvider:
    values: dict[str, object] = {
        "enabled": True,
        "worker": worker,
        "reference_index": index,
        "scheduler": HeavyModelScheduler(),
        "device": "cpu",
    }
    values.update(updates)
    return MegaLocRetrievalProvider(**values)  # type: ignore[arg-type]


def test_descriptor_requires_exact_finite_l2_vector() -> None:
    assert validate_megaloc_descriptor(unit_descriptor()) == unit_descriptor()
    with pytest.raises(ValueError, match="exactly 8448"):
        validate_megaloc_descriptor((1.0,))
    with pytest.raises(ValueError, match="finite"):
        validate_megaloc_descriptor((float("nan"),) + unit_descriptor()[1:])
    with pytest.raises(ValueError, match="L2 normalized"):
        validate_megaloc_descriptor((0.5,) + unit_descriptor()[1:])


@pytest.mark.asyncio
async def test_provider_returns_real_index_hits_with_sequence_dedup_and_diversity() -> None:
    hits = (
        reference_hit(
            "ref-a1", rank=1, similarity=0.95, latitude=41.0, longitude=29.0, sequence="seq-a"
        ),
        reference_hit(
            "ref-a2", rank=2, similarity=0.94, latitude=41.001, longitude=29.001, sequence="seq-a"
        ),
        reference_hit(
            "ref-b", rank=3, similarity=0.92, latitude=41.01, longitude=29.01, sequence="seq-b"
        ),
        reference_hit(
            "ref-c", rank=4, similarity=0.90, latitude=39.93, longitude=32.85, sequence="seq-c"
        ),
        reference_hit(
            "ref-d", rank=5, similarity=0.88, latitude=38.42, longitude=27.14, sequence="seq-d"
        ),
    )
    worker = StubWorker()
    index = StubIndex(hits)
    result = await provider(worker, index).retrieve(b"private-query")

    assert result.status == "completed"
    assert [item.rank for item in result.matches] == [1, 2, 3, 4]
    assert {item.reference_id for item in result.matches} == {
        "ref-a1",
        "ref-b",
        "ref-c",
        "ref-d",
    }
    assert result.matches[0].confidence is None
    assert result.matches[0].similarity_semantics == "cosine_similarity_not_confidence"
    assert result.matches[0].reference.source_url.startswith("https://")
    assert result.matches[0].reference.license == "CC BY 4.0"
    assert result.matches[0].reference.captured_at is not None
    assert result.matches[0].uncertainty_radius_m > 0
    assert result.source_family == "megaloc_retrieval_family"
    assert max(item.independent_sequence_support for item in result.clusters) == 2
    assert all(item.uncertainty_radius_km > 0 for item in result.clusters)
    assert "warning.megaloc.sequence_duplicates_reduced" in result.warnings
    assert index.search_calls == 1
    assert worker.calls == [
        ("load", "cpu"),
        ("describe", "cpu"),
        ("unload", "cpu"),
    ]
    dumped = result.model_dump()
    assert not _contains_key(dumped, "descriptor")
    assert "private-query" not in str(dumped)


@pytest.mark.asyncio
async def test_status_requires_real_load_inference_and_verified_index() -> None:
    index = StubIndex(())
    missing = await provider(None, index).status()
    load_pending = await provider(
        StubWorker(worker_health=health(load_verified=False, inference_verified=False)),
        index,
    ).status()
    inference_pending = await provider(
        StubWorker(worker_health=health(inference_verified=False)),
        index,
    ).status()
    inference_failed = await provider(
        StubWorker(
            worker_health=health(
                inference_verified=False,
                last_error="inference_failed",
            )
        ),
        index,
    ).status()
    load_failed = await provider(
        StubWorker(
            worker_health=health(
                load_verified=False,
                inference_verified=False,
                last_error="model_load_failed",
            )
        ),
        index,
    ).status()
    unreachable = await provider(
        StubWorker(health_error=WorkerClientError("worker_unreachable")),
        index,
    ).status()
    ready_index = StubIndex(
        (
            reference_hit(
                "ready-ref",
                rank=1,
                similarity=0.8,
                latitude=40,
                longitude=30,
                sequence="ready-seq",
            ),
        )
    )
    ready = await provider(StubWorker(), ready_index).status()

    assert (missing.state, missing.available) == ("not_installed", False)
    assert load_pending.state == "model_load_not_verified"
    assert inference_pending.state == "inference_not_verified"
    assert inference_failed.state == "inference_failed"
    assert load_failed.state == "model_load_failed"
    assert unreachable.state == "worker_unreachable"
    assert unreachable.worker_reachable is False
    assert (ready.state, ready.available) == ("ready", True)
    assert ready.load_verified and ready.real_inference_verified
    assert ready.worker_reachable is True
    assert ready.installed and ready.weights_available and ready.model_loaded


@pytest.mark.asyncio
async def test_missing_or_mismatched_index_abstains_before_descriptor_inference() -> None:
    worker = StubWorker()
    missing = await provider(worker, None).retrieve(b"private-query")
    mismatched = await provider(
        worker,
        StubIndex((), descriptor_version="different-descriptor-v1"),
    ).retrieve(b"private-query")
    assert (missing.status, missing.reason_code) == (
        "abstained",
        "reference_index_unavailable",
    )
    assert (mismatched.status, mismatched.reason_code) == (
        "abstained",
        "descriptor_version_mismatch",
    )
    assert worker.calls == []


@pytest.mark.asyncio
async def test_invalid_descriptor_and_timeout_fail_without_fake_hits() -> None:
    one_hit = StubIndex(
        (
            reference_hit(
                "only-ref",
                rank=1,
                similarity=0.7,
                latitude=40,
                longitude=30,
                sequence="only-seq",
            ),
        )
    )
    invalid = await provider(StubWorker((0.0,) * MEGALOC_DESCRIPTOR_DIMENSION), one_hit).retrieve(
        b"private-query"
    )
    timed_out = await provider(
        StubWorker(delay=0.1),
        one_hit,
        config=MegaLocRetrievalConfig(timeout_seconds=0.01),
    ).retrieve(b"private-query")
    assert (invalid.status, invalid.reason_code, invalid.matches) == (
        "failed",
        "invalid_model_output",
        (),
    )
    assert timed_out.status == "timeout"
    assert timed_out.reason_code in {"model_load_timeout", "inference_timeout"}
    assert timed_out.matches == ()


@pytest.mark.asyncio
async def test_query_semaphore_bounds_cpu_worker_concurrency() -> None:
    worker = StubWorker(delay=0.02)
    index = StubIndex(
        (
            reference_hit(
                "concurrent-ref",
                rank=1,
                similarity=0.75,
                latitude=40,
                longitude=30,
                sequence="concurrent-seq",
            ),
        )
    )
    instance = provider(
        worker,
        index,
        config=MegaLocRetrievalConfig(max_concurrent_queries=1),
    )
    left, right = await asyncio.gather(
        instance.retrieve(b"private-query"),
        instance.retrieve(b"private-query"),
    )
    assert left.status == right.status == "completed"
    assert worker.max_active == 1


@pytest.mark.asyncio
async def test_cancellation_while_waiting_does_not_leak_query_slot() -> None:
    worker = StubWorker(delay=0.04)
    index = StubIndex(
        (
            reference_hit(
                "cancel-ref",
                rank=1,
                similarity=0.75,
                latitude=40,
                longitude=30,
                sequence="cancel-seq",
            ),
        )
    )
    instance = provider(
        worker,
        index,
        config=MegaLocRetrievalConfig(max_concurrent_queries=1),
    )
    first = asyncio.create_task(instance.retrieve(b"private-query"))
    await asyncio.sleep(0.005)
    cancellation = asyncio.Event()
    second = asyncio.create_task(
        instance.retrieve(b"private-query", cancellation=cancellation)
    )
    await asyncio.sleep(0.005)
    cancellation.set()
    cancelled = await second
    completed = await first
    after = await instance.retrieve(b"private-query")
    assert (cancelled.status, cancelled.reason_code) == ("skipped", "cancelled")
    assert completed.status == after.status == "completed"
    assert worker.max_active == 1


class TransportAdapter:
    def __init__(self, descriptor: tuple[float, ...]) -> None:
        self.descriptor = descriptor
        self.loaded = False
        self.verified = False
        self.device = "cpu"

    def health(self) -> Mapping[str, JSONValue]:
        return {
            "import_ok": True,
            "weights_available": True,
            "model_loaded": self.loaded,
            "load_verified": self.loaded,
            "real_inference_verified": self.verified,
            "device": self.device,
            "model_revision": MEGALOC_MODEL_REVISION,
            "last_error": None,
        }

    def load(self, parameters: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        self.device = str(parameters["device"])
        self.loaded = True
        return {"loaded": True}

    def infer(
        self,
        image_bytes: bytes,
        parameters: dict[str, JSONValue],
    ) -> Mapping[str, JSONValue]:
        assert image_bytes.startswith(b"\x89PNG")
        assert parameters["device"] == self.device
        assert self.loaded
        self.verified = True
        return {
            "provider": "megaloc",
            "model_id": MEGALOC_MODEL_ID,
            "model_revision": MEGALOC_MODEL_REVISION,
            "source_revision": MEGALOC_SOURCE_REVISION,
            "device": self.device,
            "descriptor": list(self.descriptor),
            "descriptor_dimension": MEGALOC_DESCRIPTOR_DIMENSION,
            "descriptor_normalization": "l2",
            "descriptor_semantics": "visual_place_descriptor_not_confidence",
            "inference_ms": 4,
        }

    def unload(self, parameters: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        self.loaded = False
        return {"unloaded": True}


@contextmanager
def transport_server(adapter: TransportAdapter) -> Iterator[WorkerHTTPServer]:
    server = create_worker_server(
        adapter,
        provider="megaloc",
        provider_revision=MEGALOC_SOURCE_REVISION,
        model_revision=MEGALOC_MODEL_REVISION,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_http_worker_round_trip_validates_identity_revision_and_descriptor() -> None:
    adapter = TransportAdapter(unit_descriptor())
    with transport_server(adapter) as server:
        worker = create_megaloc_http_worker_client(
            host="127.0.0.1",
            port=server.server_address[1],
            timeout_seconds=2,
        )
        before = await worker.health()
        assert before.ready is False
        await worker.load("cpu")
        descriptor = await worker.describe(png_bytes(), device="cpu")
        after = await worker.health()
        assert descriptor == unit_descriptor()
        assert after.ready is True
        await worker.unload("cpu")
        await worker.close()

    with pytest.raises(ValueError, match="127.0.0.1"):
        create_megaloc_http_worker_client(
            host="0.0.0.0",
            port=8794,
            timeout_seconds=2,
        )


@pytest.mark.asyncio
async def test_http_worker_rejects_malformed_descriptor_and_health_revision() -> None:
    adapter = TransportAdapter((1.0,))
    with transport_server(adapter) as server:
        transport = LocalWorkerHTTPClient(
            provider="megaloc",
            host="127.0.0.1",
            port=server.server_address[1],
            provider_revision=MEGALOC_SOURCE_REVISION,
            model_revisions=(MEGALOC_MODEL_REVISION,),
            timeout_seconds=2,
        )
        worker = MegaLocHTTPWorkerClient(transport)
        await worker.load("cpu")
        with pytest.raises(WorkerClientError, match="invalid_megaloc_descriptor"):
            await worker.describe(png_bytes(), device="cpu")
        await worker.close()

    with transport_server(TransportAdapter(unit_descriptor())) as server:
        wrong_revision_transport = LocalWorkerHTTPClient(
            provider="megaloc",
            host="127.0.0.1",
            port=server.server_address[1],
            provider_revision="wrong-source-revision",
            model_revisions=(MEGALOC_MODEL_REVISION,),
            timeout_seconds=2,
        )
        worker = MegaLocHTTPWorkerClient(wrong_revision_transport)
        with pytest.raises(WorkerClientError, match="worker_revision_mismatch"):
            await worker.health()
        await worker.close()

    with transport_server(TransportAdapter(unit_descriptor())) as server:
        strict_transport = LocalWorkerHTTPClient(
            provider="megaloc",
            host="127.0.0.1",
            port=server.server_address[1],
            provider_revision=MEGALOC_SOURCE_REVISION,
            model_revisions=(MEGALOC_MODEL_REVISION,),
            timeout_seconds=2,
        )
        worker = MegaLocHTTPWorkerClient(
            strict_transport,
            model_revision="wrong-model-revision",
        )
        with pytest.raises(WorkerClientError, match="worker_revision_mismatch"):
            await worker.health()
        await worker.close()


def test_production_provider_has_no_holdout_specific_branch() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "atlaslens_api"
        / "phase6c"
        / "megaloc.py"
    ).read_text(encoding="utf-8")
    forbidden = ("kayseri", "erciyes", "talas", "4cedcf22c000")
    assert not any(term in source.casefold() for term in forbidden)


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, tuple | list):
        return any(_contains_key(item, key) for item in value)
    return False
