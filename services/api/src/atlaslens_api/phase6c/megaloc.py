from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlaslens_api.phase6b.scheduler import DeviceName, HeavyModelScheduler, ScheduledExecution
from atlaslens_api.phase6b.worker_client import (
    LocalWorkerHTTPClient,
    WorkerClientError,
    WorkerHealth,
)
from atlaslens_api.phase6c.reference_index import (
    ReferenceIndexError,
    ReferenceSearchHit,
)
from atlaslens_api.reranking.geo import geodesic_km, spherical_center

MEGALOC_MODEL_ID = "gberton/MegaLoc"
MEGALOC_SOURCE_REVISION = "1af071c68fc3ab6c6018c5c868391763516e50f7"
MEGALOC_MODEL_REVISION = "7cb9f7970d366fdf059963d04d372e503e8e9df9"
MEGALOC_DESCRIPTOR_VERSION = "megaloc-7cb9f797-max560-imagenet-v1"
MEGALOC_DESCRIPTOR_DIMENSION = 8_448


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class MegaLocWorkerDescriptor(_FrozenModel):
    """Strict worker payload; the descriptor is excluded from dumps and reprs."""

    provider: Literal["megaloc"]
    model_id: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    source_revision: str = Field(min_length=1, max_length=160)
    device: Literal["cuda", "cpu"]
    descriptor: tuple[float, ...] = Field(
        min_length=MEGALOC_DESCRIPTOR_DIMENSION,
        max_length=MEGALOC_DESCRIPTOR_DIMENSION,
        repr=False,
        exclude=True,
    )
    descriptor_dimension: Literal[8448]
    descriptor_normalization: Literal["l2"]
    descriptor_semantics: Literal["visual_place_descriptor_not_confidence"]
    inference_ms: int = Field(ge=0)

    @field_validator("descriptor")
    @classmethod
    def normalized_descriptor(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return validate_megaloc_descriptor(value)


class MegaLocSearchIndex(Protocol):
    """Read-only seam implemented by ``MegaLocReferenceIndex``."""

    @property
    def available(self) -> bool: ...

    @property
    def size(self) -> int: ...

    @property
    def index_version(self) -> str: ...

    @property
    def descriptor_version(self) -> str: ...

    def search(
        self,
        query_descriptor: object,
        *,
        top_k: int,
        max_per_sequence: int = 2,
    ) -> tuple[ReferenceSearchHit, ...]: ...


class MegaLocWorker(Protocol):
    async def health(
        self,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> WorkerHealth: ...

    async def load(
        self,
        device: DeviceName,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> None: ...

    async def describe(
        self,
        image_bytes: bytes,
        *,
        device: DeviceName,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> tuple[float, ...]: ...

    async def unload(
        self,
        device: DeviceName,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> None: ...

    async def close(self) -> None: ...


class MegaLocHTTPWorkerClient:
    """Typed MegaLoc adapter over the common loopback-only worker transport."""

    def __init__(
        self,
        client: LocalWorkerHTTPClient,
        *,
        model_id: str = MEGALOC_MODEL_ID,
        source_revision: str = MEGALOC_SOURCE_REVISION,
        model_revision: str = MEGALOC_MODEL_REVISION,
    ) -> None:
        if not model_id or not source_revision or not model_revision:
            raise ValueError("MegaLoc worker identity must be explicit")
        self._client = client
        self._model_id = model_id
        self._source_revision = source_revision
        self._model_revision = model_revision

    async def health(
        self,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> WorkerHealth:
        health = await self._client.health(
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )
        if (
            health.provider != "megaloc"
            or health.provider_revision != self._source_revision
            or health.model_revision != self._model_revision
        ):
            raise WorkerClientError("worker_revision_mismatch")
        return health

    async def load(
        self,
        device: DeviceName,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> None:
        response = await self._client.load(
            {"device": device},
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )
        if response.device != device:
            raise WorkerClientError("worker_device_mismatch")

    async def describe(
        self,
        image_bytes: bytes,
        *,
        device: DeviceName,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> tuple[float, ...]:
        response = await self._client.infer(
            image_bytes,
            {"device": device},
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )
        try:
            output = MegaLocWorkerDescriptor.model_validate(response.result)
        except ValueError as exc:
            raise WorkerClientError("invalid_megaloc_descriptor") from exc
        if (
            output.model_id != self._model_id
            or output.source_revision != self._source_revision
            or output.model_revision != self._model_revision
            or output.device != device
        ):
            raise WorkerClientError("worker_revision_mismatch")
        return output.descriptor

    async def unload(
        self,
        device: DeviceName,
        *,
        timeout_seconds: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> None:
        await self._client.unload(
            {"device": device},
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )

    async def close(self) -> None:
        await self._client.close()


def create_megaloc_http_worker_client(
    *,
    host: str,
    port: int,
    timeout_seconds: float,
    source_revision: str = MEGALOC_SOURCE_REVISION,
    model_revision: str = MEGALOC_MODEL_REVISION,
    max_image_bytes: int = 20 * 1024 * 1024,
) -> MegaLocHTTPWorkerClient:
    """Compose the strict client; ``LocalWorkerHTTPClient`` rejects non-loopback hosts."""

    client = LocalWorkerHTTPClient(
        provider="megaloc",
        host=host,
        port=port,
        provider_revision=source_revision,
        model_revisions=(model_revision,),
        timeout_seconds=timeout_seconds,
        max_image_bytes=max_image_bytes,
        max_request_bytes=min(
            64 * 1024 * 1024,
            max(1024 * 1024, max_image_bytes * 2),
        ),
        max_response_bytes=2 * 1024 * 1024,
    )
    return MegaLocHTTPWorkerClient(
        client,
        source_revision=source_revision,
        model_revision=model_revision,
    )


type MegaLocProviderState = Literal[
    "disabled",
    "not_installed",
    "worker_unreachable",
    "model_load_not_verified",
    "model_load_failed",
    "inference_not_verified",
    "inference_failed",
    "reference_index_unavailable",
    "descriptor_version_mismatch",
    "ready",
]


class MegaLocProviderCapability(_FrozenModel):
    provider: Literal["megaloc_retrieval"] = "megaloc_retrieval"
    state: MegaLocProviderState
    available: bool
    reason_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[a-z0-9][a-z0-9._-]+$",
    )
    model_id: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    source_revision: str = Field(min_length=1, max_length=160)
    query_descriptor_version: str = Field(min_length=1, max_length=160)
    device: Literal["cuda", "cpu"]
    worker_reachable: bool | None = None
    installed: bool | None = None
    weights_available: bool | None = None
    model_loaded: bool | None = None
    load_verified: bool
    real_inference_verified: bool
    index_available: bool
    index_version: str | None = Field(default=None, max_length=160)
    index_size: int = Field(ge=0)
    confidence: None = None
    confidence_semantics: Literal["uncalibrated_unavailable"] = "uncalibrated_unavailable"

    @model_validator(mode="after")
    def coherent_readiness(self) -> MegaLocProviderCapability:
        if self.available != (self.state == "ready"):
            raise ValueError("only an attested ready MegaLoc capability may be available")
        if self.available and (
            self.reason_code is not None
            or not self.load_verified
            or not self.real_inference_verified
            or not self.index_available
            or self.index_size <= 0
        ):
            raise ValueError("MegaLoc readiness requires real inference and a verified index")
        if not self.available and self.reason_code is None:
            raise ValueError("unavailable MegaLoc capability requires a reason")
        return self


class MegaLocRetrievalConfig(_FrozenModel):
    timeout_seconds: float = Field(default=60.0, gt=0, le=180)
    status_timeout_seconds: float = Field(default=2.0, gt=0, le=10)
    max_input_bytes: int = Field(default=20 * 1024 * 1024, gt=0, le=32 * 1024 * 1024)
    max_matches: int = Field(default=50, ge=1, le=100)
    search_multiplier: int = Field(default=4, ge=1, le=10)
    max_per_sequence: Literal[1] = 1
    cluster_radius_km: float = Field(default=25.0, gt=0, le=500)
    max_concurrent_queries: int = Field(default=1, ge=1, le=8)
    estimated_vram_mb: int = Field(default=3_000, ge=0, le=48_000)


class MegaLocRetrievalCluster(_FrozenModel):
    cluster_id: str = Field(pattern=r"^megaloc-cluster-[a-f0-9]{16}$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    uncertainty_radius_km: float = Field(gt=0)
    match_count: int = Field(ge=1, le=100)
    independent_sequence_support: int = Field(ge=1, le=100)
    best_similarity: float = Field(ge=-1, le=1)
    similarity_semantics: Literal["cosine_similarity_not_confidence"] = (
        "cosine_similarity_not_confidence"
    )
    confidence: None = None
    confidence_semantics: Literal["uncalibrated_unavailable"] = "uncalibrated_unavailable"
    member_reference_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    member_ranks: tuple[int, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def coherent_members(self) -> MegaLocRetrievalCluster:
        if (
            len(self.member_reference_ids) != self.match_count
            or len(self.member_ranks) != self.match_count
            or len(set(self.member_reference_ids)) != self.match_count
        ):
            raise ValueError("MegaLoc cluster support is inconsistent")
        return self


type MegaLocRetrievalState = Literal["completed", "abstained", "skipped", "failed", "timeout"]


class MegaLocRetrievalResult(_FrozenModel):
    provider: Literal["megaloc_retrieval"] = "megaloc_retrieval"
    source_family: Literal["megaloc_retrieval_family"] = "megaloc_retrieval_family"
    status: MegaLocRetrievalState
    reason_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[a-z0-9][a-z0-9._-]+$",
    )
    model_id: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    source_revision: str = Field(min_length=1, max_length=160)
    index_version: str | None = Field(default=None, max_length=160)
    query_descriptor_version: str = Field(min_length=1, max_length=160)
    device: Literal["cuda", "cpu"]
    duration_ms: int = Field(ge=0)
    score_semantics: Literal["cosine_similarity_not_confidence"] = (
        "cosine_similarity_not_confidence"
    )
    confidence: None = None
    confidence_semantics: Literal["uncalibrated_unavailable"] = "uncalibrated_unavailable"
    matches: tuple[ReferenceSearchHit, ...] = Field(default=(), max_length=100)
    clusters: tuple[MegaLocRetrievalCluster, ...] = Field(default=(), max_length=100)
    warnings: tuple[str, ...] = Field(default=(), max_length=20)
    cpu_fallback_used: bool = False

    @model_validator(mode="after")
    def coherent_result(self) -> MegaLocRetrievalResult:
        if self.status == "completed":
            if not self.matches or not self.clusters or self.reason_code is not None:
                raise ValueError("completed MegaLoc retrieval requires real matches and clusters")
            if [item.rank for item in self.matches] != list(range(1, len(self.matches) + 1)):
                raise ValueError("MegaLoc retrieval ranks must be contiguous")
        elif self.matches or self.clusters or self.reason_code is None:
            raise ValueError("non-completed MegaLoc retrieval must be an explicit abstention")
        return self


class MegaLocRetrievalProvider:
    """Lazy MegaLoc descriptor retrieval with no descriptor or image persistence."""

    def __init__(
        self,
        *,
        enabled: bool,
        worker: MegaLocWorker | None,
        reference_index: MegaLocSearchIndex | None,
        scheduler: HeavyModelScheduler,
        model_id: str = MEGALOC_MODEL_ID,
        model_revision: str = MEGALOC_MODEL_REVISION,
        source_revision: str = MEGALOC_SOURCE_REVISION,
        descriptor_version: str = MEGALOC_DESCRIPTOR_VERSION,
        device: Literal["auto", "cuda", "cpu"] = "auto",
        cuda_available: bool = False,
        allow_cpu_fallback: bool = False,
        config: MegaLocRetrievalConfig | None = None,
    ) -> None:
        if not all((model_id, model_revision, source_revision, descriptor_version)):
            raise ValueError("MegaLoc revisions and descriptor version must be explicit")
        self._enabled = enabled
        self._worker = worker
        self._index = reference_index
        self._scheduler = scheduler
        self._model_id = model_id
        self._model_revision = model_revision
        self._source_revision = source_revision
        self._descriptor_version = descriptor_version
        self._allow_cpu_fallback = allow_cpu_fallback
        self._config = config or MegaLocRetrievalConfig()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_queries)
        if device == "auto":
            self._device: DeviceName = "cuda" if cuda_available else "cpu"
        elif device == "cuda" and not cuda_available and allow_cpu_fallback:
            self._device = "cpu"
        else:
            self._device = device

    async def status(self) -> MegaLocProviderCapability:
        if not self._enabled:
            return self._capability("disabled", "disabled")
        if self._worker is None:
            return self._capability("not_installed", "isolated_worker_not_installed")
        try:
            health = await self._worker.health(
                timeout_seconds=self._config.status_timeout_seconds,
            )
        except WorkerClientError as exc:
            state: MegaLocProviderState = (
                "worker_unreachable"
                if exc.code in {"worker_unreachable", "worker_timeout", "worker_transport_failed"}
                else "not_installed"
            )
            return self._capability(state, exc.code)
        if not health.import_ok or not health.weights_available:
            return self._capability(
                "not_installed",
                "model_artifact_not_verified",
                health=health,
            )
        if not health.load_verified:
            if health.last_error is not None:
                return self._capability(
                    "model_load_failed",
                    "model_load_failed",
                    health=health,
                )
            return self._capability(
                "model_load_not_verified",
                "model_load_not_verified",
                health=health,
            )
        if not health.real_inference_verified:
            if health.last_error is not None:
                return self._capability(
                    "inference_failed",
                    "real_inference_failed",
                    health=health,
                )
            return self._capability(
                "inference_not_verified",
                "real_inference_not_verified",
                health=health,
            )
        if self._index is None or not self._index.available or self._index.size <= 0:
            return self._capability(
                "reference_index_unavailable",
                "reference_index_unavailable",
                health=health,
            )
        if self._index.descriptor_version != self._descriptor_version:
            return self._capability(
                "descriptor_version_mismatch",
                "descriptor_version_mismatch",
                health=health,
            )
        return self._capability("ready", None, health=health)

    async def retrieve(
        self,
        image_bytes: bytes,
        *,
        cancellation: asyncio.Event | None = None,
    ) -> MegaLocRetrievalResult:
        started = time.monotonic()
        stopped = cancellation or asyncio.Event()
        if not self._enabled:
            return self._empty_result("skipped", "disabled", started)
        if self._worker is None:
            return self._empty_result(
                "skipped", "isolated_worker_not_installed", started
            )
        if self._index is None or not self._index.available:
            return self._empty_result(
                "abstained", "reference_index_unavailable", started
            )
        if self._index.descriptor_version != self._descriptor_version:
            return self._empty_result(
                "abstained", "descriptor_version_mismatch", started
            )
        if (
            type(image_bytes) is not bytes
            or not 0 < len(image_bytes) <= self._config.max_input_bytes
        ):
            return self._empty_result("failed", "unsupported_input", started)
        if stopped.is_set():
            return self._empty_result("skipped", "cancelled", started)

        worker = self._worker
        stage = "model_load"
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                async with _bounded_semaphore(self._semaphore, stopped):

                    async def load(device: DeviceName) -> None:
                        nonlocal stage
                        stage = "model_load"
                        await worker.load(
                            device,
                            timeout_seconds=self._config.timeout_seconds,
                            cancellation=stopped,
                        )

                    async def describe(device: DeviceName) -> tuple[float, ...]:
                        nonlocal stage
                        stage = "inference"
                        return await worker.describe(
                            image_bytes,
                            device=device,
                            timeout_seconds=self._config.timeout_seconds,
                            cancellation=stopped,
                        )

                    async def unload(device: DeviceName) -> None:
                        await worker.unload(
                            device,
                            timeout_seconds=self._config.timeout_seconds,
                            cancellation=None,
                        )

                    execution: ScheduledExecution[tuple[float, ...]] = (
                        await self._scheduler.execute(
                            model_name=self._model_id,
                            requested_device=self._device,
                            estimated_vram_mb=self._config.estimated_vram_mb,
                            load=load,
                            infer=describe,
                            unload=unload,
                            allow_cpu_fallback=self._allow_cpu_fallback,
                        )
                    )
                    descriptor = validate_megaloc_descriptor(execution.value)
                    if stopped.is_set():
                        raise WorkerClientError("cancelled")
                    stage = "index_query"
                    requested = min(
                        1_000,
                        self._config.max_matches * self._config.search_multiplier,
                    )
                    raw_hits = await asyncio.to_thread(
                        self._index.search,
                        descriptor,
                        top_k=requested,
                        max_per_sequence=self._config.max_per_sequence,
                    )
                    # No image bytes or query descriptor are retained beyond this scope.
                    del descriptor
                    if stopped.is_set():
                        raise WorkerClientError("cancelled")
        except TimeoutError:
            reason = f"{stage}_timeout"
            return self._empty_result("timeout", reason, started)
        except WorkerClientError as exc:
            if exc.code == "cancelled":
                return self._empty_result("skipped", "cancelled", started)
            if exc.code == "worker_timeout":
                return self._empty_result("timeout", f"{stage}_timeout", started)
            if exc.code in {"worker_unreachable", "worker_transport_failed"}:
                return self._empty_result("failed", "worker_unreachable", started)
            reason = "model_load_failed" if stage == "model_load" else "inference_failed"
            return self._empty_result("failed", reason, started)
        except (ReferenceIndexError, ValueError, TypeError):
            reason = (
                "reference_index_query_failed"
                if stage == "index_query"
                else "invalid_model_output"
            )
            return self._empty_result("failed", reason, started)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - provider boundary emits only safe reason codes
            reason = "reference_index_query_failed" if stage == "index_query" else f"{stage}_failed"
            return self._empty_result("failed", reason, started)

        try:
            matches, removed = _deduplicate_and_diversify_hits(
                raw_hits,
                descriptor_version=self._descriptor_version,
                maximum=self._config.max_matches,
                cluster_radius_km=self._config.cluster_radius_km,
            )
            clusters = _clusters(matches, radius_km=self._config.cluster_radius_km)
        except (ValueError, TypeError):
            return self._empty_result("failed", "invalid_reference_index_output", started)
        if not matches:
            return self._empty_result("abstained", "no_reference_match", started)
        warnings = ["warning.megaloc.cosine_similarity_not_confidence"]
        if removed:
            warnings.append("warning.megaloc.sequence_duplicates_reduced")
        if execution.diagnostic.cpu_fallback_used:
            warnings.append("warning.megaloc.cuda_oom_cpu_fallback")
        return MegaLocRetrievalResult(
            status="completed",
            model_id=self._model_id,
            model_revision=self._model_revision,
            source_revision=self._source_revision,
            index_version=self._index.index_version,
            query_descriptor_version=self._descriptor_version,
            device=execution.diagnostic.actual_device,
            duration_ms=_elapsed_ms(started),
            matches=matches,
            clusters=clusters,
            warnings=tuple(warnings),
            cpu_fallback_used=execution.diagnostic.cpu_fallback_used,
        )

    async def close(self) -> None:
        if self._worker is not None:
            await self._worker.close()

    def _capability(
        self,
        state: MegaLocProviderState,
        reason_code: str | None,
        *,
        health: WorkerHealth | None = None,
    ) -> MegaLocProviderCapability:
        index = self._index
        index_available = index is not None and index.available and index.size > 0
        index_version = index.index_version if index_available and index is not None else None
        index_size = index.size if index_available and index is not None else 0
        return MegaLocProviderCapability(
            state=state,
            available=state == "ready",
            reason_code=reason_code,
            model_id=self._model_id,
            model_revision=self._model_revision,
            source_revision=self._source_revision,
            query_descriptor_version=self._descriptor_version,
            device=self._device,
            worker_reachable=(
                health.process_running
                if health is not None
                else False
                if state == "worker_unreachable"
                else None
            ),
            installed=health.import_ok if health is not None else None,
            weights_available=(
                health.weights_available if health is not None else None
            ),
            model_loaded=health.model_loaded if health is not None else None,
            load_verified=health.load_verified if health is not None else False,
            real_inference_verified=(
                health.real_inference_verified if health is not None else False
            ),
            index_available=index_available,
            index_version=index_version,
            index_size=index_size,
        )

    def _empty_result(
        self,
        status: Literal["abstained", "skipped", "failed", "timeout"],
        reason_code: str,
        started: float,
    ) -> MegaLocRetrievalResult:
        index = self._index
        index_version = index.index_version if index is not None and index.available else None
        return MegaLocRetrievalResult(
            status=status,
            reason_code=reason_code,
            model_id=self._model_id,
            model_revision=self._model_revision,
            source_revision=self._source_revision,
            index_version=index_version,
            query_descriptor_version=self._descriptor_version,
            device=self._device,
            duration_ms=_elapsed_ms(started),
        )


def validate_megaloc_descriptor(value: object) -> tuple[float, ...]:
    if not isinstance(value, tuple | list) or len(value) != MEGALOC_DESCRIPTOR_DIMENSION:
        raise ValueError("MegaLoc descriptor must contain exactly 8448 values")
    descriptor: list[float] = []
    squared_norm = 0.0
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError("MegaLoc descriptor values must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("MegaLoc descriptor values must be finite")
        descriptor.append(number)
        squared_norm += number * number
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError("MegaLoc descriptor must be L2 normalized")
    return tuple(descriptor)


@asynccontextmanager
async def _bounded_semaphore(
    semaphore: asyncio.Semaphore,
    cancellation: asyncio.Event,
) -> AsyncIterator[None]:
    acquire = asyncio.create_task(semaphore.acquire())
    cancelled = asyncio.create_task(cancellation.wait())
    acquired = False
    try:
        done, _ = await asyncio.wait(
            {acquire, cancelled},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancelled in done:
            if acquire in done:
                acquired = await acquire
            raise WorkerClientError("cancelled")
        acquired = await acquire
        if cancellation.is_set():
            raise WorkerClientError("cancelled")
        yield
    finally:
        if acquired:
            semaphore.release()
        acquire.cancel()
        cancelled.cancel()
        await asyncio.gather(acquire, cancelled, return_exceptions=True)


def _deduplicate_and_diversify_hits(
    raw_hits: Sequence[ReferenceSearchHit],
    *,
    descriptor_version: str,
    maximum: int,
    cluster_radius_km: float,
) -> tuple[tuple[ReferenceSearchHit, ...], int]:
    if len(raw_hits) > 1_000:
        raise ValueError("reference index returned too many hits")
    ordered = sorted(
        (ReferenceSearchHit.model_validate(item) for item in raw_hits),
        key=lambda item: (item.rank, item.distance, item.reference_id),
    )
    seen_references: set[str] = set()
    seen_sequences: set[tuple[str, str]] = set()
    deduplicated: list[ReferenceSearchHit] = []
    for hit in ordered:
        reference = hit.reference
        if reference.descriptor_version != descriptor_version:
            raise ValueError("reference descriptor version mismatch")
        if abs((1.0 - hit.similarity) - hit.distance) > 1e-5:
            raise ValueError("reference similarity and distance are inconsistent")
        sequence = (reference.source, reference.source_sequence_id)
        if hit.reference_id in seen_references or sequence in seen_sequences:
            continue
        seen_references.add(hit.reference_id)
        seen_sequences.add(sequence)
        deduplicated.append(hit)
    groups = _cluster_groups(deduplicated, radius_km=cluster_radius_km)
    groups.sort(key=lambda group: min((item.rank, item.distance) for item in group))
    diversified: list[ReferenceSearchHit] = []
    position = 0
    while len(diversified) < maximum:
        added = False
        for group in groups:
            if position < len(group):
                diversified.append(group[position])
                added = True
                if len(diversified) >= maximum:
                    break
        if not added:
            break
        position += 1
    reranked = tuple(
        hit.model_copy(update={"rank": rank})
        for rank, hit in enumerate(diversified, start=1)
    )
    return reranked, len(raw_hits) - len(deduplicated)


def _cluster_groups(
    hits: Sequence[ReferenceSearchHit],
    *,
    radius_km: float,
) -> list[list[ReferenceSearchHit]]:
    if not hits:
        return []
    parents = list(range(len(hits)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(hits)):
        left_reference = hits[left].reference
        for right in range(left + 1, len(hits)):
            right_reference = hits[right].reference
            if geodesic_km(
                (left_reference.latitude, left_reference.longitude),
                (right_reference.latitude, right_reference.longitude),
            ) <= radius_km:
                union(left, right)
    grouped: dict[int, list[ReferenceSearchHit]] = {}
    for index, hit in enumerate(hits):
        grouped.setdefault(find(index), []).append(hit)
    return [
        sorted(group, key=lambda item: (item.rank, item.distance, item.reference_id))
        for _, group in sorted(grouped.items())
    ]


def _clusters(
    matches: Sequence[ReferenceSearchHit],
    *,
    radius_km: float,
) -> tuple[MegaLocRetrievalCluster, ...]:
    clusters: list[MegaLocRetrievalCluster] = []
    for group in _cluster_groups(matches, radius_km=radius_km):
        points = tuple(
            (item.reference.latitude, item.reference.longitude) for item in group
        )
        center = spherical_center(points)
        uncertainty = max(
            geodesic_km(center, point) + item.uncertainty_radius_m / 1_000.0
            for item, point in zip(group, points, strict=True)
        )
        identity = hashlib.sha256(
            "|".join(sorted(item.reference_id for item in group)).encode()
        ).hexdigest()[:16]
        sequences = {
            (item.reference.source, item.reference.source_sequence_id) for item in group
        }
        clusters.append(
            MegaLocRetrievalCluster(
                cluster_id=f"megaloc-cluster-{identity}",
                latitude=center[0],
                longitude=center[1],
                uncertainty_radius_km=max(0.001, uncertainty),
                match_count=len(group),
                independent_sequence_support=len(sequences),
                best_similarity=max(item.similarity for item in group),
                member_reference_ids=tuple(item.reference_id for item in group),
                member_ranks=tuple(item.rank for item in group),
            )
        )
    return tuple(sorted(clusters, key=lambda item: min(item.member_ranks)))


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1_000))
