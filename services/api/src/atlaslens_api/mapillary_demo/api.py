"""Isolated local API for the private, attributed Mapillary technical demo."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Literal, Protocol

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlaslens_api.corpus_index.artifacts import sha256_path
from atlaslens_api.corpus_index.errors import ArtifactIntegrityError
from atlaslens_api.errors import AppError
from atlaslens_api.image_processing import SafeImageProcessor
from atlaslens_api.mapillary_demo.indexing import PublishedMapillaryDemoIndex
from atlaslens_api.mapillary_demo.models import (
    MAPILLARY_LICENSE_IDENTIFIER,
    MAPILLARY_LICENSE_URL,
)
from atlaslens_api.phase6b.scheduler import DeviceName
from atlaslens_api.phase6b.worker_client import WorkerClientError, WorkerHealth
from atlaslens_api.phase6c.megaloc import MegaLocWorker
from atlaslens_api.storage import LocalTemporaryStorage

MEGALOC_PHASE3B2_ARTIFACT_SHA256 = (
    "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
)
MAPILLARY_DEMO_INDEX_VERSION = "mapillary-private-demo-faiss-flatip-v1"
MAPILLARY_DEMO_COVERAGE_LABEL: Literal["Ankara reference pilot"] = (
    "Ankara reference pilot"
)
MAPILLARY_DEMO_RETRIEVAL_PROVIDER: Literal["megaloc_mapillary_faiss"] = (
    "megaloc_mapillary_faiss"
)
MAPILLARY_DEMO_RETRIEVAL_SCOPE: Literal["ankara_reference_collection"] = (
    "ankara_reference_collection"
)
MAPILLARY_DEMO_RESULT_SEMANTICS: Literal[
    "ankara_reference_collection_visual_similarity_not_general_geolocation"
] = (
    "ankara_reference_collection_visual_similarity_not_general_geolocation"
)
MAPILLARY_DEMO_SIMILARITY_SEMANTICS: Literal[
    "cosine_similarity_not_confidence"
] = "cosine_similarity_not_confidence"
MAPILLARY_DEMO_SUPPORTED_REGION: Literal["Ankara pilot collection only"] = (
    "Ankara pilot collection only"
)
MAPILLARY_DEMO_EVIDENCE_VERSION: Literal[
    "phase3b3-mapillary-ankara-pilot-v1"
] = "phase3b3-mapillary-ankara-pilot-v1"
MAPILLARY_DEMO_BENCHMARK_VERSION: Literal[
    "phase3b3-mapillary-benchmark-v1"
] = "phase3b3-mapillary-benchmark-v1"
type MapillaryDemoAnalysisScope = Literal[
    "generic_upload", "ankara_reference_pilot"
]
type MapillaryDemoCoverageStatus = Literal[
    "pilot_eligible", "insufficient", "provider_unavailable"
]
MAPILLARY_DEMO_LIMITATIONS = (
    "Bounded pilot-city coverage only; no Türkiye-wide coverage claim.",
    "Raw cosine similarity is uncalibrated and is not a probability or confidence.",
    "Reference proximity is not geographic proof; abstention remains valid.",
    "Private technical demonstration only; not cleared for public or production use.",
)
_PROBLEM_CONTENT: dict[str, object] = {
    "application/problem+json": {
        "schema": {"$ref": "#/components/schemas/ProblemDetails"}
    }
}
_DEMO_STATUS_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    403: {
        "description": "The private demo is available only to a loopback client.",
        "content": _PROBLEM_CONTENT,
    }
}
_DEMO_QUERY_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    **_DEMO_STATUS_ERROR_RESPONSES,
    413: {
        "description": "The private query image exceeds the bounded upload limit.",
        "content": _PROBLEM_CONTENT,
    },
    415: {
        "description": "The private query image type is unsupported or inconsistent.",
        "content": _PROBLEM_CONTENT,
    },
    422: {
        "description": "Authorization or bounded image validation failed.",
        "content": _PROBLEM_CONTENT,
    },
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MapillaryDemoStatusResponse(_StrictModel):
    state: Literal["disabled", "not_ready", "active"]
    enabled: bool
    available: bool
    reason_code: str | None = Field(default=None, min_length=1, max_length=120)
    city: str | None = Field(default=None, min_length=1, max_length=160)
    image_count: int = Field(ge=0)
    model_id: Literal["gberton/MegaLoc"] = "gberton/MegaLoc"
    model_version: str | None = Field(default=None, min_length=1, max_length=200)
    model_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    descriptor_dimension: Literal[8448] = 8448
    index_version: str | None = Field(default=None, min_length=1, max_length=200)
    index_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    attribution_url: Literal["https://www.mapillary.com/"] = "https://www.mapillary.com/"
    license_identifier: Literal["CC-BY-SA-4.0"] = "CC-BY-SA-4.0"
    license_url: Literal["https://creativecommons.org/licenses/by-sa/4.0/"] = (
        "https://creativecommons.org/licenses/by-sa/4.0/"
    )
    experimental_status: Literal["private_technical_demo_not_production"] = (
        "private_technical_demo_not_production"
    )
    coverage_status: Literal["limited_pilot"] = "limited_pilot"
    coverage_label: Literal["Ankara reference pilot"] = MAPILLARY_DEMO_COVERAGE_LABEL
    retrieval_provider: Literal["megaloc_mapillary_faiss"] = (
        MAPILLARY_DEMO_RETRIEVAL_PROVIDER
    )
    retrieval_scope: Literal["ankara_reference_collection"] = (
        MAPILLARY_DEMO_RETRIEVAL_SCOPE
    )
    result_semantics: Literal[
        "ankara_reference_collection_visual_similarity_not_general_geolocation"
    ] = MAPILLARY_DEMO_RESULT_SEMANTICS
    similarity_semantics: Literal["cosine_similarity_not_confidence"] = (
        MAPILLARY_DEMO_SIMILARITY_SEMANTICS
    )
    supported_region: Literal["Ankara pilot collection only"] = (
        MAPILLARY_DEMO_SUPPORTED_REGION
    )
    evidence_version: Literal["phase3b3-mapillary-ankara-pilot-v1"] = (
        MAPILLARY_DEMO_EVIDENCE_VERSION
    )
    benchmark_version: Literal["phase3b3-mapillary-benchmark-v1"] = (
        MAPILLARY_DEMO_BENCHMARK_VERSION
    )
    limitations: tuple[str, ...] = MAPILLARY_DEMO_LIMITATIONS

    @model_validator(mode="after")
    def coherent_state(self) -> MapillaryDemoStatusResponse:
        if self.state == "active":
            if (
                not self.enabled
                or not self.available
                or self.reason_code is not None
                or not self.city
                or self.image_count <= 0
                or not self.model_version
                or self.model_artifact_sha256 != MEGALOC_PHASE3B2_ARTIFACT_SHA256
                or self.index_version != MAPILLARY_DEMO_INDEX_VERSION
                or not self.index_checksum
            ):
                raise ValueError("active Mapillary demo status is incomplete")
        elif self.available or self.reason_code is None:
            raise ValueError("inactive Mapillary demo status must include a reason")
        return self


class MapillaryDemoCandidateResponse(_StrictModel):
    rank: int = Field(gt=0, le=20)
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    cosine_distance: float = Field(ge=0.0, le=2.0)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    mapillary_image_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    contributor: str | None = Field(default=None, min_length=1, max_length=240)
    source_url: str = Field(min_length=1, max_length=600)
    license_identifier: Literal["CC-BY-SA-4.0"] = "CC-BY-SA-4.0"
    license_url: Literal["https://creativecommons.org/licenses/by-sa/4.0/"] = (
        "https://creativecommons.org/licenses/by-sa/4.0/"
    )
    capture_date: date | None = None
    confidence: None = None
    confidence_semantics: Literal["uncalibrated_unavailable"] = "uncalibrated_unavailable"
    similarity_semantics: Literal["cosine_similarity_not_confidence"] = (
        MAPILLARY_DEMO_SIMILARITY_SEMANTICS
    )
    uncertainty_radius_m: float = Field(gt=0.0)
    uncertainty_semantics: Literal["presentation_radius_not_accuracy_or_probability"] = (
        "presentation_radius_not_accuracy_or_probability"
    )
    experimental_status: Literal["private_technical_demo_not_production"] = (
        "private_technical_demo_not_production"
    )

    @model_validator(mode="after")
    def coherent_similarity(self) -> MapillaryDemoCandidateResponse:
        if abs(self.cosine_distance - (1.0 - self.cosine_similarity)) > 1e-5:
            raise ValueError("cosine distance and similarity are inconsistent")
        return self


class MapillaryDemoQueryResponse(_StrictModel):
    status: Literal["completed", "abstained", "failed"]
    reason_code: str | None = Field(default=None, min_length=1, max_length=120)
    analysis_scope: MapillaryDemoAnalysisScope
    coverage_status: MapillaryDemoCoverageStatus
    coverage_label: Literal["Ankara reference pilot"] = MAPILLARY_DEMO_COVERAGE_LABEL
    retrieval_provider: Literal["megaloc_mapillary_faiss"] = (
        MAPILLARY_DEMO_RETRIEVAL_PROVIDER
    )
    retrieval_scope: Literal["ankara_reference_collection"] = (
        MAPILLARY_DEMO_RETRIEVAL_SCOPE
    )
    result_semantics: Literal[
        "ankara_reference_collection_visual_similarity_not_general_geolocation"
    ] = MAPILLARY_DEMO_RESULT_SEMANTICS
    abstained: bool
    abstention_reason: str | None = Field(default=None, min_length=1, max_length=120)
    similarity_semantics: Literal["cosine_similarity_not_confidence"] = (
        MAPILLARY_DEMO_SIMILARITY_SEMANTICS
    )
    supported_region: Literal["Ankara pilot collection only"] = (
        MAPILLARY_DEMO_SUPPORTED_REGION
    )
    evidence_version: Literal["phase3b3-mapillary-ankara-pilot-v1"] = (
        MAPILLARY_DEMO_EVIDENCE_VERSION
    )
    benchmark_version: Literal["phase3b3-mapillary-benchmark-v1"] = (
        MAPILLARY_DEMO_BENCHMARK_VERSION
    )
    city: str | None = Field(default=None, min_length=1, max_length=160)
    index_version: str | None = Field(default=None, min_length=1, max_length=200)
    candidates: tuple[MapillaryDemoCandidateResponse, ...] = Field(default=(), max_length=10)
    confidence: None = None
    confidence_semantics: Literal["uncalibrated_unavailable"] = "uncalibrated_unavailable"
    experimental_status: Literal["private_technical_demo_not_production"] = (
        "private_technical_demo_not_production"
    )
    limitations: tuple[str, ...] = MAPILLARY_DEMO_LIMITATIONS

    @model_validator(mode="after")
    def coherent_result(self) -> MapillaryDemoQueryResponse:
        if self.reason_code != self.abstention_reason:
            raise ValueError("abstention reason aliases must agree")
        if self.status == "completed":
            if (
                self.analysis_scope != "ankara_reference_pilot"
                or self.coverage_status != "pilot_eligible"
                or self.abstained
                or self.reason_code is not None
                or not self.candidates
            ):
                raise ValueError("completed demo query requires candidates")
            if [item.rank for item in self.candidates] != list(range(1, len(self.candidates) + 1)):
                raise ValueError("demo candidate ranks must be contiguous")
            if any(
                left.cosine_distance > right.cosine_distance
                for left, right in zip(self.candidates, self.candidates[1:], strict=False)
            ):
                raise ValueError("demo candidate order must preserve ascending distance")
        elif not self.abstained or self.reason_code is None or self.candidates:
            raise ValueError("non-completed demo query must explicitly abstain")
        elif self.analysis_scope == "generic_upload" and self.coverage_status != "insufficient":
            raise ValueError("generic upload must fail closed on limited coverage")
        return self


@dataclass(frozen=True, slots=True)
class MapillaryDemoIndexStatus:
    state: Literal["disabled", "not_ready", "active"]
    enabled: bool
    available: bool
    reason_code: str | None
    city: str | None = None
    image_count: int = 0
    model_version: str | None = None
    model_artifact_sha256: str | None = None
    index_version: str | None = None
    index_checksum: str | None = None


class MapillaryDemoSearchIndex(Protocol):
    def status(self) -> MapillaryDemoIndexStatus: ...

    def search(
        self, descriptor: Sequence[float], *, top_k: int
    ) -> tuple[MapillaryDemoCandidateResponse, ...]: ...


class DisabledMapillaryDemoIndex:
    def __init__(self, *, enabled: bool, reason_code: str = "explicitly_disabled") -> None:
        self._enabled = enabled
        self._reason_code = reason_code

    def status(self) -> MapillaryDemoIndexStatus:
        return MapillaryDemoIndexStatus(
            state="not_ready" if self._enabled else "disabled",
            enabled=self._enabled,
            available=False,
            reason_code=self._reason_code,
        )

    def search(
        self, descriptor: Sequence[float], *, top_k: int
    ) -> tuple[MapillaryDemoCandidateResponse, ...]:
        del descriptor, top_k
        return ()


class PublishedMapillaryDemoSearchIndex:
    """Fail-closed adapter over the checksum-bound Phase 3B3 FAISS bundle."""

    def __init__(
        self,
        *,
        enabled: bool,
        bundle_path: Path,
        expected_publication_sha256: str,
        expected_source_policy_sha256: str,
        expected_selection_lock_sha256: str,
        uncertainty_radius_m: float,
    ) -> None:
        self._enabled = enabled
        self._bundle_path = bundle_path
        self._expected_publication_sha256 = expected_publication_sha256.strip()
        self._expected_source_policy_sha256 = expected_source_policy_sha256.strip()
        self._expected_selection_lock_sha256 = expected_selection_lock_sha256.strip()
        self._uncertainty_radius_m = uncertainty_radius_m

    def status(self) -> MapillaryDemoIndexStatus:
        if not self._enabled:
            return MapillaryDemoIndexStatus(
                state="disabled",
                enabled=False,
                available=False,
                reason_code="explicitly_disabled",
            )
        if not _is_sha256(self._expected_publication_sha256):
            return self._not_ready("publication_checksum_not_configured")
        if not _is_sha256(self._expected_source_policy_sha256):
            return self._not_ready("source_policy_checksum_not_configured")
        if not _is_sha256(self._expected_selection_lock_sha256):
            return self._not_ready("selection_lock_checksum_not_configured")
        try:
            publication = self._open()
        except (ArtifactIntegrityError, OSError, ValueError):
            return self._not_ready("demo_index_incompatible")
        descriptor_spec = publication.metadata.get("descriptor_spec")
        model_version = (
            descriptor_spec.get("version")
            if isinstance(descriptor_spec, dict) and isinstance(descriptor_spec.get("version"), str)
            else None
        )
        if not model_version:
            return self._not_ready("demo_index_incompatible")
        return MapillaryDemoIndexStatus(
            state="active",
            enabled=True,
            available=True,
            reason_code=None,
            city=publication.city,
            image_count=publication.size,
            model_version=model_version,
            model_artifact_sha256=MEGALOC_PHASE3B2_ARTIFACT_SHA256,
            index_version=MAPILLARY_DEMO_INDEX_VERSION,
            index_checksum=self._expected_publication_sha256,
        )

    def search(
        self, descriptor: Sequence[float], *, top_k: int
    ) -> tuple[MapillaryDemoCandidateResponse, ...]:
        publication = self._open()
        hits = publication.search(descriptor, top_k=top_k)
        return tuple(
            candidate_from_distance(
                rank=hit.rank,
                cosine_distance=hit.cosine_distance,
                latitude=hit.latitude,
                longitude=hit.longitude,
                mapillary_image_id=hit.mapillary_image_id,
                contributor=hit.creator_id,
                source_url=hit.source_page_url,
                capture_date=_capture_date(hit.captured_at),
                uncertainty_radius_m=self._uncertainty_radius_m,
            )
            for hit in hits
        )

    def _open(self) -> PublishedMapillaryDemoIndex:
        if self._bundle_path.is_symlink():
            raise ArtifactIntegrityError("Mapillary demo bundle root cannot be a symlink")
        root = self._bundle_path.resolve()
        marker = root / "PUBLISHED.json"
        if (
            marker.is_symlink()
            or not marker.is_file()
            or sha256_path(marker) != self._expected_publication_sha256
        ):
            raise ArtifactIntegrityError("Mapillary demo publication checksum mismatch")
        publication = PublishedMapillaryDemoIndex.open(
            root,
            expected_source_policy_sha256=self._expected_source_policy_sha256,
            expected_selection_lock_sha256=self._expected_selection_lock_sha256,
        )
        for record in publication.attribution.values():
            if (
                record.license_identifier != MAPILLARY_LICENSE_IDENTIFIER
                or record.license_url != MAPILLARY_LICENSE_URL
                or not record.source_page_url.startswith("https://www.mapillary.com/app/")
                or "access_token=" in record.source_page_url.casefold()
            ):
                raise ArtifactIntegrityError("Mapillary attribution is incompatible")
        return publication

    def _not_ready(self, reason_code: str) -> MapillaryDemoIndexStatus:
        return MapillaryDemoIndexStatus(
            state="not_ready",
            enabled=True,
            available=False,
            reason_code=reason_code,
            model_artifact_sha256=MEGALOC_PHASE3B2_ARTIFACT_SHA256,
            index_version=MAPILLARY_DEMO_INDEX_VERSION,
        )


class MapillaryDemoRuntime:
    """Local worker-to-index query seam; image bytes and descriptors are ephemeral."""

    def __init__(
        self,
        *,
        index: MapillaryDemoSearchIndex,
        worker: MegaLocWorker | None,
        device: DeviceName,
        timeout_seconds: float,
        top_k: int = 5,
    ) -> None:
        if not 1 <= top_k <= 10:
            raise ValueError("Mapillary demo top_k must be between 1 and 10")
        if timeout_seconds <= 0.0 or timeout_seconds > 180.0:
            raise ValueError("Mapillary demo timeout is outside the bounded range")
        self._index = index
        self._worker = worker
        self._device = device
        self._timeout_seconds = timeout_seconds
        self._top_k = top_k
        self._semaphore = asyncio.Semaphore(1)

    async def status(self) -> MapillaryDemoStatusResponse:
        index = await asyncio.to_thread(self._index.status)
        if index.state != "active":
            return _public_status(index)
        if index.city is None or index.city.strip().casefold() != "ankara":
            return _public_status(_not_ready(index, "demo_coverage_incompatible"))
        if self._worker is None:
            return _public_status(_not_ready(index, "megaloc_worker_unavailable"))
        try:
            health = await self._worker.health(timeout_seconds=min(5.0, self._timeout_seconds))
        except WorkerClientError:
            return _public_status(_not_ready(index, "megaloc_worker_unavailable"))
        if not _worker_ready(health):
            return _public_status(_not_ready(index, "megaloc_worker_not_ready"))
        return _public_status(index)

    async def query(
        self,
        image_bytes: bytes,
        *,
        analysis_scope: MapillaryDemoAnalysisScope = "generic_upload",
    ) -> MapillaryDemoQueryResponse:
        public_status = await self.status()
        if analysis_scope == "generic_upload":
            return _abstention(
                "reference_coverage_insufficient",
                public_status,
                analysis_scope=analysis_scope,
                coverage_status="insufficient",
            )
        if public_status.state != "active" or self._worker is None:
            return _abstention(
                public_status.reason_code or "demo_not_ready",
                public_status,
                analysis_scope=analysis_scope,
                coverage_status="provider_unavailable",
            )
        if not image_bytes:
            return _abstention(
                "empty_query_image", public_status, analysis_scope=analysis_scope
            )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._semaphore:
                    await self._worker.load(self._device, timeout_seconds=self._timeout_seconds)
                    try:
                        descriptor = await self._worker.describe(
                            image_bytes,
                            device=self._device,
                            timeout_seconds=self._timeout_seconds,
                        )
                        candidates = await asyncio.to_thread(
                            self._index.search, descriptor, top_k=self._top_k
                        )
                        del descriptor
                    finally:
                        await self._worker.unload(
                            self._device, timeout_seconds=self._timeout_seconds
                        )
        except TimeoutError:
            return _abstention(
                "demo_query_timeout", public_status, analysis_scope=analysis_scope
            )
        except (ArtifactIntegrityError, WorkerClientError, ValueError, OSError):
            return _abstention(
                "demo_query_failed",
                public_status,
                analysis_scope=analysis_scope,
                failed=True,
            )
        if not candidates:
            return _abstention(
                "no_reference_match", public_status, analysis_scope=analysis_scope
            )
        if not _candidate_order_is_valid(candidates):
            return _abstention(
                "invalid_candidate_ordering",
                public_status,
                analysis_scope=analysis_scope,
                failed=True,
            )
        return MapillaryDemoQueryResponse(
            status="completed",
            reason_code=None,
            analysis_scope=analysis_scope,
            coverage_status="pilot_eligible",
            abstained=False,
            abstention_reason=None,
            city=public_status.city,
            index_version=public_status.index_version,
            candidates=candidates,
        )

    async def close(self) -> None:
        if self._worker is not None:
            await self._worker.close()


def build_mapillary_demo_router(
    *,
    runtime: MapillaryDemoRuntime,
    storage: LocalTemporaryStorage,
    image_processor: SafeImageProcessor,
    max_upload_bytes: int,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/mapillary-demo", tags=["mapillary-private-demo"])

    @router.get(
        "/status",
        response_model=MapillaryDemoStatusResponse,
        operation_id="getMapillaryDemoStatus",
        responses=_DEMO_STATUS_ERROR_RESPONSES,
    )
    async def status(request: Request) -> MapillaryDemoStatusResponse:
        _require_loopback(request)
        return await runtime.status()

    @router.post(
        "/query",
        response_model=MapillaryDemoQueryResponse,
        operation_id="queryMapillaryDemo",
        responses=_DEMO_QUERY_ERROR_RESPONSES,
    )
    async def query(
        request: Request,
        image: Annotated[UploadFile, File()],
        authorization_acknowledged: Annotated[bool, Form()],
        analysis_scope: Annotated[MapillaryDemoAnalysisScope, Form()] = "generic_upload",
    ) -> MapillaryDemoQueryResponse:
        _require_loopback(request)
        if not authorization_acknowledged:
            raise AppError(
                422,
                "authorization_required",
                "error.authorization_required",
                "Authorization acknowledgement required",
            )
        if analysis_scope == "generic_upload":
            return await runtime.query(b"", analysis_scope=analysis_scope)
        public_status = await runtime.status()
        if public_status.state != "active":
            return _abstention(
                public_status.reason_code or "demo_not_ready",
                public_status,
                analysis_scope=analysis_scope,
                coverage_status="provider_unavailable",
            )
        stored = await storage.save_upload(image, max_upload_bytes)
        normalized_key: str | None = None
        try:
            prepared = await image_processor.prepare(
                stored,
                content_type=image.content_type,
                original_filename=image.filename,
            )
            normalized_key = prepared.normalized.key
            payload = await asyncio.to_thread(
                _read_bounded, prepared.normalized.path, max_upload_bytes
            )
            return await runtime.query(payload, analysis_scope=analysis_scope)
        finally:
            try:
                await storage.delete(normalized_key)
            finally:
                await storage.delete(stored.handle.key)

    return router


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client is not None else ""
    if host == "testclient":
        return
    try:
        is_loopback = ip_address(host.split("%", maxsplit=1)[0]).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise AppError(
            403,
            "private_demo_loopback_required",
            "error.private_demo_loopback_required",
            "Private demo access requires a loopback client",
        )


def _read_bounded(path: Path, maximum: int) -> bytes:
    with path.open("rb") as handle:
        payload = handle.read(maximum + 1)
    if not payload or len(payload) > maximum:
        raise OSError("query image outside bounds")
    return payload


def _capture_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ArtifactIntegrityError("Mapillary capture date is invalid") from exc


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _worker_ready(health: WorkerHealth) -> bool:
    return bool(
        health.provider == "megaloc"
        and health.import_ok
        and health.weights_available
        and health.load_verified
        and health.real_inference_verified
    )


def _not_ready(value: MapillaryDemoIndexStatus, reason_code: str) -> MapillaryDemoIndexStatus:
    return MapillaryDemoIndexStatus(
        state="not_ready",
        enabled=value.enabled,
        available=False,
        reason_code=reason_code,
        city=value.city,
        image_count=value.image_count,
        model_version=value.model_version,
        model_artifact_sha256=value.model_artifact_sha256,
        index_version=value.index_version,
        index_checksum=value.index_checksum,
    )


def _public_status(value: MapillaryDemoIndexStatus) -> MapillaryDemoStatusResponse:
    return MapillaryDemoStatusResponse(
        state=value.state,
        enabled=value.enabled,
        available=value.available,
        reason_code=value.reason_code,
        city=value.city,
        image_count=value.image_count,
        model_version=value.model_version,
        model_artifact_sha256=value.model_artifact_sha256,
        index_version=value.index_version,
        index_checksum=value.index_checksum,
    )


def _abstention(
    reason_code: str,
    status: MapillaryDemoStatusResponse,
    *,
    analysis_scope: MapillaryDemoAnalysisScope,
    coverage_status: MapillaryDemoCoverageStatus = "pilot_eligible",
    failed: bool = False,
) -> MapillaryDemoQueryResponse:
    return MapillaryDemoQueryResponse(
        status="failed" if failed else "abstained",
        reason_code=reason_code,
        analysis_scope=analysis_scope,
        coverage_status=coverage_status,
        abstained=True,
        abstention_reason=reason_code,
        city=status.city,
        index_version=status.index_version,
    )


def _candidate_order_is_valid(
    candidates: Sequence[MapillaryDemoCandidateResponse],
) -> bool:
    return bool(candidates) and all(
        candidate.rank == position
        and (
            position == 1
            or candidates[position - 2].cosine_distance <= candidate.cosine_distance
        )
        for position, candidate in enumerate(candidates, start=1)
    )


def candidate_from_distance(
    *,
    rank: int,
    cosine_distance: float,
    latitude: float,
    longitude: float,
    mapillary_image_id: str,
    contributor: str | None,
    source_url: str,
    capture_date: date | None,
    uncertainty_radius_m: float,
) -> MapillaryDemoCandidateResponse:
    """Build one uncalibrated result from a verified index hit and attribution."""

    if not math.isfinite(cosine_distance):
        raise ValueError("cosine distance must be finite")
    bounded_distance = min(2.0, max(0.0, float(cosine_distance)))
    return MapillaryDemoCandidateResponse(
        rank=rank,
        cosine_similarity=1.0 - bounded_distance,
        cosine_distance=bounded_distance,
        latitude=latitude,
        longitude=longitude,
        mapillary_image_id=mapillary_image_id,
        contributor=contributor,
        source_url=source_url,
        capture_date=capture_date,
        uncertainty_radius_m=uncertainty_radius_m,
    )
