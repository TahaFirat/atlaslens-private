from __future__ import annotations

import ipaddress
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from atlaslens_api.evaluation.isolation import (
    IsolatedPredictionBoundary,
    IsolatedPredictionRequest,
    IsolatedPredictionResponse,
)
from atlaslens_api.evaluation.models import (
    CandidatePrediction,
    EvaluationProvider,
    ProviderPrediction,
)

_PROVIDER_ID = "atlaslens-phase6c-http"
_MODEL_REVISION = "phase6c-v1"
_PROTOCOL_INPUT_LIMIT = 8 * 1024


class Phase6CHTTPWorkerError(RuntimeError):
    """Internal error carrying only a stable, non-sensitive public code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _HTTPModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, allow_inf_nan=False)


class _AcceptedAnalysis(_HTTPModel):
    id: UUID
    status: Literal["queued"]


class _Point(_HTTPModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class _RelativeAssessment(_HTTPModel):
    relative_rank_score: float = Field(ge=0, le=1)
    reranker_version: Literal[
        "phase5b-v1", "phase6a-v1", "phase6b-v1", "phase6c-v1"
    ]


class _ReverseGeocode(_HTTPModel):
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    region: str | None = Field(default=None, max_length=160)
    city: str | None = Field(default=None, max_length=160)


class _PublicCandidate(_HTTPModel):
    rank: int = Field(ge=1, le=100)
    center: _Point
    radius_km: float = Field(gt=0)
    phase5b_assessment: _RelativeAssessment | None = None
    reverse_geocode: _ReverseGeocode | None = None


class _Abstention(_HTTPModel):
    abstained: Literal[True]


class _LeakageAudit(_HTTPModel):
    status: Literal["passed", "failed", "not_run", "incomplete"]


class _Phase6CSummary(_HTTPModel):
    pipeline_version: Literal["phase6c-v1"]
    fusion_version: Literal["phase6c-v1"]
    reference_index_version: str | None = Field(default=None, max_length=160)
    leakage_audit: _LeakageAudit
    cache_fingerprint: str = Field(pattern=r"^phase6c-v1:[a-f0-9]{64}$")


class _HTTPAnalysis(_HTTPModel):
    id: UUID
    status: Literal["queued", "processing", "completed", "failed", "deleted"]
    pipeline_version: str | None = Field(default=None, max_length=64)
    result_classification: Literal["real", "simulated"] | None = None
    candidates: tuple[_PublicCandidate, ...] = Field(default=(), max_length=100)
    abstention: _Abstention | None = None
    phase6c: _Phase6CSummary | None = None

    @model_validator(mode="after")
    def candidate_ranks_are_contiguous(self) -> _HTTPAnalysis:
        ranks = [candidate.rank for candidate in self.candidates]
        if ranks and ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("public candidate ranks are invalid")
        return self


@dataclass(frozen=True, slots=True)
class Phase6CHTTPWorkerConfig:
    api_base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 110.0
    request_timeout_seconds: float = 15.0
    poll_interval_seconds: float = 0.25
    max_candidates: int = 20
    max_input_bytes: int = 25 * 1024 * 1024
    max_response_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        _validate_loopback_base_url(self.api_base_url)
        if not 0 < self.timeout_seconds <= 3_600:
            raise ValueError("timeout_seconds must be in (0, 3600]")
        if not 0 < self.request_timeout_seconds <= 120:
            raise ValueError("request_timeout_seconds must be in (0, 120]")
        if not 0.01 <= self.poll_interval_seconds <= 10:
            raise ValueError("poll_interval_seconds must be in [0.01, 10]")
        if not 1 <= self.max_candidates <= 50:
            raise ValueError("max_candidates must be in [1, 50]")
        if not 1_024 <= self.max_input_bytes <= 100 * 1024 * 1024:
            raise ValueError("max_input_bytes is invalid")
        if not 1_024 <= self.max_response_bytes <= 16 * 1024 * 1024:
            raise ValueError("max_response_bytes is invalid")


def _validate_loopback_base_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.hostname is None
    ):
        raise ValueError("API base URL must be a plain loopback HTTP origin")
    hostname = parsed.hostname.casefold()
    if hostname != "localhost":
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise ValueError("API base URL must use a loopback host") from exc
        if not address.is_loopback:
            raise ValueError("API base URL must use a loopback host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("API base URL port is invalid") from exc
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("API base URL port is invalid")


def read_isolated_request(stream: BinaryIO) -> IsolatedPredictionRequest:
    payload = stream.read(_PROTOCOL_INPUT_LIMIT + 1)
    if not payload or len(payload) > _PROTOCOL_INPUT_LIMIT:
        raise Phase6CHTTPWorkerError("request_invalid")
    try:
        return IsolatedPredictionRequest.model_validate_json(payload)
    except (UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise Phase6CHTTPWorkerError("request_invalid") from exc


@dataclass(frozen=True, slots=True)
class Phase6CHTTPPredictionWorker:
    config: Phase6CHTTPWorkerConfig
    transport: httpx.BaseTransport | None = field(default=None, repr=False)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    sleeper: Callable[[float], None] = field(default=time.sleep, repr=False)

    def predict(self, request: IsolatedPredictionRequest) -> IsolatedPredictionResponse:
        started = self.clock()
        try:
            prediction = self._predict(request, started)
        except Phase6CHTTPWorkerError as exc:
            prediction = self._failure(exc.code, started)
        except httpx.TimeoutException:
            prediction = self._failure("atlaslens_http_timeout", started)
        except httpx.RequestError:
            prediction = self._failure("atlaslens_http_unavailable", started)
        except (OSError, ValidationError, ValueError):
            prediction = self._failure("atlaslens_http_response_invalid", started)
        except Exception:  # A process boundary must not emit exception payloads or paths.
            prediction = self._failure("worker_internal_failure", started)
        return IsolatedPredictionResponse(
            request_id=request.request_id,
            provider_id=_PROVIDER_ID,
            model_revision=_MODEL_REVISION,
            prediction=prediction,
        )

    def _predict(
        self, request: IsolatedPredictionRequest, started: float
    ) -> ProviderPrediction:
        image_path = Path(request.image_path)
        try:
            size = image_path.stat().st_size
        except OSError as exc:
            raise Phase6CHTTPWorkerError("input_image_unavailable") from exc
        if not image_path.is_file() or not 0 < size <= self.config.max_input_bytes:
            raise Phase6CHTTPWorkerError("input_image_invalid")

        deadline = started + self.config.timeout_seconds
        with httpx.Client(
            base_url=self.config.api_base_url,
            follow_redirects=False,
            trust_env=False,
            transport=self.transport,
        ) as client:
            with image_path.open("rb") as image:
                filename, content_type = _sanitized_upload_identity(image)
                image.seek(0)
                status, body = self._bounded_response(
                    client,
                    "POST",
                    "/api/v1/analyses",
                    deadline,
                    files={"image": (filename, image, content_type)},
                    data={
                        "analysis_mode": "local_only",
                        "cloud_processing_consent": "false",
                        "authorization_acknowledged": "true",
                        "allow_cloud_assist": "false",
                    },
                )
            if status != 202:
                raise Phase6CHTTPWorkerError("analysis_submission_failed")
            try:
                accepted = _AcceptedAnalysis.model_validate_json(body)
            except ValidationError as exc:
                raise Phase6CHTTPWorkerError("analysis_submission_invalid") from exc
            terminal = self._poll(client, accepted.id, deadline)

        return self._map_terminal(terminal, started)

    def _poll(
        self, client: httpx.Client, analysis_id: UUID, deadline: float
    ) -> _HTTPAnalysis:
        path = f"/api/v1/analyses/{analysis_id}"
        while True:
            if self.clock() >= deadline:
                raise Phase6CHTTPWorkerError("atlaslens_http_timeout")
            status, body = self._bounded_response(client, "GET", path, deadline)
            if status != 200:
                raise Phase6CHTTPWorkerError("analysis_poll_failed")
            try:
                analysis = _HTTPAnalysis.model_validate_json(body)
            except ValidationError as exc:
                raise Phase6CHTTPWorkerError("analysis_response_invalid") from exc
            if analysis.id != analysis_id:
                raise Phase6CHTTPWorkerError("analysis_response_id_mismatch")
            if analysis.status in {"completed", "failed", "deleted"}:
                return analysis
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise Phase6CHTTPWorkerError("atlaslens_http_timeout")
            self.sleeper(min(self.config.poll_interval_seconds, remaining))

    def _bounded_response(
        self,
        client: httpx.Client,
        method: str,
        path: str,
        deadline: float,
        *,
        files: dict[str, tuple[str, BinaryIO, str]] | None = None,
        data: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise Phase6CHTTPWorkerError("atlaslens_http_timeout")
        timeout = min(self.config.request_timeout_seconds, remaining)
        with client.stream(
            method,
            path,
            timeout=timeout,
            files=files,
            data=data,
        ) as response:
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > self.config.max_response_bytes:
                    raise Phase6CHTTPWorkerError("analysis_response_too_large")
            return response.status_code, bytes(body)

    def _map_terminal(
        self, analysis: _HTTPAnalysis, started: float
    ) -> ProviderPrediction:
        if analysis.status == "failed":
            return self._failure("analysis_failed", started)
        if analysis.status == "deleted":
            return self._failure("analysis_deleted", started)
        if (
            analysis.status != "completed"
            or analysis.pipeline_version != "phase6c-v1"
            or analysis.result_classification != "real"
            or analysis.phase6c is None
        ):
            return self._failure("phase6c_output_invalid", started)
        if (
            analysis.phase6c.reference_index_version is not None
            and analysis.phase6c.leakage_audit.status != "passed"
        ):
            return self._failure("reference_index_leakage_not_passed", started)
        if analysis.candidates and analysis.abstention is not None:
            return self._failure("analysis_outcome_invalid", started)
        if not analysis.candidates:
            if analysis.abstention is None:
                return self._failure("analysis_outcome_invalid", started)
            return ProviderPrediction(
                abstained=True,
                latency_ms=self._latency_ms(started),
                device="other",
            )

        candidates: list[CandidatePrediction] = []
        for rank, candidate in enumerate(
            analysis.candidates[: self.config.max_candidates], start=1
        ):
            assessment = candidate.phase5b_assessment
            if assessment is None:
                return self._failure("candidate_relative_score_unavailable", started)
            reverse = candidate.reverse_geocode
            candidates.append(
                CandidatePrediction(
                    rank=rank,
                    latitude=candidate.center.latitude,
                    longitude=candidate.center.longitude,
                    raw_score=assessment.relative_rank_score,
                    score_type=(
                        f"{assessment.reranker_version}:"
                        "uncalibrated_relative_rank_not_probability"
                    )[:80],
                    country_code=reverse.country_code if reverse is not None else None,
                    region=reverse.region if reverse is not None else None,
                    city_or_area=reverse.city if reverse is not None else None,
                    uncertainty_radius_km=candidate.radius_km,
                )
            )
        return ProviderPrediction(
            candidates=tuple(candidates),
            latency_ms=self._latency_ms(started),
            device="other",
        )

    def _failure(self, code: str, started: float) -> ProviderPrediction:
        return ProviderPrediction(
            failure_code=code,
            latency_ms=self._latency_ms(started),
            device="other",
        )

    def _latency_ms(self, started: float) -> float:
        return round(max(0.0, self.clock() - started) * 1_000, 3)


@dataclass(frozen=True, slots=True)
class Phase6CHTTPEvaluationProvider(EvaluationProvider):
    """Adapt the truth-free HTTP worker to the established benchmark provider seam."""

    worker: IsolatedPredictionBoundary = field(repr=False)
    provider_id: str = field(default=_PROVIDER_ID, init=False)
    model_revision: str = field(default=_MODEL_REVISION, init=False)

    def predict(self, image_path: Path) -> ProviderPrediction:
        request_id = f"benchmark-{uuid4().hex}"
        response = self.worker.predict(
            IsolatedPredictionRequest(
                request_id=request_id,
                image_path=str(image_path),
            )
        )
        if (
            response.request_id != request_id
            or response.provider_id != self.provider_id
            or response.model_revision != self.model_revision
        ):
            return ProviderPrediction(
                failure_code="phase6c_worker_response_invalid",
                latency_ms=response.prediction.latency_ms,
                device="other",
            )
        return response.prediction


def _sanitized_upload_identity(image: BinaryIO) -> tuple[str, str]:
    header = image.read(12)
    if header.startswith(b"\xff\xd8\xff"):
        return "evaluation-input.jpg", "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "evaluation-input.png", "image/png"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "evaluation-input.webp", "image/webp"
    raise Phase6CHTTPWorkerError("input_image_type_unsupported")
