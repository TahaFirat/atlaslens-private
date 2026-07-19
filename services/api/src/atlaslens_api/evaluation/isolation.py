from __future__ import annotations

import hashlib
import os
import subprocess
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlaslens_api.evaluation.holdout import (
    HoldoutManifestError,
    HoldoutManifestLoader,
    PredictionInput,
)
from atlaslens_api.evaluation.metrics import latency_distribution, ratio
from atlaslens_api.evaluation.models import (
    LatencyDistribution,
    ProviderPrediction,
    RatioMetric,
)


class PredictionBoundaryError(RuntimeError):
    """Safe error raised by an isolated prediction process."""


class _ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class IsolatedPredictionRequest(_ProtocolModel):
    protocol_version: Literal["atlaslens-isolated-prediction-v1"] = (
        "atlaslens-isolated-prediction-v1"
    )
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    image_path: str = Field(min_length=1, max_length=1_024)


class IsolatedPredictionResponse(_ProtocolModel):
    protocol_version: Literal["atlaslens-isolated-prediction-v1"] = (
        "atlaslens-isolated-prediction-v1"
    )
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    provider_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    model_revision: str = Field(min_length=1, max_length=160)
    prediction: ProviderPrediction

    @model_validator(mode="after")
    def require_candidate_uncertainty(self) -> IsolatedPredictionResponse:
        if any(
            candidate.uncertainty_radius_km is None
            for candidate in self.prediction.candidates
        ):
            raise ValueError("isolated candidates require a positive uncertainty radius")
        return self


class IsolatedPredictionBoundary(Protocol):
    def predict(self, request: IsolatedPredictionRequest) -> IsolatedPredictionResponse: ...


class IsolatedEvaluationSample(_ProtocolModel):
    evaluation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    provider_id: str
    model_revision: str
    returned_candidates: int = Field(ge=0, le=100)
    abstained: bool
    failure_code: str | None
    latency_ms: float = Field(ge=0)
    country_top1_correct: bool
    city_top1_correct: bool
    city_top3_correct: bool
    city_top5_correct: bool
    ground_truth_city_rank: int | None = Field(default=None, ge=1, le=100)


class IsolatedEvaluationReport(_ProtocolModel):
    evaluation_protocol: Literal["atlaslens-isolated-prediction-v1"] = (
        "atlaslens-isolated-prediction-v1"
    )
    manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(gt=0)
    predictions_completed_before_scoring: Literal[True] = True
    successful_inference: RatioMetric
    candidate_return: RatioMetric
    abstention: RatioMetric
    provider_failure: RatioMetric
    country_top1: RatioMetric
    city_top1: RatioMetric
    city_top3: RatioMetric
    city_top5: RatioMetric
    latency_ms: LatencyDistribution
    accuracy_claim_allowed: bool
    claim_note: Literal[
        "insufficient_sample_no_accuracy_claim",
        "sample_gate_met_dataset_review_still_required",
    ]
    samples: tuple[IsolatedEvaluationSample, ...]


def _child_environment() -> dict[str, str]:
    blocked_fragments = ("GROUND_TRUTH", "EVALUATION_MANIFEST", "HOLDOUT_MANIFEST")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(fragment in key.upper() for fragment in blocked_fragments)
    }
    environment["PYTHONUTF8"] = "1"
    return environment


@dataclass(frozen=True, slots=True)
class ProcessPredictionClient:
    """One request per child process; stdin has protocol metadata plus only the image path."""

    command: tuple[str, ...]
    timeout_seconds: float = 120.0
    max_response_bytes: int = 1024 * 1024
    working_directory: Path | None = None

    def __post_init__(self) -> None:
        if not self.command or len(self.command) > 32:
            raise ValueError("prediction command must contain 1 to 32 arguments")
        if any(not item or len(item) > 4_096 for item in self.command):
            raise ValueError("prediction command contains an invalid argument")
        if not 0 < self.timeout_seconds <= 3_600:
            raise ValueError("prediction timeout must be in (0, 3600]")
        if not 1_024 <= self.max_response_bytes <= 16 * 1024 * 1024:
            raise ValueError("prediction response bound is invalid")

    def predict(self, request: IsolatedPredictionRequest) -> IsolatedPredictionResponse:
        payload = request.model_dump_json().encode("utf-8")
        try:
            completed = subprocess.run(  # noqa: S603 - operator supplies an argv, never a shell.
                self.command,
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
                cwd=self.working_directory,
                env=_child_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PredictionBoundaryError("prediction_process_unavailable") from exc
        if completed.returncode != 0:
            raise PredictionBoundaryError("prediction_process_failed")
        if not completed.stdout or len(completed.stdout) > self.max_response_bytes:
            raise PredictionBoundaryError("prediction_response_size_invalid")
        try:
            response = IsolatedPredictionResponse.model_validate_json(completed.stdout)
        except ValueError as exc:
            raise PredictionBoundaryError("prediction_response_invalid") from exc
        if response.request_id != request.request_id:
            raise PredictionBoundaryError("prediction_response_id_mismatch")
        return response


def _normalized_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return normalized or None


class LeakageSafeEvaluationRunner:
    """Finish isolated predictions before loading labels and entering the scoring stage."""

    def __init__(
        self,
        *,
        manifest_loader: HoldoutManifestLoader | None = None,
        max_samples: int = 1_000,
        minimum_claim_samples: int = 100,
    ) -> None:
        if not 1 <= max_samples <= 10_000:
            raise ValueError("max_samples must be in [1, 10000]")
        if not 1 <= minimum_claim_samples <= 1_000_000:
            raise ValueError("minimum_claim_samples must be positive and bounded")
        self._manifest_loader = manifest_loader or HoldoutManifestLoader(
            max_records=max_samples
        )
        self._max_samples = max_samples
        self._minimum_claim_samples = minimum_claim_samples

    def run(
        self,
        manifest_path: Path,
        prediction_boundary: IsolatedPredictionBoundary,
    ) -> IsolatedEvaluationReport:
        prediction_inputs = self._manifest_loader.load_prediction_inputs(manifest_path)
        if len(prediction_inputs) > self._max_samples:
            raise HoldoutManifestError("holdout_runner_sample_limit_exceeded")

        responses = tuple(
            (item.evaluation_id, self._predict(item, prediction_boundary))
            for item in prediction_inputs
        )

        # This is intentionally the first full/truth-bearing manifest load in the run.
        truth_manifest = self._manifest_loader.load(manifest_path)
        truth = {item.metadata.id: item for item in truth_manifest.records}
        if set(truth) != {item.evaluation_id for item in prediction_inputs}:
            raise HoldoutManifestError("holdout_prediction_truth_projection_mismatch")

        samples: list[IsolatedEvaluationSample] = []
        for evaluation_id, response in responses:
            record = truth[evaluation_id].metadata
            prediction = response.prediction
            candidates = prediction.candidates
            expected_city = _normalized_label(record.city)
            city_rank = next(
                (
                    candidate.rank
                    for candidate in candidates
                    if _normalized_label(candidate.city_or_area) == expected_city
                ),
                None,
            )
            top1 = candidates[0] if candidates else None
            samples.append(
                IsolatedEvaluationSample(
                    evaluation_id=record.id,
                    provider_id=response.provider_id,
                    model_revision=response.model_revision,
                    returned_candidates=len(candidates),
                    abstained=prediction.abstained,
                    failure_code=prediction.failure_code,
                    latency_ms=prediction.latency_ms,
                    country_top1_correct=(
                        top1 is not None and top1.country_code == record.country
                    ),
                    city_top1_correct=city_rank == 1,
                    city_top3_correct=city_rank is not None and city_rank <= 3,
                    city_top5_correct=city_rank is not None and city_rank <= 5,
                    ground_truth_city_rank=city_rank,
                )
            )

        count = len(samples)
        claim_allowed = count >= self._minimum_claim_samples
        return IsolatedEvaluationReport(
            manifest_fingerprint=truth_manifest.fingerprint,
            sample_count=count,
            successful_inference=ratio(
                sum(item.failure_code is None for item in samples), count
            ),
            candidate_return=ratio(
                sum(item.returned_candidates > 0 for item in samples), count
            ),
            abstention=ratio(sum(item.abstained for item in samples), count),
            provider_failure=ratio(
                sum(item.failure_code is not None for item in samples), count
            ),
            country_top1=ratio(
                sum(item.country_top1_correct for item in samples), count
            ),
            city_top1=ratio(sum(item.city_top1_correct for item in samples), count),
            city_top3=ratio(sum(item.city_top3_correct for item in samples), count),
            city_top5=ratio(sum(item.city_top5_correct for item in samples), count),
            latency_ms=latency_distribution(item.latency_ms for item in samples),
            accuracy_claim_allowed=claim_allowed,
            claim_note=(
                "sample_gate_met_dataset_review_still_required"
                if claim_allowed
                else "insufficient_sample_no_accuracy_claim"
            ),
            samples=tuple(samples),
        )

    @staticmethod
    def _predict(
        item: PredictionInput,
        boundary: IsolatedPredictionBoundary,
    ) -> IsolatedPredictionResponse:
        request_id = "prediction-" + hashlib.sha256(
            item.evaluation_id.encode("utf-8")
        ).hexdigest()[:24]
        request = IsolatedPredictionRequest(
            request_id=request_id,
            image_path=str(item.image_path),
        )
        try:
            response = boundary.predict(request)
        except PredictionBoundaryError:
            return IsolatedPredictionResponse(
                request_id=request_id,
                provider_id="isolated-boundary",
                model_revision="unavailable",
                prediction=ProviderPrediction(
                    failure_code="prediction_boundary_failed",
                    latency_ms=0,
                    device="other",
                ),
            )
        if response.request_id != request_id:
            raise PredictionBoundaryError("prediction_response_id_mismatch")
        return response


def prediction_command(value: Sequence[str]) -> tuple[str, ...]:
    """Normalize argparse REMAINDER values without accepting shell syntax."""

    selected = tuple(value[1:] if value and value[0] == "--" else value)
    if not selected:
        raise ValueError("prediction command is required")
    return selected
