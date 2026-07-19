from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NoReturn, TextIO
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from atlaslens_api.evaluation.manifest import (
    EvaluationManifestError,
    EvaluationManifestLoader,
)
from atlaslens_api.evaluation.metrics import haversine_km
from atlaslens_api.evaluation.models import EvaluationRecord, ValidatedEvaluationAsset
from atlaslens_api.evaluation.phase6c_http import Phase6CHTTPWorkerConfig

_RADII_KM = (25, 100, 250, 750)
_MANIFEST_HEADERS = tuple(EvaluationRecord.model_fields)
_CONTENT_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_HISTORY_PAGE_SIZE = 100


class Phase6CValidationError(ValueError):
    """Stable, payload-free failure at the diagnostic boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise Phase6CValidationError("arguments_invalid")


class _InputModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class _HistoryItem(_InputModel):
    id: UUID
    created_at: datetime
    status: Literal["completed"]
    result_classification: Literal["real"]
    image_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("created_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("history timestamp must include an offset")
        return value


class _HistoryPage(_InputModel):
    items: tuple[_HistoryItem, ...] = Field(max_length=_HISTORY_PAGE_SIZE)
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=_HISTORY_PAGE_SIZE)
    offset: int = Field(ge=0)


class _CoordinateCandidate(_InputModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class _Point(_InputModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class _CenteredCandidate(_InputModel):
    center: _Point


class _Phase6BPrediction(_InputModel):
    provider: str = Field(min_length=1, max_length=80)
    candidates: tuple[_CoordinateCandidate, ...] = Field(default=(), max_length=100)


class _OCRSummary(_InputModel):
    place_evidence: tuple[_CenteredCandidate, ...] = Field(default=(), max_length=12)


class _ProviderRun(_InputModel):
    provider_id: str = Field(
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    status: Literal[
        "completed",
        "abstained",
        "skipped",
        "disabled",
        "unavailable",
        "failed",
        "timeout",
    ]
    duration_ms: int = Field(ge=0, le=3_600_000)
    candidates_produced: int = Field(ge=0, le=1_000)


class _Ablation(_InputModel):
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{0,79}$")
    candidates: tuple[_CoordinateCandidate, ...] = Field(default=(), max_length=5)


class _Phase6CSummary(_InputModel):
    pipeline_version: Literal["phase6c-v1"]
    hierarchical_candidates: tuple[_CoordinateCandidate, ...] = Field(
        default=(), max_length=64
    )
    megaloc_matches: tuple[_CoordinateCandidate, ...] = Field(default=(), max_length=200)
    providers: tuple[_ProviderRun, ...] = Field(default=(), max_length=20)
    fusion_candidates: tuple[_CoordinateCandidate, ...] = Field(default=(), max_length=50)
    ablations: tuple[_Ablation, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def unique_ablations(self) -> _Phase6CSummary:
        names = [item.profile_id for item in self.ablations]
        if len(names) != len(set(names)):
            raise ValueError("ablation profile names must be unique")
        return self


class _CompletedAnalysis(_InputModel):
    id: UUID
    status: Literal["completed"]
    pipeline_version: Literal["phase6c-v1"]
    result_classification: Literal["real"]
    candidates: tuple[_CenteredCandidate, ...] = Field(default=(), max_length=100)
    model_predictions: dict[str, _Phase6BPrediction] | None = Field(
        default=None, max_length=8
    )
    ocr: _OCRSummary | None = None
    phase6c: _Phase6CSummary


@dataclass(frozen=True, slots=True)
class Phase6CValidationConfig:
    api_base_url: str
    request_timeout_seconds: float = 30.0
    max_response_bytes: int = 16 * 1024 * 1024
    max_history_records: int = 10_000
    max_analysis_fetches: int = 2_000
    max_manifest_records: int = 5_000
    max_manifest_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        # Reuse the established strict plain-loopback origin validator.
        try:
            Phase6CHTTPWorkerConfig(api_base_url=self.api_base_url)
        except ValueError as exc:
            raise Phase6CValidationError("api_base_url_invalid") from exc
        if not 0 < self.request_timeout_seconds <= 120:
            raise ValueError("request_timeout_seconds must be in (0, 120]")
        if not 1_024 <= self.max_response_bytes <= 16 * 1024 * 1024:
            raise ValueError("max_response_bytes is invalid")
        if not 1 <= self.max_history_records <= 100_000:
            raise ValueError("max_history_records is invalid")
        if not 1 <= self.max_analysis_fetches <= self.max_history_records:
            raise ValueError("max_analysis_fetches is invalid")
        if not 1 <= self.max_manifest_records <= 10_000:
            raise ValueError("max_manifest_records is invalid")
        if not 1_024 <= self.max_manifest_bytes <= 32 * 1024 * 1024:
            raise ValueError("max_manifest_bytes is invalid")


@dataclass(frozen=True, slots=True)
class _AnalysisJoin:
    by_content_sha256: Mapping[str, _CompletedAnalysis]
    duplicate_analyses_resolved: int


class Phase6CCompletedHistoryClient:
    def __init__(
        self,
        config: Phase6CValidationConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport

    def fetch(self, content_sha256: frozenset[str]) -> _AnalysisJoin:
        if not content_sha256:
            raise Phase6CValidationError("validation_split_empty")
        with httpx.Client(
            base_url=self._config.api_base_url,
            timeout=self._config.request_timeout_seconds,
            trust_env=False,
            transport=self._transport,
        ) as client:
            history = self._history(client)
            matching = [
                item for item in history if item.image_sha256 in content_sha256
            ]
            if len(matching) > self._config.max_analysis_fetches:
                raise Phase6CValidationError("analysis_fetch_limit_exceeded")
            fetched: defaultdict[str, list[tuple[_HistoryItem, _CompletedAnalysis]]] = (
                defaultdict(list)
            )
            for item in matching:
                analysis = self._analysis(client, item.id)
                if analysis is not None:
                    if analysis.id != item.id:
                        raise Phase6CValidationError("analysis_identity_mismatch")
                    if item.image_sha256 is None:
                        continue
                    fetched[item.image_sha256].append((item, analysis))

        selected: dict[str, _CompletedAnalysis] = {}
        duplicate_count = 0
        for digest, values in fetched.items():
            newest = max(
                values,
                key=lambda value: (
                    value[0].created_at.timestamp(),
                    str(value[0].id),
                ),
            )
            selected[digest] = newest[1]
            duplicate_count += len(values) - 1
        return _AnalysisJoin(
            by_content_sha256=selected,
            duplicate_analyses_resolved=duplicate_count,
        )

    def _history(self, client: httpx.Client) -> tuple[_HistoryItem, ...]:
        offset = 0
        expected_total: int | None = None
        rows: list[_HistoryItem] = []
        while expected_total is None or offset < expected_total:
            payload = self._get_json(
                client,
                "/api/v1/analyses",
                params={
                    "limit": str(_HISTORY_PAGE_SIZE),
                    "offset": str(offset),
                    "status": "completed",
                    "classification": "real",
                },
            )
            try:
                page = _HistoryPage.model_validate(payload)
            except ValidationError as exc:
                raise Phase6CValidationError("history_response_invalid") from exc
            if page.offset != offset:
                raise Phase6CValidationError("history_response_invalid")
            if expected_total is None:
                expected_total = page.total
                if expected_total > self._config.max_history_records:
                    raise Phase6CValidationError("history_limit_exceeded")
            elif page.total != expected_total:
                raise Phase6CValidationError("history_changed")
            if not page.items and offset < expected_total:
                raise Phase6CValidationError("history_response_invalid")
            rows.extend(page.items)
            offset += len(page.items)
        if len(rows) != expected_total:
            raise Phase6CValidationError("history_response_invalid")
        return tuple(rows)

    def _analysis(
        self, client: httpx.Client, analysis_id: UUID
    ) -> _CompletedAnalysis | None:
        payload = self._get_json(
            client,
            f"/api/v1/analyses/{analysis_id}",
            not_found_is_none=True,
        )
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise Phase6CValidationError("analysis_response_invalid")
        if payload.get("pipeline_version") != "phase6c-v1" or not isinstance(
            payload.get("phase6c"), dict
        ):
            return None
        try:
            return _CompletedAnalysis.model_validate(payload)
        except ValidationError as exc:
            raise Phase6CValidationError("analysis_response_invalid") from exc

    def _get_json(
        self,
        client: httpx.Client,
        path: str,
        *,
        params: dict[str, str] | None = None,
        not_found_is_none: bool = False,
    ) -> Any | None:
        with client.stream("GET", path, params=params) as response:
            if not_found_is_none and response.status_code == 404:
                return None
            if response.status_code != 200:
                raise Phase6CValidationError("loopback_api_unavailable")
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > self._config.max_response_bytes:
                    raise Phase6CValidationError("api_response_too_large")
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Phase6CValidationError("api_response_invalid") from exc


def _validation_hash_inventory(
    manifest_path: Path,
    *,
    max_records: int,
    max_bytes: int,
) -> frozenset[str]:
    try:
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or manifest_path.stat().st_size > max_bytes
        ):
            raise Phase6CValidationError("manifest_invalid")
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _MANIFEST_HEADERS:
                raise Phase6CValidationError("manifest_invalid")
            hashes: list[str] = []
            row_count = 0
            for raw in reader:
                row_count += 1
                if row_count > max_records:
                    raise Phase6CValidationError("manifest_record_limit_exceeded")
                if raw.get("split") != "validation":
                    continue
                digest = raw.get("content_sha256", "")
                if not _CONTENT_SHA256.fullmatch(digest):
                    raise Phase6CValidationError("manifest_invalid")
                hashes.append(digest)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Phase6CValidationError("manifest_invalid") from exc
    if not hashes:
        raise Phase6CValidationError("validation_split_empty")
    if len(hashes) != len(set(hashes)):
        raise Phase6CValidationError("manifest_invalid")
    return frozenset(hashes)


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6) if denominator else None,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(
    values: Sequence[float], *, unit: Literal["km", "ms"]
) -> dict[str, int | float | None]:
    suffix = unit
    return {
        "denominator": len(values),
        f"mean_{suffix}": round(statistics.fmean(values), 3) if values else None,
        f"median_{suffix}": round(statistics.median(values), 3) if values else None,
        f"p95_{suffix}": (
            round(value, 3) if (value := _percentile(values, 0.95)) is not None else None
        ),
    }


@dataclass(slots=True)
class _CandidateAggregate:
    candidate_count: int = 0
    nearest_errors: list[float] = field(default_factory=list)
    hits: Counter[int] = field(default_factory=Counter)

    def observe(
        self,
        candidates: Sequence[_CoordinateCandidate | _CenteredCandidate],
        *,
        truth_latitude: float,
        truth_longitude: float,
    ) -> None:
        self.candidate_count += len(candidates)
        errors = [
            haversine_km(
                truth_latitude,
                truth_longitude,
                point.latitude,
                point.longitude,
            )
            for item in candidates
            for point in [item.center if isinstance(item, _CenteredCandidate) else item]
        ]
        if not errors:
            return
        nearest = min(errors)
        self.nearest_errors.append(nearest)
        for radius in _RADII_KM:
            if nearest <= radius:
                self.hits[radius] += 1

    def report(self, sample_count: int) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "coverage": _ratio(len(self.nearest_errors), sample_count),
            "nearest_error_km": _distribution(self.nearest_errors, unit="km"),
            "recall_within_km": {
                str(radius): _ratio(self.hits[radius], sample_count)
                for radius in _RADII_KM
            },
        }


@dataclass(slots=True)
class _ProviderAggregate:
    observed_samples: int = 0
    runs: int = 0
    candidate_return_samples: int = 0
    candidates_produced: int = 0
    statuses: Counter[str] = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)


def _component_candidates(
    analysis: _CompletedAnalysis,
) -> dict[str, tuple[_CoordinateCandidate | _CenteredCandidate, ...]]:
    predictions = tuple((analysis.model_predictions or {}).values())
    osv = tuple(
        candidate
        for prediction in predictions
        if prediction.provider.casefold().startswith("osv5m")
        for candidate in prediction.candidates
    )
    plonk = tuple(
        candidate
        for prediction in predictions
        if prediction.provider.casefold().startswith("plonk")
        for candidate in prediction.candidates
    )
    return {
        "hierarchical": analysis.phase6c.hierarchical_candidates,
        "megaloc": analysis.phase6c.megaloc_matches,
        "osv5m": osv,
        "plonk": plonk,
        "ocr_place_evidence": (
            analysis.ocr.place_evidence if analysis.ocr is not None else ()
        ),
        "final_fusion": analysis.phase6c.fusion_candidates,
        "public_candidates": analysis.candidates,
    }


def _score(
    assets: Sequence[ValidatedEvaluationAsset],
    join: _AnalysisJoin,
) -> dict[str, Any]:
    sample_count = len(assets)
    components = {
        name: _CandidateAggregate()
        for name in (
            "hierarchical",
            "megaloc",
            "osv5m",
            "plonk",
            "ocr_place_evidence",
            "final_fusion",
            "public_candidates",
        )
    }
    ablation_names = sorted(
        {
            item.profile_id
            for analysis in join.by_content_sha256.values()
            for item in analysis.phase6c.ablations
        }
    )
    ablations = {name: _CandidateAggregate() for name in ablation_names}
    providers: defaultdict[str, _ProviderAggregate] = defaultdict(_ProviderAggregate)
    matched = 0

    for asset in assets:
        truth = asset.record
        analysis = join.by_content_sha256.get(truth.content_sha256)
        candidate_groups = _component_candidates(analysis) if analysis is not None else {}
        if analysis is not None:
            matched += 1
        for name, aggregate in components.items():
            aggregate.observe(
                candidate_groups.get(name, ()),
                truth_latitude=truth.true_latitude,
                truth_longitude=truth.true_longitude,
            )
        analysis_ablations = (
            {item.profile_id: item.candidates for item in analysis.phase6c.ablations}
            if analysis is not None
            else {}
        )
        for name, aggregate in ablations.items():
            aggregate.observe(
                analysis_ablations.get(name, ()),
                truth_latitude=truth.true_latitude,
                truth_longitude=truth.true_longitude,
            )
        if analysis is None:
            continue
        seen_provider_names: set[str] = set()
        provider_with_candidates: set[str] = set()
        for provider in analysis.phase6c.providers:
            provider_aggregate = providers[provider.provider_id]
            provider_aggregate.runs += 1
            provider_aggregate.statuses[provider.status] += 1
            provider_aggregate.latencies_ms.append(float(provider.duration_ms))
            provider_aggregate.candidates_produced += provider.candidates_produced
            seen_provider_names.add(provider.provider_id)
            if provider.candidates_produced:
                provider_with_candidates.add(provider.provider_id)
        for name in seen_provider_names:
            providers[name].observed_samples += 1
        for name in provider_with_candidates:
            providers[name].candidate_return_samples += 1

    return {
        "schema_version": "atlaslens-phase6c-validation-diagnostic-v1",
        "prediction_boundary": "completed_history_get_only_no_inference",
        "split": "validation",
        "sample_count": sample_count,
        "analysis_join": {
            "matched_completed_phase6c": matched,
            "unmatched": sample_count - matched,
            "coverage": _ratio(matched, sample_count),
            "duplicate_analyses_resolved": join.duplicate_analyses_resolved,
        },
        "components": {
            name: aggregate.report(sample_count)
            for name, aggregate in components.items()
        },
        "ablations": {
            name: aggregate.report(sample_count)
            for name, aggregate in ablations.items()
        },
        "providers": {
            name: {
                "observed_samples": aggregate.observed_samples,
                "observed_runs": aggregate.runs,
                "coverage": _ratio(aggregate.observed_samples, sample_count),
                "candidate_return": _ratio(
                    aggregate.candidate_return_samples, sample_count
                ),
                "candidates_produced": aggregate.candidates_produced,
                "status_counts": dict(sorted(aggregate.statuses.items())),
                "latency_ms": _distribution(aggregate.latencies_ms, unit="ms"),
            }
            for name, aggregate in sorted(providers.items())
        },
        "no_claim": True,
        "claim_reason": "diagnostic_raw_candidate_recall_not_accuracy_claim",
    }


def build_validation_diagnostic(
    *,
    config: Phase6CValidationConfig,
    manifest_path: Path,
    asset_root: Path,
    allowed_licenses: frozenset[str],
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    # This first pass intentionally reads only split and content hash. It neither
    # validates nor projects truth fields to the prediction/history boundary.
    content_sha256 = _validation_hash_inventory(
        manifest_path,
        max_records=config.max_manifest_records,
        max_bytes=config.max_manifest_bytes,
    )
    join = Phase6CCompletedHistoryClient(config, transport=transport).fetch(content_sha256)

    # Truth is parsed and validated only after all GET-only prediction fetches finish.
    manifest = EvaluationManifestLoader(allowed_licenses=allowed_licenses).load(
        manifest_path, asset_root
    )
    assets = tuple(asset for asset in manifest.assets if asset.record.split == "validation")
    if frozenset(asset.record.content_sha256 for asset in assets) != content_sha256:
        raise Phase6CValidationError("manifest_changed")
    return _score(assets, join)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Phase6CValidationError("output_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise Phase6CValidationError("output_exists")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Score completed Phase 6C validation history.")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--allowed-license", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-response-mb", type=int, default=16)
    parser.add_argument("--max-history-records", type=int, default=10_000)
    parser.add_argument("--max-analysis-fetches", type=int, default=2_000)
    parser.add_argument("--max-manifest-records", type=int, default=5_000)
    parser.add_argument("--max-manifest-mb", type=int, default=8)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    try:
        args = build_parser().parse_args(tuple(argv) if argv is not None else sys.argv[1:])
        config = Phase6CValidationConfig(
            api_base_url=args.api_base_url,
            request_timeout_seconds=args.request_timeout_seconds,
            max_response_bytes=args.max_response_mb * 1024 * 1024,
            max_history_records=args.max_history_records,
            max_analysis_fetches=args.max_analysis_fetches,
            max_manifest_records=args.max_manifest_records,
            max_manifest_bytes=args.max_manifest_mb * 1024 * 1024,
        )
        report = build_validation_diagnostic(
            config=config,
            manifest_path=args.manifest,
            asset_root=args.asset_root,
            allowed_licenses=frozenset(str(item) for item in args.allowed_license),
            transport=transport,
        )
        _write_report(args.output, report)
    except Phase6CValidationError as exc:
        error_stream.write(
            json.dumps(
                {
                    "event": "phase6c_validation_diagnostic_failed",
                    "reason_code": exc.code,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        return 2
    except (EvaluationManifestError, OSError, ValueError, httpx.HTTPError):
        error_stream.write(
            '{"event":"phase6c_validation_diagnostic_failed",'
            '"reason_code":"diagnostic_failed"}\n'
        )
        return 2
    except Exception:  # Keep unexpected payload-bearing exceptions out of the console.
        error_stream.write(
            '{"event":"phase6c_validation_diagnostic_failed",'
            '"reason_code":"diagnostic_failed"}\n'
        )
        return 2
    output_stream.write(
        json.dumps(
            {
                "event": "phase6c_validation_diagnostic_completed",
                "sample_count": report["sample_count"],
                "matched_completed_phase6c": report["analysis_join"][
                    "matched_completed_phase6c"
                ],
                "no_claim": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0
