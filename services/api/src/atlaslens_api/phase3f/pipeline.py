"""Resumable Phase 3F corpus/descriptor/benchmark state machine."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from .acquisition import AcquisitionGuard
from .benchmark import AbstentionThresholdLock, BenchmarkResult
from .coverage import CoverageSelectionLock
from .splits import SealedSplit

DESCRIPTOR_DIMENSION: Final = 8_448
MAX_LOCAL_DERIVED_ARTIFACT_BYTES: Final = 5 * 1024 * 1024 * 1024
MODEL_SHA256: Final = "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
MODEL_REVISION: Final = "7cb9f7970d366fdf059963d04d372e503e8e9df9"
PROVIDER_REVISION: Final = "1af071c68fc3ab6c6018c5c868391763516e50f7"

PREREGISTERED_GATES: Mapping[str, object] = {
    "minimum_in_domain_cities": 5,
    "minimum_ood_cities": 2,
    "all_leakage_invariants": True,
    "per_city_minimums": True,
    "city_top1_minimum": 0.50,
    "city_top3_minimum": 0.75,
    "accepted_city_top1_minimum": 0.70,
    "accepted_query_coverage_minimum": 0.25,
    "ood_false_accept_maximum": 0.15,
    "ood_abstention_minimum": 0.70,
    "accepted_median_geodesic_error_km_maximum": 75.0,
    "maximum_accepted_city_share": 0.40,
    "secret_security_artifact_integrity": True,
}
PREREGISTERED_GATES_SHA256: Final = hashlib.sha256(
    json.dumps(PREREGISTERED_GATES, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


class PipelineStage(StrEnum):
    CREATED = "CREATED"
    SELECTION_LOCKED = "SELECTION_LOCKED"
    ACQUISITION_COMPLETE = "ACQUISITION_COMPLETE"
    SPLIT_SEALED = "SPLIT_SEALED"
    DESCRIPTORS_PUBLISHED = "DESCRIPTORS_PUBLISHED"
    THRESHOLD_LOCKED = "THRESHOLD_LOCKED"
    HOLDOUT_EVALUATED = "HOLDOUT_EVALUATED"
    FINALIZED = "FINALIZED"
    COVERAGE_INSUFFICIENT = "COVERAGE_INSUFFICIENT"


class PipelineStateError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class DescriptorPublication:
    selection_lock_sha256: str
    split_lock_sha256: str
    source_policy_sha256: str
    descriptor_publication_sha256: str
    index_sha256: str
    city_scope: tuple[str, ...]
    created_at: datetime
    descriptor_dimension: int = DESCRIPTOR_DIMENSION
    descriptor_finite: bool = True
    descriptor_l2_normalized: bool = True
    index_backend: str = "faiss.IndexFlatIP"
    offline_model_loading: bool = True

    def __post_init__(self) -> None:
        for value in (
            self.selection_lock_sha256,
            self.split_lock_sha256,
            self.source_policy_sha256,
            self.descriptor_publication_sha256,
            self.index_sha256,
        ):
            _require_sha256(value)
        if self.descriptor_dimension != DESCRIPTOR_DIMENSION:
            raise ValueError("MegaLoc descriptor dimension must be 8448")
        if not self.descriptor_finite or not self.descriptor_l2_normalized:
            raise ValueError("descriptor publication must be finite and L2-normalized")
        if self.index_backend != "faiss.IndexFlatIP":
            raise ValueError("Phase 3F permits only deterministic FAISS IndexFlatIP")
        if not self.offline_model_loading:
            raise ValueError("model loading must be offline")
        if len(self.city_scope) < 5 or len(set(self.city_scope)) != len(self.city_scope):
            raise ValueError("descriptor city scope is invalid")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("publication time must be timezone-aware")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": "atlaslens-phase3f-descriptor-publication-v1",
            "model_sha256": MODEL_SHA256,
            "model_revision": MODEL_REVISION,
            "provider_revision": PROVIDER_REVISION,
            "selection_lock_sha256": self.selection_lock_sha256,
            "split_lock_sha256": self.split_lock_sha256,
            "source_policy_sha256": self.source_policy_sha256,
            "descriptor_publication_sha256": self.descriptor_publication_sha256,
            "index_sha256": self.index_sha256,
            "descriptor_dimension": self.descriptor_dimension,
            "descriptor_finite": self.descriptor_finite,
            "descriptor_l2_normalized": self.descriptor_l2_normalized,
            "index_backend": self.index_backend,
            "offline_model_loading": self.offline_model_loading,
            "city_scope": list(self.city_scope),
            "created_at": self.created_at.astimezone(UTC).isoformat(),
        }


class Phase3FPipeline:
    """Persist only sanitized state; raw media and evaluator truth stay outside it."""

    def __init__(self, state_path: Path, run_id: str, document: dict[str, object]) -> None:
        self._state_path = state_path.resolve()
        self._run_id = run_id
        self._document = document

    @classmethod
    def create(cls, state_path: Path, *, run_id: str) -> Phase3FPipeline:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("run_id must be opaque 128-bit lowercase hex")
        resolved = state_path.resolve()
        if resolved.exists() or resolved.is_symlink():
            raise PipelineStateError("PHASE3F_STATE_ALREADY_EXISTS")
        document: dict[str, object] = {
            "schema": "atlaslens-phase3f-pipeline-state-v1",
            "run_id": run_id,
            "stage": PipelineStage.CREATED.value,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "preregistered_gates": dict(PREREGISTERED_GATES),
            "preregistered_gates_sha256": PREREGISTERED_GATES_SHA256,
            "holdout_open_count": 0,
            "events": [],
        }
        pipeline = cls(resolved, run_id, document)
        pipeline._persist()
        return pipeline

    @classmethod
    def resume(cls, state_path: Path) -> Phase3FPipeline:
        resolved = state_path.resolve()
        if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size > 1_048_576:
            raise PipelineStateError("PHASE3F_STATE_INVALID")
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PipelineStateError("PHASE3F_STATE_INVALID") from exc
        if not isinstance(value, dict):
            raise PipelineStateError("PHASE3F_STATE_INVALID")
        document = {str(key): item for key, item in value.items()}
        run_id = document.get("run_id")
        if (
            document.get("schema") != "atlaslens-phase3f-pipeline-state-v1"
            or not isinstance(run_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", run_id)
            or document.get("preregistered_gates_sha256") != PREREGISTERED_GATES_SHA256
            or document.get("preregistered_gates") != PREREGISTERED_GATES
        ):
            raise PipelineStateError("PHASE3F_STATE_INVALID")
        try:
            PipelineStage(str(document.get("stage")))
        except ValueError as exc:
            raise PipelineStateError("PHASE3F_STATE_INVALID") from exc
        return cls(resolved, run_id, document)

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage(str(self._document["stage"]))

    @property
    def document(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(json.dumps(self._document)))

    def lock_selection(self, lock: CoverageSelectionLock) -> None:
        self._transition(
            PipelineStage.CREATED,
            PipelineStage.SELECTION_LOCKED,
            {
                "selection_lock_sha256": lock.lock_sha256,
                "in_domain_cities": [record.city for record in lock.in_domain],
                "ood_cities": [record.city for record in lock.ood],
                "selection_stage": "metadata_only_before_image_acquisition",
            },
        )

    def complete_acquisition(self, guard: AcquisitionGuard) -> None:
        if guard.in_flight != 0:
            raise PipelineStateError("PHASE3F_ACQUISITION_IN_FLIGHT")
        self._transition(
            PipelineStage.SELECTION_LOCKED,
            PipelineStage.ACQUISITION_COMPLETE,
            {"acquisition": guard.receipt()},
        )

    def record_split(self, split: SealedSplit) -> None:
        if not split.leakage.passed:
            raise PipelineStateError("PHASE3F_LEAKAGE_VIOLATION")
        self._transition(
            PipelineStage.ACQUISITION_COMPLETE,
            PipelineStage.SPLIT_SEALED,
            {
                "split_lock_sha256": split.split_lock_sha256,
                "holdout_seal_sha256": split.holdout_seal_sha256,
                "evaluator_truth_sha256": split.evaluator_truth_sha256,
                "ground_truth_in_inference_state": False,
                "leakage": split.leakage.document(),
            },
        )

    def record_descriptor_publication(self, publication: DescriptorPublication) -> None:
        if publication.selection_lock_sha256 != self._document.get("selection_lock_sha256"):
            raise PipelineStateError("PHASE3F_SELECTION_LOCK_MISMATCH")
        if publication.split_lock_sha256 != self._document.get("split_lock_sha256"):
            raise PipelineStateError("PHASE3F_SPLIT_LOCK_MISMATCH")
        self._transition(
            PipelineStage.SPLIT_SEALED,
            PipelineStage.DESCRIPTORS_PUBLISHED,
            {"descriptor_publication": publication.document()},
        )

    def lock_threshold(self, threshold: AbstentionThresholdLock) -> None:
        if threshold.selection_lock_sha256 != self._document.get("selection_lock_sha256"):
            raise PipelineStateError("PHASE3F_SELECTION_LOCK_MISMATCH")
        if threshold.split_lock_sha256 != self._document.get("split_lock_sha256"):
            raise PipelineStateError("PHASE3F_SPLIT_LOCK_MISMATCH")
        self._transition(
            PipelineStage.DESCRIPTORS_PUBLISHED,
            PipelineStage.THRESHOLD_LOCKED,
            {
                "threshold_lock_sha256": threshold.lock_sha256,
                "threshold": threshold.document(),
            },
        )

    def record_benchmark(self, benchmark: BenchmarkResult) -> None:
        holdout_open_count = self._document.get("holdout_open_count")
        if (
            isinstance(holdout_open_count, bool)
            or not isinstance(holdout_open_count, int)
            or holdout_open_count != 0
        ):
            raise PipelineStateError("PHASE3F_HOLDOUT_ALREADY_OPENED")
        if benchmark.threshold_lock_sha256 != self._document.get("threshold_lock_sha256"):
            raise PipelineStateError("PHASE3F_THRESHOLD_LOCK_MISMATCH")
        self._document["holdout_open_count"] = 1
        self._transition(
            PipelineStage.THRESHOLD_LOCKED,
            PipelineStage.HOLDOUT_EVALUATED,
            {
                "benchmark_result_sha256": benchmark.result_sha256,
                "outcome": benchmark.outcome,
                "activation_gates": benchmark.gates,
            },
        )

    def finalize(self) -> None:
        self._transition(
            PipelineStage.HOLDOUT_EVALUATED,
            PipelineStage.FINALIZED,
            {"finalized": True},
        )

    def finalize_coverage_insufficient(self, *, reason_code: str) -> None:
        if self.stage not in {PipelineStage.CREATED, PipelineStage.SELECTION_LOCKED}:
            raise PipelineStateError("PHASE3F_COVERAGE_TERMINAL_STAGE_INVALID")
        if not re.fullmatch(r"[A-Z0-9_]{3,64}", reason_code):
            raise ValueError("coverage reason code is invalid")
        self._transition(
            self.stage,
            PipelineStage.COVERAGE_INSUFFICIENT,
            {
                "outcome": "COVERAGE_INSUFFICIENT",
                "reason_code": reason_code,
                "image_acquisition_started": False,
                "finalized": True,
            },
        )

    def _transition(
        self,
        expected: PipelineStage,
        next_stage: PipelineStage,
        updates: Mapping[str, object],
    ) -> None:
        if self.stage != expected:
            raise PipelineStateError("PHASE3F_STATE_TRANSITION_REFUSED")
        self._document.update(updates)
        self._document["stage"] = next_stage.value
        self._document["updated_at"] = datetime.now(UTC).isoformat()
        events = self._document.get("events")
        if not isinstance(events, list) or len(events) >= 32:
            raise PipelineStateError("PHASE3F_STATE_INVALID")
        events.append({"stage": next_stage.value, "at": self._document["updated_at"]})
        self._persist()

    def _persist(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        if self._state_path.parent.is_symlink():
            raise PipelineStateError("PHASE3F_STATE_PATH_INVALID")
        payload = json.dumps(
            self._document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        temporary = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        if temporary.exists() or temporary.is_symlink():
            raise PipelineStateError("PHASE3F_STATE_TEMP_EXISTS")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._state_path)
        finally:
            temporary.unlink(missing_ok=True)


def validate_derived_bundle_size(total_bytes: int) -> None:
    if total_bytes < 0 or total_bytes > MAX_LOCAL_DERIVED_ARTIFACT_BYTES:
        raise PipelineStateError("PHASE3F_DERIVED_ARTIFACT_CAP_REACHED")


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("expected lowercase SHA-256")


__all__ = [
    "DescriptorPublication",
    "Phase3FPipeline",
    "PipelineStage",
    "PipelineStateError",
    "PREREGISTERED_GATES",
    "PREREGISTERED_GATES_SHA256",
    "validate_derived_bundle_size",
]
