from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from .errors import CheckpointCompatibilityError, CheckpointError
from .models import OPAQUE_ID_PATTERN, SHA256_PATTERN, StrictModel
from .safety import atomic_write_model, read_bounded_bytes, resolve_output_file


class PipelineState(StrEnum):
    CREATED = "CREATED"
    MANIFEST_VALIDATED = "MANIFEST_VALIDATED"
    RIGHTS_VALIDATED = "RIGHTS_VALIDATED"
    INGESTED = "INGESTED"
    DEDUPLICATED = "DEDUPLICATED"
    SPLIT_LOCKED = "SPLIT_LOCKED"
    DESCRIPTORS_BUILT = "DESCRIPTORS_BUILT"
    INDEX_BUILT = "INDEX_BUILT"
    BENCHMARKED = "BENCHMARKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


PIPELINE_SEQUENCE = (
    PipelineState.CREATED,
    PipelineState.MANIFEST_VALIDATED,
    PipelineState.RIGHTS_VALIDATED,
    PipelineState.INGESTED,
    PipelineState.DEDUPLICATED,
    PipelineState.SPLIT_LOCKED,
    PipelineState.DESCRIPTORS_BUILT,
    PipelineState.INDEX_BUILT,
    PipelineState.BENCHMARKED,
    PipelineState.COMPLETED,
)
_STATE_POSITION = {state: index for index, state in enumerate(PIPELINE_SEQUENCE)}
_ARTIFACT_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,239}$")


class TransitionEvent(StrictModel):
    operation: Literal["initialize", "begin", "complete", "fail", "cancel", "resume"]
    from_state: PipelineState
    to_state: PipelineState
    at: datetime

    @field_validator("at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checkpoint event must include a timezone")
        return value.astimezone(UTC)


class StepProgress(StrictModel):
    state: PipelineState
    completed_units: int = Field(ge=0)
    total_units: int = Field(ge=0)
    last_completed_asset_id: str | None = Field(default=None, pattern=OPAQUE_ID_PATTERN)
    partial_artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def require_bounded_progress(self) -> Self:
        if self.completed_units > self.total_units:
            raise ValueError("completed progress exceeds total")
        return self


class PipelineCheckpoint(StrictModel):
    schema_version: Literal["atlaslens-corpus-checkpoint-v1"] = (
        "atlaslens-corpus-checkpoint-v1"
    )
    run_id: str = Field(pattern=OPAQUE_ID_PATTERN)
    state: PipelineState
    last_successful_state: PipelineState
    active_state: PipelineState | None = None
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    manifest_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    source_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    progress: StepProgress | None = None
    failure_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,79}$")
    history: tuple[TransitionEvent, ...]
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checkpoint timestamp must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def require_consistent_state(self) -> Self:
        if self.last_successful_state not in _STATE_POSITION:
            raise ValueError("last successful state cannot be terminal failure")
        if self.state in _STATE_POSITION and self.state != self.last_successful_state:
            raise ValueError("current and last successful states must agree")
        if self.state in {PipelineState.FAILED, PipelineState.CANCELLED} and self.active_state:
            raise ValueError("terminal failure state cannot remain active")
        if self.active_state is not None:
            expected_position = _STATE_POSITION[self.last_successful_state] + 1
            if (
                expected_position >= len(PIPELINE_SEQUENCE)
                or self.active_state != PIPELINE_SEQUENCE[expected_position]
            ):
                raise ValueError("active checkpoint state is not the next transition")
            if self.progress is not None and self.progress.state != self.active_state:
                raise ValueError("progress does not match active state")
        elif self.progress is not None:
            raise ValueError("progress requires an active state")
        for key, digest in self.artifact_sha256.items():
            if (
                _ARTIFACT_KEY.fullmatch(key) is None
                or ".." in key.split("/")
                or re.fullmatch(SHA256_PATTERN, digest) is None
            ):
                raise ValueError("checkpoint artifact record is invalid")
        return self


class CheckpointStore:
    """Atomic state-machine persistence shared by ingestion and later index stages."""

    def __init__(self, work_root: Path, checkpoint_path: Path | str) -> None:
        self.path = resolve_output_file(work_root, checkpoint_path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> PipelineCheckpoint:
        if not self.path.exists() or self.path.is_symlink() or not self.path.is_file():
            raise CheckpointError("checkpoint_missing")
        try:
            payload = read_bounded_bytes(
                self.path,
                max_bytes=2 * 1024 * 1024,
                error_prefix="checkpoint",
            )
            return PipelineCheckpoint.model_validate(json.loads(payload))
        except CheckpointError:
            raise
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise CheckpointError("checkpoint_invalid") from exc

    def initialize(
        self,
        *,
        run_id: str,
        manifest_sha256: str,
        manifest_schema_sha256: str,
        source_policy_sha256: str,
        config_sha256: str,
        resume: bool,
    ) -> PipelineCheckpoint:
        expected = {
            "manifest_sha256": manifest_sha256,
            "manifest_schema_sha256": manifest_schema_sha256,
            "source_policy_sha256": source_policy_sha256,
            "config_sha256": config_sha256,
        }
        if self.exists():
            if not resume:
                raise CheckpointError("checkpoint_exists_resume_required")
            checkpoint = self.load()
            self.verify_compatibility(checkpoint, **expected)
            if checkpoint.run_id != run_id:
                raise CheckpointCompatibilityError("checkpoint_run_id_mismatch")
            if checkpoint.state == PipelineState.COMPLETED:
                return checkpoint
            if (
                checkpoint.state in {PipelineState.FAILED, PipelineState.CANCELLED}
                or checkpoint.active_state is not None
            ):
                now = datetime.now(UTC)
                resumed = checkpoint.model_copy(
                    update={
                        "state": checkpoint.last_successful_state,
                        "active_state": None,
                        "progress": None,
                        "failure_code": None,
                        "history": checkpoint.history
                        + (
                            TransitionEvent(
                                operation="resume",
                                from_state=checkpoint.state,
                                to_state=checkpoint.last_successful_state,
                                at=now,
                            ),
                        ),
                        "updated_at": now,
                    }
                )
                return self._write(resumed)
            return checkpoint
        now = datetime.now(UTC)
        checkpoint = PipelineCheckpoint(
            run_id=run_id,
            state=PipelineState.CREATED,
            last_successful_state=PipelineState.CREATED,
            manifest_sha256=manifest_sha256,
            manifest_schema_sha256=manifest_schema_sha256,
            source_policy_sha256=source_policy_sha256,
            config_sha256=config_sha256,
            history=(
                TransitionEvent(
                    operation="initialize",
                    from_state=PipelineState.CREATED,
                    to_state=PipelineState.CREATED,
                    at=now,
                ),
            ),
            updated_at=now,
        )
        return self._write(checkpoint)

    @staticmethod
    def verify_compatibility(
        checkpoint: PipelineCheckpoint,
        *,
        manifest_sha256: str,
        manifest_schema_sha256: str,
        source_policy_sha256: str,
        config_sha256: str,
    ) -> None:
        comparisons = {
            "manifest": (checkpoint.manifest_sha256, manifest_sha256),
            "manifest_schema": (
                checkpoint.manifest_schema_sha256,
                manifest_schema_sha256,
            ),
            "source_policy": (checkpoint.source_policy_sha256, source_policy_sha256),
            "config": (checkpoint.config_sha256, config_sha256),
        }
        for name, (actual, expected) in comparisons.items():
            if actual != expected:
                raise CheckpointCompatibilityError(f"checkpoint_{name}_mismatch")

    def begin(self, target: PipelineState) -> PipelineCheckpoint:
        checkpoint = self.load()
        if checkpoint.state not in _STATE_POSITION:
            raise CheckpointError("checkpoint_resume_required")
        if checkpoint.state == PipelineState.COMPLETED:
            if target == PipelineState.COMPLETED:
                return checkpoint
            raise CheckpointError("checkpoint_already_completed")
        if checkpoint.active_state is not None:
            if checkpoint.active_state == target:
                return checkpoint
            raise CheckpointError("checkpoint_step_already_active")
        next_position = _STATE_POSITION[checkpoint.state] + 1
        if target != PIPELINE_SEQUENCE[next_position]:
            raise CheckpointError("checkpoint_transition_invalid")
        now = datetime.now(UTC)
        updated = checkpoint.model_copy(
            update={
                "active_state": target,
                "history": checkpoint.history
                + (
                    TransitionEvent(
                        operation="begin",
                        from_state=checkpoint.state,
                        to_state=target,
                        at=now,
                    ),
                ),
                "updated_at": now,
            }
        )
        return self._write(updated)

    def update_progress(self, progress: StepProgress) -> PipelineCheckpoint:
        checkpoint = self.load()
        if checkpoint.active_state != progress.state:
            raise CheckpointError("checkpoint_progress_state_mismatch")
        updated = checkpoint.model_copy(
            update={"progress": progress, "updated_at": datetime.now(UTC)}
        )
        return self._write(updated)

    def complete(
        self,
        target: PipelineState,
        *,
        artifact_sha256: Mapping[str, str] | None = None,
    ) -> PipelineCheckpoint:
        checkpoint = self.load()
        additions = dict(artifact_sha256 or {})
        if checkpoint.state == target and checkpoint.active_state is None:
            for key, digest in additions.items():
                if checkpoint.artifact_sha256.get(key) != digest:
                    raise CheckpointError("checkpoint_idempotent_artifact_mismatch")
            return checkpoint
        if checkpoint.active_state != target:
            raise CheckpointError("checkpoint_completion_without_active_step")
        merged_artifacts = dict(checkpoint.artifact_sha256)
        for key, digest in additions.items():
            if (
                _ARTIFACT_KEY.fullmatch(key) is None
                or ".." in key.split("/")
                or re.fullmatch(SHA256_PATTERN, digest) is None
            ):
                raise CheckpointError("checkpoint_artifact_invalid")
            existing = merged_artifacts.get(key)
            if existing is not None and existing != digest:
                raise CheckpointError("checkpoint_artifact_hash_mismatch")
            merged_artifacts[key] = digest
        now = datetime.now(UTC)
        updated = checkpoint.model_copy(
            update={
                "state": target,
                "last_successful_state": target,
                "active_state": None,
                "progress": None,
                "failure_code": None,
                "artifact_sha256": merged_artifacts,
                "history": checkpoint.history
                + (
                    TransitionEvent(
                        operation="complete",
                        from_state=checkpoint.state,
                        to_state=target,
                        at=now,
                    ),
                ),
                "updated_at": now,
            }
        )
        return self._write(updated)

    def fail(self, reason_code: str) -> PipelineCheckpoint:
        checkpoint = self.load()
        if checkpoint.state == PipelineState.COMPLETED:
            raise CheckpointError("completed_checkpoint_cannot_fail")
        if re.fullmatch(r"^[a-z][a-z0-9_]{0,79}$", reason_code) is None:
            reason_code = "pipeline_failed"
        now = datetime.now(UTC)
        updated = checkpoint.model_copy(
            update={
                "state": PipelineState.FAILED,
                "active_state": None,
                "progress": None,
                "failure_code": reason_code,
                "history": checkpoint.history
                + (
                    TransitionEvent(
                        operation="fail",
                        from_state=checkpoint.state,
                        to_state=PipelineState.FAILED,
                        at=now,
                    ),
                ),
                "updated_at": now,
            }
        )
        return self._write(updated)

    def cancel(self) -> PipelineCheckpoint:
        checkpoint = self.load()
        if checkpoint.state == PipelineState.COMPLETED:
            raise CheckpointError("completed_checkpoint_cannot_cancel")
        now = datetime.now(UTC)
        updated = checkpoint.model_copy(
            update={
                "state": PipelineState.CANCELLED,
                "active_state": None,
                "progress": None,
                "failure_code": None,
                "history": checkpoint.history
                + (
                    TransitionEvent(
                        operation="cancel",
                        from_state=checkpoint.state,
                        to_state=PipelineState.CANCELLED,
                        at=now,
                    ),
                ),
                "updated_at": now,
            }
        )
        return self._write(updated)

    def _write(self, checkpoint: PipelineCheckpoint) -> PipelineCheckpoint:
        atomic_write_model(self.path, checkpoint, require_idempotent=False)
        return checkpoint


def state_at_or_after(current: PipelineState, target: PipelineState) -> bool:
    if current not in _STATE_POSITION or target not in _STATE_POSITION:
        return False
    return _STATE_POSITION[current] >= _STATE_POSITION[target]


__all__ = [
    "CheckpointStore",
    "PIPELINE_SEQUENCE",
    "PipelineCheckpoint",
    "PipelineState",
    "StepProgress",
    "TransitionEvent",
    "state_at_or_after",
]
