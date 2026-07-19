from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CalibrationCompatibilityError(ValueError):
    pass


_CALIBRATED_MINIMUM_SPLIT_COUNT = 50
_CALIBRATED_REQUIRED_METRICS = frozenset(
    {
        "fit_event_rate",
        "validation_event_rate",
        "test_event_rate",
        "validation_brier",
        "validation_ece",
        "test_brier",
        "test_ece",
    }
)


class CalibrationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    artifact_version: Literal["phase5-calibration-v1"] = "phase5-calibration-v1"
    calibration_state: Literal["preliminary", "calibrated"]
    provider_id: str = Field(min_length=1, max_length=80)
    model_revision: str = Field(min_length=1, max_length=160)
    event: Literal["country_correct", "within_25_km", "within_200_km", "within_750_km"]
    method: Literal["logistic"]
    feature_names: tuple[str, ...] = Field(min_length=1, max_length=16)
    coefficients: tuple[float, ...] = Field(min_length=1, max_length=16)
    intercept: float
    fit_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_count: int = Field(gt=0)
    validation_count: int = Field(gt=0)
    test_count: int = Field(gt=0)
    created_at: datetime
    metrics: dict[str, float]
    applicability_limits: tuple[str, ...] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def validate_artifact(self) -> CalibrationArtifact:
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("calibration features must be unique")
        if len(self.feature_names) != len(self.coefficients):
            raise ValueError("calibration coefficient count mismatch")
        if len({self.fit_fingerprint, self.validation_fingerprint, self.test_fingerprint}) != 3:
            raise ValueError("calibration splits must have distinct fingerprints")
        values = (*self.coefficients, self.intercept, *self.metrics.values())
        if not all(math.isfinite(value) for value in values):
            raise ValueError("calibration artifact contains non-finite values")
        if self.calibration_state == "calibrated":
            if min(self.fit_count, self.validation_count, self.test_count) < (
                _CALIBRATED_MINIMUM_SPLIT_COUNT
            ):
                raise ValueError("calibrated artifact split is too small")
            if not set(self.metrics) >= _CALIBRATED_REQUIRED_METRICS:
                raise ValueError("calibrated artifact promotion metrics are incomplete")
            if any(
                not 0 <= self.metrics[name] <= 1
                for name in _CALIBRATED_REQUIRED_METRICS
            ):
                raise ValueError("calibrated artifact promotion metrics are invalid")
            if any(
                not 0 < self.metrics[name] < 1
                for name in (
                    "fit_event_rate",
                    "validation_event_rate",
                    "test_event_rate",
                )
            ):
                raise ValueError("calibrated artifact lacks positive and negative classes")
        return self

    @classmethod
    def from_path(cls, path: Path) -> CalibrationArtifact:
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CalibrationCompatibilityError("calibration artifact is invalid") from exc

    def assert_compatible(
        self,
        *,
        provider_id: str,
        model_revision: str,
        feature_names: tuple[str, ...],
        fit_fingerprint: str,
        validation_fingerprint: str,
        test_fingerprint: str,
        fit_count: int,
        validation_count: int,
        test_count: int,
    ) -> None:
        if (
            self.provider_id != provider_id
            or self.model_revision != model_revision
            or self.feature_names != feature_names
            or self.fit_fingerprint != fit_fingerprint
            or self.validation_fingerprint != validation_fingerprint
            or self.test_fingerprint != test_fingerprint
            or self.fit_count != fit_count
            or self.validation_count != validation_count
            or self.test_count != test_count
        ):
            raise CalibrationCompatibilityError("calibration artifact is incompatible")


class ScoreCalibrator(Protocol):
    calibration_state: Literal["uncalibrated", "preliminary", "calibrated"]

    def probability(self, features: dict[str, float]) -> float | None: ...


class UncalibratedCalibrator:
    calibration_state: Literal["uncalibrated"] = "uncalibrated"

    def probability(self, features: dict[str, float]) -> None:
        del features
        return None


class LogisticArtifactCalibrator:
    def __init__(
        self,
        artifact: CalibrationArtifact,
        *,
        provider_id: str,
        model_revision: str,
        feature_names: tuple[str, ...],
        fit_fingerprint: str,
        validation_fingerprint: str,
        test_fingerprint: str,
        fit_count: int,
        validation_count: int,
        test_count: int,
    ) -> None:
        artifact.assert_compatible(
            provider_id=provider_id,
            model_revision=model_revision,
            feature_names=feature_names,
            fit_fingerprint=fit_fingerprint,
            validation_fingerprint=validation_fingerprint,
            test_fingerprint=test_fingerprint,
            fit_count=fit_count,
            validation_count=validation_count,
            test_count=test_count,
        )
        self.artifact = artifact
        self.calibration_state = artifact.calibration_state

    def probability(self, features: dict[str, float]) -> float | None:
        if self.calibration_state != "calibrated":
            return None
        if set(features) != set(self.artifact.feature_names):
            raise CalibrationCompatibilityError("calibration features are incompatible")
        values = [features[name] for name in self.artifact.feature_names]
        if not all(math.isfinite(value) for value in values):
            raise CalibrationCompatibilityError("calibration features are invalid")
        linear = self.artifact.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.artifact.coefficients, values, strict=True)
        )
        if linear >= 0:
            return 1 / (1 + math.exp(-linear))
        exp_value = math.exp(linear)
        return exp_value / (1 + exp_value)
