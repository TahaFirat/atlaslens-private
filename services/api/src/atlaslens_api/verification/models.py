from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VerificationStatus = Literal["supported", "inconclusive", "contradicted", "unavailable", "failed"]


@dataclass(frozen=True, slots=True)
class VerificationLimits:
    max_dimension: int = 1280
    max_features: int = 1600
    ratio_threshold: float = 0.75
    ransac_threshold_px: float = 4.0
    min_keypoints: int = 20
    min_matches: int = 12
    min_inliers: int = 10
    min_inlier_ratio: float = 0.35
    min_coverage: float = 0.025
    min_descriptor_uniqueness: float = 0.3
    min_global_descriptor_uniqueness: float = 0.8
    max_median_residual_px: float = 4.0
    timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if self.max_dimension < 64 or self.max_features < 32 or self.timeout_ms < 0:
            raise ValueError("verification limits must be positive and bounded")
        for value in (
            self.ratio_threshold,
            self.min_inlier_ratio,
            self.min_coverage,
            self.min_descriptor_uniqueness,
            self.min_global_descriptor_uniqueness,
        ):
            if not 0 < value <= 1:
                raise ValueError("verification ratios must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class GeometricVerificationResult:
    status: VerificationStatus
    reason_code: str
    provider_id: str
    provider_version: str
    model: Literal["homography", "none"]
    query_keypoints: int
    reference_keypoints: int
    candidate_matches: int
    inliers: int
    inlier_ratio: float
    coverage: float
    median_residual_px: float | None
    resized_query: tuple[int, int]
    resized_reference: tuple[int, int]
    runtime_ms: int
    limits: VerificationLimits
