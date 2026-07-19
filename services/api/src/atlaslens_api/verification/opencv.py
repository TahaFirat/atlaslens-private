from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import GeometricVerificationResult, VerificationLimits, VerificationStatus
from .protocols import CancellationSignal, LocalFeatureProvider, LocalFeatureSet


class InvalidVerificationImage(ValueError):
    pass


class VerificationCancelled(RuntimeError):
    pass


class VerificationTimedOut(TimeoutError):
    pass


@dataclass(frozen=True, slots=True)
class _OrbFeatures:
    points: NDArray[np.float32]
    descriptors: NDArray[np.uint8]
    image_size: tuple[int, int]


class OrbLocalFeatureProvider:
    provider_id = "opencv_orb"
    provider_version = "phase4-v1"

    def extract(self, image_bytes: bytes, limits: VerificationLimits) -> LocalFeatureSet:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if image is None or image.ndim != 2 or image.size == 0:
            raise InvalidVerificationImage("image could not be decoded")
        height, width = image.shape
        scale = min(1.0, limits.max_dimension / max(width, height))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        orb = cv2.ORB.create(
            nfeatures=limits.max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=19,
            patchSize=31,
            fastThreshold=12,
        )
        mask = np.full(image.shape, 255, dtype=np.uint8)
        keypoints, descriptors = orb.detectAndCompute(image, mask)
        points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(
            -1, 2
        )
        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)
        return _OrbFeatures(
            points, np.asarray(descriptors, dtype=np.uint8), (image.shape[1], image.shape[0])
        )


class MutualRatioFeatureMatcher:
    def match(
        self, query: LocalFeatureSet, reference: LocalFeatureSet, limits: VerificationLimits
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        if len(query.descriptors) < 2 or len(reference.descriptors) < 2:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy(), 0.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        forward = matcher.knnMatch(query.descriptors, reference.descriptors, k=2)
        reverse = matcher.knnMatch(reference.descriptors, query.descriptors, k=2)

        def accepted(matches: Sequence[Sequence[cv2.DMatch]]) -> dict[int, int]:
            selected: dict[int, int] = {}
            for pair in matches:
                if len(pair) == 2 and pair[0].distance < limits.ratio_threshold * pair[1].distance:
                    selected[pair[0].queryIdx] = pair[0].trainIdx
            return selected

        forward_map = accepted(forward)
        reverse_map = accepted(reverse)
        pairs = sorted(
            (query_id, ref_id)
            for query_id, ref_id in forward_map.items()
            if reverse_map.get(ref_id) == query_id
        )
        if not pairs:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy(), 0.0
        query_ids = np.asarray([pair[0] for pair in pairs], dtype=np.intp)
        reference_ids = np.asarray([pair[1] for pair in pairs], dtype=np.intp)
        matched_descriptors = query.descriptors[query_ids]
        uniqueness = len(np.unique(matched_descriptors, axis=0)) / len(matched_descriptors)
        return query.points[query_ids], reference.points[reference_ids], float(uniqueness)


class OpenCvGeometricVerifier:
    provider_id = "opencv_orb_homography"
    provider_version = "phase4-v1"
    _opencv_lock = threading.Lock()

    def __init__(
        self,
        feature_provider: LocalFeatureProvider | None = None,
        matcher: MutualRatioFeatureMatcher | None = None,
    ) -> None:
        self._features = feature_provider or OrbLocalFeatureProvider()
        self._matcher = matcher or MutualRatioFeatureMatcher()

    def verify(
        self,
        query_image: bytes,
        reference_image: bytes,
        *,
        limits: VerificationLimits | None = None,
        cancellation: CancellationSignal | None = None,
    ) -> GeometricVerificationResult:
        policy = limits or VerificationLimits()
        started = time.perf_counter()
        query_size = (0, 0)
        reference_size = (0, 0)
        query_count = reference_count = matches = 0

        def check() -> None:
            if cancellation is not None and cancellation.is_cancelled():
                raise VerificationCancelled
            if (time.perf_counter() - started) * 1000 > policy.timeout_ms:
                raise VerificationTimedOut

        def result(
            status: VerificationStatus,
            reason: str,
            *,
            model: Literal["homography", "none"] = "none",
            inliers: int = 0,
            ratio: float = 0.0,
            coverage: float = 0.0,
            residual: float | None = None,
        ) -> GeometricVerificationResult:
            return GeometricVerificationResult(
                status=status,
                reason_code=reason,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                model=model,
                query_keypoints=query_count,
                reference_keypoints=reference_count,
                candidate_matches=matches,
                inliers=inliers,
                inlier_ratio=round(ratio, 6),
                coverage=round(coverage, 6),
                median_residual_px=None if residual is None else round(residual, 4),
                resized_query=query_size,
                resized_reference=reference_size,
                runtime_ms=max(0, round((time.perf_counter() - started) * 1000)),
                limits=policy,
            )

        try:
            check()
            with self._opencv_lock:
                cv2.setRNGSeed(0)
                query = self._features.extract(query_image, policy)
                check()
                reference = self._features.extract(reference_image, policy)
                check()
                query_size, reference_size = query.image_size, reference.image_size
                query_count, reference_count = len(query.points), len(reference.points)
                if min(query_count, reference_count) < policy.min_keypoints:
                    return result("inconclusive", "insufficient_local_features")
                query_points, reference_points, uniqueness = self._matcher.match(
                    query, reference, policy
                )
                matches = len(query_points)
                check()
                if matches < policy.min_matches:
                    return result("contradicted", "insufficient_mutual_matches")
                if uniqueness < policy.min_descriptor_uniqueness:
                    return result("inconclusive", "repeated_pattern_ambiguity")
                if (
                    min(
                        _descriptor_uniqueness(query.descriptors),
                        _descriptor_uniqueness(reference.descriptors),
                    )
                    < policy.min_global_descriptor_uniqueness
                ):
                    return result("inconclusive", "repeated_pattern_ambiguity")
                if _regular_spatial_pattern(query_points) or _regular_spatial_pattern(
                    reference_points
                ):
                    return result("inconclusive", "repeated_pattern_ambiguity")
                homography, mask = cv2.findHomography(
                    query_points,
                    reference_points,
                    cv2.RANSAC,
                    policy.ransac_threshold_px,
                    maxIters=2000,
                    confidence=0.995,
                )
                check()
            if homography is None or mask is None or not np.isfinite(homography).all():
                return result("contradicted", "geometric_model_not_found")
            homography_matrix: NDArray[np.float64] = np.asarray(homography, dtype=np.float64)
            determinant = float(np.linalg.det(homography_matrix[:2, :2]))
            if not math.isfinite(determinant) or abs(determinant) < 1e-6:
                return result("inconclusive", "degenerate_geometric_model")
            inlier_mask = mask.ravel().astype(bool)
            inlier_count = int(inlier_mask.sum())
            inlier_ratio = inlier_count / matches
            if inlier_count < 4:
                return result(
                    "contradicted",
                    "too_few_geometric_inliers",
                    inliers=inlier_count,
                    ratio=inlier_ratio,
                )
            query_inliers = query_points[inlier_mask]
            reference_inliers = reference_points[inlier_mask]
            projected = cv2.perspectiveTransform(
                query_inliers.reshape(-1, 1, 2), homography_matrix
            ).reshape(-1, 2)
            residual = float(np.median(np.linalg.norm(projected - reference_inliers, axis=1)))
            coverage = min(
                _point_coverage(query_inliers, query_size),
                _point_coverage(reference_inliers, reference_size),
            )
            if (
                inlier_count >= policy.min_inliers
                and inlier_ratio >= policy.min_inlier_ratio
                and coverage >= policy.min_coverage
                and residual <= policy.max_median_residual_px
            ):
                return result(
                    "supported",
                    "geometric_consistency_supported",
                    model="homography",
                    inliers=inlier_count,
                    ratio=inlier_ratio,
                    coverage=coverage,
                    residual=residual,
                )
            return result(
                "contradicted",
                "geometric_consistency_below_threshold",
                model="homography",
                inliers=inlier_count,
                ratio=inlier_ratio,
                coverage=coverage,
                residual=residual,
            )
        except VerificationCancelled:
            return result("failed", "cancelled")
        except VerificationTimedOut:
            return result("failed", "timeout")
        except InvalidVerificationImage:
            return result("failed", "invalid_image")
        except cv2.error:
            return result("failed", "opencv_failure")


class UnavailableGeometricVerifier:
    provider_id = "disabled_geometry"
    provider_version = "phase4-v1"

    def verify(
        self,
        query_image: bytes,
        reference_image: bytes,
        *,
        limits: VerificationLimits | None = None,
        cancellation: CancellationSignal | None = None,
    ) -> GeometricVerificationResult:
        del query_image, reference_image, cancellation
        policy = limits or VerificationLimits()
        return GeometricVerificationResult(
            status="unavailable",
            reason_code="provider_unavailable",
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            model="none",
            query_keypoints=0,
            reference_keypoints=0,
            candidate_matches=0,
            inliers=0,
            inlier_ratio=0.0,
            coverage=0.0,
            median_residual_px=None,
            resized_query=(0, 0),
            resized_reference=(0, 0),
            runtime_ms=0,
            limits=policy,
        )


def _point_coverage(points: NDArray[np.float32], size: tuple[int, int]) -> float:
    width, height = size
    if len(points) < 2 or width <= 0 or height <= 0:
        return 0.0
    extent = np.ptp(points, axis=0)
    return float(np.clip((extent[0] * extent[1]) / (width * height), 0.0, 1.0))


def _regular_spatial_pattern(points: NDArray[np.float32]) -> bool:
    if len(points) < 30:
        return False
    sample = points[: min(len(points), 400)]
    differences = sample[:, None, :] - sample[None, :, :]
    squared = np.sum(differences * differences, axis=2)
    np.fill_diagonal(squared, np.inf)
    nearest = np.sqrt(np.min(squared, axis=1))
    mean = float(np.mean(nearest))
    return mean > 0 and float(np.std(nearest) / mean) < 0.18


def _descriptor_uniqueness(descriptors: NDArray[np.uint8]) -> float:
    if len(descriptors) == 0:
        return 0.0
    return len(np.unique(descriptors, axis=0)) / len(descriptors)
