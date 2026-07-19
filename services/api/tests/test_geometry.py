from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

import cv2
import numpy as np
import pytest

from atlaslens_api.verification import (
    OpenCvGeometricVerifier,
    UnavailableGeometricVerifier,
    VerificationLimits,
)


def _scene(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((480, 640, 3), 28, dtype=np.uint8)
    for index in range(45):
        center = tuple(int(value) for value in rng.integers([25, 25], [615, 455]))
        color = tuple(int(value) for value in rng.integers(70, 245, size=3))
        radius = int(rng.integers(5, 22))
        cv2.circle(image, center, radius, color, 2)
        cv2.putText(
            image, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )
    cv2.rectangle(image, (55, 65), (270, 205), (235, 215, 85), 4)
    cv2.line(image, (20, 430), (610, 250), (80, 230, 220), 5)
    return image


def _encoded(image: np.ndarray) -> bytes:
    ok, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return payload.tobytes()


def _translation(image: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(image, np.float32([[1, 0, 24], [0, 1, 17]]), (640, 480))


def _rotation(image: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(image, cv2.getRotationMatrix2D((320, 240), 8, 0.96), (640, 480))


def _perspective(image: np.ndarray) -> np.ndarray:
    source = np.float32([[0, 0], [639, 0], [639, 479], [0, 479]])
    target = np.float32([[25, 18], [610, 5], [628, 455], [12, 470]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (640, 480))


def _crop(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image[55:430, 70:585], (640, 480), interpolation=cv2.INTER_LINEAR)


@pytest.mark.parametrize(
    "transform",
    [lambda image: image.copy(), _translation, _rotation, _perspective, _crop],
    ids=["identical", "translation", "rotation", "perspective", "crop"],
)
def test_geometry_supports_generated_consistent_views(
    transform: Callable[[np.ndarray], np.ndarray],
) -> None:
    source = _scene()
    result = OpenCvGeometricVerifier().verify(_encoded(transform(source)), _encoded(source))
    assert result.status == "supported", result
    assert result.model == "homography"
    assert result.inliers >= result.limits.min_inliers
    assert result.inlier_ratio >= result.limits.min_inlier_ratio
    assert result.coverage > 0


def test_brightness_change_remains_supported() -> None:
    source = _scene()
    brighter = cv2.convertScaleAbs(source, alpha=1.12, beta=24)
    assert (
        OpenCvGeometricVerifier().verify(_encoded(brighter), _encoded(source)).status == "supported"
    )


def test_blur_is_bounded_and_never_fabricates_metrics() -> None:
    source = _scene()
    blurred = cv2.GaussianBlur(source, (17, 17), 5)
    result = OpenCvGeometricVerifier().verify(_encoded(blurred), _encoded(source))
    assert result.status in {"supported", "inconclusive", "contradicted"}
    assert 0 <= result.inlier_ratio <= 1
    assert 0 <= result.coverage <= 1


def test_unrelated_images_are_not_supported() -> None:
    result = OpenCvGeometricVerifier().verify(_encoded(_scene(99)), _encoded(_scene(7)))
    assert result.status in {"inconclusive", "contradicted"}


def test_low_texture_is_inconclusive() -> None:
    flat = np.full((360, 480, 3), 127, dtype=np.uint8)
    result = OpenCvGeometricVerifier().verify(_encoded(flat), _encoded(flat))
    assert (result.status, result.reason_code) == ("inconclusive", "insufficient_local_features")


def test_repeated_pattern_does_not_claim_support() -> None:
    tile = np.indices((480, 640)).sum(axis=0) // 24 % 2
    checker = np.repeat((tile * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    result = OpenCvGeometricVerifier().verify(
        _encoded(np.roll(checker, 48, axis=1)), _encoded(checker)
    )
    assert result.status != "supported", result


def test_malformed_images_fail_safely() -> None:
    result = OpenCvGeometricVerifier().verify(b"not-an-image", b"also-invalid")
    assert (result.status, result.reason_code) == ("failed", "invalid_image")


class _Cancelled:
    def is_cancelled(self) -> bool:
        return True


def test_cancellation_is_reported_without_exception() -> None:
    payload = _encoded(_scene())
    result = OpenCvGeometricVerifier().verify(payload, payload, cancellation=_Cancelled())
    assert (result.status, result.reason_code) == ("failed", "cancelled")


def test_timeout_is_bounded() -> None:
    payload = _encoded(_scene())
    result = OpenCvGeometricVerifier().verify(
        payload, payload, limits=VerificationLimits(timeout_ms=0)
    )
    assert (result.status, result.reason_code) == ("failed", "timeout")


def test_unavailable_provider_is_explicit() -> None:
    result = UnavailableGeometricVerifier().verify(b"", b"")
    assert (result.status, result.reason_code) == ("unavailable", "provider_unavailable")


def test_result_contains_diagnostics_but_no_descriptors_or_paths() -> None:
    payload = _encoded(_scene())
    result = OpenCvGeometricVerifier().verify(
        payload, payload, limits=VerificationLimits(max_dimension=320, max_features=300)
    )
    serialized = asdict(result)
    assert result.resized_query[0] <= 320 and result.resized_query[1] <= 320
    assert result.query_keypoints <= 300
    assert "descriptors" not in repr(serialized).lower()
    assert "path" not in repr(serialized).lower()
