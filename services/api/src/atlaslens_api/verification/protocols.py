from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .models import GeometricVerificationResult, VerificationLimits


class CancellationSignal(Protocol):
    def is_cancelled(self) -> bool: ...


class LocalFeatureSet(Protocol):
    @property
    def points(self) -> NDArray[np.float32]: ...

    @property
    def descriptors(self) -> NDArray[np.uint8]: ...

    @property
    def image_size(self) -> tuple[int, int]: ...


class LocalFeatureProvider(Protocol):
    provider_id: str
    provider_version: str

    def extract(self, image_bytes: bytes, limits: VerificationLimits) -> LocalFeatureSet: ...


class FeatureMatcher(Protocol):
    def match(
        self, query: LocalFeatureSet, reference: LocalFeatureSet, limits: VerificationLimits
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]: ...


class GeometricVerifier(Protocol):
    def verify(
        self,
        query_image: bytes,
        reference_image: bytes,
        *,
        limits: VerificationLimits | None = None,
        cancellation: CancellationSignal | None = None,
    ) -> GeometricVerificationResult: ...
