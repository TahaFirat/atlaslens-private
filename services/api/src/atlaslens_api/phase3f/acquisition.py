"""Hard acquisition caps and provenance admission for the Phase 3F cloud job."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final
from urllib.parse import parse_qsl, urlsplit

MAX_IMAGES: Final = 4_000
MAX_MEDIA_BYTES: Final = 8 * 1024 * 1024 * 1024
MAX_REQUESTS: Final = 6_000
MAX_CONCURRENCY: Final = 2
MAX_RENDITION_EDGE: Final = 1_024
MAX_RETRY_ATTEMPTS: Final = 3


class AcquisitionRefused(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ProvenanceSidecar:
    mapillary_image_id: str
    source_page: str
    contributor_attribution: str
    capture_date: datetime
    source_policy_receipt_sha256: str
    revoked: bool = False

    def validate(self) -> None:
        if not self.mapillary_image_id or len(self.mapillary_image_id) > 128:
            raise AcquisitionRefused("MAPILLARY_PROVENANCE_INVALID")
        parsed = urlsplit(self.source_page)
        if parsed.scheme != "https" or parsed.hostname not in {
            "www.mapillary.com",
            "mapillary.com",
        }:
            raise AcquisitionRefused("MAPILLARY_SOURCE_PAGE_INVALID")
        sensitive = {
            "access_token",
            "token",
            "signature",
            "sig",
            "expires",
            "authorization",
        }
        if any(key.casefold() in sensitive for key, _ in parse_qsl(parsed.query)):
            raise AcquisitionRefused("MAPILLARY_EPHEMERAL_URL_REFUSED")
        if not self.contributor_attribution.strip():
            raise AcquisitionRefused("MAPILLARY_ATTRIBUTION_MISSING")
        if self.capture_date.tzinfo is None or self.capture_date.utcoffset() is None:
            raise AcquisitionRefused("MAPILLARY_CAPTURE_DATE_INVALID")
        if len(self.source_policy_receipt_sha256) != 64:
            raise AcquisitionRefused("MAPILLARY_SOURCE_POLICY_INVALID")
        if self.revoked:
            raise AcquisitionRefused("MAPILLARY_PROVENANCE_REVOKED")


@dataclass(frozen=True, slots=True)
class PrivacyReview:
    passed: bool
    face_reidentification_performed: bool = False
    plate_reidentification_performed: bool = False
    raw_ocr_generated: bool = False

    def validate(self) -> None:
        if (
            not self.passed
            or self.face_reidentification_performed
            or self.plate_reidentification_performed
            or self.raw_ocr_generated
        ):
            raise AcquisitionRefused("MAPILLARY_PRIVACY_REVIEW_FAILED")


@dataclass(slots=True)
class AcquisitionGuard:
    request_count: int = 0
    image_count: int = 0
    media_bytes: int = 0
    in_flight: int = 0
    stopped: bool = False

    def begin_request(self) -> None:
        if self.stopped:
            raise AcquisitionRefused("MAPILLARY_ACQUISITION_STOPPED")
        if self.in_flight >= MAX_CONCURRENCY:
            raise AcquisitionRefused("MAPILLARY_CONCURRENCY_CAP_REACHED")
        if self.request_count >= MAX_REQUESTS:
            self.stopped = True
            raise AcquisitionRefused("MAPILLARY_REQUEST_CAP_REACHED")
        self.request_count += 1
        self.in_flight += 1

    def finish_request(
        self,
        *,
        status_code: int,
        media_bytes: int = 0,
        admitted_image: bool = False,
        provenance: ProvenanceSidecar | None = None,
        privacy_review: PrivacyReview | None = None,
    ) -> None:
        if self.in_flight <= 0:
            raise AcquisitionRefused("MAPILLARY_REQUEST_ACCOUNTING_INVALID")
        self.in_flight -= 1
        if status_code in {401, 403}:
            self.stopped = True
            raise AcquisitionRefused("MAPILLARY_TOKEN_REJECTED")
        if media_bytes < 0:
            raise AcquisitionRefused("MAPILLARY_MEDIA_BYTES_INVALID")
        if not admitted_image:
            return
        if provenance is None or privacy_review is None:
            raise AcquisitionRefused("MAPILLARY_ADMISSION_EVIDENCE_MISSING")
        provenance.validate()
        privacy_review.validate()
        if self.image_count >= MAX_IMAGES:
            self.stopped = True
            raise AcquisitionRefused("MAPILLARY_IMAGE_CAP_REACHED")
        if self.media_bytes + media_bytes > MAX_MEDIA_BYTES:
            self.stopped = True
            raise AcquisitionRefused("MAPILLARY_MEDIA_BYTE_CAP_REACHED")
        self.image_count += 1
        self.media_bytes += media_bytes

    def should_retry(self, *, status_code: int, completed_attempts: int) -> bool:
        if status_code in {401, 403}:
            self.stopped = True
            raise AcquisitionRefused("MAPILLARY_TOKEN_REJECTED")
        if completed_attempts < 0:
            raise ValueError("completed_attempts must be non-negative")
        transient = status_code == 429 or 500 <= status_code <= 599
        return bool(
            transient
            and completed_attempts < MAX_RETRY_ATTEMPTS
            and self.request_count < MAX_REQUESTS
            and not self.stopped
        )

    def stop_new_acquisition(self) -> None:
        self.stopped = True

    def receipt(self) -> dict[str, object]:
        return {
            "schema": "atlaslens-phase3f-acquisition-guard-v1",
            "request_count": self.request_count,
            "image_count": self.image_count,
            "media_bytes": self.media_bytes,
            "in_flight": self.in_flight,
            "stopped": self.stopped,
            "caps": {
                "requests": MAX_REQUESTS,
                "images": MAX_IMAGES,
                "media_bytes": MAX_MEDIA_BYTES,
                "concurrency": MAX_CONCURRENCY,
                "rendition_max_edge": MAX_RENDITION_EDGE,
                "retry_attempts": MAX_RETRY_ATTEMPTS,
            },
            "raw_ocr_generated": False,
            "reidentification_performed": False,
            "signed_media_urls_retained": False,
        }


__all__ = [
    "AcquisitionGuard",
    "AcquisitionRefused",
    "PrivacyReview",
    "ProvenanceSidecar",
]
