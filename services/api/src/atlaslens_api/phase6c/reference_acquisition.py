from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from atlaslens_api.gazetteer import GazetteerResolver
from atlaslens_api.phase6c.catalogue import CoordinateCatalogue, load_coordinate_catalogue
from atlaslens_api.phase6c.reference_index import (
    REFERENCE_INPUT_SCHEMA_VERSION,
    ReferenceBuildInput,
    ReferenceBuildRecord,
)

ACQUISITION_PLAN_SCHEMA_VERSION: Final = "atlaslens-reference-acquisition-plan-v1"
ACQUISITION_RECEIPT_SCHEMA_VERSION: Final = "atlaslens-reference-acquisition-receipt-v1"
KARTAVIEW_REVIEWED_TERMS_VERSION: Final = "2025-06-17"
MAPILLARY_LICENSE: Final = "CC BY-SA 4.0"
KARTAVIEW_LICENSE: Final = "CC BY-SA 4.0"
CC_BY_SA_4_URL: Final = "https://creativecommons.org/licenses/by-sa/4.0/"
DEFAULT_MAX_IMAGES: Final = 2_025
DEFAULT_MAX_PER_PROVINCE: Final = 25
DEFAULT_MAX_PER_SEQUENCE: Final = 4
DEFAULT_MAX_DISK_BYTES: Final = 2 * 1024 * 1024 * 1024
DEFAULT_ESTIMATED_IMAGE_BYTES: Final = 512 * 1024
LARGE_INDEX_IMAGE_THRESHOLD: Final = 5_000

_OPAQUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DHASH = re.compile(r"^[0-9a-f]{16}$")
_TOKEN_QUERY_KEYS = frozenset(
    {"access_token", "authorization", "client_token", "secret", "token"}
)
_SUPPORTED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}

SourceName = Literal["mapillary", "kartaview"]
Urbanicity = Literal["urban", "rural", "unknown"]
Season = Literal["winter", "spring", "summer", "autumn", "unknown"]


class ReferenceAcquisitionError(RuntimeError):
    """Operator-safe acquisition failure with no secret, URL, or coordinate detail."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class MapillaryBackendSettings(BaseSettings):
    """The only supported Mapillary secret boundary is the backend environment."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        case_sensitive=True,
    )

    enabled: bool = Field(default=False, validation_alias="MAPILLARY_ENABLED")
    access_token: SecretStr | None = Field(
        default=None,
        validation_alias="MAPILLARY_ACCESS_TOKEN",
        repr=False,
    )
    max_images: int = Field(
        default=DEFAULT_MAX_IMAGES,
        ge=81,
        le=8_100,
        validation_alias="MAPILLARY_MAX_IMAGES",
    )
    image_width: int = Field(
        default=1024,
        validation_alias="MAPILLARY_IMAGE_WIDTH",
    )

    @field_validator("image_width")
    @classmethod
    def supported_image_width(cls, value: int) -> int:
        if value not in {256, 1024, 2048}:
            raise ValueError("unsupported Mapillary image width")
        return value

    @model_validator(mode="after")
    def require_enabled_token(self) -> MapillaryBackendSettings:
        if self.enabled and (
            self.access_token is None or not self.access_token.get_secret_value().strip()
        ):
            raise ValueError("mapillary_token_required")
        return self


def load_mapillary_backend_settings(env_file: Path) -> MapillaryBackendSettings:
    selected = env_file.expanduser().resolve()
    if selected.is_symlink():
        raise ReferenceAcquisitionError("backend_env_invalid")
    try:
        return MapillaryBackendSettings(_env_file=selected if selected.is_file() else None)
    except ValueError as exc:
        raise ReferenceAcquisitionError("mapillary_configuration_invalid") from exc


class AcquisitionPolicy(_FrozenModel):
    max_images: int = Field(default=DEFAULT_MAX_IMAGES, ge=81, le=8_100)
    max_per_province: int = Field(default=DEFAULT_MAX_PER_PROVINCE, ge=1, le=100)
    max_per_sequence: int = Field(default=DEFAULT_MAX_PER_SEQUENCE, ge=1, le=10)
    discovery_oversample: int = Field(default=4, ge=1, le=10)
    max_disk_bytes: int = Field(default=DEFAULT_MAX_DISK_BYTES, gt=0)
    estimated_image_bytes: int = Field(default=DEFAULT_ESTIMATED_IMAGE_BYTES, gt=0)
    max_image_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=16_000_000, gt=0)
    mapillary_bbox_half_span_degrees: float = Field(default=0.08, gt=0, le=0.25)

    @model_validator(mode="after")
    def validate_total_bound(self) -> AcquisitionPolicy:
        if self.max_images > 81 * self.max_per_province:
            raise ValueError("max_images exceeds the province-balanced capacity")
        return self


class ProvinceAcquisitionTarget(_FrozenModel):
    admin1_code: str = Field(pattern=r"^[0-9]{1,3}$")
    province: str = Field(min_length=1, max_length=160)
    center_latitude: float = Field(ge=-90, le=90)
    center_longitude: float = Field(ge=-180, le=180)
    requested_images: int = Field(ge=1, le=100)
    urbanicity_strata: tuple[Urbanicity, ...] = ("urban", "rural", "unknown")
    heading_bins: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    season_strata: tuple[Season, ...] = (
        "winter",
        "spring",
        "summer",
        "autumn",
        "unknown",
    )


class ReferenceAcquisitionPlan(_FrozenModel):
    schema_version: Literal["atlaslens-reference-acquisition-plan-v1"]
    plan_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,159}$")
    catalogue_version: str = Field(min_length=1, max_length=180)
    catalogue_sha256: str = Field(pattern=_SHA256.pattern)
    policy: AcquisitionPolicy
    provinces: tuple[ProvinceAcquisitionTarget, ...]

    @model_validator(mode="after")
    def require_complete_balanced_province_plan(self) -> ReferenceAcquisitionPlan:
        if len(self.provinces) != 81:
            raise ValueError("the acquisition plan must represent all 81 provinces")
        if len({item.admin1_code for item in self.provinces}) != 81:
            raise ValueError("province admin codes must be unique")
        if len({item.province.casefold() for item in self.provinces}) != 81:
            raise ValueError("province names must be unique")
        total = sum(item.requested_images for item in self.provinces)
        if total > self.policy.max_images:
            raise ValueError("province plan exceeds the global image limit")
        if any(item.requested_images > self.policy.max_per_province for item in self.provinces):
            raise ValueError("province plan exceeds its per-province limit")
        return self

    @property
    def estimated_image_count(self) -> int:
        return sum(item.requested_images for item in self.provinces)

    @property
    def estimated_storage_bytes(self) -> int:
        return self.estimated_image_count * self.policy.estimated_image_bytes

    @property
    def requires_large_confirmation(self) -> bool:
        return self.estimated_image_count > LARGE_INDEX_IMAGE_THRESHOLD

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def build_acquisition_plan(
    *,
    policy: AcquisitionPolicy | None = None,
    catalogue: CoordinateCatalogue | None = None,
    plan_version: str = "turkiye-balanced-v1",
) -> ReferenceAcquisitionPlan:
    selected_policy = policy or AcquisitionPolicy()
    selected_catalogue = catalogue or load_coordinate_catalogue()
    provinces = sorted(
        selected_catalogue.records_of_kind("turkiye_province"),
        key=lambda item: (int(item.admin1_code or "0"), item.name.casefold()),
    )
    remaining = selected_policy.max_images
    quotas: list[int] = []
    for index in range(len(provinces)):
        slots_left = len(provinces) - index
        fair_share = max(1, remaining // slots_left)
        quota = min(selected_policy.max_per_province, fair_share)
        quotas.append(quota)
        remaining -= quota
    targets = tuple(
        ProvinceAcquisitionTarget(
            admin1_code=cast(str, record.admin1_code),
            province=record.name,
            center_latitude=record.latitude,
            center_longitude=record.longitude,
            requested_images=quota,
        )
        for record, quota in zip(provinces, quotas, strict=True)
    )
    plan = ReferenceAcquisitionPlan(
        schema_version=ACQUISITION_PLAN_SCHEMA_VERSION,
        plan_version=plan_version,
        catalogue_version=selected_catalogue.catalogue_version,
        catalogue_sha256=selected_catalogue.digest,
        policy=selected_policy,
        provinces=targets,
    )
    if plan.estimated_storage_bytes > selected_policy.max_disk_bytes:
        raise ReferenceAcquisitionError("estimated_storage_exceeds_disk_limit")
    return plan


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str


UrlValidator = Callable[[str], None]


class HttpTransport(Protocol):
    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_bytes: int,
        validate_url: UrlValidator,
    ) -> HttpResponse: ...


class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, validate_url: UrlValidator) -> None:
        self._validate_url = validate_url
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        self._validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibHttpTransport:
    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_bytes: int,
        validate_url: UrlValidator,
    ) -> HttpResponse:
        validate_url(url)
        request = urllib.request.Request(url, headers=dict(headers))  # noqa: S310
        opener = urllib.request.build_opener(_CheckedRedirectHandler(validate_url))
        try:
            with opener.open(request, timeout=timeout_seconds) as response:  # noqa: S310
                final_url = str(response.geturl())
                validate_url(final_url)
                body = cast(bytes, response.read(max_bytes + 1))
                status = int(response.status)
                response_headers = {str(key): str(value) for key, value in response.headers.items()}
        except urllib.error.HTTPError as exc:
            final_url = str(exc.geturl() or url)
            validate_url(final_url)
            body = exc.read(max_bytes + 1)
            status = int(exc.code)
            response_headers = {str(key): str(value) for key, value in exc.headers.items()}
        except (OSError, ValueError) as exc:
            raise ReferenceAcquisitionError("network_request_failed") from exc
        if len(body) > max_bytes:
            raise ReferenceAcquisitionError("network_response_too_large")
        return HttpResponse(status, response_headers, body, final_url)


def _host_matches(hostname: str, allowed: frozenset[str]) -> bool:
    host = hostname.casefold().rstrip(".")
    return any(host == item or host.endswith(f".{item}") for item in allowed)


def _url_validator(allowed_hosts: frozenset[str]) -> UrlValidator:
    def validate(url: str) -> None:
        parsed = urllib.parse.urlsplit(url)
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query_keys = {key.casefold() for key, _ in query_pairs}
        if (
            parsed.scheme.casefold() != "https"
            or parsed.hostname is None
            or not _host_matches(parsed.hostname, allowed_hosts)
            or parsed.port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or query_keys.intersection(_TOKEN_QUERY_KEYS)
        ):
            raise ReferenceAcquisitionError("unapproved_remote_url")

    return validate


class RetryingHttpClient:
    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        minimum_interval_seconds: float = 0.25,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            minimum_interval_seconds < 0
            or timeout_seconds <= 0
            or max_retries < 1
            or max_retries > 6
        ):
            raise ValueError("invalid HTTP client bounds")
        self._transport = transport or UrllibHttpTransport()
        self._minimum_interval = minimum_interval_seconds
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at: float | None = None
        self._json_cache: dict[str, dict[str, Any]] = {}

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        allowed_hosts: frozenset[str],
        max_bytes: int = 4 * 1024 * 1024,
    ) -> dict[str, Any]:
        cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cached = self._json_cache.get(cache_key)
        if cached is not None:
            return cached
        response = self._request(
            url,
            headers=headers,
            allowed_hosts=allowed_hosts,
            max_bytes=max_bytes,
        )
        try:
            value = json.loads(response.body)
        except (UnicodeError, ValueError) as exc:
            raise ReferenceAcquisitionError("remote_json_invalid") from exc
        if not isinstance(value, dict):
            raise ReferenceAcquisitionError("remote_json_invalid")
        typed = cast(dict[str, Any], value)
        self._json_cache[cache_key] = typed
        return typed

    def get_bytes(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        allowed_hosts: frozenset[str],
        max_bytes: int,
    ) -> bytes:
        return self._request(
            url,
            headers=headers,
            allowed_hosts=allowed_hosts,
            max_bytes=max_bytes,
        ).body

    def _request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        allowed_hosts: frozenset[str],
        max_bytes: int,
    ) -> HttpResponse:
        validate_url = _url_validator(allowed_hosts)
        validate_url(url)
        for attempt in range(self._max_retries):
            if self._last_request_at is not None:
                remaining = self._minimum_interval - (
                    self._monotonic() - self._last_request_at
                )
                if remaining > 0:
                    self._sleep(remaining)
            try:
                response = self._transport.request(
                    url,
                    headers=headers,
                    timeout_seconds=self._timeout,
                    max_bytes=max_bytes,
                    validate_url=validate_url,
                )
            except ReferenceAcquisitionError as exc:
                self._last_request_at = self._monotonic()
                if str(exc) != "network_request_failed" or attempt + 1 >= self._max_retries:
                    raise
                self._sleep(min(4.0, 0.25 * (2**attempt)))
                continue
            self._last_request_at = self._monotonic()
            if response.status_code == 200:
                return response
            if response.status_code != 429 and not 500 <= response.status_code <= 599:
                raise ReferenceAcquisitionError("remote_request_rejected")
            if attempt + 1 >= self._max_retries:
                break
            retry_after = _bounded_retry_after(response.headers.get("Retry-After"))
            self._sleep(retry_after if retry_after is not None else min(4.0, 0.25 * (2**attempt)))
        raise ReferenceAcquisitionError("remote_retry_exhausted")


def _bounded_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return min(30.0, max(0.0, parsed)) if math.isfinite(parsed) else None


class RemoteReferenceCandidate(_FrozenModel):
    reference_id: str = Field(pattern=_OPAQUE.pattern)
    source: SourceName
    source_image_id: str = Field(pattern=_OPAQUE.pattern)
    source_sequence_id: str = Field(pattern=_OPAQUE.pattern)
    source_url: str = Field(min_length=1, max_length=800)
    download_url: SecretStr = Field(repr=False)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    coordinate_uncertainty_m: float = Field(gt=0, le=1_000_000)
    heading_degrees: float | None = Field(default=None, ge=0, lt=360)
    captured_at: datetime | None = None
    license: str = Field(min_length=1, max_length=500)
    license_url: str = Field(min_length=1, max_length=800)
    attribution: str = Field(min_length=1, max_length=500)
    urbanicity: Urbanicity = "unknown"

    @field_validator("source_url", "license_url")
    @classmethod
    def stable_https_url(cls, value: str) -> str:
        parsed = urllib.parse.urlsplit(value)
        if (
            parsed.scheme.casefold() != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("stable provenance URL required")
        return value

    @field_validator("captured_at")
    @classmethod
    def timezone_required(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("capture time must be timezone-aware")
        return value

    @property
    def season(self) -> Season:
        if self.captured_at is None:
            return "unknown"
        month = self.captured_at.month
        if month in {12, 1, 2}:
            return "winter"
        if month in {3, 4, 5}:
            return "spring"
        if month in {6, 7, 8}:
            return "summer"
        return "autumn"

    @property
    def heading_bin(self) -> int | None:
        if self.heading_degrees is None:
            return None
        return int(self.heading_degrees // 45) % 8


class ReferenceSourceAdapter(Protocol):
    @property
    def source(self) -> SourceName: ...

    @property
    def terms_version(self) -> str: ...

    def discover(
        self,
        target: ProvinceAcquisitionTarget,
        *,
        limit: int,
    ) -> tuple[RemoteReferenceCandidate, ...]: ...

    def download(self, candidate: RemoteReferenceCandidate, *, max_bytes: int) -> bytes: ...


def _external_opaque(value: object) -> str | None:
    normalized = str(value).strip() if value is not None else ""
    return normalized if _OPAQUE.fullmatch(normalized) else None


def _reference_id(source: SourceName, image_id: str) -> str:
    digest = hashlib.sha256(f"{source}:{image_id}".encode()).hexdigest()[:32]
    return f"{source}-{digest}"


def _float(value: object) -> float | None:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _heading(value: object) -> float | None:
    number = _float(value)
    return number % 360.0 if number is not None else None


def _mapillary_capture_time(value: object) -> datetime | None:
    number = _float(value)
    if number is None or number <= 0:
        return None
    try:
        return datetime.fromtimestamp(number / 1000.0, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _timezone_capture_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


class MapillaryGraphAdapter:
    source: Literal["mapillary"] = "mapillary"
    terms_version = "mapillary-cc-by-sa-reviewed-2025-05-12"
    _METADATA_HOSTS = frozenset({"graph.mapillary.com"})
    _IMAGE_HOSTS = frozenset({"fbcdn.net", "mapillary.com"})

    def __init__(
        self,
        settings: MapillaryBackendSettings,
        *,
        http: RetryingHttpClient | None = None,
        bbox_half_span_degrees: float = 0.08,
    ) -> None:
        if not settings.enabled or settings.access_token is None:
            raise ReferenceAcquisitionError("mapillary_not_enabled")
        if not 0 < bbox_half_span_degrees <= 0.25:
            raise ValueError("invalid Mapillary bounding box")
        self._token = settings.access_token
        self._image_width = settings.image_width
        self._max_images = settings.max_images
        self._http = http or RetryingHttpClient()
        self._half_span = bbox_half_span_degrees

    def discover(
        self,
        target: ProvinceAcquisitionTarget,
        *,
        limit: int,
    ) -> tuple[RemoteReferenceCandidate, ...]:
        bounded_limit = min(100, self._max_images, max(1, limit))
        fields = (
            "id,computed_geometry,computed_compass_angle,compass_angle,captured_at,"
            f"sequence,creator,thumb_{self._image_width}_url"
        )
        bbox = ",".join(
            f"{value:.6f}"
            for value in (
                target.center_longitude - self._half_span,
                target.center_latitude - self._half_span,
                target.center_longitude + self._half_span,
                target.center_latitude + self._half_span,
            )
        )
        query = urllib.parse.urlencode(
            {"bbox": bbox, "fields": fields, "limit": str(bounded_limit)}
        )
        payload = self._http.get_json(
            f"https://graph.mapillary.com/images?{query}",
            headers={"Authorization": f"OAuth {self._token.get_secret_value()}"},
            allowed_hosts=self._METADATA_HOSTS,
        )
        data = payload.get("data")
        if not isinstance(data, list):
            raise ReferenceAcquisitionError("mapillary_metadata_invalid")
        candidates: list[RemoteReferenceCandidate] = []
        for item in data[:bounded_limit]:
            candidate = self._parse_item(item)
            if candidate is not None:
                candidates.append(candidate)
        return tuple(candidates)

    def _parse_item(self, value: object) -> RemoteReferenceCandidate | None:
        if not isinstance(value, dict):
            return None
        item = cast(dict[str, Any], value)
        image_id = _external_opaque(item.get("id"))
        geometry = item.get("computed_geometry")
        if not isinstance(geometry, dict):
            return None
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            return None
        longitude, latitude = _float(coordinates[0]), _float(coordinates[1])
        creator = item.get("creator")
        username = creator.get("username") if isinstance(creator, dict) else None
        image_url = item.get(f"thumb_{self._image_width}_url")
        sequence = item.get("sequence")
        if isinstance(sequence, dict):
            sequence = sequence.get("id")
        sequence_id = _external_opaque(sequence)
        if (
            image_id is None
            or sequence_id is None
            or latitude is None
            or longitude is None
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
            or not isinstance(username, str)
            or not username.strip()
            or not isinstance(image_url, str)
        ):
            return None
        try:
            _url_validator(self._IMAGE_HOSTS)(image_url)
            return RemoteReferenceCandidate(
                reference_id=_reference_id("mapillary", image_id),
                source="mapillary",
                source_image_id=image_id,
                source_sequence_id=sequence_id,
                source_url=f"https://graph.mapillary.com/{image_id}",
                download_url=SecretStr(image_url),
                latitude=latitude,
                longitude=longitude,
                coordinate_uncertainty_m=50.0,
                heading_degrees=_heading(
                    _first(item, "computed_compass_angle", "compass_angle")
                ),
                captured_at=_mapillary_capture_time(item.get("captured_at")),
                license=MAPILLARY_LICENSE,
                license_url=CC_BY_SA_4_URL,
                attribution=f"Mapillary image by {username.strip()}; CC BY-SA 4.0",
            )
        except (ReferenceAcquisitionError, ValueError):
            return None

    def download(self, candidate: RemoteReferenceCandidate, *, max_bytes: int) -> bytes:
        if candidate.source != self.source:
            raise ReferenceAcquisitionError("reference_source_mismatch")
        return self._http.get_bytes(
            candidate.download_url.get_secret_value(),
            headers={},
            allowed_hosts=self._IMAGE_HOSTS,
            max_bytes=max_bytes,
        )


class KartaViewApiAdapter:
    source: Literal["kartaview"] = "kartaview"
    terms_version = KARTAVIEW_REVIEWED_TERMS_VERSION
    _METADATA_HOSTS = frozenset({"api.openstreetcam.org"})
    _IMAGE_HOSTS = frozenset({"openstreetcam.org", "kartaview.org"})

    def __init__(
        self,
        *,
        accepted_terms_version: str,
        http: RetryingHttpClient | None = None,
    ) -> None:
        if accepted_terms_version != KARTAVIEW_REVIEWED_TERMS_VERSION:
            raise ReferenceAcquisitionError("kartaview_terms_acceptance_required")
        self._http = http or RetryingHttpClient()

    def discover(
        self,
        target: ProvinceAcquisitionTarget,
        *,
        limit: int,
    ) -> tuple[RemoteReferenceCandidate, ...]:
        query = urllib.parse.urlencode(
            {
                "lat": f"{target.center_latitude:.6f}",
                "lng": f"{target.center_longitude:.6f}",
                # KartaView's documented nearby-photo radius is capped at 2 km.
                # A zoom-only query can validly return API code 601 even where
                # the radius endpoint reports coverage.
                "radius": "2000",
                "join": "sequence",
                "orderBy": "id",
                "orderDirection": "desc",
            }
        )
        payload = self._http.get_json(
            f"https://api.openstreetcam.org/2.0/photo/?{query}",
            headers={"User-Agent": "AtlasLens/0.1 reference-acquisition"},
            allowed_hosts=self._METADATA_HOSTS,
        )
        status = payload.get("status")
        if (
            isinstance(status, dict)
            and status.get("apiCode") == 601
            and status.get("httpCode") == 200
        ):
            # KartaView represents a valid zero-coverage response as HTTP 200 with
            # status.apiCode 601 and no result object.
            return ()
        result = payload.get("result")
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list):
            raise ReferenceAcquisitionError("kartaview_metadata_invalid")
        candidates: list[RemoteReferenceCandidate] = []
        for item in data[: min(max(1, limit), 250)]:
            candidate = self._parse_item(item)
            if candidate is not None:
                candidates.append(candidate)
        return tuple(candidates)

    def _parse_item(self, value: object) -> RemoteReferenceCandidate | None:
        if not isinstance(value, dict):
            return None
        item = cast(dict[str, Any], value)
        raw_image_id = _first(item, "id", "photoId", "imageId")
        raw_sequence = _first(item, "sequenceId", "sequence_id", "sequence")
        if isinstance(raw_sequence, dict):
            raw_sequence = _first(raw_sequence, "id", "sequenceId")
        latitude = _float(_first(item, "lat", "latitude", "gpsLat"))
        longitude = _float(_first(item, "lng", "longitude", "gpsLng", "lon"))
        image_url = _first(
            item,
            "imageLthUrl",
            "fileurlLTh",
            "imageThUrl",
            "fileurlTh",
            "imageProcUrl",
            "fileurlProc",
            "fileurl",
            "lth_name",
            "procUrl",
            "imageUrl",
            "url",
        )
        sequence_index = _external_opaque(
            _first(item, "sequenceIndex", "sequence_index")
        )
        image_id = _external_opaque(raw_image_id)
        sequence_id = _external_opaque(raw_sequence)
        if (
            image_id is None
            or sequence_id is None
            or sequence_index is None
            or latitude is None
            or longitude is None
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
            or not isinstance(image_url, str)
        ):
            return None
        try:
            _url_validator(self._IMAGE_HOSTS)(image_url)
            return RemoteReferenceCandidate(
                reference_id=_reference_id("kartaview", image_id),
                source="kartaview",
                source_image_id=image_id,
                source_sequence_id=sequence_id,
                source_url=(
                    f"https://kartaview.org/details/{sequence_id}/{sequence_index}/track-info"
                ),
                download_url=SecretStr(image_url),
                latitude=latitude,
                longitude=longitude,
                coordinate_uncertainty_m=75.0,
                heading_degrees=_heading(
                    _first(item, "heading", "gpsHeading", "cameraHeading")
                ),
                captured_at=_timezone_capture_time(
                    _first(
                        item,
                        "shotDate",
                        "capturedAt",
                        "captured_at",
                        "captureDate",
                    )
                ),
                license=KARTAVIEW_LICENSE,
                license_url=CC_BY_SA_4_URL,
                attribution="© Grab and KartaView Contributors",
            )
        except (ReferenceAcquisitionError, ValueError):
            return None

    def download(self, candidate: RemoteReferenceCandidate, *, max_bytes: int) -> bytes:
        if candidate.source != self.source:
            raise ReferenceAcquisitionError("reference_source_mismatch")
        return self._http.get_bytes(
            candidate.download_url.get_secret_value(),
            headers={"User-Agent": "AtlasLens/0.1 reference-acquisition"},
            allowed_hosts=self._IMAGE_HOSTS,
            max_bytes=max_bytes,
        )


def _first(item: Mapping[str, Any], *names: str) -> object | None:
    for name in names:
        if name in item and item[name] is not None:
            value: object = item[name]
            return value
    return None


def _normalized_place_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if character.isalnum())


class ResolvedReferenceCandidate(_FrozenModel):
    candidate: RemoteReferenceCandidate
    province: str = Field(min_length=1, max_length=160)
    city: str | None = Field(default=None, max_length=160)


class LocalAdministrativeResolver:
    """Accepts only locally resolved TR records matching the planned province stratum."""

    def __init__(self, resolver: GazetteerResolver, *, maximum_city_distance_km: float = 25.0):
        if maximum_city_distance_km <= 0:
            raise ValueError("city distance must be positive")
        self._resolver = resolver
        self._maximum_city_distance = maximum_city_distance_km

    def resolve(
        self,
        target: ProvinceAcquisitionTarget,
        candidate: RemoteReferenceCandidate,
    ) -> ResolvedReferenceCandidate | None:
        try:
            place = self._resolver.resolve(candidate.latitude, candidate.longitude)
        except (OSError, RuntimeError, ValueError):
            return None
        if (
            place.country_code != "TR"
            or place.region is None
            or _normalized_place_name(place.region) != _normalized_place_name(target.province)
        ):
            return None
        city = (
            place.city
            if place.city is not None and place.distance_km <= self._maximum_city_distance
            else None
        )
        return ResolvedReferenceCandidate(
            candidate=candidate,
            province=place.region,
            city=city,
        )


def stratified_select(
    candidates: Sequence[ResolvedReferenceCandidate],
    *,
    limit: int,
    max_per_sequence: int,
) -> tuple[ResolvedReferenceCandidate, ...]:
    if limit < 1 or max_per_sequence < 1:
        raise ValueError("selection bounds must be positive")
    remaining = sorted(
        {item.candidate.reference_id: item for item in candidates}.values(),
        key=lambda item: item.candidate.reference_id,
    )
    selected: list[ResolvedReferenceCandidate] = []
    sequence_counts: Counter[tuple[SourceName, str]] = Counter()
    source_counts: Counter[SourceName] = Counter()
    heading_counts: Counter[int | None] = Counter()
    season_counts: Counter[Season] = Counter()
    urbanicity_counts: Counter[Urbanicity] = Counter()
    while remaining and len(selected) < limit:
        eligible = [
            item
            for item in remaining
            if sequence_counts[
                (item.candidate.source, item.candidate.source_sequence_id)
            ]
            < max_per_sequence
        ]
        if not eligible:
            break
        choice = min(
            eligible,
            key=lambda item: (
                sequence_counts[(item.candidate.source, item.candidate.source_sequence_id)],
                source_counts[item.candidate.source],
                heading_counts[item.candidate.heading_bin],
                season_counts[item.candidate.season],
                urbanicity_counts[item.candidate.urbanicity],
                item.candidate.reference_id,
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
        sequence_counts[(choice.candidate.source, choice.candidate.source_sequence_id)] += 1
        source_counts[choice.candidate.source] += 1
        heading_counts[choice.candidate.heading_bin] += 1
        season_counts[choice.candidate.season] += 1
        urbanicity_counts[choice.candidate.urbanicity] += 1
    return tuple(selected)


class AcquisitionReceiptEntry(_FrozenModel):
    reference_id: str = Field(pattern=_OPAQUE.pattern)
    source: SourceName
    source_image_id: str = Field(pattern=_OPAQUE.pattern)
    source_sequence_id: str = Field(pattern=_OPAQUE.pattern)
    relative_path: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,239}$")
    sha256: str = Field(pattern=_SHA256.pattern)
    perceptual_hash: str = Field(pattern=_DHASH.pattern)
    size_bytes: int = Field(gt=0)


class ReferenceAcquisitionReceipt(_FrozenModel):
    schema_version: Literal["atlaslens-reference-acquisition-receipt-v1"]
    plan_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,159}$")
    plan_sha256: str = Field(pattern=_SHA256.pattern)
    source_terms_versions: dict[SourceName, str]
    updated_at: datetime
    completed: bool
    entries: tuple[AcquisitionReceiptEntry, ...]

    @field_validator("updated_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("receipt time must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def unique_entries(self) -> ReferenceAcquisitionReceipt:
        identifiers = [item.reference_id for item in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("receipt reference IDs must be unique")
        return self


class ReferenceAcquisitionResult(_FrozenModel):
    status: Literal["complete", "partial", "empty"]
    planned_images: int = Field(ge=0)
    discovered_images: int = Field(ge=0)
    selected_images: int = Field(ge=0)
    newly_downloaded_images: int = Field(ge=0)
    resumed_images: int = Field(ge=0)
    total_images: int = Field(ge=0)
    covered_province_count: int = Field(ge=0, le=81)
    missing_province_count: int = Field(ge=0, le=81)
    source_counts: dict[SourceName, int]
    storage_bytes: int = Field(ge=0)


def _validated_image(
    payload: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
) -> tuple[str, str, str]:
    if not payload or len(payload) > max_bytes:
        raise ReferenceAcquisitionError("downloaded_image_size_invalid")
    digest = hashlib.sha256(payload).hexdigest()
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            if (
                source.format not in _SUPPORTED_FORMATS
                or getattr(source, "n_frames", 1) != 1
                or source.width <= 0
                or source.height <= 0
                or source.width * source.height > max_pixels
            ):
                raise ReferenceAcquisitionError("downloaded_image_invalid")
            grayscale = ImageOps.exif_transpose(source).convert("L").resize((9, 8))
            pixels = list(grayscale.getdata())
            image_format = cast(str, source.format)
    except ReferenceAcquisitionError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ReferenceAcquisitionError("downloaded_image_invalid") from exc
    number = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            number = (number << 1) | int(pixels[offset + column + 1] > pixels[offset + column])
    return digest, f"{number:016x}", _SUPPORTED_FORMATS[image_format]


def _atomic_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _safe_local_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    candidate = (root / relative).resolve()
    if relative.is_absolute() or not candidate.is_relative_to(root) or candidate.is_symlink():
        raise ReferenceAcquisitionError("receipt_asset_invalid")
    return candidate


class ReferenceCorpusAcquirer:
    MANIFEST_FILENAME = "reference-input.json"
    RECEIPT_FILENAME = "acquisition-receipt.json"

    def __init__(
        self,
        *,
        sources: Sequence[ReferenceSourceAdapter],
        administrative_resolver: LocalAdministrativeResolver,
    ) -> None:
        if not sources or len({source.source for source in sources}) != len(sources):
            raise ReferenceAcquisitionError("reference_sources_invalid")
        self._sources = {source.source: source for source in sources}
        self._resolver = administrative_resolver

    def acquire(
        self,
        plan: ReferenceAcquisitionPlan,
        output_root: Path,
    ) -> ReferenceAcquisitionResult:
        root = output_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise ReferenceAcquisitionError("output_root_invalid")
        manifest_path = root / self.MANIFEST_FILENAME
        receipt_path = root / self.RECEIPT_FILENAME
        records, receipt_entries = self._load_resume_state(
            root,
            manifest_path,
            receipt_path,
            plan,
        )
        resumed = len(records)
        discovered_count = 0
        selected_by_province: dict[str, tuple[ResolvedReferenceCandidate, ...]] = {}
        for target in plan.provinces:
            discovered: list[ResolvedReferenceCandidate] = []
            source_limit = min(
                250,
                target.requested_images * plan.policy.discovery_oversample,
            )
            for source in self._sources.values():
                remote = source.discover(target, limit=source_limit)
                discovered_count += len(remote)
                for candidate in remote:
                    resolved = self._resolver.resolve(target, candidate)
                    if resolved is not None:
                        discovered.append(resolved)
            selected_by_province[target.admin1_code] = stratified_select(
                discovered,
                limit=target.requested_images,
                max_per_sequence=plan.policy.max_per_sequence,
            )

        selected = _round_robin_candidates(plan, selected_by_province)
        existing_ids = {record.reference_id for record in records}
        storage_bytes = sum(entry.size_bytes for entry in receipt_entries)
        newly_downloaded = 0
        for item in selected:
            candidate = item.candidate
            if candidate.reference_id in existing_ids:
                continue
            source = self._sources[candidate.source]
            payload = source.download(candidate, max_bytes=plan.policy.max_image_bytes)
            digest, perceptual_hash, suffix = _validated_image(
                payload,
                max_bytes=plan.policy.max_image_bytes,
                max_pixels=plan.policy.max_image_pixels,
            )
            if any(
                entry.sha256 == digest
                or (int(entry.perceptual_hash, 16) ^ int(perceptual_hash, 16)).bit_count()
                <= 4
                for entry in receipt_entries
            ):
                continue
            if storage_bytes + len(payload) > plan.policy.max_disk_bytes:
                raise ReferenceAcquisitionError("actual_storage_exceeds_disk_limit")
            relative_path = f"images/{candidate.source}/{candidate.reference_id}{suffix}"
            asset_path = _safe_local_file(root, relative_path)
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = asset_path.with_suffix(asset_path.suffix + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, asset_path)
            record = ReferenceBuildRecord(
                reference_id=candidate.reference_id,
                source=candidate.source,
                source_family=f"{candidate.source}_street_imagery_family",
                source_image_id=candidate.source_image_id,
                source_sequence_id=candidate.source_sequence_id,
                source_url=candidate.source_url,
                latitude=candidate.latitude,
                longitude=candidate.longitude,
                coordinate_uncertainty_m=candidate.coordinate_uncertainty_m,
                heading_degrees=candidate.heading_degrees,
                captured_at=candidate.captured_at,
                country="TR",
                province=item.province,
                city=item.city,
                license=candidate.license,
                license_url=candidate.license_url,
                attribution=candidate.attribution,
                asset_key=f"reference/{candidate.reference_id}",
                image_path=relative_path,
                expected_sha256=digest,
                expected_perceptual_hash=perceptual_hash,
            )
            entry = AcquisitionReceiptEntry(
                reference_id=candidate.reference_id,
                source=candidate.source,
                source_image_id=candidate.source_image_id,
                source_sequence_id=candidate.source_sequence_id,
                relative_path=relative_path,
                sha256=digest,
                perceptual_hash=perceptual_hash,
                size_bytes=len(payload),
            )
            records.append(record)
            receipt_entries.append(entry)
            existing_ids.add(candidate.reference_id)
            storage_bytes += len(payload)
            newly_downloaded += 1
            self._write_state(
                manifest_path,
                receipt_path,
                plan,
                records,
                receipt_entries,
                source_terms_versions={
                    name: source.terms_version for name, source in self._sources.items()
                },
                completed=False,
            )

        self._write_state(
            manifest_path,
            receipt_path,
            plan,
            records,
            receipt_entries,
            source_terms_versions={
                name: source.terms_version for name, source in self._sources.items()
            },
            completed=True,
        )
        covered = len({record.province for record in records})
        source_counts = Counter(entry.source for entry in receipt_entries)
        status: Literal["complete", "partial", "empty"]
        if not records:
            status = "empty"
        elif covered == 81 and len(records) >= plan.estimated_image_count:
            status = "complete"
        else:
            status = "partial"
        return ReferenceAcquisitionResult(
            status=status,
            planned_images=plan.estimated_image_count,
            discovered_images=discovered_count,
            selected_images=len(selected),
            newly_downloaded_images=newly_downloaded,
            resumed_images=resumed,
            total_images=len(records),
            covered_province_count=covered,
            missing_province_count=81 - covered,
            source_counts=dict(source_counts),
            storage_bytes=storage_bytes,
        )

    def _load_resume_state(
        self,
        root: Path,
        manifest_path: Path,
        receipt_path: Path,
        plan: ReferenceAcquisitionPlan,
    ) -> tuple[list[ReferenceBuildRecord], list[AcquisitionReceiptEntry]]:
        if not manifest_path.exists() and not receipt_path.exists():
            return [], []
        if not manifest_path.is_file() or not receipt_path.is_file():
            raise ReferenceAcquisitionError("resume_state_incomplete")
        try:
            manifest = ReferenceBuildInput.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            receipt = ReferenceAcquisitionReceipt.model_validate_json(
                receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise ReferenceAcquisitionError("resume_state_invalid") from exc
        if receipt.plan_version != plan.plan_version or receipt.plan_sha256 != plan.digest:
            raise ReferenceAcquisitionError("resume_plan_mismatch")
        current_terms = {
            name: source.terms_version for name, source in self._sources.items()
        }
        if any(
            receipt.source_terms_versions.get(name) != version
            for name, version in current_terms.items()
        ):
            raise ReferenceAcquisitionError("resume_terms_mismatch")
        records = list(manifest.records)
        entries = list(receipt.entries)
        records_by_id = {record.reference_id: record for record in records}
        receipt_ids = {entry.reference_id for entry in entries}
        if not receipt_ids.issubset(records_by_id):
            raise ReferenceAcquisitionError("resume_state_invalid")
        for reference_id in sorted(records_by_id.keys() - receipt_ids):
            record = records_by_id[reference_id]
            if (
                record.source not in {"mapillary", "kartaview"}
                or record.expected_sha256 is None
                or record.expected_perceptual_hash is None
            ):
                raise ReferenceAcquisitionError("resume_state_invalid")
            path = _safe_local_file(root, record.image_path)
            if not path.is_file():
                raise ReferenceAcquisitionError("resume_asset_invalid")
            entries.append(
                AcquisitionReceiptEntry(
                    reference_id=record.reference_id,
                    source=record.source,
                    source_image_id=record.source_image_id,
                    source_sequence_id=record.source_sequence_id,
                    relative_path=record.image_path,
                    sha256=record.expected_sha256,
                    perceptual_hash=record.expected_perceptual_hash,
                    size_bytes=path.stat().st_size,
                )
            )
        for entry in entries:
            path = _safe_local_file(root, entry.relative_path)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry.sha256:
                raise ReferenceAcquisitionError("resume_asset_invalid")
        return records, entries

    @staticmethod
    def _write_state(
        manifest_path: Path,
        receipt_path: Path,
        plan: ReferenceAcquisitionPlan,
        records: Sequence[ReferenceBuildRecord],
        entries: Sequence[AcquisitionReceiptEntry],
        *,
        source_terms_versions: Mapping[SourceName, str],
        completed: bool,
    ) -> None:
        ordered_records = tuple(sorted(records, key=lambda item: item.reference_id))
        ordered_entries = tuple(sorted(entries, key=lambda item: item.reference_id))
        _atomic_model(
            manifest_path,
            ReferenceBuildInput(
                schema_version=REFERENCE_INPUT_SCHEMA_VERSION,
                records=ordered_records,
            ),
        )
        _atomic_model(
            receipt_path,
            ReferenceAcquisitionReceipt(
                schema_version=ACQUISITION_RECEIPT_SCHEMA_VERSION,
                plan_version=plan.plan_version,
                plan_sha256=plan.digest,
                source_terms_versions=dict(source_terms_versions),
                updated_at=datetime.now(UTC),
                completed=completed,
                entries=ordered_entries,
            ),
        )


def _round_robin_candidates(
    plan: ReferenceAcquisitionPlan,
    selected_by_province: Mapping[str, Sequence[ResolvedReferenceCandidate]],
) -> tuple[ResolvedReferenceCandidate, ...]:
    ordered: list[ResolvedReferenceCandidate] = []
    maximum = max((len(items) for items in selected_by_province.values()), default=0)
    for position in range(maximum):
        for target in plan.provinces:
            candidates = selected_by_province.get(target.admin1_code, ())
            if position < len(candidates):
                ordered.append(candidates[position])
                if len(ordered) >= plan.policy.max_images:
                    return tuple(ordered)
    return tuple(ordered)


__all__ = [
    "ACQUISITION_PLAN_SCHEMA_VERSION",
    "ACQUISITION_RECEIPT_SCHEMA_VERSION",
    "KARTAVIEW_REVIEWED_TERMS_VERSION",
    "AcquisitionPolicy",
    "HttpResponse",
    "HttpTransport",
    "KartaViewApiAdapter",
    "LocalAdministrativeResolver",
    "MapillaryBackendSettings",
    "MapillaryGraphAdapter",
    "ProvinceAcquisitionTarget",
    "ReferenceAcquisitionError",
    "ReferenceAcquisitionPlan",
    "ReferenceAcquisitionResult",
    "ReferenceCorpusAcquirer",
    "RemoteReferenceCandidate",
    "ResolvedReferenceCandidate",
    "RetryingHttpClient",
    "build_acquisition_plan",
    "load_mapillary_backend_settings",
    "stratified_select",
]
