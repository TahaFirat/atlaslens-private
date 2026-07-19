from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, cast

from PIL import Image, UnidentifiedImageError

from atlaslens_api.dataset_acquisition.models import (
    AcquisitionReceipt,
    AcquisitionReceiptEntry,
    CommonsReviewRecord,
    SamplingCell,
)
from atlaslens_api.retrieval.manifest import EXTENDED_MANIFEST_COLUMNS

_API_URL = "https://commons.wikimedia.org/w/api.php"
_ALLOWED_MEDIA_HOST = "upload.wikimedia.org"
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
_REVIEW_COLUMNS = tuple(CommonsReviewRecord.model_fields)
_TAG = re.compile(r"<[^>]+>")


class DatasetAcquisitionError(RuntimeError):
    pass


JsonTransport = Callable[[str, dict[str, str], str], dict[str, Any]]
BinaryTransport = Callable[[str, str], bytes]


class _MediaRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_media_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_media_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _ALLOWED_MEDIA_HOST
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise DatasetAcquisitionError("unapproved_media_source")


def _json_transport(url: str, params: dict[str, str], user_agent: str) -> dict[str, Any]:
    if url != _API_URL:
        raise DatasetAcquisitionError("unapproved_metadata_source")
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(  # noqa: S310 - exact HTTPS endpoint is fixed above.
        f"{url}?{query}", headers={"User-Agent": user_agent}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            payload = response.read(4 * 1024 * 1024 + 1)
    except OSError as exc:
        raise DatasetAcquisitionError("metadata_request_failed") from exc
    if len(payload) > 4 * 1024 * 1024:
        raise DatasetAcquisitionError("metadata_response_too_large")
    try:
        value = json.loads(payload)
    except (UnicodeError, ValueError) as exc:
        raise DatasetAcquisitionError("invalid_metadata_response") from exc
    if not isinstance(value, dict):
        raise DatasetAcquisitionError("invalid_metadata_response")
    return value


def _binary_transport(url: str, user_agent: str) -> bytes:
    _validate_media_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})  # noqa: S310
    opener = urllib.request.build_opener(_MediaRedirectHandler())
    try:
        with opener.open(request, timeout=60) as response:  # noqa: S310
            _validate_media_url(response.geturl())
            payload = cast(bytes, response.read(_MAX_DOWNLOAD_BYTES + 1))
    except OSError as exc:
        raise DatasetAcquisitionError("media_download_failed") from exc
    if not payload or len(payload) > _MAX_DOWNLOAD_BYTES:
        raise DatasetAcquisitionError("media_download_size_invalid")
    return payload


def _plain(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    return _TAG.sub("", html.unescape(str(value))).strip()


def _fingerprint(record: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def average_hash(payload: bytes) -> str:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            pixels = list(
                image.convert("L").resize((8, 8), Image.Resampling.LANCZOS).getdata()
            )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise DatasetAcquisitionError("invalid_downloaded_image") from exc
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return f"{value:016x}"


class CommonsApiClient:
    def __init__(
        self,
        *,
        user_agent: str,
        allowed_licenses: frozenset[str],
        json_transport: JsonTransport = _json_transport,
        binary_transport: BinaryTransport = _binary_transport,
        minimum_request_interval_seconds: float = 0.1,
        max_retries: int = 3,
    ) -> None:
        if (
            not user_agent.strip()
            or not allowed_licenses
            or minimum_request_interval_seconds < 0
            or max_retries < 1
        ):
            raise ValueError("Commons requires a User-Agent and explicit license allowlist")
        self.user_agent = user_agent
        self.allowed_licenses = allowed_licenses
        self._json_transport = json_transport
        self._binary_transport = binary_transport
        self._minimum_interval = minimum_request_interval_seconds
        self._max_retries = max_retries
        self._last_request = 0.0

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        for attempt in range(self._max_retries):
            remaining = self._minimum_interval - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
            payload = self._json_transport(_API_URL, params, self.user_agent)
            self._last_request = time.monotonic()
            error = payload.get("error")
            if not isinstance(error, dict) or error.get("code") != "maxlag":
                return payload
            if attempt + 1 < self._max_retries:
                time.sleep(min(2.0, 0.25 * (2**attempt)))
        raise DatasetAcquisitionError("commons_maxlag_retry_exhausted")

    def review_title(
        self,
        title: str,
        *,
        fallback_country: str,
        fallback_region: str | None,
        continent: Literal[
            "Africa", "Asia", "Europe", "North America", "South America", "Oceania"
        ],
        geographic_cell: str,
    ) -> CommonsReviewRecord:
        normalized = title.strip()
        if not normalized or len(normalized) > 500:
            raise DatasetAcquisitionError("invalid_commons_title")
        if not normalized.startswith("File:"):
            normalized = f"File:{normalized}"
        payload = self._request(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "prop": "coordinates|imageinfo",
                "titles": normalized,
                "coprop": "type|name|dim|country|region|globe",
                "iiprop": "url|sha1|mime|extmetadata",
                "iiurlwidth": "1024",
                "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit",
                "maxlag": "2",
            }
        )
        try:
            pages = payload["query"]["pages"]
            page = pages[0]
            image_info = page["imageinfo"][0]
            coordinate = page["coordinates"][0]
            metadata = image_info["extmetadata"]
            page_id = int(page["pageid"])
            latitude = float(coordinate["lat"])
            longitude = float(coordinate["lon"])
            globe = str(coordinate.get("globe", "earth"))
            api_country = _plain(coordinate.get("country"))
            api_region = _plain(coordinate.get("region"))
            source_sha1 = str(image_info["sha1"])
            download_url = str(image_info.get("thumburl") or image_info["url"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DatasetAcquisitionError("incomplete_commons_metadata") from exc
        if globe != "earth" or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise DatasetAcquisitionError("invalid_commons_coordinate")
        _validate_media_url(download_url)
        license_name = _plain(metadata.get("LicenseShortName"))
        license_url = _plain(metadata.get("LicenseUrl"))
        attribution = _plain(metadata.get("Artist")) or _plain(metadata.get("Credit"))
        if license_name not in self.allowed_licenses:
            raise DatasetAcquisitionError("license_not_allowlisted")
        if not license_url.startswith("https://") or not attribution:
            raise DatasetAcquisitionError("incomplete_commons_attribution")
        source_url = f"https://commons.wikimedia.org/?curid={page_id}"
        country = api_country or fallback_country.strip()
        region = api_region or (fallback_region.strip() if fallback_region else None)
        if not country or not geographic_cell.strip():
            raise DatasetAcquisitionError("incomplete_sampling_provenance")
        fingerprint_fields: dict[str, object] = {
            "page_id": page_id,
            "title": str(page["title"]),
            "source_url": source_url,
            "download_url": download_url,
            "source_sha1": source_sha1,
            "latitude": latitude,
            "longitude": longitude,
            "license": license_name,
            "license_url": license_url,
            "attribution": attribution[:500],
            "country": country,
            "region": region,
            "continent": continent,
            "geographic_cell": geographic_cell,
            "country_source": (
                "mediawiki_coordinates" if api_country else "operator_sampling"
            ),
            "region_source": (
                "mediawiki_coordinates"
                if api_region
                else "operator_sampling"
                if region
                else "unavailable"
            ),
        }
        return CommonsReviewRecord(
            approved=False,
            page_id=page_id,
            title=str(page["title"]),
            source_record_id=str(page_id),
            source_url=source_url,
            download_url=download_url,
            source_sha1=source_sha1,
            latitude=latitude,
            longitude=longitude,
            country=country,
            region=region,
            continent=continent,
            geographic_cell=geographic_cell,
            country_source=(
                "mediawiki_coordinates" if api_country else "operator_sampling"
            ),
            region_source=(
                "mediawiki_coordinates"
                if api_region
                else "operator_sampling"
                if region
                else "unavailable"
            ),
            license=license_name,
            license_url=license_url,
            attribution=attribution[:500],
            metadata_fingerprint=_fingerprint(fingerprint_fields),
        )

    def discover_cell(self, cell: SamplingCell) -> list[CommonsReviewRecord]:
        titles: dict[int, str] = {}
        continuation: str | None = None
        candidate_target = min(500, max(50, cell.target * 4))
        while len(titles) < candidate_target:
            params = {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "generator": "geosearch",
                "ggsnamespace": "6",
                "ggscoord": f"{cell.latitude}|{cell.longitude}",
                "ggsradius": str(cell.radius_m),
                "ggslimit": str(candidate_target),
                "maxlag": "2",
            }
            if continuation is not None:
                params["ggscontinue"] = continuation
            payload = self._request(params)
            try:
                pages = payload["query"]["pages"]
            except (KeyError, TypeError) as exc:
                raise DatasetAcquisitionError("invalid_discovery_response") from exc
            if not isinstance(pages, list):
                raise DatasetAcquisitionError("invalid_discovery_response")
            for page in pages:
                if isinstance(page, dict) and "pageid" in page and "title" in page:
                    titles[int(page["pageid"])] = str(page["title"])
            next_value = payload.get("continue")
            continuation = (
                str(next_value["ggscontinue"])
                if isinstance(next_value, dict) and "ggscontinue" in next_value
                else None
            )
            if continuation is None:
                break
        records: list[CommonsReviewRecord] = []
        for _, title in sorted(titles.items()):
            try:
                records.append(
                    self.review_title(
                        title,
                        fallback_country=cell.country,
                        fallback_region=cell.region,
                        continent=cell.continent,
                        geographic_cell=cell.cell_id,
                    )
                )
            except DatasetAcquisitionError as exc:
                if str(exc) in {
                    "license_not_allowlisted",
                    "incomplete_commons_metadata",
                    "incomplete_commons_attribution",
                    "invalid_commons_coordinate",
                }:
                    continue
                raise
            if len(records) >= cell.target:
                break
        return records

    def download(self, url: str) -> bytes:
        return self._binary_transport(url, self.user_agent)


def write_review(path: Path, records: Iterable[CommonsReviewRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_REVIEW_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump(mode="json"))
    os.replace(temporary, path)


def read_review(path: Path) -> list[CommonsReviewRecord]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _REVIEW_COLUMNS:
                raise DatasetAcquisitionError("invalid_review_headers")
            return [CommonsReviewRecord.model_validate(row) for row in reader]
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        if isinstance(exc, DatasetAcquisitionError):
            raise
        raise DatasetAcquisitionError("invalid_review_file") from exc


class CommonsAcquirer:
    def __init__(self, client: CommonsApiClient) -> None:
        self.client = client

    def acquire(
        self,
        review_path: Path,
        output_root: Path,
        manifest_path: Path,
        *,
        excluded_hashes: frozenset[str],
        excluded_perceptual_hashes: tuple[int, ...],
        excluded_capture_families: frozenset[str],
    ) -> AcquisitionReceipt:
        approved = [record for record in read_review(review_path) if record.approved]
        if not approved:
            raise DatasetAcquisitionError("review_has_no_approved_records")
        if any(record.coordinate_kind == "unknown" for record in approved):
            raise DatasetAcquisitionError("coordinate_provenance_not_approved")
        root = output_root.expanduser().resolve()
        assets = root / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        receipt_path = root / "acquisition.json"
        receipt = self._load_receipt(receipt_path)
        entries = dict(receipt.entries)
        rows: list[dict[str, object]] = []
        seen_hashes = set(excluded_hashes)
        seen_perceptual = list(excluded_perceptual_hashes)
        for reviewed in approved:
            current = self.client.review_title(
                reviewed.title,
                fallback_country=reviewed.country,
                fallback_region=reviewed.region,
                continent=reviewed.continent,
                geographic_cell=reviewed.geographic_cell,
            )
            if current.metadata_fingerprint != reviewed.metadata_fingerprint:
                raise DatasetAcquisitionError("commons_metadata_changed_after_review")
            family = f"commons-file:{reviewed.page_id}"
            if family in excluded_capture_families:
                raise DatasetAcquisitionError("evaluation_capture_family_overlap")
            asset_key = f"commons:{reviewed.page_id}:{reviewed.source_sha1[:24]}"
            existing = entries.get(asset_key)
            if existing is not None:
                destination = (root / existing.relative_path).resolve()
                if (
                    destination.is_relative_to(root)
                    and destination.is_file()
                    and hashlib.sha256(destination.read_bytes()).hexdigest()
                    == existing.content_sha256
                    and existing.metadata_fingerprint == reviewed.metadata_fingerprint
                ):
                    rows.append(self._manifest_row(reviewed, existing, family))
                    seen_hashes.add(existing.content_sha256)
                    seen_perceptual.append(int(existing.perceptual_hash, 16))
                    continue
            payload = self.client.download(reviewed.download_url)
            digest = hashlib.sha256(payload).hexdigest()
            perceptual = average_hash(payload)
            perceptual_int = int(perceptual, 16)
            if digest in seen_hashes:
                raise DatasetAcquisitionError("duplicate_reference_content")
            if any((perceptual_int ^ previous).bit_count() <= 4 for previous in seen_perceptual):
                raise DatasetAcquisitionError("near_duplicate_reference_content")
            suffix = self._validated_suffix(payload)
            relative = f"assets/{digest}{suffix}"
            destination = root / relative
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            entry = AcquisitionReceiptEntry(
                asset_key=asset_key,
                relative_path=relative,
                source_record_id=reviewed.source_record_id,
                content_sha256=digest,
                perceptual_hash=perceptual,
                metadata_fingerprint=reviewed.metadata_fingerprint,
            )
            entries[asset_key] = entry
            self._write_receipt(receipt_path, AcquisitionReceipt(entries=entries))
            rows.append(self._manifest_row(reviewed, entry, family))
            seen_hashes.add(digest)
            seen_perceptual.append(perceptual_int)
        self._write_manifest(manifest_path, rows)
        return AcquisitionReceipt(entries=entries)

    @staticmethod
    def _load_receipt(path: Path) -> AcquisitionReceipt:
        if not path.exists():
            return AcquisitionReceipt(entries={})
        try:
            return AcquisitionReceipt.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DatasetAcquisitionError("invalid_acquisition_receipt") from exc

    @staticmethod
    def _write_receipt(path: Path, receipt: AcquisitionReceipt) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _validated_suffix(payload: bytes) -> str:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
                image_format = image.format
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise DatasetAcquisitionError("invalid_downloaded_image") from exc
        suffixes = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
        try:
            return suffixes[image_format or ""]
        except KeyError as exc:
            raise DatasetAcquisitionError("unsupported_downloaded_image") from exc

    @staticmethod
    def _manifest_row(
        reviewed: CommonsReviewRecord,
        entry: AcquisitionReceiptEntry,
        family: str,
    ) -> dict[str, object]:
        return dict(
            zip(
                EXTENDED_MANIFEST_COLUMNS,
                (
                    entry.relative_path,
                    reviewed.latitude,
                    reviewed.longitude,
                    reviewed.country,
                    reviewed.region or "",
                    "",
                    reviewed.license,
                    "Wikimedia Commons",
                    "reviewed official MediaWiki metadata",
                    entry.asset_key,
                    reviewed.source_record_id,
                    reviewed.source_url,
                    reviewed.license_url,
                    reviewed.attribution,
                    str(reviewed.display_allowed).lower(),
                    "commons_reviewed",
                    family,
                    reviewed.coordinate_kind,
                    "",
                    "",
                    "",
                    entry.perceptual_hash,
                    "mediawiki-api-review-v1",
                    reviewed.continent,
                    reviewed.geographic_cell,
                ),
                strict=True,
            )
        )

    @staticmethod
    def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EXTENDED_MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
