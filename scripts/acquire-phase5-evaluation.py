"""Acquire the explicit, license-reviewed Phase 5 Wikimedia Commons slice.

The command is intentionally separate from application startup and model
installation. It creates local, metadata-free evaluation derivatives under
'.local'; no asset is committed or used for training.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / ".local" / "acceptance-images"
EVALUATION_ROOT = ROOT / ".local" / "evaluation"
MANIFEST_PATH = EVALUATION_ROOT / "manifest.csv"
SOURCES_PATH = EVALUATION_ROOT / "source-records.json"
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AtlasLens-Phase5-Evaluation/1.0 (local license-reviewed smoke slice)"
ALLOWED_API_HOSTS = frozenset({"commons.wikimedia.org"})
ALLOWED_IMAGE_HOSTS = frozenset({"upload.wikimedia.org"})
MAX_API_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000


@dataclass(frozen=True, slots=True)
class Source:
    title: str
    latitude: float
    longitude: float
    country_code: str
    region: str
    city_or_area: str
    continent: str
    license: str
    author: str
    scene_category: str
    geographic_cell: str
    truth_location_kind: str = "camera_position"


SOURCES = (
    Source(
        "File:Eiffel Tower viewed from the Champ de Mars.jpg",
        48.856342,
        2.297417,
        "FR",
        "Île-de-France",
        "Paris",
        "Europe",
        "CC-BY-SA-4.0",
        "APK",
        "urban_landmark",
        "fr-paris",
    ),
    Source(
        "File:Sydney Harbour including Harbour bridge and Opera House.jpg",
        -33.849008,
        151.205951,
        "AU",
        "New South Wales",
        "Sydney",
        "Oceania",
        "CC0-1.0",
        "Gibrate1",
        "urban_harbour",
        "au-sydney",
    ),
    Source(
        "File:Fort Point View of Golden Gate Bridge.jpg",
        37.808997,
        -122.473878,
        "US",
        "California",
        "San Francisco",
        "North America",
        "CC0-1.0",
        "CaryXz5",
        "bridge_landmark",
        "us-san-francisco",
    ),
    Source(
        "File:00 1031 Brasilien, Rio de Janeiro.jpg",
        -22.952000,
        -43.211481,
        "BR",
        "Rio de Janeiro",
        "Rio de Janeiro",
        "South America",
        "CC-BY-SA-4.0",
        "W. Bulach",
        "urban_landscape",
        "br-rio",
    ),
    Source(
        "File:Chureito Pagoda and Mount Fuji 20241022.jpg",
        35.501427,
        138.801575,
        "JP",
        "Yamanashi",
        "Fujiyoshida",
        "Asia",
        "CC-BY-4.0",
        "Supanut Arunoprayote",
        "mountain_landmark",
        "jp-fujiyoshida",
    ),
    Source(
        "File:Table mountain 1 – Panorama (Greg Zaal and Rico Cilliers via Poly Haven).jpg",
        -33.958547,
        18.405612,
        "ZA",
        "Western Cape",
        "Cape Town",
        "Africa",
        "CC0-1.0",
        "Greg Zaal and Rico Cilliers",
        "panoramic_landscape",
        "za-cape-town",
    ),
)

HEADERS = (
    "image_asset_key",
    "local_reference",
    "true_latitude",
    "true_longitude",
    "country_code",
    "region",
    "city_or_area",
    "continent",
    "source",
    "source_record_id",
    "license",
    "attribution",
    "split",
    "scene_category",
    "geographic_cell",
    "capture_family_id",
    "content_sha256",
    "perceptual_hash",
)


class ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: frozenset[str]) -> None:
        self._allowed_hosts = allowed_hosts

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request:
        _validate_url(new_url, self._allowed_hosts)
        redirected = super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )
        if redirected is None:
            raise RuntimeError("official source redirect was rejected")
        return redirected


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: object) -> str:
    parser = _TextExtractor()
    parser.feed(str(value))
    parser.close()
    return " ".join(html.unescape(" ".join(parser.parts)).split())


def _author_assertions(source: Source) -> tuple[str, ...]:
    if source.author == "Greg Zaal and Rico Cilliers":
        return ("Greg Zaal", "Rico Cilliers")
    return (source.author,)


def _validate_url(url: str, allowed_hosts: frozenset[str]) -> None:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError("unexpected official source URL")


def _fetch(url: str, *, allowed_hosts: frozenset[str], limit: int) -> bytes:
    _validate_url(url, allowed_hosts)
    opener = urllib.request.build_opener(ValidatingRedirectHandler(allowed_hosts))
    request = urllib.request.Request(  # noqa: S310 - URL is strictly HTTPS/allowlisted
        url, headers={"User-Agent": USER_AGENT}
    )
    context = ssl.create_default_context()
    opener.add_handler(urllib.request.HTTPSHandler(context=context))
    for attempt in range(4):
        try:
            with opener.open(request, timeout=60) as response:  # noqa: S310
                _validate_url(response.geturl(), allowed_hosts)
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) > limit:
                    raise RuntimeError("official source response exceeds size bound")
                chunks: list[bytes] = []
                total = 0
                while chunk := response.read(min(1024 * 1024, limit + 1 - total)):
                    total += len(chunk)
                    if total > limit:
                        raise RuntimeError("official source response exceeds size bound")
                    chunks.append(chunk)
                return b"".join(chunks)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 503} or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                requested_delay = float(retry_after) if retry_after is not None else 0.0
            except ValueError:
                requested_delay = 0.0
            time.sleep(min(30.0, max(requested_delay, float(2 ** (attempt + 1)))))
    raise RuntimeError("official source request retries were exhausted")


def _api_record(source: Source) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": "1280",
            "titles": source.title,
        }
    )
    payload = json.loads(
        _fetch(f"{API_URL}?{query}", allowed_hosts=ALLOWED_API_HOSTS, limit=MAX_API_BYTES)
    )
    pages = payload.get("query", {}).get("pages", [])
    if len(pages) != 1 or pages[0].get("missing") is True:
        raise RuntimeError("reviewed Commons record is unavailable")
    page = pages[0]
    if page.get("title") != source.title or not isinstance(page.get("pageid"), int):
        raise RuntimeError("reviewed Commons record identity changed")
    image_info = page.get("imageinfo", [])
    if len(image_info) != 1 or image_info[0].get("mime") != "image/jpeg":
        raise RuntimeError("reviewed Commons media type changed")
    thumb_url = image_info[0].get("thumburl")
    if not isinstance(thumb_url, str):
        raise RuntimeError("reviewed Commons thumbnail is unavailable")
    metadata = image_info[0].get("extmetadata", {})
    api_license = str(metadata.get("LicenseShortName", {}).get("value", ""))
    accepted = {
        "CC-BY-SA-4.0": {"CC BY-SA 4.0"},
        "CC-BY-4.0": {"CC BY 4.0"},
        "CC0-1.0": {"CC0", "CC0 1.0"},
    }[source.license]
    if api_license not in accepted:
        raise RuntimeError("reviewed Commons license changed")
    artist = _plain_text(metadata.get("Artist", {}).get("value", ""))
    credit = _plain_text(metadata.get("Credit", {}).get("value", ""))
    combined_credit = f"{artist} {credit}".casefold()
    if not artist or any(
        assertion.casefold() not in combined_credit
        for assertion in _author_assertions(source)
    ):
        raise RuntimeError("reviewed Commons attribution changed")
    license_url = str(metadata.get("LicenseUrl", {}).get("value", ""))
    parsed_license = urllib.parse.urlparse(license_url)
    if (
        parsed_license.scheme not in {"http", "https"}
        or parsed_license.hostname != "creativecommons.org"
    ):
        raise RuntimeError("reviewed Commons license URL changed")
    attribution_required = str(
        metadata.get("AttributionRequired", {}).get("value", "")
    ).casefold()
    expected_attribution = source.license != "CC0-1.0"
    if attribution_required not in {"true", "false"} or (
        attribution_required == "true"
    ) != expected_attribution:
        raise RuntimeError("reviewed Commons attribution requirement changed")
    for key, expected in (
        ("GPSLatitude", source.latitude),
        ("GPSLongitude", source.longitude),
    ):
        value = metadata.get(key, {}).get("value")
        if value in (None, ""):
            raise RuntimeError("reviewed Commons camera coordinates are unavailable")
        if not math.isclose(float(value), expected, rel_tol=0, abs_tol=0.002):
            raise RuntimeError("reviewed Commons coordinates changed")
    attribution_text = artist
    if credit and all(
        assertion.casefold() in credit.casefold()
        for assertion in _author_assertions(source)
    ):
        attribution_text = credit
    return {
        "pageid": page["pageid"],
        "title": page["title"],
        "page_url": "https://commons.wikimedia.org/wiki/"
        + urllib.parse.quote(source.title.replace(" ", "_"), safe=":()_,-"),
        "thumb_url": thumb_url,
        "api_license": api_license,
        "official_artist": artist,
        "official_credit": credit,
        "license_url": license_url,
        "attribution_required": expected_attribution,
        "attribution_text": attribution_text[:300],
        "truth_location_kind": source.truth_location_kind,
    }


def _metadata_free_jpeg(payload: bytes, destination: Path) -> None:
    with Image.open(io.BytesIO(payload)) as opened:
        opened.verify()
    with Image.open(io.BytesIO(payload)) as opened:
        normalized = ImageOps.exif_transpose(opened).convert("RGB")
        normalized.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        normalized.save(buffer, "JPEG", quality=92, optimize=True)
    destination.write_bytes(buffer.getvalue())
    with Image.open(destination) as check:
        if check.getexif():
            raise RuntimeError("evaluation derivative still contains EXIF")


def _average_hash(path: Path) -> str:
    with Image.open(path) as opened:
        pixels = list(
            opened.convert("L").resize((8, 8), Image.Resampling.LANCZOS).getdata()
        )
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return f"{value:016x}"


def main() -> int:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    source_records: list[dict[str, Any]] = []
    for index, source in enumerate(SOURCES, start=1):
        official = _api_record(source)
        payload = _fetch(
            official["thumb_url"],
            allowed_hosts=ALLOWED_IMAGE_HOSTS,
            limit=MAX_IMAGE_BYTES,
        )
        filename = f"asset-{index:02d}.jpg"
        destination = ASSET_ROOT / filename
        _metadata_free_jpeg(payload, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        source_id = str(official["pageid"])
        attribution = (
            f"{official['attribution_text']}; {source.license}; Wikimedia Commons; "
            "local resized metadata-free evaluation derivative"
        )
        rows.append(
            {
                "image_asset_key": f"commons:{source_id}",
                "local_reference": filename,
                "true_latitude": f"{source.latitude:.6f}",
                "true_longitude": f"{source.longitude:.6f}",
                "country_code": source.country_code,
                "region": source.region,
                "city_or_area": source.city_or_area,
                "continent": source.continent,
                "source": official["page_url"],
                "source_record_id": source_id,
                "license": source.license,
                "attribution": attribution,
                "split": "test",
                "scene_category": source.scene_category,
                "geographic_cell": source.geographic_cell,
                "capture_family_id": f"commons-file:{source_id}",
                "content_sha256": digest,
                "perceptual_hash": _average_hash(destination),
            }
        )
        source_records.append(
            {
                **asdict(source),
                **official,
                "local_asset": filename,
                "content_sha256": digest,
                "derivative": "resized JPEG with metadata removed for local evaluation",
            }
        )
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    SOURCES_PATH.write_text(
        json.dumps(source_records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "acquired",
                "image_count": len(rows),
                "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
                "assets": str(ASSET_ROOT.relative_to(ROOT)),
                "images_contain_exif": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
