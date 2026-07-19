"""Freeze a deterministic, licensed Wikimedia Commons Phase 5B test slice.

This is an explicit operator command. It uses only the official MediaWiki API,
stores metadata-free local derivatives under ``.local`` and never runs during
application startup. Selection is deterministic from predeclared geographic cells
and happens before any Phase 5B model result is observed.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
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
OUTPUT_ROOT = ROOT / ".local" / "phase5b-evaluation"
ASSET_ROOT = OUTPUT_ROOT / "images"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
SOURCES_PATH = OUTPUT_ROOT / "source-records.json"
CHECKPOINT_PATH = OUTPUT_ROOT / "selection-checkpoint.json"
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AtlasLens-Phase5B-Evaluation/1.0 (contact: local-operator)"
API_HOSTS = frozenset({"commons.wikimedia.org"})
IMAGE_HOSTS = frozenset({"upload.wikimedia.org"})
MAX_API_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000


@dataclass(frozen=True, slots=True)
class SamplingCell:
    key: str
    latitude: float
    longitude: float
    country_code: str
    region: str
    city_or_area: str
    continent: str
    scene_category: str


CELLS = (
    SamplingCell(
        "tr-istanbul", 41.0082, 28.9784, "TR", "Istanbul", "Istanbul", "Europe", "text_rich_urban"
    ),
    SamplingCell(
        "gb-london", 51.5074, -0.1278, "GB", "England", "London", "Europe", "text_rich_urban"
    ),
    SamplingCell(
        "de-berlin", 52.5200, 13.4050, "DE", "Berlin", "Berlin", "Europe", "text_rich_urban"
    ),
    SamplingCell("kr-seoul", 37.5665, 126.9780, "KR", "Seoul", "Seoul", "Asia", "text_rich_urban"),
    SamplingCell(
        "th-bangkok", 13.7563, 100.5018, "TH", "Bangkok", "Bangkok", "Asia", "text_rich_urban"
    ),
    SamplingCell(
        "in-mumbai",
        18.9388,
        72.8354,
        "IN",
        "Maharashtra",
        "Mumbai",
        "Asia",
        "text_rich_urban",
    ),
    SamplingCell(
        "ca-toronto",
        43.6532,
        -79.3832,
        "CA",
        "Ontario",
        "Toronto",
        "North America",
        "text_rich_urban",
    ),
    SamplingCell(
        "ar-buenos-aires",
        -34.6037,
        -58.3816,
        "AR",
        "Buenos Aires",
        "Buenos Aires",
        "South America",
        "text_rich_urban",
    ),
    SamplingCell(
        "za-cape-town",
        -33.9249,
        18.4241,
        "ZA",
        "Western Cape",
        "Cape Town",
        "Africa",
        "urban_road",
    ),
    SamplingCell(
        "gh-accra", 5.6037, -0.1870, "GH", "Greater Accra", "Accra", "Africa", "urban_road"
    ),
    SamplingCell("vn-hanoi", 21.0278, 105.8342, "VN", "Hanoi", "Hanoi", "Asia", "urban_road"),
    SamplingCell(
        "au-melbourne",
        -37.8136,
        144.9631,
        "AU",
        "Victoria",
        "Melbourne",
        "Oceania",
        "urban_road",
    ),
    SamplingCell(
        "nl-amsterdam",
        52.3676,
        4.9041,
        "NL",
        "North Holland",
        "Amsterdam",
        "Europe",
        "urban_road",
    ),
    SamplingCell(
        "ma-casablanca",
        33.5731,
        -7.5898,
        "MA",
        "Casablanca-Settat",
        "Casablanca",
        "Africa",
        "urban_road",
    ),
    SamplingCell(
        "is-vik",
        63.4194,
        -19.0060,
        "IS",
        "Southern Region",
        "Vik",
        "Europe",
        "natural_difficult_viewpoint",
    ),
    SamplingCell(
        "no-trolltunga",
        60.1240,
        6.7400,
        "NO",
        "Vestland",
        "Trolltunga",
        "Europe",
        "natural_difficult_viewpoint",
    ),
    SamplingCell(
        "ke-maasai-mara", -1.4931, 35.1439, "KE", "Narok", "Maasai Mara", "Africa", "rural_natural"
    ),
    SamplingCell(
        "nz-wanaka", -44.6700, 169.1300, "NZ", "Otago", "Wanaka", "Oceania", "rural_natural"
    ),
    SamplingCell(
        "ar-perito-moreno",
        -50.4967,
        -73.1377,
        "AR",
        "Santa Cruz",
        "Perito Moreno Glacier",
        "South America",
        "natural_difficult_viewpoint",
    ),
    SamplingCell(
        "cl-atacama",
        -22.9087,
        -68.1997,
        "CL",
        "Antofagasta",
        "San Pedro de Atacama",
        "South America",
        "rural_road",
    ),
    SamplingCell(
        "ma-erg-chebbi",
        31.1450,
        -4.0100,
        "MA",
        "Draa-Tafilalet",
        "Erg Chebbi",
        "Africa",
        "rural_natural",
    ),
    SamplingCell(
        "ca-banff", 51.1784, -115.5708, "CA", "Alberta", "Banff", "North America", "rural_road"
    ),
    SamplingCell(
        "ch-lauterbrunnen", 46.5935, 7.9091, "CH", "Bern", "Lauterbrunnen", "Europe", "rural_road"
    ),
    SamplingCell("in-munnar", 10.0889, 77.0595, "IN", "Kerala", "Munnar", "Asia", "rural_natural"),
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

LICENSES = {
    "CC0": "CC0-1.0",
    "CC0 1.0": "CC0-1.0",
    "CC BY 2.0": "CC-BY-2.0",
    "CC BY 2.5": "CC-BY-2.5",
    "CC BY 3.0": "CC-BY-3.0",
    "CC BY 4.0": "CC-BY-4.0",
    "CC BY-SA 2.0": "CC-BY-SA-2.0",
    "CC BY-SA 2.5": "CC-BY-SA-2.5",
    "CC BY-SA 3.0": "CC-BY-SA-3.0",
    "CC BY-SA 4.0": "CC-BY-SA-4.0",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain(value: object) -> str:
    parser = _TextExtractor()
    parser.feed(str(value))
    parser.close()
    return " ".join(html.unescape(" ".join(parser.parts)).split())


def _validate_url(url: str, hosts: frozenset[str]) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
    ):
        raise RuntimeError("official source URL is outside the allowlist")


class _Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, hosts: frozenset[str]) -> None:
        self.hosts = hosts

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        _validate_url(newurl, self.hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(url: str, hosts: frozenset[str], limit: int) -> bytes:
    _validate_url(url, hosts)
    opener = urllib.request.build_opener(
        _Redirects(hosts), urllib.request.HTTPSHandler(context=ssl.create_default_context())
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    for attempt in range(6):
        try:
            with opener.open(request, timeout=60) as response:  # noqa: S310
                _validate_url(response.geturl(), hosts)
                chunks: list[bytes] = []
                size = 0
                while chunk := response.read(min(1024 * 1024, limit + 1 - size)):
                    size += len(chunk)
                    if size > limit:
                        raise RuntimeError("official response exceeds its size bound")
                    chunks.append(chunk)
                return b"".join(chunks)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 503} or attempt == 5:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                requested = float(retry_after) if retry_after else 0.0
            except ValueError:
                requested = 0.0
            time.sleep(min(30.0, max(requested, float(2 ** (attempt + 1)))))
    raise RuntimeError("official request retries were exhausted")


def _api(params: dict[str, str]) -> dict[str, Any]:
    time.sleep(0.75)
    query = urllib.parse.urlencode(
        {"action": "query", "format": "json", "formatversion": "2", "maxlag": "5", **params}
    )
    return json.loads(_fetch(f"{API_URL}?{query}", API_HOSTS, MAX_API_BYTES))


def _candidates(cell: SamplingCell) -> list[dict[str, Any]]:
    payload = _api(
        {
            "generator": "geosearch",
            "ggsprimary": "all",
            "ggsnamespace": "6",
            "ggsradius": "10000",
            "ggslimit": "100",
            "ggscoord": f"{cell.latitude}|{cell.longitude}",
            "prop": "coordinates|imageinfo",
            "coprimary": "all",
            "iiprop": "url|mime|size|sha1|extmetadata",
            "iiurlwidth": "1280",
        }
    )
    pages = payload.get("query", {}).get("pages", [])
    if not isinstance(pages, list):
        return []
    return sorted(
        (page for page in pages if isinstance(page, dict)),
        key=lambda page: hashlib.sha256(
            f"phase5b-eval-v1|{cell.key}|{page.get('pageid')}".encode()
        ).hexdigest(),
    )


def _accepted(page: dict[str, Any]) -> dict[str, Any] | None:
    infos = page.get("imageinfo")
    coordinates = page.get("coordinates")
    if not isinstance(page.get("pageid"), int) or not isinstance(infos, list) or len(infos) != 1:
        return None
    if not isinstance(coordinates, list) or not coordinates:
        return None
    info = infos[0]
    if (
        info.get("mime") != "image/jpeg"
        or int(info.get("width", 0)) < 640
        or int(info.get("height", 0)) < 480
    ):
        return None
    metadata = info.get("extmetadata")
    if not isinstance(metadata, dict):
        return None
    license_id = LICENSES.get(str(metadata.get("LicenseShortName", {}).get("value", "")))
    artist = _plain(metadata.get("Artist", {}).get("value", ""))
    restrictions = _plain(metadata.get("Restrictions", {}).get("value", ""))
    attribution_required = str(
        metadata.get("AttributionRequired", {}).get("value", "")
    ).casefold()
    thumb_url = info.get("thumburl")
    license_url = str(metadata.get("LicenseUrl", {}).get("value", ""))
    parsed_license = urllib.parse.urlsplit(license_url)
    if (
        license_id is None
        or not artist
        or restrictions
        or attribution_required not in {"true", "false"}
        or (license_id != "CC0-1.0" and attribution_required != "true")
        or not isinstance(thumb_url, str)
        or parsed_license.hostname != "creativecommons.org"
    ):
        return None
    coordinate = coordinates[0]
    latitude = float(coordinate.get("lat"))
    longitude = float(coordinate.get("lon"))
    if (
        not -90 <= latitude <= 90
        or not -180 <= longitude <= 180
        or (latitude == 0 and longitude == 0)
    ):
        return None
    return {
        "pageid": int(page["pageid"]),
        "title": str(page.get("title", "")),
        "latitude": latitude,
        "longitude": longitude,
        "license": license_id,
        "license_url": license_url,
        "artist": artist[:300],
        "thumb_url": thumb_url,
        "source_sha1": str(info.get("sha1", "")),
        "original_width": int(info.get("width", 0)),
        "original_height": int(info.get("height", 0)),
    }


def _derivative(payload: bytes, destination: Path) -> None:
    with Image.open(io.BytesIO(payload)) as opened:
        opened.verify()
    with Image.open(io.BytesIO(payload)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        image.save(destination, "JPEG", quality=92, optimize=True)
    with Image.open(destination) as check:
        if check.getexif():
            raise RuntimeError("evaluation derivative retained EXIF")


def _average_hash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((8, 8), Image.Resampling.LANCZOS).getdata())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return f"{value:016x}"


def main() -> int:
    if MANIFEST_PATH.exists() or SOURCES_PATH.exists():
        raise RuntimeError("the Phase 5B evaluation slice is already frozen")
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, dict[str, Any]] = {}
    if CHECKPOINT_PATH.is_file():
        loaded = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            checkpoint = {
                str(key): value for key, value in loaded.items() if isinstance(value, dict)
            }
    rows: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    used_page_ids: set[int] = set()
    used_hashes: list[int] = []
    for index, cell in enumerate(CELLS, 1):
        selected: dict[str, Any] | None = None
        destination = ASSET_ROOT / f"asset-{index:02d}.jpg"
        cached = checkpoint.get(cell.key)
        if cached is not None and destination.is_file():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest == cached.get("content_sha256"):
                selected = cached
                used_page_ids.add(int(selected["pageid"]))
                used_hashes.append(int(str(selected["perceptual_hash"]), 16))
        pages = [] if selected is not None else _candidates(cell)
        for page in pages:
            record = _accepted(page)
            if record is None or record["pageid"] in used_page_ids:
                continue
            payload = _fetch(record["thumb_url"], IMAGE_HOSTS, MAX_IMAGE_BYTES)
            _derivative(payload, destination)
            perceptual = int(_average_hash(destination), 16)
            if any((perceptual ^ previous).bit_count() <= 4 for previous in used_hashes):
                destination.unlink(missing_ok=True)
                continue
            record["perceptual_hash"] = f"{perceptual:016x}"
            selected = record
            used_page_ids.add(record["pageid"])
            used_hashes.append(perceptual)
            break
        if selected is None:
            raise RuntimeError(f"no reviewed image passed for sampling cell {cell.key}")
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        selected["content_sha256"] = digest
        checkpoint[cell.key] = selected
        CHECKPOINT_PATH.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        page_url = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(
            selected["title"].replace(" ", "_"), safe=":()_,-"
        )
        attribution = (
            f"{selected['artist']}; {selected['license']}; Wikimedia Commons; "
            "local resized metadata-free evaluation derivative"
        )
        rows.append(
            {
                "image_asset_key": f"commons:{selected['pageid']}",
                "local_reference": destination.name,
                "true_latitude": f"{selected['latitude']:.6f}",
                "true_longitude": f"{selected['longitude']:.6f}",
                "country_code": cell.country_code,
                "region": cell.region,
                "city_or_area": cell.city_or_area,
                "continent": cell.continent,
                "source": page_url,
                "source_record_id": str(selected["pageid"]),
                "license": selected["license"],
                "attribution": attribution,
                "split": "test",
                "scene_category": cell.scene_category,
                "geographic_cell": cell.key,
                "capture_family_id": f"commons-file:{selected['pageid']}",
                "content_sha256": digest,
                "perceptual_hash": selected["perceptual_hash"],
            }
        )
        records.append({**asdict(cell), **selected, "page_url": page_url, "content_sha256": digest})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(canonical).hexdigest()
    with MANIFEST_PATH.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    SOURCES_PATH.write_text(
        json.dumps(
            {
                "schema": "phase5b-fixed-evaluation-v1",
                "fingerprint": fingerprint,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": "frozen", "image_count": len(rows), "fingerprint": fingerprint},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
