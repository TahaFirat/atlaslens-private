from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ISO_A2 = re.compile(r"^[A-Z]{2}$")


def _country_code(*values: object) -> str | None:
    for value in values:
        candidate = str(value or "")
        if _ISO_A2.fullmatch(candidate):
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _natural_earth(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    countries: list[dict[str, Any]] = []
    places: list[dict[str, Any]] = []
    for feature in payload["features"]:
        properties = feature["properties"]
        geometry = feature["geometry"]
        if path.name.startswith("ne_110m_admin_0"):
            longitude = float(properties["LABEL_X"])
            latitude = float(properties["LABEL_Y"])
            countries.append(
                {
                    "id": f"ne-country-{int(properties['NE_ID'])}",
                    "kind": "country",
                    "name": str(properties.get("NAME_EN") or properties["ADMIN"]),
                    "country_code": _country_code(
                        properties.get("ISO_A2"), properties.get("ISO_A2_EH")
                    ),
                    "latitude": latitude,
                    "longitude": longitude,
                    "source_record_id": str(properties["NE_ID"]),
                }
            )
        else:
            longitude, latitude = geometry["coordinates"]
            places.append(
                {
                    "id": f"ne-place-{int(properties['ne_id'])}",
                    "kind": "populated_place",
                    "name": str(properties.get("namepar") or properties["name"]),
                    "country_code": _country_code(properties.get("iso_a2")),
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "source_record_id": str(properties["ne_id"]),
                    "population": int(properties.get("pop_max") or 0),
                    "rank": int(properties.get("scalerank") or 0),
                }
            )
    return countries, places


def _turkiye_provinces(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        lines = archive.read("TR.txt").decode("utf-8").splitlines()
    provinces: list[dict[str, Any]] = []
    for line in lines:
        fields = line.split("\t")
        if fields[6:8] != ["A", "ADM1"]:
            continue
        provinces.append(
            {
                "id": f"geonames-adm1-{fields[0]}",
                "kind": "turkiye_province",
                "name": fields[1],
                "country_code": "TR",
                "latitude": float(fields[4]),
                "longitude": float(fields[5]),
                "source_record_id": fields[0],
                "admin1_code": fields[10],
                "source_modified": fields[18],
            }
        )
    if len(provinces) != 81 or len({item["admin1_code"] for item in provinces}) != 81:
        raise ValueError("GeoNames Türkiye extract must contain exactly 81 unique ADM1 rows")
    return provinces


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pinned Phase 6C coordinate catalogue")
    parser.add_argument("--countries", type=Path, required=True)
    parser.add_argument("--places", type=Path, required=True)
    parser.add_argument("--turkiye", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    countries, _ = _natural_earth(args.countries)
    _, places = _natural_earth(args.places)
    provinces = _turkiye_provinces(args.turkiye)
    records = sorted(
        [*countries, *places, *provinces],
        key=lambda item: (item["kind"], item["country_code"] or "", item["name"], item["id"]),
    )
    payload = {
        "schema_version": "atlaslens-coordinate-catalogue-v1",
        "catalogue_version": "natural-earth-5.1.2_geonames-tr-2026-07-14",
        "built_at": datetime.now(UTC).isoformat(),
        "sources": [
            {
                "name": "Natural Earth",
                "version": "5.1.2",
                "license": "public-domain",
                "url": "https://github.com/nvkelso/natural-earth-vector/tree/v5.1.2",
                "files": [
                    {"name": args.countries.name, "sha256": _sha256(args.countries)},
                    {"name": args.places.name, "sha256": _sha256(args.places)},
                ],
            },
            {
                "name": "GeoNames",
                "version": "TR country extract fetched 2026-07-14",
                "license": "CC-BY-4.0",
                "url": "https://download.geonames.org/export/dump/TR.zip",
                "files": [{"name": args.turkiye.name, "sha256": _sha256(args.turkiye)}],
            },
        ],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "countries": len(countries),
                "populated_places": len(places),
                "turkiye_provinces": len(provinces),
                "records": len(records),
                "output_sha256": _sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
