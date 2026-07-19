from __future__ import annotations

import hashlib
import io
import sqlite3
import zipfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from atlaslens_api.gazetteer import (
    ForwardGazetteerManager,
    GazetteerMetadata,
    SQLiteForwardGazetteerResolver,
)
from atlaslens_api.gazetteer.management import (
    FORWARD_ARTIFACTS,
    ArtifactReceipt,
    ArtifactSpec,
)
from atlaslens_api.place_evidence.models import OCRTextObservation
from atlaslens_api.place_evidence.normalization import (
    detect_script,
    normalize_search_text,
    secondary_search_key,
)
from atlaslens_api.place_evidence.service import PlaceEvidenceService


def geonames_row(
    geoname_id: int,
    name: str,
    ascii_name: str,
    alternates: str,
    latitude: float,
    longitude: float,
    country_code: str,
    admin1: str,
    population: int,
    feature_code: str = "PPL",
) -> str:
    fields = [
        str(geoname_id),
        name,
        ascii_name,
        alternates,
        str(latitude),
        str(longitude),
        "P",
        feature_code,
        country_code,
        "",
        admin1,
        "",
        "",
        "",
        str(population),
        "",
        "0",
        "UTC",
        "2026-01-01",
    ]
    return "\t".join(fields) + "\n"


def forward_payloads() -> dict[str, bytes]:
    rows = "".join(
        (
            geonames_row(
                1,
                "İstanbul",
                "Istanbul",
                "Constantinople,İstambul",
                41.01,
                28.97,
                "TR",
                "34",
                15_000_000,
                "PPLC",
            ),
            geonames_row(
                2,
                "Paris",
                "Paris",
                "",
                48.85,
                2.35,
                "FR",
                "11",
                2_000_000,
                "PPLC",
            ),
            geonames_row(
                3,
                "Paris",
                "Paris",
                "",
                33.66,
                -95.55,
                "US",
                "TX",
                25_000,
            ),
        )
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("cities500.txt", rows)
    country_rows = (
        "TR\tTUR\t792\tTU\tTürkiye\tİstanbul\t783562\t85000000\tAS\t.tr\tTRY\tLira\t90\t\t\ttr\t298795\n"
        "FR\tFRA\t250\tFR\tFrance\tParis\t551695\t68000000\tEU\t.fr\tEUR\tEuro\t33\t\t\tfr\t3017382\n"
        "US\tUSA\t840\tUS\tUnited States\tWashington\t9629091\t330000000\tNA\t.us\t"
        "USD\tDollar\t1\t\t\ten\t6252001\n"
    ).encode()
    return {
        "cities500.zip": archive.getvalue(),
        "countryInfo.txt": country_rows,
        "admin1CodesASCII.txt": (
            "TR.34\tİstanbul\tIstanbul\t1\n"
            "FR.11\tÎle-de-France\tIle-de-France\t2\n"
            "US.TX\tTexas\tTexas\t3\n"
        ).encode(),
    }


class FixtureDownloader:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def download(self, spec: ArtifactSpec, destination: Path) -> ArtifactReceipt:
        payload = self.payloads[spec.filename]
        destination.write_bytes(payload)
        return ArtifactReceipt(
            filename=spec.filename,
            url=spec.url,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )


def metadata() -> GazetteerMetadata:
    return GazetteerMetadata(
        dataset="GeoNames cities500 forward search",
        version="atlaslens-geonames-forward-v2",
        source="GeoNames",
        license="CC BY 4.0",
    )


def installed_manager(tmp_path: Path) -> ForwardGazetteerManager:
    manager = ForwardGazetteerManager(
        tmp_path / "gazetteer",
        downloader=FixtureDownloader(forward_payloads()),
        now=lambda: datetime(2026, 7, 11, tzinfo=UTC),
    )
    assert manager.install().status == "ready"
    return manager


def test_forward_v2_installs_separately_and_preserves_v1_directory(tmp_path: Path) -> None:
    root = tmp_path / "gazetteer"
    v1 = root / "geonames-cities15000"
    v1.mkdir(parents=True)
    sentinel = v1 / "do-not-touch"
    sentinel.write_text("v1")
    manager = ForwardGazetteerManager(
        root,
        downloader=FixtureDownloader(forward_payloads()),
        now=lambda: datetime(2026, 7, 11, tzinfo=UTC),
    )
    info = manager.install()
    assert info.schema_version == "atlaslens-geonames-forward-v2"
    assert info.place_count == 3
    assert info.alias_count >= 10
    assert sentinel.read_text() == "v1"
    assert [spec.filename for spec in FORWARD_ARTIFACTS] == [
        "cities500.zip",
        "countryInfo.txt",
        "admin1CodesASCII.txt",
    ]
    assert manager.verify().status == "ready"


def test_forward_lookup_preserves_ambiguity_and_unicode_aliases(tmp_path: Path) -> None:
    manager = installed_manager(tmp_path)
    resolver = SQLiteForwardGazetteerResolver(manager.database_path, metadata())
    istanbul = resolver.search("Istanbul")
    assert istanbul[0].matched_entity == "İstanbul"
    assert istanbul[0].country_code == "TR"
    assert istanbul[0].text_similarity == pytest.approx(1)
    paris = resolver.search("Paris")
    assert len(paris) == 2
    assert {match.country_code for match in paris} == {"FR", "US"}
    assert all(match.ambiguity_count == 2 for match in paris)


def test_country_region_and_domain_suffix_are_structured_aliases(tmp_path: Path) -> None:
    manager = installed_manager(tmp_path)
    resolver = SQLiteForwardGazetteerResolver(manager.database_path, metadata())
    assert resolver.search("Türkiye")[0].match_type == "country"
    assert resolver.search("Ile-de-France")[0].match_type == "region"
    suffix = resolver.search(".tr")
    assert suffix[0].match_type == "domain_suffix"
    assert suffix[0].country_code == "TR"


def test_place_service_uses_real_forward_resolver_without_collapsing_paris(
    tmp_path: Path,
) -> None:
    manager = installed_manager(tmp_path)
    resolver = SQLiteForwardGazetteerResolver(manager.database_path, metadata())
    service = PlaceEvidenceService(resolver, metadata())
    matches = service.resolve(
        (
            OCRTextObservation(
                text="Welcome to Paris Station",
                confidence=0.94,
                script="latin",
                language_hints=("und-Latn",),
            ),
        )
    )
    paris = [match for match in matches if match.matched_entity == "Paris"]
    assert len(paris) == 2
    assert all(match.ambiguity_count == 2 for match in paris)
    assert all(match.evidence_strength < 0.8 for match in paris)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("İSTANBUL", "istanbul"),
        ("Москва", "москва"),
        ("السَّلَام", "السلام"),
        ("東京", "東京"),
    ],
)
def test_multilingual_search_normalization(text: str, expected: str) -> None:
    assert secondary_search_key(text) == expected


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("İstanbul", "latin"),
        ("Москва", "cyrillic"),
        ("القاهرة", "arabic"),
        ("東京", "han"),
        ("東京駅へ", "japanese"),
        ("서울", "hangul"),
    ],
)
def test_script_detection(text: str, script: str) -> None:
    assert detect_script(text) == script
    assert normalize_search_text(text)


def test_forward_database_contains_prefix_and_fts_indexes(tmp_path: Path) -> None:
    manager = installed_manager(tmp_path)
    uri = f"file:{manager.database_path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        objects = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            )
        }
    assert "ix_forward_alias_normalized" in objects
    assert "aliases_fts" in objects
