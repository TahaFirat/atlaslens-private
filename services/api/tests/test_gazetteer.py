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
    CoordinateFallbackResolver,
    GazetteerMetadata,
    SQLiteGazetteerResolver,
)
from atlaslens_api.gazetteer.cli import main as gazetteer_main
from atlaslens_api.gazetteer.management import (
    ARTIFACTS,
    ArtifactReceipt,
    ArtifactSpec,
    GazetteerInfo,
    GazetteerManager,
    OfficialHTTPSDownloader,
)


def create_database(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """
            CREATE TABLE places (
                name TEXT NOT NULL,
                country_code TEXT,
                country TEXT,
                region TEXT,
                city TEXT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                population INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "Fixture City",
                "TR",
                "Fixture Country",
                "Fixture Region",
                "Fixture City",
                41.0,
                29.0,
                100,
            ),
        )
        connection.commit()


def metadata() -> GazetteerMetadata:
    return GazetteerMetadata(
        dataset="GeoNames-compatible fixture",
        version="test-v1",
        source="offline-test-fixture",
        license="CC-BY-4.0-test-only",
    )


def test_offline_sqlite_resolver_returns_nearest_attributed_place(tmp_path: Path) -> None:
    database = tmp_path / "places.sqlite3"
    create_database(database)
    resolved = SQLiteGazetteerResolver(database, metadata()).resolve(41.01, 29.01)
    assert resolved.label == "Fixture City"
    assert resolved.country_code == "TR"
    assert resolved.distance_km > 0
    assert resolved.source == "offline-test-fixture"
    assert resolved.dataset_version == "test-v1"


def test_resolver_uses_coordinate_fallback_without_inventing_place(tmp_path: Path) -> None:
    database = tmp_path / "places.sqlite3"
    create_database(database)
    resolved = SQLiteGazetteerResolver(
        database, metadata(), search_radius_km=5
    ).resolve(-20.0, -20.0)
    assert resolved.label == "-20.000, -20.000"
    assert resolved.country_code is None
    assert resolved.source == "coordinate_fallback"


def test_coordinate_fallback_is_deterministic() -> None:
    resolver = CoordinateFallbackResolver()
    assert resolver.resolve(1.23456, -2.34567) == resolver.resolve(1.23456, -2.34567)


def geonames_city_row() -> str:
    fields = [
        "1",
        "Fixture City",
        "Fixture City",
        "",
        "41.0",
        "29.0",
        "P",
        "PPL",
        "TR",
        "",
        "34",
        "",
        "",
        "",
        "15000",
        "",
        "0",
        "Europe/Istanbul",
        "2026-01-01",
    ]
    return "\t".join(fields) + "\n"


def artifact_payloads(*, unsafe_archive: bool = False) -> dict[str, bytes]:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        name = "../cities15000.txt" if unsafe_archive else "cities15000.txt"
        zipped.writestr(name, geonames_city_row())
    return {
        "cities15000.zip": archive.getvalue(),
        "countryInfo.txt": b"TR\tTUR\t792\tTU\tFixture Country\n",
        "admin1CodesASCII.txt": b"TR.34\tFixture Region\tFixture Region\t1\n",
    }


class FixtureDownloader:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.urls: list[str] = []

    def download(self, spec: ArtifactSpec, destination: Path) -> ArtifactReceipt:
        self.urls.append(spec.url)
        payload = self.payloads[spec.filename]
        destination.write_bytes(payload)
        return ArtifactReceipt(
            filename=spec.filename,
            url=spec.url,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )


def test_manager_installs_verifies_and_removes_offline_fixture(tmp_path: Path) -> None:
    downloader = FixtureDownloader(artifact_payloads())
    manager = GazetteerManager(
        tmp_path / "cache",
        downloader=downloader,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    installed = manager.install()
    assert installed.status == "ready"
    assert installed.place_count == 1
    assert downloader.urls == [spec.url for spec in ARTIFACTS]
    assert manager.verify().status == "ready"
    assert manager.info().schema_version == "atlaslens-geonames-v1"
    resolved = SQLiteGazetteerResolver(manager.database_path, metadata()).resolve(41.0, 29.0)
    assert resolved.label == "Fixture City"

    with pytest.raises(ValueError, match="confirmation"):
        manager.remove(confirmed=False)
    assert manager.remove(confirmed=True).status == "not_installed"
    assert not manager.database_path.exists()


def test_verify_detects_artifact_and_database_tampering(tmp_path: Path) -> None:
    manager = GazetteerManager(
        tmp_path / "artifact-cache", downloader=FixtureDownloader(artifact_payloads())
    )
    manager.install()
    (manager.database_path.parent / "countryInfo.txt").write_bytes(b"tampered")
    assert manager.verify().reason_code == "artifact_checksum_mismatch"

    database_manager = GazetteerManager(
        tmp_path / "database-cache", downloader=FixtureDownloader(artifact_payloads())
    )
    database_manager.install()
    with closing(sqlite3.connect(database_manager.database_path)) as connection:
        connection.execute("UPDATE places SET name = 'Changed'")
        connection.commit()
    assert database_manager.verify().reason_code == "database_checksum_mismatch"


def test_unsafe_archive_is_rejected_and_staging_is_cleaned(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    manager = GazetteerManager(
        root, downloader=FixtureDownloader(artifact_payloads(unsafe_archive=True))
    )
    with pytest.raises(ValueError, match="archive"):
        manager.install()
    assert manager.info().status == "not_installed"
    assert not list(root.glob(".staging-*"))
    assert not (root / ".install.lock").exists()


@pytest.mark.parametrize(
    "url",
    [
        "http://download.geonames.org/export/dump/cities15000.zip",
        "https://example.com/export/dump/cities15000.zip",
        "https://download.geonames.org/other/cities15000.zip",
        "https://download.geonames.org/export/dump/cities15000.zip?mirror=1",
    ],
)
def test_official_downloader_rejects_urls_outside_allowlist(url: str) -> None:
    with pytest.raises(ValueError, match="allowlist"):
        OfficialHTTPSDownloader.validate_url(url)


class StubManager:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root

    def info(self) -> GazetteerInfo:
        return GazetteerInfo(
            status="ready",
            dataset="GeoNames cities15000",
            schema_version="atlaslens-geonames-v1",
            place_count=1,
            license="CC BY 4.0",
        )


def test_cli_info_json_is_safe_and_path_free(tmp_path: Path, capsys) -> None:
    exit_code = gazetteer_main(
        ["--cache-root", str(tmp_path), "--json", "info"],
        manager_factory=StubManager,
    )
    rendered = capsys.readouterr().out
    assert exit_code == 0
    assert '"status":"ready"' in rendered
    assert str(tmp_path) not in rendered
