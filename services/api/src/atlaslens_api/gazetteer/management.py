from __future__ import annotations

import csv
import hashlib
import os
import shutil
import sqlite3
import stat
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import IO, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from atlaslens_api.place_evidence.normalization import (
    detect_script,
    normalize_search_text,
    secondary_search_key,
)

OFFICIAL_HOST = "download.geonames.org"
OFFICIAL_BASE_URL = f"https://{OFFICIAL_HOST}/export/dump"
SCHEMA_VERSION = "atlaslens-geonames-v1"
INSTALLATION_NAME = "geonames-cities15000"
FORWARD_SCHEMA_VERSION = "atlaslens-geonames-forward-v2"
FORWARD_INSTALLATION_NAME = "geonames-forward-v2"


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    filename: str
    url: str
    max_bytes: int


ARTIFACTS = (
    ArtifactSpec("cities15000.zip", f"{OFFICIAL_BASE_URL}/cities15000.zip", 100_000_000),
    ArtifactSpec("countryInfo.txt", f"{OFFICIAL_BASE_URL}/countryInfo.txt", 5_000_000),
    ArtifactSpec(
        "admin1CodesASCII.txt",
        f"{OFFICIAL_BASE_URL}/admin1CodesASCII.txt",
        10_000_000,
    ),
)

FORWARD_ARTIFACTS = (
    ArtifactSpec("cities500.zip", f"{OFFICIAL_BASE_URL}/cities500.zip", 150_000_000),
    ArtifactSpec("countryInfo.txt", f"{OFFICIAL_BASE_URL}/countryInfo.txt", 5_000_000),
    ArtifactSpec(
        "admin1CodesASCII.txt",
        f"{OFFICIAL_BASE_URL}/admin1CodesASCII.txt",
        10_000_000,
    ),
)


class ManagementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactReceipt(ManagementModel):
    filename: str
    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)


class InstallationReceipt(ManagementModel):
    schema_version: Literal["atlaslens-geonames-v1"]
    dataset: Literal["GeoNames cities15000"]
    dataset_source: Literal["https://download.geonames.org/export/dump/"]
    license: Literal["CC BY 4.0"]
    attribution: Literal["GeoNames"]
    installed_at: datetime
    place_count: int = Field(gt=0)
    database_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    database_size_bytes: int = Field(gt=0)
    artifacts: tuple[ArtifactReceipt, ...] = Field(min_length=3, max_length=3)


class ForwardInstallationReceipt(ManagementModel):
    schema_version: Literal["atlaslens-geonames-forward-v2"]
    dataset: Literal["GeoNames cities500 forward search"]
    dataset_source: Literal["https://download.geonames.org/export/dump/"]
    license: Literal["CC BY 4.0"]
    attribution: Literal["GeoNames"]
    installed_at: datetime
    place_count: int = Field(gt=0)
    alias_count: int = Field(gt=0)
    database_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    database_size_bytes: int = Field(gt=0)
    artifacts: tuple[ArtifactReceipt, ...] = Field(min_length=3, max_length=3)


class GazetteerInfo(ManagementModel):
    status: Literal["not_installed", "ready", "invalid"]
    dataset: str
    schema_version: str | None = None
    place_count: int = Field(ge=0)
    alias_count: int = Field(default=0, ge=0)
    installed_at: datetime | None = None
    license: str | None = None
    reason_code: str | None = None


class ArtifactDownloader(Protocol):
    def download(self, spec: ArtifactSpec, destination: Path) -> ArtifactReceipt: ...


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        OfficialHTTPSDownloader.validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class OfficialHTTPSDownloader:
    def __init__(self, *, timeout_seconds: float = 60.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("download timeout must be positive")
        self._timeout = timeout_seconds
        self._opener = urllib.request.build_opener(_SafeRedirectHandler())

    @staticmethod
    def validate_url(url: str) -> None:
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != OFFICIAL_HOST
            or parsed.port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or not parsed.path.startswith("/export/dump/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("gazetteer URL is outside the official HTTPS allowlist")

    def download(self, spec: ArtifactSpec, destination: Path) -> ArtifactReceipt:
        self.validate_url(spec.url)
        request = urllib.request.Request(  # noqa: S310 - strict HTTPS allowlist above
            spec.url,
            headers={"User-Agent": "AtlasLens-GeoNames-Installer/1"},
            method="GET",
        )
        digest = hashlib.sha256()
        size = 0
        with self._opener.open(request, timeout=self._timeout) as response:
            final_url = response.geturl()
            self.validate_url(final_url)
            with destination.open("xb") as output:
                while chunk := response.read(64 * 1024):
                    size += len(chunk)
                    if size > spec.max_bytes:
                        raise ValueError("official gazetteer artifact exceeds its size bound")
                    digest.update(chunk)
                    output.write(chunk)
        if size <= 0:
            raise ValueError("official gazetteer artifact is empty")
        return ArtifactReceipt(
            filename=spec.filename,
            url=spec.url,
            sha256=digest.hexdigest(),
            size_bytes=size,
        )


class GazetteerManager:
    def __init__(
        self,
        cache_root: Path,
        *,
        downloader: ArtifactDownloader | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._root = cache_root.expanduser().resolve()
        self._target = self._root / INSTALLATION_NAME
        self._downloader = downloader or OfficialHTTPSDownloader()
        self._now = now or (lambda: datetime.now(UTC))

    @property
    def database_path(self) -> Path:
        return self._target / "places.sqlite3"

    def install(self) -> GazetteerInfo:
        self._root.mkdir(parents=True, exist_ok=True)
        if self._target.exists() or self._target.is_symlink():
            current = self.verify()
            if current.status == "ready":
                return current
            raise ValueError("an invalid gazetteer installation already exists")
        staging = self._root / f".staging-{uuid4().hex}"
        lock = self._root / ".install.lock"
        try:
            with self._exclusive_lock(lock):
                if self._target.exists() or self._target.is_symlink():
                    current = self.verify()
                    if current.status == "ready":
                        return current
                    raise ValueError("an invalid gazetteer installation already exists")
                staging.mkdir(exist_ok=False)
                receipts = tuple(
                    self._downloader.download(spec, staging / spec.filename)
                    for spec in ARTIFACTS
                )
                place_count = self._build_database(staging)
                database_sha256, database_size = self._hash_file(
                    staging / "places.sqlite3"
                )
                receipt = InstallationReceipt(
                    schema_version=SCHEMA_VERSION,
                    dataset="GeoNames cities15000",
                    dataset_source="https://download.geonames.org/export/dump/",
                    license="CC BY 4.0",
                    attribution="GeoNames",
                    installed_at=self._now(),
                    place_count=place_count,
                    database_sha256=database_sha256,
                    database_size_bytes=database_size,
                    artifacts=receipts,
                )
                receipt_path = staging / "receipt.json"
                receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
                self._fsync_file(receipt_path)
                verified = self._verify_directory(staging)
                if verified.status != "ready":
                    raise ValueError("staged gazetteer verification failed")
                os.replace(staging, self._target)
                return verified
        finally:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)

    def verify(self) -> GazetteerInfo:
        if not self._target.exists():
            return GazetteerInfo(
                status="not_installed", dataset="GeoNames cities15000", place_count=0
            )
        if self._target.is_symlink() or not self._target.is_dir():
            return self._invalid("unsafe_installation_path")
        return self._verify_directory(self._target)

    def info(self) -> GazetteerInfo:
        return self.verify()

    def remove(self, *, confirmed: bool) -> GazetteerInfo:
        if not confirmed:
            raise ValueError("gazetteer removal requires explicit confirmation")
        if not self._target.exists():
            return GazetteerInfo(
                status="not_installed", dataset="GeoNames cities15000", place_count=0
            )
        if self._target.is_symlink() or self._target.resolve().parent != self._root:
            raise ValueError("unsafe gazetteer installation path")
        shutil.rmtree(self._target)
        return GazetteerInfo(
            status="not_installed", dataset="GeoNames cities15000", place_count=0
        )

    def _verify_directory(self, directory: Path) -> GazetteerInfo:
        try:
            if directory.is_symlink():
                return self._invalid("unsafe_installation_path")
            receipt = InstallationReceipt.model_validate_json(
                (directory / "receipt.json").read_text(encoding="utf-8")
            )
            expected = {spec.filename: spec.url for spec in ARTIFACTS}
            if {
                artifact.filename: artifact.url for artifact in receipt.artifacts
            } != expected:
                return self._invalid("artifact_manifest_invalid")
            for artifact in receipt.artifacts:
                path = directory / artifact.filename
                if path.is_symlink() or not path.is_file():
                    return self._invalid("artifact_missing")
                digest, size = self._hash_file(path)
                if digest != artifact.sha256 or size != artifact.size_bytes:
                    return self._invalid("artifact_checksum_mismatch")
            database = directory / "places.sqlite3"
            if database.is_symlink() or not database.is_file():
                return self._invalid("database_missing")
            database_digest, database_size = self._hash_file(database)
            if (
                database_digest != receipt.database_sha256
                or database_size != receipt.database_size_bytes
            ):
                return self._invalid("database_checksum_mismatch")
            uri = f"file:{database.resolve().as_posix()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                integrity = connection.execute("PRAGMA integrity_check").fetchone()
                count = int(connection.execute("SELECT COUNT(*) FROM places").fetchone()[0])
            if integrity != ("ok",) or count != receipt.place_count:
                return self._invalid("database_invalid")
        except (OSError, ValueError, sqlite3.Error):
            return self._invalid("receipt_or_database_invalid")
        return GazetteerInfo(
            status="ready",
            dataset=receipt.dataset,
            schema_version=receipt.schema_version,
            place_count=receipt.place_count,
            installed_at=receipt.installed_at,
            license=receipt.license,
        )

    def _build_database(self, staging: Path) -> int:
        countries = self._read_countries(staging / "countryInfo.txt")
        regions = self._read_regions(staging / "admin1CodesASCII.txt")
        archive = staging / "cities15000.zip"
        with zipfile.ZipFile(archive) as zipped:
            entries = [entry for entry in zipped.infolist() if not entry.is_dir()]
            if len(entries) != 1 or entries[0].filename != "cities15000.txt":
                raise ValueError("unexpected GeoNames archive contents")
            entry = entries[0]
            self._validate_zip_entry(entry)
            if entry.file_size > 500_000_000:
                raise ValueError("expanded GeoNames artifact exceeds its size bound")
            database = staging / "places.sqlite3"
            with closing(sqlite3.connect(database)) as connection, zipped.open(entry) as raw:
                self._create_schema(connection)
                count = self._insert_places(connection, raw, countries, regions)
                connection.commit()
        self._fsync_file(database)
        if count <= 0:
            raise ValueError("GeoNames artifact contains no usable places")
        return count

    @staticmethod
    def _validate_zip_entry(entry: zipfile.ZipInfo) -> None:
        path = PurePosixPath(entry.filename)
        unix_mode = (entry.external_attr >> 16) & 0xFFFF
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in entry.filename
            or stat.S_ISLNK(unix_mode)
            or entry.compress_size <= 0
        ):
            raise ValueError("unsafe GeoNames archive entry")

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
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
        connection.execute("CREATE INDEX ix_places_latitude ON places(latitude)")
        connection.execute("CREATE INDEX ix_places_longitude ON places(longitude)")

    @staticmethod
    def _insert_places(
        connection: sqlite3.Connection,
        raw: IO[bytes],
        countries: dict[str, str],
        regions: dict[str, str],
    ) -> int:
        import io

        count = 0
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
        batch: list[tuple[object, ...]] = []
        for row in reader:
            if len(row) < 19:
                raise ValueError("malformed GeoNames city row")
            country_code = row[8].strip().upper()
            name = row[2].strip() or row[1].strip()
            latitude = float(row[4])
            longitude = float(row[5])
            population = max(0, int(row[14] or 0))
            if not name or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError("invalid GeoNames city values")
            batch.append(
                (
                    name,
                    country_code or None,
                    countries.get(country_code),
                    regions.get(f"{country_code}.{row[10].strip()}"),
                    name,
                    latitude,
                    longitude,
                    population,
                )
            )
            if len(batch) >= 2_000:
                connection.executemany("INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                count += len(batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
            count += len(batch)
        return count

    @staticmethod
    def _read_countries(path: Path) -> dict[str, str]:
        countries: dict[str, str] = {}
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    raise ValueError("malformed GeoNames country row")
                countries[fields[0].upper()] = fields[4]
        return countries

    @staticmethod
    def _read_regions(path: Path) -> dict[str, str]:
        regions: dict[str, str] = {}
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    raise ValueError("malformed GeoNames admin row")
                regions[fields[0]] = fields[2] or fields[1]
        return regions

    @staticmethod
    def _hash_file(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            while chunk := source.read(64 * 1024):
                size += len(chunk)
                digest.update(chunk)
        return digest.hexdigest(), size

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("r+b") as file:
            os.fsync(file.fileno())

    @staticmethod
    @contextmanager
    def _exclusive_lock(path: Path) -> Iterator[None]:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            yield
        finally:
            os.close(descriptor)
            path.unlink(missing_ok=True)

    @staticmethod
    def _invalid(reason: str) -> GazetteerInfo:
        return GazetteerInfo(
            status="invalid",
            dataset="GeoNames cities15000",
            place_count=0,
            reason_code=reason,
        )


@dataclass(frozen=True, slots=True)
class _CountryRecord:
    name: str
    capital: str
    tld: str


class ForwardGazetteerManager:
    """Installs a separate forward index without mutating the verified v1 artifact."""

    def __init__(
        self,
        cache_root: Path,
        *,
        downloader: ArtifactDownloader | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._root = cache_root.expanduser().resolve()
        self._target = self._root / FORWARD_INSTALLATION_NAME
        self._downloader = downloader or OfficialHTTPSDownloader()
        self._now = now or (lambda: datetime.now(UTC))

    @property
    def database_path(self) -> Path:
        return self._target / "places-forward.sqlite3"

    def install(self) -> GazetteerInfo:
        self._root.mkdir(parents=True, exist_ok=True)
        if self._target.exists() or self._target.is_symlink():
            current = self.verify()
            if current.status == "ready":
                return current
            raise ValueError("an invalid forward gazetteer installation already exists")
        staging = self._root / f".forward-staging-{uuid4().hex}"
        lock = self._root / ".forward-install.lock"
        try:
            with GazetteerManager._exclusive_lock(lock):
                if self._target.exists() or self._target.is_symlink():
                    current = self.verify()
                    if current.status == "ready":
                        return current
                    raise ValueError("an invalid forward gazetteer installation already exists")
                staging.mkdir(exist_ok=False)
                receipts = tuple(
                    self._downloader.download(spec, staging / spec.filename)
                    for spec in FORWARD_ARTIFACTS
                )
                place_count, alias_count = self._build_database(staging)
                database_sha256, database_size = GazetteerManager._hash_file(
                    staging / "places-forward.sqlite3"
                )
                receipt = ForwardInstallationReceipt(
                    schema_version=FORWARD_SCHEMA_VERSION,
                    dataset="GeoNames cities500 forward search",
                    dataset_source="https://download.geonames.org/export/dump/",
                    license="CC BY 4.0",
                    attribution="GeoNames",
                    installed_at=self._now(),
                    place_count=place_count,
                    alias_count=alias_count,
                    database_sha256=database_sha256,
                    database_size_bytes=database_size,
                    artifacts=receipts,
                )
                receipt_path = staging / "receipt.json"
                receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
                GazetteerManager._fsync_file(receipt_path)
                verified = self._verify_directory(staging)
                if verified.status != "ready":
                    raise ValueError("staged forward gazetteer verification failed")
                os.replace(staging, self._target)
                return verified
        finally:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)

    def verify(self) -> GazetteerInfo:
        if not self._target.exists():
            return self._not_installed()
        if self._target.is_symlink() or not self._target.is_dir():
            return self._invalid("unsafe_installation_path")
        return self._verify_directory(self._target)

    def info(self) -> GazetteerInfo:
        return self.verify()

    def remove(self, *, confirmed: bool) -> GazetteerInfo:
        if not confirmed:
            raise ValueError("forward gazetteer removal requires explicit confirmation")
        if not self._target.exists():
            return self._not_installed()
        if self._target.is_symlink() or self._target.resolve().parent != self._root:
            raise ValueError("unsafe forward gazetteer installation path")
        shutil.rmtree(self._target)
        return self._not_installed()

    def _verify_directory(self, directory: Path) -> GazetteerInfo:
        try:
            if directory.is_symlink():
                return self._invalid("unsafe_installation_path")
            receipt = ForwardInstallationReceipt.model_validate_json(
                (directory / "receipt.json").read_text(encoding="utf-8")
            )
            expected = {spec.filename: spec.url for spec in FORWARD_ARTIFACTS}
            if {
                artifact.filename: artifact.url for artifact in receipt.artifacts
            } != expected:
                return self._invalid("artifact_manifest_invalid")
            for artifact in receipt.artifacts:
                path = directory / artifact.filename
                if path.is_symlink() or not path.is_file():
                    return self._invalid("artifact_missing")
                digest, size = GazetteerManager._hash_file(path)
                if digest != artifact.sha256 or size != artifact.size_bytes:
                    return self._invalid("artifact_checksum_mismatch")
            database = directory / "places-forward.sqlite3"
            if database.is_symlink() or not database.is_file():
                return self._invalid("database_missing")
            database_digest, database_size = GazetteerManager._hash_file(database)
            if (
                database_digest != receipt.database_sha256
                or database_size != receipt.database_size_bytes
            ):
                return self._invalid("database_checksum_mismatch")
            uri = f"file:{database.resolve().as_posix()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                integrity = connection.execute("PRAGMA integrity_check").fetchone()
                place_count = int(connection.execute("SELECT COUNT(*) FROM places").fetchone()[0])
                alias_count = int(connection.execute("SELECT COUNT(*) FROM aliases").fetchone()[0])
            if (
                integrity != ("ok",)
                or place_count != receipt.place_count
                or alias_count != receipt.alias_count
            ):
                return self._invalid("database_invalid")
        except (OSError, ValueError, sqlite3.Error):
            return self._invalid("receipt_or_database_invalid")
        return GazetteerInfo(
            status="ready",
            dataset=receipt.dataset,
            schema_version=receipt.schema_version,
            place_count=receipt.place_count,
            alias_count=receipt.alias_count,
            installed_at=receipt.installed_at,
            license=receipt.license,
        )

    def _build_database(self, staging: Path) -> tuple[int, int]:
        countries = self._read_country_records(staging / "countryInfo.txt")
        regions = GazetteerManager._read_regions(staging / "admin1CodesASCII.txt")
        archive = staging / "cities500.zip"
        with zipfile.ZipFile(archive) as zipped:
            entries = [entry for entry in zipped.infolist() if not entry.is_dir()]
            if len(entries) != 1 or entries[0].filename != "cities500.txt":
                raise ValueError("unexpected GeoNames forward archive contents")
            entry = entries[0]
            GazetteerManager._validate_zip_entry(entry)
            if entry.file_size > 1_000_000_000:
                raise ValueError("expanded GeoNames forward artifact exceeds its size bound")
            database = staging / "places-forward.sqlite3"
            with closing(sqlite3.connect(database)) as connection, zipped.open(entry) as raw:
                self._create_schema(connection)
                place_count = self._insert_places(connection, raw, countries, regions)
                self._insert_country_and_region_aliases(connection, countries, regions)
                connection.execute(
                    "INSERT INTO aliases_fts(rowid, normalized_name) "
                    "SELECT alias_id, normalized_name FROM aliases"
                )
                alias_count = int(connection.execute("SELECT COUNT(*) FROM aliases").fetchone()[0])
                connection.commit()
        GazetteerManager._fsync_file(database)
        if place_count <= 0 or alias_count <= 0:
            raise ValueError("GeoNames forward artifact contains no usable places")
        return place_count, alias_count

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE places (
                geoname_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                ascii_name TEXT NOT NULL,
                country_code TEXT,
                country TEXT,
                region TEXT,
                city TEXT,
                admin1_code TEXT,
                feature_class TEXT NOT NULL,
                feature_code TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                population INTEGER NOT NULL
            );
            CREATE TABLE aliases (
                alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
                place_id INTEGER NOT NULL REFERENCES places(geoname_id),
                alias TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                secondary_key TEXT NOT NULL,
                script TEXT NOT NULL,
                language_hint TEXT,
                match_type TEXT NOT NULL,
                UNIQUE(place_id, normalized_name, match_type)
            );
            CREATE INDEX ix_forward_places_country_admin
                ON places(country_code, admin1_code, population DESC);
            CREATE INDEX ix_forward_alias_normalized ON aliases(normalized_name);
            CREATE INDEX ix_forward_alias_secondary ON aliases(secondary_key);
            CREATE VIRTUAL TABLE aliases_fts USING fts5(
                normalized_name,
                content='aliases',
                content_rowid='alias_id',
                tokenize='unicode61 remove_diacritics 0'
            );
            """
        )

    @classmethod
    def _insert_places(
        cls,
        connection: sqlite3.Connection,
        raw: IO[bytes],
        countries: dict[str, _CountryRecord],
        regions: dict[str, str],
    ) -> int:
        import io

        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
        count = 0
        for row in reader:
            if len(row) < 19:
                raise ValueError("malformed GeoNames forward city row")
            geoname_id = int(row[0])
            canonical_name = row[1].strip() or row[2].strip()
            ascii_name = row[2].strip() or canonical_name
            latitude = float(row[4])
            longitude = float(row[5])
            country_code = row[8].strip().upper()
            admin1_code = row[10].strip()
            population = max(0, int(row[14] or 0))
            if (
                geoname_id <= 0
                or not canonical_name
                or not -90 <= latitude <= 90
                or not -180 <= longitude <= 180
            ):
                raise ValueError("invalid GeoNames forward city values")
            country = countries.get(country_code)
            region = regions.get(f"{country_code}.{admin1_code}")
            connection.execute(
                "INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    geoname_id,
                    canonical_name[:200],
                    ascii_name[:200],
                    country_code or None,
                    country.name if country else None,
                    region,
                    canonical_name[:160],
                    admin1_code or None,
                    row[6].strip()[:1],
                    row[7].strip()[:10],
                    latitude,
                    longitude,
                    population,
                ),
            )
            match_type = "district" if row[7].strip() == "PPLX" else "city"
            aliases = [canonical_name, ascii_name, *row[3].split(",")]
            for alias in list(
                dict.fromkeys(item.strip() for item in aliases if item.strip())
            )[:64]:
                cls._insert_alias(connection, geoname_id, alias, match_type=match_type)
            count += 1
        return count

    @classmethod
    def _insert_country_and_region_aliases(
        cls,
        connection: sqlite3.Connection,
        countries: dict[str, _CountryRecord],
        regions: dict[str, str],
    ) -> None:
        for country_code, country in countries.items():
            capital_key = normalize_search_text(country.capital)
            row = connection.execute(
                """
                SELECT p.geoname_id
                FROM places AS p
                LEFT JOIN aliases AS a ON a.place_id = p.geoname_id
                WHERE p.country_code = ?
                ORDER BY CASE WHEN p.feature_code = 'PPLC' THEN 0
                              WHEN a.normalized_name = ? THEN 1 ELSE 2 END,
                         p.population DESC, p.geoname_id ASC
                LIMIT 1
                """,
                (country_code, capital_key),
            ).fetchone()
            if row is None:
                continue
            place_id = int(row[0])
            cls._insert_alias(connection, place_id, country.name, match_type="country")
            if country.tld:
                cls._insert_alias(
                    connection, place_id, f".{country.tld.lstrip('.')}", match_type="domain_suffix"
                )
        for combined_code, region_name in regions.items():
            if "." not in combined_code:
                continue
            country_code, admin1_code = combined_code.split(".", 1)
            row = connection.execute(
                """
                SELECT geoname_id FROM places
                WHERE country_code = ? AND admin1_code = ?
                ORDER BY CASE WHEN feature_code LIKE 'PPLA%' THEN 0 ELSE 1 END,
                         population DESC, geoname_id ASC
                LIMIT 1
                """,
                (country_code, admin1_code),
            ).fetchone()
            if row is not None:
                cls._insert_alias(
                    connection, int(row[0]), region_name, match_type="region"
                )

    @staticmethod
    def _insert_alias(
        connection: sqlite3.Connection, place_id: int, alias: str, *, match_type: str
    ) -> None:
        alias = alias.strip()[:200]
        normalized = normalize_search_text(alias)
        if len(normalized) < 2:
            return
        connection.execute(
            """
            INSERT OR IGNORE INTO aliases(
                place_id, alias, normalized_name, secondary_key,
                script, language_hint, match_type
            ) VALUES (?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                place_id,
                alias,
                normalized,
                secondary_search_key(normalized),
                detect_script(alias),
                match_type,
            ),
        )

    @staticmethod
    def _read_country_records(path: Path) -> dict[str, _CountryRecord]:
        countries: dict[str, _CountryRecord] = {}
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 10:
                    raise ValueError("malformed GeoNames forward country row")
                countries[fields[0].upper()] = _CountryRecord(
                    name=fields[4].strip(),
                    capital=fields[5].strip(),
                    tld=fields[9].strip(),
                )
        return countries

    @staticmethod
    def _not_installed() -> GazetteerInfo:
        return GazetteerInfo(
            status="not_installed", dataset="GeoNames cities500 forward search", place_count=0
        )

    @staticmethod
    def _invalid(reason: str) -> GazetteerInfo:
        return GazetteerInfo(
            status="invalid",
            dataset="GeoNames cities500 forward search",
            place_count=0,
            reason_code=reason,
        )
