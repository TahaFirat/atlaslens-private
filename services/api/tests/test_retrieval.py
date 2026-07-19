from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from PIL import Image
from sqlalchemy import inspect

import atlaslens_api.retrieval.cli as retrieval_cli
from atlaslens_api.database import AnalysisRow, create_database_engine
from atlaslens_api.retrieval.cli import main as cli_main
from atlaslens_api.retrieval.errors import (
    IndexIntegrityError,
    ManifestValidationError,
    MetadataConflictError,
    ProviderUnavailableError,
)
from atlaslens_api.retrieval.faiss_index import FaissImageIndex, QdrantImageIndex
from atlaslens_api.retrieval.manifest import MANIFEST_COLUMNS, validate_manifest
from atlaslens_api.retrieval.metadata import (
    RetrievalImageRow,
    SQLAlchemyImageMetadataRepository,
)
from atlaslens_api.retrieval.models import Embedding, EmbeddingSpec, ImageMetadata
from atlaslens_api.retrieval.providers import UnavailableEmbeddingProvider
from atlaslens_api.retrieval.service import (
    FaissNearestNeighborEngine,
    ManifestDatasetImporter,
    verify_metadata_consistency,
)


class DeterministicTestEmbeddingProvider:
    """Test-only adapter; production code never fabricates embeddings."""

    spec = EmbeddingSpec(provider="test-only", version="1", dimension=3)
    available = True
    unavailable_reason = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, image_path: Path) -> Embedding:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        self.calls.append(image_path.name)
        return Embedding(self.spec, rgb.mean(axis=(0, 1)))


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path, format="PNG")


def _manifest(path: Path, rows: Sequence[Sequence[str]], *, bom: bool = False) -> None:
    encoding = "utf-8-sig" if bom else "utf-8"
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(rows)


def _row(
    name: str,
    *,
    latitude: str = "41.0082",
    longitude: str = "28.9784",
    country: str = "TR",
    region: str = "Istanbul",
    city: str = "Istanbul",
    license_value: str = "CC BY 4.0",
    source: str = "test fixture",
    notes: str = "not persisted",
) -> list[str]:
    return [
        name,
        latitude,
        longitude,
        country,
        region,
        city,
        license_value,
        source,
        notes,
    ]


def _stack(
    tmp_path: Path, provider: Any | None = None
) -> tuple[
    Any,
    FaissImageIndex,
    SQLAlchemyImageMetadataRepository,
    ManifestDatasetImporter,
    Path,
]:
    provider = provider or DeterministicTestEmbeddingProvider()
    index_dir = tmp_path / "index"
    index = FaissImageIndex(index_dir, provider.spec)
    engine = create_database_engine(f"sqlite:///{(index_dir / 'metadata.sqlite3').as_posix()}")
    repository = SQLAlchemyImageMetadataRepository(engine)
    importer = ManifestDatasetImporter(provider, index, repository)
    return provider, index, repository, importer, index_dir


def _metadata(
    *,
    image_id: str,
    index_id: int,
    content_hash: str,
    provider: str = "test-only",
    version: str = "1",
) -> ImageMetadata:
    return ImageMetadata(
        image_id=UUID(image_id),
        index_id=index_id,
        latitude=41.0,
        longitude=29.0,
        country="TR",
        region=None,
        city="Istanbul",
        source="test fixture",
        license="CC BY 4.0",
        capture_type="user_provided",
        hash=content_hash,
        embedding_provider=provider,
        embedding_version=version,
    )


def test_manifest_accepts_bom_and_validates_every_row_before_mutation(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _image(images / "one.png", (255, 0, 0))
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [_row("one.png"), _row("missing.png")], bom=True)
    provider, _, _, importer, index_dir = _stack(tmp_path)

    with pytest.raises(ManifestValidationError):
        importer.create(manifest, images)

    assert provider.calls == []
    assert not index_dir.exists()


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [("nan", "0"), ("inf", "0"), ("91", "0"), ("0", "-181"), ("bad", "0")],
)
def test_manifest_rejects_invalid_coordinates(
    tmp_path: Path, latitude: str, longitude: str
) -> None:
    images = tmp_path / "images"
    _image(images / "one.png", (255, 0, 0))
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [_row("one.png", latitude=latitude, longitude=longitude)])

    with pytest.raises(ManifestValidationError):
        validate_manifest(manifest, images)


def test_manifest_rejects_duplicate_or_wrong_headers(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_path,latitude,longitude,country,region,city,license,source,source\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError, match="duplicate headers"):
        validate_manifest(manifest, images)


def test_manifest_rejects_non_utf8_input(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"image_path,latitude\n\xff")

    with pytest.raises(ManifestValidationError, match="UTF-8"):
        validate_manifest(manifest, images)


def test_header_only_manifest_is_a_noop_even_when_provider_is_unavailable(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [])
    provider = UnavailableEmbeddingProvider(
        EmbeddingSpec(provider="disabled", version="1", dimension=1), "disabled"
    )
    _, _, _, importer, index_dir = _stack(tmp_path, provider)

    summary = importer.create(manifest, images)

    assert summary.model_dump() == {"validated": 0, "added": 0, "duplicates": 0, "index_size": 0}
    assert not index_dir.exists()


def test_nonempty_import_with_unavailable_provider_creates_no_artifacts(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _image(images / "one.png", (255, 0, 0))
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [_row("one.png")])
    provider = UnavailableEmbeddingProvider(
        EmbeddingSpec(provider="disabled", version="1", dimension=1), "disabled"
    )
    _, _, _, importer, index_dir = _stack(tmp_path, provider)

    with pytest.raises(ProviderUnavailableError):
        importer.create(manifest, images)
    assert not index_dir.exists()


def test_duplicate_is_idempotent_and_conflicting_metadata_is_rejected(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _image(images / "one.png", (255, 0, 0))
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [_row("one.png"), _row("one.png")])
    provider, index, _, importer, _ = _stack(tmp_path)

    first = importer.create(manifest, images)
    second = importer.create(manifest, images)
    assert (first.added, first.duplicates, second.added, second.duplicates) == (1, 1, 0, 2)
    assert index.size == 1
    assert provider.calls == ["one.png"]

    _manifest(manifest, [_row("one.png", city="Ankara")])
    with pytest.raises(MetadataConflictError):
        importer.create(manifest, images)
    assert index.size == 1


def test_incremental_append_does_not_reembed_existing_content(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _image(images / "one.png", (255, 0, 0))
    _image(images / "two.png", (0, 255, 0))
    manifest = tmp_path / "manifest.csv"
    provider, index, repository, importer, index_dir = _stack(tmp_path)
    _manifest(manifest, [_row("one.png")])
    importer.create(manifest, images)
    first_ids = index.ids()

    reloaded = FaissImageIndex.open(index_dir)
    incremental = ManifestDatasetImporter(provider, reloaded, repository)
    _manifest(manifest, [_row("one.png"), _row("two.png", city="Bursa")])
    result = incremental.create(manifest, images)

    assert result.added == 1
    assert first_ids < reloaded.ids()
    assert provider.calls == ["one.png", "two.png"]
    assert FaissImageIndex.open(index_dir).ids() == reloaded.ids()


def test_nearest_neighbor_retrieval_uses_cosine_distance_and_stable_uuid_ties(
    tmp_path: Path,
) -> None:
    _, index, repository, _, index_dir = _stack(tmp_path)
    index_dir.mkdir(parents=True)
    repository.initialize()
    later_uuid = "00000000-0000-0000-0000-000000000002"
    earlier_uuid = "00000000-0000-0000-0000-000000000001"
    rows = [
        _metadata(image_id=later_uuid, index_id=1, content_hash="a" * 64),
        _metadata(image_id=earlier_uuid, index_id=2, content_hash="b" * 64),
        _metadata(
            image_id="00000000-0000-0000-0000-000000000003",
            index_id=3,
            content_hash="c" * 64,
        ),
    ]
    staged, _ = repository.stage(rows)
    repository.activate([row.index_id for row in staged])
    spec = index.spec
    index.append(
        [1, 2, 3],
        [
            Embedding(spec, np.array([1, 0, 0], dtype=np.float32)),
            Embedding(spec, np.array([1, 0, 0], dtype=np.float32)),
            Embedding(spec, np.array([0, 1, 0], dtype=np.float32)),
        ],
    )
    index.persist()

    hits = FaissNearestNeighborEngine(index, repository).retrieve(
        Embedding(spec, np.array([1, 0, 0], dtype=np.float32)), 2
    )

    assert [str(hit.metadata.image_id) for hit in hits] == [earlier_uuid, later_uuid]
    assert [hit.distance for hit in hits] == pytest.approx([0.0, 0.0])
    assert all(hit.provider == spec for hit in hits)


def test_reload_diagnostics_and_metadata_persistence(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _image(images / "one.png", (0, 0, 255))
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [_row("one.png", notes="must not persist")])
    _, index, repository, importer, index_dir = _stack(tmp_path)
    importer.create(manifest, images)

    reloaded = FaissImageIndex.open(index_dir)
    diagnostics = reloaded.verify()
    reopened_repository = SQLAlchemyImageMetadataRepository(
        create_database_engine(f"sqlite:///{(index_dir / 'metadata.sqlite3').as_posix()}")
    )
    verify_metadata_consistency(reloaded, reopened_repository)
    row = next(iter(reopened_repository.get_by_index_ids(list(reloaded.ids())).values()))
    reloaded_hits = FaissNearestNeighborEngine(reloaded, reopened_repository).retrieve(
        Embedding(reloaded.spec, np.array([0, 0, 1], dtype=np.float32)), 1
    )

    assert diagnostics.status == "ready"
    assert diagnostics.index_size == 1
    assert diagnostics.dimension == 3
    assert diagnostics.embedding_provider == "test-only"
    assert diagnostics.storage_size_bytes > 0
    assert row.city == "Istanbul"
    assert reloaded_hits[0].metadata == row
    assert reloaded_hits[0].distance == pytest.approx(0.0)
    assert index.size == reloaded.size
    reopened_repository.close()


def test_metadata_schema_has_no_blob_filename_notes_or_path_columns(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    engine = create_database_engine(f"sqlite:///{(index_dir / 'metadata.sqlite3').as_posix()}")
    repository = SQLAlchemyImageMetadataRepository(engine)
    repository.initialize()
    names = {column["name"] for column in inspect(engine).get_columns("retrieval_images")}

    assert {"image_path", "path", "filename", "blob", "image_blob", "Notes"}.isdisjoint(names)
    assert {
        "image_id",
        "latitude",
        "longitude",
        "country",
        "region",
        "city",
        "source",
        "license",
        "capture_type",
        "hash",
        "embedding_provider",
        "embedding_version",
    } <= names


def test_alembic_migration_creates_retrieval_metadata_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    database = tmp_path / "migrated.sqlite3"
    root = Path(__file__).parents[1]
    config = Config(root / "alembic.ini")
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")

    command.upgrade(config, "head")

    schema = inspect(create_database_engine(f"sqlite:///{database.as_posix()}"))
    assert "retrieval_images" in schema.get_table_names()
    assert schema.get_pk_constraint("retrieval_images")["constrained_columns"] == ["index_id"]


@pytest.mark.parametrize("include_retrieval", [False, True])
def test_alembic_adopts_compatible_pre_migration_runtime_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_retrieval: bool
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    database = tmp_path / "legacy.sqlite3"
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    AnalysisRow.__table__.create(engine)
    if include_retrieval:
        RetrievalImageRow.__table__.create(engine)
    engine.dispose()
    root = Path(__file__).parents[1]
    config = Config(root / "alembic.ini")
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")

    command.upgrade(config, "head")

    migrated = create_database_engine(f"sqlite:///{database.as_posix()}")
    schema = inspect(migrated)
    assert {"alembic_version", "analyses", "retrieval_images"} <= set(schema.get_table_names())
    with migrated.connect() as connection:
        version = connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar()
    assert version == "0009"
    assert "phase5b_diagnostics_json" in {
        column["name"] for column in schema.get_columns("analyses")
    }
    assert {
        "result_classification",
        "simulation_json",
        "provider_comparisons_json",
        "scene_analysis_json",
    } <= {column["name"] for column in schema.get_columns("analyses")}
    migrated.dispose()


def test_same_image_id_can_be_stored_for_multiple_embedding_versions(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    engine = create_database_engine(f"sqlite:///{(index_dir / 'metadata.sqlite3').as_posix()}")
    repository = SQLAlchemyImageMetadataRepository(engine)
    repository.initialize()
    image_id = "00000000-0000-0000-0000-000000000001"
    rows = [
        _metadata(image_id=image_id, index_id=1, content_hash="a" * 64),
        _metadata(
            image_id=image_id,
            index_id=2,
            content_hash="a" * 64,
            provider="future-provider",
            version="2",
        ),
    ]
    staged, duplicates = repository.stage(rows)
    repository.activate([row.index_id for row in staged])

    assert duplicates == 0
    assert repository.active_index_ids() == {1, 2}


def test_index_detects_checksum_corruption(tmp_path: Path) -> None:
    _, index, _, _, index_dir = _stack(tmp_path)
    index.append([1], [Embedding(index.spec, np.array([1, 0, 0], dtype=np.float32))])
    index.persist()
    with (index_dir / "vectors.faiss").open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(IndexIntegrityError):
        FaissImageIndex.open(index_dir)


def test_qdrant_boundary_reports_unavailable() -> None:
    adapter = QdrantImageIndex(EmbeddingSpec(provider="test", version="1", dimension=3))
    assert adapter.diagnostics().status == "unavailable"
    with pytest.raises(ProviderUnavailableError):
        adapter.search(Embedding(adapter.spec, np.array([1, 0, 0], dtype=np.float32)), top_k=1)


def test_cli_info_verify_and_guarded_remove(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    images = tmp_path / "images"
    _image(images / "one.png", (255, 0, 0))
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [_row("one.png")])
    _, _, repository, importer, index_dir = _stack(tmp_path)
    importer.create(manifest, images)
    repository.close()

    assert cli_main(["embeddings", "info", "--index-dir", str(index_dir)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["status"] == "ready"
    assert cli_main(["embeddings", "verify", "--index-dir", str(index_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["index_size"] == 1
    assert cli_main(["embeddings", "remove", "--index-dir", str(index_dir)]) == 2
    assert index_dir.exists()
    capsys.readouterr()
    assert cli_main(["embeddings", "remove", "--index-dir", str(index_dir), "--yes"]) == 0
    assert not index_dir.exists()


def test_cli_unavailable_create_does_not_create_index(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _image(images / "one.png", (255, 0, 0))
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [_row("one.png")])
    index_dir = tmp_path / "index"

    result = cli_main(
        [
            "embeddings",
            "create",
            "--manifest",
            str(manifest),
            "--input-root",
            str(images),
            "--index-dir",
            str(index_dir),
        ]
    )

    assert result == 2
    assert not index_dir.exists()


def test_cli_header_only_create_is_a_successful_noop(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [])
    index_dir = tmp_path / "index"

    result = cli_main(
        [
            "embeddings",
            "create",
            "--manifest",
            str(manifest),
            "--input-root",
            str(images),
            "--index-dir",
            str(index_dir),
        ]
    )

    assert result == 0
    assert not index_dir.exists()


def test_cli_create_reopens_and_appends_to_an_existing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "images"
    _image(images / "one.png", (255, 0, 0))
    _image(images / "two.png", (0, 255, 0))
    manifest = tmp_path / "manifest.csv"
    index_dir = tmp_path / "index"
    provider = DeterministicTestEmbeddingProvider()
    monkeypatch.setattr(retrieval_cli, "production_embedding_provider", lambda _: provider)
    arguments = [
        "embeddings",
        "create",
        "--manifest",
        str(manifest),
        "--input-root",
        str(images),
        "--index-dir",
        str(index_dir),
    ]

    _manifest(manifest, [_row("one.png")])
    assert cli_main(arguments) == 0
    _manifest(manifest, [_row("one.png"), _row("two.png", city="Bursa")])
    assert cli_main(arguments) == 0

    assert FaissImageIndex.open(index_dir).size == 2
    assert provider.calls == ["one.png", "two.png"]


def test_cli_refuses_to_remove_through_directory_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "index.json").write_text("{}", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")

    assert cli_main(["embeddings", "remove", "--index-dir", str(link), "--yes"]) == 2
    assert (target / "index.json").exists()
