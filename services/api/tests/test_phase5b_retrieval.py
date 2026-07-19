from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from atlaslens_api.dataset_acquisition.cli import run_dataset_command
from atlaslens_api.dataset_acquisition.commons import CommonsApiClient, average_hash
from atlaslens_api.dataset_acquisition.models import SamplingCell
from atlaslens_api.dataset_acquisition.validation import (
    EvaluationExclusions,
    LicensedDatasetValidator,
)
from atlaslens_api.model_management.manifest import (
    load_embedding_manifest,
    siglip2_manifest_path,
)
from atlaslens_api.model_management.models import EmbeddingModelManifest
from atlaslens_api.model_management.service import EmbeddingModelManagementService
from atlaslens_api.retrieval.manifest import EXTENDED_MANIFEST_COLUMNS
from atlaslens_api.retrieval.siglip2 import Siglip2EmbeddingProvider


def _installed_embedding_management(tmp_path: Path) -> EmbeddingModelManagementService:
    weights = b"safe-siglip2-test-weights"
    manifest = load_embedding_manifest(siglip2_manifest_path()).model_copy(
        update={"weights_sha256": hashlib.sha256(weights).hexdigest()}
    )

    def fetch(spec: EmbeddingModelManifest, destination: Path) -> str:
        for relative in spec.required_files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(weights if relative == spec.weights_filename else b"{}")
        return spec.revision

    service = EmbeddingModelManagementService(
        tmp_path / "models", manifest, snapshot_fetcher=fetch
    )
    service.install()
    return service


def test_siglip2_manifest_and_receipt_are_pinned_and_tamper_evident(tmp_path: Path) -> None:
    manifest = load_embedding_manifest(siglip2_manifest_path())
    assert manifest.revision == "f775b65a79762255128c981547af89addcfe0f88"
    assert manifest.embedding_dimension == 768
    assert manifest.license == "Apache-2.0"

    service = _installed_embedding_management(tmp_path)
    assert service.info(verify=True).status == "verified"
    (service.snapshot_directory() / "config.json").write_bytes(b"tampered")
    assert service.info(verify=True).status == "invalid"


class _Processor:
    def __call__(self, *, images: list[Image.Image], return_tensors: str) -> dict[str, Any]:
        assert return_tensors == "pt"
        means = [float(np.asarray(image, dtype=np.float32).mean()) for image in images]
        return {
            "pixel_values": torch.tensor(means, dtype=torch.float32)
            .reshape(-1, 1)
            .repeat(1, 768)
        }


class _Model:
    def get_image_features(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values


def test_siglip2_provider_is_lazy_batched_normalized_and_reused(tmp_path: Path) -> None:
    service = _installed_embedding_management(tmp_path)
    loads = 0

    def load(_snapshot: Path, device: str):  # type: ignore[no-untyped-def]
        nonlocal loads
        loads += 1
        assert device == "cpu"
        return _Model(), _Processor(), torch

    provider = Siglip2EmbeddingProvider(
        service,
        requested_device="cpu",
        runtime_loader=load,
        device_selector=lambda _: "cpu",
    )
    assert provider.available is True
    assert provider.loaded is False
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (16, 12), (50, 100, 150)).save(first)
    Image.new("RGB", (16, 12), (10, 20, 30)).save(second)

    embeddings = provider.embed_many([first, second])
    repeated = provider.embed(first)

    assert loads == 1
    assert provider.loaded is True
    assert len(embeddings) == 2
    assert all(item.spec.dimension == 768 for item in embeddings)
    assert all(np.linalg.norm(item.vector) == pytest.approx(1.0) for item in embeddings)
    assert repeated.vector == pytest.approx(embeddings[0].vector)


def _commons_page(title: str, page_id: int) -> dict[str, Any]:
    return {
        "query": {
            "pages": [
                {
                    "pageid": page_id,
                    "title": title,
                    "coordinates": [
                        {
                            "lat": 41.01,
                            "lon": 28.97,
                            "globe": "earth",
                            "country": "TR",
                            "region": "Istanbul",
                        }
                    ],
                    "imageinfo": [
                        {
                            "sha1": f"source{page_id}",
                            "thumburl": "https://upload.wikimedia.org/example.jpg",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY 4.0"},
                                "LicenseUrl": {
                                    "value": "https://creativecommons.org/licenses/by/4.0/"
                                },
                                "Artist": {"value": "<b>Public creator</b>"},
                            },
                        }
                    ],
                }
            ]
        }
    }


def test_commons_discovery_is_metadata_only_deterministic_and_keeps_geography() -> None:
    def transport(_url: str, params: dict[str, str], _user_agent: str) -> dict[str, Any]:
        if params.get("generator") == "geosearch":
            return {
                "query": {
                    "pages": [
                        {"pageid": 2, "title": "File:Two.jpg"},
                        {"pageid": 1, "title": "File:One.jpg"},
                    ]
                }
            }
        title = params["titles"]
        return _commons_page(title, 1 if title.endswith("One.jpg") else 2)

    client = CommonsApiClient(
        user_agent="AtlasLens-test/1 (test@example.invalid)",
        allowed_licenses=frozenset({"CC BY 4.0"}),
        json_transport=transport,
        minimum_request_interval_seconds=0,
    )
    records = client.discover_cell(
        SamplingCell(
            cell_id="tr-istanbul",
            latitude=41.0,
            longitude=29.0,
            radius_m=10_000,
            target=2,
            country="TR",
            region="Istanbul",
            continent="Asia",
        )
    )
    assert [record.page_id for record in records] == [1, 2]
    assert all(record.approved is False for record in records)
    assert all(record.country == "TR" and record.continent == "Asia" for record in records)
    assert records[0].attribution == "Public creator"


def test_extended_dataset_validation_blocks_overlap_and_reports_distribution(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "dataset"
    assets.mkdir()
    image = assets / "reference.png"
    Image.new("RGB", (16, 16), (20, 80, 160)).save(image)
    payload = image.read_bytes()
    perceptual = average_hash(payload)
    row = (
        "reference.png",
        "41.0",
        "29.0",
        "TR",
        "Istanbul",
        "Istanbul",
        "CC BY 4.0",
        "Wikimedia Commons",
        "reviewed",
        "commons:1:source",
        "1",
        "https://commons.wikimedia.org/?curid=1",
        "https://creativecommons.org/licenses/by/4.0/",
        "Public creator",
        "false",
        "commons_reviewed",
        "commons-file:1",
        "object",
        "",
        "",
        "",
        perceptual,
        "mediawiki-api-review-v1",
        "Asia",
        "tr-istanbul",
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(EXTENDED_MANIFEST_COLUMNS)
        writer.writerow(row)
    exclusions = EvaluationExclusions(
        fingerprint="a" * 64,
        content_hashes=frozenset(),
        perceptual_hashes=(),
        capture_families=frozenset(),
    )
    report = LicensedDatasetValidator(
        allowed_licenses=frozenset({"CC BY 4.0"}),
        minimum_count=1,
        required_continents=1,
        minimum_countries=1,
    ).validate(manifest, assets, exclusions)
    assert report.image_count == 1
    assert report.country_counts == {"TR": 1}
    assert report.continent_counts == {"Asia": 1}


def test_kartaview_acquisition_requires_separate_terms_approval() -> None:
    with pytest.raises(RuntimeError, match="kartaview_terms_approval_required"):
        run_dataset_command(
            ["acquire", "kartaview", "--allow-license", "CC BY-SA 4.0"]
        )
