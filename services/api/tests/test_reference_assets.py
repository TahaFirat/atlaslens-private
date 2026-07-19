from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from atlaslens_api.reference_assets import (
    DatasetManifestEntry,
    InMemoryReferenceAssetResolver,
    LicensedDatasetReferenceAssetResolver,
    ObjectStorageReferenceAssetResolver,
    ReferenceAsset,
    ReferenceResolutionError,
)


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 20, 30)).save(buffer, "JPEG")
    return buffer.getvalue()


def entry(relative_path: str, *, display_allowed: bool = False) -> DatasetManifestEntry:
    return DatasetManifestEntry(
        relative_path=relative_path,
        source="licensed-fixture",
        license="CC0-1.0",
        display_allowed=display_allowed,
    )


def test_licensed_resolver_reads_valid_image_and_denies_display_by_default(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "reference.jpg").write_bytes(jpeg_bytes())
    resolver = LicensedDatasetReferenceAssetResolver(root, {"asset:001": entry("reference.jpg")})
    asset = resolver.resolve("asset:001")
    assert asset.media_type == "image/jpeg"
    assert asset.display_allowed is False
    assert asset.content.startswith(b"\xff\xd8")
    assert str(root) not in repr(asset)


def test_display_requires_explicit_manifest_permission(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "reference.jpg").write_bytes(jpeg_bytes())
    resolver = LicensedDatasetReferenceAssetResolver(
        root, {"asset:shown": entry("reference.jpg", display_allowed=True)}
    )
    assert resolver.resolve("asset:shown").display_allowed is True


@pytest.mark.parametrize("relative_path", ["../escape.jpg", "sub/../../escape.jpg"])
def test_traversal_is_rejected_without_path_disclosure(tmp_path: Path, relative_path: str) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    resolver = LicensedDatasetReferenceAssetResolver(root, {"asset:bad": entry(relative_path)})
    with pytest.raises(ReferenceResolutionError) as error:
        resolver.resolve("asset:bad")
    assert str(tmp_path) not in str(error.value)


def test_symlink_escape_is_rejected_where_supported(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "reference.jpg").write_bytes(jpeg_bytes())
    link = root / "escape.jpg"
    try:
        link.symlink_to(outside / "reference.jpg")
    except OSError:
        pytest.skip("host does not permit symlink creation")
    resolver = LicensedDatasetReferenceAssetResolver(root, {"asset:escape": entry("escape.jpg")})
    with pytest.raises(ReferenceResolutionError, match="path was rejected"):
        resolver.resolve("asset:escape")


def test_non_image_and_oversized_assets_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "bad.jpg").write_bytes(b"not an image")
    invalid = LicensedDatasetReferenceAssetResolver(root, {"asset:bad": entry("bad.jpg")})
    with pytest.raises(ReferenceResolutionError, match="supported image"):
        invalid.resolve("asset:bad")

    (root / "large.jpg").write_bytes(jpeg_bytes())
    oversized = LicensedDatasetReferenceAssetResolver(
        root, {"asset:large": entry("large.jpg")}, max_asset_bytes=10
    )
    with pytest.raises(ReferenceResolutionError, match="size limit"):
        oversized.resolve("asset:large")


def test_in_memory_and_future_object_storage_boundaries() -> None:
    asset = ReferenceAsset(
        asset_key="asset:test",
        content=jpeg_bytes(),
        media_type="image/jpeg",
        source="generated-test",
        license="CC0-1.0",
    )
    memory = InMemoryReferenceAssetResolver({"asset:test": asset})
    assert memory.resolve("asset:test") is asset
    with pytest.raises(ReferenceResolutionError, match="not found"):
        memory.resolve("asset:missing")

    object_storage = ObjectStorageReferenceAssetResolver()
    assert object_storage.available is False
    with pytest.raises(ReferenceResolutionError, match="unavailable"):
        object_storage.resolve("asset:test")


def test_opaque_keys_are_validated(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    with pytest.raises(ReferenceResolutionError, match="opaque"):
        LicensedDatasetReferenceAssetResolver(root, {"../secret": entry("reference.jpg")})
