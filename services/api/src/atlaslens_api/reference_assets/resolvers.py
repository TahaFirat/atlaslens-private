from __future__ import annotations

import io
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

_OPAQUE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class ReferenceResolutionError(RuntimeError):
    """Safe resolver error that contains no filesystem path."""


class DatasetManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    relative_path: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=500)
    license: str = Field(min_length=1, max_length=500)
    attribution: str = Field(default="attribution unavailable", min_length=1, max_length=500)
    display_allowed: bool = False


@dataclass(frozen=True, slots=True)
class ReferenceAsset:
    asset_key: str
    content: bytes
    media_type: str
    source: str
    license: str
    attribution: str = "attribution unavailable"
    display_allowed: bool = False

    def __repr__(self) -> str:
        return (
            "ReferenceAsset(asset_key=<opaque>, content=<redacted>, "
            f"media_type={self.media_type!r}, display_allowed={self.display_allowed!r})"
        )


class ReferenceAssetResolver(Protocol):
    available: bool

    def resolve(self, asset_key: str) -> ReferenceAsset: ...


def _validate_key(asset_key: str) -> None:
    if not _OPAQUE_KEY.fullmatch(asset_key):
        raise ReferenceResolutionError("invalid opaque reference key")


def _validated_image(payload: bytes) -> str:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ReferenceResolutionError("reference asset is not a supported image") from exc
    media_type = _MEDIA_TYPES.get(image_format or "")
    if media_type is None:
        raise ReferenceResolutionError("reference asset is not a supported image")
    return media_type


class LicensedDatasetReferenceAssetResolver:
    available = True

    def __init__(
        self,
        dataset_root: Path,
        manifest: Mapping[str, DatasetManifestEntry],
        *,
        max_asset_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        if max_asset_bytes < 1:
            raise ValueError("max_asset_bytes must be positive")
        try:
            self._root = dataset_root.resolve(strict=True)
        except OSError as exc:
            raise ReferenceResolutionError("licensed dataset root is unavailable") from exc
        if not self._root.is_dir():
            raise ReferenceResolutionError("licensed dataset root is unavailable")
        for asset_key in manifest:
            _validate_key(asset_key)
        self._manifest = dict(manifest)
        self._max_asset_bytes = max_asset_bytes

    def resolve(self, asset_key: str) -> ReferenceAsset:
        _validate_key(asset_key)
        entry = self._manifest.get(asset_key)
        if entry is None:
            raise ReferenceResolutionError("reference asset was not found")
        relative = Path(entry.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReferenceResolutionError("reference asset path was rejected")
        try:
            candidate = (self._root / relative).resolve(strict=True)
            candidate.relative_to(self._root)
        except (OSError, ValueError) as exc:
            raise ReferenceResolutionError("reference asset path was rejected") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise ReferenceResolutionError("reference asset is not a regular file")
        try:
            with candidate.open("rb") as stream:
                payload = stream.read(self._max_asset_bytes + 1)
        except OSError as exc:
            raise ReferenceResolutionError("reference asset could not be read") from exc
        if len(payload) > self._max_asset_bytes:
            raise ReferenceResolutionError("reference asset exceeded the size limit")
        media_type = _validated_image(payload)
        return ReferenceAsset(
            asset_key=asset_key,
            content=payload,
            media_type=media_type,
            source=entry.source,
            license=entry.license,
            attribution=entry.attribution,
            display_allowed=entry.display_allowed,
        )


class InMemoryReferenceAssetResolver:
    available = True

    def __init__(self, assets: Mapping[str, ReferenceAsset]) -> None:
        for key, asset in assets.items():
            _validate_key(key)
            if key != asset.asset_key:
                raise ValueError("asset key does not match its manifest key")
            if _validated_image(asset.content) != asset.media_type:
                raise ValueError("asset media type does not match its content")
        self._assets = dict(assets)

    def resolve(self, asset_key: str) -> ReferenceAsset:
        _validate_key(asset_key)
        try:
            return self._assets[asset_key]
        except KeyError as exc:
            raise ReferenceResolutionError("reference asset was not found") from exc


class ObjectStorageReferenceAssetResolver:
    available = False

    def resolve(self, asset_key: str) -> ReferenceAsset:
        _validate_key(asset_key)
        raise ReferenceResolutionError("object-storage reference resolver is unavailable")
