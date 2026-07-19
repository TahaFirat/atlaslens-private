from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import AssetSafetyError

_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
}


@dataclass(frozen=True, slots=True)
class ImageIdentity:
    sha256: str
    perceptual_hash: str
    perceptual_hash_algorithm: str
    width_px: int
    height_px: int
    mime_type: str
    size_bytes: int


def sha256_file(path: Path, *, max_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(chunk)
                if total > max_bytes:
                    raise AssetSafetyError("asset_size_limit_exceeded")
                digest.update(chunk)
    except AssetSafetyError:
        raise
    except OSError as exc:
        raise AssetSafetyError("asset_read_failed") from exc
    if total < 1:
        raise AssetSafetyError("asset_empty")
    return digest.hexdigest(), total


def _dhash64(image: Image.Image) -> str:
    """dHash64-v1: EXIF transpose, grayscale, 9x8 Lanczos, left > right bits."""

    normalized = ImageOps.exif_transpose(image).convert("L").resize(
        (9, 8),
        Image.Resampling.LANCZOS,
    )
    pixels = list(normalized.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{value:016x}"


def inspect_image(path: Path, *, max_bytes: int, max_pixels: int) -> ImageIdentity:
    sha256, size_bytes = sha256_file(path, max_bytes=max_bytes)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = image.format or ""
                width, height = image.size
                if width < 1 or height < 1 or width * height > max_pixels:
                    raise AssetSafetyError("asset_pixel_limit_exceeded")
                mime_type = _MIME_BY_FORMAT.get(image_format)
                if mime_type is None:
                    raise AssetSafetyError("asset_image_format_unsupported")
                image.load()
                perceptual_hash = _dhash64(image)
    except AssetSafetyError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise AssetSafetyError("asset_pixel_limit_exceeded") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise AssetSafetyError("asset_image_invalid") from exc
    return ImageIdentity(
        sha256=sha256,
        perceptual_hash=perceptual_hash,
        perceptual_hash_algorithm="dhash64-v1",
        width_px=width,
        height_px=height,
        mime_type=mime_type,
        size_bytes=size_bytes,
    )


def perceptual_hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right) or len(left) % 2 or not left:
        raise ValueError("perceptual hashes must have equal non-empty byte lengths")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("perceptual hashes must be hexadecimal") from exc


__all__ = [
    "ImageIdentity",
    "inspect_image",
    "perceptual_hamming_distance",
    "sha256_file",
]
