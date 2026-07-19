from __future__ import annotations

import asyncio
import base64
import io
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from atlaslens_api.config import Settings
from atlaslens_api.errors import AppError
from atlaslens_api.schemas import ImageSummary
from atlaslens_api.storage import LocalImageHandle, LocalTemporaryStorage, StoredUpload

_MIME_BY_FORMAT = {
    "jpeg": {"image/jpeg", "image/jpg"},
    "png": {"image/png"},
    "webp": {"image/webp"},
}
_EXTENSIONS_BY_FORMAT = {
    "jpeg": {".jpg", ".jpeg"},
    "png": {".png"},
    "webp": {".webp"},
}


@dataclass(frozen=True, slots=True)
class PreparedImage:
    original: LocalImageHandle
    normalized: LocalImageHandle
    summary: ImageSummary


@dataclass(frozen=True, slots=True)
class CloudSafeDerivative:
    data_url: str
    byte_count: int
    width: int
    height: int

    def __repr__(self) -> str:
        return "CloudSafeDerivative(<redacted>)"


def detect_signature(path: Path) -> str | None:
    with path.open("rb") as file:
        header = file.read(16)
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    return None


def validate_declared_type(
    detected_format: str | None,
    *,
    content_type: str | None,
    original_filename: str | None,
) -> str:
    if detected_format is None:
        raise AppError(
            415, "unsupported_media_type", "error.unsupported_media", "Unsupported image"
        )
    normalized_mime = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
    if normalized_mime not in _MIME_BY_FORMAT[detected_format]:
        raise AppError(415, "mime_mismatch", "error.mime_mismatch", "Image type mismatch")
    suffix = Path(original_filename or "").suffix.lower()
    if suffix not in _EXTENSIONS_BY_FORMAT[detected_format]:
        raise AppError(
            415,
            "extension_mismatch",
            "error.extension_mismatch",
            "Image extension mismatch",
        )
    return detected_format


class SafeImageProcessor:
    def __init__(self, settings: Settings, storage: LocalTemporaryStorage) -> None:
        self._settings = settings
        self._storage = storage

    async def prepare(
        self,
        upload: StoredUpload,
        *,
        content_type: str | None,
        original_filename: str | None,
    ) -> PreparedImage:
        detected = await asyncio.to_thread(detect_signature, upload.handle.path)
        image_format = validate_declared_type(
            detected,
            content_type=content_type,
            original_filename=original_filename,
        )
        normalized = await self._storage.create_temporary(".png")
        try:
            summary = await asyncio.to_thread(
                self._decode_and_normalize,
                upload.handle.path,
                normalized.path,
                image_format,
                upload.sha256,
            )
        except BaseException:
            await self._storage.delete(normalized.key)
            raise
        return PreparedImage(original=upload.handle, normalized=normalized, summary=summary)

    def _decode_and_normalize(
        self,
        source: Path,
        destination: Path,
        image_format: str,
        sha256: str,
    ) -> ImageSummary:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        Image.MAX_IMAGE_PIXELS = self._settings.max_decoded_pixels
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(source) as probe:
                    actual_format = (probe.format or "").lower()
                    if actual_format == "jpg":
                        actual_format = "jpeg"
                    if actual_format != image_format:
                        raise AppError(
                            415,
                            "decoder_format_mismatch",
                            "error.decoder_format_mismatch",
                            "Image type mismatch",
                        )
                    width, height = probe.size
                    frames = int(getattr(probe, "n_frames", 1))
                    if frames != 1 or bool(getattr(probe, "is_animated", False)):
                        raise AppError(
                            422,
                            "animated_image_not_supported",
                            "error.animated_image_not_supported",
                            "Animated images are not supported",
                        )
                    self._validate_dimensions(width, height)
                    probe.verify()

                with Image.open(source) as decoded:
                    width, height = decoded.size
                    self._validate_dimensions(width, height)
                    exif = decoded.getexif()
                    exif_present = bool(exif)
                    normalized = ImageOps.exif_transpose(decoded)
                    normalized.load()
                    if normalized.mode not in {"RGB", "RGBA", "L"}:
                        normalized = normalized.convert("RGB")
                    normalized.save(destination, format="PNG", optimize=False)
                    normalized_width, normalized_height = normalized.size
        except AppError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise AppError(
                413,
                "decoded_image_too_large",
                "error.decoded_image_too_large",
                "Decoded image too large",
            ) from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise AppError(
                422, "malformed_image", "error.malformed_image", "Malformed image"
            ) from exc

        return ImageSummary(
            format=image_format,
            width=normalized_width,
            height=normalized_height,
            megapixels=round((normalized_width * normalized_height) / 1_000_000, 4),
            sha256=sha256,
            orientation_normalized=True,
            exif_present=exif_present,
        )

    def _validate_dimensions(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise AppError(422, "invalid_dimensions", "error.invalid_dimensions", "Invalid image")
        if (
            width > self._settings.max_image_dimension
            or height > self._settings.max_image_dimension
        ):
            raise AppError(
                413,
                "image_dimension_too_large",
                "error.image_dimension_too_large",
                "Image dimensions too large",
            )
        if width * height > self._settings.max_decoded_pixels:
            raise AppError(
                413,
                "decoded_image_too_large",
                "error.decoded_image_too_large",
                "Decoded image too large",
            )

    async def cloud_derivative(self, normalized: LocalImageHandle) -> CloudSafeDerivative:
        return await asyncio.to_thread(self._cloud_derivative_sync, normalized.path)

    @staticmethod
    def _cloud_derivative_sync(path: Path) -> CloudSafeDerivative:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((1568, 1568), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85, optimize=True, exif=b"")
            payload = buffer.getvalue()
            encoded = base64.b64encode(payload).decode("ascii")
            return CloudSafeDerivative(
                data_url=f"data:image/jpeg;base64,{encoded}",
                byte_count=len(payload),
                width=image.width,
                height=image.height,
            )
