"""In-memory, metadata-free image preparation for NVIDIA vision requests."""

from __future__ import annotations

import base64
import hashlib
import io
import warnings
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import NvidiaPreprocessingError

CropKind = Literal["full", "centre", "upper", "lower"]

_MAX_SOURCE_BYTES = 25 * 1024 * 1024
_MAX_DECODED_PIXELS = 40_000_000
# NVIDIA's hosted endpoint requires images larger than 180 KB to use a separate
# asset-upload API.  Phase 3C1 deliberately avoids that extra persistence surface.
_MAX_DERIVATIVE_BYTES = 180_000
_MAX_DERIVATIVE_SET_BYTES = 4 * _MAX_DERIVATIVE_BYTES
_SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
_MEDIA_TYPES = {
    "JPEG": frozenset({"image/jpeg", "image/jpg"}),
    "PNG": frozenset({"image/png"}),
    "WEBP": frozenset({"image/webp"}),
}
_CROP_ORDER: tuple[CropKind, ...] = ("full", "centre", "upper", "lower")
_APP_MARKERS = frozenset(range(0xE0, 0xF0))
_STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD8)})


def _jpeg_has_metadata_or_is_malformed(payload: bytes) -> bool:
    """Parse JPEG markers and reject every APP/COM segment and malformed tail."""

    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        return True
    position = 2
    in_scan = False
    while position < len(payload):
        if in_scan:
            marker_start = payload.find(b"\xff", position)
            if marker_start < 0:
                return True
            position = marker_start
        elif payload[position] != 0xFF:
            return True

        while position < len(payload) and payload[position] == 0xFF:
            position += 1
        if position >= len(payload):
            return True
        marker = payload[position]
        position += 1

        if in_scan and marker == 0x00:
            continue
        if marker in _STANDALONE_MARKERS:
            continue
        in_scan = False
        if marker in _APP_MARKERS or marker == 0xFE:
            return True
        if marker == 0xD9:
            return position != len(payload)
        if marker == 0xD8 or position + 2 > len(payload):
            return True
        segment_length = int.from_bytes(payload[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(payload):
            return True
        position += segment_length
        if marker == 0xDA:
            in_scan = True
    return True


def _strip_jpeg_header_metadata(payload: bytes) -> bytes:
    """Remove encoder APP/COM headers before constructing a guarded derivative."""

    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        raise NvidiaPreprocessingError("nvidia_image_derivative_invalid")
    output = bytearray(payload[:2])
    position = 2
    while position < len(payload):
        segment_start = position
        if payload[position] != 0xFF:
            raise NvidiaPreprocessingError("nvidia_image_derivative_invalid")
        while position < len(payload) and payload[position] == 0xFF:
            position += 1
        if position >= len(payload):
            raise NvidiaPreprocessingError("nvidia_image_derivative_invalid")
        marker = payload[position]
        position += 1
        if marker == 0xDA:
            output.extend(payload[segment_start:])
            return bytes(output)
        if marker == 0xD9:
            output.extend(payload[segment_start:position])
            if position != len(payload):
                raise NvidiaPreprocessingError("nvidia_image_derivative_invalid")
            return bytes(output)
        if marker in _STANDALONE_MARKERS:
            output.extend(payload[segment_start:position])
            continue
        if marker == 0xD8 or position + 2 > len(payload):
            raise NvidiaPreprocessingError("nvidia_image_derivative_invalid")
        segment_length = int.from_bytes(payload[position : position + 2], "big")
        segment_end = position + segment_length
        if segment_length < 2 or segment_end > len(payload):
            raise NvidiaPreprocessingError("nvidia_image_derivative_invalid")
        if marker not in _APP_MARKERS and marker != 0xFE:
            output.extend(payload[segment_start:segment_end])
        position = segment_end
    raise NvidiaPreprocessingError("nvidia_image_derivative_invalid")


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """One derivative; encoded bytes are intentionally non-representable."""

    kind: CropKind
    jpeg_bytes: bytes = field(repr=False)
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.kind not in _CROP_ORDER or not self.jpeg_bytes:
            raise NvidiaPreprocessingError("nvidia_image_derivative_invalid")
        if len(self.jpeg_bytes) > _MAX_DERIVATIVE_BYTES or self.width <= 0 or self.height <= 0:
            raise NvidiaPreprocessingError("nvidia_image_derivative_invalid")
        try:
            with Image.open(io.BytesIO(self.jpeg_bytes)) as probe:
                if (
                    (probe.format or "").upper() != "JPEG"
                    or probe.mode != "RGB"
                    or probe.size != (self.width, self.height)
                    or int(getattr(probe, "n_frames", 1)) != 1
                    or bool(probe.getexif())
                    or bool(probe.info.get("icc_profile"))
                    or _jpeg_has_metadata_or_is_malformed(self.jpeg_bytes)
                ):
                    raise NvidiaPreprocessingError("nvidia_image_derivative_invalid")
                probe.verify()
        except NvidiaPreprocessingError:
            raise
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            raise NvidiaPreprocessingError("nvidia_image_derivative_invalid") from None

    @property
    def byte_count(self) -> int:
        return len(self.jpeg_bytes)

    def data_url(self) -> str:
        """Build the ephemeral OpenAI-compatible data URL in memory."""

        encoded = base64.b64encode(self.jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def __repr__(self) -> str:
        return (
            f"PreparedImage(kind={self.kind!r}, jpeg_bytes=<redacted>, "
            f"width={self.width}, height={self.height})"
        )


@dataclass(frozen=True, slots=True)
class PreparedImageSet:
    """Derivatives from exactly one source image and its internal cache identity."""

    images: tuple[PreparedImage, ...]
    _source_sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        kinds = tuple(image.kind for image in self.images)
        _validate_crop_kinds(kinds)
        if len(self.images) > 4 or self.total_bytes > _MAX_DERIVATIVE_SET_BYTES:
            raise NvidiaPreprocessingError("nvidia_image_derivative_set_too_large")
        if len(self._source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self._source_sha256
        ):
            raise NvidiaPreprocessingError("nvidia_image_identity_invalid")

    @property
    def cache_identity_sha256(self) -> str:
        """Return the full local identity only to internal cache/evaluation callers."""

        return self._source_sha256

    @property
    def total_bytes(self) -> int:
        return sum(image.byte_count for image in self.images)

    def __repr__(self) -> str:
        return (
            "PreparedImageSet(images="
            f"{self.images!r}, source_sha256=<redacted>, total_bytes={self.total_bytes})"
        )


def _validate_crop_kinds(crop_kinds: tuple[CropKind, ...]) -> None:
    if not crop_kinds or crop_kinds[0] != "full":
        raise NvidiaPreprocessingError("nvidia_crop_policy_invalid")
    selected = set(crop_kinds)
    expected = tuple(kind for kind in _CROP_ORDER if kind in selected)
    if len(selected) != len(crop_kinds) or crop_kinds != expected:
        raise NvidiaPreprocessingError("nvidia_crop_policy_invalid")


def _crop_box(kind: CropKind, width: int, height: int) -> tuple[int, int, int, int]:
    if kind == "full":
        return 0, 0, width, height
    if kind == "centre":
        return (
            width * 15 // 100,
            height * 15 // 100,
            max(width * 85 // 100, 1),
            max(height * 85 // 100, 1),
        )
    if kind == "upper":
        return 0, 0, width, max(height * 60 // 100, 1)
    return 0, height * 45 // 100, width, height


def _encode_derivative(
    source: Image.Image,
    *,
    kind: CropKind,
    maximum_edge: int,
    jpeg_quality: int,
) -> PreparedImage:
    selected = source.crop(_crop_box(kind, source.width, source.height))
    edges = tuple(
        dict.fromkeys(
            max(256, min(maximum_edge, edge))
            for edge in (maximum_edge, 1_280, 1_024, 896, 768, 640, 512, 384, 256)
        )
    )
    qualities = tuple(
        dict.fromkeys(min(jpeg_quality, quality) for quality in (jpeg_quality, 78, 72, 65, 58, 50))
    )
    attempted_sizes: set[tuple[int, int]] = set()
    for edge in edges:
        candidate = selected.copy()
        candidate.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        if candidate.size in attempted_sizes:
            continue
        attempted_sizes.add(candidate.size)
        for quality in qualities:
            output = io.BytesIO()
            candidate.save(
                output,
                format="JPEG",
                quality=quality,
                optimize=True,
                exif=b"",
                icc_profile=None,
            )
            encoded = _strip_jpeg_header_metadata(output.getvalue())
            if encoded and len(encoded) <= _MAX_DERIVATIVE_BYTES:
                return PreparedImage(
                    kind=kind,
                    jpeg_bytes=encoded,
                    width=candidate.width,
                    height=candidate.height,
                )
    raise NvidiaPreprocessingError("nvidia_image_derivative_too_large")


def prepare_nvidia_images(
    payload: bytes,
    *,
    declared_media_type: str | None = None,
    maximum_edge: int = 1_280,
    jpeg_quality: int = 82,
    crop_kinds: tuple[CropKind, ...] = ("full",),
) -> PreparedImageSet:
    """Decode one image, orient it and create bounded EXIF-free JPEGs in memory.

    The full source SHA-256 is retained only inside the returned object for an
    internal cache/evaluation identity.  It is never placed in the cloud payload.
    """

    if not payload or len(payload) > _MAX_SOURCE_BYTES:
        raise NvidiaPreprocessingError("nvidia_image_source_size_invalid")
    if not 256 <= maximum_edge <= 1_600 or not 50 <= jpeg_quality <= 90:
        raise NvidiaPreprocessingError("nvidia_image_policy_invalid")
    _validate_crop_kinds(crop_kinds)
    normalized_media_type = declared_media_type.casefold().strip() if declared_media_type else None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as probe:
                image_format = (probe.format or "").upper()
                if image_format not in _SUPPORTED_FORMATS:
                    raise NvidiaPreprocessingError("nvidia_image_format_unsupported")
                if normalized_media_type is not None and normalized_media_type not in _MEDIA_TYPES[
                    image_format
                ]:
                    raise NvidiaPreprocessingError("nvidia_image_media_type_mismatch")
                width, height = probe.size
                if (
                    width <= 0
                    or height <= 0
                    or width * height > _MAX_DECODED_PIXELS
                    or int(getattr(probe, "n_frames", 1)) != 1
                    or bool(getattr(probe, "is_animated", False))
                ):
                    raise NvidiaPreprocessingError("nvidia_image_dimensions_invalid")
                probe.verify()

            with Image.open(io.BytesIO(payload)) as opened:
                oriented = ImageOps.exif_transpose(opened)
                oriented.load()
                source = oriented.convert("RGB")
                images = tuple(
                    _encode_derivative(
                        source,
                        kind=kind,
                        maximum_edge=maximum_edge,
                        jpeg_quality=jpeg_quality,
                    )
                    for kind in crop_kinds
                )
    except NvidiaPreprocessingError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, MemoryError):
        raise NvidiaPreprocessingError("nvidia_image_decompression_refused") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise NvidiaPreprocessingError("nvidia_image_malformed") from None

    prepared = PreparedImageSet(
        images=images,
        _source_sha256=hashlib.sha256(payload).hexdigest(),
    )
    if prepared.total_bytes > _MAX_DERIVATIVE_SET_BYTES:
        raise NvidiaPreprocessingError("nvidia_image_derivative_set_too_large")
    return prepared
