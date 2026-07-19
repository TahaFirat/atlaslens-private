from __future__ import annotations

import base64
import hashlib
import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

_MAX_SOURCE_BYTES = 25 * 1024 * 1024
_MAX_DECODED_PIXELS = 40_000_000
_MAX_DERIVATIVE_BYTES = 2 * 1024 * 1024
_SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


@dataclass(frozen=True, slots=True)
class ReducedImageDerivative:
    source_sha256: str
    jpeg_bytes: bytes
    width: int
    height: int

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def __repr__(self) -> str:
        return (
            "ReducedImageDerivative(source_sha256=<redacted>, jpeg_bytes=<redacted>, "
            f"width={self.width}, height={self.height})"
        )


def create_reduced_image_derivative(
    payload: bytes,
    *,
    maximum_edge: int = 768,
    jpeg_quality: int = 75,
) -> ReducedImageDerivative:
    """Decode, orient, resize and re-encode without carrying source metadata."""

    if not payload or len(payload) > _MAX_SOURCE_BYTES:
        raise ValueError("image payload is empty or too large")
    if not 256 <= maximum_edge <= 768 or not 50 <= jpeg_quality <= 85:
        raise ValueError("unsafe derivative settings")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as probe:
                if (probe.format or "").upper() not in _SUPPORTED_FORMATS:
                    raise ValueError("unsupported image format")
                width, height = probe.size
                if (
                    width <= 0
                    or height <= 0
                    or width * height > _MAX_DECODED_PIXELS
                    or int(getattr(probe, "n_frames", 1)) != 1
                    or bool(getattr(probe, "is_animated", False))
                ):
                    raise ValueError("unsupported image dimensions or frames")
                probe.verify()

            with Image.open(io.BytesIO(payload)) as source:
                oriented = ImageOps.exif_transpose(source)
                oriented.load()
                image = oriented.convert("RGB")
                image.thumbnail((maximum_edge, maximum_edge), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.save(
                    output,
                    format="JPEG",
                    quality=jpeg_quality,
                    optimize=True,
                    exif=b"",
                )
                encoded = output.getvalue()
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("decoded image is too large") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("malformed image") from exc

    if not encoded or len(encoded) > _MAX_DERIVATIVE_BYTES:
        raise ValueError("reduced image exceeds the derivative bound")
    return ReducedImageDerivative(
        source_sha256=hashlib.sha256(payload).hexdigest(),
        jpeg_bytes=encoded,
        width=image.width,
        height=image.height,
    )
