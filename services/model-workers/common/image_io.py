from __future__ import annotations

import base64
import binascii
import struct

from common.protocol import WorkerProtocolError

_PNG = b"\x89PNG\r\n\x1a\n"


def decode_bounded_image(
    encoded: str,
    *,
    max_image_bytes: int,
    max_decoded_pixels: int,
    max_dimension: int,
) -> bytes:
    if not encoded or len(encoded) > ((max_image_bytes + 2) // 3) * 4:
        raise WorkerProtocolError("image_too_large", status=413)
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise WorkerProtocolError("invalid_image_encoding") from exc
    if not 0 < len(image_bytes) <= max_image_bytes:
        raise WorkerProtocolError("image_too_large", status=413)
    dimensions = inspect_image_dimensions(image_bytes)
    if dimensions is None:
        raise WorkerProtocolError("unsupported_or_invalid_image", status=422)
    width, height = dimensions
    if (
        width <= 0
        or height <= 0
        or width > max_dimension
        or height > max_dimension
        or width * height > max_decoded_pixels
    ):
        raise WorkerProtocolError("decoded_image_too_large", status=413)
    return image_bytes


def inspect_image_dimensions(payload: bytes) -> tuple[int, int] | None:
    if len(payload) >= 24 and payload.startswith(_PNG) and payload[12:16] == b"IHDR":
        return struct.unpack(">II", payload[16:24])
    if len(payload) >= 12 and payload[:2] == b"\xff\xd8":
        return _jpeg_dimensions(payload)
    if len(payload) >= 30 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return _webp_dimensions(payload)
    return None


def _jpeg_dimensions(payload: bytes) -> tuple[int, int] | None:
    offset = 2
    while offset + 4 <= len(payload):
        if payload[offset] != 0xFF:
            return None
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            return None
        marker = payload[offset]
        offset += 1
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            return None
        if offset + 2 > len(payload):
            return None
        segment_length = int.from_bytes(payload[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(payload):
            return None
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if segment_length < 7:
                return None
            height = int.from_bytes(payload[offset + 3 : offset + 5], "big")
            width = int.from_bytes(payload[offset + 5 : offset + 7], "big")
            return width, height
        offset += segment_length
    return None


def _webp_dimensions(payload: bytes) -> tuple[int, int] | None:
    chunk = payload[12:16]
    if chunk == b"VP8X" and len(payload) >= 30:
        width = 1 + int.from_bytes(payload[24:27], "little")
        height = 1 + int.from_bytes(payload[27:30], "little")
        return width, height
    if chunk == b"VP8L" and len(payload) >= 25 and payload[20] == 0x2F:
        bits = int.from_bytes(payload[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 " and len(payload) >= 30 and payload[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(payload[26:28], "little") & 0x3FFF
        height = int.from_bytes(payload[28:30], "little") & 0x3FFF
        return width, height
    return None
