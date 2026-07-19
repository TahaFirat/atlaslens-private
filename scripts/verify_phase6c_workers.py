from __future__ import annotations

import argparse
import asyncio
import json
import math
from io import BytesIO
from typing import cast

from atlaslens_api.phase6b.worker_client import WorkerClientError
from atlaslens_api.phase6c.megaloc import DeviceName, create_megaloc_http_worker_client
from PIL import Image, ImageDraw


def _verification_image() -> bytes:
    image = Image.new("RGB", (224, 224), (23, 37, 51))
    draw = ImageDraw.Draw(image)
    for offset in range(0, 224, 28):
        draw.rectangle((0, offset, 223, min(223, offset + 12)), fill=(80, 105, 65))
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    image.close()
    return output.getvalue()


async def _verify(port: int, device: DeviceName, timeout_seconds: float) -> dict[str, object]:
    client = create_megaloc_http_worker_client(
        host="127.0.0.1",
        port=port,
        timeout_seconds=timeout_seconds,
    )
    try:
        before = await client.health(timeout_seconds=min(10, timeout_seconds))
        await client.load(device, timeout_seconds=timeout_seconds)
        descriptor = await client.describe(
            _verification_image(),
            device=device,
            timeout_seconds=timeout_seconds,
        )
        after = await client.health(timeout_seconds=min(10, timeout_seconds))
        norm = math.sqrt(sum(value * value for value in descriptor))
        if (
            len(descriptor) != 8_448
            or abs(norm - 1.0) > 1e-3
            or not after.load_verified
            or not after.real_inference_verified
        ):
            raise WorkerClientError("megaloc_real_inference_not_verified")
        return {
            "schema_version": "atlaslens-phase6c-worker-verification-v1",
            "provider": after.provider,
            "provider_revision": after.provider_revision,
            "model_revision": after.model_revision,
            "device": after.device,
            "import_ok": after.import_ok,
            "weights_available": after.weights_available,
            "load_verified": after.load_verified,
            "real_inference_verified": after.real_inference_verified,
            "descriptor_dimension": len(descriptor),
            "descriptor_norm": round(norm, 7),
            "previously_loaded": before.model_loaded,
        }
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(prog="verify-phase6c-workers")
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    try:
        result = asyncio.run(
            _verify(args.port, cast(DeviceName, args.device), args.timeout_seconds)
        )
    except (OSError, ValueError, WorkerClientError):
        print(json.dumps({"status": "failed", "reason_code": "worker_verification_failed"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
