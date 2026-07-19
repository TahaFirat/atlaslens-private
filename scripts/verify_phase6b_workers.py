#!/usr/bin/env python3
"""Perform one real, offline inference against each selected local worker."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _smoke_image() -> bytes:
    image = Image.new("RGB", (1200, 360), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf"
    try:
        font = ImageFont.truetype(str(font_path), 72)
    except OSError:
        font = ImageFont.load_default()
    draw.text((50, 45), "ISTANBUL ATLAS", fill="black", font=font)
    draw.text((50, 165), "2026 MERKEZ", fill="black", font=font)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    image.close()
    return output.getvalue()


async def _verify(project_root: Path, providers: tuple[str, ...]) -> dict[str, object]:
    sys.path.insert(0, str(project_root / "services" / "api" / "src"))
    from atlaslens_api.phase6b.worker_client import (  # noqa: PLC0415
        LocalWorkerHTTPClient,
        OSV5MHTTPWorkerClient,
        PaddleOCRHTTPWorkerClient,
        PlonkHTTPWorkerClient,
    )

    image_bytes = _smoke_image()
    results: dict[str, object] = {}
    if "osv5m" in providers:
        client = LocalWorkerHTTPClient(
            provider="osv5m",
            host="127.0.0.1",
            port=8791,
            provider_revision="4e6075387ecde4255410785ffb83830c9aa099f6",
            model_revisions=("71548b90ac4a1aa7c37839841f411a06da82b1a6",),
            timeout_seconds=180,
        )
        worker = OSV5MHTTPWorkerClient(client, model_id="osv5m/baseline")
        await worker.load("cpu")
        prediction = await worker.predict_radians(image_bytes, device="cpu")
        if not isinstance(prediction, tuple) or len(prediction) != 2:
            raise RuntimeError("osv5m_invalid_real_output")
        health = await worker.health()
        if not health.ready:
            raise RuntimeError("osv5m_inference_not_verified")
        await worker.unload("cpu")
        results["osv5m"] = {
            "state": "ready",
            "load_verified": health.load_verified,
            "inference_verified": health.real_inference_verified,
            "device": health.device,
        }
    if "plonk" in providers:
        yfcc_id = "nicolas-dufour/PLONK_YFCC"
        client = LocalWorkerHTTPClient(
            provider="plonk",
            host="127.0.0.1",
            port=8792,
            provider_revision="76d46410910c9dfec9e19ed371450ebc7051cdf3",
            model_revisions=(
                "scene-routed",
                "e23229f4dd91d52560e8827f5bb2c68257fa162f",
                "4f358d09938a89ed239a847777729e95c5d187bc",
                "8da6edcbdd01ff04a61f9d06e2de23ea300d1a35",
            ),
            timeout_seconds=180,
        )
        worker = PlonkHTTPWorkerClient(client)
        await worker.load(yfcc_id, "cpu")
        prediction = await worker.sample(
            image_bytes,
            model_id=yfcc_id,
            sample_count=4,
            device="cpu",
        )
        if len(prediction.samples_degrees) != 4:
            raise RuntimeError("plonk_invalid_real_output")
        health = await worker.health()
        if not health.ready:
            raise RuntimeError("plonk_inference_not_verified")
        await worker.unload(yfcc_id, "cpu")
        results["plonk"] = {
            "state": "ready",
            "verified_specialization": yfcc_id,
            "load_verified": health.load_verified,
            "inference_verified": health.real_inference_verified,
            "device": health.device,
        }
    if "paddleocr" in providers:
        client = LocalWorkerHTTPClient(
            provider="paddleocr",
            host="127.0.0.1",
            port=8793,
            provider_revision="3.7.0",
            model_revisions=("PP-OCRv5_server_det+latin_PP-OCRv5_mobile_rec",),
            timeout_seconds=60,
        )
        worker = PaddleOCRHTTPWorkerClient(client)
        await worker.load()
        lines = await worker.infer(
            image_bytes,
            scales=(1.0,),
            timeout_seconds=60,
            cancellation=asyncio.Event(),
        )
        recognized = " ".join(line.text for line in lines).casefold()
        if not any(token in recognized for token in ("istanbul", "atlas", "2026", "merkez")):
            raise RuntimeError("paddleocr_meaningful_token_missing")
        health = await worker.health()
        if not health.ready:
            raise RuntimeError("paddleocr_inference_not_verified")
        await client.unload({"device": "cpu"})
        results["paddleocr"] = {
            "state": "ready",
            "meaningful_token_verified": True,
            "load_verified": health.load_verified,
            "inference_verified": health.real_inference_verified,
            "device": health.device,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("osv5m", "plonk", "paddleocr"),
        default=("osv5m", "plonk", "paddleocr"),
    )
    args = parser.parse_args()
    report = asyncio.run(_verify(args.project_root.resolve(), tuple(args.providers)))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
