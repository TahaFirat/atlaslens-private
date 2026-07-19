#!/usr/bin/env python3
"""Run a real RapidOCR process inference and persist non-sensitive proof."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _image_bytes() -> bytes:
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


async def _verify(project_root: Path, model_root: Path) -> dict[str, object]:
    sys.path.insert(0, str(project_root / "services" / "api" / "src"))
    from atlaslens_api.phase6b.runtime_verification import (  # noqa: PLC0415
        write_rapidocr_runtime_verification,
    )
    from atlaslens_api.providers.rapidocr import (  # noqa: PLC0415
        load_verified_rapidocr_runtime,
    )
    from atlaslens_api.providers.rapidocr_worker import (  # noqa: PLC0415
        ProcessRapidOCRWorker,
    )

    runtime = load_verified_rapidocr_runtime(model_root, device="cpu")
    if runtime is None:
        raise RuntimeError("rapidocr_artifacts_not_verified")
    worker = ProcessRapidOCRWorker(runtime.resolve_profiles(), device="cpu")
    try:
        batch = await worker.infer(
            _image_bytes(),
            timeout_seconds=60,
            cancellation=asyncio.Event(),
        )
    finally:
        await worker.close()
    recognized = " ".join(line.text for line in batch.lines).casefold()
    if not any(token in recognized for token in ("istanbul", "atlas", "2026", "merkez")):
        raise RuntimeError("rapidocr_meaningful_token_missing")
    write_rapidocr_runtime_verification(
        project_root,
        model_root,
        source_revision="7fe716f8e38bb9a43f2680159f38deb14d8b1930",
        model_revision="rapidocr-3.9.1",
        model_load_succeeded=True,
        real_inference_succeeded=True,
    )
    return {
        "provider": "rapidocr",
        "state": "ready",
        "device": batch.device,
        "load_verified": True,
        "inference_verified": True,
        "meaningful_token_verified": True,
        "detection_count": len(batch.lines),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(
        _verify(args.project_root.resolve(), args.model_root.expanduser().resolve())
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
