from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any

MEGALOC_SOURCE_REVISION = "1af071c68fc3ab6c6018c5c868391763516e50f7"
MEGALOC_MODEL_REVISION = "7cb9f7970d366fdf059963d04d372e503e8e9df9"
MEGALOC_WEIGHT_SHA256 = "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="verify-phase6c-models")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--artifacts-only", action="store_true")
    return parser.parse_args()


def _load_adapter_module(project_root: Path) -> ModuleType:
    workers_root = project_root / "services" / "model-workers"
    if str(workers_root) not in sys.path:
        sys.path.insert(0, str(workers_root))
    path = workers_root / "megaloc" / "adapter.py"
    spec = importlib.util.spec_from_file_location("atlaslens_phase6c_megaloc_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("megaloc_adapter_import_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verification_image() -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (224, 224), (31, 43, 57))
    draw = ImageDraw.Draw(image)
    for offset in range(0, 224, 28):
        draw.rectangle((offset, 0, min(223, offset + 13), 223), fill=(92, 113, 71))
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    image.close()
    return output.getvalue()


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _safe_summary(
    health: dict[str, object], *, inference: dict[str, object] | None
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": "atlaslens-phase6c-model-verification-v1",
        "provider": "megaloc",
        "source_revision": health.get("source_revision"),
        "model_revision": health.get("model_revision"),
        "weights_available": health.get("weights_available"),
        "import_ok": health.get("import_ok"),
        "load_verified": health.get("load_verified"),
        "real_inference_verified": health.get("real_inference_verified"),
        "device": health.get("device"),
        "descriptor_dimension": health.get("descriptor_dimension"),
    }
    if inference is not None:
        values = inference.get("descriptor")
        if not isinstance(values, list):
            raise RuntimeError("invalid_descriptor")
        norm = math.sqrt(sum(float(value) ** 2 for value in values))
        summary["descriptor_norm"] = round(norm, 7)
        summary["inference_ms"] = inference.get("inference_ms")
    return summary


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    module = _load_adapter_module(project_root)
    artifacts = module.MegaLocArtifacts(
        source_dir=project_root / ".local" / "vendor" / "megaloc",
        model_dir=project_root / ".local" / "models" / "phase6c" / "megaloc",
        source_revision=MEGALOC_SOURCE_REVISION,
        model_revision=MEGALOC_MODEL_REVISION,
        weight_sha256=MEGALOC_WEIGHT_SHA256,
    )
    module.verify_weight(artifacts.weights_path, MEGALOC_WEIGHT_SHA256)
    adapter = module.MegaLocAdapter(artifacts, maximum_edge=224)
    initial = dict(adapter.health())
    if not initial.get("weights_available"):
        raise RuntimeError("megaloc_artifacts_unavailable")
    if args.artifacts_only:
        print(json.dumps(_safe_summary(initial, inference=None), sort_keys=True))
        return 0

    device = _select_device(args.device)
    inference: dict[str, object] | None = None
    try:
        adapter.load({"device": device})
        inference = dict(adapter.infer(_verification_image(), {"device": device}))
        final = dict(adapter.health())
        if not final.get("load_verified") or not final.get("real_inference_verified"):
            raise RuntimeError("megaloc_real_inference_not_verified")
        print(json.dumps(_safe_summary(final, inference=inference), sort_keys=True))
    finally:
        adapter.unload({})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
