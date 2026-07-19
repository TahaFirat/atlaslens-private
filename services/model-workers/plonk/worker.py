from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


MODEL_WORKERS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(MODEL_WORKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_WORKERS_ROOT))

from common.server import run_worker  # noqa: E402

from adapter import PlonkAdapter, PlonkArtifacts, PlonkModelArtifact  # noqa: E402


SOURCE_REVISION = "76d46410910c9dfec9e19ed371450ebc7051cdf3"
MODEL_REVISIONS = {
    "nicolas-dufour/PLONK_OSV_5M": "e23229f4dd91d52560e8827f5bb2c68257fa162f",
    "nicolas-dufour/PLONK_YFCC": "4f358d09938a89ed239a847777729e95c5d187bc",
    "nicolas-dufour/PLONK_iNaturalist": "8da6edcbdd01ff04a61f9d06e2de23ea300d1a35",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="atlaslens-plonk-worker")
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(
            os.environ.get(
                "PLONK_MODEL_ROOT",
                PROJECT_ROOT / ".local/models/phase6b/plonk",
            )
        ),
    )
    parser.add_argument(
        "--streetclip-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "PLONK_STREETCLIP_DIR",
                PROJECT_ROOT / ".local/models/phase6b/plonk/auxiliary/streetclip",
            )
        ),
    )
    parser.add_argument(
        "--dinov2-repo-dir",
        type=Path,
        default=Path(
            os.environ.get("PLONK_DINOV2_REPO_DIR", PROJECT_ROOT / ".local/vendor/dinov2")
        ),
    )
    parser.add_argument(
        "--dinov2-weights",
        type=Path,
        default=Path(
            os.environ.get(
                "PLONK_DINOV2_WEIGHTS",
                PROJECT_ROOT
                / ".local/models/phase6b/plonk/torch/hub/checkpoints"
                / "dinov2_vitl14_reg4_pretrain.pth",
            )
        ),
    )
    parser.add_argument("--max-image-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--max-image-pixels", type=int, default=40_000_000)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--localizability-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65_535:
        raise SystemExit("port must be between 1 and 65535")
    if args.max_image_bytes <= 0 or args.max_image_pixels <= 0:
        raise SystemExit("image bounds must be positive")
    model_root = args.model_root.resolve()
    models = {
        "nicolas-dufour/PLONK_OSV_5M": PlonkModelArtifact(
            model_id="nicolas-dufour/PLONK_OSV_5M",
            revision=MODEL_REVISIONS["nicolas-dufour/PLONK_OSV_5M"],
            model_dir=model_root / "osv",
            conditioning="streetclip",
        ),
        "nicolas-dufour/PLONK_YFCC": PlonkModelArtifact(
            model_id="nicolas-dufour/PLONK_YFCC",
            revision=MODEL_REVISIONS["nicolas-dufour/PLONK_YFCC"],
            model_dir=model_root / "yfcc",
            conditioning="dinov2",
        ),
        "nicolas-dufour/PLONK_iNaturalist": PlonkModelArtifact(
            model_id="nicolas-dufour/PLONK_iNaturalist",
            revision=MODEL_REVISIONS["nicolas-dufour/PLONK_iNaturalist"],
            model_dir=model_root / "inat",
            conditioning="dinov2",
        ),
    }
    adapter = PlonkAdapter(
        PlonkArtifacts(
            models=models,
            source_revision=SOURCE_REVISION,
            streetclip_dir=args.streetclip_dir.resolve(),
            dinov2_repo_dir=args.dinov2_repo_dir.resolve(),
            dinov2_weights_path=args.dinov2_weights.resolve(),
        ),
        max_input_bytes=args.max_image_bytes,
        max_image_pixels=args.max_image_pixels,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        localizability_samples=args.localizability_samples,
    )
    run_worker(
        adapter,
        provider="plonk",
        provider_revision=SOURCE_REVISION,
        model_revision="scene-routed",
        host=args.host,
        port=args.port,
        max_request_bytes=max(args.max_image_bytes * 2, 1024 * 1024),
        max_image_bytes=args.max_image_bytes,
        max_response_bytes=2 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
