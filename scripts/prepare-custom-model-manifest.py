"""Prepare a manifest for an already-exported custom ONNX artifact.

This utility does not train, export, download, or fabricate a model. It hashes an
operator-supplied artifact and writes the strict AtlasLens registration manifest.
JSON is emitted because it is also valid YAML 1.2 and can safely use a `.yaml`
filename without an additional script dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("--artifact", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--model-id", required=True)
    command.add_argument("--provider-id", required=True)
    command.add_argument("--model-version", required=True)
    command.add_argument("--implementation-revision", required=True)
    command.add_argument("--dataset-fingerprint", required=True)
    command.add_argument("--training-code-revision", required=True)
    command.add_argument("--training-completed-at", required=True)
    command.add_argument("--license-name", required=True)
    command.add_argument("--license-review-reference", required=True)
    command.add_argument(
        "--commercial-use", choices=("allowed", "restricted", "unknown"), required=True
    )
    command.add_argument("--preprocessing-version", required=True)
    command.add_argument("--input-tensor", required=True)
    command.add_argument("--coordinates-output", required=True)
    command.add_argument("--scores-output", required=True)
    command.add_argument(
        "--normalization", choices=("zero_to_one", "mean_std"), required=True
    )
    command.add_argument("--mean", type=float, nargs=3)
    command.add_argument("--std", type=float, nargs=3)
    command.add_argument("--score-type", required=True)
    command.add_argument("--width", type=int, default=384)
    command.add_argument("--height", type=int, default=384)
    command.add_argument("--top-k", type=int, default=5)
    return command


def main() -> int:
    args = parser().parse_args()
    artifact = args.artifact.expanduser()
    output = args.output.expanduser()
    if artifact.is_symlink() or not artifact.is_file() or artifact.suffix.lower() != ".onnx":
        raise SystemExit("artifact must be a regular ONNX file")
    output_is_unsafe = (
        output.exists()
        or output.is_symlink()
        or output.parent.resolve() != artifact.parent.resolve()
    )
    if output_is_unsafe:
        raise SystemExit("output must be a new file beside the artifact")
    if len(args.dataset_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in args.dataset_fingerprint
    ):
        raise SystemExit("dataset fingerprint must be lowercase SHA-256")
    datetime.fromisoformat(args.training_completed_at.replace("Z", "+00:00"))
    size = artifact.stat().st_size
    if size <= 0:
        raise SystemExit("artifact is empty")
    if args.normalization == "mean_std" and (args.mean is None or args.std is None):
        raise SystemExit("mean_std normalization requires --mean and --std")
    if args.normalization == "zero_to_one" and (args.mean is not None or args.std is not None):
        raise SystemExit("zero_to_one normalization does not accept --mean/--std")
    if not args.score_type.startswith("uncalibrated_"):
        raise SystemExit("score type must explicitly begin with uncalibrated_")
    input_contract = {
        "tensor_name": args.input_tensor,
        "width": args.width,
        "height": args.height,
        "color_space": "RGB",
        "layout": "NCHW",
        "dtype": "float32",
        "resize_method": "bicubic",
        "normalization": args.normalization,
        "preprocessing_version": args.preprocessing_version,
    }
    if args.normalization == "mean_std":
        input_contract["mean"] = args.mean
        input_contract["std"] = args.std
    payload = {
        "schema_version": "atlaslens-trained-artifact-v1",
        "model_id": args.model_id,
        "provider_id": args.provider_id,
        "model_version": args.model_version,
        "implementation_revision": args.implementation_revision,
        "task": "global_geolocation",
        "artifact_format": "onnx",
        "runtime_adapter": "onnx-coordinate-v1",
        "artifact_file": artifact.name,
        "artifact_sha256": sha256(artifact),
        "artifact_size_bytes": size,
        "input": input_contract,
        "output": {
            "type": "top_k_coordinates",
            "coordinate_order": "lat_lon",
            "top_k": args.top_k,
            "score_type": args.score_type,
            "output_schema_version": "coordinate-topk-v1",
            "coordinates_output": args.coordinates_output,
            "scores_output": args.scores_output,
        },
        "training": {
            "dataset_fingerprint": args.dataset_fingerprint,
            "code_revision": args.training_code_revision,
            "completed_at": args.training_completed_at,
        },
        "evaluation": {
            "report_required_before_promotion": True,
            "minimum_sample_count": 30,
            "primary_minimum_sample_count": 100,
            "maximum_country_top1_regression": 0.05,
            "maximum_recall_200km_regression": 0.05,
            "maximum_median_error_increase": 25.0,
            "maximum_latency_p95_ms": 5000.0,
        },
        "license": {
            "name": args.license_name,
            "status": "approved",
            "commercial_use": args.commercial_use,
            "review_reference": args.license_review_reference,
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
