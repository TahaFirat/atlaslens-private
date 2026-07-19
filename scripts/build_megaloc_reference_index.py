#!/usr/bin/env python3
"""Build a checked, bounded MegaLoc reference index from explicit licensed inputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from contextlib import suppress
from pathlib import Path

import numpy as np
from atlaslens_api.phase6b.worker_client import LocalWorkerHTTPClient, WorkerClientError
from atlaslens_api.phase6c.reference_index import (
    MegaLocDescriptorSpec,
    ReferenceIndexBuildError,
    ReferenceIndexBuildPolicy,
    build_reference_index,
    estimate_index_bytes,
    load_descriptor_matrix,
    load_reference_build_input,
)

MEGALOC_MODEL_ID = "gberton/MegaLoc"
MEGALOC_SOURCE_REVISION = "1af071c68fc3ab6c6018c5c868391763516e50f7"
MEGALOC_MODEL_REVISION = "7cb9f7970d366fdf059963d04d372e503e8e9df9"
MEGALOC_DESCRIPTOR_VERSION = "megaloc-7cb9f797-max560-imagenet-v1"
MEGALOC_DIMENSION = 8_448


class MegaLocWorkerDescriptorProvider:
    def __init__(
        self,
        *,
        port: int,
        device: str,
        timeout_seconds: float,
        spec: MegaLocDescriptorSpec,
    ) -> None:
        self._device = device
        self._spec = spec
        self._client = LocalWorkerHTTPClient(
            provider="megaloc",
            host="127.0.0.1",
            port=port,
            provider_revision=spec.source_revision,
            model_revisions=(spec.model_revision,),
            timeout_seconds=timeout_seconds,
            max_response_bytes=2 * 1024 * 1024,
        )
        asyncio.run(self._client.load({"device": device}))

    def describe(self, image_path: Path) -> np.ndarray:
        response = asyncio.run(
            self._client.infer(
                image_path.read_bytes(),
                {"device": self._device},
            )
        )
        result = response.result
        if (
            result.get("model_id") != self._spec.model_id
            or result.get("model_revision") != self._spec.model_revision
            or result.get("source_revision") != self._spec.source_revision
            or result.get("descriptor_dimension") != self._spec.dimension
            or result.get("descriptor_normalization") != "l2"
        ):
            raise ReferenceIndexBuildError("worker_descriptor_identity_mismatch")
        descriptor = result.get("descriptor")
        if not isinstance(descriptor, list):
            raise ReferenceIndexBuildError("worker_descriptor_invalid")
        return np.asarray(descriptor, dtype=np.float32)

    def close(self) -> None:
        asyncio.run(self._client.close())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-version", required=True)
    parser.add_argument(
        "--descriptor-matrix",
        type=Path,
        help="Optional real float32 .npy matrix aligned with input records.",
    )
    parser.add_argument("--worker-port", type=int, default=8794)
    parser.add_argument("--worker-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--worker-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-images", type=int, default=5_000)
    parser.add_argument("--max-per-sequence", type=int, default=10)
    parser.add_argument("--max-per-province", type=int, default=100)
    parser.add_argument("--max-disk-gb", type=float, default=2.0)
    parser.add_argument("--perceptual-hash-distance", type=int, default=4)
    parser.add_argument("--descriptor-duplicate-distance", type=float)
    parser.add_argument("--exclude-sha256", action="append", default=[])
    parser.add_argument("--exclude-perceptual-hash", action="append", default=[])
    parser.add_argument("--confirm-large-index", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    provider: MegaLocWorkerDescriptorProvider | None = None
    try:
        if (
            not math.isfinite(args.max_disk_gb)
            or args.max_disk_gb <= 0
            or not 1 <= args.worker_port <= 65_535
        ):
            raise ReferenceIndexBuildError("build_bounds_invalid")
        spec = MegaLocDescriptorSpec(
            model_id=MEGALOC_MODEL_ID,
            model_revision=MEGALOC_MODEL_REVISION,
            source_revision=MEGALOC_SOURCE_REVISION,
            descriptor_version=MEGALOC_DESCRIPTOR_VERSION,
            dimension=MEGALOC_DIMENSION,
        )
        build_input = load_reference_build_input(args.input_manifest)
        prospective_count = min(len(build_input.records), max(0, args.max_images))
        estimated_bytes = estimate_index_bytes(prospective_count, spec.dimension)
        print(
            json.dumps(
                {
                    "event": "reference_index_estimate",
                    "input_records": len(build_input.records),
                    "maximum_output_records": prospective_count,
                    "estimated_index_bytes": estimated_bytes,
                },
                sort_keys=True,
            )
        )
        if args.max_images > 5_000 and not args.confirm_large_index:
            raise ReferenceIndexBuildError("large_index_confirmation_required")
        max_disk_bytes = round(args.max_disk_gb * 1024**3)
        policy = ReferenceIndexBuildPolicy(
            max_images=args.max_images,
            max_per_sequence=args.max_per_sequence,
            max_per_province=args.max_per_province,
            max_disk_bytes=max_disk_bytes,
            perceptual_hash_hamming_threshold=args.perceptual_hash_distance,
            descriptor_cosine_distance_threshold=args.descriptor_duplicate_distance,
            excluded_sha256=frozenset(args.exclude_sha256),
            excluded_perceptual_hashes=frozenset(args.exclude_perceptual_hash),
        )
        matrix = (
            load_descriptor_matrix(args.descriptor_matrix, spec)
            if args.descriptor_matrix is not None
            else None
        )
        needs_worker = matrix is None and any(
            record.descriptor_path is None for record in build_input.records
        )
        if needs_worker:
            provider = MegaLocWorkerDescriptorProvider(
                port=args.worker_port,
                device=args.worker_device,
                timeout_seconds=args.worker_timeout_seconds,
                spec=spec,
            )
        result = build_reference_index(
            input_manifest=args.input_manifest,
            input_root=args.input_root,
            output_directory=args.output,
            index_version=args.index_version,
            descriptor_spec=spec,
            policy=policy,
            descriptor_provider=provider,
            descriptor_matrix=matrix,
            replace=args.replace,
        )
        print(
            json.dumps(
                {
                    "event": "reference_index_built",
                    "index_version": result.manifest.index_version,
                    "count": result.manifest.count,
                    "independent_sequences": result.manifest.independent_sequences,
                    "excluded_count": result.manifest.deduplication.excluded_count,
                    "descriptor_version": result.manifest.descriptor.descriptor_version,
                },
                sort_keys=True,
            )
        )
        return 0
    except (ReferenceIndexBuildError, WorkerClientError, ValueError, OSError) as exc:
        code = str(exc)
        print(json.dumps({"event": "reference_index_build_failed", "reason_code": code}))
        return 1
    finally:
        if provider is not None:
            with suppress(WorkerClientError, OSError):
                provider.close()


if __name__ == "__main__":
    sys.exit(main())
