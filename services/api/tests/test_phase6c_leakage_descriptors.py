from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from atlaslens_api.evaluation.descriptors import (
    LeakageDescriptorBuildError,
    build_phase6c_leakage_descriptor_artifact,
)
from atlaslens_api.evaluation.leakage import load_descriptor_artifact
from atlaslens_api.phase6b.worker_client import WorkerHealth
from atlaslens_api.phase6c.megaloc import (
    MEGALOC_DESCRIPTOR_DIMENSION,
    MEGALOC_DESCRIPTOR_VERSION,
    MEGALOC_MODEL_ID,
    MEGALOC_MODEL_REVISION,
    MEGALOC_SOURCE_REVISION,
)
from atlaslens_api.phase6c.reference_index import REFERENCE_INPUT_SCHEMA_VERSION


def _image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    Image.fromarray(
        rng.integers(0, 256, size=(36, 48, 3), dtype=np.uint8)
    ).save(path, "PNG")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _holdout(path: Path, records: list[tuple[str, Path, str]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-geolocation-evaluation-v1",
                "records": [
                    {
                        "id": evaluation_id,
                        "path": str(image_path),
                        "country": "TR",
                        "city": secret_truth,
                        "split": "final_holdout",
                        "usage": "holdout_only",
                        "allow_reference_index": False,
                        "allow_training": False,
                        "allow_prompt_ground_truth": False,
                    }
                    for evaluation_id, image_path, secret_truth in records
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _record(reference_id: str, image_path: str) -> dict[str, object]:
    return {
        "reference_id": reference_id,
        "source": "manual",
        "source_family": "licensed_manual_family",
        "source_image_id": f"source-{reference_id}",
        "source_sequence_id": f"sequence-{reference_id}",
        "source_url": f"https://example.test/metadata/{reference_id}",
        "latitude": 39.0,
        "longitude": 35.0,
        "coordinate_uncertainty_m": 25.0,
        "heading_degrees": None,
        "captured_at": "2025-01-01T00:00:00Z",
        "country": "TR",
        "province": "Test Province",
        "city": None,
        "license": "Test-licensed fixture",
        "license_url": "https://example.test/license",
        "attribution": "Test fixture author",
        "asset_key": f"reference/{reference_id}",
        "image_path": image_path,
    }


def _reference_input(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": REFERENCE_INPUT_SCHEMA_VERSION,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path


class _FakeWorker:
    def __init__(
        self,
        *,
        source_revision: str = MEGALOC_SOURCE_REVISION,
        invalid_descriptor: bool = False,
    ) -> None:
        self.source_revision = source_revision
        self.invalid_descriptor = invalid_descriptor
        self.events: list[str] = []
        self.image_payloads: list[bytes] = []

    async def health(self, **_: Any) -> WorkerHealth:
        self.events.append("health")
        return WorkerHealth(
            schema_version="atlaslens-worker-v1",
            provider="megaloc",
            provider_revision=self.source_revision,
            model_revision=MEGALOC_MODEL_REVISION,
            device="cuda",
            process_running=True,
            import_ok=True,
            weights_available=True,
            model_loaded=False,
            load_verified=True,
            real_inference_verified=True,
            last_error=None,
        )

    async def load(self, device: str, **_: Any) -> None:
        self.events.append(f"load:{device}")

    async def describe(self, image_bytes: bytes, *, device: str, **_: Any) -> tuple[float, ...]:
        self.events.append(f"describe:{device}")
        self.image_payloads.append(image_bytes)
        if self.invalid_descriptor:
            return (1.0, 0.0)
        descriptor = np.zeros(MEGALOC_DESCRIPTOR_DIMENSION, dtype=np.float32)
        position = int(hashlib.sha256(image_bytes).hexdigest()[:8], 16) % len(descriptor)
        descriptor[position] = 1.0
        return tuple(float(value) for value in descriptor)

    async def unload(self, device: str, **_: Any) -> None:
        self.events.append(f"unload:{device}")

    async def close(self) -> None:
        self.events.append("close")


@pytest.mark.asyncio
async def test_real_descriptor_artifact_uses_prediction_projection_and_actual_sha_keys(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    unselected = tmp_path / "unselected.png"
    reference_root = tmp_path / "references"
    reference_root.mkdir()
    first = reference_root / "first.png"
    second = reference_root / "second.png"
    for path, seed in ((target, 1), (unselected, 2), (first, 3), (second, 4)):
        _image(path, seed)
    secret = "TRUTH_MUST_NEVER_ENTER_WORKER_PAYLOAD"
    holdout = _holdout(
        tmp_path / "holdout.json",
        [("selected", target, secret), ("not-selected", unselected, "OTHER_SECRET")],
    )
    reference_input = _reference_input(
        reference_root / "reference-input.json",
        [_record("one", "first.png"), _record("two", "second.png")],
    )
    output = tmp_path / "leakage-descriptors.npz"
    worker = _FakeWorker()

    result = await build_phase6c_leakage_descriptor_artifact(
        holdout_manifest=holdout,
        holdout_id="selected",
        reference_input=reference_input,
        reference_root=reference_root,
        output=output,
        worker=worker,  # type: ignore[arg-type]
    )

    expected_hashes = {_sha256(target), _sha256(first), _sha256(second)}
    artifact = load_descriptor_artifact(output)
    assert set(artifact.vectors) == expected_hashes
    assert artifact.provider == "megaloc"
    assert artifact.version == MEGALOC_DESCRIPTOR_VERSION
    assert result.input_image_count == 3
    assert result.unique_content_count == 3
    assert worker.image_payloads == [
        target.read_bytes(),
        first.read_bytes(),
        second.read_bytes(),
    ]
    assert all(secret.encode() not in payload for payload in worker.image_payloads)
    assert unselected.read_bytes() not in worker.image_payloads
    with np.load(output, allow_pickle=False) as payload:
        assert str(payload["model_id"].item()) == MEGALOC_MODEL_ID
        assert str(payload["model_revision"].item()) == MEGALOC_MODEL_REVISION
        assert str(payload["source_revision"].item()) == MEGALOC_SOURCE_REVISION
        assert int(payload["dimension"].item()) == MEGALOC_DESCRIPTOR_DIMENSION
        assert payload["vectors"].shape == (3, MEGALOC_DESCRIPTOR_DIMENSION)
    assert worker.events == [
        "health",
        "load:cuda",
        "describe:cuda",
        "describe:cuda",
        "describe:cuda",
        "unload:cuda",
        "close",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_path", ("../outside.png", "missing.png"))
async def test_invalid_reference_paths_fail_before_worker_contact(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    target = tmp_path / "target.png"
    outside = tmp_path / "outside.png"
    reference_root = tmp_path / "references"
    reference_root.mkdir()
    _image(target, 1)
    _image(outside, 2)
    holdout = _holdout(tmp_path / "holdout.json", [("selected", target, "SECRET")])
    reference_input = _reference_input(
        reference_root / "reference-input.json",
        [_record("one", invalid_path)],
    )
    worker = _FakeWorker()

    with pytest.raises(
        LeakageDescriptorBuildError,
        match="reference_image_(path_invalid|unavailable)",
    ):
        await build_phase6c_leakage_descriptor_artifact(
            holdout_manifest=holdout,
            holdout_id="selected",
            reference_input=reference_input,
            reference_root=reference_root,
            output=tmp_path / "output.npz",
            worker=worker,  # type: ignore[arg-type]
        )
    assert worker.events == []


@pytest.mark.asyncio
async def test_duplicate_reference_paths_fail_before_worker_contact(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    reference_root = tmp_path / "references"
    reference_root.mkdir()
    reference = reference_root / "same.png"
    _image(target, 1)
    _image(reference, 2)
    holdout = _holdout(tmp_path / "holdout.json", [("selected", target, "SECRET")])
    reference_input = _reference_input(
        reference_root / "reference-input.json",
        [_record("one", "same.png"), _record("two", "same.png")],
    )
    worker = _FakeWorker()

    with pytest.raises(LeakageDescriptorBuildError, match="duplicate_reference_path"):
        await build_phase6c_leakage_descriptor_artifact(
            holdout_manifest=holdout,
            holdout_id="selected",
            reference_input=reference_input,
            reference_root=reference_root,
            output=tmp_path / "output.npz",
            worker=worker,  # type: ignore[arg-type]
        )
    assert worker.events == []


@pytest.mark.asyncio
async def test_worker_identity_and_dimension_are_strict_and_output_is_atomic(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    reference_root = tmp_path / "references"
    reference_root.mkdir()
    reference = reference_root / "reference.png"
    _image(target, 1)
    _image(reference, 2)
    holdout = _holdout(tmp_path / "holdout.json", [("selected", target, "SECRET")])
    reference_input = _reference_input(
        reference_root / "reference-input.json",
        [_record("one", "reference.png")],
    )
    output = tmp_path / "output.npz"
    output.write_bytes(b"existing-operator-artifact")

    wrong_identity = _FakeWorker(source_revision="wrong-revision")
    with pytest.raises(LeakageDescriptorBuildError, match="worker_identity_mismatch"):
        await build_phase6c_leakage_descriptor_artifact(
            holdout_manifest=holdout,
            holdout_id="selected",
            reference_input=reference_input,
            reference_root=reference_root,
            output=output,
            worker=wrong_identity,  # type: ignore[arg-type]
        )
    assert output.read_bytes() == b"existing-operator-artifact"

    invalid_dimension = _FakeWorker(invalid_descriptor=True)
    with pytest.raises(LeakageDescriptorBuildError, match="worker_descriptor_invalid"):
        await build_phase6c_leakage_descriptor_artifact(
            holdout_manifest=holdout,
            holdout_id="selected",
            reference_input=reference_input,
            reference_root=reference_root,
            output=output,
            worker=invalid_dimension,  # type: ignore[arg-type]
        )
    assert output.read_bytes() == b"existing-operator-artifact"
    assert not list(tmp_path.glob(".output.npz.*.tmp"))
