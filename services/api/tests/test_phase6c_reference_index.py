from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from atlaslens_api.phase6c.reference_index import (
    EXCLUSIONS_FILENAME,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    REFERENCE_INPUT_SCHEMA_VERSION,
    VECTORS_FILENAME,
    MegaLocDescriptorSpec,
    MegaLocReferenceIndex,
    ReferenceBuildInput,
    ReferenceIndexBuildPolicy,
    ReferenceIndexIntegrityError,
    build_reference_index,
    open_reference_index,
)


def _write_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="PNG")


def _record(
    *,
    reference_id: str,
    image_path: str,
    sequence_id: str,
    source_image_id: str,
    province: str = "Ankara",
) -> dict[str, object]:
    return {
        "reference_id": reference_id,
        "source": "manual",
        "source_family": "licensed_manual_family",
        "source_image_id": source_image_id,
        "source_sequence_id": sequence_id,
        "source_url": f"https://example.test/metadata/{source_image_id}",
        "latitude": 39.9334,
        "longitude": 32.8597,
        "coordinate_uncertainty_m": 12.0,
        "heading_degrees": None,
        "captured_at": "2024-05-01T12:00:00Z",
        "country": "TR",
        "province": province,
        "city": "Ankara",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Fixture author",
        "asset_key": f"reference/{reference_id}",
        "image_path": image_path,
    }


def _manifest(path: Path, records: list[dict[str, object]]) -> Path:
    manifest = path / "input.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": REFERENCE_INPUT_SCHEMA_VERSION,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _spec(dimension: int = 3) -> MegaLocDescriptorSpec:
    return MegaLocDescriptorSpec(
        model_id="gberton/MegaLoc",
        model_revision="test-model-revision",
        source_revision="test-source-revision",
        descriptor_version="test-descriptor-v1",
        dimension=dimension,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_verify_and_normalized_cosine_search_with_sequence_cap(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    records: list[dict[str, object]] = []
    for position, sequence in enumerate(("sequence-a", "sequence-a", "sequence-b")):
        filename = f"image-{position}.png"
        _write_image(input_root / filename, seed=position + 1)
        records.append(
            _record(
                reference_id=f"reference-{position}",
                image_path=filename,
                sequence_id=sequence,
                source_image_id=f"source-{position}",
            )
        )
    matrix = np.asarray(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    output = tmp_path / "index"

    result = build_reference_index(
        input_manifest=_manifest(input_root, records),
        input_root=input_root,
        output_directory=output,
        index_version="turkey-test-v1",
        descriptor_spec=_spec(),
        descriptor_matrix=matrix,
        policy=ReferenceIndexBuildPolicy(perceptual_hash_hamming_threshold=0),
        built_at=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result.manifest.count == 3
    assert result.manifest.independent_sequences == 2
    assert result.manifest.vectors.sha256 == _sha256(output / VECTORS_FILENAME)
    assert result.manifest.metadata.sha256 == _sha256(output / METADATA_FILENAME)
    assert result.manifest.exclusions.sha256 == _sha256(output / EXCLUSIONS_FILENAME)
    index = MegaLocReferenceIndex.open(output)
    assert index.available is True
    assert index.index_version == "turkey-test-v1"
    assert index.descriptor_version == "test-descriptor-v1"
    hits = index.search([10.0, 0.0, 0.0], top_k=2, max_per_sequence=1)
    assert [hit.reference_id for hit in hits] == ["reference-0", "reference-2"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].similarity == pytest.approx(1.0)
    assert hits[0].similarity_semantics == "cosine_similarity_not_confidence"
    assert hits[0].distance == pytest.approx(0.0)
    assert hits[0].confidence is None
    assert hits[0].distance_semantics == "cosine_distance_not_confidence"
    assert all(hit.uncertainty_radius_m > 0 for hit in hits)

    persisted_metadata = (output / METADATA_FILENAME).read_text(encoding="utf-8")
    assert "image-0.png" not in persisted_metadata
    assert "latitude" in persisted_metadata


def test_build_records_exact_exclusion_duplicate_and_sequence_limit(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    target = input_root / "excluded.png"
    accepted = input_root / "accepted.png"
    another = input_root / "another.png"
    _write_image(target, seed=10)
    _write_image(accepted, seed=11)
    _write_image(another, seed=12)
    records = [
        _record(
            reference_id="excluded-reference",
            image_path=target.name,
            sequence_id="sequence-x",
            source_image_id="source-excluded",
        ),
        _record(
            reference_id="accepted-reference",
            image_path=accepted.name,
            sequence_id="sequence-a",
            source_image_id="source-accepted",
        ),
        _record(
            reference_id="exact-duplicate",
            image_path=accepted.name,
            sequence_id="sequence-b",
            source_image_id="source-duplicate",
        ),
        _record(
            reference_id="sequence-overflow",
            image_path=another.name,
            sequence_id="sequence-a",
            source_image_id="source-overflow",
        ),
    ]
    output = tmp_path / "index"
    result = build_reference_index(
        input_manifest=_manifest(input_root, records),
        input_root=input_root,
        output_directory=output,
        index_version="dedup-test-v1",
        descriptor_spec=_spec(),
        descriptor_matrix=np.eye(4, 3, dtype=np.float32),
        policy=ReferenceIndexBuildPolicy(
            max_per_sequence=1,
            perceptual_hash_hamming_threshold=0,
            excluded_sha256=frozenset({_sha256(target)}),
        ),
    )

    assert result.manifest.count == 1
    assert result.manifest.deduplication.counts_by_reason == {
        "duplicate_sha256": 1,
        "excluded_sha256": 1,
        "sequence_limit": 1,
    }
    exclusions = json.loads((output / EXCLUSIONS_FILENAME).read_text(encoding="utf-8"))
    assert {item["reference_id"] for item in exclusions["exclusions"]} == {
        "excluded-reference",
        "exact-duplicate",
        "sequence-overflow",
    }
    assert all("image_path" not in item for item in exclusions["exclusions"])


def test_open_detects_checksum_tampering_and_optional_open_is_safe(tmp_path: Path) -> None:
    missing = open_reference_index(tmp_path / "missing")
    assert missing.index is None
    assert missing.diagnostics.status == "unavailable"
    assert missing.diagnostics.reason_code == "index_unavailable"

    input_root = tmp_path / "inputs"
    input_root.mkdir()
    image_path = input_root / "image.png"
    _write_image(image_path, seed=20)
    output = tmp_path / "index"
    build_reference_index(
        input_manifest=_manifest(
            input_root,
            [
                _record(
                    reference_id="reference-a",
                    image_path=image_path.name,
                    sequence_id="sequence-a",
                    source_image_id="source-a",
                )
            ],
        ),
        input_root=input_root,
        output_directory=output,
        index_version="tamper-test-v1",
        descriptor_spec=_spec(),
        descriptor_matrix=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        policy=ReferenceIndexBuildPolicy(perceptual_hash_hamming_threshold=0),
    )
    vector_path = output / VECTORS_FILENAME
    contents = bytearray(vector_path.read_bytes())
    contents[-1] ^= 0x01
    vector_path.write_bytes(contents)

    with pytest.raises(ReferenceIndexIntegrityError, match="checksum"):
        MegaLocReferenceIndex.open(output)
    invalid = open_reference_index(output)
    assert invalid.index is None
    assert invalid.diagnostics.status == "invalid"


def test_empty_index_is_valid_and_unknown_ground_truth_fields_are_rejected(
    tmp_path: Path,
) -> None:
    invalid = _record(
        reference_id="reference-a",
        image_path="image.png",
        sequence_id="sequence-a",
        source_image_id="source-a",
    )
    invalid["ground_truth_label"] = "must-not-enter-inference-artifact"
    with pytest.raises(ValueError, match="ground_truth_label"):
        ReferenceBuildInput.model_validate(
            {"schema_version": REFERENCE_INPUT_SCHEMA_VERSION, "records": [invalid]}
        )

    input_root = tmp_path / "inputs"
    input_root.mkdir()
    output = tmp_path / "empty-index"
    build_reference_index(
        input_manifest=_manifest(input_root, []),
        input_root=input_root,
        output_directory=output,
        index_version="empty-test-v1",
        descriptor_spec=_spec(),
        descriptor_matrix=np.empty((0, 3), dtype=np.float32),
    )
    opened = open_reference_index(output)
    assert opened.index is not None
    assert opened.diagnostics.status == "empty"
    assert opened.index.search([1.0, 0.0, 0.0], top_k=5) == ()
    assert (output / MANIFEST_FILENAME).is_file()


def test_positive_uncertainty_and_timezone_are_mandatory() -> None:
    record = _record(
        reference_id="reference-a",
        image_path="image.png",
        sequence_id="sequence-a",
        source_image_id="source-a",
    )
    record["coordinate_uncertainty_m"] = 0
    with pytest.raises(ValueError, match="coordinate_uncertainty_m"):
        ReferenceBuildInput.model_validate(
            {"schema_version": REFERENCE_INPUT_SCHEMA_VERSION, "records": [record]}
        )
    record["coordinate_uncertainty_m"] = 1
    record["captured_at"] = "2024-01-01T12:00:00"
    with pytest.raises(ValueError, match="timezone"):
        ReferenceBuildInput.model_validate(
            {"schema_version": REFERENCE_INPUT_SCHEMA_VERSION, "records": [record]}
        )
    record["captured_at"] = None
    record["source_url"] = "https://example.test/item?access_token=secret"
    with pytest.raises(ValueError, match="credential-free"):
        ReferenceBuildInput.model_validate(
            {"schema_version": REFERENCE_INPUT_SCHEMA_VERSION, "records": [record]}
        )
    record["source_url"] = "https://example.test/item"
    record["asset_key"] = "reference/../private"
    with pytest.raises(ValueError, match="traversal"):
        ReferenceBuildInput.model_validate(
            {"schema_version": REFERENCE_INPUT_SCHEMA_VERSION, "records": [record]}
        )
