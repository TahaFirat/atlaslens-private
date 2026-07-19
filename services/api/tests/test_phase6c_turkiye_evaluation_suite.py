from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from atlaslens_api.evaluation.manifest import EvaluationManifestLoader
from atlaslens_api.evaluation.turkiye_suite import (
    EvaluationSuiteBuildError,
    EvaluationSuiteBuildPolicy,
    TurkiyeEvaluationSuiteBuilder,
)
from atlaslens_api.phase6c.reference_index import ReferenceBuildInput, ReferenceBuildRecord

LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def _dhash(path: Path) -> str:
    with Image.open(path) as source:
        pixels = list(source.convert("L").resize((9, 8)).getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column + 1] > pixels[offset + column]
            )
    return f"{value:016x}"


def _ahash(path: Path) -> str:
    with Image.open(path) as source:
        pixels = list(source.convert("L").resize((8, 8), Image.Resampling.LANCZOS).getdata())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return f"{value:016x}"


def _distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _write_unique_image(
    root: Path,
    name: str,
    seed: str,
    *,
    forbidden_dhashes: set[str],
    forbidden_ahashes: set[str],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    for attempt in range(100):
        pixels = [
            hashlib.sha256(f"{seed}:{attempt}:{index}".encode()).digest()[0]
            for index in range(9 * 8)
        ]
        image = Image.new("L", (9, 8))
        image.putdata(pixels)
        image.resize((45, 40), Image.Resampling.NEAREST).save(path, format="PNG")
        if all(_distance(_dhash(path), value) > 8 for value in forbidden_dhashes) and all(
            _distance(_ahash(path), value) > 8 for value in forbidden_ahashes
        ):
            forbidden_dhashes.add(_dhash(path))
            forbidden_ahashes.add(_ahash(path))
            return path
    raise AssertionError("could not construct a distinct image fixture")


def _record(
    root: Path,
    path: Path,
    identifier: str,
    *,
    image_id: str | None = None,
    sequence: str | None = None,
    city: str | None = "Fixture City",
    province: str = "Fixture Province",
    expected_hashes: bool = True,
) -> ReferenceBuildRecord:
    relative = path.relative_to(root).as_posix()
    return ReferenceBuildRecord(
        reference_id=identifier,
        source="kartaview",
        source_family="kartaview_street_imagery_family",
        source_image_id=image_id or f"image-{identifier}",
        source_sequence_id=sequence or f"sequence-{identifier}",
        source_url=(
            f"https://kartaview.org/details/{sequence or f'sequence-{identifier}'}/"
            f"{image_id or f'image-{identifier}'}/track-info"
        ),
        latitude=39.0 + len(identifier) / 100,
        longitude=32.0 + len(identifier) / 100,
        coordinate_uncertainty_m=75,
        heading_degrees=90,
        captured_at=None,
        country="TR",
        province=province,
        city=city,
        license=LICENSE,
        license_url=LICENSE_URL,
        attribution="© Fixture source contributors",
        asset_key=f"reference/{identifier}",
        image_path=relative,
        expected_sha256=(
            hashlib.sha256(path.read_bytes()).hexdigest() if expected_hashes else None
        ),
        expected_perceptual_hash=_dhash(path) if expected_hashes else None,
    )


def _write_input(path: Path, records: list[ReferenceBuildRecord]) -> Path:
    path.write_text(
        ReferenceBuildInput(
            schema_version="atlaslens-megaloc-reference-input-v1",
            records=tuple(records),
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def _policy(*, minimum_records: int = 3) -> EvaluationSuiteBuildPolicy:
    return EvaluationSuiteBuildPolicy(
        allowed_licenses=frozenset({LICENSE}),
        minimum_records=minimum_records,
        development_fraction=0.6,
    )


def test_builder_excludes_cross_index_hash_source_and_capture_leakage(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    retrieval_root = tmp_path / "retrieval"
    forbidden_dhashes: set[str] = set()
    forbidden_ahashes: set[str] = set()

    retrieval_exact = _write_unique_image(
        retrieval_root,
        "exact.png",
        "retrieval-exact",
        forbidden_dhashes=forbidden_dhashes,
        forbidden_ahashes=forbidden_ahashes,
    )
    retrieval_near = _write_unique_image(
        retrieval_root,
        "near.png",
        "retrieval-near",
        forbidden_dhashes=forbidden_dhashes,
        forbidden_ahashes=forbidden_ahashes,
    )
    retrieval_source = _write_unique_image(
        retrieval_root,
        "source.png",
        "retrieval-source",
        forbidden_dhashes=forbidden_dhashes,
        forbidden_ahashes=forbidden_ahashes,
    )
    retrieval_sequence = _write_unique_image(
        retrieval_root,
        "sequence.png",
        "retrieval-sequence",
        forbidden_dhashes=forbidden_dhashes,
        forbidden_ahashes=forbidden_ahashes,
    )
    retrieval_records = [
        _record(retrieval_root, retrieval_exact, "retrieval-exact"),
        _record(retrieval_root, retrieval_near, "retrieval-near"),
        _record(
            retrieval_root,
            retrieval_source,
            "retrieval-source",
            image_id="shared-source-image",
        ),
        _record(
            retrieval_root,
            retrieval_sequence,
            "retrieval-sequence",
            sequence="shared-capture-sequence",
        ),
    ]

    candidate_root.mkdir(parents=True)
    candidate_exact = candidate_root / "candidate-exact.png"
    candidate_exact.write_bytes(retrieval_exact.read_bytes())
    candidate_near = candidate_root / "candidate-near.png"
    with Image.open(retrieval_near) as source:
        metadata = PngInfo()
        metadata.add_text("fixture", "different encoded bytes")
        source.save(candidate_near, format="PNG", pnginfo=metadata)
    assert candidate_near.read_bytes() != retrieval_near.read_bytes()
    assert _dhash(candidate_near) == _dhash(retrieval_near)

    candidate_source = _write_unique_image(
        candidate_root,
        "candidate-source.png",
        "candidate-source",
        forbidden_dhashes=forbidden_dhashes,
        forbidden_ahashes=forbidden_ahashes,
    )
    candidate_sequence = _write_unique_image(
        candidate_root,
        "candidate-sequence.png",
        "candidate-sequence",
        forbidden_dhashes=forbidden_dhashes,
        forbidden_ahashes=forbidden_ahashes,
    )
    accepted_one = _write_unique_image(
        candidate_root,
        "accepted-one.png",
        "accepted-one",
        forbidden_dhashes=forbidden_dhashes,
        forbidden_ahashes=forbidden_ahashes,
    )
    accepted_two = _write_unique_image(
        candidate_root,
        "accepted-two.png",
        "accepted-two",
        forbidden_dhashes=forbidden_dhashes,
        forbidden_ahashes=forbidden_ahashes,
    )
    accepted_three = _write_unique_image(
        candidate_root,
        "accepted-three.png",
        "accepted-three",
        forbidden_dhashes=forbidden_dhashes,
        forbidden_ahashes=forbidden_ahashes,
    )
    missing_city = _write_unique_image(
        candidate_root,
        "missing-city.png",
        "missing-city",
        forbidden_dhashes=forbidden_dhashes,
        forbidden_ahashes=forbidden_ahashes,
    )
    candidate_records = [
        _record(candidate_root, candidate_exact, "candidate-exact"),
        _record(candidate_root, candidate_near, "candidate-near"),
        _record(
            candidate_root,
            candidate_source,
            "candidate-source",
            image_id="shared-source-image",
        ),
        _record(
            candidate_root,
            candidate_sequence,
            "candidate-sequence",
            sequence="shared-capture-sequence",
        ),
        _record(candidate_root, missing_city, "candidate-missing-city", city=None),
        _record(
            candidate_root,
            accepted_one,
            "accepted-one",
            sequence="accepted-family-one",
        ),
        _record(
            candidate_root,
            accepted_two,
            "accepted-two",
            sequence="accepted-family-one",
        ),
        _record(
            candidate_root,
            accepted_three,
            "accepted-three",
            sequence="accepted-family-two",
        ),
    ]
    candidate_input = _write_input(candidate_root / "reference-input.json", candidate_records)
    retrieval_input = _write_input(retrieval_root / "reference-input.json", retrieval_records)

    builder = TurkiyeEvaluationSuiteBuilder(_policy())
    result = builder.build(
        candidate_reference_input=candidate_input,
        candidate_root=candidate_root,
        retrieval_reference_input=retrieval_input,
        retrieval_root=retrieval_root,
    )
    manifest = tmp_path / "evaluation.csv"
    receipt = tmp_path / "evaluation-receipt.json"
    builder.write(
        result,
        manifest_path=manifest,
        receipt_path=receipt,
        protected_inputs=(candidate_input, retrieval_input),
    )

    assert result.receipt.status == "ready"
    assert result.receipt.accepted_count == 3
    assert result.receipt.exclusion_counts["retrieval_sha256_overlap"] == 1
    assert result.receipt.exclusion_counts["retrieval_dhash_near_duplicate"] == 1
    assert result.receipt.exclusion_counts["retrieval_source_image_overlap"] == 1
    assert result.receipt.exclusion_counts["retrieval_capture_family_overlap"] == 1
    assert result.receipt.exclusion_counts["missing_locally_resolved_city"] == 1
    assert result.receipt.coordinate_uncertainty_verified_positive
    assert set(result.receipt.split_counts) == {"development", "validation"}
    assert all(result.receipt.split_counts.values())

    validated = EvaluationManifestLoader(allowed_licenses=frozenset({LICENSE})).load(
        manifest, candidate_root
    )
    records = [item.record for item in validated.assets]
    assert len(records) == 3
    assert {record.split for record in records} == {"calibration", "validation"}
    assert {record.source for record in records} == {"kartaview"}
    assert all(record.country_code == "TR" for record in records)
    assert all(-90 <= record.true_latitude <= 90 for record in records)
    assert all(-180 <= record.true_longitude <= 180 for record in records)
    assert all(record.city_or_area and record.region for record in records)
    assert all(record.license == LICENSE and record.attribution for record in records)
    same_family = [
        record
        for record in records
        if record.source_record_id
        in {"image-accepted-one", "image-accepted-two"}
    ]
    assert len(same_family) == 2
    assert len({record.split for record in same_family}) == 1
    assert len({record.capture_family_id for record in same_family}) == 1

    receipt_text = receipt.read_text(encoding="utf-8")
    assert "accepted-one" not in receipt_text
    assert "Fixture City" not in receipt_text
    assert "true_latitude" not in receipt_text
    assert "local_reference" not in receipt_text


def test_split_is_deterministic_by_capture_family_and_never_uses_holdout_split(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    retrieval_root = tmp_path / "retrieval"
    candidate_root.mkdir()
    retrieval_root.mkdir()
    dhashes: set[str] = set()
    ahashes: set[str] = set()
    records: list[ReferenceBuildRecord] = []
    for index, sequence in enumerate(("family-a", "family-a", "family-b", "family-c")):
        path = _write_unique_image(
            candidate_root,
            f"image-{index}.png",
            f"split-{index}",
            forbidden_dhashes=dhashes,
            forbidden_ahashes=ahashes,
        )
        records.append(
            _record(
                candidate_root,
                path,
                f"split-{index}",
                sequence=sequence,
            )
        )
    first_input = _write_input(candidate_root / "first.json", records)
    second_input = _write_input(candidate_root / "second.json", list(reversed(records)))
    retrieval_input = _write_input(retrieval_root / "reference-input.json", [])
    builder = TurkiyeEvaluationSuiteBuilder(_policy(minimum_records=4))

    first = builder.build(
        candidate_reference_input=first_input,
        candidate_root=candidate_root,
        retrieval_reference_input=retrieval_input,
        retrieval_root=retrieval_root,
    )
    second = builder.build(
        candidate_reference_input=second_input,
        candidate_root=candidate_root,
        retrieval_reference_input=retrieval_input,
        retrieval_root=retrieval_root,
    )

    assert first.manifest_csv == second.manifest_csv
    assert first.receipt.status == "ready"
    rows = list(csv.DictReader(first.manifest_csv.splitlines()))
    assert {row["split"] for row in rows} == {"calibration", "validation"}
    family_rows = [
        row
        for row in rows
        if row["source_record_id"] in {"image-split-0", "image-split-1"}
    ]
    assert len({row["split"] for row in family_rows}) == 1
    assert len({row["capture_family_id"] for row in family_rows}) == 1


def test_empty_and_small_inputs_have_honest_non_claim_status(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    retrieval_root = tmp_path / "retrieval"
    candidate_root.mkdir()
    retrieval_root.mkdir()
    retrieval_input = _write_input(retrieval_root / "reference-input.json", [])
    empty_input = _write_input(candidate_root / "empty.json", [])
    builder = TurkiyeEvaluationSuiteBuilder(_policy(minimum_records=100))

    empty = builder.build(
        candidate_reference_input=empty_input,
        candidate_root=candidate_root,
        retrieval_reference_input=retrieval_input,
        retrieval_root=retrieval_root,
    )
    assert empty.receipt.status == "empty"
    assert empty.receipt.accepted_count == 0
    assert not empty.receipt.accuracy_claim_allowed
    assert not empty.receipt.coordinate_uncertainty_verified_positive
    assert len(empty.manifest_csv.splitlines()) == 1

    image = _write_unique_image(
        candidate_root,
        "small.png",
        "small",
        forbidden_dhashes=set(),
        forbidden_ahashes=set(),
    )
    small_input = _write_input(
        candidate_root / "small.json",
        [_record(candidate_root, image, "small")],
    )
    small = builder.build(
        candidate_reference_input=small_input,
        candidate_root=candidate_root,
        retrieval_reference_input=retrieval_input,
        retrieval_root=retrieval_root,
    )
    assert small.receipt.status == "insufficient"
    assert small.receipt.accepted_count == 1
    assert not small.receipt.accuracy_claim_allowed


@pytest.mark.parametrize(
    ("field", "value"),
    (("coordinate_uncertainty_m", 0), ("attribution", "")),
)
def test_invalid_coordinate_or_provenance_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    candidate_root = tmp_path / "candidate"
    retrieval_root = tmp_path / "retrieval"
    candidate_root.mkdir()
    retrieval_root.mkdir()
    image = _write_unique_image(
        candidate_root,
        "candidate.png",
        "invalid-metadata",
        forbidden_dhashes=set(),
        forbidden_ahashes=set(),
    )
    record = _record(candidate_root, image, "candidate")
    payload = ReferenceBuildInput(
        schema_version="atlaslens-megaloc-reference-input-v1",
        records=(record,),
    ).model_dump(mode="json")
    payload["records"][0][field] = value
    candidate_input = candidate_root / "reference-input.json"
    candidate_input.write_text(json.dumps(payload), encoding="utf-8")
    retrieval_input = _write_input(retrieval_root / "reference-input.json", [])

    with pytest.raises(EvaluationSuiteBuildError, match="reference_input_invalid"):
        TurkiyeEvaluationSuiteBuilder(_policy()).build(
            candidate_reference_input=candidate_input,
            candidate_root=candidate_root,
            retrieval_reference_input=retrieval_input,
            retrieval_root=retrieval_root,
        )


def test_operator_help_and_implementation_have_no_target_specific_constants() -> None:
    repository = Path(__file__).resolve().parents[3]
    script = repository / "scripts" / "build_turkiye_evaluation_manifest.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "services" / "api" / "src")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert "--candidate-reference-input" in result.stdout
    assert "--retrieval-reference-input" in result.stdout
    assert "--allowed-license" in result.stdout
    module = (
        repository
        / "services"
        / "api"
        / "src"
        / "atlaslens_api"
        / "evaluation"
        / "turkiye_suite.py"
    )
    source = (module.read_text(encoding="utf-8") + script.read_text(encoding="utf-8")).casefold()
    for forbidden in ("erciyes", "talas", "private-operator-image.example.jpg", "province id 38"):
        assert forbidden not in source
