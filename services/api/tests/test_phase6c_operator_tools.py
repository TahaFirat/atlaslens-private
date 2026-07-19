from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"


def _image(path: Path, *, rotated_from: Path | None = None) -> None:
    if rotated_from is not None:
        with Image.open(rotated_from) as source:
            source.rotate(90, expand=True).save(path, "PNG")
        return
    image = Image.new("RGB", (320, 240), (18, 37, 59))
    draw = ImageDraw.Draw(image)
    for index in range(12):
        x = 12 + index * 23
        draw.rectangle(
            (x, 10 + (index % 4) * 35, x + 14, 220 - (index % 3) * 17),
            fill=((index * 31) % 255, 220 - index * 9, 30 + index * 13),
        )
    draw.ellipse((75, 55, 245, 195), outline=(250, 240, 20), width=8)
    draw.line((0, 230, 319, 25), fill=(10, 255, 180), width=5)
    image.save(path, "PNG")


def _unrelated_image(path: Path, *, variant: int = 0) -> None:
    image = Image.new("RGB", (320, 240))
    image.putdata(
        [
            (
                (x * 17 + y * 73 + variant * 41) % 256,
                (x * 97 + y * 11 + variant * 67) % 256,
                (x * 29 + y * 43 + variant * 83) % 256,
            )
            for y in range(240)
            for x in range(320)
        ]
    )
    image.save(path, "PNG")


def _dhash64(path: Path) -> str:
    with Image.open(path) as source:
        values = list(source.convert("L").resize((9, 8)).getdata())
    number = 0
    for row in range(8):
        for column in range(8):
            number = (number << 1) | int(
                values[row * 9 + column + 1] > values[row * 9 + column]
            )
    return f"{number:016x}"


def _holdout(path: Path, target: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-geolocation-evaluation-v1",
                "records": [
                    {
                        "id": "user-holdout-001",
                        "path": str(target),
                        "country": "TR",
                        "city": "Evaluation City",
                        "split": "final_holdout",
                        "usage": "holdout_only",
                        "allow_reference_index": False,
                        "allow_training": False,
                        "allow_prompt_ground_truth": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _reference_record(
    reference_id: str,
    relative: str,
    image: Path,
    *,
    asset_key: str | None = None,
) -> dict[str, object]:
    return {
        "reference_id": reference_id,
        "source": "mapillary",
        "source_family": "mapillary_public",
        "source_image_id": f"image-{reference_id}",
        "source_sequence_id": f"sequence-{reference_id}",
        "source_url": f"https://www.mapillary.com/app/{reference_id}",
        "latitude": 39.0,
        "longitude": 35.0,
        "coordinate_uncertainty_m": 50.0,
        "heading_degrees": 90.0,
        "captured_at": "2024-01-01T00:00:00Z",
        "country": "Turkiye",
        "province": "Ankara",
        "city": "Ankara",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Fixture creator",
        "asset_key": asset_key or f"reference/{reference_id}",
        "image_path": relative,
        "descriptor_path": None,
        "expected_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "expected_perceptual_hash": _dhash64(image),
    }


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_root = str(ROOT / "services" / "api" / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, environment.get("PYTHONPATH")) if item
    )
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_audit_cli_accepts_phase6c_reference_input_and_excludes_target_duplicates(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    _image(target)
    corpus = tmp_path / "corpus"
    images = corpus / "images"
    images.mkdir(parents=True)
    exact = images / "exact.png"
    rotated = images / "rotated.png"
    exact.write_bytes(target.read_bytes())
    _image(rotated, rotated_from=target)
    reference_input = corpus / "reference-input.json"
    reference_input.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-megaloc-reference-input-v1",
                "records": [
                    _reference_record("exact", "images/exact.png", exact),
                    _reference_record("rotated", "images/rotated.png", rotated),
                ],
            }
        ),
        encoding="utf-8",
    )
    holdout = tmp_path / "holdout.json"
    _holdout(holdout, target)
    output = tmp_path / "report.json"

    result = _run(
        "audit_reference_leakage.py",
        "--holdout-manifest",
        str(holdout),
        "--reference-input",
        str(reference_input),
        "--reference-root",
        str(corpus),
        "--output",
        str(output),
    )

    assert result.returncode == 1
    summary = json.loads(result.stdout)
    report = json.loads(output.read_text(encoding="utf-8"))
    reasons = {
        item["reference_key"]: set(item["reasons"]) for item in report["exclusions"]
    }
    assert summary["status"] == "failed"
    assert report["checked_reference_count"] == 2
    assert "target_sha256_match" in reasons["reference/exact"]
    assert "target_perceptual_near_duplicate" in reasons["reference/rotated"]
    assert report["checks"]["descriptor_similarity"] == (
        "not_run_no_real_descriptor_artifact"
    )


def test_audit_cli_rejects_reference_input_path_escape_without_disclosing_paths(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    outside = tmp_path / "outside.png"
    _image(target)
    _image(outside)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    reference_input = corpus / "reference-input.json"
    reference_input.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-megaloc-reference-input-v1",
                "records": [_reference_record("escape", "../outside.png", outside)],
            }
        ),
        encoding="utf-8",
    )
    holdout = tmp_path / "holdout.json"
    _holdout(holdout, target)
    output = tmp_path / "report.json"

    result = _run(
        "audit_reference_leakage.py",
        "--holdout-manifest",
        str(holdout),
        "--reference-input",
        str(reference_input),
        "--reference-root",
        str(corpus),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert json.loads(result.stdout) == {
        "status": "error",
        "code": "leakage_audit_failed",
    }
    assert not output.exists()
    assert str(outside) not in result.stdout + result.stderr


def test_audit_cli_atomically_writes_filtered_input_then_clean_rerun_passes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    _image(target)
    corpus = tmp_path / "corpus"
    images = corpus / "images"
    images.mkdir(parents=True)
    exact = images / "exact.png"
    safe = images / "safe.png"
    safe_two = images / "safe-two.png"
    exact.write_bytes(target.read_bytes())
    _unrelated_image(safe)
    _unrelated_image(safe_two, variant=1)
    reference_input = corpus / "reference-input.json"
    reference_input.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-megaloc-reference-input-v1",
                "records": [
                    _reference_record("exact", "images/exact.png", exact),
                    _reference_record("safe", "images/safe.png", safe),
                    _reference_record("safe-two", "images/safe-two.png", safe_two),
                ],
            }
        ),
        encoding="utf-8",
    )
    original = reference_input.read_text(encoding="utf-8")
    holdout = tmp_path / "holdout.json"
    _holdout(holdout, target)
    report = tmp_path / "initial-report.json"
    filtered = corpus / "reference-input.filtered.json"

    first = _run(
        "audit_reference_leakage.py",
        "--holdout-manifest",
        str(holdout),
        "--reference-input",
        str(reference_input),
        "--reference-root",
        str(corpus),
        "--write-filtered-reference-input",
        str(filtered),
        "--output",
        str(report),
    )

    assert first.returncode == 1
    summary = json.loads(first.stdout)
    assert summary["status"] == "failed"
    assert summary["filtered_output_written"] is True
    assert summary["filtered_reference_count"] == 2
    assert str(filtered) not in first.stdout
    assert "reference/exact" not in first.stdout
    assert reference_input.read_text(encoding="utf-8") == original
    filtered_payload = json.loads(filtered.read_text(encoding="utf-8"))
    assert filtered_payload["schema_version"] == (
        "atlaslens-megaloc-reference-input-v1"
    )
    assert [item["reference_id"] for item in filtered_payload["records"]] == [
        "safe",
        "safe-two",
    ]
    assert exact.is_file() and safe.is_file() and safe_two.is_file()
    assert not list(corpus.glob(f".{filtered.name}.*.tmp"))

    rerun_report = tmp_path / "filtered-report.json"
    rerun = _run(
        "audit_reference_leakage.py",
        "--holdout-manifest",
        str(holdout),
        "--reference-input",
        str(filtered),
        "--reference-root",
        str(corpus),
        "--output",
        str(rerun_report),
    )

    assert rerun.returncode == 0
    assert json.loads(rerun.stdout)["status"] == "passed"


def test_filtered_input_refuses_when_every_reference_is_excluded(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    _image(target)
    corpus = tmp_path / "corpus"
    images = corpus / "images"
    images.mkdir(parents=True)
    exact = images / "exact.png"
    exact.write_bytes(target.read_bytes())
    reference_input = corpus / "reference-input.json"
    reference_input.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-megaloc-reference-input-v1",
                "records": [_reference_record("exact", "images/exact.png", exact)],
            }
        ),
        encoding="utf-8",
    )
    holdout = tmp_path / "holdout.json"
    _holdout(holdout, target)
    filtered = corpus / "reference-input.filtered.json"

    result = _run(
        "audit_reference_leakage.py",
        "--holdout-manifest",
        str(holdout),
        "--reference-input",
        str(reference_input),
        "--reference-root",
        str(corpus),
        "--write-filtered-reference-input",
        str(filtered),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert json.loads(result.stdout) == {
        "status": "error",
        "code": "leakage_audit_failed",
    }
    assert not filtered.exists()


def test_filtered_input_refuses_ambiguous_exclusion_key_mapping(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    _image(target)
    corpus = tmp_path / "corpus"
    images = corpus / "images"
    images.mkdir(parents=True)
    exact = images / "exact.png"
    safe = images / "safe.png"
    exact.write_bytes(target.read_bytes())
    _unrelated_image(safe)
    reference_input = corpus / "reference-input.json"
    reference_input.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-megaloc-reference-input-v1",
                "records": [
                    _reference_record(
                        "exact",
                        "images/exact.png",
                        exact,
                        asset_key="reference/shared",
                    ),
                    _reference_record(
                        "safe",
                        "images/safe.png",
                        safe,
                        asset_key="reference/shared",
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    holdout = tmp_path / "holdout.json"
    _holdout(holdout, target)
    filtered = corpus / "reference-input.filtered.json"

    result = _run(
        "audit_reference_leakage.py",
        "--holdout-manifest",
        str(holdout),
        "--reference-input",
        str(reference_input),
        "--reference-root",
        str(corpus),
        "--write-filtered-reference-input",
        str(filtered),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert json.loads(result.stdout) == {
        "status": "error",
        "code": "leakage_audit_failed",
    }
    assert "reference/shared" not in result.stdout
    assert not filtered.exists()


def test_filtered_input_option_is_invalid_with_legacy_csv_interface(tmp_path: Path) -> None:
    result = _run(
        "audit_reference_leakage.py",
        "--reference-manifest",
        str(tmp_path / "references.csv"),
        "--reference-root",
        str(tmp_path),
        "--write-filtered-reference-input",
        str(tmp_path / "filtered.json"),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "requires --reference-input" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize(
    ("script", "expected_option"),
    [
        ("build_turkey_reference_index.py", "--input-manifest"),
        ("verify_reference_index.py", "--index"),
    ],
)
def test_reference_index_compatibility_entrypoints_support_help(
    script: str,
    expected_option: str,
) -> None:
    result = _run(script, "--help")

    assert result.returncode == 0
    assert expected_option in result.stdout


def test_audit_help_preserves_csv_interface_and_documents_phase6c_json() -> None:
    result = _run("audit_reference_leakage.py", "--help")

    assert result.returncode == 0
    assert "--reference-manifest" in result.stdout
    assert "--reference-input" in result.stdout
    assert "--write-filtered-reference-input" in result.stdout
