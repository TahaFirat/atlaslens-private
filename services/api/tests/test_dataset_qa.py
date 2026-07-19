from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import piexif
import pytest
from PIL import Image

from atlaslens_api.dataset_acquisition.cli import (
    build_dataset_parser,
    run_dataset_command,
)
from atlaslens_api.dataset_acquisition.cli import (
    main as dataset_main,
)
from atlaslens_api.dataset_qa import (
    DatasetQAError,
    DatasetQAPolicy,
    DatasetQAReportCatalog,
    DatasetQAReportWriter,
    DatasetQAScanner,
    safe_output_directory,
)
from atlaslens_api.schemas import DatasetQAReport

_HEADERS = (
    "asset_key",
    "image_path",
    "mask_path",
    "latitude",
    "longitude",
    "split",
    "capture_family_id",
    "sequence_id",
    "class_id",
    "label",
    "country_code",
)
_NOW = datetime(2026, 7, 12, tzinfo=UTC)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def row(name: str, key: str, **values: str) -> dict[str, str]:
    result = dict.fromkeys(_HEADERS, "")
    result.update({"asset_key": key, "image_path": name})
    result.update(values)
    return result


def image(path: Path, *, size: tuple[int, int] = (24, 16), value: int = 100) -> None:
    Image.new("RGB", size, (value, value, value)).save(path)


def scanner(policy: DatasetQAPolicy | None = None) -> DatasetQAScanner:
    return DatasetQAScanner(policy, clock=lambda: _NOW)


def codes(report: DatasetQAReport) -> set[str]:
    return {item.code for item in report.issues}


def test_image_quality_corruption_and_dimension_checks_are_structured(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image(images / "blank.jpg", value=0)
    image(images / "oversized.png", size=(32, 8), value=120)
    (images / "corrupt.jpg").write_bytes(b"truncated-image")
    policy = DatasetQAPolicy(max_side=24, max_decoded_pixels=10_000)

    report = scanner(policy).scan(images=images, report_id="image-quality").report

    assert {
        "image.near_blank",
        "image.blur",
        "image.underexposed",
        "image.dimensions_invalid",
        "image.corrupt_or_truncated",
    } <= codes(report)
    assert report.summary.scanned_images == 3
    assert all(item.asset_key.startswith("asset-") for item in report.issues)
    assert report.checks["images"] == "failed"


def test_mask_checks_cover_missing_corrupt_classes_dimensions_and_components(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    names = (
        "missing",
        "empty",
        "corrupt",
        "invalid",
        "mismatch",
        "background",
        "tiny",
        "palette",
    )
    for name in names:
        image(images / f"{name}.png")
    (masks / "empty.png").write_bytes(b"")
    (masks / "corrupt.png").write_bytes(b"not-a-mask")
    Image.new("L", (24, 16), 9).save(masks / "invalid.png")
    Image.new("L", (8, 8), 1).save(masks / "mismatch.png")
    Image.new("L", (24, 16), 0).save(masks / "background.png")
    tiny = Image.new("L", (24, 16), 0)
    tiny.putpixel((1, 1), 1)
    tiny.save(masks / "tiny.png")
    palette = Image.new("P", (24, 16), 1)
    palette.save(masks / "palette.png")
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [row(f"{name}.png", f"mask-{name}", mask_path=f"{name}.png") for name in names],
    )

    report = (
        scanner()
        .scan(
            images=images,
            masks=masks,
            manifest=manifest,
            report_id="mask-quality",
            allowed_class_ids=frozenset({0, 1}),
        )
        .report
    )

    assert {
        "mask.missing",
        "mask.empty_file",
        "mask.corrupt",
        "mask.invalid_class_id",
        "mask.dimension_mismatch",
        "mask.all_background",
        "mask.empty_foreground",
        "mask.tiny_components",
        "mask.coverage_too_high",
    } <= codes(report)
    assert report.summary.scanned_masks == 7
    assert report.summary.dataset_type == "segmentation"
    assert report.checks["masks"] == "failed"


def test_gps_duplicates_labels_and_split_family_sequence_leakage(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image(images / "first.jpg", value=80)
    (images / "second.jpg").write_bytes((images / "first.jpg").read_bytes())
    with Image.open(images / "first.jpg") as source:
        source.save(images / "near.jpg", quality=60)
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            row(
                "first.jpg",
                "asset-first",
                latitude="0",
                longitude="0",
                split="train",
                capture_family_id="family-a",
                sequence_id="sequence-a",
                label="one",
            ),
            row(
                "second.jpg",
                "asset-second",
                latitude="120",
                longitude="40",
                split="test",
                capture_family_id="family-a",
                sequence_id="sequence-a",
                label="two",
            ),
            row("near.jpg", "asset-near", split="validation", label="one"),
        ],
    )

    report = scanner().scan(images=images, manifest=manifest, report_id="leakage").report

    assert {
        "gps.zero_zero",
        "gps.possibly_swapped",
        "gps.out_of_range",
        "duplicate.exact",
        "duplicate.perceptual",
        "labels.conflicting_duplicate",
        "leakage.duplicate_cross_split",
        "leakage.capture_family_cross_split",
        "leakage.sequence_cross_split",
    } <= codes(report)
    assert "country_boundary_checks_unavailable" in report.limitations
    assert report.checks["country_boundary"] == "unavailable"


def test_exif_conflict_is_bucketed_without_coordinates(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    exif = {
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: ((20, 1), (0, 1), (0, 1)),
            piexif.GPSIFD.GPSLongitudeRef: "E",
            piexif.GPSIFD.GPSLongitude: ((20, 1), (0, 1), (0, 1)),
        }
    }
    Image.new("RGB", (24, 16), (50, 80, 120)).save(images / "gps.jpg", exif=piexif.dump(exif))
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [row("gps.jpg", "opaque-gps", latitude="10", longitude="10")],
    )

    report = scanner().scan(images=images, manifest=manifest, report_id="exif-conflict").report
    issue = next(item for item in report.issues if item.code == "gps.exif_conflict")
    assert issue.safe_metrics == {"distance_bucket": "100_km_or_more"}
    assert set(issue.safe_metrics) == {"distance_bucket"}


def test_reports_are_stable_read_only_atomic_safe_and_catalogued(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    pattern = Image.new("RGB", (32, 24))
    pattern.putdata(
        [((index * 31) % 255, (index * 17) % 255, (index * 7) % 255) for index in range(768)]
    )
    source = images / "private-original-name.jpg"
    pattern.save(source)
    before = source.read_bytes()
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [row(source.name, "opaque-one")])
    output = tmp_path / "catalog" / "qa-stable"
    output.parent.mkdir()
    destination = safe_output_directory(output, (images,))

    first = scanner().scan(
        images=images,
        manifest=manifest,
        report_id="qa-stable",
        contact_sheet=True,
    )
    second = scanner().scan(
        images=images,
        manifest=manifest,
        report_id="qa-stable",
        contact_sheet=True,
    )
    assert first.report == second.report
    DatasetQAReportWriter().write(first, destination)

    assert source.read_bytes() == before
    for filename in ("report.json", "issues.csv", "summary.md", "report.html", "contact-sheet.jpg"):
        assert (output / filename).is_file()
    serialized = (output / "report.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert source.name not in serialized
    with Image.open(output / "contact-sheet.jpg") as contact:
        assert not contact.getexif()
        assert max(contact.size) <= 8 * DatasetQAPolicy().contact_sheet_thumbnail
    catalog = DatasetQAReportCatalog(output.parent)
    assert catalog.get_report("qa-stable") == first.report
    assert catalog.list_reports().reports == [first.report.summary]
    with pytest.raises(DatasetQAError):
        catalog.get("../qa-stable")
    payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
    payload["issues"].append(
        {
            "code": "image.blur",
            "severity": "warning",
            "asset_key": "../private-path",
            "field": "image",
            "message_key": "dataset_qa.image.blur",
            "safe_metrics": {},
        }
    )
    (output / "report.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetQAError, match="issue_unsafe"):
        catalog.get("qa-stable")


def test_qa_rejects_containment_apply_and_preserves_existing_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image(images / "inside.jpg")
    image(tmp_path / "outside.jpg")
    manifest = tmp_path / "unsafe.csv"
    write_manifest(manifest, [row("../outside.jpg", "unsafe")])
    with pytest.raises(DatasetQAError, match="unsafe"):
        scanner().scan(images=images, manifest=manifest, report_id="unsafe")
    with pytest.raises(DatasetQAError, match="inside_source"):
        safe_output_directory(images / "report", (images,))

    parser = build_dataset_parser()
    assert (
        parser.parse_args(["qa", "--images", str(images), "--output", str(tmp_path / "qa")]).command
        == "qa"
    )
    assert parser.parse_args(["qa-report", "--output", str(tmp_path / "qa")]).command == "qa-report"
    old = parser.parse_args(
        [
            "review",
            "commons",
            "--titles",
            "titles.txt",
            "--output",
            "review.json",
            "--user-agent",
            "operator",
            "--country",
            "TR",
            "--continent",
            "Asia",
            "--geographic-cell",
            "cell",
            "--allow-license",
            "CC0-1.0",
        ]
    )
    assert old.command == "review" and old.source == "commons"
    assert (
        dataset_main(
            [
                "qa",
                "--images",
                str(images),
                "--output",
                str(tmp_path / "qa-apply"),
                "--apply",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "qa_apply_not_supported" in captured.err


def test_cli_qa_and_qa_report_round_trip(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image(images / "image.jpg", value=90)
    output = tmp_path / "qa-cli"
    result = run_dataset_command(["qa", "--images", str(images), "--output", str(output)])
    assert result["summary"]["report_id"] == "qa-cli"
    loaded = run_dataset_command(["qa-report", "--output", str(output)])
    assert loaded == result
    review = run_dataset_command(["review", "--qa-report", str(output), "--severity", "warning"])
    assert review["status"] == "review"
    assert review["read_only"] is True
