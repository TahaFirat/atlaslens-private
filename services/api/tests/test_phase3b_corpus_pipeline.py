from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from atlaslens_api.corpus_pipeline import (
    CheckpointCompatibilityError,
    CheckpointStore,
    CorpusIngestor,
    CorpusPipelineError,
    IngestedAsset,
    IngestionConfig,
    ManifestAsset,
    ManifestValidationError,
    PipelineInterrupted,
    PipelineState,
    SourcePolicy,
    filter_leakage,
    inspect_image,
    load_ingested_corpus,
    load_split_lock,
    perceptual_hamming_distance,
    plan_revocation,
    read_manifest,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
MANIFEST_SCHEMA = REPOSITORY_ROOT / "config" / "corpus" / "manifest-schema-v1.json"
SOURCE_POLICY = REPOSITORY_ROOT / "config" / "corpus" / "source-policy-v1.json"
FIRST_PARTY_SOURCE = "First-party AtlasLens-captured imagery"


def _pixels(seed: int) -> list[tuple[int, int, int]]:
    values = []
    for y in range(24):
        for x in range(32):
            value = (x * (17 + seed * 3) + y * (31 + seed * 5) + (x * y % 19) * 7) % 256
            values.append((value, (value * 3 + seed * 29) % 256, (255 - value + seed) % 256))
    return values


def _write_png(
    corpus_root: Path,
    asset_id: str,
    *,
    seed: int,
    compress_level: int = 6,
) -> Path:
    assets = corpus_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    path = assets / f"{asset_id}.png"
    image = Image.new("RGB", (32, 24))
    image.putdata(_pixels(seed))
    image.save(path, format="PNG", compress_level=compress_level)
    return path


def _manifest_row(
    path: Path,
    asset_id: str,
    *,
    role: str = "train",
    latitude: float = 39.0,
    longitude: float = 35.0,
    sequence_id: str | None = None,
    capture_run_id: str | None = None,
    source_name: str = FIRST_PARTY_SOURCE,
    decision: str = "FIRST_PARTY_ONLY",
    acquisition_ready: bool = True,
    revocation_status: str = "ACTIVE",
    privacy_state: str = "BLURRED_BY_ATLASLENS",
) -> dict[str, object]:
    identity = inspect_image(path, max_bytes=1024 * 1024, max_pixels=10_000)
    tier = "tier_3_locked_holdout" if role == "holdout" else "tier_1_national_recall"
    return {
        "corpus_version": "atlaslens-turkiye-corpus-synthetic-v1",
        "asset_id": asset_id,
        "source_asset_id": f"source-{asset_id}",
        "source_name": source_name,
        "source_url": f"https://fixture.invalid/assets/{asset_id}",
        "contributor_or_owner": "synthetic-fixture-owner",
        "capture_timestamp": "2026-07-16T08:00:00Z",
        "acquisition_timestamp": "2026-07-16T09:00:00Z",
        "latitude": latitude,
        "longitude": longitude,
        "coordinate_accuracy_m": 5.0,
        "heading_degrees": 90.0,
        "sequence_id": sequence_id,
        "capture_run_id": capture_run_id,
        "image_sha256": identity.sha256,
        "perceptual_hash": identity.perceptual_hash,
        "perceptual_hash_algorithm": "dhash64-v1",
        "width_px": identity.width_px,
        "height_px": identity.height_px,
        "mime_type": identity.mime_type,
        "asset_type": "street_level",
        "province_code": "TR-38",
        "urbanicity": "urban_core",
        "road_class": "local_urban_street",
        "scene_type": "city_center",
        "road_context": "ordinary_road",
        "terrain_class": "flat",
        "vegetation_state": "leaf_on",
        "season": "summer",
        "license_identifier": "AtlasLens-first-party-fixture",
        "license_url": "https://fixture.invalid/license",
        "attribution_text": "Synthetic fixture; not production imagery",
        "source_policy_decision": decision,
        "commercial_use_decision": decision,
        "derivative_index_decision": decision,
        "personal_data_blur_state": privacy_state,
        "deletion_revocation_state": {
            "status": revocation_status,
            "last_checked_at": "2026-07-16T09:00:00Z",
            "request_id": "request-1" if revocation_status != "ACTIVE" else None,
        },
        "provenance_receipt": {
            "receipt_id": f"receipt-{asset_id}",
            "source_policy_version": "2026-07-16",
            "evidence_date": "2026-07-16",
            "acquisition_method": "first_party_capture",
            "receipt_sha256": "a" * 64,
        },
        "spatial_split": f"cell-{asset_id}",
        "role": role,
        "tier": tier,
        "acquisition_ready": acquisition_ready,
    }


def _write_manifest(corpus_root: Path, rows: list[dict[str, object]]) -> Path:
    path = corpus_root / "manifest.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _ingestor(
    corpus_root: Path,
    work_root: Path,
    *,
    checkpoint_root: Path | None = None,
    strict: bool = True,
    hamming_threshold: int = 0,
    spatial_radius_m: float = 50.0,
) -> CorpusIngestor:
    return CorpusIngestor(
        corpus_root=corpus_root,
        work_root=work_root,
        checkpoint_root=checkpoint_root,
        manifest_schema_path=MANIFEST_SCHEMA,
        source_policy_path=SOURCE_POLICY,
        config=IngestionConfig(
            strict_rejections=strict,
            near_duplicate_hamming_threshold=hamming_threshold,
            spatial_leakage_radius_m=spatial_radius_m,
            descriptor_version="synthetic-descriptor-v1",
            index_version="synthetic-index-v1",
        ),
    )


def test_phase3a_schema_compatible_manifest_and_rights_admission(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    image = _write_png(corpus_root, "asset-1", seed=1)
    manifest = _write_manifest(corpus_root, [_manifest_row(image, "asset-1")])

    result = read_manifest(
        corpus_root,
        manifest,
        MANIFEST_SCHEMA,
        config=IngestionConfig(),
    )
    policy = SourcePolicy.load(SOURCE_POLICY)
    admission = policy.enforce(result.assets[0])

    assert result.rejections == ()
    assert admission.source_id == "atlaslens_first_party_imagery"
    assert admission.permission_state == "validated"
    assert result.assets[0].model_dump(mode="json").keys() <= ManifestAsset.model_fields.keys()


def test_parent_and_tile_provenance_survives_ingestion(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    work_root = tmp_path / "work"
    image = _write_png(corpus_root, "lineage", seed=2)
    row = _manifest_row(image, "lineage")
    row["parent_source_asset_id"] = "parent-panorama-17"
    row["source_tile_id"] = "source-tile-42"
    manifest = _write_manifest(corpus_root, [row])

    result = _ingestor(corpus_root, work_root).ingest(
        manifest,
        checkpoint_path="checkpoint.json",
    )
    persisted = load_ingested_corpus(work_root)

    assert result.corpus.assets[0].parent_source_asset_id == "parent-panorama-17"
    assert result.corpus.assets[0].source_tile_id == "source-tile-42"
    assert persisted.assets[0].parent_source_asset_id == "parent-panorama-17"
    assert persisted.assets[0].source_tile_id == "source-tile-42"


def test_rights_privacy_revocation_and_coordinate_gates_fail_closed(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    images = [
        _write_png(corpus_root, "blocked", seed=1),
        _write_png(corpus_root, "revoked", seed=2),
        _write_png(corpus_root, "privacy", seed=3),
        _write_png(corpus_root, "coordinate", seed=4),
        _write_png(corpus_root, "valid", seed=5),
    ]
    policy = SourcePolicy.load(SOURCE_POLICY)
    blocked_source = next(
        entry.source_name
        for entry in policy.document.sources
        if entry.current_decision == "BLOCKED"
    )
    rows = [
        _manifest_row(
            images[0],
            "blocked",
            source_name=blocked_source,
            decision="BLOCKED",
            acquisition_ready=False,
        ),
        _manifest_row(
            images[1],
            "revoked",
            revocation_status="REVOKED",
            acquisition_ready=False,
        ),
        _manifest_row(
            images[2],
            "privacy",
            privacy_state="REVIEW_REQUIRED",
            acquisition_ready=False,
        ),
        _manifest_row(images[3], "coordinate"),
        _manifest_row(images[4], "valid"),
    ]
    rows[3]["latitude"] = 91.0
    manifest = _write_manifest(corpus_root, rows)

    with pytest.raises(CorpusPipelineError):
        _ingestor(corpus_root, tmp_path / "strict-work").prepare(manifest)

    result = _ingestor(corpus_root, tmp_path / "work", strict=False).prepare(manifest)
    reasons = {rejection.asset_id: rejection.reason_code for rejection in result.corpus.rejections}
    assert [asset.asset_id for asset in result.corpus.assets] == ["valid"]
    assert reasons["blocked"] == "source_policy_not_acquisition_ready"
    assert reasons["revoked"] in {"asset_not_acquisition_ready", "asset_revoked_or_unavailable"}
    assert reasons["privacy"] in {"asset_not_acquisition_ready", "privacy_review_incomplete"}
    assert reasons["coordinate"] == "manifest_row_invalid"


def test_missing_provenance_is_a_manifest_rejection(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    image = _write_png(corpus_root, "asset-1", seed=1)
    row = _manifest_row(image, "asset-1")
    row.pop("provenance_receipt")
    manifest = _write_manifest(corpus_root, [row])

    result = read_manifest(
        corpus_root,
        manifest,
        MANIFEST_SCHEMA,
        config=IngestionConfig(strict_rejections=False),
    )

    assert result.assets == ()
    assert result.rejections[0].reason_code == "manifest_row_invalid"


def test_manifest_traversal_and_asset_symlink_escape_are_rejected(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    outside_manifest = tmp_path / "outside.json"
    outside_manifest.write_text("[]", encoding="utf-8")

    with pytest.raises(CorpusPipelineError, match="path_traversal_rejected"):
        read_manifest(
            corpus_root,
            Path("..") / "outside.json",
            MANIFEST_SCHEMA,
            config=IngestionConfig(),
        )

    outside_image = _write_png(tmp_path / "outside-corpus", "linked", seed=4)
    valid_image = _write_png(corpus_root, "valid", seed=5)
    assets = corpus_root / "assets"
    assets.mkdir(exist_ok=True)
    linked = assets / "linked.png"
    try:
        linked.symlink_to(outside_image)
    except OSError:
        pytest.skip("host does not permit symlink creation")
    row = _manifest_row(outside_image, "linked")
    manifest = _write_manifest(corpus_root, [row, _manifest_row(valid_image, "valid")])
    result = _ingestor(corpus_root, tmp_path / "work", strict=False).prepare(manifest)
    assert result.corpus.rejections[0].reason_code == "path_link_rejected"


def test_sha256_exact_and_dhash_near_duplicates_are_deterministic(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    first = _write_png(corpus_root, "first", seed=7, compress_level=0)
    exact = corpus_root / "assets" / "exact.png"
    shutil.copyfile(first, exact)
    near = _write_png(corpus_root, "near", seed=7, compress_level=9)
    identities = [
        inspect_image(path, max_bytes=1024 * 1024, max_pixels=10_000)
        for path in (first, exact, near)
    ]

    assert identities[0].sha256 == identities[1].sha256
    assert identities[0].sha256 != identities[2].sha256
    assert identities[0].perceptual_hash == identities[2].perceptual_hash
    assert perceptual_hamming_distance(
        identities[0].perceptual_hash, identities[2].perceptual_hash
    ) == 0

    manifest = _write_manifest(
        corpus_root,
        [
            _manifest_row(first, "first", role="holdout"),
            _manifest_row(exact, "exact"),
            _manifest_row(near, "near"),
        ],
    )
    result = _ingestor(corpus_root, tmp_path / "work", strict=False).prepare(manifest)
    reasons = {rejection.asset_id: rejection.reason_code for rejection in result.corpus.rejections}
    assert [asset.asset_id for asset in result.corpus.assets] == ["first"]
    assert reasons == {"exact": "exact_duplicate", "near": "near_duplicate"}


def test_sequence_capture_run_and_spatial_cross_split_leakage() -> None:
    def asset(
        asset_id: str,
        *,
        role: str,
        digest: str,
        phash: str,
        latitude: float,
        sequence: str | None = None,
        capture_run: str | None = None,
        spatial_split: str | None = None,
        parent_source_asset_id: str | None = None,
        source_tile_id: str | None = None,
        source_id: str = "atlaslens_first_party_imagery",
        source_name: str = FIRST_PARTY_SOURCE,
    ) -> IngestedAsset:
        tier = "tier_3_locked_holdout" if role == "holdout" else "tier_1_national_recall"
        return IngestedAsset(
            asset_id=asset_id,
            source_asset_id=f"source-{asset_id}",
            parent_source_asset_id=parent_source_asset_id,
            source_tile_id=source_tile_id,
            source_id=source_id,
            source_name=source_name,
            file_locator=f"assets/{asset_id}.png",
            image_sha256=digest * 64,
            perceptual_hash=phash,
            rights_decision="FIRST_PARTY_ONLY",
            license_identifier="fixture",
            license_url="https://fixture.invalid/license",
            attribution_text="fixture",
            provenance_receipt_id=f"receipt-{asset_id}",
            provenance_receipt_sha256="f" * 64,
            privacy_review_state="NOT_DETECTED",
            role=role,
            tier=tier,
            spatial_split=spatial_split or f"cell-{asset_id}",
            sampling_cell=spatial_split or f"cell-{asset_id}",
            province_code="TR-38",
            latitude=latitude,
            longitude=35.0,
            coordinate_accuracy_m=5,
            contributor_or_owner="fixture-owner",
            capture_timestamp="2026-07-16T08:00:00Z",
            capture_run_id=capture_run,
            sequence_id=sequence,
            descriptor_version="descriptor-v1",
            index_version="index-v1",
        )

    values = (
        asset(
            "holdout",
            role="holdout",
            digest="a",
            phash="0000000000000000",
            latitude=39.0,
            sequence="sequence-a",
            capture_run="run-a",
            parent_source_asset_id="parent-shared",
            source_tile_id="tile-shared",
        ),
        asset(
            "sequence-leak",
            role="train",
            digest="b",
            phash="1111111111111111",
            latitude=40.0,
            sequence="sequence-a",
            spatial_split="cell-holdout",
        ),
        asset(
            "run-leak",
            role="train",
            digest="c",
            phash="3333333333333333",
            latitude=41.0,
            capture_run="run-a",
            source_id="cross_source_partner",
            source_name="Cross-source synthetic partner",
        ),
        asset(
            "spatial-leak",
            role="train",
            digest="d",
            phash="7777777777777777",
            latitude=39.0001,
        ),
        asset(
            "a-sequence-base",
            role="train",
            digest="e",
            phash="5555555555555555",
            latitude=42.0,
            sequence="sequence-spatial",
            spatial_split="cell-sequence-a",
        ),
        asset(
            "z-sequence-spatial-leak",
            role="train",
            digest="f",
            phash="9999999999999999",
            latitude=43.0,
            sequence="sequence-spatial",
            spatial_split="cell-sequence-b",
        ),
        asset(
            "parent-leak",
            role="train",
            digest="1",
            phash="aaaaaaaaaaaaaaaa",
            latitude=44.0,
            parent_source_asset_id="parent-shared",
        ),
        asset(
            "tile-leak",
            role="train",
            digest="2",
            phash="bbbbbbbbbbbbbbbb",
            latitude=45.0,
            source_tile_id="tile-shared",
        ),
    )

    decision = filter_leakage(
        values,
        near_duplicate_hamming_threshold=0,
        spatial_leakage_radius_m=50,
    )

    assert [item.asset_id for item in decision.accepted] == ["a-sequence-base", "holdout"]
    assert decision.excluded_reason_by_asset == {
        "parent-leak": "source_parent_or_tile_cross_role",
        "run-leak": "capture_run_cross_split",
        "sequence-leak": "sequence_cross_split",
        "spatial-leak": "spatial_cross_split",
        "tile-leak": "source_parent_or_tile_cross_role",
        "z-sequence-spatial-leak": "sequence_cross_spatial_split",
    }


def test_atomic_ingestion_interruption_resume_idempotence_and_config_mismatch(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "corpus"
    work_root = tmp_path / "work"
    checkpoint_root = tmp_path / "checkpoints"
    image = _write_png(corpus_root, "asset-1", seed=11)
    manifest = _write_manifest(corpus_root, [_manifest_row(image, "asset-1")])
    ingestor = _ingestor(corpus_root, work_root, checkpoint_root=checkpoint_root)

    with pytest.raises(PipelineInterrupted):
        ingestor.ingest(
            manifest,
            checkpoint_path="checkpoint.json",
            interrupt_before_complete=PipelineState.INGESTED,
        )
    interrupted = CheckpointStore(checkpoint_root, "checkpoint.json").load()
    assert interrupted.state == PipelineState.RIGHTS_VALIDATED
    assert interrupted.active_state == PipelineState.INGESTED
    assert (checkpoint_root / "checkpoint.json").is_file()
    assert not (work_root / "checkpoint.json").exists()

    resumed = ingestor.ingest(manifest, checkpoint_path="checkpoint.json", resume=True)
    before = {
        name: (work_root / name).read_bytes()
        for name in (
            "ingested-assets.json",
            "corpus.json",
            "leakage-report.json",
            "split-lock.json",
        )
    }
    repeated = ingestor.ingest(manifest, checkpoint_path="checkpoint.json", resume=True)

    assert resumed.checkpoint.state == PipelineState.SPLIT_LOCKED
    assert repeated.checkpoint.state == PipelineState.SPLIT_LOCKED
    assert resumed.split_lock.lock_sha256 == repeated.split_lock.lock_sha256
    assert before == {name: (work_root / name).read_bytes() for name in before}

    incompatible = CorpusIngestor(
        corpus_root=corpus_root,
        work_root=work_root,
        checkpoint_root=checkpoint_root,
        manifest_schema_path=MANIFEST_SCHEMA,
        source_policy_path=SOURCE_POLICY,
        config=IngestionConfig(spatial_leakage_radius_m=51),
    )
    with pytest.raises(CheckpointCompatibilityError, match="checkpoint_config_mismatch"):
        incompatible.ingest(manifest, checkpoint_path="checkpoint.json", resume=True)


def test_split_lock_detects_tampering(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    work_root = tmp_path / "work"
    image = _write_png(corpus_root, "asset-1", seed=13)
    manifest = _write_manifest(corpus_root, [_manifest_row(image, "asset-1")])
    ingested = _ingestor(corpus_root, work_root).ingest(
        manifest,
        checkpoint_path="checkpoint.json",
    )
    assert load_split_lock(work_root) == ingested.split_lock

    split_path = work_root / "split-lock.json"
    payload = ingested.split_lock.model_dump(mode="json")
    payload["assignments"][0]["spatial_split"] = "tampered-cell"
    split_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="split_lock_invalid"):
        load_split_lock(work_root)

    split_path.write_text("{", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="split_lock_invalid"):
        load_split_lock(work_root)


def test_revocation_impact_is_deterministic_and_artifact_scoped(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    image = _write_png(corpus_root, "asset-1", seed=15)
    manifest = _write_manifest(corpus_root, [_manifest_row(image, "asset-1")])
    corpus = _ingestor(corpus_root, tmp_path / "work").prepare(manifest).corpus

    first = plan_revocation(
        corpus.assets,
        ["asset-1"],
        artifact_membership={"asset-1": ["index/shard-0001", "descriptors/batch-0001"]},
    )
    second = plan_revocation(
        corpus.assets,
        ["asset-1"],
        artifact_membership={"asset-1": ["descriptors/batch-0001", "index/shard-0001"]},
    )

    assert first == second
    assert first.rebuild_required
    assert first.impacted_artifacts == ("descriptors/batch-0001", "index/shard-0001")
    assert first.evidence_semantics.endswith("not_legal_or_cryptographic_certification")
