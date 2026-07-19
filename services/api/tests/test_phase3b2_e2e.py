from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from atlaslens_api.capture_import import (
    CaptureImporter,
    CaptureImportError,
    CapturePlan,
    FirstPartyRights,
    PrivacyDecision,
    RevocationUpdate,
    build_capture_manifest,
    revoke_capture_asset,
    update_capture_privacy,
)
from atlaslens_api.corpus_cli import main as corpus_cli_main
from atlaslens_api.corpus_index.benchmark import (
    BenchmarkQuery,
    compute_locked_holdout_hash,
    evaluate_locked_holdout,
)
from atlaslens_api.corpus_index.descriptors import build_descriptors
from atlaslens_api.corpus_index.index import build_index
from atlaslens_api.corpus_index.models import (
    AssetProvenance,
    DescriptorAsset,
    FloatMatrix,
)
from atlaslens_api.corpus_pipeline import (
    CorpusIngestor,
    IngestionConfig,
    ManifestAsset,
    inspect_image,
)
from atlaslens_api.megaloc_adapter import (
    MegaLocAdapterConfig,
    MegaLocApproval,
    MegaLocDescriptorProvider,
)
from atlaslens_api.turkiye_reference import TurkiyeReferenceIndexProvider

REPOSITORY_ROOT = Path(__file__).parents[3]
MANIFEST_SCHEMA = REPOSITORY_ROOT / "config" / "corpus" / "manifest-schema-v1.json"
SOURCE_POLICY = REPOSITORY_ROOT / "config" / "corpus" / "source-policy-v1.json"
PREPROCESSING_VERSION = "rgb-imagenet-max560-multiple14-v1"


def _write_synthetic_png(root: Path, asset_id: str, *, seed: int = 1) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{asset_id}.png"
    image = Image.new("RGB", (32, 24))
    image.putdata(
        [
            (
                (x * (17 + seed) + y * (31 + seed * 2)) % 256,
                (x * (11 + seed * 3) + y * 23 + seed * 29) % 256,
                (255 - x * 5 - y * (3 + seed) + seed * 7) % 256,
            )
            for y in range(24)
            for x in range(32)
        ]
    )
    image.save(path, format="PNG")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approved_megaloc_config(root: Path) -> MegaLocAdapterConfig:
    source_dir = root / "source"
    artifact_dir = root / "artifacts"
    receipt_dir = root / "receipts"
    source_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    receipt_dir.mkdir()
    source_code = source_dir / "model.py"
    source_license = source_dir / "LICENSE"
    artifact = artifact_dir / "model.safetensors"
    receipt = receipt_dir / "artifact.json"
    source_code.write_text("# generated test-only source marker\n", encoding="utf-8")
    source_license.write_text("Apache-2.0 generated fixture\n", encoding="utf-8")
    artifact.write_bytes(b"phase3b2-e2e-synthetic-safetensors-placeholder")
    artifact_sha256 = _sha256(artifact)
    model_revision = "7cb9f797-synthetic"
    descriptor_version = (
        f"megaloc-{model_revision[:8]}-{artifact_sha256[:12]}-max560-imagenet-rgb-v1"
    )
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "atlaslens-model-receipt-v1",
                "provider": "megaloc",
                "source_revision": "synthetic-source-revision",
                "model_id": "megaloc",
                "model_revision": model_revision,
                "weight_file": artifact.name,
                "weight_size": artifact.stat().st_size,
                "weight_sha256": artifact_sha256,
                "license": "Apache-2.0",
                "datasets_downloaded": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return MegaLocAdapterConfig(
        project_root=root,
        source_dir=source_dir,
        source_code_path=source_code,
        source_code_sha256=_sha256(source_code),
        source_revision="synthetic-source-revision",
        source_license_path=source_license,
        source_license_sha256=_sha256(source_license),
        source_license_spdx="Apache-2.0",
        artifact_path=artifact,
        artifact_sha256=artifact_sha256,
        artifact_size=artifact.stat().st_size,
        artifact_format="safetensors",
        receipt_path=receipt,
        receipt_sha256=_sha256(receipt),
        smoke_receipt_path=receipt_dir / "smoke.json",
        model_id="megaloc",
        model_revision=model_revision,
        weight_license_spdx="Apache-2.0",
        descriptor_version=descriptor_version,
        descriptor_dimension=8448,
        preprocessing_id=PREPROCESSING_VERSION,
        maximum_edge=560,
        device="cpu",
        max_batch_size=4,
        max_input_bytes=1024 * 1024,
        max_image_pixels=1_000_000,
        deterministic_algorithms=True,
        reduce_batch_on_cuda_oom=True,
        approval=MegaLocApproval(
            artifact_hash_approved=True,
            source_code_approved=True,
            source_license_approved=True,
            weight_license_approved=True,
            descriptor_contract_approved=True,
            preprocessing_approved=True,
            production_use_approved=True,
            approval_record_id="synthetic-approval",
            weight_license_record_id="synthetic-license",
        ),
    )


class _InjectedMegaLocBackend:
    def __init__(self, device: str) -> None:
        self._device = device
        self.calls: list[int] = []

    @property
    def device(self) -> str:
        return self._device

    def describe_batch(self, locators: Sequence[Path]) -> FloatMatrix:
        self.calls.append(len(locators))
        vectors = np.zeros((len(locators), 8448), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors

    def recover_cuda_oom(self) -> None:
        return None

    def close(self) -> None:
        return None


def _megaloc_provider(
    config: MegaLocAdapterConfig,
) -> tuple[MegaLocDescriptorProvider, list[_InjectedMegaLocBackend]]:
    created: list[_InjectedMegaLocBackend] = []

    def factory(
        _config: MegaLocAdapterConfig,
        device: str,
    ) -> _InjectedMegaLocBackend:
        backend = _InjectedMegaLocBackend(device)
        created.append(backend)
        return backend

    return MegaLocDescriptorProvider(config, backend_factory=factory), created


def _write_track(path: Path, *, invalid_gps: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first_latitude = "95.0" if invalid_gps else "38.7205"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<gpx version="1.1"><trk><trkseg>'
        f'<trkpt lat="{first_latitude}" lon="35.4826">'
        "<time>2026-07-17T08:00:00Z</time></trkpt>"
        '<trkpt lat="38.7205" lon="35.4826">'
        "<time>2026-07-17T08:00:05Z</time></trkpt>"
        '<trkpt lat="38.72104" lon="35.4826">'
        "<time>2026-07-17T08:00:15Z</time></trkpt>"
        '<trkpt lat="38.72158" lon="35.4826">'
        "<time>2026-07-17T08:00:25Z</time></trkpt>"
        "</trkseg></trk></gpx>",
        encoding="utf-8",
    )


def _capture_plan(*, track_locator: str = "tracks/route.gpx") -> CapturePlan:
    return CapturePlan(
        plan_id="phase3b2-e2e-plan",
        input_mode="ordered_frames_gpx",
        corpus_version="atlaslens-turkiye-corpus-phase3b2-e2e-v1",
        media_locators=(
            "frames/reference-1.png",
            "frames/reference-duplicate.png",
            "frames/reference-2.png",
            "frames/reference-3.png",
        ),
        track_locator=track_locator,
        frame_timestamps=(
            datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC),
            datetime(2026, 7, 17, 8, 0, 5, tzinfo=UTC),
            datetime(2026, 7, 17, 8, 0, 15, tzinfo=UTC),
            datetime(2026, 7, 17, 8, 0, 25, tzinfo=UTC),
        ),
        acquisition_timestamp=datetime(2026, 7, 17, 9, 0, tzinfo=UTC),
        capture_run_id="phase3b2-e2e-run",
        sequence_id="phase3b2-e2e-sequence",
        device_id="phase3b2-e2e-device",
        contributor_or_owner="phase3b2-synthetic-owner",
        province_code="TR-38",
        spatial_split="phase3b2-e2e-cell",
        sample_distance_m=50,
        urbanicity="urban_core",
        road_class="local_urban_street",
        scene_type="city_center",
        road_context="ordinary_road",
        terrain_class="flat",
        vegetation_state="leaf_on",
        season="summer",
        rights=FirstPartyRights(
            source_policy_version="2026-07-16",
            source_policy_evidence_date=datetime(2026, 7, 16, tzinfo=UTC).date(),
            license_identifier="AtlasLens-first-party-synthetic-fixture",
            license_url="https://fixture.invalid/license",
            attribution_text="Generated synthetic fixture; never production imagery",
            capture_policy_sha256="1" * 64,
            contributor_assignment_sha256="2" * 64,
            privacy_notice_sha256="3" * 64,
        ),
    )


def _write_capture_inputs(input_root: Path, *, invalid_gps: bool = False) -> None:
    _write_synthetic_png(input_root / "frames", "reference-1", seed=11)
    _write_synthetic_png(input_root / "frames", "reference-duplicate", seed=11)
    _write_synthetic_png(input_root / "frames", "reference-2", seed=29)
    _write_synthetic_png(input_root / "frames", "reference-3", seed=47)
    _write_track(input_root / "tracks" / "route.gpx", invalid_gps=invalid_gps)


def _phase3b1_manifest_row(path: Path, asset_id: str) -> dict[str, object]:
    identity = inspect_image(path, max_bytes=1024 * 1024, max_pixels=10_000)
    return {
        "corpus_version": "atlaslens-turkiye-corpus-phase3b2-synthetic-v1",
        "asset_id": asset_id,
        "source_asset_id": f"capture-{asset_id}",
        "source_name": "First-party AtlasLens-captured imagery",
        "source_url": f"https://fixture.invalid/capture/{asset_id}",
        "contributor_or_owner": "phase3b2-synthetic-owner",
        "capture_timestamp": "2026-07-17T08:00:00Z",
        "acquisition_timestamp": "2026-07-17T09:00:00Z",
        "latitude": 38.7205,
        "longitude": 35.4826,
        "coordinate_accuracy_m": 5.0,
        "heading_degrees": 90.0,
        "sequence_id": "synthetic-sequence-1",
        "capture_run_id": "synthetic-run-1",
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
        "license_identifier": "AtlasLens-first-party-synthetic-fixture",
        "license_url": "https://fixture.invalid/license",
        "attribution_text": "Generated synthetic fixture; never production imagery",
        "source_policy_decision": "FIRST_PARTY_ONLY",
        "commercial_use_decision": "FIRST_PARTY_ONLY",
        "derivative_index_decision": "FIRST_PARTY_ONLY",
        "personal_data_blur_state": "BLURRED_BY_ATLASLENS",
        "deletion_revocation_state": {
            "status": "ACTIVE",
            "last_checked_at": "2026-07-17T09:00:00Z",
            "request_id": None,
        },
        "provenance_receipt": {
            "receipt_id": f"receipt-{asset_id}",
            "source_policy_version": "2026-07-16",
            "evidence_date": "2026-07-17",
            "acquisition_method": "first_party_capture",
            "receipt_sha256": "a" * 64,
        },
        "spatial_split": "phase3b2-synthetic-cell-1",
        "role": "train",
        "tier": "tier_1_national_recall",
        "acquisition_ready": True,
    }


def test_generated_capture_row_is_phase3b1_manifest_compatible(tmp_path: Path) -> None:
    image = _write_synthetic_png(tmp_path / "capture", "reference-1")
    row = _phase3b1_manifest_row(image, "reference-1")

    parsed = ManifestAsset.model_validate_json(json.dumps(row))

    assert parsed.asset_id == "reference-1"
    assert parsed.capture_run_id == "synthetic-run-1"
    assert parsed.sequence_id == "synthetic-sequence-1"
    assert parsed.personal_data_blur_state == "BLURRED_BY_ATLASLENS"


def test_capture_e2e_rejects_invalid_gps_before_import(tmp_path: Path) -> None:
    input_root = tmp_path / "invalid-gps-input"
    _write_capture_inputs(input_root, invalid_gps=True)

    with pytest.raises(CaptureImportError) as error:
        CaptureImporter(input_root, tmp_path / "invalid-gps-work", _capture_plan()).inspect()

    assert error.value.code == "gpx_point_invalid"
    assert not (tmp_path / "invalid-gps-work").exists()


def test_capture_privacy_states_redaction_and_manifest_gate(tmp_path: Path) -> None:
    input_root = tmp_path / "privacy-input"
    work_root = tmp_path / "privacy-work"
    _write_capture_inputs(input_root)
    imported = CaptureImporter(input_root, work_root, _capture_plan()).import_capture()
    assets = imported.inventory.assets
    assert len(assets) == 3
    assert len(list((work_root / "quarantine").glob("*.png"))) == 3
    assert not (work_root / "assets").exists()

    needs_redaction = update_capture_privacy(
        work_root,
        PrivacyDecision(
            asset_id=assets[1].asset_id,
            state="needs_redaction",
            reviewed_by="phase3b2-human-reviewer",
            reviewed_at=datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
            decision_receipt_sha256="a" * 64,
        ),
    )
    rejected = update_capture_privacy(
        work_root,
        PrivacyDecision(
            asset_id=assets[2].asset_id,
            state="rejected",
            reviewed_by="phase3b2-human-reviewer",
            reviewed_at=datetime(2026, 7, 17, 12, 1, tzinfo=UTC),
            decision_receipt_sha256="b" * 64,
        ),
    )
    assert needs_redaction.assets[1].file_locator.startswith("quarantine/")
    assert rejected.assets[2].file_locator.startswith("quarantine/")
    excluded = build_capture_manifest(work_root)
    assert excluded.status == "empty"
    assert excluded.excluded_by_reason == {
        "privacy_pending": 1,
        "privacy_rejected": 1,
        "privacy_needs_redaction": 1,
        "revoked": 0,
    }

    approved_redaction = PrivacyDecision(
        asset_id=assets[1].asset_id,
        state="approved",
        reviewed_by="phase3b2-human-reviewer",
        reviewed_at=datetime(2026, 7, 17, 12, 2, tzinfo=UTC),
        decision_receipt_sha256="c" * 64,
        redaction_applied=True,
        redaction_receipt_sha256="d" * 64,
    )
    with pytest.raises(CaptureImportError) as missing_copy:
        update_capture_privacy(work_root, approved_redaction)
    assert missing_copy.value.code == "redacted_copy_required"

    redacted_root = tmp_path / "manual-redaction"
    redacted_path = _write_synthetic_png(redacted_root, "redacted-copy", seed=83)
    with pytest.raises(CaptureImportError) as traversal:
        update_capture_privacy(
            work_root,
            approved_redaction,
            redacted_input_root=redacted_root,
            redacted_copy=Path("../outside-redacted.png"),
        )
    assert traversal.value.code == "redacted_copy_invalid"

    prior = needs_redaction.assets[1]
    redacted_identity = inspect_image(
        redacted_path,
        max_bytes=1024 * 1024,
        max_pixels=10_000,
    )
    reviewed = update_capture_privacy(
        work_root,
        approved_redaction,
        redacted_input_root=redacted_root,
        redacted_copy=Path("redacted-copy.png"),
    )
    published = reviewed.assets[1]
    assert published.file_locator.startswith("assets/")
    assert published.image_sha256 == redacted_identity.sha256
    assert published.perceptual_hash == redacted_identity.perceptual_hash
    assert published.image_sha256 != prior.image_sha256
    assert published.perceptual_hash != prior.perceptual_hash
    assert published.privacy.redaction_applied is True
    assert (work_root / "originals" / f"{published.asset_id}.png").is_file()

    manifest = build_capture_manifest(work_root)
    assert manifest.eligible_count == 1
    assert manifest.excluded_count == 2
    rows = json.loads((work_root / "manifest.json").read_text(encoding="utf-8"))
    assert rows[0]["personal_data_blur_state"] == "BLURRED_BY_ATLASLENS"


def test_capture_redacted_copy_symlink_is_rejected(tmp_path: Path) -> None:
    input_root = tmp_path / "symlink-input"
    work_root = tmp_path / "symlink-work"
    _write_capture_inputs(input_root)
    imported = CaptureImporter(input_root, work_root, _capture_plan()).import_capture()
    asset = imported.inventory.assets[0]
    redacted_root = tmp_path / "symlink-redaction"
    outside = _write_synthetic_png(tmp_path / "outside-redaction", "outside", seed=97)
    redacted_root.mkdir()
    link = redacted_root / "linked.png"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation is unavailable on this Windows host")

    with pytest.raises(CaptureImportError) as error:
        update_capture_privacy(
            work_root,
            PrivacyDecision(
                asset_id=asset.asset_id,
                state="approved",
                reviewed_by="phase3b2-human-reviewer",
                reviewed_at=datetime(2026, 7, 17, 12, 3, tzinfo=UTC),
                decision_receipt_sha256="e" * 64,
                redaction_applied=True,
                redaction_receipt_sha256="f" * 64,
            ),
            redacted_input_root=redacted_root,
            redacted_copy=Path("linked.png"),
        )
    assert error.value.code == "redacted_copy_invalid"


def test_capture_cli_dry_run_does_not_create_work_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = tmp_path / "dry-run-input"
    work_root = tmp_path / "dry-run-work"
    _write_capture_inputs(input_root)
    (input_root / "capture-plan.json").write_text(
        json.dumps(_capture_plan().model_dump(mode="json")),
        encoding="utf-8",
    )

    exit_code = corpus_cli_main(
        [
            "import-capture",
            "--input-root",
            str(input_root),
            "--work-root",
            str(work_root),
            "--plan",
            "capture-plan.json",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    result = json.loads(captured.out)
    assert result["status"] == "dry_run"
    assert result["input_count"] == 4
    assert result["sampled_count"] == 3
    assert result["would_import_count"] == 3
    assert not work_root.exists()


def test_capture_import_to_megaloc_faiss_locked_query_and_api_states(
    tmp_path: Path,
    client_factory: Any,
) -> None:
    input_root = tmp_path / "input"
    capture_root = tmp_path / "capture-work"
    _write_capture_inputs(input_root)
    importer = CaptureImporter(input_root, capture_root, _capture_plan())

    inspection = importer.inspect()
    assert inspection.media_count == 4
    assert inspection.track_point_count == 4
    assert inspection.synchronized_count == 4
    synchronized = importer.synchronize()
    assert synchronized.track_point_count == 4
    assert len(synchronized.frames) == 4
    sampled = importer.sample()
    assert sampled.input_count == 4
    assert sampled.sampled_count == 3
    assert sampled.stationary_deduplicated_count == 1
    assert sampled.distance_excluded_count == 0

    imported = importer.import_capture()
    assert imported.status == "completed"
    assert imported.input_count == 4
    assert imported.sampled_count == 3
    assert imported.imported_count == 3
    assert imported.pending_privacy_count == 3
    quarantine_files = sorted((capture_root / "quarantine").glob("*.png"))
    assert len(quarantine_files) == 3
    assert not (capture_root / "assets").exists()
    pending_manifest = build_capture_manifest(capture_root)
    assert pending_manifest.status == "empty"
    assert pending_manifest.excluded_count == 3
    assert pending_manifest.excluded_by_reason["privacy_pending"] == 3

    asset_ids = [asset.asset_id for asset in imported.inventory.assets]
    assert len(asset_ids) == 3
    for index in (0, 2):
        reviewed = update_capture_privacy(
            capture_root,
            PrivacyDecision(
                asset_id=asset_ids[index],
                state="approved",
                reviewed_by="phase3b2-synthetic-reviewer",
                reviewed_at=datetime(2026, 7, 17, 10, index, tzinfo=UTC),
                decision_receipt_sha256=f"{index + 4:064x}",
            ),
        )
        assert reviewed.assets[index].file_locator.startswith("assets/")
    revoked = revoke_capture_asset(
        capture_root,
        RevocationUpdate(
            asset_id=asset_ids[2],
            request_id="phase3b2-synthetic-revocation",
            receipt_sha256="6" * 64,
            revoked_at=datetime(2026, 7, 17, 11, 0, tzinfo=UTC),
        ),
    )
    assert revoked.assets[0].privacy.state == "approved"
    assert revoked.assets[0].revocation_status == "ACTIVE"
    assert revoked.assets[1].privacy.state == "pending"
    assert revoked.assets[2].revocation_status == "REVOKED"
    assert len(list((capture_root / "assets").glob("*.png"))) == 1
    assert len(list((capture_root / "quarantine").glob("*.png"))) == 2
    manifest_result = build_capture_manifest(capture_root)
    assert manifest_result.status == "ready"
    assert manifest_result.eligible_count == 1
    assert manifest_result.excluded_count == 2
    assert manifest_result.excluded_by_reason["privacy_pending"] == 1
    assert manifest_result.excluded_by_reason["revoked"] == 1

    megaloc_config = _approved_megaloc_config(tmp_path / "megaloc")
    megaloc, backends = _megaloc_provider(megaloc_config)
    descriptor_version = megaloc.spec.version
    artifact_sha256 = megaloc_config.artifact_sha256

    corpus_work = tmp_path / "corpus-work"
    ingestion = CorpusIngestor(
        corpus_root=capture_root,
        work_root=corpus_work,
        checkpoint_root=tmp_path / "corpus-checkpoint",
        manifest_schema_path=MANIFEST_SCHEMA,
        source_policy_path=SOURCE_POLICY,
        config=IngestionConfig(
            strict_rejections=True,
            near_duplicate_hamming_threshold=0,
            spatial_leakage_radius_m=50.0,
            descriptor_version=descriptor_version,
            index_version="phase3b2-e2e-index-v1",
        ),
    ).ingest(Path("manifest.json"), checkpoint_path=Path("pipeline-state.json"))
    assert len(ingestion.corpus.assets) == 1
    admitted = ingestion.corpus.assets[0]
    descriptor_asset = DescriptorAsset(
        locator=capture_root / admitted.file_locator,
        provenance=AssetProvenance(
            asset_id=admitted.asset_id,
            source_id=admitted.source_id,
            rights_decision=admitted.rights_decision,
            license_record_id=admitted.provenance_receipt_id,
            provenance_summary="generated first-party Phase 3B2 synthetic fixture",
            content_sha256=admitted.image_sha256,
            split=admitted.role,
            rights_validated=True,
            province=admitted.province_code,
            latitude=admitted.latitude,
            longitude=admitted.longitude,
            contributor_id=admitted.contributor_or_owner,
            capture_run_id=admitted.capture_run_id,
            sequence_id=admitted.sequence_id,
            sampling_cell=admitted.spatial_split,
        ),
    )
    smoke = megaloc.run_smoke(descriptor_asset.locator)
    assert smoke.descriptor_dimension == 8448
    assert smoke.artifact_sha256 == artifact_sha256
    assert megaloc.runtime_kind == "test_only"
    descriptors = build_descriptors(
        (descriptor_asset,),
        megaloc,
        work_dir=tmp_path / "descriptor-work",
        output_dir=tmp_path / "descriptors",
        manifest_hash=ingestion.corpus.manifest_sha256,
        source_policy_hash=ingestion.corpus.source_policy_sha256,
        batch_size=1,
    )
    query = BenchmarkQuery(
        query_id="phase3b2-locked-synthetic-query",
        descriptor=np.concatenate(
            (np.asarray([1.0], dtype=np.float32), np.zeros(8447, dtype=np.float32))
        ),
        relevant_asset_ids=(admitted.asset_id,),
        content_sha256="7" * 64,
        holdout_asset_id="phase3b2-synthetic-holdout",
        source_id="phase3b2-independent-holdout-source",
        contributor_id="phase3b2-independent-holdout-owner",
        capture_run_id="phase3b2-independent-holdout-run",
        sequence_id="phase3b2-independent-holdout-sequence",
        sampling_cell="phase3b2-independent-holdout-cell",
        province="Synthetic",
        latitude=41.0,
        longitude=40.0,
    )
    holdout_hash = compute_locked_holdout_hash((query,))
    split_lock_hash = "8" * 64
    index_path = tmp_path / "faiss-index"
    index = build_index(
        descriptors.dataset,
        output_dir=index_path,
        backend="faiss",
        index_version="phase3b2-e2e-index-v1",
        source_policy_hash=ingestion.corpus.source_policy_sha256,
        locked_holdout_hash=holdout_hash,
        split_lock_hash=split_lock_hash,
    )
    result = evaluate_locked_holdout(
        index,
        (query,),
        expected_holdout_hash=holdout_hash,
        expected_split_lock_hash=split_lock_hash,
    )
    assert index.backend == "faiss-flat-ip-v1"
    assert backends[0].calls == [1, 1]
    assert index.spec.artifact_sha256 == artifact_sha256
    assert index.spec.preprocessing_version == PREPROCESSING_VERSION
    metadata = json.loads((index_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["descriptor_spec"]["artifact_sha256"] == artifact_sha256
    assert metadata["descriptor_spec"]["preprocessing_version"] == PREPROCESSING_VERSION
    assert result.recall_at_1 == 1.0
    assert result.to_json()["accuracy_semantics"].endswith("not confidence")

    disabled = TurkiyeReferenceIndexProvider(
        enabled=False,
        index_path=index_path,
        source_policy_path=SOURCE_POLICY,
        descriptor_provider="megaloc",
        descriptor_version=descriptor_version,
        descriptor_dimension=8448,
    )
    assert disabled.status().state == "disabled"
    blank = TurkiyeReferenceIndexProvider(
        enabled=True,
        index_path=index_path,
        source_policy_path=SOURCE_POLICY,
        descriptor_provider="megaloc",
        descriptor_version=descriptor_version,
        descriptor_dimension=8448,
    )
    assert blank.status().reason_code == "descriptor_artifact_not_approved"
    ready = TurkiyeReferenceIndexProvider(
        enabled=True,
        index_path=index_path,
        source_policy_path=SOURCE_POLICY,
        descriptor_provider="megaloc",
        descriptor_version=descriptor_version,
        descriptor_dimension=8448,
        descriptor_artifact_sha256=artifact_sha256,
        descriptor_preprocessing_version=PREPROCESSING_VERSION,
        descriptor_artifact_approved=True,
    )
    assert ready.status().state == "ready"
    api = client_factory(turkiye_reference_provider=ready)
    capability = api.get("/api/v1/capabilities").json()["providers"]["turkiye_reference_index"]
    assert capability["operational_status"] == "ready"

    mismatch = TurkiyeReferenceIndexProvider(
        enabled=True,
        index_path=index_path,
        source_policy_path=SOURCE_POLICY,
        descriptor_provider="megaloc",
        descriptor_version=descriptor_version,
        descriptor_dimension=8448,
        descriptor_artifact_sha256="9" * 64,
        descriptor_preprocessing_version="wrong-preprocessing",
        descriptor_artifact_approved=True,
    )
    assert mismatch.status().state == "not_ready"
    forbidden = TurkiyeReferenceIndexProvider(
        enabled=True,
        index_path=index_path,
        source_policy_path=SOURCE_POLICY,
        descriptor_provider="atlaslens-test-only-deterministic",
        descriptor_version=descriptor_version,
        descriptor_dimension=8448,
        descriptor_artifact_sha256=artifact_sha256,
        descriptor_preprocessing_version=PREPROCESSING_VERSION,
        descriptor_artifact_approved=True,
    )
    assert forbidden.status().reason_code == "test_descriptor_provider_forbidden"
