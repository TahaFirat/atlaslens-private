from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from atlaslens_api.corpus_cli import COMMANDS, build_parser
from atlaslens_api.corpus_cli import main as corpus_cli_main
from atlaslens_api.corpus_index.benchmark import (
    BenchmarkQuery,
    compute_locked_holdout_hash,
)
from atlaslens_api.corpus_index.errors import HoldoutIntegrityError
from atlaslens_api.corpus_index.index import PublishedCorpusIndex
from atlaslens_api.corpus_index.models import DescriptorSpec, FloatMatrix
from atlaslens_api.corpus_index.providers import TestOnlyDeterministicDescriptorProvider
from atlaslens_api.corpus_pipeline import (
    CheckpointStore,
    IngestionConfig,
    PipelineState,
    SourcePolicy,
    inspect_image,
)
from atlaslens_api.corpus_workflow import CorpusWorkflow, CorpusWorkflowPaths

REPOSITORY_ROOT = Path(__file__).parents[3]
MANIFEST_SCHEMA = REPOSITORY_ROOT / "config" / "corpus" / "manifest-schema-v1.json"
SOURCE_POLICY = REPOSITORY_ROOT / "config" / "corpus" / "source-policy-v1.json"
FIRST_PARTY_SOURCE = "First-party AtlasLens-captured imagery"


def _pixels(seed: int) -> list[tuple[int, int, int]]:
    return [
        (
            (x * (17 + seed * 3) + y * (31 + seed * 5) + (x * y % 19) * 7) % 256,
            (x * (11 + seed) + y * (23 + seed * 2) + seed * 29) % 256,
            (255 - x * 5 - y * 3 + seed * 7) % 256,
        )
        for y in range(24)
        for x in range(32)
    ]


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


def _row(
    path: Path,
    asset_id: str,
    *,
    role: str = "train",
    latitude: float = 38.0,
    longitude: float = 34.0,
    contributor: str | None = None,
    sequence_id: str | None = None,
    capture_run_id: str | None = None,
    source_name: str = FIRST_PARTY_SOURCE,
    decision: str = "FIRST_PARTY_ONLY",
    acquisition_ready: bool = True,
    revocation_status: str = "ACTIVE",
) -> dict[str, object]:
    identity = inspect_image(path, max_bytes=1024 * 1024, max_pixels=10_000)
    return {
        "corpus_version": "atlaslens-turkiye-corpus-synthetic-e2e-v1",
        "asset_id": asset_id,
        "source_asset_id": f"source-{asset_id}",
        "source_name": source_name,
        "source_url": f"https://fixture.invalid/assets/{asset_id}",
        "contributor_or_owner": contributor or f"synthetic-owner-{asset_id}",
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
        "license_identifier": "AtlasLens-first-party-synthetic-fixture",
        "license_url": "https://fixture.invalid/license",
        "attribution_text": "Synthetic fixture; never production imagery",
        "source_policy_decision": decision,
        "commercial_use_decision": decision,
        "derivative_index_decision": decision,
        "personal_data_blur_state": "BLURRED_BY_ATLASLENS",
        "deletion_revocation_state": {
            "status": revocation_status,
            "last_checked_at": "2026-07-16T09:00:00Z",
            "request_id": "synthetic-revocation" if revocation_status != "ACTIVE" else None,
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
        "tier": ("tier_3_locked_holdout" if role == "holdout" else "tier_1_national_recall"),
        "acquisition_ready": acquisition_ready,
    }


class _InterruptAfterOneBatch:
    runtime_kind = "test_only"

    def __init__(self, delegate: TestOnlyDeterministicDescriptorProvider) -> None:
        self._delegate = delegate
        self.calls = 0

    @property
    def spec(self) -> DescriptorSpec:
        return self._delegate.spec

    def describe_batch(self, locators: Sequence[Path]) -> FloatMatrix:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("synthetic interruption")
        return self._delegate.describe_batch(locators)


def test_synthetic_corpus_to_locked_benchmark_with_resume(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoints"

    holdout = _write_png(corpus_root, "a-holdout", seed=41)
    reference_1 = _write_png(corpus_root, "b-reference-1", seed=11, compress_level=0)
    reference_2 = _write_png(corpus_root, "c-reference-2", seed=17, compress_level=0)
    reference_3 = _write_png(corpus_root, "d-reference-3", seed=23)
    exact = corpus_root / "assets" / "z-exact-duplicate.png"
    shutil.copyfile(reference_1, exact)
    near = _write_png(corpus_root, "z-near-duplicate", seed=17, compress_level=9)
    sequence_leak = _write_png(corpus_root, "z-sequence-leak", seed=31)
    spatial_leak = _write_png(corpus_root, "z-spatial-leak", seed=37)
    blocked = _write_png(corpus_root, "z-blocked", seed=43)
    revoked = _write_png(corpus_root, "z-revoked", seed=47)
    invalid = _write_png(corpus_root, "z-invalid", seed=53)

    policy = SourcePolicy.load(SOURCE_POLICY)
    holdout_source = next(
        entry
        for entry in policy.document.sources
        if entry.source_id == "wikimedia_commons"
    )
    blocked_source = next(
        entry.source_name
        for entry in policy.document.sources
        if entry.current_decision == "BLOCKED"
    )
    rows = [
        _row(
            holdout,
            "a-holdout",
            role="holdout",
            latitude=39.0,
            longitude=35.0,
            contributor="synthetic-holdout-owner",
            sequence_id="holdout-sequence",
            capture_run_id="holdout-run",
            source_name=holdout_source.source_name,
            decision=holdout_source.current_decision,
        ),
        _row(reference_1, "b-reference-1", latitude=38.0, longitude=34.0),
        _row(reference_2, "c-reference-2", latitude=37.0, longitude=33.0),
        _row(reference_3, "d-reference-3", latitude=36.0, longitude=32.0),
        _row(exact, "z-exact-duplicate", latitude=35.0, longitude=31.0),
        _row(near, "z-near-duplicate", latitude=34.0, longitude=30.0),
        _row(
            sequence_leak,
            "z-sequence-leak",
            latitude=33.0,
            longitude=29.0,
            contributor="synthetic-holdout-owner",
            sequence_id="holdout-sequence",
            source_name=holdout_source.source_name,
            decision=holdout_source.current_decision,
        ),
        _row(spatial_leak, "z-spatial-leak", latitude=39.0001, longitude=35.0),
        _row(
            blocked,
            "z-blocked",
            source_name=blocked_source,
            decision="BLOCKED",
            acquisition_ready=False,
        ),
        _row(
            revoked,
            "z-revoked",
            acquisition_ready=False,
            revocation_status="REVOKED",
        ),
        _row(invalid, "z-invalid"),
    ]
    rows[-1]["latitude"] = 91.0
    manifest = corpus_root / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")

    config = IngestionConfig(
        strict_rejections=False,
        near_duplicate_hamming_threshold=0,
        spatial_leakage_radius_m=50,
        descriptor_version="synthetic-descriptor-v1",
        index_version="synthetic-index-v1",
    )
    paths = CorpusWorkflowPaths(
        corpus_root=corpus_root,
        work_dir=work_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
    )
    workflow = CorpusWorkflow(
        manifest_path=manifest,
        manifest_schema_path=MANIFEST_SCHEMA,
        source_policy_path=SOURCE_POLICY,
        paths=paths,
        ingestion_config=config,
    )

    ingestion = workflow.ingest()
    assert {asset.asset_id for asset in ingestion.corpus.assets} == {
        "a-holdout",
        "b-reference-1",
        "c-reference-2",
        "d-reference-3",
    }
    reasons = {
        rejection.asset_id: rejection.reason_code for rejection in ingestion.corpus.rejections
    }
    assert reasons == {
        "z-blocked": "source_policy_not_acquisition_ready",
        "z-exact-duplicate": "exact_duplicate",
        "z-invalid": "manifest_row_invalid",
        "z-near-duplicate": "near_duplicate",
        "z-revoked": "asset_revoked_or_unavailable",
        "z-sequence-leak": "sequence_cross_spatial_split",
        "z-spatial-leak": "spatial_cross_split",
    }
    assert ingestion.checkpoint.state == PipelineState.SPLIT_LOCKED

    base_provider = TestOnlyDeterministicDescriptorProvider(dimension=8)
    interrupted_provider = _InterruptAfterOneBatch(base_provider)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        workflow.build_descriptors(interrupted_provider, batch_size=1)
    interrupted = CheckpointStore(checkpoint_dir, "pipeline-state.json").load()
    assert interrupted.state == PipelineState.FAILED
    assert interrupted.last_successful_state == PipelineState.SPLIT_LOCKED

    descriptors = workflow.build_descriptors(base_provider, batch_size=1, resume=True)
    assert descriptors.resumed_assets == 1
    assert descriptors.generated_assets == 2
    assert descriptors.dataset.count == 3
    assert {asset.asset_id for asset in descriptors.dataset.assets} == {
        "b-reference-1",
        "c-reference-2",
        "d-reference-3",
    }

    query_vector = np.array(
        base_provider.describe_batch([reference_1])[0],
        dtype=np.float32,
        copy=True,
    )
    query_vector[0] += 0.01
    query_vector /= np.linalg.norm(query_vector)
    holdout_identity = inspect_image(
        holdout,
        max_bytes=1024 * 1024,
        max_pixels=10_000,
    )
    query = BenchmarkQuery(
        query_id="synthetic-query-1",
        descriptor=query_vector,
        relevant_asset_ids=("b-reference-1",),
        content_sha256=holdout_identity.sha256,
        holdout_asset_id="a-holdout",
        province="TR-38",
        latitude=39.0,
        longitude=35.0,
        source_id=holdout_source.source_id,
        contributor_id="synthetic-holdout-owner",
        capture_run_id="holdout-run",
        sequence_id="holdout-sequence",
        sampling_cell="cell-a-holdout",
    )
    holdout_hash = compute_locked_holdout_hash((query,))
    forged_query = replace(query, contributor_id="forged-contributor")
    with pytest.raises(HoldoutIntegrityError, match="identity"):
        workflow.build_index(
            (forged_query,),
            backend="exact",
            index_version="synthetic-index-v1",
            shard_size=2,
            locked_holdout_hash=compute_locked_holdout_hash((forged_query,)),
            resume=True,
        )
    index = workflow.build_index(
        (query,),
        backend="exact",
        index_version="synthetic-index-v1",
        shard_size=2,
        locked_holdout_hash=holdout_hash,
        resume=True,
    )
    assert index.size == 3
    assert index.split_lock_hash == ingestion.split_lock.lock_sha256
    benchmark = workflow.evaluate(
        (query,),
        expected_holdout_hash=holdout_hash,
        resume=True,
    )
    assert benchmark.query_count == 1
    assert benchmark.recall_at_1 == 1.0
    assert benchmark.recall_at_5 == 1.0
    assert benchmark.recall_at_10 == 1.0
    assert benchmark.abstention_coverage == 1.0

    completed = workflow.finalize(resume=True)
    assert completed.state == PipelineState.COMPLETED
    published = PublishedCorpusIndex.open(output_dir / "index")
    impact = published.revocation_impact("b-reference-1")
    rebuild = published.rebuild_plan(["b-reference-1"])
    assert impact.found
    assert impact.shard_ids
    assert rebuild.removed_asset_count == 1
    assert rebuild.retained_asset_count == 2
    inventory = json.loads((output_dir / "artifact-inventory.json").read_text("utf-8"))
    assert inventory["schema"] == "atlaslens-phase3b-artifact-inventory-v1"
    assert len(inventory["artifacts"]) == len(published.artifact_inventory())

    cli_config = tmp_path / "cli-config.json"
    cli_config.write_text(
        json.dumps({"ingestion": config.model_dump(mode="json"), "pipeline": {}}),
        encoding="utf-8",
    )
    common = [
        "--manifest",
        str(manifest),
        "--manifest-schema",
        str(MANIFEST_SCHEMA),
        "--source-policy",
        str(SOURCE_POLICY),
        "--corpus-root",
        str(corpus_root),
        "--work-dir",
        str(tmp_path / "cli-work"),
        "--output-dir",
        str(tmp_path / "cli-output"),
        "--checkpoint",
        str(tmp_path / "cli-checkpoint"),
        "--config",
        str(cli_config),
    ]
    assert corpus_cli_main(["run-pipeline", *common, "--dry-run"]) == 0
    assert corpus_cli_main(["run-pipeline", *common]) == 3
    assert not (tmp_path / "cli-work").exists()
    assert corpus_cli_main(["inspect-checkpoint", "--checkpoint", str(checkpoint_dir)]) == 0


@pytest.mark.parametrize("command", COMMANDS)
def test_each_corpus_command_exposes_help(command: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([command, "--help"])
    assert exit_info.value.code == 0
