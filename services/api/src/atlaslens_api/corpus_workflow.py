"""Integrated, resumable Phase 3B corpus-to-index workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from atlaslens_api.corpus_index.artifacts import atomic_write_json, sha256_path
from atlaslens_api.corpus_index.benchmark import (
    BenchmarkQuery,
    BenchmarkResult,
    evaluate_locked_holdout,
    verify_locked_holdout_hash,
)
from atlaslens_api.corpus_index.descriptors import (
    DescriptorBuildResult,
    DescriptorDataset,
    build_descriptors,
)
from atlaslens_api.corpus_index.errors import HoldoutIntegrityError
from atlaslens_api.corpus_index.index import PublishedCorpusIndex, build_index
from atlaslens_api.corpus_index.models import AssetProvenance, DescriptorAsset
from atlaslens_api.corpus_index.providers import DescriptorProvider
from atlaslens_api.corpus_pipeline import (
    CheckpointError,
    CheckpointStore,
    CorpusIngestor,
    CorpusPreparation,
    IngestedCorpus,
    IngestionConfig,
    IngestionResult,
    PipelineCheckpoint,
    PipelineState,
    SplitLock,
    load_ingested_corpus,
    load_split_lock,
    state_at_or_after,
)
from atlaslens_api.corpus_pipeline.safety import (
    ensure_safe_output_root,
    resolve_contained_file,
)


@dataclass(frozen=True, slots=True)
class CorpusWorkflowPaths:
    corpus_root: Path
    work_dir: Path
    output_dir: Path
    checkpoint_dir: Path

    @property
    def descriptor_work_dir(self) -> Path:
        return self.work_dir / "descriptor-work"

    @property
    def descriptor_output_dir(self) -> Path:
        return self.output_dir / "descriptors"

    @property
    def index_output_dir(self) -> Path:
        return self.output_dir / "index"

    @property
    def benchmark_output_path(self) -> Path:
        return self.output_dir / "benchmark.json"

    @property
    def inventory_output_path(self) -> Path:
        return self.output_dir / "artifact-inventory.json"


@dataclass(frozen=True, slots=True)
class CorpusWorkflowResult:
    accepted_assets: int
    rejected_assets: int
    resumed_descriptors: int
    generated_descriptors: int
    index_assets: int
    benchmark: BenchmarkResult
    checkpoint: PipelineCheckpoint


class CorpusWorkflow:
    """Bind ingestion and retrieval components without selecting a model implicitly."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_schema_path: Path,
        source_policy_path: Path,
        paths: CorpusWorkflowPaths,
        ingestion_config: IngestionConfig | None = None,
    ) -> None:
        if paths.work_dir.resolve() == paths.output_dir.resolve():
            raise ValueError("work and output directories must differ")
        self.manifest_path = manifest_path
        self.source_policy_path = source_policy_path
        self.paths = paths
        self.ingestion_config = ingestion_config or IngestionConfig()
        self.ingestor = CorpusIngestor(
            corpus_root=paths.corpus_root,
            work_root=paths.work_dir,
            checkpoint_root=paths.checkpoint_dir,
            manifest_schema_path=manifest_schema_path,
            source_policy_path=source_policy_path,
            config=self.ingestion_config,
        )

    def prepare(self) -> CorpusPreparation:
        """Perform the no-write manifest, rights, identity, and leakage preview."""
        return self.ingestor.prepare(self.manifest_path)

    def ingest(
        self,
        *,
        resume: bool = False,
        interrupt_before_complete: PipelineState | None = None,
    ) -> IngestionResult:
        ensure_safe_output_root(self.paths.work_dir)
        ensure_safe_output_root(self.paths.output_dir)
        ensure_safe_output_root(self.paths.checkpoint_dir)
        return self.ingestor.ingest(
            self.manifest_path,
            checkpoint_path=Path("pipeline-state.json"),
            resume=resume,
            interrupt_before_complete=interrupt_before_complete,
        )

    def build_descriptors(
        self,
        provider: DescriptorProvider,
        *,
        batch_size: int,
        resume: bool = False,
    ) -> DescriptorBuildResult:
        corpus = load_ingested_corpus(self.paths.work_dir)
        store, checkpoint = self._resumable_store(resume=resume)
        if not state_at_or_after(checkpoint.last_successful_state, PipelineState.SPLIT_LOCKED):
            raise CheckpointError("checkpoint_split_not_locked")
        transition_required = not state_at_or_after(
            checkpoint.last_successful_state,
            PipelineState.DESCRIPTORS_BUILT,
        )
        if transition_required:
            store.begin(PipelineState.DESCRIPTORS_BUILT)
        try:
            result = build_descriptors(
                _descriptor_assets(corpus, self.paths.corpus_root),
                provider,
                work_dir=self.paths.descriptor_work_dir,
                output_dir=self.paths.descriptor_output_dir,
                manifest_hash=corpus.manifest_sha256,
                source_policy_hash=corpus.source_policy_sha256,
                batch_size=batch_size,
            )
            if transition_required:
                marker = self.paths.descriptor_output_dir / "PUBLISHED.json"
                store.complete(
                    PipelineState.DESCRIPTORS_BUILT,
                    artifact_sha256={"descriptors/PUBLISHED.json": sha256_path(marker)},
                )
            return result
        except Exception:
            if transition_required:
                store.fail("descriptor_build_failed")
            raise

    def build_index(
        self,
        queries: Sequence[BenchmarkQuery],
        *,
        backend: str,
        index_version: str,
        shard_size: int,
        locked_holdout_hash: str,
        resume: bool = False,
    ) -> PublishedCorpusIndex:
        if backend not in {"exact", "faiss"}:
            raise ValueError("index backend must be exact or faiss")
        split_lock_hash = self.validate_holdout(
            queries,
            expected_holdout_hash=locked_holdout_hash,
        )
        selected_backend = cast(Literal["exact", "faiss"], backend)
        store, checkpoint = self._resumable_store(resume=resume)
        if not state_at_or_after(
            checkpoint.last_successful_state,
            PipelineState.DESCRIPTORS_BUILT,
        ):
            raise CheckpointError("checkpoint_descriptors_not_built")
        transition_required = not state_at_or_after(
            checkpoint.last_successful_state,
            PipelineState.INDEX_BUILT,
        )
        if transition_required:
            store.begin(PipelineState.INDEX_BUILT)
        try:
            dataset = DescriptorDataset.open(self.paths.descriptor_output_dir)
            index = build_index(
                dataset,
                output_dir=self.paths.index_output_dir,
                backend=selected_backend,
                index_version=index_version,
                source_policy_hash=dataset.source_policy_hash,
                shard_size=shard_size,
                locked_holdout_hash=locked_holdout_hash,
                split_lock_hash=split_lock_hash,
            )
            if transition_required:
                marker = self.paths.index_output_dir / "PUBLISHED.json"
                store.complete(
                    PipelineState.INDEX_BUILT,
                    artifact_sha256={"index/PUBLISHED.json": sha256_path(marker)},
                )
            return index
        except Exception:
            if transition_required:
                store.fail("index_build_failed")
            raise

    def evaluate(
        self,
        queries: Sequence[BenchmarkQuery],
        *,
        expected_holdout_hash: str,
        abstain_if_distance_gt: float | None = None,
        resume: bool = False,
    ) -> BenchmarkResult:
        split_lock_hash = self.validate_holdout(
            queries,
            expected_holdout_hash=expected_holdout_hash,
        )
        store, checkpoint = self._resumable_store(resume=resume)
        if not state_at_or_after(checkpoint.last_successful_state, PipelineState.INDEX_BUILT):
            raise CheckpointError("checkpoint_index_not_built")
        transition_required = not state_at_or_after(
            checkpoint.last_successful_state,
            PipelineState.BENCHMARKED,
        )
        if transition_required:
            store.begin(PipelineState.BENCHMARKED)
        try:
            index = PublishedCorpusIndex.open(
                self.paths.index_output_dir,
                expected_locked_holdout_hash=expected_holdout_hash,
                expected_split_lock_hash=split_lock_hash,
            )
            result = evaluate_locked_holdout(
                index,
                queries,
                expected_holdout_hash=expected_holdout_hash,
                expected_split_lock_hash=split_lock_hash,
                abstain_if_distance_gt=abstain_if_distance_gt,
            )
            atomic_write_json(self.paths.benchmark_output_path, result.to_json())
            if transition_required:
                store.complete(
                    PipelineState.BENCHMARKED,
                    artifact_sha256={
                        "benchmark.json": sha256_path(self.paths.benchmark_output_path)
                    },
                )
            return result
        except Exception:
            if transition_required:
                store.fail("benchmark_failed")
            raise

    def finalize(self, *, resume: bool = False) -> PipelineCheckpoint:
        store, checkpoint = self._resumable_store(resume=resume)
        if checkpoint.last_successful_state == PipelineState.COMPLETED:
            return checkpoint
        if checkpoint.last_successful_state != PipelineState.BENCHMARKED:
            raise CheckpointError("checkpoint_benchmark_not_completed")
        store.begin(PipelineState.COMPLETED)
        try:
            index = PublishedCorpusIndex.open(self.paths.index_output_dir)
            inventory = {
                "schema": "atlaslens-phase3b-artifact-inventory-v1",
                "artifacts": [item.to_json() for item in index.artifact_inventory()],
            }
            atomic_write_json(self.paths.inventory_output_path, inventory)
            return store.complete(
                PipelineState.COMPLETED,
                artifact_sha256={
                    "artifact-inventory.json": sha256_path(self.paths.inventory_output_path)
                },
            )
        except Exception:
            store.fail("pipeline_finalize_failed")
            raise

    def run(
        self,
        provider: DescriptorProvider,
        queries: Sequence[BenchmarkQuery],
        *,
        descriptor_batch_size: int,
        index_backend: str,
        index_version: str,
        index_shard_size: int,
        expected_holdout_hash: str,
        abstain_if_distance_gt: float | None = None,
        resume: bool = False,
    ) -> CorpusWorkflowResult:
        ingestion = self.ingest(resume=resume)
        descriptors = self.build_descriptors(
            provider,
            batch_size=descriptor_batch_size,
            resume=resume,
        )
        index = self.build_index(
            queries,
            backend=index_backend,
            index_version=index_version,
            shard_size=index_shard_size,
            locked_holdout_hash=expected_holdout_hash,
            resume=resume,
        )
        benchmark = self.evaluate(
            queries,
            expected_holdout_hash=expected_holdout_hash,
            abstain_if_distance_gt=abstain_if_distance_gt,
            resume=resume,
        )
        checkpoint = self.finalize(resume=resume)
        return CorpusWorkflowResult(
            accepted_assets=len(ingestion.corpus.assets),
            rejected_assets=len(ingestion.corpus.rejections),
            resumed_descriptors=descriptors.resumed_assets,
            generated_descriptors=descriptors.generated_assets,
            index_assets=index.size,
            benchmark=benchmark,
            checkpoint=checkpoint,
        )

    def validate_holdout(
        self,
        queries: Sequence[BenchmarkQuery],
        *,
        expected_holdout_hash: str,
    ) -> str:
        """Bind every query to the admitted corpus and tamper-evident split lock."""

        verify_locked_holdout_hash(queries, expected_holdout_hash)
        corpus = load_ingested_corpus(self.paths.work_dir)
        split_lock = load_split_lock(self.paths.work_dir)
        _verify_holdout_binding(corpus, split_lock, queries)
        return split_lock.lock_sha256

    def _resumable_store(
        self,
        *,
        resume: bool,
    ) -> tuple[CheckpointStore, PipelineCheckpoint]:
        store = CheckpointStore(self.paths.checkpoint_dir, Path("pipeline-state.json"))
        checkpoint = store.load()
        needs_resume = (
            checkpoint.state in {PipelineState.FAILED, PipelineState.CANCELLED}
            or checkpoint.active_state is not None
        )
        if needs_resume:
            if not resume:
                raise CheckpointError("checkpoint_resume_required")
            checkpoint = store.initialize(
                run_id=checkpoint.run_id,
                manifest_sha256=checkpoint.manifest_sha256,
                manifest_schema_sha256=checkpoint.manifest_schema_sha256,
                source_policy_sha256=checkpoint.source_policy_sha256,
                config_sha256=checkpoint.config_sha256,
                resume=True,
            )
        return store, checkpoint


def _descriptor_assets(corpus: IngestedCorpus, corpus_root: Path) -> tuple[DescriptorAsset, ...]:
    assets: list[DescriptorAsset] = []
    for asset in corpus.assets:
        if asset.role == "holdout":
            continue
        locator = resolve_contained_file(corpus_root, asset.file_locator)
        assets.append(
            DescriptorAsset(
                locator=locator,
                provenance=AssetProvenance(
                    asset_id=asset.asset_id,
                    source_id=asset.source_id,
                    rights_decision=asset.rights_decision,
                    license_record_id=asset.provenance_receipt_id,
                    provenance_summary=(
                        f"{asset.source_id}; license={asset.license_identifier}; "
                        f"receipt={asset.provenance_receipt_id}"
                    ),
                    content_sha256=asset.image_sha256,
                    split=asset.role,
                    rights_validated=asset.permission_state == "validated",
                    revoked=asset.revoked,
                    province=asset.province_code,
                    latitude=asset.latitude,
                    longitude=asset.longitude,
                    contributor_id=asset.contributor_or_owner,
                    capture_run_id=asset.capture_run_id,
                    sequence_id=asset.sequence_id,
                    sampling_cell=asset.sampling_cell,
                ),
            )
        )
    return tuple(assets)


def _verify_holdout_binding(
    corpus: IngestedCorpus,
    split_lock: SplitLock,
    queries: Sequence[BenchmarkQuery],
) -> None:
    if (
        split_lock.manifest_sha256 != corpus.manifest_sha256
        or split_lock.config_sha256 != corpus.config_sha256
    ):
        raise HoldoutIntegrityError("split lock does not match the admitted corpus")

    assignments = {item.asset_id: item for item in split_lock.assignments}
    if set(assignments) != {asset.asset_id for asset in corpus.assets}:
        raise HoldoutIntegrityError("split lock does not cover the admitted corpus")
    for asset in corpus.assets:
        assignment = assignments[asset.asset_id]
        if (
            assignment.image_sha256 != asset.image_sha256
            or assignment.role != asset.role
            or assignment.tier != asset.tier
            or assignment.spatial_split != asset.spatial_split
        ):
            raise HoldoutIntegrityError("split assignment does not match the admitted corpus")

    holdout = {asset.asset_id: asset for asset in corpus.assets if asset.role == "holdout"}
    query_asset_ids = [query.holdout_asset_id for query in queries]
    if (
        not holdout
        or len(query_asset_ids) != len(set(query_asset_ids))
        or set(query_asset_ids) != set(holdout)
    ):
        raise HoldoutIntegrityError("holdout queries do not map one-to-one to the locked split")

    reference_asset_ids = {asset.asset_id for asset in corpus.assets if asset.role != "holdout"}
    for query in queries:
        asset = holdout[query.holdout_asset_id]
        if (
            query.content_sha256 != asset.image_sha256
            or query.source_id != asset.source_id
            or query.contributor_id != asset.contributor_or_owner
            or query.capture_run_id != asset.capture_run_id
            or query.sequence_id != asset.sequence_id
            or query.sampling_cell != asset.sampling_cell
            or query.province != asset.province_code
            or query.latitude != asset.latitude
            or query.longitude != asset.longitude
        ):
            raise HoldoutIntegrityError(
                "holdout query identity does not match the admitted locked asset"
            )
        if not set(query.relevant_asset_ids).issubset(reference_asset_ids):
            raise HoldoutIntegrityError("holdout relevance IDs are outside the reference split")
