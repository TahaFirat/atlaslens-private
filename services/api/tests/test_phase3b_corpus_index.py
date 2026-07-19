from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from atlaslens_api.corpus_index.benchmark import (
    BenchmarkQuery,
    LeakageViolation,
    compute_locked_holdout_hash,
    detect_prebenchmark_leakage,
    evaluate_locked_holdout,
)
from atlaslens_api.corpus_index.descriptors import (
    DescriptorDataset,
    build_descriptors,
    build_production_descriptors,
)
from atlaslens_api.corpus_index.errors import (
    ArtifactIntegrityError,
    BenchmarkLeakageError,
    CheckpointCompatibilityError,
    DescriptorValidationError,
    HoldoutIntegrityError,
    IndexCompatibilityError,
    ProviderNotConfiguredError,
    ProviderRegistrationError,
)
from atlaslens_api.corpus_index.index import (
    PublishedCorpusIndex,
    build_index,
    write_rebuild_plan,
)
from atlaslens_api.corpus_index.models import (
    AssetProvenance,
    DescriptorAsset,
    DescriptorSpec,
    FloatMatrix,
    ProviderApproval,
    RuntimeKind,
    normalize_vector,
)
from atlaslens_api.corpus_index.providers import (
    ProductionDescriptorRegistry,
    TestOnlyDeterministicDescriptorProvider,
)

MANIFEST_HASH = "a" * 64
SOURCE_POLICY_HASH = "b" * 64
SPLIT_LOCK_HASH = "c" * 64


class StaticDescriptorProvider:
    spec = DescriptorSpec(provider_id="synthetic-static", version="1", dimension=3)

    def __init__(
        self,
        vectors: dict[str, Sequence[float]],
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self.vectors = vectors
        self.fail_on_call = fail_on_call
        self.calls = 0

    @property
    def runtime_kind(self) -> RuntimeKind:
        return "test_only"

    def describe_batch(self, locators: Sequence[Path]) -> FloatMatrix:
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("synthetic interruption")
        return np.asarray([self.vectors[path.name] for path in locators], dtype=np.float32)


class ProductionStubProvider(StaticDescriptorProvider):
    @property
    def runtime_kind(self) -> RuntimeKind:
        return "production"


class InvalidDescriptorProvider(StaticDescriptorProvider):
    def __init__(self, value: FloatMatrix) -> None:
        super().__init__({})
        self.value = value

    def describe_batch(self, locators: Sequence[Path]) -> FloatMatrix:
        return self.value


def _make_assets(tmp_path: Path) -> tuple[list[DescriptorAsset], dict[str, list[float]]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = {
        "asset-a": ([1.0, 0.0, 0.0], "Ankara", 39.9334, 32.8597),
        "asset-b": ([0.0, 1.0, 0.0], "Kayseri", 38.7205, 35.4826),
        "asset-c": ([0.0, 0.0, 1.0], "Sivas", 39.7505, 37.0150),
    }
    assets: list[DescriptorAsset] = []
    vectors: dict[str, list[float]] = {}
    for asset_id, (vector, province, latitude, longitude) in rows.items():
        locator = tmp_path / f"{asset_id}.fixture"
        content = f"synthetic-numeric-fixture:{asset_id}".encode()
        locator.write_bytes(content)
        vectors[locator.name] = vector
        assets.append(
            DescriptorAsset(
                locator=locator,
                provenance=AssetProvenance(
                    asset_id=asset_id,
                    source_id="first-party-test-fixture",
                    rights_decision="FIRST_PARTY_ONLY",
                    license_record_id=f"license-{asset_id}",
                    provenance_summary="generated synthetic numeric fixture",
                    content_sha256=hashlib.sha256(content).hexdigest(),
                    split="reference",
                    rights_validated=True,
                    province=province,
                    latitude=latitude,
                    longitude=longitude,
                    contributor_id=f"reference-contributor-{asset_id}",
                    capture_run_id=f"reference-run-{asset_id}",
                    sequence_id=f"reference-sequence-{asset_id}",
                    sampling_cell=f"reference-cell-{asset_id}",
                ),
            )
        )
    return assets, vectors


def _build_dataset(tmp_path: Path) -> tuple[DescriptorDataset, list[DescriptorAsset]]:
    assets, vectors = _make_assets(tmp_path / "fixtures")
    result = build_descriptors(
        assets,
        StaticDescriptorProvider(vectors),
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "descriptors",
        manifest_hash=MANIFEST_HASH,
        source_policy_hash=SOURCE_POLICY_HASH,
        batch_size=2,
    )
    return result.dataset, assets


def _queries(*, distant_second: bool = False) -> tuple[BenchmarkQuery, BenchmarkQuery]:
    return (
        BenchmarkQuery(
            query_id="holdout-ankara",
            descriptor=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            relevant_asset_ids=("asset-a",),
            content_sha256="d" * 64,
            holdout_asset_id="holdout-asset-ankara",
            source_id="holdout-source-ankara",
            province="Ankara",
            latitude=39.9534,
            longitude=32.8597,
            contributor_id="holdout-contributor-a",
            capture_run_id="holdout-run-a",
            sequence_id="holdout-sequence-a",
            sampling_cell="holdout-cell-a",
        ),
        BenchmarkQuery(
            query_id="holdout-kayseri",
            descriptor=np.asarray(
                [-1.0, 0.0, 0.0] if distant_second else [0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
            relevant_asset_ids=("asset-b",),
            content_sha256="e" * 64,
            holdout_asset_id="holdout-asset-kayseri",
            source_id="holdout-source-kayseri",
            province="Kayseri",
            latitude=38.7405,
            longitude=35.4826,
            contributor_id="holdout-contributor-b",
            capture_run_id="holdout-run-b",
            sequence_id="holdout-sequence-b",
            sampling_cell="holdout-cell-b",
        ),
    )


def test_benchmark_query_requires_complete_locked_lineage_and_coordinates() -> None:
    query = _queries()[0]
    assert {
        "holdout_asset_id",
        "source_id",
        "contributor_id",
        "capture_run_id",
        "sequence_id",
        "sampling_cell",
        "province",
        "latitude",
        "longitude",
    } <= query.lock_record().keys()
    for field in (
        "holdout_asset_id",
        "source_id",
        "contributor_id",
        "capture_run_id",
        "sequence_id",
        "sampling_cell",
        "province",
    ):
        with pytest.raises(ValueError, match="must not be blank"):
            replace(query, **{field: " "})
    with pytest.raises(ValueError, match="latitude"):
        replace(query, latitude=float("nan"))


def test_production_registry_fails_closed_and_rejects_test_provider(tmp_path: Path) -> None:
    registry = ProductionDescriptorRegistry()
    with pytest.raises(ProviderNotConfiguredError):
        registry.active()

    test_provider = TestOnlyDeterministicDescriptorProvider(dimension=3)
    approval = ProviderApproval(
        provider_id=test_provider.spec.provider_id,
        version=test_provider.spec.version,
        dimension=test_provider.spec.dimension,
        artifact_sha256="1" * 64,
        license_record_id="test-license",
        rights_approved=True,
    )
    with pytest.raises(ProviderRegistrationError, match="test-only"):
        registry.register(test_provider, approval)
    with pytest.raises(ProviderNotConfiguredError):
        registry.activate(test_provider.spec.identity)

    assets, vectors = _make_assets(tmp_path)
    with pytest.raises(ProviderNotConfiguredError):
        build_production_descriptors(
            assets,
            registry,
            work_dir=tmp_path / "production-work",
            output_dir=tmp_path / "production-output",
            manifest_hash=MANIFEST_HASH,
            source_policy_hash=SOURCE_POLICY_HASH,
        )
    assert not (tmp_path / "production-work").exists()
    assert not (tmp_path / "production-output").exists()

    production = ProductionStubProvider(vectors)
    denied = ProviderApproval(
        provider_id=production.spec.provider_id,
        version=production.spec.version,
        dimension=production.spec.dimension,
        artifact_sha256="2" * 64,
        license_record_id="review-record",
        rights_approved=False,
    )
    with pytest.raises(ProviderRegistrationError, match="not approved"):
        registry.register(production, denied)
    approved = ProviderApproval(
        provider_id=production.spec.provider_id,
        version=production.spec.version,
        dimension=production.spec.dimension,
        artifact_sha256="2" * 64,
        license_record_id="review-record",
        rights_approved=True,
    )
    registry.register(production, approved)
    registry.activate(production.spec.identity)
    assert registry.active() is production


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1.0, np.nan, 0.0], dtype=np.float32),
        np.asarray([1.0, np.inf, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
    ],
)
def test_vector_validation_rejects_corrupt_values(value: np.ndarray) -> None:
    with pytest.raises(DescriptorValidationError):
        normalize_vector(value, dimension=3)


def test_descriptor_generation_resumes_and_rejects_incompatible_checkpoint(
    tmp_path: Path,
) -> None:
    assets, vectors = _make_assets(tmp_path / "fixtures")
    arguments = {
        "work_dir": tmp_path / "work",
        "output_dir": tmp_path / "descriptors",
        "manifest_hash": MANIFEST_HASH,
        "source_policy_hash": SOURCE_POLICY_HASH,
    }
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        build_descriptors(
            assets,
            StaticDescriptorProvider(vectors, fail_on_call=2),
            batch_size=2,
            **arguments,
        )
    assert not (tmp_path / "descriptors").exists()
    checkpoint = json.loads((tmp_path / "work" / "descriptor-checkpoint.json").read_text())
    assert checkpoint["completed_assets"] == 2

    with pytest.raises(CheckpointCompatibilityError, match="incompatible"):
        build_descriptors(
            assets,
            StaticDescriptorProvider(vectors),
            batch_size=1,
            **arguments,
        )
    resumed = build_descriptors(
        assets,
        StaticDescriptorProvider(vectors),
        batch_size=2,
        **arguments,
    )
    assert resumed.resumed_assets == 2
    assert resumed.generated_assets == 1
    assert resumed.dataset.count == 3
    assert np.linalg.norm(resumed.dataset.vectors, axis=1) == pytest.approx([1.0, 1.0, 1.0])

    idempotent = build_descriptors(
        assets,
        StaticDescriptorProvider(vectors),
        batch_size=2,
        **arguments,
    )
    assert idempotent.resumed_assets == 3
    assert idempotent.generated_assets == 0


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[np.nan, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[np.inf, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
    ],
)
def test_descriptor_generation_rejects_bad_provider_batch(
    tmp_path: Path, invalid: FloatMatrix
) -> None:
    assets, _ = _make_assets(tmp_path / "fixtures")
    with pytest.raises(DescriptorValidationError):
        build_descriptors(
            assets[:1],
            InvalidDescriptorProvider(invalid),
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "descriptors",
            manifest_hash=MANIFEST_HASH,
            source_policy_hash=SOURCE_POLICY_HASH,
        )
    assert not (tmp_path / "descriptors").exists()


def test_exact_index_is_deterministic_and_checksum_bound(tmp_path: Path) -> None:
    dataset, _ = _build_dataset(tmp_path)
    index = build_index(
        dataset,
        output_dir=tmp_path / "exact-index",
        backend="exact",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
        shard_size=2,
    )
    hits = index.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)
    assert [hit.reference_asset_id for hit in hits] == ["asset-a", "asset-b", "asset-c"]
    assert hits[0].cosine_distance == pytest.approx(0.0)
    assert hits[0].provenance_summary == "generated synthetic numeric fixture"
    assert index.size == 3
    assert len(index.artifact_inventory()) == 6

    reopened = PublishedCorpusIndex.open(
        tmp_path / "exact-index",
        expected_spec=dataset.spec,
        expected_source_policy_hash=SOURCE_POLICY_HASH,
    )
    assert reopened.search(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), top_k=1)[
        0
    ].reference_asset_id == "asset-b"
    with pytest.raises(IndexCompatibilityError):
        PublishedCorpusIndex.open(
            tmp_path / "exact-index",
            expected_spec=DescriptorSpec("wrong", "1", 3),
        )

    vector_path = tmp_path / "exact-index" / "shards" / "shard-000000" / "vectors.npy"
    with vector_path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        PublishedCorpusIndex.open(tmp_path / "exact-index")


def test_real_faiss_backend_publishes_and_matches_exact_search(tmp_path: Path) -> None:
    dataset, _ = _build_dataset(tmp_path)
    exact = build_index(
        dataset,
        output_dir=tmp_path / "exact",
        backend="exact",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
        shard_size=2,
    )
    faiss_index = build_index(
        dataset,
        output_dir=tmp_path / "faiss",
        backend="faiss",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
        shard_size=2,
    )
    query = np.asarray([0.2, 0.8, 0.0], dtype=np.float32)
    assert [hit.reference_asset_id for hit in faiss_index.search(query, top_k=3)] == [
        hit.reference_asset_id for hit in exact.search(query, top_k=3)
    ]
    assert faiss_index.backend == "faiss-flat-ip-v1"
    metadata = json.loads((tmp_path / "faiss" / "metadata.json").read_text())
    assert metadata["descriptor_spec"]["dimension"] == 3
    assert metadata["source_policy_hash"] == SOURCE_POLICY_HASH


def test_index_requires_paired_holdout_and_split_locks(tmp_path: Path) -> None:
    dataset, _ = _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="provided together"):
        build_index(
            dataset,
            output_dir=tmp_path / "partial-holdout",
            backend="exact",
            index_version="phase3b-test-v1",
            source_policy_hash=SOURCE_POLICY_HASH,
            locked_holdout_hash="d" * 64,
        )
    with pytest.raises(ValueError, match="provided together"):
        build_index(
            dataset,
            output_dir=tmp_path / "partial-split",
            backend="exact",
            index_version="phase3b-test-v1",
            source_policy_hash=SOURCE_POLICY_HASH,
            split_lock_hash=SPLIT_LOCK_HASH,
        )

    unbound = build_index(
        dataset,
        output_dir=tmp_path / "unbound-index",
        backend="exact",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
    )
    assert unbound.locked_holdout_hash is None
    assert unbound.split_lock_hash is None
    with pytest.raises(IndexCompatibilityError, match="must be paired"):
        PublishedCorpusIndex.open(
            tmp_path / "unbound-index",
            expected_locked_holdout_hash="d" * 64,
        )

    metadata_path = tmp_path / "unbound-index" / "metadata.json"
    marker_path = tmp_path / "unbound-index" / "PUBLISHED.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["locked_holdout_hash"] = "d" * 64
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    marker = json.loads(marker_path.read_text())
    marker["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="not paired"):
        PublishedCorpusIndex.open(tmp_path / "unbound-index")


def test_revocation_impact_and_rebuild_plan_are_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    dataset, _ = _build_dataset(tmp_path)
    index = build_index(
        dataset,
        output_dir=tmp_path / "index",
        backend="exact",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
        shard_size=2,
    )
    impact = index.revocation_impact("asset-b")
    assert impact.found is True
    assert impact.shard_ids == ("shard-000000",)
    assert set(impact.artifact_paths) == {
        "PUBLISHED.json",
        "metadata.json",
        "shards/shard-000000/assets.json",
        "shards/shard-000000/vectors.npy",
    }
    assert index.revocation_impact("missing").found is False

    first = index.rebuild_plan(["missing", "asset-b", "asset-b"])
    second = index.rebuild_plan(["asset-b", "missing"])
    assert first == second
    assert first.removed_asset_count == 1
    assert first.retained_asset_count == 2
    assert first.missing_asset_ids == ("missing",)
    receipt = tmp_path / "rebuild-plan.json"
    write_rebuild_plan(receipt, first)
    assert json.loads(receipt.read_text())["plan_id"] == first.plan_id


def test_locked_benchmark_metrics_province_error_and_abstention(tmp_path: Path) -> None:
    dataset, _ = _build_dataset(tmp_path)
    queries = _queries()
    holdout_hash = compute_locked_holdout_hash(queries)
    index = build_index(
        dataset,
        output_dir=tmp_path / "index",
        backend="faiss",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
        locked_holdout_hash=holdout_hash,
        split_lock_hash=SPLIT_LOCK_HASH,
    )
    result = evaluate_locked_holdout(
        index,
        queries,
        expected_holdout_hash=holdout_hash,
        expected_split_lock_hash=SPLIT_LOCK_HASH,
    )
    assert (result.recall_at_1, result.recall_at_5, result.recall_at_10) == (1.0, 1.0, 1.0)
    assert result.province_recall["Ankara"].recall_at_1 == 1.0
    assert result.province_recall["Kayseri"].query_count == 1
    assert result.median_haversine_error_m is not None
    assert result.median_haversine_error_m > 1_000.0
    assert result.p90_haversine_error_m is not None
    assert result.p90_haversine_error_m >= result.median_haversine_error_m
    assert result.abstention_coverage == 1.0

    distant_queries = _queries(distant_second=True)
    distant_hash = compute_locked_holdout_hash(distant_queries)
    distant_index = build_index(
        dataset,
        output_dir=tmp_path / "distant-index",
        backend="exact",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
        locked_holdout_hash=distant_hash,
        split_lock_hash=SPLIT_LOCK_HASH,
    )
    abstained = evaluate_locked_holdout(
        distant_index,
        distant_queries,
        expected_holdout_hash=distant_hash,
        expected_split_lock_hash=SPLIT_LOCK_HASH,
        abstain_if_distance_gt=0.5,
    )
    assert abstained.abstention_coverage == 0.5
    assert abstained.abstained_queries == 1
    assert abstained.recall_at_1 == 0.5


def test_benchmark_rejects_holdout_tampering_and_leakage_before_scoring(tmp_path: Path) -> None:
    dataset, assets = _build_dataset(tmp_path)
    queries = _queries()
    holdout_hash = compute_locked_holdout_hash(queries)
    index = build_index(
        dataset,
        output_dir=tmp_path / "index",
        backend="exact",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
        locked_holdout_hash=holdout_hash,
        split_lock_hash=SPLIT_LOCK_HASH,
    )
    with pytest.raises(HoldoutIntegrityError, match="mismatch"):
        evaluate_locked_holdout(
            index,
            queries,
            expected_holdout_hash="f" * 64,
            expected_split_lock_hash=SPLIT_LOCK_HASH,
        )

    leaked = (
        BenchmarkQuery(
            query_id="leaked",
            descriptor=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            relevant_asset_ids=("asset-a",),
            content_sha256=assets[0].provenance.content_sha256,
            holdout_asset_id="leaked-holdout-asset",
            source_id="independent-leaked-source",
            contributor_id="independent-leaked-contributor",
            capture_run_id="independent-leaked-run",
            sequence_id="independent-leaked-sequence",
            sampling_cell="independent-leaked-cell",
            province="Ankara",
            latitude=39.9534,
            longitude=32.8597,
        ),
    )
    leaked_hash = compute_locked_holdout_hash(leaked)
    leaked_index = build_index(
        dataset,
        output_dir=tmp_path / "leaked-index",
        backend="exact",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
        locked_holdout_hash=leaked_hash,
        split_lock_hash=SPLIT_LOCK_HASH,
    )
    with pytest.raises(BenchmarkLeakageError, match="rejected"):
        evaluate_locked_holdout(
            leaked_index,
            leaked,
            expected_holdout_hash=leaked_hash,
            expected_split_lock_hash=SPLIT_LOCK_HASH,
        )

    spatial = detect_prebenchmark_leakage(
        dataset.assets,
        (
            BenchmarkQuery(
                query_id="spatial",
                descriptor=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                relevant_asset_ids=("asset-a",),
                content_sha256="c" * 64,
                holdout_asset_id="spatial-holdout-asset",
                source_id="independent-spatial-source",
                contributor_id="independent-spatial-contributor",
                capture_run_id="independent-spatial-run",
                sequence_id="independent-spatial-sequence",
                sampling_cell="independent-spatial-cell",
                province="Ankara",
                latitude=39.9334,
                longitude=32.8597,
            ),
        ),
        minimum_spatial_separation_m=50.0,
    )
    assert any(violation.kind == "spatial_separation" for violation in spatial)

    source_overlap = detect_prebenchmark_leakage(
        dataset.assets,
        (
            BenchmarkQuery(
                query_id="source-overlap",
                descriptor=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                relevant_asset_ids=("asset-a",),
                content_sha256="9" * 64,
                holdout_asset_id="source-overlap-holdout-asset",
                source_id=dataset.assets[0].source_id,
                contributor_id="independent-source-contributor",
                capture_run_id="independent-source-run",
                sequence_id="independent-source-sequence",
                sampling_cell="independent-source-cell",
                province="Ankara",
                latitude=41.0,
                longitude=32.0,
            ),
        ),
    )
    assert any(violation.kind == "source_id" for violation in source_overlap)

    spatial_query = (
        BenchmarkQuery(
            query_id="spatial-overlap",
            descriptor=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            relevant_asset_ids=("asset-a",),
            content_sha256="8" * 64,
            holdout_asset_id="spatial-overlap-holdout-asset",
            source_id="independent-holdout-source",
            contributor_id="spatial-overlap-contributor",
            capture_run_id="spatial-overlap-run",
            sequence_id="spatial-overlap-sequence",
            sampling_cell="spatial-overlap-cell",
            province="Ankara",
            latitude=39.9334,
            longitude=32.8597,
        ),
    )
    spatial_hash = compute_locked_holdout_hash(spatial_query)
    spatial_index = build_index(
        dataset,
        output_dir=tmp_path / "spatial-index",
        backend="exact",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
        locked_holdout_hash=spatial_hash,
        split_lock_hash=SPLIT_LOCK_HASH,
    )
    with pytest.raises(BenchmarkLeakageError, match="rejected"):
        evaluate_locked_holdout(
            spatial_index,
            spatial_query,
            expected_holdout_hash=spatial_hash,
            expected_split_lock_hash=SPLIT_LOCK_HASH,
        )

    def always_reject(
        references: Sequence[AssetProvenance],
        benchmark_queries: Sequence[BenchmarkQuery],
    ) -> Sequence[LeakageViolation]:
        return [LeakageViolation(benchmark_queries[0].query_id, references[0].asset_id, "hook")]

    with pytest.raises(BenchmarkLeakageError):
        evaluate_locked_holdout(
            index,
            queries,
            expected_holdout_hash=holdout_hash,
            expected_split_lock_hash=SPLIT_LOCK_HASH,
            leakage_hook=always_reject,
        )


def test_benchmark_rejects_nonpositive_or_nonfinite_spatial_gate(tmp_path: Path) -> None:
    dataset, _ = _build_dataset(tmp_path)
    queries = _queries()
    holdout_hash = compute_locked_holdout_hash(queries)
    index = build_index(
        dataset,
        output_dir=tmp_path / "index",
        backend="exact",
        index_version="phase3b-test-v1",
        source_policy_hash=SOURCE_POLICY_HASH,
        locked_holdout_hash=holdout_hash,
        split_lock_hash=SPLIT_LOCK_HASH,
    )
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="strictly positive"):
            evaluate_locked_holdout(
                index,
                queries,
                expected_holdout_hash=holdout_hash,
                expected_split_lock_hash=SPLIT_LOCK_HASH,
                minimum_spatial_separation_m=invalid,
            )


def test_descriptor_builder_rejects_revoked_or_unvalidated_assets(tmp_path: Path) -> None:
    assets, vectors = _make_assets(tmp_path / "fixtures")
    original = assets[0]
    rejected = AssetProvenance(
        **{
            **original.provenance.to_json(),
            "rights_validated": False,
            "revoked": True,
        }
    )
    with pytest.raises(DescriptorValidationError, match="rights-valid"):
        build_descriptors(
            [DescriptorAsset(original.locator, rejected)],
            StaticDescriptorProvider(vectors),
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "descriptors",
            manifest_hash=MANIFEST_HASH,
            source_policy_hash=SOURCE_POLICY_HASH,
        )
