from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import atlaslens_api.mapillary_demo.runner as mapillary_runner_module
from atlaslens_api.corpus_index.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_json,
    sha256_path,
)
from atlaslens_api.corpus_index.errors import ArtifactIntegrityError
from atlaslens_api.corpus_index.models import DescriptorSpec, FloatMatrix
from atlaslens_api.mapillary_demo.benchmark import (
    evaluate_mapillary_holdout,
    run_operator_query_smoke,
)
from atlaslens_api.mapillary_demo.indexing import (
    MEGALOC_PHASE3B2_ARTIFACT_SHA256,
    MapillaryPrivateDemoDescriptorProvider,
    PublishedMapillaryDemoIndex,
    build_mapillary_demo_index,
)
from atlaslens_api.mapillary_demo.manifest import (
    ingest_with_phase3b1,
    materialize_phase3b1_manifest,
)
from atlaslens_api.mapillary_demo.models import (
    AcquiredImage,
    AcquisitionCheckpoint,
    AcquisitionManifest,
    CoverageAudit,
    CoverageSummary,
    GeoPoint,
)
from atlaslens_api.mapillary_demo.runner import (
    Phase3B3RunError,
    Phase3B3RunPaths,
    _validate_performance_receipt,
    _write_performance_receipt,
    execute_private_demo_run,
    plan_private_demo_run,
)
from atlaslens_api.mapillary_demo.selection import (
    SelectionLimits,
    choose_pilot_city,
    select_locked_split,
)
from atlaslens_api.megaloc_adapter import provider as megaloc_provider_module
from atlaslens_api.megaloc_adapter.errors import MegaLocAdapterError
from atlaslens_api.megaloc_adapter.models import (
    MEGALOC_PREPROCESSING_ID,
    MegaLocAdapterConfig,
    MegaLocApproval,
)
from atlaslens_api.megaloc_adapter.provider import (
    MegaLocDescriptorProvider,
    PrivateDemoExecutionMetrics,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
SOURCE_POLICY = REPOSITORY_ROOT / "config" / "phase3b3" / "mapillary-source-policy-v1.json"
MANIFEST_SCHEMA = REPOSITORY_ROOT / "config" / "corpus" / "manifest-schema-v1.json"


def _summary(
    aoi_id: str,
    display_name: str,
    *,
    region_kind: str,
    province: str,
    city: str | None,
    images: int,
    cells: int,
    sequences: int,
    contributors: int,
    eligible: bool = True,
) -> CoverageSummary:
    return CoverageSummary.model_validate(
        {
            "aoi_id": aoi_id,
            "aoi_version": "v1",
            "display_name": display_name,
            "region_kind": region_kind,
            "province": province,
            "city": city,
            "image_count": images,
            "rejected_item_count": 0,
            "sequence_count": sequences,
            "contributor_count": contributors,
            "capture_year_distribution": {"2024": images},
            "spatial_cell_count": cells,
            "compass_direction_bins": {"N": images},
            "approximate_road_km": 100.0,
            "approximate_images_per_road_km": images / 100.0,
            "approximate_sequence_density": sequences / 100.0,
            "estimated_selected_download_bytes": images * 1024,
            "expected_descriptor_bytes": images * 8448 * 4,
            "expected_index_bytes": images * 8448 * 4,
            "eligible_for_selection": eligible,
            "metadata_sha256": hashlib.sha256(aoi_id.encode()).hexdigest(),
        }
    )


def _audit(
    *,
    kayseri_images: int = 1_200,
    kayseri_cells: int = 20,
    kayseri_sequences: int = 12,
    selected: str = "kayseri-urban-v1",
) -> CoverageAudit:
    return CoverageAudit(
        catalog_version="phase3b3-aoi-v1",
        audited_at=datetime(2026, 7, 17, 9, 0, tzinfo=UTC),
        request_count=5,
        page_count=5,
        regions=(
            _summary(
                "kayseri-urban-v1",
                "Kayseri urban area",
                region_kind="urban",
                province="Kayseri",
                city="Kayseri",
                images=kayseri_images,
                cells=kayseri_cells,
                sequences=kayseri_sequences,
                contributors=8,
            ),
            _summary(
                "ankara-urban-v1",
                "Ankara urban area",
                region_kind="urban",
                province="Ankara",
                city="Ankara",
                images=1_100,
                cells=50,
                sequences=40,
                contributors=20,
            ),
            _summary(
                "sivas-urban-v1",
                "Sivas urban area",
                region_kind="urban",
                province="Sivas",
                city="Sivas",
                images=500,
                cells=30,
                sequences=15,
                contributors=7,
            ),
            _summary(
                "kayseri-ankara-corridor-v1",
                "Kayseri-Ankara corridor",
                region_kind="corridor",
                province="Kayseri-Ankara",
                city=None,
                images=300,
                cells=40,
                sequences=10,
                contributors=6,
            ),
            _summary(
                "kayseri-sivas-corridor-v1",
                "Kayseri-Sivas corridor",
                region_kind="corridor",
                province="Kayseri-Sivas",
                city=None,
                images=250,
                cells=35,
                sequences=9,
                contributors=5,
            ),
        ),
        selected_aoi_id=selected,
        selection_reason="synthetic audit fixture",
    )


def _write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def _acquisition(tmp_path: Path) -> tuple[AcquisitionManifest, Path]:
    root = tmp_path / "acquisition"
    policy_hash = sha256_path(SOURCE_POLICY)
    specifications = (
        ("a", "seq-a", "creator-a", 35.00000, 1),
        ("a-adjacent", "seq-a", "creator-a", 35.00004, 2),
        ("b", "seq-b", "creator-b", 35.00018, 3),
        ("c", "seq-c", "creator-c", 35.00100, 4),
        ("d", "seq-d", "creator-d", 35.00122, 5),
        ("e", "seq-e", "creator-e", 35.00200, 6),
        ("f", "seq-f", "creator-f", 35.00225, 7),
        ("g", "seq-g", "creator-g", 35.00300, 8),
    )
    assets: list[AcquiredImage] = []
    total_bytes = 0
    for index, (image_id, sequence, creator, longitude, seed) in enumerate(specifications):
        relative = f"images/{image_id}.png"
        path = root / relative
        _write_image(path, seed)
        digest = sha256_path(path)
        size = path.stat().st_size
        total_bytes += size
        assets.append(
            AcquiredImage(
                mapillary_image_id=image_id,
                computed_geometry=GeoPoint(coordinates=(longitude, 38.7205)),
                captured_at=datetime(2024 + index % 2, 1, 1, tzinfo=UTC),
                compass_angle=90.0,
                sequence_id=sequence,
                creator_id=creator,
                width_px=48,
                height_px=48,
                source_page_url=f"https://www.mapillary.com/app/?pKey={image_id}",
                attribution_text=f"Mapillary contributor {creator} / {image_id}",
                acquired_at=datetime(2026, 7, 17, 10, index, tzinfo=UTC),
                raw_sha256=digest,
                normalized_sha256=digest,
                relative_path=relative,
                byte_size=size,
                normalized_byte_size=size,
            )
        )
    return (
        AcquisitionManifest(
            aoi_id="kayseri-urban-v1",
            aoi_version="v1",
            source_policy_receipt_sha256=policy_hash,
            coverage_audit_sha256="a" * 64,
            created_at=datetime(2026, 7, 17, 11, 0, tzinfo=UTC),
            request_count=9,
            page_count=1,
            downloaded_bytes=total_bytes,
            assets=tuple(assets),
        ),
        root,
    )


@pytest.fixture
def locked_split(tmp_path: Path) -> tuple[Any, Path]:
    acquisition, root = _acquisition(tmp_path)
    split = select_locked_split(
        acquisition,
        _audit(),
        acquisition_root=root,
        province_code="TR-38",
        limits=SelectionLimits(reference_target=3, holdout_target=2, hard_image_cap=8),
    )
    return split, root


def test_city_selection_requires_explicit_distribution_gate_and_truthful_fallback() -> None:
    fallback = choose_pilot_city(
        _audit(
            kayseri_images=1_500,
            kayseri_cells=9,
            kayseri_sequences=20,
            selected="ankara-urban-v1",
        )
    )
    assert fallback.city == "Ankara"
    assert not fallback.kayseri_eligible

    too_few_sequences = choose_pilot_city(
        _audit(
            kayseri_images=1_500,
            kayseri_cells=20,
            kayseri_sequences=1,
            selected="ankara-urban-v1",
        )
    )
    assert too_few_sequences.city == "Ankara"

    kayseri = choose_pilot_city(_audit())
    assert kayseri.city == "Kayseri"
    assert kayseri.kayseri_eligible


def test_split_is_sequence_aware_geographic_and_reports_underfill(
    locked_split: tuple[Any, Path],
) -> None:
    split, _ = locked_split
    split.verify()
    reference_sequences = {asset.lineage_sequence for asset in split.references}
    holdout_sequences = {asset.lineage_sequence for asset in split.holdout}
    assert not reference_sequences & holdout_sequences
    assert split.suppressed_stationary_or_adjacent == 1
    assert split.reference_shortfall == 0
    assert split.holdout_shortfall == 0
    assert all(split.positive_reference_ids[query.asset_id]["1000"] for query in split.holdout)

    with pytest.raises(ValueError, match="checksum mismatch"):
        replace(split, lock_sha256="0" * 64).verify()


class _SyntheticExecutor:
    def __init__(
        self,
        *,
        keys: dict[str, int] | None = None,
        artifact_sha256: str = MEGALOC_PHASE3B2_ARTIFACT_SHA256,
        fail_on_call: int | None = None,
    ) -> None:
        self._spec = DescriptorSpec(
            provider_id="megaloc",
            version=("megaloc-test-7cb9f797-d4f9f2bcb600-max560-imagenet"),
            dimension=8448,
            artifact_sha256=artifact_sha256,
            preprocessing_version=MEGALOC_PREPROCESSING_ID,
        )
        self.keys = keys or {}
        self.fail_on_call = fail_on_call
        self.calls = 0

    @property
    def spec(self) -> DescriptorSpec:
        return self._spec

    def describe_batch_for_private_demo(
        self,
        locators: list[Path] | tuple[Path, ...],
        *,
        authorization_id: str,
        source_policy_sha256: str,
    ) -> FloatMatrix:
        assert authorization_id == "phase3b3-mapillary-private-demo-v1"
        assert len(source_policy_sha256) == 64
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("synthetic descriptor interruption")
        matrix = np.zeros((len(locators), 8448), dtype=np.float32)
        for row, locator in enumerate(locators):
            fallback = int(
                hashlib.sha256(locator.name.encode()).hexdigest()[:8],
                16,
            )
            key = self.keys.get(locator.stem, fallback)
            matrix[row, key % 8448] = 1.0
        return matrix


def _adapter_config(tmp_path: Path) -> MegaLocAdapterConfig:
    approval = MegaLocApproval(
        artifact_hash_approved=False,
        source_code_approved=False,
        source_license_approved=False,
        weight_license_approved=False,
        descriptor_contract_approved=False,
        preprocessing_approved=False,
        production_use_approved=False,
        approval_record_id="not-approved",
        weight_license_record_id="not-approved",
    )
    return MegaLocAdapterConfig(
        project_root=tmp_path,
        source_dir=tmp_path / "source",
        source_code_path=tmp_path / "source" / "megaloc_model.py",
        source_code_sha256="1" * 64,
        source_revision="1af071c68fc3ab6c6018c5c868391763516e50f7",
        source_license_path=tmp_path / "source" / "LICENSE",
        source_license_sha256="2" * 64,
        source_license_spdx="MIT",
        artifact_path=tmp_path / "model.safetensors",
        artifact_sha256=MEGALOC_PHASE3B2_ARTIFACT_SHA256,
        artifact_size=1,
        artifact_format="safetensors",
        receipt_path=tmp_path / "receipt.json",
        receipt_sha256="3" * 64,
        smoke_receipt_path=tmp_path / "smoke.json",
        model_id="gberton/MegaLoc",
        model_revision="7cb9f7970d366fdf059963d04d372e503e8e9df9",
        weight_license_spdx="MIT",
        descriptor_version=("megaloc-7cb9f797-d4f9f2bcb600-max560-imagenet-private-demo"),
        descriptor_dimension=8448,
        preprocessing_id=MEGALOC_PREPROCESSING_ID,
        maximum_edge=560,
        device="cpu",
        max_batch_size=2,
        max_input_bytes=1024,
        max_image_pixels=10_000,
        deterministic_algorithms=True,
        reduce_batch_on_cuda_oom=True,
        approval=approval,
    )


class _InjectedBackend:
    @property
    def device(self) -> str:
        return "cpu"

    def describe_batch(self, locators: list[Path] | tuple[Path, ...]) -> FloatMatrix:
        return np.ones((len(locators), 8448), dtype=np.float32) / np.sqrt(8448)

    def recover_cuda_oom(self) -> None:
        return None

    def close(self) -> None:
        return None


class _MetricBackend(_InjectedBackend):
    def __init__(self) -> None:
        self.reset_calls = 0

    @property
    def peak_vram_bytes(self) -> int:
        return 0

    def reset_peak_vram(self) -> None:
        self.reset_calls += 1


def test_private_demo_metrics_accumulate_and_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _MetricBackend()
    monkeypatch.setattr(
        megaloc_provider_module,
        "_verify_integrity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        megaloc_provider_module,
        "_verify_smoke_receipt",
        lambda _config: True,
    )
    monkeypatch.setattr(
        megaloc_provider_module,
        "_create_torch_backend",
        lambda _config, _device: backend,
    )
    ticks = iter((10.0, 10.25, 20.0, 20.5))
    monkeypatch.setattr(megaloc_provider_module, "perf_counter", lambda: next(ticks))
    locators = tuple(tmp_path / f"image-{index}.bin" for index in range(3))
    for locator in locators:
        locator.write_bytes(b"synthetic-private-demo-image")

    provider = MegaLocDescriptorProvider(_adapter_config(tmp_path))
    try:
        first = provider.describe_batch_for_private_demo(
            locators[:2],
            authorization_id="phase3b3-mapillary-private-demo-v1",
            source_policy_sha256="a" * 64,
        )
        second = provider.describe_batch_for_private_demo(
            locators[2:],
            authorization_id="phase3b3-mapillary-private-demo-v1",
            source_policy_sha256="a" * 64,
        )
        assert first.shape == (2, 8448)
        assert second.shape == (1, 8448)
        metrics = provider.private_demo_metrics
        assert metrics.device == "cpu"
        assert metrics.image_count == 3
        assert metrics.batch_call_count == 2
        assert metrics.elapsed_seconds == pytest.approx(0.75)
        assert metrics.peak_vram_bytes == 0

        provider.reset_private_demo_metrics()
        reset = provider.private_demo_metrics
        assert reset.image_count == 0
        assert reset.batch_call_count == 0
        assert reset.elapsed_seconds == 0.0
        assert backend.reset_calls == 1
    finally:
        provider.close()


def test_private_demo_artifact_and_megaloc_authorization_gates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact Phase 3B2"):
        MapillaryPrivateDemoDescriptorProvider(
            _SyntheticExecutor(artifact_sha256="f" * 64),
            source_policy_sha256="a" * 64,
        )

    config = _adapter_config(tmp_path)
    injected = MegaLocDescriptorProvider(
        config,
        backend_factory=lambda _config, _device: _InjectedBackend(),  # type: ignore[arg-type]
    )
    with pytest.raises(MegaLocAdapterError, match="MEGALOC_PRIVATE_DEMO_NOT_AUTHORIZED"):
        injected.describe_batch_for_private_demo(
            (), authorization_id="wrong", source_policy_sha256="a" * 64
        )
    with pytest.raises(MegaLocAdapterError, match="MEGALOC_PRIVATE_DEMO_POLICY_INVALID"):
        injected.describe_batch_for_private_demo(
            (),
            authorization_id="phase3b3-mapillary-private-demo-v1",
            source_policy_sha256="invalid",
        )
    with pytest.raises(MegaLocAdapterError, match="MEGALOC_PRIVATE_DEMO_REAL_BACKEND_REQUIRED"):
        injected.describe_batch_for_private_demo(
            (),
            authorization_id="phase3b3-mapillary-private-demo-v1",
            source_policy_sha256="a" * 64,
        )
    assert injected.runtime_kind == "test_only"
    assert not injected.readiness.ready

    production = MegaLocDescriptorProvider(config)
    with pytest.raises(MegaLocAdapterError, match="MEGALOC_LICENSE_APPROVAL_MISSING"):
        _ = production.runtime_kind
    assert not production.readiness.ready


def test_phase3b1_descriptor_resume_faiss_metadata_and_geographic_benchmark(
    tmp_path: Path,
    locked_split: tuple[Any, Path],
) -> None:
    split, acquisition_root = locked_split
    corpus_root = tmp_path / "corpus"
    manifest = materialize_phase3b1_manifest(
        split,
        acquisition_root=acquisition_root,
        corpus_root=corpus_root,
        manifest_path=None,
        source_policy_path=SOURCE_POLICY,
        coordinate_uncertainty_m=100.0,
    )
    ingestion = ingest_with_phase3b1(
        manifest,
        work_root=tmp_path / "ingestion",
        checkpoint_root=tmp_path / "checkpoints",
        manifest_schema_path=MANIFEST_SCHEMA,
        source_policy_path=SOURCE_POLICY,
    )
    reference_key = {asset.asset_id: index + 1 for index, asset in enumerate(split.references)}
    keys = dict(reference_key)
    for query in split.holdout:
        positive = split.positive_reference_ids[query.asset_id]["1000"][0]
        keys[query.asset_id] = reference_key[positive]

    with pytest.raises(RuntimeError, match="synthetic descriptor interruption"):
        build_mapillary_demo_index(
            ingestion,
            split,
            _SyntheticExecutor(keys=keys, fail_on_call=2),
            corpus_root=corpus_root,
            descriptor_work_root=tmp_path / "descriptor-work",
            descriptor_output_root=tmp_path / "descriptors",
            output_dir=tmp_path / "index-bundle",
            batch_size=1,
        )

    result = build_mapillary_demo_index(
        ingestion,
        split,
        _SyntheticExecutor(keys=keys),
        corpus_root=corpus_root,
        descriptor_work_root=tmp_path / "descriptor-work",
        descriptor_output_root=tmp_path / "descriptors",
        output_dir=tmp_path / "index-bundle",
        batch_size=1,
    )
    assert result.reference_descriptors.resumed_assets == 1
    assert result.publication.index.backend == "faiss-flat-ip-v1"
    assert result.publication.size == len(split.references)
    assert result.publication.metadata["model_artifact_sha256"] == (
        MEGALOC_PHASE3B2_ARTIFACT_SHA256
    )
    assert result.publication.metadata["selection_lock_sha256"] == split.lock_sha256
    assert result.publication.metadata["source_policy_sha256"] == sha256_path(SOURCE_POLICY)
    assert {
        record.mapillary_image_id
        for record in result.publication.attribution.values()
        if record.split == "reference"
    } == {asset.mapillary_image_id for asset in split.references}

    benchmark = evaluate_mapillary_holdout(
        result.publication,
        result.holdout_descriptors.dataset,
        split,
        expected_selection_lock_sha256=split.lock_sha256,
    )
    assert set(benchmark.geographic_recall) == {"25", "100", "500", "1000"}
    assert benchmark.query_count == len(split.holdout)
    assert benchmark.geographic_recall["1000"].recall_at_1 == 1.0
    assert benchmark.city_accuracy == 1.0
    assert benchmark.province_accuracy == 1.0

    operator_image = corpus_root / "assets" / f"{split.holdout[0].asset_id}.png"
    smoke = run_operator_query_smoke(
        operator_image,
        _SyntheticExecutor(keys=keys),
        result.publication,
        claimed_city="Kayseri",
        abstain_if_cosine_distance_gt=2.0,
    )
    assert len(smoke.candidates) <= 5
    assert smoke.claimed_city_appears
    assert not smoke.exact_ground_truth_available
    assert all(candidate.source_page_url.startswith("https://") for candidate in smoke.candidates)

    attribution_path = result.publication.directory / "mapillary-attribution.json"
    value = json.loads(attribution_path.read_text(encoding="utf-8"))
    value["records"][0]["attribution_text"] = "tampered"
    attribution_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="attribution checksum"):
        PublishedMapillaryDemoIndex.open(result.publication.directory)


class _ClosableSyntheticExecutor(_SyntheticExecutor):
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        super().__init__(fail_on_call=fail_on_call)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_runner_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition, pilot_root = _acquisition(tmp_path)
    paths = Phase3B3RunPaths.for_test(
        project_root=REPOSITORY_ROOT,
        pilot_root=pilot_root,
    )
    audit = _audit()
    atomic_write_json(paths.coverage_audit, audit.model_dump(mode="json"))
    acquisition = acquisition.model_copy(
        update={"coverage_audit_sha256": sha256_path(paths.coverage_audit)}
    )
    atomic_write_json(paths.acquisition_manifest, acquisition.model_dump(mode="json"))
    checkpoint = AcquisitionCheckpoint(
        plan_sha256="b" * 64,
        status="completed",
        completed_image_ids=tuple(sorted(asset.mapillary_image_id for asset in acquisition.assets)),
        downloaded_bytes=acquisition.downloaded_bytes,
        request_count=acquisition.request_count,
        page_count=acquisition.page_count,
        updated_at=datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
    )
    atomic_write_json(paths.acquisition_checkpoint, checkpoint.model_dump(mode="json"))

    interrupted = _ClosableSyntheticExecutor(fail_on_call=1)
    with pytest.raises(RuntimeError, match="synthetic descriptor interruption"):
        execute_private_demo_run(paths, _executor_for_test=interrupted)
    assert interrupted.closed
    assert paths.locked_split.is_file()
    assert not paths.benchmark.exists()
    locked_bytes = paths.locked_split.read_bytes()

    completed = _ClosableSyntheticExecutor()
    receipt = execute_private_demo_run(paths, _executor_for_test=completed)
    assert completed.closed
    assert paths.locked_split.read_bytes() == locked_bytes
    assert paths.benchmark.is_file()
    assert paths.aggregate_receipt.is_file()
    assert receipt.value["model_artifact_sha256"] == MEGALOC_PHASE3B2_ARTIFACT_SHA256
    assert receipt.value["individual_asset_records_in_receipt"] is False
    assert receipt.value["network_used_by_runner"] is False
    assert receipt.value["index_validated"] is True
    assert receipt.value["attribution_validated"] is True
    assert receipt.value["manifest_sha256"] == sha256_path(paths.acquisition_manifest)
    safe_output = json.dumps(receipt.to_json(), sort_keys=True).casefold()
    assert "mapillary_image_id" not in safe_output
    assert "source_page_url" not in safe_output
    assert "latitude" not in safe_output
    assert "access_token" not in safe_output

    resumed = _ClosableSyntheticExecutor()
    resumed_receipt = execute_private_demo_run(paths, _executor_for_test=resumed)
    assert resumed.closed
    assert resumed_receipt.to_json() == receipt.to_json()

    plan = plan_private_demo_run(paths)
    expected_count = len(plan.split.references) + len(plan.split.holdout)
    metrics = PrivateDemoExecutionMetrics(
        device="cuda",
        image_count=expected_count,
        batch_call_count=2,
        elapsed_seconds=1.25,
        peak_vram_bytes=1_024,
    )
    _write_performance_receipt(paths, plan, metrics, expected_count)
    _validate_performance_receipt(paths, plan, expected_count)
    frozen = paths.performance_receipt.read_bytes()
    performance = json.loads(frozen)
    assert performance["descriptor_count"] == expected_count
    assert performance["network_used"] is False
    assert performance["measurement_descriptors_persisted"] is False
    _write_performance_receipt(paths, plan, metrics, expected_count)
    assert paths.performance_receipt.read_bytes() == frozen

    tampered_values: tuple[tuple[str, object], ...] = (
        ("technical_demo_only", False),
        ("network_used", True),
        ("measurement_descriptors_persisted", True),
        ("descriptor_dimension", 1),
        ("runtime_seconds", -1.0),
        ("device", "cpu"),
    )
    for key, bad_value in tampered_values:
        tampered = dict(performance)
        tampered[key] = bad_value
        base = {
            item_key: item for item_key, item in tampered.items() if item_key != "receipt_sha256"
        }
        tampered["receipt_sha256"] = sha256_json(base)
        paths.performance_receipt.write_bytes(canonical_json_bytes(tampered))
        with pytest.raises(
            Phase3B3RunError,
            match="PHASE3B3_PERFORMANCE_RECEIPT_INVALID",
        ):
            _validate_performance_receipt(paths, plan, expected_count)
    paths.performance_receipt.write_bytes(frozen)

    monkeypatch.setattr(
        mapillary_runner_module,
        "_is_link",
        lambda path: path == paths.performance_receipt,
    )
    with pytest.raises(Phase3B3RunError, match="PHASE3B3_REQUIRED_FILE_INVALID"):
        _validate_performance_receipt(paths, plan, expected_count)
