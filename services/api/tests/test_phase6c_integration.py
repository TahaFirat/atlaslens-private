from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import cast

import pytest

from atlaslens_api.inference.models import (
    InferenceCandidate,
    InferenceProvenance,
    InferenceResult,
)
from atlaslens_api.phase6c.fusion import (
    OSVDirectRegressionEvidence,
    Phase6CEvidenceCandidate,
    Phase6CGeographicFusionEngine,
    PlonkOSVSampleEvidence,
)
from atlaslens_api.phase6c.hierarchical import (
    GeoCLIPHierarchicalSearchProvider,
    HierarchicalSearchCandidate,
    HierarchicalSearchResult,
)
from atlaslens_api.phase6c.integration import (
    Phase6CCacheVersions,
    Phase6CPipelineExtension,
    derive_turkiye_signal_groups,
)
from atlaslens_api.phase6c.megaloc import (
    MEGALOC_DESCRIPTOR_VERSION,
    MEGALOC_MODEL_ID,
    MEGALOC_MODEL_REVISION,
    MEGALOC_SOURCE_REVISION,
    MegaLocRetrievalCluster,
    MegaLocRetrievalProvider,
    MegaLocRetrievalResult,
)
from atlaslens_api.phase6c.reference_index import ReferenceRecord, ReferenceSearchHit
from atlaslens_api.providers.base import ProviderDescriptor
from atlaslens_api.schemas import Phase6CLeakageAuditSummary


class _Hierarchy:
    def __init__(self) -> None:
        self.signal_count: int | None = None

    async def search(
        self,
        _image_bytes: bytes,
        *,
        turkiye_signal_count: int,
        cancellation: asyncio.Event,
    ) -> HierarchicalSearchResult:
        assert not cancellation.is_set()
        self.signal_count = turkiye_signal_count
        return HierarchicalSearchResult(
            catalogue_version="catalogue-v1",
            global_grid_points_scored=256,
            catalogue_points_scored=501,
            turkiye_refinement_triggered=turkiye_signal_count >= 2,
            trigger_signal_count=turkiye_signal_count,
            candidates=(
                HierarchicalSearchCandidate(
                    candidate_id="hier-0123456789abcdefabcd",
                    latitude=39.0,
                    longitude=32.0,
                    search_level="catalogue",
                    grid_resolution_km=50.0,
                    provider_score=0.7,
                    provider_rank=1,
                    diversity_cluster="mode-01",
                    nearest_name="Ankara",
                    nearest_kind="place",
                    nearest_distance_km=2.5,
                ),
            ),
            duration_ms=5,
            cache_keys=("global-v1", "catalogue-v1"),
        )


class _MegaLoc:
    async def retrieve(
        self, _image_bytes: bytes, *, cancellation: asyncio.Event
    ) -> MegaLocRetrievalResult:
        assert not cancellation.is_set()
        reference = ReferenceRecord(
            reference_id="reference-1",
            source="manual",
            source_family="licensed-source",
            source_image_id="source-image-1",
            source_sequence_id="sequence-1",
            source_url="https://example.org/reference/1",
            latitude=39.01,
            longitude=32.01,
            coordinate_uncertainty_m=25.0,
            heading_degrees=None,
            captured_at=None,
            country="Türkiye",
            province="Ankara",
            city="Ankara",
            license="CC-BY-4.0",
            license_url="https://example.org/license",
            attribution="Example contributor",
            asset_key="licensed/reference-1.jpg",
            sha256="1" * 64,
            perceptual_hash="2" * 16,
            descriptor_version=MEGALOC_DESCRIPTOR_VERSION,
        )
        hit = ReferenceSearchHit(
            reference_id=reference.reference_id,
            rank=1,
            similarity=0.8,
            distance=0.2,
            uncertainty_radius_m=25.0,
            reference=reference,
        )
        cluster = MegaLocRetrievalCluster(
            cluster_id="megaloc-cluster-0123456789abcdef",
            latitude=39.01,
            longitude=32.01,
            uncertainty_radius_km=1.0,
            match_count=1,
            independent_sequence_support=1,
            best_similarity=0.8,
            member_reference_ids=(reference.reference_id,),
            member_ranks=(1,),
        )
        return MegaLocRetrievalResult(
            status="completed",
            model_id=MEGALOC_MODEL_ID,
            model_revision=MEGALOC_MODEL_REVISION,
            source_revision=MEGALOC_SOURCE_REVISION,
            index_version="index-v1",
            query_descriptor_version=MEGALOC_DESCRIPTOR_VERSION,
            device="cuda",
            duration_ms=7,
            matches=(hit,),
            clusters=(cluster,),
        )

    async def close(self) -> None:
        return None


def _geoclip() -> InferenceResult:
    return InferenceResult(
        provider_id="geoclip-global-v1",
        provider_revision="1.0.0",
        model_name="GeoCLIP",
        model_revision="1.2.0",
        runtime_revision="runtime-v1",
        classification="real",
        status="succeeded",
        device="cuda",
        dtype="float32",
        runtime_ms=4,
        score_semantics="gallery_softmax_not_calibrated",
        normalization_method="stable_rank",
        calibration_state="uncalibrated",
        candidates=(
            InferenceCandidate(
                rank=1,
                original_rank=1,
                latitude=39.0,
                longitude=32.0,
                raw_score=0.6,
                limitations=("uncalibrated",),
            ),
        ),
        provenance=InferenceProvenance(
            provider_id="geoclip-global-v1",
            provider_revision="1.0.0",
            model_revision="1.2.0",
            runtime_revision="runtime-v1",
            source_kind="verified_model",
        ),
    )


def _versions(
    *,
    index: str | None = "index-v1",
    leakage_attestation: str = "not_run",
    ocr_version: str = "ocr:1",
) -> Phase6CCacheVersions:
    return Phase6CCacheVersions(
        pipeline_version="phase6c-v1",
        provider_versions=("geoclip:1", "megaloc:1"),
        model_revisions=("geoclip-model:1", "megaloc-model:1"),
        fusion_config_version="phase6c-v1",
        fusion_config_sha256="a" * 64,
        reference_index_version=index,
        ocr_version=ocr_version,
        openai_prompt_version="prompt:1",
        reference_index_leakage_attestation=leakage_attestation,
    )


def _extension(
    hierarchy: _Hierarchy | None,
    megaloc: _MegaLoc | None,
    *,
    versions: Phase6CCacheVersions | None = None,
    leakage_audit: Phase6CLeakageAuditSummary | None = None,
) -> Phase6CPipelineExtension:
    return Phase6CPipelineExtension(
        hierarchical=cast(GeoCLIPHierarchicalSearchProvider | None, hierarchy),
        megaloc=cast(MegaLocRetrievalProvider | None, megaloc),
        fusion=Phase6CGeographicFusionEngine(),
        cache_versions=versions or _versions(),
        ocr_descriptor=ProviderDescriptor(
            id="ocr-local",
            kind="ocr",
            version="1",
            execution_boundary="local",
            criticality="optional",
            available=True,
        ),
        leakage_audit=leakage_audit,
    )


@pytest.mark.asyncio
async def test_phase6c_publishes_only_independent_group_agreement(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"private-query")
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    hierarchy = _Hierarchy()
    extension = _extension(hierarchy, _MegaLoc())

    result = await extension.run(
        image,
        image_sha256=image_sha256,
        geoclip=_geoclip(),
        phase6b=None,
        ocr_outcome=None,
        cancellation=asyncio.Event(),
    )

    assert result.replacement_eligible
    assert result.candidate_batch is not None
    assert all(item.confidence is None for item in result.candidate_batch.candidates)
    assert all(item.radius_km > 0 for item in result.candidate_batch.candidates)
    assert result.summary.publication_candidate_count >= 1
    assert {item.profile_id for item in result.summary.ablations} == {
        "geoclip_original_only",
        "geoclip_original_hierarchical",
        "geoclip_osv5m",
        "geoclip_plonk",
        "geoclip_megaloc",
        "all_local_without_ocr",
        "all_local_with_ocr",
        "all_local_with_g3",
        "optional_openai_assist",
        "phase6c_final",
    }
    assert next(
        item for item in result.summary.ablations if item.profile_id == "phase6c_final"
    ).candidate_count == len(result.summary.fusion_candidates)
    assert result.summary.hierarchical_candidates
    assert result.summary.megaloc_matches
    assert result.summary.hierarchical_candidates[0].diversity_cluster == "mode-01"
    assert result.summary.hierarchical_candidates[0].nearest_name == "Ankara"
    assert result.summary.megaloc_matches[0].province == "Ankara"
    assert result.summary.megaloc_matches[0].city == "Ankara"
    assert result.summary.fusion_candidates
    assert result.summary.cache_fingerprint.startswith("phase6c-v1:")
    assert result.summary.leakage_audit.status == "not_run"
    assert hierarchy.signal_count == 2
    assert any(
        item.provider_id == "g3" and item.status == "disabled"
        for item in result.summary.providers
    )


@pytest.mark.asyncio
async def test_attested_leakage_summary_is_persisted_without_private_truth(
    tmp_path: Path,
) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"private-query")
    audit = Phase6CLeakageAuditSummary(
        status="passed",
        audit_version="atlaslens-leakage-audit-v1",
        report_fingerprint="a" * 64,
        references_checked=1,
        references_excluded=0,
    )
    result = await _extension(None, None, leakage_audit=audit).run(
        image,
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        geoclip=_geoclip(),
        phase6b=None,
        ocr_outcome=None,
        cancellation=asyncio.Event(),
    )

    assert result.summary.leakage_audit == audit
    dumped = result.summary.leakage_audit.model_dump()
    assert "holdout_id" not in dumped
    assert "holdout_sha256" not in dumped


@pytest.mark.asyncio
async def test_same_family_recall_is_persisted_but_not_published(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"private-query")
    hierarchy = _Hierarchy()
    result = await _extension(hierarchy, None).run(
        image,
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        geoclip=_geoclip(),
        phase6b=None,
        ocr_outcome=None,
        cancellation=asyncio.Event(),
    )

    assert result.candidate_batch is None
    assert not result.replacement_eligible
    assert result.summary.publication_candidate_count == 0
    assert result.summary.fusion_candidates
    assert result.summary.hierarchical_candidates
    assert hierarchy.signal_count == 1


def test_cache_fingerprint_covers_image_and_reference_index() -> None:
    first = _versions(index="index-v1")
    second = _versions(index="index-v2")
    image_a = "a" * 64
    image_b = "b" * 64

    assert first.fingerprint(image_a) != first.fingerprint(image_b)
    assert first.fingerprint(image_a) != second.fingerprint(image_a)
    assert first.fingerprint(image_a) != _versions(
        leakage_attestation=f"passed:{'c' * 64}"
    ).fingerprint(image_a)
    assert first.fingerprint(image_a) != _versions(
        ocr_version=f"ocr:1:phase6c-v1:{'d' * 64}"
    ).fingerprint(image_a)


def test_turkiye_signal_count_deduplicates_correlated_sources() -> None:
    direct = OSVDirectRegressionEvidence(
        provider="osv5m",
        model_id="osv",
        model_revision="1",
        candidates=(
            Phase6CEvidenceCandidate(
                candidate_id="osv-candidate",
                latitude=39.0,
                longitude=32.0,
                provider_rank=1,
                uncertainty_radius_km=100.0,
                provenance="osv:1",
            ),
        ),
    )
    correlated = PlonkOSVSampleEvidence(
        provider="plonk",
        model_id="plonk-osv",
        model_revision="1",
        candidates=(
            Phase6CEvidenceCandidate(
                candidate_id="plonk-candidate",
                latitude=39.1,
                longitude=32.1,
                raw_value=0.5,
                provider_rank=1,
                uncertainty_radius_km=100.0,
                provenance="plonk:1",
            ),
        ),
    )

    assert derive_turkiye_signal_groups((direct, correlated)) == ("osv5m_training",)
