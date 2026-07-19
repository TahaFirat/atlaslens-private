from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from atlaslens_api.fusion import (
    CandidateBatch,
    CandidateDraft,
    DeterministicCandidateFusionService,
)
from atlaslens_api.schemas import (
    Candidate,
    Evidence,
    GeoClipClusterSummary,
    GeoJsonPoint,
    GeoPoint,
    Provenance,
)


def provenance(provider_id: str = "test-provider") -> Provenance:
    return Provenance(
        provider_id=provider_id,
        provider_kind="test",
        provider_version="1",
        execution_boundary="local",
        output_schema_version="test-v1",
    )


def evidence(evidence_id: str) -> Evidence:
    return Evidence(
        id=evidence_id,
        type="future_evidence_type",
        label="evidence.test",
        display_value=None,
        confidence=0.3,
        confidence_basis="test",
        source="test-provider",
        sensitive=False,
        provenance=provenance(),
    )


def draft(
    candidate_id: str,
    evidence_id: str,
    *,
    latitude: float = 10,
    longitude: float = 20,
    confidence: float = 0.3,
    verification: str = "unverified_model",
) -> CandidateDraft:
    metadata = verification == "metadata_only"
    return CandidateDraft(
        id=candidate_id,
        center=GeoPoint(latitude=latitude, longitude=longitude),
        radius_km=1 if metadata else 25,
        uncertainty_basis="test",
        confidence=0.95 if metadata else confidence,
        confidence_kind="source_reliability" if metadata else "uncalibrated_score",
        confidence_basis="test",
        granularity="exact_metadata" if metadata else "region",
        source="test-provider",
        evidence_ids=[evidence_id],
        evidence_summary=f"summary.{evidence_id}",
        provenance=[provenance(candidate_id)],
        verification_status=verification,
        verified=False,
    )


def test_candidate_validation_rejects_invalid_uncertainty() -> None:
    valid = {
        "id": "candidate",
        "rank": 1,
        "center": {"latitude": 0, "longitude": 0},
        "geometry": {"type": "Point", "coordinates": [0, 0]},
        "radius_km": -1,
        "uncertainty_basis": "test",
        "confidence": 0.3,
        "confidence_kind": "uncalibrated_score",
        "confidence_basis": "test",
        "granularity": "region",
        "source": "test",
        "evidence_ids": ["e1"],
        "evidence_summary": "test",
        "provenance": [provenance().model_dump()],
        "verification_status": "unverified_model",
        "verified": False,
    }
    with pytest.raises(ValidationError):
        Candidate.model_validate(valid)


def test_fusion_deduplicates_across_antimeridian_and_preserves_provenance() -> None:
    fusion = DeterministicCandidateFusionService()
    left = draft("a", "e1", longitude=179.999, confidence=0.4)
    right = draft("b", "e2", longitude=-179.999, confidence=0.2)
    result = fusion.fuse(
        [evidence("e1"), evidence("e2")],
        [CandidateBatch("one", (right,)), CandidateBatch("two", (left,))],
    )
    assert result.abstention is None
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.id == "a"
    assert candidate.confidence == 0.4
    assert candidate.evidence_ids == ["e1", "e2"]
    assert len(candidate.provenance) == 2


def test_fusion_is_deterministic_and_exif_ranks_first() -> None:
    fusion = DeterministicCandidateFusionService()
    metadata = draft("metadata", "exif", latitude=30, longitude=30, verification="metadata_only")
    vision = draft("vision", "visual", latitude=-20, longitude=-20, confidence=0.4)
    evidence_items = [evidence("exif"), evidence("visual")]
    one = fusion.fuse(
        evidence_items,
        [CandidateBatch("vision", (vision,)), CandidateBatch("metadata", (metadata,))],
    )
    two = fusion.fuse(
        deepcopy(evidence_items),
        [CandidateBatch("metadata", (metadata,)), CandidateBatch("vision", (vision,))],
    )
    assert [item.model_dump() for item in one.candidates] == [
        item.model_dump() for item in two.candidates
    ]
    assert one.candidates[0].verification_status == "metadata_only"
    assert [item.rank for item in one.candidates] == [1, 2]


def test_invalid_evidence_reference_is_rejected_and_fusion_abstains() -> None:
    result = DeterministicCandidateFusionService().fuse(
        [], [CandidateBatch("test", (draft("bad", "missing"),))]
    )
    assert result.candidates == []
    assert result.abstention is not None
    assert result.abstention.reason_code == "insufficient_geographic_evidence"


def test_fusion_keeps_nearby_but_distinct_geoclip_clusters() -> None:
    first = draft("cluster-a", "e1", latitude=40.0, longitude=29.0).model_copy(
        update={
            "geoclip_cluster": GeoClipClusterSummary(
                cluster_id="geoclip-cluster-a",
                member_count=1,
                member_ranks=[1],
                max_raw_similarity=0.9,
                mean_raw_similarity=0.9,
                raw_score_type="uncalibrated_gallery_softmax",
                cluster_support=0.9,
            )
        }
    )
    second = draft(
        "cluster-b",
        "e2",
        latitude=40.0001,
        longitude=29.0001,
    ).model_copy(
        update={
            "geoclip_cluster": GeoClipClusterSummary(
                cluster_id="geoclip-cluster-b",
                member_count=1,
                member_ranks=[2],
                max_raw_similarity=0.8,
                mean_raw_similarity=0.8,
                raw_score_type="uncalibrated_gallery_softmax",
                cluster_support=0.8,
            )
        }
    )

    result = DeterministicCandidateFusionService().fuse(
        [evidence("e1"), evidence("e2")],
        [CandidateBatch("phase6a", (first, second))],
    )

    assert result.abstention is None
    assert [item.id for item in result.candidates] == ["cluster-a", "cluster-b"]
    assert [
        item.geoclip_cluster.cluster_id for item in result.candidates if item.geoclip_cluster
    ] == [
        "geoclip-cluster-a",
        "geoclip-cluster-b",
    ]


def test_geojson_is_longitude_first() -> None:
    point = GeoJsonPoint(coordinates=(29.0, 41.0))
    assert point.coordinates == (29.0, 41.0)
