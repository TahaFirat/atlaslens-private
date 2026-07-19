from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from atlaslens_api.cases import (
    AdjudicationDecision,
    CalibrationState,
    CaseDomainError,
    CasePurpose,
    CaseSensitivity,
    CaseStatus,
    EvidenceType,
    HypothesisOrigin,
    MediaStorageState,
    sanitize_display_filename,
)
from atlaslens_api.cases.domain import (
    GENESIS_AUDIT_HASH,
    ActorType,
    canonical_audit_hash,
    canonical_json_bytes,
    safe_json_object,
    validate_case_values,
    validate_https_url,
    validate_wgs84,
)


def test_phase2_enum_values_match_the_persisted_contract() -> None:
    assert {item.value for item in CasePurpose} == {
        "journalism",
        "humanitarian",
        "disaster_response",
        "insurance",
        "authorized_security_research",
        "other",
    }
    assert {item.value for item in CaseStatus} == {
        "open",
        "under_review",
        "resolved",
        "archived",
    }
    assert {item.value for item in CaseSensitivity} == {
        "standard",
        "sensitive",
        "conflict_related",
    }
    assert {item.value for item in MediaStorageState} == {
        "ephemeral",
        "deleted_after_analysis",
        "unavailable",
        "externally_managed",
    }
    assert HypothesisOrigin.OPERATOR_CORRECTION.value == "operator_correction"
    assert CalibrationState.UNCALIBRATED.value == "uncalibrated"
    assert AdjudicationDecision.NEEDS_MORE_EVIDENCE.value == "needs_more_evidence"
    assert EvidenceType.RETRIEVAL_MATCH.value == "retrieval_match"


def test_case_requires_authorization_source_context_and_other_detail() -> None:
    common = {
        "title": "Synthetic investigation",
        "description": None,
        "purpose": CasePurpose.JOURNALISM,
        "purpose_detail": None,
        "source_context": "Synthetic public-interest test fixture",
        "authorization_attested": True,
        "created_by_actor_id": "synthetic-analyst",
        "retention_policy": "temporary",
        "workspace_id": "local-default",
    }
    values = validate_case_values(**common)
    assert values[0] == "Synthetic investigation"

    with pytest.raises(CaseDomainError, match="authorization"):
        validate_case_values(**{**common, "authorization_attested": False})
    with pytest.raises(CaseDomainError, match="source_context"):
        validate_case_values(**{**common, "source_context": " "})
    with pytest.raises(CaseDomainError, match="purpose_detail"):
        validate_case_values(**{**common, "purpose": CasePurpose.OTHER, "purpose_detail": None})


def test_filename_and_payload_guards_do_not_retain_private_shapes() -> None:
    assert sanitize_display_filename(r"C:\private\operator image?.jpg") == "operator image_.jpg"
    assert sanitize_display_filename("../../synthetic.png") == "synthetic.png"
    assert safe_json_object({"summary": "synthetic", "score": 0.25}) == {
        "summary": "synthetic",
        "score": 0.25,
    }
    for unsafe in (
        {"raw_ocr": "private text"},
        {"image_bytes": "base64"},
        {"api_key": "placeholder"},
        {"original_filename": "private.jpg"},
    ):
        with pytest.raises(CaseDomainError, match="sensitive payload key"):
            safe_json_object(unsafe)


def test_metadata_url_policy_preserves_safe_https_and_mapillary_attribution() -> None:
    ordinary = "https://example.invalid/public/source"
    mapillary = "https://www.mapillary.com/app/?pKey=test-only-reference"

    assert validate_https_url(ordinary, "source_url") == ordinary
    assert validate_https_url(mapillary, "source_url") == mapillary


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "https://example.invalid/public/source?view=attribution",
        "https://example.invalid/public/source?",
        "https://example.invalid/public/source#fragment",
        "https://www.mapillary.com/app/?pKey=",
        "https://www.mapillary.com/app/?pKey=first&pKey=second",
        "https://www.mapillary.com/app/?pKey=test-only&signature=test-only",
        "https://www.mapillary.com/app/?pKey=test-only&expires=1",
        "https://www.mapillary.com:443/app/?pKey=test-only",
        "https://www.mapillary.com:/app/?pKey=test-only",
        "https://mapillary.com/app/?pKey=test-only",
        "https://www.mapillary.com/other/?pKey=test-only",
        "https://operator:test-only@example.invalid/public/source",
    ],
)
def test_metadata_url_policy_rejects_nonpersistent_or_noncanonical_urls(
    unsafe_url: str,
) -> None:
    with pytest.raises(CaseDomainError) as invalid:
        validate_https_url(unsafe_url, "source_url")

    assert invalid.value.code == "invalid_source_url"


def test_metadata_url_policy_caps_internal_writes_at_response_limit() -> None:
    oversized = "https://example.invalid/" + ("a" * 777)
    assert len(oversized) == 801

    with pytest.raises(CaseDomainError) as invalid:
        validate_https_url(oversized, "archive_url")

    assert invalid.value.code == "archive_url_too_long"


@pytest.mark.parametrize(
    ("latitude", "longitude", "radius"),
    [(91.0, 0.0, 10.0), (0.0, -181.0, 10.0), (0.0, 0.0, 0.0)],
)
def test_wgs84_and_positive_uncertainty_are_required(
    latitude: float, longitude: float, radius: float
) -> None:
    with pytest.raises(CaseDomainError):
        validate_wgs84(latitude, longitude, radius)
    validate_wgs84(39.0, 35.0, 1.0)


def test_audit_hash_is_canonical_and_binds_order_and_payload() -> None:
    event_id = UUID(int=1)
    case_id = UUID(int=2)
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    first_payload = {"z": 1, "a": "synthetic"}
    second_payload = {"a": "synthetic", "z": 1}
    assert canonical_json_bytes(first_payload) == canonical_json_bytes(second_payload)

    original = canonical_audit_hash(
        event_id=event_id,
        case_id=case_id,
        sequence_number=1,
        event_type="case.created",
        actor_id="synthetic-analyst",
        actor_type=ActorType.OPERATOR,
        payload=first_payload,
        created_at=timestamp,
        previous_event_hash=GENESIS_AUDIT_HASH,
    )
    reordered_keys = canonical_audit_hash(
        event_id=event_id,
        case_id=case_id,
        sequence_number=1,
        event_type="case.created",
        actor_id="synthetic-analyst",
        actor_type=ActorType.OPERATOR,
        payload=second_payload,
        created_at=timestamp,
        previous_event_hash=GENESIS_AUDIT_HASH,
    )
    changed_order = canonical_audit_hash(
        event_id=event_id,
        case_id=case_id,
        sequence_number=2,
        event_type="case.created",
        actor_id="synthetic-analyst",
        actor_type=ActorType.OPERATOR,
        payload=second_payload,
        created_at=timestamp,
        previous_event_hash=GENESIS_AUDIT_HASH,
    )
    assert original == reordered_keys
    assert original != changed_order
