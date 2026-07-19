from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PHASE3_SCRIPTS = ROOT / "scripts" / "phase3"
sys.path.insert(0, str(PHASE3_SCRIPTS))

from validate_phase3_plan import (  # noqa: E402
    Phase3ValidationError,
    load_json_object,
    load_manifest_rows,
    validate_budget,
    validate_manifest_rows,
    validate_manifest_schema,
    validate_repository,
    validate_sampling,
    validate_source_policy,
)

SOURCE_PATH = ROOT / "config" / "corpus" / "source-policy-v1.json"
SCHEMA_PATH = ROOT / "config" / "corpus" / "manifest-schema-v1.json"
SAMPLING_PATH = ROOT / "config" / "corpus" / "turkiye-pilot-sampling-v1.json"
LEAKAGE_PATH = ROOT / "config" / "corpus" / "leakage-policy-v1.json"
BUDGET_PATH = ROOT / "config" / "cloud" / "runpod-phase3-budget-v1.json"


def _configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_json_object(SOURCE_PATH),
        load_json_object(SCHEMA_PATH),
        load_json_object(SAMPLING_PATH),
        load_json_object(LEAKAGE_PATH),
    )


def _row(
    asset_id: str = "asset-001",
    *,
    sha: str = "a" * 64,
    phash: str = "0123456789abcdef",
) -> dict[str, Any]:
    return {
        "corpus_version": "atlaslens-turkiye-corpus-test-v1",
        "asset_id": asset_id,
        "source_asset_id": f"source-{asset_id}",
        "source_name": "First-party AtlasLens-captured imagery",
        "source_url": f"atlaslens://capture/{asset_id}",
        "contributor_or_owner": "AtlasLens test owner",
        "capture_timestamp": "2026-01-01T10:00:00+00:00",
        "acquisition_timestamp": "2026-02-01T10:00:00+00:00",
        "latitude": 37.0,
        "longitude": 35.0,
        "coordinate_accuracy_m": 12.5,
        "heading_degrees": 90.0,
        "sequence_id": f"sequence-{asset_id}",
        "capture_run_id": f"run-{asset_id}",
        "image_sha256": sha,
        "perceptual_hash": phash,
        "perceptual_hash_algorithm": "dhash64-v1",
        "width_px": 1920,
        "height_px": 1080,
        "mime_type": "image/jpeg",
        "asset_type": "street_level",
        "province_code": "TR-01",
        "urbanicity": "rural_road",
        "road_class": "national_or_state_road",
        "scene_type": "rural_road",
        "road_context": "rural_road",
        "terrain_class": "flat",
        "vegetation_state": "sparse_or_arid",
        "season": "winter",
        "license_identifier": "ATLASLENS-FIRST-PARTY-V1",
        "license_url": "https://atlaslens.local/rights/first-party-v1",
        "attribution_text": "AtlasLens first-party capture",
        "source_policy_decision": "FIRST_PARTY_ONLY",
        "commercial_use_decision": "FIRST_PARTY_ONLY",
        "derivative_index_decision": "FIRST_PARTY_ONLY",
        "personal_data_blur_state": "BLURRED_BY_ATLASLENS",
        "deletion_revocation_state": {
            "status": "ACTIVE",
            "last_checked_at": "2026-07-15T10:00:00+00:00",
            "request_id": None,
        },
        "provenance_receipt": {
            "receipt_id": f"receipt-{asset_id}",
            "source_policy_version": "2026-07-16",
            "evidence_date": "2026-07-16",
            "acquisition_method": "first_party_capture",
            "receipt_sha256": "b" * 64,
        },
        "spatial_split": "development-west",
        "role": "development",
        "tier": "tier_1_national_recall",
        "acquisition_ready": True,
    }


def test_checked_in_phase3_configs_pass_offline_validation() -> None:
    result = validate_repository(ROOT)

    assert result == {
        "status": "passed",
        "schema_version": "atlaslens-phase3-validator-v1",
        "validated_config_count": 5,
        "manifest_row_count": 0,
        "network_calls": 0,
        "downloads": 0,
        "image_bytes_read": 0,
    }


def test_valid_first_party_manifest_row_passes() -> None:
    source, schema, _, leakage = _configs()

    validate_manifest_rows([_row()], source, schema, leakage)


def test_blocked_source_is_rejected_from_acquisition_ready_manifest() -> None:
    source, schema, _, leakage = _configs()
    row = _row()
    row.update(
        {
            "source_name": "Google Street View imagery",
            "source_policy_decision": "BLOCKED",
            "commercial_use_decision": "BLOCKED",
            "derivative_index_decision": "BLOCKED",
        }
    )

    with pytest.raises(Phase3ValidationError, match="manifest_blocked_or_unknown_source"):
        validate_manifest_rows([row], source, schema, leakage)


def test_required_attribution_is_rejected_when_empty_in_substance() -> None:
    source, schema, _, leakage = _configs()
    row = _row()
    row.update(
        {
            "source_name": "Wikimedia Commons",
            "source_policy_decision": "GO_WITH_ATTRIBUTION",
            "commercial_use_decision": "GO_WITH_ATTRIBUTION",
            "derivative_index_decision": "GO_WITH_ATTRIBUTION",
            "attribution_text": "N/A",
        }
    )

    with pytest.raises(Phase3ValidationError, match="manifest_attribution_required"):
        validate_manifest_rows([row], source, schema, leakage)


def test_metadata_license_cannot_be_used_as_imagery_permission() -> None:
    source, schema, _, leakage = _configs()
    row = _row()
    row.update(
        {
            "source_name": "OpenStreetMap metadata",
            "source_policy_decision": "GO_WITH_ATTRIBUTION",
            "commercial_use_decision": "GO_WITH_ATTRIBUTION",
            "derivative_index_decision": "GO_WITH_ATTRIBUTION",
        }
    )

    with pytest.raises(
        Phase3ValidationError,
        match="manifest_metadata_not_imagery",
    ):
        validate_manifest_rows([row], source, schema, leakage)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("latitude", 91.0, "manifest_coordinates"),
        ("longitude", -181.0, "manifest_coordinates"),
        ("coordinate_accuracy_m", 0, "manifest_coordinate_accuracy"),
        ("image_sha256", "not-a-hash", "manifest_sha256"),
    ],
)
def test_manifest_coordinate_hash_and_radius_guards(
    field: str,
    value: object,
    code: str,
) -> None:
    source, schema, _, leakage = _configs()
    row = _row()
    row[field] = value

    with pytest.raises(Phase3ValidationError, match=code):
        validate_manifest_rows([row], source, schema, leakage)


def test_duplicate_asset_id_and_content_are_rejected() -> None:
    source, schema, _, leakage = _configs()
    duplicate_id = _row()
    second = _row()

    with pytest.raises(Phase3ValidationError, match="manifest_asset_id_duplicate"):
        validate_manifest_rows([duplicate_id, second], source, schema, leakage)

    second = _row("asset-002")
    with pytest.raises(Phase3ValidationError, match="manifest_exact_hash_duplicate"):
        validate_manifest_rows([duplicate_id, second], source, schema, leakage)


def test_sequence_cross_split_is_rejected() -> None:
    source, schema, _, leakage = _configs()
    first = _row()
    second = _row("asset-002", sha="c" * 64, phash="fedcba9876543210")
    second["sequence_id"] = first["sequence_id"]
    second["spatial_split"] = "development-east"
    second["longitude"] = 36.0

    with pytest.raises(Phase3ValidationError, match="manifest_sequence_cross_split"):
        validate_manifest_rows([first, second], source, schema, leakage)


def test_holdout_spatial_conflict_and_dense_province_are_rejected() -> None:
    source, schema, _, leakage = _configs()
    reference = _row()
    holdout = _row("holdout-001", sha="c" * 64, phash="fedcba9876543210")
    holdout.update(
        {
            "role": "holdout",
            "tier": "tier_3_locked_holdout",
            "province_code": "TR-02",
            "spatial_split": "holdout-02",
            "longitude": 35.001,
        }
    )

    with pytest.raises(Phase3ValidationError, match="manifest_spatial_split_conflict"):
        validate_manifest_rows([reference, holdout], source, schema, leakage)

    holdout["longitude"] = 36.0
    holdout["province_code"] = "TR-38"
    with pytest.raises(Phase3ValidationError, match="manifest_dense_province_holdout"):
        validate_manifest_rows([reference, holdout], source, schema, leakage)


def test_revoked_asset_cannot_be_acquisition_ready() -> None:
    source, schema, _, leakage = _configs()
    row = _row()
    row["deletion_revocation_state"]["status"] = "REVOKED"

    with pytest.raises(Phase3ValidationError, match="manifest_deletion_not_active"):
        validate_manifest_rows([row], source, schema, leakage)


def test_budget_requires_approval_and_bounded_gpu_hours() -> None:
    sampling = load_json_object(SAMPLING_PATH)
    budget = load_json_object(BUDGET_PATH)

    no_approval = copy.deepcopy(budget)
    no_approval["approval_required"] = False
    with pytest.raises(Phase3ValidationError, match="budget_approval_required"):
        validate_budget(no_approval, sampling)

    negative = copy.deepcopy(budget)
    negative["maximum_gpu_hours"] = -1
    with pytest.raises(Phase3ValidationError, match="budget_gpu_hours_bounded"):
        validate_budget(negative, sampling)

    unbounded = copy.deepcopy(budget)
    unbounded["maximum_gpu_hours"] = float("inf")
    with pytest.raises(Phase3ValidationError, match="budget_maximum_hours"):
        validate_budget(unbounded, sampling)

    missing_backups = copy.deepcopy(budget)
    missing_backups["required_backup_outputs"].pop()
    with pytest.raises(Phase3ValidationError, match="budget_backup_outputs"):
        validate_budget(missing_backups, sampling)

    unsafe_network_storage = copy.deepcopy(budget)
    unsafe_network_storage["storage_security"][
        "application_layer_encryption_required"
    ] = False
    with pytest.raises(
        Phase3ValidationError,
        match="budget_storage_security_application_layer_encryption_required",
    ):
        validate_budget(unsafe_network_storage, sampling)


def test_sampling_requires_explicit_asset_types_and_split_storage_math() -> None:
    sampling = load_json_object(SAMPLING_PATH)
    option = sampling["options"][1]
    option["street_reference_count"] -= 1

    with pytest.raises(
        Phase3ValidationError, match="sampling_reference_asset_type_math"
    ):
        validate_sampling(sampling)

    sampling = load_json_object(SAMPLING_PATH)
    del sampling["options"][1]["storage_gb"]["faiss_parquet_index"]
    with pytest.raises(Phase3ValidationError, match="sampling_storage_value"):
        validate_sampling(sampling)


def test_manifest_rejects_unknown_tier2_strata() -> None:
    source, schema, _, leakage = _configs()
    row = _row()
    row["road_context"] = "invented_road_type"

    with pytest.raises(Phase3ValidationError, match="manifest_road_context"):
        validate_manifest_rows([row], source, schema, leakage)


def test_unlicensed_source_cannot_be_marked_acquisition_ready() -> None:
    policy = load_json_object(SOURCE_PATH)
    sample = next(item for item in policy["sources"] if item["source_id"] == "sample4geo_code")
    sample["current_decision"] = "GO"

    with pytest.raises(Phase3ValidationError, match="unlicensed_source_acquisition_ready"):
        validate_source_policy(policy)


def test_source_policy_requires_caching_and_component_rights_separation() -> None:
    policy = load_json_object(SOURCE_PATH)
    del policy["sources"][0]["caching_permission"]
    with pytest.raises(Phase3ValidationError, match="source_record_fields"):
        validate_source_policy(policy)

    policy = load_json_object(SOURCE_PATH)
    policy["component_rights_separation"]["megaloc"]["future_atlaslens_inputs"][
        "decision"
    ] = "GO"
    with pytest.raises(
        Phase3ValidationError, match="source_megaloc_rights_separation"
    ):
        validate_source_policy(policy)


def test_manifest_schema_cannot_drop_required_governance_field() -> None:
    schema = load_json_object(SCHEMA_PATH)
    schema["required"].remove("deletion_revocation_state")

    with pytest.raises(Phase3ValidationError, match="manifest_required_fields"):
        validate_manifest_schema(schema)


def test_manifest_loader_rejects_image_suffix_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_read(*args: object, **kwargs: object) -> str:
        raise AssertionError("unsupported manifest path was opened")

    monkeypatch.setattr(Path, "read_text", fail_if_read)

    with pytest.raises(Phase3ValidationError, match="manifest_suffix"):
        load_manifest_rows(Path("must-not-open.jpg"))


def test_cli_reports_zero_network_download_and_image_reads() -> None:
    result = subprocess.run(
        [sys.executable, str(PHASE3_SCRIPTS / "validate_phase3_plan.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["network_calls"] == 0
    assert payload["downloads"] == 0
    assert payload["image_bytes_read"] == 0
