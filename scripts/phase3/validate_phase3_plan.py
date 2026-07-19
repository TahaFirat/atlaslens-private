#!/usr/bin/env python3
"""Validate Product Phase 3A planning controls without network or image access."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
DECISIONS = frozenset(
    {
        "GO",
        "GO_WITH_ATTRIBUTION",
        "FIRST_PARTY_ONLY",
        "WRITTEN_PERMISSION_REQUIRED",
        "LEGAL_REVIEW_REQUIRED",
        "RESEARCH_ONLY",
        "BLOCKED",
        "UNKNOWN",
    }
)
ACQUISITION_READY = frozenset({"GO", "GO_WITH_ATTRIBUTION", "FIRST_PARTY_ONLY"})
OPTION_IDS = ("minimal_proof", "recommended_mvp", "expansion")
PROVINCE_CODES = frozenset(f"TR-{index:02d}" for index in range(1, 82))
DENSE_PROVINCE_CODES = frozenset({"TR-06", "TR-38", "TR-58"})
REQUIRED_SOURCE_IDS = frozenset(
    {
        "atlaslens_first_party_imagery",
        "atlaslens_partner_imagery",
        "kartaview",
        "mapillary",
        "panoramax",
        "wikimedia_commons",
        "openstreetmap_metadata",
        "geonames_metadata",
        "copernicus_sentinel",
        "openaerialmap",
        "commercial_aerial_satellite",
        "google_street_view",
        "google_maps_earth_screenshots",
        "megaloc_code_model",
        "sample4geo_code",
        "sample4geo_pretrained_weights",
        "cvusa",
        "cvact",
        "vigor",
        "university_1652",
        "osv5m",
        "plonk_models_lineage",
        "turkiye_hgm_orthophoto",
    }
)
SOURCE_FIELDS = frozenset(
    {
        "source_id",
        "source_name",
        "official_url",
        "asset_type",
        "owner_controller",
        "code_license",
        "weight_license",
        "imagery_data_license",
        "commercial_use",
        "redistribution_rights",
        "derivative_work_rights",
        "embedding_index_rights",
        "caching_permission",
        "attribution_requirements",
        "share_alike_risk",
        "api_terms",
        "caching_indexing_permission",
        "deletion_revocation_requirements",
        "rate_limits",
        "personal_data_considerations",
        "current_decision",
        "evidence_date",
        "unresolved_question",
        "required_next_action",
        "primary_evidence_urls",
        "license_absence_evidence",
    }
)
MANIFEST_FIELDS = frozenset(
    {
        "corpus_version",
        "asset_id",
        "source_asset_id",
        "source_name",
        "source_url",
        "contributor_or_owner",
        "capture_timestamp",
        "acquisition_timestamp",
        "latitude",
        "longitude",
        "coordinate_accuracy_m",
        "heading_degrees",
        "sequence_id",
        "capture_run_id",
        "image_sha256",
        "perceptual_hash",
        "perceptual_hash_algorithm",
        "width_px",
        "height_px",
        "mime_type",
        "asset_type",
        "province_code",
        "urbanicity",
        "road_class",
        "scene_type",
        "road_context",
        "terrain_class",
        "vegetation_state",
        "season",
        "license_identifier",
        "license_url",
        "attribution_text",
        "source_policy_decision",
        "commercial_use_decision",
        "derivative_index_decision",
        "personal_data_blur_state",
        "deletion_revocation_state",
        "provenance_receipt",
        "spatial_split",
        "role",
        "tier",
        "acquisition_ready",
    }
)
MANIFEST_STRATUM_ENUMS = {
    "road_context": frozenset(
        {
            "divided_highway",
            "ordinary_road",
            "rural_road",
            "junction",
            "service_area",
            "not_applicable",
            "unknown",
        }
    ),
    "terrain_class": frozenset(
        {"mountainous", "flat", "rolling", "mixed", "unknown"}
    ),
    "vegetation_state": frozenset(
        {
            "leaf_on",
            "leaf_off_or_dormant",
            "cropland_active",
            "sparse_or_arid",
            "snow_cover",
            "mixed",
            "not_applicable",
            "unknown",
        }
    ),
    "season": frozenset({"winter", "spring", "summer", "autumn", "unknown"}),
}
LEAKAGE_CHECKS = frozenset(
    {
        "exact_sha256",
        "perceptual_near_duplicate",
        "crop_overlap",
        "rotation_mirror_equivalence",
        "sequence_adjacency",
        "same_capture_run",
        "same_contributor_or_source_asset",
        "descriptor_similarity",
        "spatial_buffer",
        "temporal_proximity",
        "satellite_patch_overlap",
        "operator_smoke_image_exclusion",
        "benchmark_truth_isolation",
    }
)
REQUIRED_BACKUP_OUTPUTS = frozenset(
    {
        "governed_source_archive",
        "normalized_derivative_version",
        "corpus_manifest",
        "provenance_receipts",
        "descriptor_index_version",
        "deletion_revocation_map",
        "cost_receipt",
        "validation_report",
        "checkpoint_completion_receipts",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PHASH_RE = re.compile(r"^[0-9a-f]{16,128}$")


class Phase3ValidationError(ValueError):
    """A stable, path-free Phase 3 planning validation failure."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Phase3ValidationError(code)


def _object(value: object, code: str) -> dict[str, Any]:
    _require(isinstance(value, dict), code)
    return cast(dict[str, Any], value)


def _array(value: object, code: str) -> list[Any]:
    _require(isinstance(value, list), code)
    return cast(list[Any], value)


def _text(value: object, code: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), code)
    return cast(str, value)


def _number(value: object, code: str) -> float:
    _require(isinstance(value, int | float) and not isinstance(value, bool), code)
    result = float(cast(int | float, value))
    _require(math.isfinite(result), code)
    return result


def _uri(value: object, code: str) -> str:
    result = _text(value, code)
    parsed = urlparse(result)
    _require(bool(parsed.scheme) and (bool(parsed.netloc) or bool(parsed.path)), code)
    return result


def _close(actual: float, expected: float, code: str, tolerance: float = 1e-8) -> None:
    _require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), code)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase3ValidationError("json_invalid_or_unreadable") from error
    return _object(raw, "json_root_not_object")


def validate_source_policy(policy: dict[str, Any]) -> None:
    _require(
        policy.get("schema_version") == "atlaslens-source-policy-v1", "source_schema"
    )
    _require(
        set(_array(policy.get("decision_values"), "source_decisions")) == DECISIONS,
        "source_decisions",
    )
    _require(
        set(_array(policy.get("acquisition_ready_decisions"), "source_ready_decisions"))
        == ACQUISITION_READY,
        "source_ready_decisions",
    )
    _text(policy.get("policy_version"), "source_policy_version")
    _text(policy.get("evidence_date"), "source_evidence_date")
    sources = _array(policy.get("sources"), "source_records")
    ids: set[str] = set()
    names: set[str] = set()
    for raw in sources:
        source = _object(raw, "source_record")
        _require(source.keys() >= SOURCE_FIELDS, "source_record_fields")
        source_id = _text(source.get("source_id"), "source_id")
        source_name = _text(source.get("source_name"), "source_name")
        _require(source_id not in ids and source_name not in names, "source_duplicate")
        ids.add(source_id)
        names.add(source_name)
        _uri(source.get("official_url"), "source_official_url")
        _text(source.get("caching_permission"), "source_caching_permission")
        decision = _text(source.get("current_decision"), "source_decision")
        _require(decision in DECISIONS, "source_decision")
        _text(source.get("evidence_date"), "source_record_evidence_date")
        evidence = _array(
            source.get("primary_evidence_urls"), "source_primary_evidence"
        )
        _require(bool(evidence), "source_primary_evidence")
        for url in evidence:
            _uri(url, "source_primary_evidence_url")
        license_values = [
            _text(source.get(field), "source_license_field")
            for field in ("code_license", "weight_license", "imagery_data_license")
        ]
        missing_license = any("NONE_FOUND" in item for item in license_values)
        absence = source.get("license_absence_evidence")
        if missing_license:
            absence_record = _object(absence, "source_license_absence_evidence")
            _require(
                decision not in ACQUISITION_READY, "unlicensed_source_acquisition_ready"
            )
            _require(
                bool(
                    _array(
                        absence_record.get("applies_to"),
                        "source_license_absence_applies",
                    )
                ),
                "source_license_absence_applies",
            )
            checked = _array(
                absence_record.get("checked_urls"), "source_license_absence_urls"
            )
            _require(bool(checked), "source_license_absence_urls")
            for url in checked:
                _uri(url, "source_license_absence_url")
            _text(absence_record.get("finding"), "source_license_absence_finding")
        else:
            _require(absence is None, "unexpected_license_absence_evidence")
    _require(ids == REQUIRED_SOURCE_IDS, "source_required_coverage")
    separation = _object(
        policy.get("component_rights_separation"), "source_component_separation"
    )
    megaloc = _object(separation.get("megaloc"), "source_megaloc_separation")
    megaloc_code = _object(megaloc.get("code"), "source_megaloc_code")
    megaloc_weights = _object(
        megaloc.get("released_weights"), "source_megaloc_weights"
    )
    megaloc_corpora = _object(
        megaloc.get("training_corpora"), "source_megaloc_corpora"
    )
    megaloc_inputs = _object(
        megaloc.get("future_atlaslens_inputs"), "source_megaloc_inputs"
    )
    _require(
        megaloc.get("aggregate_source_id") == "megaloc_code_model"
        and megaloc_code.get("decision") == "GO_WITH_ATTRIBUTION"
        and megaloc_weights.get("artifact_license_decision")
        == "GO_WITH_ATTRIBUTION"
        and megaloc_weights.get("commercial_deployment_decision")
        == "LEGAL_REVIEW_REQUIRED"
        and megaloc_corpora.get("decision") == "LEGAL_REVIEW_REQUIRED"
        and len(set(_array(megaloc_corpora.get("named_sources"), "source_megaloc_corpus_names")))
        >= 5
        and megaloc_inputs.get("decision") == "UNKNOWN",
        "source_megaloc_rights_separation",
    )
    sample4geo = _object(
        separation.get("sample4geo"), "source_sample4geo_separation"
    )
    sample_code = _object(sample4geo.get("code"), "source_sample4geo_code")
    sample_weights = _object(
        sample4geo.get("released_weights"), "source_sample4geo_weights"
    )
    sample_datasets = {
        _text(
            _object(item, "source_sample4geo_dataset").get("source_id"),
            "source_sample4geo_dataset_id",
        ): _text(
            _object(item, "source_sample4geo_dataset").get("decision"),
            "source_sample4geo_dataset_decision",
        )
        for item in _array(
            sample4geo.get("referenced_datasets"), "source_sample4geo_datasets"
        )
    }
    _require(
        sample_code
        == {"source_id": "sample4geo_code", "decision": "BLOCKED"}
        and sample_weights
        == {"source_id": "sample4geo_pretrained_weights", "decision": "BLOCKED"}
        and sample_datasets
        == {
            "cvusa": "BLOCKED",
            "cvact": "BLOCKED",
            "vigor": "BLOCKED",
            "university_1652": "RESEARCH_ONLY",
        },
        "source_sample4geo_rights_separation",
    )
    by_id = {
        _text(_object(item, "source_record").get("source_id"), "source_id"): _object(
            item, "source_record"
        )
        for item in sources
    }
    for source_id in (
        "google_street_view",
        "google_maps_earth_screenshots",
        "sample4geo_code",
        "sample4geo_pretrained_weights",
    ):
        _require(
            by_id[source_id].get("current_decision") == "BLOCKED",
            "mandatory_blocked_source",
        )
    _require(
        by_id["megaloc_code_model"].get("current_decision") != "GO",
        "megaloc_rights_not_separate",
    )


def validate_manifest_schema(schema: dict[str, Any]) -> None:
    _require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        "manifest_draft",
    )
    _require(schema.get("type") == "object", "manifest_schema_type")
    _require(
        schema.get("additionalProperties") is False, "manifest_additional_properties"
    )
    required = set(_array(schema.get("required"), "manifest_required"))
    properties = _object(schema.get("properties"), "manifest_properties")
    _require(required >= MANIFEST_FIELDS, "manifest_required_fields")
    _require(required <= properties.keys(), "manifest_required_properties")
    decisions = _object(
        _object(schema.get("$defs"), "manifest_defs").get("sourceDecision"),
        "manifest_decision_def",
    )
    _require(
        set(_array(decisions.get("enum"), "manifest_decisions")) == DECISIONS,
        "manifest_decisions",
    )
    for field, expected in MANIFEST_STRATUM_ENUMS.items():
        definition = _object(properties.get(field), f"manifest_{field}_definition")
        _require(
            definition.get("type") == "string"
            and set(_array(definition.get("enum"), f"manifest_{field}_enum"))
            == expected,
            f"manifest_{field}_enum",
        )


def validate_sampling(sampling: dict[str, Any]) -> None:
    _require(
        sampling.get("schema_version") == "atlaslens-turkiye-pilot-sampling-v1",
        "sampling_schema",
    )
    _require(
        sampling.get("plan_version") == "turkiye-pilot-2026-07-16-v1",
        "sampling_plan_version",
    )
    _require(sampling.get("status") == "planning_only", "sampling_status")
    country = _object(sampling.get("country"), "sampling_country")
    _require(country.get("iso_3166_1_alpha_2") == "TR", "sampling_country")
    _require(country.get("province_count") == 81, "sampling_province_count")
    admission = _object(sampling.get("source_admission"), "sampling_source_admission")
    _require(
        set(_array(admission.get("allowed_decisions"), "sampling_allowed"))
        == ACQUISITION_READY,
        "sampling_allowed",
    )
    _require(admission.get("legal_clearance_claimed") is False, "sampling_legal_claim")
    assumptions = _object(sampling.get("assumptions"), "sampling_assumptions")
    _require(
        assumptions.get("megaloc_descriptor_values_per_reference") == 8448,
        "sampling_descriptor_values",
    )
    _require(
        assumptions.get("megaloc_descriptor_dtype") == "float32",
        "sampling_descriptor_dtype",
    )
    _require(
        assumptions.get("megaloc_descriptor_bytes_per_reference") == 33792,
        "sampling_descriptor_bytes",
    )

    tiers = _object(sampling.get("tiers"), "sampling_tiers")
    tier1 = _object(tiers.get("tier_1_national_recall"), "sampling_tier1")
    provinces = _array(tier1.get("provinces"), "sampling_provinces")
    province_codes: set[str] = set()
    province_names: set[str] = set()
    for raw in provinces:
        province = _object(raw, "sampling_province")
        code = _text(province.get("province_code"), "sampling_province_code")
        name = _text(province.get("name"), "sampling_province_name")
        _require(
            code not in province_codes and name not in province_names,
            "sampling_province_duplicate",
        )
        province_codes.add(code)
        province_names.add(name)
        minima = _object(
            province.get("minimum_references_by_option"), "sampling_province_minima"
        )
        for option_id in OPTION_IDS:
            _require(
                _number(minima.get(option_id), "sampling_province_minimum") > 0,
                "sampling_province_minimum",
            )
    _require(
        province_codes == PROVINCE_CODES and len(province_names) == 81,
        "sampling_all_provinces",
    )

    tier2 = _object(tiers.get("tier_2_dense_corridor"), "sampling_tier2")
    targets = _array(tier2.get("targets"), "sampling_tier2_targets")
    target_ids = {
        _text(
            _object(item, "sampling_tier2_target").get("target_id"),
            "sampling_target_id",
        )
        for item in targets
    }
    _require(
        target_ids
        == {
            "kayseri_city",
            "ankara_city",
            "sivas_city",
            "kayseri_ankara_corridor",
            "kayseri_sivas_corridor",
        },
        "sampling_dense_targets",
    )
    required_strata = _object(
        tier2.get("required_strata"), "sampling_tier2_required_strata"
    )
    road_stratum = _object(
        required_strata.get("road_context"), "sampling_tier2_road_context"
    )
    terrain_stratum = _object(
        required_strata.get("terrain_class"), "sampling_tier2_terrain"
    )
    _require(
        set(_array(road_stratum.get("required_values"), "sampling_tier2_roads"))
        == {
            "divided_highway",
            "ordinary_road",
            "rural_road",
            "junction",
            "service_area",
        },
        "sampling_tier2_roads",
    )
    _require(
        set(
            _array(
                terrain_stratum.get("required_values"), "sampling_tier2_terrains"
            )
        )
        == {"mountainous", "flat"},
        "sampling_tier2_terrains",
    )
    for stratum_name in ("vegetation_state", "season"):
        stratum = _object(
            required_strata.get(stratum_name), f"sampling_tier2_{stratum_name}"
        )
        _require(
            len(
                set(
                    _array(
                        stratum.get("required_values_where_observable"),
                        f"sampling_tier2_{stratum_name}_values",
                    )
                )
            )
            >= 4,
            f"sampling_tier2_{stratum_name}_values",
        )

    tier3 = _object(tiers.get("tier_3_locked_holdout"), "sampling_tier3")
    holdout_counts = _object(
        tier3.get("query_count_by_option"), "sampling_holdout_counts"
    )
    _require(
        set(
            _array(
                tier3.get("excluded_dense_province_codes"),
                "sampling_holdout_exclusions",
            )
        )
        == DENSE_PROVINCE_CODES,
        "sampling_holdout_exclusions",
    )
    _require(
        _number(tier3.get("dense_corridor_share"), "sampling_dense_share") == 0,
        "sampling_dense_share",
    )
    _require(
        tier3.get("model_selection_prohibited") is True,
        "sampling_holdout_model_selection",
    )
    _require(
        _number(
            tier3.get("minimum_publishable_query_count"), "sampling_publishable_count"
        )
        >= 100,
        "sampling_publishable_count",
    )
    _require(
        tier3.get("maximum_queries_per_sequence") == 1
        and tier3.get("maximum_queries_per_capture_run") == 1
        and tier3.get("reference_source_name_overlap_allowed") is False,
        "sampling_holdout_independence",
    )
    difficulty = _object(
        tier3.get("difficulty_minimum_shares"), "sampling_holdout_difficulty"
    )
    difficulty_values = [
        _number(value, "sampling_holdout_difficulty_share")
        for value in difficulty.values()
    ]
    _require(
        set(difficulty) == {"easy", "medium", "hard"}
        and all(0 < value < 1 for value in difficulty_values)
        and sum(difficulty_values) <= 1,
        "sampling_holdout_difficulty",
    )

    options = _array(sampling.get("options"), "sampling_options")
    by_option: dict[str, dict[str, Any]] = {}
    for raw in options:
        option = _object(raw, "sampling_option")
        option_id = _text(option.get("option_id"), "sampling_option_id")
        _require(option_id not in by_option, "sampling_option_duplicate")
        by_option[option_id] = option
    _require(set(by_option) == set(OPTION_IDS), "sampling_option_coverage")

    image_bytes = _number(
        assumptions.get("compressed_image_bytes_per_governed_asset"),
        "sampling_image_bytes",
    )
    derivative_bytes = _number(
        assumptions.get("normalized_derivative_bytes_per_governed_asset"),
        "sampling_derivative_bytes",
    )
    descriptor_bytes = _number(
        assumptions.get("megaloc_descriptor_bytes_per_reference"),
        "sampling_descriptor_bytes",
    )
    metadata_bytes = _number(
        assumptions.get("metadata_bytes_per_governed_asset"), "sampling_metadata_bytes"
    )
    ann_fraction = _number(
        assumptions.get("faiss_parquet_index_overhead_fraction_of_descriptor_bytes"),
        "sampling_ann_fraction",
    )
    staging_fraction = _number(
        assumptions.get("staging_space_fraction_of_primary_footprint"),
        "sampling_staging",
    )
    rollback_fraction = _number(
        assumptions.get("rollback_version_space_fraction_of_primary_footprint"),
        "sampling_rollback",
    )
    local_copies = _number(
        assumptions.get("local_descriptor_index_and_metadata_copies"),
        "sampling_local_copies",
    )
    _require(
        ann_fraction > 0
        and staging_fraction > 0
        and rollback_fraction > 0
        and local_copies >= 1,
        "sampling_storage_assumptions",
    )
    report_allowances = _object(
        assumptions.get("local_report_allowance_gb_by_option"),
        "sampling_report_allowances",
    )
    throughput = _number(
        assumptions.get("descriptor_generation_references_per_gpu_hour"),
        "sampling_throughput",
    )
    retry = _number(
        assumptions.get("descriptor_generation_retry_factor"), "sampling_retry"
    )
    eval_configs = _number(
        assumptions.get("benchmark_configurations_per_holdout"), "sampling_eval_configs"
    )
    eval_throughput = _number(
        assumptions.get("evaluation_query_configuration_runs_per_gpu_hour"),
        "sampling_eval_throughput",
    )

    tier1_minima = _object(
        tier1.get("minimum_references_per_province_by_option"), "sampling_tier1_minima"
    )
    for option_id, option in by_option.items():
        references = int(
            _number(option.get("reference_image_count"), "sampling_reference_count")
        )
        holdout = int(
            _number(option.get("holdout_query_count"), "sampling_holdout_count")
        )
        governed = int(
            _number(option.get("governed_asset_count"), "sampling_governed_count")
        )
        tier1_count = int(
            _number(option.get("tier_1_reference_count"), "sampling_tier1_count")
        )
        tier2_count = int(
            _number(option.get("tier_2_reference_count"), "sampling_tier2_count")
        )
        street_count = int(
            _number(option.get("street_reference_count"), "sampling_street_count")
        )
        aerial_count = int(
            _number(
                option.get("aerial_satellite_patch_count"), "sampling_aerial_count"
            )
        )
        _require(governed == references + holdout, "sampling_governed_math")
        _require(tier1_count + tier2_count == references, "sampling_reference_math")
        _require(
            street_count + aerial_count == references
            and street_count == references
            and aerial_count == 0,
            "sampling_reference_asset_type_math",
        )
        _require(
            holdout
            == int(_number(holdout_counts.get(option_id), "sampling_holdout_config")),
            "sampling_holdout_math",
        )
        _require(holdout >= 100, "sampling_holdout_minimum")
        _require(
            tier1_count
            == 81 * int(_number(tier1_minima.get(option_id), "sampling_tier1_minimum")),
            "sampling_tier1_math",
        )
        target_sum = sum(
            int(
                _number(
                    _object(
                        _object(item, "sampling_target").get(
                            "reference_targets_by_option"
                        ),
                        "sampling_target_counts",
                    ).get(option_id),
                    "sampling_target_count",
                )
            )
            for item in targets
        )
        _require(target_sum == tier2_count, "sampling_tier2_math")
        sequence_counts = _object(
            tier3.get("minimum_distinct_sequences_by_option"),
            "sampling_holdout_sequences",
        )
        run_counts = _object(
            tier3.get("minimum_distinct_capture_runs_by_option"),
            "sampling_holdout_runs",
        )
        _require(
            int(
                _number(
                    sequence_counts.get(option_id),
                    "sampling_holdout_sequence_count",
                )
            )
            == holdout
            and int(
                _number(run_counts.get(option_id), "sampling_holdout_run_count")
            )
            == holdout,
            "sampling_holdout_independent_count",
        )

        storage = _object(option.get("storage_gb"), "sampling_storage")
        compressed = governed * image_bytes / 1_000_000_000
        normalized = governed * derivative_bytes / 1_000_000_000
        descriptors = references * descriptor_bytes / 1_000_000_000
        faiss_parquet_index = descriptors * ann_fraction
        metadata = governed * metadata_bytes / 1_000_000_000
        primary = compressed + normalized + descriptors + faiss_parquet_index + metadata
        staging = primary * staging_fraction
        rollback = primary * rollback_fraction
        total = primary + staging + rollback
        local_output = local_copies * (
            descriptors + faiss_parquet_index + metadata
        ) + _number(
            report_allowances.get(option_id), "sampling_report_allowance"
        )
        for key, expected in (
            ("compressed_images", compressed),
            ("normalized_derivatives", normalized),
            ("megaloc_descriptors", descriptors),
            ("faiss_parquet_index", faiss_parquet_index),
            ("metadata", metadata),
            ("primary_footprint", primary),
            ("staging_space", staging),
            ("rollback_version_space", rollback),
            ("total_cloud_requirement", total),
            ("estimated_local_output", local_output),
        ):
            _close(
                _number(storage.get(key), "sampling_storage_value"),
                expected,
                f"sampling_storage_{key}",
            )

        compute = _object(option.get("compute_hours"), "sampling_compute")
        descriptor_hours = math.ceil((references / throughput * retry) * 4) / 4
        evaluation_hours = holdout * eval_configs / eval_throughput
        _close(
            _number(
                compute.get("descriptor_generation_gpu"), "sampling_descriptor_hours"
            ),
            descriptor_hours,
            "sampling_descriptor_hours",
        )
        _close(
            _number(compute.get("evaluation_gpu"), "sampling_evaluation_hours"),
            evaluation_hours,
            "sampling_evaluation_hours",
        )
        _close(
            _number(compute.get("total_gpu"), "sampling_total_hours"),
            descriptor_hours + evaluation_hours,
            "sampling_total_hours",
        )

    gates = _object(sampling.get("continuation_gates"), "sampling_gates")
    benchmark = _object(gates.get("benchmark"), "sampling_benchmark_gates")
    _require(
        _number(benchmark.get("minimum_holdout_queries"), "sampling_gate_holdout")
        >= 100,
        "sampling_gate_holdout",
    )
    _require(
        "lower bound above zero"
        in _text(benchmark.get("baseline_rule"), "sampling_baseline_gate"),
        "sampling_baseline_gate",
    )


def validate_leakage_policy(policy: dict[str, Any]) -> None:
    _require(
        policy.get("schema_version") == "atlaslens-corpus-leakage-policy-v1",
        "leakage_schema",
    )
    _require(
        policy.get("policy_version") == "turkiye-pilot-leakage-2026-07-16-v1",
        "leakage_policy_version",
    )
    _require(policy.get("status") == "planning_only", "leakage_status")
    checks = _object(policy.get("checks"), "leakage_checks")
    _require(set(checks) == LEAKAGE_CHECKS, "leakage_check_coverage")
    for name, raw in checks.items():
        check = _object(raw, f"leakage_check_{name}")
        _require(check.get("enabled") is True, f"leakage_check_disabled_{name}")
        _text(check.get("action"), f"leakage_action_{name}")
    operator = _object(
        checks.get("operator_smoke_image_exclusion"), "leakage_operator_check"
    )
    _require(operator.get("known_operator_image_count") == 2, "leakage_operator_count")
    _require(
        operator.get("image_bytes_or_hashes_read_in_phase_3a") is False,
        "leakage_operator_read",
    )
    phase = _object(policy.get("phase_3a_execution"), "leakage_phase3a")
    for key in (
        "network_calls",
        "downloads",
        "reads_image_bytes",
        "generates_descriptors",
        "uses_operator_images",
    ):
        _require(phase.get(key) is False, f"leakage_phase3a_{key}")
    thresholds = _object(policy.get("thresholds"), "leakage_thresholds")
    for key in (
        "dhash64_hamming_distance_maximum",
        "phash_hamming_distance_maximum",
        "descriptor_cosine_auto_exclude_minimum",
        "descriptor_cosine_manual_review_minimum",
        "query_to_query_holdout_buffer_m",
        "same_view_reference_to_holdout_buffer_m",
        "temporal_proximity_days",
        "sequence_adjacent_frame_window",
    ):
        _require(
            _number(thresholds.get(key), f"leakage_threshold_{key}") > 0,
            f"leakage_threshold_{key}",
        )
    _require(
        _number(
            thresholds.get("satellite_footprint_overlap_fraction_maximum"),
            "leakage_satellite_overlap",
        )
        == 0,
        "leakage_satellite_overlap",
    )


def validate_budget(budget: dict[str, Any], sampling: dict[str, Any]) -> None:
    _require(budget.get("version") == "runpod-phase3-budget-v1", "budget_version")
    _require(
        budget.get("phase") == "3A" and budget.get("enabled") is False, "budget_phase"
    )
    _require(budget.get("approval_required") is True, "budget_approval_required")
    _require(budget.get("currency") == "USD", "budget_currency")
    pricing_date = _text(budget.get("pricing_snapshot_date"), "budget_pricing_date")
    _require(
        bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", pricing_date)),
        "budget_pricing_date",
    )
    _require(budget.get("platform") == "runpod_secure_cloud_pod", "budget_platform")
    _require(budget.get("purchase_model") == "on_demand", "budget_purchase_model")
    fallback = _object(budget.get("fallback_platform"), "budget_fallback")
    _require(fallback.get("platform") == "local_rtx_4060", "budget_fallback")
    _require(_number(fallback.get("vram_gb"), "budget_fallback_vram") == 8, "budget_fallback_vram")
    _require(
        set(_array(fallback.get("allowed_scenarios"), "budget_fallback_scenarios"))
        == {"minimal_proof"},
        "budget_fallback_scenarios",
    )
    _require(
        _number(fallback.get("cloud_compute_cost_usd"), "budget_fallback_cost") == 0,
        "budget_fallback_cost",
    )
    rate_record = _object(
        budget.get("gpu_hourly_rate_assumption"), "budget_rate_record"
    )
    rate = _number(rate_record.get("usd_per_gpu_hour"), "budget_gpu_rate")
    _require(
        rate > 0 and _number(rate_record.get("vram_gb"), "budget_vram") >= 8,
        "budget_gpu_shape",
    )
    _uri(rate_record.get("source_url"), "budget_rate_source")
    storage_rate = _number(budget.get("storage_monthly_rate"), "budget_storage_rate")
    _require(storage_rate >= 0, "budget_storage_rate")
    storage_security = _object(
        budget.get("storage_security"), "budget_storage_security"
    )
    _require(
        storage_security.get("selected_persistent_storage") == "network_volume",
        "budget_storage_security",
    )
    provider_controls = _object(
        storage_security.get("provider_encryption_controls"),
        "budget_storage_provider_controls",
    )
    _require(
        provider_controls.get("volume_disk_optional_toggle_exposed") is True
        and provider_controls.get("volume_disk_selected") is False
        and provider_controls.get("container_disk_equivalent_toggle_exposed") is False
        and provider_controls.get("network_volume_equivalent_toggle_exposed") is False,
        "budget_storage_provider_controls",
    )
    _uri(provider_controls.get("source_url"), "budget_storage_security_url")
    _uri(storage_security.get("dpa_source_url"), "budget_storage_dpa_url")
    _uri(
        storage_security.get("terms_shared_responsibility_source_url"),
        "budget_storage_terms_url",
    )
    for key in (
        "dpa_broader_at_rest_representation_recorded",
        "application_layer_encryption_required",
        "plaintext_persistent_storage_prohibited",
        "customer_managed_application_keys_kept_outside_runpod",
        "professional_review_required",
        "launch_blocked_until_review_complete",
    ):
        _require(storage_security.get(key) is True, f"budget_storage_security_{key}")
    expected = _number(budget.get("expected_gpu_hours"), "budget_expected_hours")
    maximum = _number(budget.get("maximum_gpu_hours"), "budget_maximum_hours")
    soft = _number(budget.get("soft_budget"), "budget_soft")
    hard = _number(budget.get("hard_budget"), "budget_hard")
    _require(0 < expected <= maximum and maximum < 10_000, "budget_gpu_hours_bounded")
    _require(0 < soft <= hard and math.isfinite(hard), "budget_caps")
    stop = _object(budget.get("automatic_stop_deadline"), "budget_stop")
    _require(
        stop.get("absolute_utc_required_before_launch") is True, "budget_absolute_stop"
    )
    _require(
        0 < _number(stop.get("idle_shutdown_minutes"), "budget_idle") <= 60,
        "budget_idle",
    )
    _require(
        _number(stop.get("pod_terminate_after_hours"), "budget_terminate_hours")
        <= maximum,
        "budget_terminate_hours",
    )
    alerts = _array(budget.get("alert_thresholds"), "budget_alerts")
    alert_metrics: set[str] = set()
    for raw_alert in alerts:
        alert = _object(raw_alert, "budget_alert")
        metric = _text(alert.get("metric"), "budget_alert_metric")
        _require(metric not in alert_metrics, "budget_alert_duplicate")
        alert_metrics.add(metric)
        values = [
            _number(value, "budget_alert_value")
            for value in _array(alert.get("values"), "budget_alert_values")
        ]
        _require(
            bool(values)
            and values == sorted(set(values))
            and all(0 < value < 1 for value in values),
            "budget_alert_values",
        )
        _text(alert.get("action"), "budget_alert_action")
    _require(
        alert_metrics
        == {"fraction_of_hard_budget", "fraction_of_maximum_gpu_hours"},
        "budget_alert_coverage",
    )
    watchdog = _object(budget.get("watchdog_thresholds"), "budget_watchdog")
    _require(
        set(_array(watchdog.get("caps"), "budget_watchdog_caps"))
        == {"hard_budget", "maximum_gpu_hours"},
        "budget_watchdog_caps",
    )
    stop_fraction = _number(
        watchdog.get("stop_new_work_fraction"), "budget_watchdog_stop"
    )
    terminate_fraction = _number(
        watchdog.get("terminate_fraction"), "budget_watchdog_terminate"
    )
    _require(
        0 < stop_fraction < terminate_fraction == 1
        and watchdog.get("external_lifecycle_watchdog_required") is True,
        "budget_watchdog",
    )
    _require(budget.get("required_output_backup") is True, "budget_backup")
    _require(
        set(_array(budget.get("required_backup_outputs"), "budget_backup_outputs"))
        == REQUIRED_BACKUP_OUTPUTS,
        "budget_backup_outputs",
    )
    backup_policy = _object(budget.get("backup_policy"), "budget_backup_policy")
    _require(
        backup_policy.get("runpod_is_system_of_record") is False
        and backup_policy.get("runpod_volume_counts_as_backup_copy") is False
        and backup_policy.get("independent_corpus_system_of_record_required_before_upload")
        is True
        and _number(
            backup_policy.get("minimum_verified_copies_before_volume_deletion"),
            "budget_backup_copies",
        )
        >= 2
        and backup_policy.get("hash_algorithm") == "sha256"
        and backup_policy.get("repository_destination_prohibited") is True
        and backup_policy.get("d_drive_destination_prohibited") is True,
        "budget_backup_policy",
    )
    runtime_paths = _object(budget.get("runtime_paths"), "budget_runtime_paths")
    for key in (
        "network_volume_mount",
        "corpus",
        "manifests",
        "model_cache",
        "checkpoints",
        "descriptor_outputs",
        "revocation",
        "ephemeral_plaintext_shard",
    ):
        _require(
            _text(runtime_paths.get(key), f"budget_runtime_path_{key}").startswith("/"),
            f"budget_runtime_path_{key}",
        )
    transfer = _object(budget.get("transfer_policy"), "budget_transfer")
    _text(transfer.get("default_mode"), "budget_transfer_mode")
    _require(
        transfer.get("direct_source_to_volume_default") is False
        and set(
            _array(
                transfer.get("direct_source_to_volume_allowed_decisions"),
                "budget_transfer_decisions",
            )
        )
        <= ACQUISITION_READY
        and transfer.get("official_bulk_or_api_terms_required") is True
        and transfer.get("provenance_receipt_required") is True
        and transfer.get("long_lived_source_token_on_pod_prohibited") is True
        and transfer.get("scraping_prohibited") is True,
        "budget_transfer",
    )
    recovery = _object(budget.get("interruption_recovery"), "budget_recovery")
    _require(
        0
        < _number(
            recovery.get("checkpoint_max_references"),
            "budget_checkpoint_references",
        )
        <= 10_000
        and 0
        < _number(
            recovery.get("checkpoint_max_minutes"), "budget_checkpoint_minutes"
        )
        <= 60
        and recovery.get("new_on_demand_pod_same_datacenter") is True
        and recovery.get("reuse_existing_network_volume") is True
        and recovery.get("community_cloud_failover") is False,
        "budget_recovery",
    )
    teardown = _object(
        budget.get("shutdown_teardown_policy"), "budget_teardown"
    )
    teardown_steps = set(
        _array(teardown.get("normal_and_abort_sequence"), "budget_teardown_steps")
    )
    _require(
        teardown.get("idle_minutes") == stop.get("idle_shutdown_minutes")
        and teardown.get("pod_final_state") == "terminate"
        and {
            "sync_required_backup_outputs",
            "verify_file_counts_and_sha256",
            "terminate_pod",
            "delete_network_volume_and_record_receipt",
        }
        <= teardown_steps
        and teardown.get("second_verified_backup_before_volume_deletion") is True
        and teardown.get("deletion_receipt_required") is True,
        "budget_teardown",
    )
    limits = _object(budget.get("resource_limits"), "budget_resource_limits")
    _require(
        limits.get("gpu_count") == 1 and limits.get("maximum_concurrent_pods") == 1,
        "budget_single_pod",
    )
    _require(
        limits.get("maximum_network_volumes") == 1 and limits.get("auto_pay") is False,
        "budget_resource_cap",
    )
    _require(
        limits.get("network_volume_tier") == "standard"
        and limits.get("public_http_ports") == 0
        and limits.get("notify_low_balance") is True,
        "budget_resource_shape",
    )
    container_image = _text(limits.get("container_image"), "budget_container_image")
    _require("@sha256:" in container_image, "budget_container_image")
    _require(
        bool(SHA256_RE.fullmatch(container_image.rsplit("@sha256:", 1)[1])),
        "budget_container_image",
    )
    prohibited = set(_array(budget.get("prohibited_resources"), "budget_prohibited"))
    for required in (
        "community_cloud",
        "savings_plan",
        "three_month_commitment",
        "six_month_commitment",
        "multi_gpu_pod",
        "auto_pay",
    ):
        _require(required in prohibited, "budget_prohibited_coverage")

    sampling_options = {
        _text(
            _object(item, "sampling_option").get("option_id"), "sampling_option_id"
        ): _object(item, "sampling_option")
        for item in _array(sampling.get("options"), "sampling_options")
    }
    scenarios = _array(budget.get("scenarios"), "budget_scenarios")
    by_scenario: dict[str, dict[str, Any]] = {}
    container_rate = _number(
        _object(budget.get("cost_assumptions"), "budget_cost_assumptions").get(
            "container_disk_usd_per_gb_month_running"
        ),
        "budget_container_rate",
    )
    month_hours = _number(
        _object(budget.get("cost_assumptions"), "budget_cost_assumptions").get(
            "month_hours"
        ),
        "budget_month_hours",
    )
    for raw in scenarios:
        scenario = _object(raw, "budget_scenario")
        scenario_id = _text(scenario.get("id"), "budget_scenario_id")
        _require(scenario_id not in by_scenario, "budget_scenario_duplicate")
        by_scenario[scenario_id] = scenario
        sample = sampling_options[scenario_id]
        references = int(
            _number(scenario.get("reference_image_count"), "budget_reference_count")
        )
        holdout = int(
            _number(scenario.get("holdout_query_count"), "budget_holdout_count")
        )
        governed = int(
            _number(scenario.get("governed_asset_count"), "budget_governed_count")
        )
        _require(
            governed == references + holdout and holdout >= 100, "budget_asset_math"
        )
        _require(
            references == sample.get("reference_image_count")
            and holdout == sample.get("holdout_query_count"),
            "budget_sampling_counts",
        )
        sample_storage = _object(sample.get("storage_gb"), "budget_sampling_storage")
        _close(
            _number(
                scenario.get("corpus_storage_required_gb"), "budget_corpus_storage"
            ),
            _number(
                sample_storage.get("total_cloud_requirement"), "budget_sample_total"
            ),
            "budget_sampling_storage",
            tolerance=1e-6,
        )
        _close(
            _number(scenario.get("local_output_gb"), "budget_local_output"),
            _number(
                sample_storage.get("estimated_local_output"), "budget_sample_local"
            ),
            "budget_sampling_local",
            tolerance=1e-6,
        )
        _require(
            _number(scenario.get("network_volume_gb"), "budget_volume")
            >= _number(
                scenario.get("corpus_storage_required_gb"), "budget_corpus_storage"
            ),
            "budget_volume_capacity",
        )
        descriptor_hours = _number(
            scenario.get("descriptor_generation_gpu_hours"), "budget_descriptor_hours"
        )
        evaluation_hours = _number(
            scenario.get("evaluation_gpu_hours"), "budget_evaluation_hours"
        )
        setup_hours = _number(
            scenario.get("setup_validation_gpu_hours"), "budget_setup_hours"
        )
        expected_hours = _number(
            scenario.get("expected_gpu_hours"), "budget_scenario_expected"
        )
        max_hours = _number(scenario.get("maximum_gpu_hours"), "budget_scenario_max")
        _close(
            expected_hours,
            descriptor_hours + evaluation_hours + setup_hours,
            "budget_expected_math",
        )
        _require(0 < expected_hours <= max_hours < 10_000, "budget_scenario_hours")
        session_hours = _number(
            scenario.get("pod_session_max_hours"), "budget_session_hours"
        )
        session_count = _number(
            scenario.get("maximum_session_count"), "budget_session_count"
        )
        _require(session_hours * session_count >= max_hours, "budget_session_cap")
        compute = expected_hours * rate
        container = (
            _number(scenario.get("container_disk_gb"), "budget_container_gb")
            * container_rate
            * expected_hours
            / month_hours
        )
        storage = (
            _number(scenario.get("network_volume_gb"), "budget_volume") * storage_rate
        )
        _close(
            _number(scenario.get("one_time_compute_cost"), "budget_compute_cost"),
            compute,
            "budget_compute_cost",
            tolerance=1e-6,
        )
        _close(
            _number(
                scenario.get("temporary_container_storage_cost"),
                "budget_container_cost",
            ),
            container,
            "budget_container_cost",
            tolerance=1e-6,
        )
        _close(
            _number(
                scenario.get("one_time_container_storage_cost"),
                "budget_container_cost_explicit",
            ),
            container,
            "budget_container_cost_explicit",
            tolerance=1e-6,
        )
        _close(
            _number(
                scenario.get("ongoing_monthly_storage_cost"), "budget_storage_cost"
            ),
            storage,
            "budget_storage_cost",
            tolerance=1e-6,
        )
        _close(
            _number(
                scenario.get("recurring_network_storage_monthly_cost"),
                "budget_storage_cost_explicit",
            ),
            storage,
            "budget_storage_cost_explicit",
            tolerance=1e-6,
        )
        _close(
            _number(scenario.get("estimated_cost"), "budget_estimated_cost"),
            compute + container + storage,
            "budget_estimated_cost",
            tolerance=1e-6,
        )
        _close(
            _number(
                scenario.get("estimated_first_month_cost"),
                "budget_estimated_first_month",
            ),
            compute + container + storage,
            "budget_estimated_first_month",
            tolerance=1e-6,
        )
        maximum_container = (
            _number(scenario.get("container_disk_gb"), "budget_container_gb")
            * container_rate
            * max_hours
            / month_hours
        )
        _close(
            _number(scenario.get("maximum_compute_cost"), "budget_max_compute"),
            max_hours * rate,
            "budget_max_compute",
            tolerance=1e-6,
        )
        _close(
            _number(
                scenario.get("maximum_container_storage_cost"),
                "budget_max_container",
            ),
            maximum_container,
            "budget_max_container",
            tolerance=1e-6,
        )
        maximum_first_month = max_hours * rate + maximum_container + storage
        _close(
            _number(
                scenario.get("maximum_first_month_cost"),
                "budget_max_first_month",
            ),
            maximum_first_month,
            "budget_max_first_month",
            tolerance=1e-6,
        )
        scenario_soft = _number(scenario.get("soft_budget"), "budget_scenario_soft")
        scenario_hard = _number(scenario.get("hard_budget"), "budget_scenario_hard")
        _require(
            maximum_first_month <= scenario_soft <= scenario_hard,
            "budget_scenario_caps",
        )
        _close(
            _number(
                scenario.get("hard_budget_headroom_after_maximum_first_month_cost"),
                "budget_headroom",
            ),
            scenario_hard - maximum_first_month,
            "budget_headroom",
            tolerance=1e-6,
        )
        _close(
            _number(scenario.get("maximum_authorized_cost"), "budget_authorized"),
            scenario_hard,
            "budget_authorized",
        )
    _require(set(by_scenario) == set(OPTION_IDS), "budget_scenario_coverage")
    recommended = by_scenario["recommended_mvp"]
    for top_key, scenario_key in (
        ("storage_gb", "network_volume_gb"),
        ("expected_gpu_hours", "expected_gpu_hours"),
        ("maximum_gpu_hours", "maximum_gpu_hours"),
        ("soft_budget", "soft_budget"),
        ("hard_budget", "hard_budget"),
    ):
        _close(
            _number(budget.get(top_key), f"budget_top_{top_key}"),
            _number(recommended.get(scenario_key), f"budget_rec_{scenario_key}"),
            f"budget_recommended_{top_key}",
        )


def _parse_timestamp(value: object, code: str) -> None:
    if value is None:
        return
    text = _text(value, code).replace("Z", "+00:00")
    try:
        datetime.fromisoformat(text)
    except ValueError as error:
        raise Phase3ValidationError(code) from error


def _haversine_m(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6_371_008.8 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _validate_spatial_records(
    records: list[tuple[str, str, float, float]],
    *,
    holdout_buffer_m: float,
    reference_buffer_m: float,
) -> None:
    record_buckets: dict[
        tuple[int, int],
        list[tuple[str, str, float, float]],
    ]

    def nearby(
        buckets: dict[
            tuple[int, int],
            list[tuple[str, str, float, float]],
        ],
        *,
        latitude: float,
        longitude: float,
        cell_degrees: float,
        distance_m: float,
    ) -> list[tuple[str, str, float, float]]:
        latitude_cell = math.floor((latitude + 90) / cell_degrees)
        longitude_cell = math.floor((longitude + 180) / cell_degrees)
        latitude_span = math.ceil(distance_m / (111_320 * cell_degrees))
        longitude_scale = max(0.05, abs(math.cos(math.radians(latitude))))
        longitude_span = math.ceil(
            distance_m / (111_320 * longitude_scale * cell_degrees)
        )
        found: list[tuple[str, str, float, float]] = []
        for lat_offset in range(-latitude_span, latitude_span + 1):
            for lon_offset in range(-longitude_span, longitude_span + 1):
                found.extend(
                    buckets.get(
                        (latitude_cell + lat_offset, longitude_cell + lon_offset),
                        [],
                    )
                )
        return found

    reference_cell = reference_buffer_m / 111_320
    exact_split_cell = 1.0 / 111_320
    reference_buckets: dict[tuple[int, int], list[tuple[str, str, float, float]]] = {}
    holdout_buckets: dict[tuple[int, int], list[tuple[str, str, float, float]]] = {}
    split_buckets: dict[tuple[int, int], list[tuple[str, str, float, float]]] = {}
    for current in records:
        role, spatial_split, latitude, longitude = current
        comparisons: list[tuple[list[tuple[str, str, float, float]], float]]
        if role == "holdout":
            comparisons = [
                (
                    nearby(
                        holdout_buckets,
                        latitude=latitude,
                        longitude=longitude,
                        cell_degrees=reference_cell,
                        distance_m=holdout_buffer_m,
                    ),
                    holdout_buffer_m,
                ),
                (
                    nearby(
                        reference_buckets,
                        latitude=latitude,
                        longitude=longitude,
                        cell_degrees=reference_cell,
                        distance_m=reference_buffer_m,
                    ),
                    reference_buffer_m,
                ),
            ]
            record_buckets = holdout_buckets
        else:
            comparisons = [
                (
                    nearby(
                        holdout_buckets,
                        latitude=latitude,
                        longitude=longitude,
                        cell_degrees=reference_cell,
                        distance_m=reference_buffer_m,
                    ),
                    reference_buffer_m,
                ),
                (
                    nearby(
                        split_buckets,
                        latitude=latitude,
                        longitude=longitude,
                        cell_degrees=exact_split_cell,
                        distance_m=1.0,
                    ),
                    1.0,
                ),
            ]
            record_buckets = reference_buckets
        for nearby_records, buffer_m in comparisons:
            for _, prior_split, prior_lat, prior_lon in nearby_records:
                if buffer_m == 1.0 and spatial_split == prior_split:
                    continue
                _require(
                    _haversine_m((latitude, longitude), (prior_lat, prior_lon))
                    >= buffer_m,
                    "manifest_spatial_split_conflict",
                )
        cell = reference_cell
        key = (
            math.floor((latitude + 90) / cell),
            math.floor((longitude + 180) / cell),
        )
        record_buckets.setdefault(key, []).append(current)
        if role != "holdout":
            split_key = (
                math.floor((latitude + 90) / exact_split_cell),
                math.floor((longitude + 180) / exact_split_cell),
            )
            split_buckets.setdefault(split_key, []).append(current)


def validate_manifest_rows(
    rows: list[dict[str, Any]],
    source_policy: dict[str, Any],
    schema: dict[str, Any],
    leakage_policy: dict[str, Any],
) -> None:
    required = set(_array(schema.get("required"), "manifest_required"))
    allowed = set(_object(schema.get("properties"), "manifest_properties"))
    source_by_name = {
        _text(_object(raw, "source_record").get("source_name"), "source_name"): _object(
            raw, "source_record"
        )
        for raw in _array(source_policy.get("sources"), "source_records")
    }
    policy_version = _text(source_policy.get("policy_version"), "source_policy_version")
    asset_ids: set[str] = set()
    sha_to_asset: dict[str, str] = {}
    phash_to_role: dict[str, str] = {}
    sequence_splits: dict[tuple[str, str], str] = {}
    capture_roles: dict[tuple[str, str], str] = {}
    source_asset_roles: dict[tuple[str, str], str] = {}
    parent_roles: dict[tuple[str, str], str] = {}
    spatial_records: list[tuple[str, str, float, float]] = []

    for row in rows:
        _require(required <= row.keys(), "manifest_row_missing_field")
        _require(row.keys() <= allowed, "manifest_row_unknown_field")
        asset_id = _text(row.get("asset_id"), "manifest_asset_id")
        _require(asset_id not in asset_ids, "manifest_asset_id_duplicate")
        asset_ids.add(asset_id)
        source_name = _text(row.get("source_name"), "manifest_source_name")
        _require(source_name in source_by_name, "manifest_source_unknown")
        source = source_by_name[source_name]
        policy_decision = _text(
            source.get("current_decision"), "manifest_policy_decision"
        )
        _require(
            row.get("source_policy_decision") == policy_decision,
            "manifest_policy_decision_drift",
        )
        _require(
            source.get("source_id")
            not in {"openstreetmap_metadata", "geonames_metadata"},
            "manifest_metadata_not_imagery",
        )
        acquisition_ready = row.get("acquisition_ready")
        _require(isinstance(acquisition_ready, bool), "manifest_acquisition_ready_type")
        if acquisition_ready:
            _require(
                policy_decision not in {"UNKNOWN", "BLOCKED"},
                "manifest_blocked_or_unknown_source",
            )
            _require(
                policy_decision in ACQUISITION_READY,
                "manifest_source_not_acquisition_ready",
            )
            for field in (
                "source_policy_decision",
                "commercial_use_decision",
                "derivative_index_decision",
            ):
                _require(
                    row.get(field) in ACQUISITION_READY, "manifest_decision_not_ready"
                )
            _require(
                row.get("personal_data_blur_state")
                in {"NOT_DETECTED", "BLURRED_AT_SOURCE", "BLURRED_BY_ATLASLENS"},
                "manifest_blur_not_ready",
            )
        if policy_decision == "FIRST_PARTY_ONLY":
            _require(
                source.get("source_id") == "atlaslens_first_party_imagery",
                "manifest_first_party_scope",
            )

        _uri(row.get("source_url"), "manifest_source_url")
        _uri(row.get("license_url"), "manifest_license_url")
        _text(row.get("license_identifier"), "manifest_license")
        attribution = _text(row.get("attribution_text"), "manifest_attribution")
        if policy_decision == "GO_WITH_ATTRIBUTION":
            _require(
                attribution.upper() not in {"N/A", "NONE", "NOT_APPLICABLE"},
                "manifest_attribution_required",
            )
        latitude = _number(row.get("latitude"), "manifest_latitude")
        longitude = _number(row.get("longitude"), "manifest_longitude")
        _require(
            -90 <= latitude <= 90 and -180 <= longitude <= 180, "manifest_coordinates"
        )
        _require(
            _number(row.get("coordinate_accuracy_m"), "manifest_coordinate_accuracy")
            > 0,
            "manifest_coordinate_accuracy",
        )
        heading = row.get("heading_degrees")
        if heading is not None:
            heading_value = _number(heading, "manifest_heading")
            _require(0 <= heading_value < 360, "manifest_heading")
        sha = _text(row.get("image_sha256"), "manifest_sha256")
        phash = _text(row.get("perceptual_hash"), "manifest_perceptual_hash")
        _require(bool(SHA256_RE.fullmatch(sha)), "manifest_sha256")
        _require(bool(PHASH_RE.fullmatch(phash)), "manifest_perceptual_hash")
        _require(sha not in sha_to_asset, "manifest_exact_hash_duplicate")
        sha_to_asset[sha] = asset_id
        role = _text(row.get("role"), "manifest_role")
        _require(
            role in {"train", "development", "validation", "holdout"}, "manifest_role"
        )
        if phash in phash_to_role:
            _require(
                phash_to_role[phash] == role, "manifest_perceptual_hash_cross_role"
            )
        else:
            phash_to_role[phash] = role
        for field, allowed_values in MANIFEST_STRATUM_ENUMS.items():
            _require(
                _text(row.get(field), f"manifest_{field}") in allowed_values,
                f"manifest_{field}",
            )
        _require(
            _number(row.get("width_px"), "manifest_width") > 0
            and _number(row.get("height_px"), "manifest_height") > 0,
            "manifest_dimensions",
        )
        _require(
            _text(row.get("mime_type"), "manifest_mime_type")
            in {"image/jpeg", "image/png", "image/webp"},
            "manifest_mime_type",
        )
        spatial_split = _text(row.get("spatial_split"), "manifest_spatial_split")
        province = _text(row.get("province_code"), "manifest_province")
        _require(province in PROVINCE_CODES, "manifest_province")
        if role == "holdout":
            _require(
                province not in DENSE_PROVINCE_CODES, "manifest_dense_province_holdout"
            )
        _parse_timestamp(row.get("capture_timestamp"), "manifest_capture_timestamp")
        _parse_timestamp(
            row.get("acquisition_timestamp"), "manifest_acquisition_timestamp"
        )

        deletion = _object(
            row.get("deletion_revocation_state"), "manifest_deletion_state"
        )
        _require(
            deletion.get("status")
            in {"ACTIVE", "QUARANTINED", "DELETION_PENDING", "DELETED", "REVOKED"},
            "manifest_deletion_status",
        )
        if acquisition_ready:
            _require(deletion.get("status") == "ACTIVE", "manifest_deletion_not_active")
        _parse_timestamp(deletion.get("last_checked_at"), "manifest_deletion_timestamp")
        provenance = _object(row.get("provenance_receipt"), "manifest_provenance")
        for field in (
            "receipt_id",
            "source_policy_version",
            "evidence_date",
            "acquisition_method",
            "receipt_sha256",
        ):
            _require(field in provenance, "manifest_provenance_fields")
        _require(
            provenance.get("source_policy_version") == policy_version,
            "manifest_provenance_policy_version",
        )
        _require(
            bool(
                SHA256_RE.fullmatch(
                    _text(
                        provenance.get("receipt_sha256"), "manifest_provenance_sha256"
                    )
                )
            ),
            "manifest_provenance_sha256",
        )

        sequence_id = row.get("sequence_id")
        if isinstance(sequence_id, str) and sequence_id:
            key = (source_name, sequence_id)
            previous = sequence_splits.setdefault(key, spatial_split)
            _require(previous == spatial_split, "manifest_sequence_cross_split")
        capture_run = row.get("capture_run_id")
        contributor = _text(row.get("contributor_or_owner"), "manifest_contributor")
        if isinstance(capture_run, str) and capture_run:
            key = (contributor, capture_run)
            previous = capture_roles.setdefault(key, role)
            _require(previous == role, "manifest_capture_run_cross_role")
        source_asset_id = _text(row.get("source_asset_id"), "manifest_source_asset_id")
        source_key = (source_name, source_asset_id)
        previous_role = source_asset_roles.setdefault(source_key, role)
        _require(previous_role == role, "manifest_source_asset_cross_role")
        for field in ("source_tile_id", "parent_source_asset_id"):
            value = row.get(field)
            if isinstance(value, str) and value:
                parent_key = (source_name, value)
                prior = parent_roles.setdefault(parent_key, role)
                _require(prior == role, "manifest_parent_asset_cross_role")
        spatial_records.append((role, spatial_split, latitude, longitude))

    thresholds = _object(leakage_policy.get("thresholds"), "leakage_thresholds")
    holdout_buffer = _number(
        thresholds.get("query_to_query_holdout_buffer_m"), "manifest_holdout_buffer"
    )
    reference_buffer = _number(
        thresholds.get("same_view_reference_to_holdout_buffer_m"),
        "manifest_reference_buffer",
    )
    _validate_spatial_records(
        spatial_records,
        holdout_buffer_m=holdout_buffer,
        reference_buffer_m=reference_buffer,
    )


def _coerce_csv_value(key: str, value: str | None) -> Any:
    if value is None:
        return None
    stripped = value.strip()
    if stripped in {"", "null"}:
        return None
    if key in {"deletion_revocation_state", "provenance_receipt"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as error:
            raise Phase3ValidationError("manifest_csv_nested_json") from error
    if key == "acquisition_ready":
        _require(stripped.lower() in {"true", "false"}, "manifest_csv_boolean")
        return stripped.lower() == "true"
    if key in {"latitude", "longitude", "coordinate_accuracy_m", "heading_degrees"}:
        try:
            return float(stripped)
        except ValueError as error:
            raise Phase3ValidationError("manifest_csv_number") from error
    if key in {"width_px", "height_px"}:
        try:
            return int(stripped)
        except ValueError as error:
            raise Phase3ValidationError("manifest_csv_integer") from error
    return stripped


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    _require(
        path.suffix.lower() in {".json", ".jsonl", ".ndjson", ".csv"}, "manifest_suffix"
    )
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return [
                    {key: _coerce_csv_value(key, value) for key, value in row.items()}
                    for row in csv.DictReader(handle)
                    if key_values_present(row)
                ]
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            rows: list[dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(_object(json.loads(line), "manifest_jsonl_row"))
            return rows
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase3ValidationError("manifest_invalid_or_unreadable") from error
    if isinstance(raw, list):
        return [_object(item, "manifest_row") for item in raw]
    root = _object(raw, "manifest_root")
    for key in ("records", "assets"):
        if key in root:
            return [
                _object(item, "manifest_row")
                for item in _array(root[key], "manifest_rows")
            ]
    return [root]


def key_values_present(row: dict[str, str | None]) -> bool:
    return any(value is not None and value.strip() for value in row.values())


def validate_repository(root: Path, manifest: Path | None = None) -> dict[str, Any]:
    source = load_json_object(root / "config" / "corpus" / "source-policy-v1.json")
    schema = load_json_object(root / "config" / "corpus" / "manifest-schema-v1.json")
    sampling = load_json_object(
        root / "config" / "corpus" / "turkiye-pilot-sampling-v1.json"
    )
    leakage = load_json_object(root / "config" / "corpus" / "leakage-policy-v1.json")
    budget = load_json_object(
        root / "config" / "cloud" / "runpod-phase3-budget-v1.json"
    )
    validate_source_policy(source)
    validate_manifest_schema(schema)
    validate_sampling(sampling)
    validate_leakage_policy(leakage)
    validate_budget(budget, sampling)
    row_count = 0
    if manifest is not None:
        rows = load_manifest_rows(manifest)
        _require(bool(rows), "manifest_empty")
        validate_manifest_rows(rows, source, schema, leakage)
        row_count = len(rows)
    return {
        "status": "passed",
        "schema_version": "atlaslens-phase3-validator-v1",
        "validated_config_count": 5,
        "manifest_row_count": row_count,
        "network_calls": 0,
        "downloads": 0,
        "image_bytes_read": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = cast(Path, args.root)
    manifest = cast(Path | None, args.manifest)
    try:
        result = validate_repository(
            root.resolve(), manifest.resolve() if manifest else None
        )
    except Phase3ValidationError as error:
        print(json.dumps({"status": "failed", "code": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
