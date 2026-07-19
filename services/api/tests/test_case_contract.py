from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI

from atlaslens_api.case_routes import router

CONTRACT = Path(__file__).resolve().parents[3] / "packages" / "contracts" / "openapi.yaml"

EXPECTED_OPERATIONS = {
    ("/api/v1/cases", "post"): "createCase",
    ("/api/v1/cases", "get"): "listCases",
    ("/api/v1/cases/{case_id}", "get"): "getCase",
    ("/api/v1/cases/{case_id}", "patch"): "patchCase",
    ("/api/v1/cases/{case_id}/media", "post"): "createCaseMedia",
    ("/api/v1/cases/{case_id}/media", "get"): "listCaseMedia",
    ("/api/v1/cases/{case_id}/analyses/{analysis_id}/link", "post"): "linkCaseAnalysis",
    (
        "/api/v1/cases/{case_id}/analyses/{analysis_id}/materialize-evidence",
        "post",
    ): "materializeCaseEvidence",
    ("/api/v1/cases/{case_id}/evidence", "get"): "listCaseEvidence",
    ("/api/v1/cases/{case_id}/hypotheses", "get"): "listCaseHypotheses",
    (
        "/api/v1/cases/{case_id}/hypotheses/{hypothesis_id}/adjudications",
        "post",
    ): "createHypothesisAdjudication",
    ("/api/v1/cases/{case_id}/operator-hypotheses", "post"): "createOperatorHypothesis",
    ("/api/v1/cases/{case_id}/audit-events", "get"): "listCaseAuditEvents",
    ("/api/v1/cases/{case_id}/audit-integrity", "get"): "getCaseAuditIntegrity",
}


def _contract() -> dict[str, Any]:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def _without_docs(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_docs(nested)
            for key, nested in value.items()
            if key not in {"description", "summary"}
        }
    if isinstance(value, list):
        return [_without_docs(nested) for nested in value]
    return value


def test_case_paths_match_router_and_use_safe_error_envelope() -> None:
    document = _contract()
    app = FastAPI()
    app.include_router(router)
    generated = app.openapi()

    assert sum(path.startswith("/api/v1/cases") for path in document["paths"]) == 11
    for (path, method), operation_id in EXPECTED_OPERATIONS.items():
        operation = document["paths"][path][method]
        assert operation["operationId"] == operation_id
        assert generated["paths"][path][method]["operationId"] == operation_id
        assert operation["responses"]["422"] == {"$ref": "#/components/responses/ProblemResponse"}
        assert not {"put", "delete"}.intersection(document["paths"][path])
        generated_operation = generated["paths"][path][method]
        for code in list(generated_operation["responses"]):
            if int(code) >= 400:
                generated_operation["responses"][code] = {
                    "$ref": "#/components/responses/ProblemResponse"
                }
        assert _without_docs(operation) == _without_docs(generated_operation)


def test_case_components_match_runtime_models_with_non_recursive_json() -> None:
    document = _contract()
    app = FastAPI()
    app.include_router(router)
    generated_schemas = app.openapi()["components"]["schemas"]
    manual_schemas = document["components"]["schemas"]

    for name, schema in generated_schemas.items():
        if name in {"HTTPValidationError", "ValidationError"}:
            continue
        assert _without_docs(manual_schemas[name]) == _without_docs(schema), name

    json_value = manual_schemas["JsonValue"]
    assert [item["type"] for item in json_value["anyOf"]] == [
        "string",
        "integer",
        "number",
        "boolean",
        "null",
    ]
    assert "#/components/schemas/JsonValue" not in str(json_value)


def test_case_creation_contract_requires_product_controls_without_tenant_claim() -> None:
    schemas = _contract()["components"]["schemas"]
    create = schemas["CaseCreateRequest"]
    required = set(create["required"])

    assert {
        "title",
        "purpose",
        "source_context",
        "sensitivity",
        "authorization_attested",
        "retention_policy",
        "created_by_actor_id",
    } <= required
    assert "workspace_id" not in create["properties"]
    assert create["properties"]["authorization_attested"]["const"] is True
    assert schemas["CasePurpose"]["enum"] == [
        "journalism",
        "humanitarian",
        "disaster_response",
        "insurance",
        "authorized_security_research",
        "other",
    ]
    assert schemas["CaseStatus"]["enum"] == [
        "open",
        "under_review",
        "resolved",
        "archived",
    ]
    assert schemas["CaseSensitivity"]["enum"] == [
        "standard",
        "sensitive",
        "conflict_related",
    ]
    assert (
        "production tenant isolation is not claimed"
        in schemas["CaseView"]["properties"]["workspace_id"]["description"]
    )


def test_media_contract_is_image_metadata_only_and_reuses_analysis_workflow() -> None:
    document = _contract()
    schemas = document["components"]["schemas"]
    media = schemas["CaseMediaCreateRequest"]
    properties = media["properties"]

    assert schemas["MediaType"]["enum"] == ["image"]
    assert schemas["MediaSourceType"]["enum"] == [
        "upload",
        "source_url",
        "external_archive",
        "other",
    ]
    assert schemas["MediaStorageState"]["enum"] == [
        "ephemeral",
        "deleted_after_analysis",
        "unavailable",
        "externally_managed",
    ]
    assert not {
        "image",
        "image_bytes",
        "raw_image",
        "path",
        "storage_path",
        "raw_ocr",
    }.intersection(properties)
    assert "POST /api/v1/analyses" in media["description"]

    existing_analysis = document["paths"]["/api/v1/analyses"]["post"]
    assert existing_analysis["operationId"] == "createAnalysis"
    assert "multipart/form-data" in existing_analysis["requestBody"]["content"]
    assert "202" in existing_analysis["responses"]


def test_evidence_hypothesis_and_adjudication_contracts_avoid_false_precision() -> None:
    schemas = _contract()["components"]["schemas"]
    evidence = schemas["EvidenceView"]
    hypothesis = schemas["HypothesisView"]

    assert "Immutable normalized evidence" in evidence["description"]
    assert "confidence" not in hypothesis["properties"]
    assert hypothesis["properties"]["uncertainty_radius_m"]["exclusiveMinimum"] == 0
    assert schemas["CalibrationState"]["enum"] == [
        "calibrated",
        "uncalibrated",
        "not_applicable",
    ]
    assert schemas["AdjudicationDecision"]["enum"] == [
        "accepted",
        "rejected",
        "needs_more_evidence",
        "withdrawn",
    ]
    assert "Append-only" in schemas["AdjudicationView"]["description"]
    assert "separate operator-origin" in schemas["OperatorHypothesisCreateRequest"]["description"]


def test_audit_contract_is_tamper_evident_but_not_certified_evidence() -> None:
    document = _contract()
    schemas = document["components"]["schemas"]
    integrity = schemas["AuditIntegrityView"]

    assert integrity["properties"]["legally_certified_evidence"]["const"] is False
    assert "not legally certified evidence" in integrity["description"]
    assert schemas["AuditActorType"]["enum"] == ["operator", "system", "model"]
    audit_path = document["paths"]["/api/v1/cases/{case_id}/audit-events"]
    assert set(audit_path) == {"get"}


def test_every_local_component_reference_resolves() -> None:
    document = _contract()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/components/"):
                node: object = document
                for part in reference.removeprefix("#/").split("/"):
                    assert isinstance(node, dict)
                    assert part in node
                    node = node[part]
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(document)
