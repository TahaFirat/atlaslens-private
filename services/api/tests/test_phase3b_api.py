from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from atlaslens_api.corpus_index.descriptors import DescriptorDataset
from atlaslens_api.corpus_index.index import build_index
from atlaslens_api.corpus_index.models import AssetProvenance, DescriptorSpec
from atlaslens_api.turkiye_reference import TurkiyeReferenceIndexProvider
from conftest import image_bytes, upload, wait_for_terminal


def _published_index(tmp_path: Path) -> tuple[Path, Path, DescriptorSpec]:
    policy_path = tmp_path / "source-policy.json"
    policy_path.write_text('{"version":"source-policy-test"}\n', encoding="utf-8")
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    spec = DescriptorSpec(provider_id="approved-fixture-adapter", version="fixture-v1", dimension=4)
    provenance = AssetProvenance(
        asset_id="fixture-reference-1",
        source_id="atlaslens_first_party_imagery",
        rights_decision="FIRST_PARTY_ONLY",
        license_record_id="fixture-license-receipt",
        provenance_summary="synthetic first-party test fixture",
        content_sha256="d" * 64,
        split="reference",
        rights_validated=True,
        province="TR-38",
        latitude=38.7,
        longitude=35.5,
    )
    dataset = DescriptorDataset(
        tmp_path / "descriptors",
        spec,
        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        (provenance,),
        {
            "config_hash": "a" * 64,
            "input_hash": "b" * 64,
            "source_policy_hash": policy_hash,
            "manifest_hash": "c" * 64,
        },
    )
    index_path = tmp_path / "published-index"
    build_index(
        dataset,
        output_dir=index_path,
        backend="exact",
        index_version="phase3b-api-fixture-v1",
        source_policy_hash=policy_hash,
        shard_size=8,
    )
    return index_path, policy_path, spec


def _provider(
    index_path: Path,
    policy_path: Path,
    spec: DescriptorSpec,
) -> TurkiyeReferenceIndexProvider:
    return TurkiyeReferenceIndexProvider(
        enabled=True,
        index_path=index_path,
        source_policy_path=policy_path,
        descriptor_provider=spec.provider_id,
        descriptor_version=spec.version,
        descriptor_dimension=spec.dimension,
    )


def test_api_reports_turkiye_reference_disabled_by_default(client: Any) -> None:
    capability = client.get("/api/v1/capabilities").json()["providers"]["turkiye_reference_index"]
    assert capability["enabled"] is False
    assert capability["available"] is False
    assert capability["operational_status"] == "disabled"

    providers = client.get("/api/v1/providers").json()["providers"]
    status = next(item for item in providers if item["provider_id"] == "turkiye-reference-index-v1")
    assert status["mode"] == "disabled"
    assert status["status"] == "disabled"
    readiness = client.get("/api/v1/ready")
    assert readiness.status_code == 200
    assert readiness.json()["checks"]["turkiye_reference_index"] == "ok"


def test_api_reports_not_ready_for_missing_configured_index(
    client_factory: Any,
    tmp_path: Path,
) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text("{}\n", encoding="utf-8")
    client = client_factory(
        turkiye_reference_index_enabled=True,
        turkiye_reference_index_path=tmp_path / "missing-index",
        turkiye_reference_source_policy_path=policy,
        turkiye_reference_descriptor_id="approved-provider",
        turkiye_reference_descriptor_version="v1",
        turkiye_reference_descriptor_dimension=4,
    )
    capability = client.get("/api/v1/capabilities").json()["providers"]["turkiye_reference_index"]
    assert capability["operational_status"] == "not_ready"
    assert capability["reason_code"] == "index_artifacts_missing"
    readiness = client.get("/api/v1/ready")
    assert readiness.status_code == 503
    assert readiness.json()["checks"]["turkiye_reference_index"] == "error"


def test_verified_index_is_ready_and_returns_distance_not_confidence(
    client_factory: Any,
    tmp_path: Path,
) -> None:
    index_path, policy_path, spec = _published_index(tmp_path)
    provider = _provider(index_path, policy_path, spec)
    client = client_factory(turkiye_reference_provider=provider)

    capability = client.get("/api/v1/capabilities").json()["providers"]["turkiye_reference_index"]
    assert capability["operational_status"] == "ready"
    assert capability["available"] is True
    assert capability["calibration_state"] == "uncalibrated"
    assert client.get("/api/v1/ready").status_code == 200

    hits = provider.search_vector([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].rank == 1
    assert hits[0].cosine_distance == 0.0
    assert hits[0].reference_asset_id == "fixture-reference-1"
    assert hits[0].provenance_summary == "synthetic first-party test fixture"
    assert not hasattr(hits[0], "confidence")


def test_test_only_descriptor_identity_cannot_make_api_ready(tmp_path: Path) -> None:
    index_path, policy_path, spec = _published_index(tmp_path)
    provider = TurkiyeReferenceIndexProvider(
        enabled=True,
        index_path=index_path,
        source_policy_path=policy_path,
        descriptor_provider="atlaslens-test-only-deterministic",
        descriptor_version=spec.version,
        descriptor_dimension=spec.dimension,
    )
    status = provider.status()
    assert status.state == "not_ready"
    assert status.reason_code == "test_descriptor_provider_forbidden"


def test_existing_analysis_route_remains_compatible(client: Any) -> None:
    accepted = upload(client, image_bytes())
    assert accepted.status_code == 202
    result = wait_for_terminal(client, accepted.json()["id"])
    assert result["status"] == "completed"
