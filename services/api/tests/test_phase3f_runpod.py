from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from atlaslens_api.phase3f.runpod import (
    GRAPHQL_URL,
    MAPILLARY_ENV_REFERENCE,
    POD_CONNECTIVITY_TIMEOUT_SECONDS,
    POD_VOLUME_GB,
    REST_BASE_URL,
    GPUOffer,
    PodAllocationSimulationDiagnostic,
    PodConnection,
    PodConnectivityProgressDiagnostic,
    PodGPUAttestationProgressDiagnostic,
    RunPodAPIError,
    RunPodConfig,
    RunPodCreateError,
    RunPodV1Client,
    attest_image_identity,
    build_create_payload,
    simulate_pod_allocation_state_machine,
    validate_create_payload_contract,
)
from atlaslens_api.phase3f.safety import (
    MAX_RUNTIME_SECONDS,
    Phase3FSafetyError,
    PodRecord,
    PodRequest,
    SinglePodSession,
)

TOKEN = "runpod-test-token-never-log"
IMAGE = "registry.example/atlaslens@sha256:" + "a" * 64
PREFERENCES = (
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 3090",
)
SSH_PUBLIC_KEY = "ssh-ed25519 " + "A" * 68 + " atlaslens-phase3f"
_DEFAULT = object()
_MISSING_FIELD = object()
FIXTURES = Path(__file__).with_name("fixtures") / "phase3f"


def _config(**overrides: object) -> RunPodConfig:
    values: dict[str, object] = {
        "image_name": IMAGE,
        "gpu_type_preferences": PREFERENCES,
        "container_disk_gb": 40,
        "min_gpu_memory_gb": 16,
        "ssh_public_key": SSH_PUBLIC_KEY,
    }
    values.update(overrides)
    return RunPodConfig(**values)  # type: ignore[arg-type]


def _request(*, public_ports: tuple[int, ...] = (22,)) -> PodRequest:
    return PodRequest(
        run_marker="phase3f-run-001",
        idempotency_key="phase3f-create-001",
        hourly_cost_usd=Decimal("0.50"),
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
        gpu_type_id="NVIDIA RTX A5000",
        public_ports=public_ports,
    )


def _offer() -> GPUOffer:
    return GPUOffer(
        gpu_type_id="NVIDIA RTX A5000",
        display_name="RTX A5000",
        memory_gb=24,
        hourly_price=Decimal("0.27"),
        stock_status="Low",
        cloud_type="SECURE",
        secure_cloud=True,
        community_cloud=False,
        available_gpu_counts=None,
    )


def _detail_row(
    gpu_type_id: str,
    *,
    display_name: str,
    memory_gb: int,
    price: object,
    counts: object = _DEFAULT,
    stock_status: object = "Low",
    secure_cloud: object = True,
    community_cloud: object = False,
    community_price: object = None,
) -> dict[str, object]:
    return {
        "id": gpu_type_id,
        "displayName": display_name,
        "memoryInGb": memory_gb,
        "secureCloud": secure_cloud,
        "communityCloud": community_cloud,
        "securePrice": (
            None
            if price is None
            else {
                "stockStatus": stock_status,
                "uninterruptablePrice": price,
                "availableGpuCounts": [1] if counts is _DEFAULT else counts,
            }
        ),
        "communityPrice": (
            None
            if community_price is None
            else {
                "stockStatus": stock_status,
                "uninterruptablePrice": community_price,
                "availableGpuCounts": [1] if counts is _DEFAULT else counts,
            }
        ),
    }


def _gpu_list_response() -> dict[str, object]:
    return {
        "data": {
            "gpuTypes": [
                {
                    "id": "NVIDIA RTX A4000",
                    "displayName": "RTX A4000",
                    "memoryInGb": 16,
                },
                {
                    "id": "NVIDIA RTX A5000",
                    "displayName": "RTX A5000",
                    "memoryInGb": 24,
                },
            ]
        }
    }


def _created_payload(
    *,
    interruptible: object = False,
    cost_per_hr: object = "0.45",
    gpu_type_id: str = "NVIDIA RTX A5000",
    desired_status: str = "RUNNING",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "created-pod",
        "name": "phase3f-run-001",
        "containerDiskInGb": 40,
        "endpointId": None,
        "networkVolume": None,
        "volumeInGb": POD_VOLUME_GB,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "gpu": {"id": gpu_type_id, "count": 1},
        "costPerHr": cost_per_hr,
        "desiredStatus": desired_status,
    }
    if interruptible is not _MISSING_FIELD:
        payload["interruptible"] = interruptible
    return payload


def test_image_identity_requires_the_exact_digest_without_normalization() -> None:
    assert attest_image_identity(IMAGE, {"imageName": IMAGE}) == IMAGE
    assert attest_image_identity(IMAGE, {}) is None
    with pytest.raises(RunPodAPIError, match="REMOTE_IMAGE_IDENTITY_MISMATCH"):
        attest_image_identity(IMAGE, {"imageName": "registry.example/atlaslens:latest"})
    with pytest.raises(RunPodAPIError, match="REMOTE_IMAGE_IDENTITY_MISMATCH"):
        attest_image_identity(
            IMAGE,
            {"imageName": "registry.example/atlaslens@sha256:" + "b" * 64},
        )


def test_create_response_diagnostic_records_secret_free_image_identity() -> None:
    client = RunPodV1Client(api_token=TOKEN, config=_config())
    body = bytearray(json.dumps({"id": "pod", "imageName": IMAGE}).encode())
    client._observe_create_response(201, body, "application/json")  # noqa: SLF001
    diagnostic = client.last_create_response_diagnostic
    assert diagnostic is not None
    assert diagnostic.image_name_present is True
    assert diagnostic.observed_image_name == IMAGE
    assert diagnostic.image_digest_match is True
    assert TOKEN not in repr(diagnostic.to_public_dict())


def _attested_pod_payload(
    *,
    interruptible: object = False,
    cost_per_hr: object = "0.45",
    gpu_shape: str = "gpu.id",
    gpu_type_id: str = "NVIDIA RTX A5000",
    gpu_count: int = 1,
    desired_status: str = "RUNNING",
) -> dict[str, object]:
    payload = _created_payload(
        interruptible=interruptible,
        cost_per_hr=cost_per_hr,
        gpu_type_id=gpu_type_id,
        desired_status=desired_status,
    )
    payload["publicIp"] = "192.0.2.10"
    payload["templateId"] = None
    if gpu_shape == "machine.gpuTypeId":
        payload.pop("gpu")
        payload["machine"] = {
            "gpuTypeId": gpu_type_id,
            "gpuType": {"count": gpu_count},
        }
    elif gpu_shape == "machine.gpuType.id":
        payload.pop("gpu")
        payload["machine"] = {
            "gpuType": {"id": gpu_type_id, "count": gpu_count}
        }
    else:
        payload["gpu"] = {"id": gpu_type_id, "count": gpu_count}
    return payload


def _graphql_response(
    request: httpx.Request,
    *,
    a5000_price: object = "0.45",
    a5000_counts: object = _DEFAULT,
    a5000_status: object = "Low",
    a4000_price: object = "0.29",
    a4000_counts: object = _DEFAULT,
    a4000_status: object = "Low",
) -> httpx.Response:
    body = json.loads(request.content)
    operation = body.get("operationName")
    if operation == "AtlasLensPhase3FGpuTypes":
        return httpx.Response(200, json=_gpu_list_response())
    assert operation == "AtlasLensPhase3FGpuDetails"
    query = body["query"]
    rows: list[dict[str, object]] = []
    if "NVIDIA RTX A5000" in query:
        rows.append(
            _detail_row(
                "NVIDIA RTX A5000",
                display_name="RTX A5000",
                memory_gb=24,
                price=a5000_price,
                counts=a5000_counts,
                stock_status=a5000_status,
                community_cloud=True,
            )
        )
    if "NVIDIA RTX A4000" in query:
        rows.append(
            _detail_row(
                "NVIDIA RTX A4000",
                display_name="RTX A4000",
                memory_gb=16,
                price=a4000_price,
                counts=a4000_counts,
                stock_status=a4000_status,
                community_cloud=True,
            )
        )
    return httpx.Response(
        200,
        json={"data": {f"g{index}": [row] for index, row in enumerate(rows)}},
    )


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> RunPodV1Client:
    return RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
    )


def _assert_secret_safe(request: httpx.Request) -> None:
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in str(request.url)
    assert request.url.query == b"" or b"api_key" not in request.url.query


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.parametrize(
    ("response", "expected_gpu_path", "expected_count_path"),
    [
        (
            {
                "desiredStatus": "RUNNING",
                "gpuCount": 1,
                "machine": {"gpuTypeId": "NVIDIA L4"},
            },
            "machine.gpuTypeId",
            "gpuCount",
        ),
        (
            {
                "desiredStatus": "RUNNING",
                "gpu": {"id": "NVIDIA L4", "count": 1},
            },
            "gpu.id",
            "gpu.count",
        ),
        (
            {
                "desiredStatus": "RUNNING",
                "machine": {"gpuType": {"id": "NVIDIA L4", "count": 1}},
            },
            "machine.gpuType.id",
            "machine.gpuType.count",
        ),
    ],
)
def test_gpu_count_normalizes_all_provider_response_paths(
    response: dict[str, object],
    expected_gpu_path: str,
    expected_count_path: str,
) -> None:
    result = simulate_pod_allocation_state_machine(
        response,
        expected_gpu_id="NVIDIA L4",
    )

    assert isinstance(result, PodAllocationSimulationDiagnostic)
    assert result.gpu_attestation_outcome == "passed"
    assert result.normalized_gpu_path == expected_gpu_path
    assert result.normalized_gpu_count_path == expected_count_path
    assert result.gpu_count == 1


def test_gpu_count_accepts_multiple_consistent_response_paths() -> None:
    result = simulate_pod_allocation_state_machine(
        {
            "desiredStatus": "RUNNING",
            "gpuCount": 1,
            "gpu": {"id": "NVIDIA L4", "count": 1},
            "machine": {
                "gpuTypeId": "NVIDIA L4",
                "gpuType": {"id": "NVIDIA L4", "count": 1},
            },
        },
        expected_gpu_id="NVIDIA L4",
    )

    assert result.normalized_gpu_count_path == "gpuCount"
    assert result.gpu_count == 1


@pytest.mark.parametrize(
    "value",
    [True, False, "1", 1.0, [], {}],
)
def test_gpu_count_rejects_non_json_integer_types(value: object) -> None:
    with pytest.raises(RunPodAPIError, match="POD_GPU_COUNT_INVALID"):
        simulate_pod_allocation_state_machine(
            {
                "desiredStatus": "RUNNING",
                "gpuCount": value,
                "machine": {"gpuTypeId": "NVIDIA L4"},
            },
            expected_gpu_id="NVIDIA L4",
        )


@pytest.mark.parametrize("value", [-1, 0, 2])
def test_gpu_count_rejects_integer_values_other_than_one(value: int) -> None:
    with pytest.raises(RunPodAPIError, match="POD_GPU_COUNT_MISMATCH"):
        simulate_pod_allocation_state_machine(
            {
                "desiredStatus": "RUNNING",
                "gpuCount": value,
                "machine": {"gpuTypeId": "NVIDIA L4"},
            },
            expected_gpu_id="NVIDIA L4",
        )


def test_gpu_count_rejects_conflicting_paths() -> None:
    with pytest.raises(RunPodAPIError, match="POD_GPU_COUNT_MISMATCH"):
        simulate_pod_allocation_state_machine(
            {
                "desiredStatus": "RUNNING",
                "gpuCount": 1,
                "gpu": {"id": "NVIDIA L4", "count": 2},
            },
            expected_gpu_id="NVIDIA L4",
        )


@pytest.mark.parametrize(
    "response",
    [
        {"desiredStatus": "RUNNING", "machine": {"gpuTypeId": "NVIDIA L4"}},
        {
            "desiredStatus": "RUNNING",
            "gpuCount": None,
            "gpu": {"id": "NVIDIA L4", "count": None},
            "machine": {
                "gpuTypeId": "NVIDIA L4",
                "gpuType": {"count": None},
            },
        },
    ],
)
def test_gpu_count_missing_or_null_has_specific_pending_terminal_code(
    response: dict[str, object],
) -> None:
    with pytest.raises(
        RunPodAPIError,
        match="POD_GPU_COUNT_ATTESTATION_TIMEOUT",
    ):
        simulate_pod_allocation_state_machine(
            response,
            expected_gpu_id="NVIDIA L4",
        )


def test_exact_live_shape_mutation_free_simulation_reaches_connectivity() -> None:
    fixture = json.loads(
        (FIXTURES / "runpod_create_201_top_level_gpu_count.json").read_text(
            encoding="utf-8"
        )
    )

    public = simulate_pod_allocation_state_machine(
        fixture,
        expected_gpu_id="NVIDIA L4",
    ).to_public_dict()

    assert public == {
        "gpu_attestation_outcome": "passed",
        "normalized_gpu_path": "machine.gpuTypeId",
        "normalized_gpu_count_path": "gpuCount",
        "gpu_count": 1,
        "next_state": "connectivity",
        "local_blockers": [],
        "network_calls": 0,
        "cloud_mutations": 0,
        "create_attempts": 0,
        "secret_free": True,
    }
    assert "created-pod" not in repr(public)
    assert TOKEN not in repr(public)


def test_inventory_lists_only_documented_rest_v1_resources() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        _assert_secret_safe(request)
        assert str(request.url).startswith(REST_BASE_URL)
        path = request.url.path
        if path.endswith("/pods"):
            assert request.url.params["includeMachine"] == "true"
            return httpx.Response(200, json=[{"id": "pod-1", "name": "phase3f-existing"}])
        if path.endswith("/endpoints"):
            return httpx.Response(200, json=[{"id": "endpoint-1"}])
        if path.endswith("/networkvolumes"):
            return httpx.Response(200, json=[{"id": "volume-1"}])
        if path.endswith("/templates"):
            return httpx.Response(200, json=[{"id": "template-1"}])
        raise AssertionError(path)

    with _client(handler) as client:
        inventory = client.inventory()
        assert repr(client) == "RunPodV1Client(api_token=<redacted>)"
    assert inventory.pods == (PodRecord("pod-1", "phase3f-existing"),)
    assert inventory.endpoint_ids == ("endpoint-1",)
    assert inventory.network_volume_ids == ("volume-1",)
    assert inventory.template_ids == ("template-1",)
    assert [request.method for request in requests] == ["GET", "GET", "GET", "GET"]


def test_account_billing_snapshot_is_authenticated_read_only_and_exact_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        _assert_secret_safe(request)
        assert str(request.url) == GRAPHQL_URL
        assert request.method == "POST"
        body = json.loads(request.content)
        assert body["operationName"] == "AtlasLensPhase3FAccountBilling"
        query = body["query"]
        assert "myself" in query
        assert "clientBalance" in query
        assert "currentSpendPerHr" in query
        assert "mutation" not in query.lower()
        return httpx.Response(
            200,
            json={
                "data": {
                    "myself": {
                        "clientBalance": "9.17",
                        "currentSpendPerHr": "0",
                    }
                }
            },
        )

    with _client(handler) as client:
        snapshot = client.account_billing_snapshot()
        assert client.api_request_count == 1
        assert client.cloud_mutation_count == 0

    assert snapshot.client_balance_usd == Decimal("9.17")
    assert snapshot.current_spend_per_hour_usd == Decimal("0")
    assert len(requests) == 1
    assert TOKEN not in repr(snapshot)


def test_account_billing_graphql_error_redacts_secret_and_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _assert_secret_safe(request)
        return httpx.Response(
            200,
            json={
                "errors": [
                    {
                        "message": f"Bearer {TOKEN} https://provider.invalid/private",
                        "extensions": {"code": "FORBIDDEN"},
                    }
                ],
                "data": {"myself": None},
            },
        )

    with _client(handler) as client:
        with pytest.raises(RunPodAPIError, match="BILLING_RECONCILIATION_GRAPHQL_ERRORS"):
            client.account_billing_snapshot()
        diagnostic = client.last_graphql_errors[0]

    assert diagnostic.message == "<redacted>"
    assert diagnostic.secret_free is True
    assert TOKEN not in repr(diagnostic)


@pytest.mark.parametrize(
    "myself",
    (
        None,
        {},
        {"clientBalance": None, "currentSpendPerHr": "0"},
        {"clientBalance": "9.17", "currentSpendPerHr": "0", "extra": "refuse"},
    ),
)
def test_account_billing_snapshot_fails_closed_on_unknown_schema(myself: object) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"myself": myself}})

    with _client(handler) as client:
        with pytest.raises(RunPodAPIError):
            client.account_billing_snapshot()
        assert client.cloud_mutation_count == 0


def test_gpu_offer_uses_official_list_then_detail_schema_and_lowest_price() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        _assert_secret_safe(request)
        assert str(request.url) == GRAPHQL_URL
        assert request.method == "POST"
        body = json.loads(request.content)
        assert "gpuTypes" in body["query"]
        assert "mutation" not in body["query"].lower()
        return _graphql_response(request)

    with _client(handler) as client:
        offer = client.select_gpu_offer(max_hourly_price=Decimal("0.50"))
        report = client.last_availability_report
    assert calls == 2
    assert offer.gpu_type_id == "NVIDIA RTX A5000"
    assert offer.memory_gb == 24
    assert offer.hourly_price == Decimal("0.45")
    assert offer.capacity_confirmed is True
    assert offer.capacity_evidence == "available_gpu_counts"
    assert report is not None
    assert report.graphql_request_count == 2
    assert report.response_schema_classification == "official_gpuTypes_list_and_detail_lists"


def test_malformed_or_unavailable_candidate_does_not_poison_valid_candidate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _graphql_response(
            request,
            a5000_counts="not-a-list",
            a4000_price="0.31",
            a4000_counts=[1],
            a4000_status="mEdIuM",
        )

    with _client(handler) as client:
        offer = client.select_gpu_offer(max_hourly_price=Decimal("0.50"))
        report = client.last_availability_report

    assert offer.gpu_type_id == "NVIDIA RTX A4000"
    assert report is not None
    rejected = [item for item in report.candidates if not item.accepted]
    assert any(item.classification == "available_gpu_counts_invalid" for item in rejected)


@pytest.mark.parametrize("status", ["Low", "Medium", "High"])
def test_null_counts_with_advertised_stock_are_eligible_but_unconfirmed(
    status: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _graphql_response(
            request,
            a5000_price="0.45",
            a5000_counts=None,
            a5000_status=status,
            a4000_price="0.60",
        )

    with _client(handler) as client:
        report = client.check_gpu_availability(max_hourly_price=Decimal("0.50"))

    offer = report.selected_offer
    assert offer is not None
    assert offer.gpu_type_id == "NVIDIA RTX A5000"
    assert offer.available_gpu_counts is None
    assert offer.capacity_confirmed is False
    assert offer.capacity_evidence == "advertised_stock_status"
    assert any(
        item.accepted
        and item.classification == "capacity_unconfirmed_but_advertised"
        for item in report.candidates
    )
    public = report.to_public_dict()
    assert public["capacity_confirmed"] is False
    assert public["capacity_evidence"] == "advertised_stock_status"
    assert public["create_attempt_limit"] == 1


@pytest.mark.parametrize("status", ["High", "medium", "LOW"])
def test_documented_stock_status_values_are_case_insensitive(status: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _graphql_response(
            request,
            a5000_price="0.30",
            a5000_status=status,
            a4000_price="0.60",
        )

    with _client(handler) as client:
        offer = client.select_gpu_offer(max_hourly_price=Decimal("0.50"))
    assert offer.gpu_type_id == "NVIDIA RTX A5000"


@pytest.mark.parametrize(
    ("price", "counts", "status", "classification"),
    [
        (None, [1], "Low", "lowest_price_unavailable"),
        ("0.30", [1], None, "stock_unavailable"),
        ("0.30", [1], "None", "stock_unavailable"),
        ("0.30", [], "Low", "gpu_count_one_unavailable"),
        ("0.30", "not-a-list", "Low", "available_gpu_counts_invalid"),
        ("not-numeric", [1], "Low", "gpu_price_invalid"),
        ("0.51", [1], "Low", "gpu_price_above_limit"),
    ],
)
def test_unavailable_or_malformed_candidate_is_aggregated_not_global_schema_error(
    price: object,
    counts: object,
    status: object,
    classification: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _graphql_response(
            request,
            a5000_price=price,
            a5000_counts=counts,
            a5000_status=status,
            a4000_price="0.60",
        )

    with _client(handler) as client:
        report = client.check_gpu_availability(max_hourly_price=Decimal("0.50"))

    assert report.selected_offer is None
    assert classification in {item.classification for item in report.candidates}


def test_graphql_errors_fail_closed_and_are_secret_redacted() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "errors": [
                    {
                        "message": f"provider refused Bearer {TOKEN}",
                        "extensions": {"code": "FORBIDDEN"},
                    }
                ]
            },
        )

    with _client(handler) as client:
        with pytest.raises(
            RunPodAPIError,
            match="GPU_AVAILABILITY_GRAPHQL_ERRORS",
        ) as error:
            client.check_gpu_availability(max_hourly_price=Decimal("0.50"))
        diagnostics = client.last_graphql_errors

    assert TOKEN not in str(error.value)
    assert len(diagnostics) == 1
    assert TOKEN not in diagnostics[0].message
    assert diagnostics[0].code == "FORBIDDEN"
    assert diagnostics[0].secret_free is True


def test_memory_below_minimum_is_rejected_before_detail_query() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["operationName"] == "AtlasLensPhase3FGpuTypes"
        return httpx.Response(
            200,
            json={
                "data": {
                    "gpuTypes": [
                        {
                            "id": "NVIDIA undersized fixture",
                            "displayName": "Undersized fixture",
                            "memoryInGb": 15,
                        }
                    ]
                }
            },
        )

    with _client(handler) as client:
        report = client.check_gpu_availability(max_hourly_price=Decimal("0.50"))

    assert calls == 1
    assert report.selected_offer is None
    assert report.candidates[0].classification == "memory_below_minimum"


def test_equal_price_uses_declared_preference_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _graphql_response(
            request,
            a5000_price="0.30",
            a4000_price="0.30",
        )

    with _client(handler) as client:
        offer = client.select_gpu_offer(max_hourly_price=Decimal("0.50"))
    assert offer.gpu_type_id == "NVIDIA RTX A5000"


def test_preferred_gpu_beats_cheaper_other_gpu() -> None:
    gpu_rows = [
        {"id": "NVIDIA RTX A4000", "displayName": "RTX A4000", "memoryInGb": 16},
        {"id": "NVIDIA L4", "displayName": "L4", "memoryInGb": 24},
    ]
    details = {
        "NVIDIA RTX A4000": _detail_row(
            "NVIDIA RTX A4000",
            display_name="RTX A4000",
            memory_gb=16,
            price="0.24",
            counts=None,
        ),
        "NVIDIA L4": _detail_row(
            "NVIDIA L4",
            display_name="L4",
            memory_gb=24,
            price="0.39",
            counts=None,
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["operationName"] == "AtlasLensPhase3FGpuTypes":
            return httpx.Response(200, json={"data": {"gpuTypes": gpu_rows}})
        query = body["query"]
        ordered_ids = sorted(details, key=query.index)
        return httpx.Response(
            200,
            json={
                "data": {
                    f"g{index}": [details[gpu_id]]
                    for index, gpu_id in enumerate(ordered_ids)
                }
            },
        )

    with _client(handler) as client:
        offer = client.select_gpu_offer(max_hourly_price=Decimal("0.50"))

    assert offer.gpu_type_id == "NVIDIA L4"
    assert offer.hourly_price == Decimal("0.39")


def test_a5000_preference_beats_l4_and_cheaper_other() -> None:
    gpu_rows = [
        {"id": "NVIDIA RTX A4000", "displayName": "RTX A4000", "memoryInGb": 16},
        {"id": "NVIDIA L4", "displayName": "L4", "memoryInGb": 24},
        {"id": "NVIDIA RTX A5000", "displayName": "RTX A5000", "memoryInGb": 24},
    ]
    details = {
        row["id"]: _detail_row(
            str(row["id"]),
            display_name=str(row["displayName"]),
            memory_gb=int(row["memoryInGb"]),
            price={
                "NVIDIA RTX A4000": "0.24",
                "NVIDIA L4": "0.39",
                "NVIDIA RTX A5000": "0.49",
            }[str(row["id"])],
            counts=None,
        )
        for row in gpu_rows
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["operationName"] == "AtlasLensPhase3FGpuTypes":
            return httpx.Response(200, json={"data": {"gpuTypes": gpu_rows}})
        query = body["query"]
        ordered_ids = sorted(details, key=query.index)
        return httpx.Response(
            200,
            json={
                "data": {
                    f"g{index}": [details[gpu_id]]
                    for index, gpu_id in enumerate(ordered_ids)
                }
            },
        )

    with _client(handler) as client:
        offer = client.select_gpu_offer(max_hourly_price=Decimal("0.50"))

    assert offer.gpu_type_id == "NVIDIA RTX A5000"
    assert offer.hourly_price == Decimal("0.49")


def test_secure_cloud_is_preferred_for_a_secure_advertised_offer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["operationName"] == "AtlasLensPhase3FGpuTypes":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "gpuTypes": [
                            {
                                "id": "NVIDIA L4",
                                "displayName": "L4",
                                "memoryInGb": 24,
                            }
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "g0": [
                        _detail_row(
                            "NVIDIA L4",
                            display_name="L4",
                            memory_gb=24,
                            price="0.39",
                            counts=None,
                            community_cloud=True,
                            community_price="0.20",
                        )
                    ]
                }
            },
        )

    with _client(handler) as client:
        offer = client.select_gpu_offer(max_hourly_price=Decimal("0.50"))

    assert offer.cloud_type == "SECURE"
    assert offer.hourly_price == Decimal("0.39")


def test_pre_create_revalidation_failure_prevents_rest_mutation() -> None:
    detail_calls = 0
    rest_create_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal detail_calls, rest_create_calls
        if str(request.url) != GRAPHQL_URL:
            rest_create_calls += 1
            raise AssertionError("REST create must not run after failed revalidation")
        body = json.loads(request.content)
        if body["operationName"] == "AtlasLensPhase3FGpuTypes":
            return _graphql_response(request)
        detail_calls += 1
        if detail_calls == 1:
            return _graphql_response(
                request,
                a5000_price="0.30",
                a4000_price="0.60",
            )
        return _graphql_response(
            request,
            a5000_price="0.30",
            a5000_status=None,
        )

    with _client(handler) as client:
        offer = client.select_gpu_offer(max_hourly_price=Decimal("0.50"))
        request = _request()
        assert request.gpu_type_id == offer.gpu_type_id
        with pytest.raises(RunPodAPIError, match="NO_ELIGIBLE_GPU_OFFER"):
            client.create_pod(request)
        assert client.cloud_mutation_count == 0

    assert detail_calls == 2
    assert rest_create_calls == 0


def test_graphql_query_uses_provider_spelling() -> None:
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        queries.append(body["query"])
        return _graphql_response(request)

    with _client(handler) as client:
        client.check_gpu_availability(max_hourly_price=Decimal("0.50"))

    rendered = "\n".join(queries)
    assert "uninterruptablePrice" in rendered
    assert "uninterruptiblePrice" not in rendered


def test_create_uses_minimal_official_gpu_payload_with_pod_volume() -> None:
    create_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_secret_safe(request)
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "GET":
            if request.url.path.endswith("/pods/created-pod"):
                return httpx.Response(200, json=_attested_pod_payload())
            if request.url.path.endswith("/pods"):
                return httpx.Response(
                    200,
                    json=[{"id": "created-pod", "name": "phase3f-run-001"}],
                )
            return httpx.Response(200, json=[])
        assert request.url.path.endswith("/pods")
        create_requests.append(request)
        body = json.loads(request.content)
        assert body == {
            "name": "phase3f-run-001",
            "imageName": IMAGE,
            "cloudType": "SECURE",
            "computeType": "GPU",
            "gpuTypeIds": ["NVIDIA RTX A5000"],
            "gpuTypePriority": "custom",
            "gpuCount": 1,
            "interruptible": False,
            "containerDiskInGb": 40,
            "volumeInGb": 20,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "env": {
                "MAPILLARY_ACCESS_TOKEN": MAPILLARY_ENV_REFERENCE,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PUBLIC_KEY": SSH_PUBLIC_KEY,
            },
        }
        assert isinstance(body["interruptible"], bool)
        assert body["interruptible"] is False
        assert body["gpuCount"] == 1
        assert body["gpuTypeIds"] == ["NVIDIA RTX A5000"]
        assert body["volumeInGb"] == 20
        assert "templateId" not in body
        assert "endpointId" not in body
        assert "networkVolumeId" not in body
        assert "vcpuCount" not in body
        assert "globalNetworking" not in body
        return httpx.Response(
            201,
            json={
                "id": "created-pod",
                "name": "phase3f-run-001",
                "containerDiskInGb": 40,
                "interruptible": False,
                "endpointId": None,
                "networkVolume": None,
                "volumeInGb": 20,
                "volumeMountPath": "/workspace",
                "ports": ["22/tcp"],
                "gpu": {"id": "NVIDIA RTX A5000", "count": 1},
                "costPerHr": "0.45",
                "desiredStatus": "RUNNING",
            },
        )

    config = RunPodConfig(
        image_name=IMAGE,
        gpu_type_preferences=PREFERENCES,
        ssh_public_key=SSH_PUBLIC_KEY,
    )
    with RunPodV1Client(
        api_token=TOKEN,
        config=config,
        transport=httpx.MockTransport(handler),
    ) as client:
        pod = client.create_pod(_request(public_ports=(22,)))
        diagnostic = client.last_create_response_diagnostic
        attestation = client.last_rental_attestation
    assert pod == PodRecord("created-pod", "phase3f-run-001")
    assert len(create_requests) == 1
    assert create_requests[0].method == "POST"
    assert diagnostic is not None
    assert diagnostic.http_status == 201
    assert diagnostic.top_level_keys == (
        "containerDiskInGb",
        "costPerHr",
        "desiredStatus",
        "endpointId",
        "gpu",
        "id",
        "interruptible",
        "name",
        "networkVolume",
        "ports",
        "volumeInGb",
        "volumeMountPath",
    )
    assert diagnostic.interruptible_present is True
    assert diagnostic.interruptible_json_type == "boolean"
    assert attestation is not None
    assert attestation.evidence == "explicit_interruptible_false"
    assert attestation.explicit_false_source == "create_response"
    assert attestation.get_verification_http_status == 200
    assert attestation.normalized_gpu_path == "gpu.id"
    assert attestation.gpu_poll_count == 1


@pytest.mark.parametrize(
    ("gpu_shape", "expected_path"),
    [
        ("machine.gpuTypeId", "machine.gpuTypeId"),
        ("machine.gpuType.id", "machine.gpuType.id"),
    ],
)
def test_sanitized_machine_only_create_fixture_polls_authenticated_gpu_shape(
    gpu_shape: str,
    expected_path: str,
) -> None:
    create_fixture = json.loads(
        (FIXTURES / "runpod_create_201_machine_no_gpu.json").read_text(
            encoding="utf-8"
        )
    )
    detail_calls = 0
    post_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal detail_calls, post_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        path = request.url.path
        if request.method == "POST":
            post_calls += 1
            return httpx.Response(201, json=create_fixture)
        if path.endswith("/pods/created-pod"):
            detail_calls += 1
            if detail_calls == 1:
                pending = _attested_pod_payload(gpu_shape=gpu_shape)
                pending["machine"] = None
                pending["publicIp"] = None
                pending["desiredStatus"] = None
                pending.pop("gpu", None)
                return httpx.Response(200, json=pending)
            return httpx.Response(
                200,
                json=_attested_pod_payload(gpu_shape=gpu_shape),
            )
        if path.endswith("/pods"):
            return httpx.Response(
                200,
                json=[{"id": "created-pod", "name": "phase3f-run-001"}],
            )
        return httpx.Response(200, json=[])

    clock = _FakeClock()
    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ) as client:
        assert client.create_pod(_request()) == PodRecord(
            "created-pod", "phase3f-run-001"
        )
        attestation = client.last_rental_attestation

    assert post_calls == 1
    assert detail_calls == 2
    assert attestation is not None
    assert attestation.normalized_gpu_path == expected_path
    assert attestation.gpu_poll_count == 2
    assert attestation.gpu_count == 1


def test_top_level_create_count_survives_sparse_get_and_starts_connectivity() -> None:
    create_fixture = json.loads(
        (FIXTURES / "runpod_create_201_top_level_gpu_count.json").read_text(
            encoding="utf-8"
        )
    )
    create_fixture["machine"] = {"gpuTypeId": "NVIDIA RTX A5000"}
    create_fixture["costPerHr"] = "0.45"
    post_calls = 0
    exact_get_calls = 0
    clock = _FakeClock()

    def sparse_get(*, ip_ready: bool) -> dict[str, object]:
        payload = _attested_pod_payload(gpu_shape="machine.gpuTypeId")
        payload["machine"] = {"gpuTypeId": "NVIDIA RTX A5000"}
        payload["publicIp"] = "192.0.2.10" if ip_ready else None
        payload["portMappings"] = {"22": 10341} if ip_ready else None
        return payload

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_calls, exact_get_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            post_calls += 1
            return httpx.Response(201, json=create_fixture)
        if request.url.path.endswith("/pods/created-pod"):
            exact_get_calls += 1
            return httpx.Response(
                200,
                json=sparse_get(ip_ready=exact_get_calls >= 3),
            )
        if request.url.path.endswith("/pods"):
            return httpx.Response(
                200,
                json=[{"id": "created-pod", "name": "phase3f-run-001"}],
            )
        return httpx.Response(200, json=[])

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        attestation_timeout_seconds=1,
        connectivity_timeout_seconds=10,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ) as client:
        pod = client.create_pod(_request())
        create_diagnostic = client.last_create_response_diagnostic
        allocation = client.last_gpu_attestation_progress
        attestation = client.last_rental_attestation
        connection = client.await_pod_connectivity(
            pod,
            _request(),
            ssh_probe=lambda _connection, _timeout: True,
        )
        connectivity = client.last_connectivity_progress

    assert post_calls == 1
    assert exact_get_calls == 3
    assert clock.now == 2
    assert create_diagnostic is not None
    assert "gpuCount" in create_diagnostic.top_level_keys
    assert attestation is not None
    assert attestation.gpu_count == 1
    assert attestation.normalized_gpu_count_path == "gpuCount"
    assert allocation is not None
    assert allocation.outcome == "passed"
    assert allocation.normalized_gpu_count_path == "gpuCount"
    assert allocation.gpu_count == 1
    assert connectivity is not None
    assert connectivity.outcome == "ready"
    assert connection.gpu_display_name == "RTX A5000"


def test_progress_callback_failure_preserves_receipt_bound_cleanup_carrier() -> None:
    pod_present = False
    deleted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            pod_present = True
            return httpx.Response(201, json=_created_payload())
        if request.method == "DELETE":
            deleted.append(request.url.path)
            pod_present = False
            return httpx.Response(204)
        if request.url.path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        return httpx.Response(200, json=[])

    def fail_receipt_write(_diagnostic: object) -> None:
        raise OSError("simulated receipt write failure")

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        record_gpu_attestation_progress=fail_receipt_write,
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(
            RunPodCreateError,
            match="create_response_validation_failed",
        ):
            session.execute(_request(), lambda _lease: None)

    assert deleted == ["/v1/pods/created-pod"]
    assert pod_present is False
    assert session.last_audit is not None
    assert session.last_audit.termination_verified is True


@pytest.mark.parametrize(
    ("get_overrides", "expected_code"),
    [
        ({"gpu_type_id": "NVIDIA L4"}, "pod_gpu_attestation_gpu_mismatch"),
        ({"gpu_count": 2}, "POD_GPU_COUNT_MISMATCH"),
        ({"cost_per_hr": "0.46"}, "pod_gpu_attestation_price_mismatch"),
        ({"interruptible": True}, "pod_interruptible_true"),
        ({"interruptible": 0}, "pod_interruptible_type_invalid"),
        ({"interruptible": []}, "pod_interruptible_type_invalid"),
        ({"interruptible": {}}, "pod_interruptible_type_invalid"),
        ({"desired_status": "EXITED"}, "pod_gpu_attestation_status_mismatch"),
    ],
)
def test_authenticated_poll_mismatch_fails_once_and_terminates_bound_pod(
    get_overrides: dict[str, object],
    expected_code: str,
) -> None:
    create_fixture = json.loads(
        (FIXTURES / "runpod_create_201_machine_no_gpu.json").read_text(
            encoding="utf-8"
        )
    )
    pod_present = False
    post_calls = 0
    deleted: list[str] = []
    progress: list[PodGPUAttestationProgressDiagnostic] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present, post_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        path = request.url.path
        if request.method == "POST":
            post_calls += 1
            pod_present = True
            return httpx.Response(201, json=create_fixture)
        if request.method == "DELETE":
            deleted.append(path)
            pod_present = False
            return httpx.Response(204)
        if path.endswith("/pods/created-pod"):
            return httpx.Response(200, json=_attested_pod_payload(**get_overrides))
        if path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        return httpx.Response(200, json=[])

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        record_gpu_attestation_progress=progress.append,
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(RunPodCreateError, match=expected_code):
            session.execute(_request(), lambda _lease: None)

    assert post_calls == 1
    assert deleted == ["/v1/pods/created-pod"]
    assert pod_present is False
    final_progress = progress[-1]
    assert final_progress.outcome == "failed"
    assert final_progress.poll_count == 1
    if expected_code == "pod_gpu_attestation_gpu_mismatch":
        assert final_progress.observed_gpu_id is None
        assert final_progress.observed_gpu_id_sha256 == hashlib.sha256(
            b"NVIDIA L4"
        ).hexdigest()
        assert "NVIDIA L4" not in repr(final_progress.to_public_dict())


@pytest.mark.parametrize("invalid_value", [0, [], {}])
def test_invalid_create_interruptible_type_terminates_exact_bound_pod(
    invalid_value: object,
) -> None:
    pod_present = False
    post_calls = 0
    deleted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present, post_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        path = request.url.path
        if request.method == "POST":
            post_calls += 1
            pod_present = True
            return httpx.Response(
                201,
                json=_created_payload(interruptible=invalid_value),
            )
        if request.method == "DELETE":
            deleted.append(path)
            pod_present = False
            return httpx.Response(204)
        if path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        return httpx.Response(200, json=[])

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(
            RunPodCreateError,
            match="create_response_interruptible_invalid",
        ):
            session.execute(_request(), lambda _lease: None)

    assert post_calls == 1
    assert deleted == ["/v1/pods/created-pod"]
    assert pod_present is False


def test_gpu_attestation_timeout_is_bounded_and_never_retries_create() -> None:
    create_fixture = json.loads(
        (FIXTURES / "runpod_create_201_machine_no_gpu.json").read_text(
            encoding="utf-8"
        )
    )
    pod_present = False
    post_calls = 0
    detail_calls = 0
    deleted: list[str] = []
    clock = _FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present, post_calls, detail_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        path = request.url.path
        if request.method == "POST":
            post_calls += 1
            pod_present = True
            return httpx.Response(201, json=create_fixture)
        if request.method == "DELETE":
            deleted.append(path)
            pod_present = False
            return httpx.Response(204)
        if path.endswith("/pods/created-pod"):
            detail_calls += 1
            return httpx.Response(
                200,
                json={
                    "id": "created-pod",
                    "name": "phase3f-run-001",
                    "desiredStatus": None,
                    "machine": None,
                    "publicIp": None,
                    "endpointId": None,
                    "networkVolume": None,
                    "templateId": None,
                },
            )
        if path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        return httpx.Response(200, json=[])

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        attestation_timeout_seconds=10,
        attestation_poll_seconds=5,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(
            RunPodCreateError,
            match="POD_GPU_COUNT_ATTESTATION_TIMEOUT",
        ):
            session.execute(_request(), lambda _lease: None)
        progress = client.last_gpu_attestation_progress

    assert post_calls == 1
    assert detail_calls == 2
    assert deleted == ["/v1/pods/created-pod"]
    assert progress is not None
    assert progress.outcome == "failed"
    assert progress.poll_elapsed_seconds == 10


def test_pod_disappearing_during_authenticated_poll_fails_closed() -> None:
    create_fixture = json.loads(
        (FIXTURES / "runpod_create_201_machine_no_gpu.json").read_text(
            encoding="utf-8"
        )
    )
    post_calls = 0
    delete_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_calls, delete_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            post_calls += 1
            return httpx.Response(201, json=create_fixture)
        if request.method == "DELETE":
            delete_calls += 1
            return httpx.Response(404, json={"code": "NOT_FOUND"})
        if request.url.path.endswith("/pods/created-pod"):
            return httpx.Response(404, json={"code": "NOT_FOUND"})
        return httpx.Response(200, json=[])

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(
            RunPodCreateError,
            match="POD_GPU_ATTESTATION_POD_MISSING",
        ):
            session.execute(_request(), lambda _lease: None)

    assert post_calls == 1
    assert delete_calls == 1
    assert session.last_audit is not None
    assert session.last_audit.termination_verified is True


def test_offline_create_contract_accepts_twenty_gib_pod_volume() -> None:
    payload, report = build_create_payload(_config(), _request(), _offer())

    assert payload["volumeInGb"] == 20
    assert "networkVolumeId" not in payload
    assert report.valid is True
    public = report.to_public_dict()
    assert public["payload_contract_valid"] is True
    assert public["secret_values_included"] is False
    assert MAPILLARY_ENV_REFERENCE not in repr(public)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("volumeInGb", 0),
        lambda payload: payload.__setitem__("volumeInGb", -1),
        lambda payload: payload.__setitem__("containerDiskInGb", 0),
        lambda payload: payload.__setitem__("containerDiskInGb", 41),
        lambda payload: payload.__setitem__("gpuCount", "1"),
        lambda payload: payload.__setitem__("interruptible", "false"),
        lambda payload: payload.__setitem__("gpuTypeIds", ["NVIDIA L4"]),
        lambda payload: payload.__setitem__("networkVolumeId", "volume-id"),
        lambda payload: payload.__setitem__("unknownField", True),
    ],
)
def test_offline_create_contract_rejects_invalid_or_unknown_fields(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    payload, _report = build_create_payload(_config(), _request(), _offer())
    mutate(payload)

    with pytest.raises(RunPodAPIError, match="RUNPOD_CREATE_PAYLOAD_INVALID"):
        validate_create_payload_contract(
            payload,
            expected_gpu_type_id="NVIDIA RTX A5000",
            expected_image_name=IMAGE,
            expected_cloud_type="SECURE",
        )


def test_offline_create_contract_rejects_non_reference_secret_value() -> None:
    payload, _report = build_create_payload(_config(), _request(), _offer())
    environment = payload["env"]
    assert isinstance(environment, dict)
    environment["MAPILLARY_ACCESS_TOKEN"] = TOKEN

    with pytest.raises(RunPodAPIError, match="RUNPOD_CREATE_PAYLOAD_INVALID"):
        validate_create_payload_contract(
            payload,
            expected_gpu_type_id="NVIDIA RTX A5000",
            expected_image_name=IMAGE,
            expected_cloud_type="SECURE",
        )


@pytest.mark.parametrize(
    ("interruptible", "expected_type"),
    [
        (_MISSING_FIELD, "missing"),
        (None, "null"),
        ("false", "string"),
    ],
)
def test_indeterminate_create_interruptible_uses_authenticated_get_verification(
    interruptible: object,
    expected_type: str,
) -> None:
    get_calls = 0
    bound: list[PodRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_calls
        _assert_secret_safe(request)
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "GET":
            get_calls += 1
            if request.url.path.endswith("/pods/created-pod"):
                return httpx.Response(200, json=_attested_pod_payload())
            if request.url.path.endswith("/pods"):
                return httpx.Response(
                    200,
                    json=[{"id": "created-pod", "name": "phase3f-run-001"}],
                )
            return httpx.Response(200, json=[])
        assert request.method == "POST"
        return httpx.Response(
            201,
            json=_created_payload(interruptible=interruptible),
        )

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        bind_created_pod=bound.append,
    ) as client:
        pod = client.create_pod(_request())
        diagnostic = client.last_create_response_diagnostic
        attestation = client.last_rental_attestation

    assert pod == PodRecord("created-pod", "phase3f-run-001")
    assert bound == [pod]
    assert get_calls == 5
    assert diagnostic is not None
    assert diagnostic.http_status == 201
    assert diagnostic.id_present is True
    assert diagnostic.interruptible_json_type == expected_type
    assert diagnostic.classification == "http_201_create_success_object"
    assert attestation is not None
    assert attestation.evidence == "explicit_interruptible_false"
    assert attestation.explicit_false_source == "authenticated_get"
    assert attestation.get_verification_http_status == 200


@pytest.mark.parametrize(
    ("cost_per_hr", "expected_cost", "expected_delta"),
    [
        ("0.45", Decimal("0.45"), Decimal("0.00")),
        (0.45, Decimal("0.45"), Decimal("0.00")),
        ("0.455", Decimal("0.455"), Decimal("0.005")),
    ],
)
@pytest.mark.parametrize(
    ("interruptible_value", "expected_type"),
    [
        (_MISSING_FIELD, "missing"),
        (None, "null"),
        ("false", "string"),
    ],
)
def test_indeterminate_fields_attest_on_demand_price_within_tolerance_and_cleanup(
    cost_per_hr: object,
    expected_cost: Decimal,
    expected_delta: Decimal,
    interruptible_value: object,
    expected_type: str,
) -> None:
    pod_present = False
    delete_calls = 0
    recorded: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present, delete_calls
        _assert_secret_safe(request)
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        path = request.url.path
        if request.method == "POST":
            pod_present = True
            return httpx.Response(
                201,
                json=_created_payload(
                    interruptible=interruptible_value,
                    cost_per_hr=cost_per_hr,
                ),
            )
        if request.method == "DELETE":
            assert path.endswith("/pods/created-pod")
            delete_calls += 1
            pod_present = False
            return httpx.Response(204)
        if path.endswith("/pods/created-pod"):
            payload = _attested_pod_payload(
                interruptible=interruptible_value,
                cost_per_hr=cost_per_hr,
            )
            if interruptible_value is not _MISSING_FIELD:
                payload["interruptible"] = interruptible_value
            return httpx.Response(200, json=payload)
        if path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        if path.endswith(("/endpoints", "/networkvolumes", "/templates")):
            return httpx.Response(200, json=[])
        raise AssertionError(path)

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        record_rental_attestation=recorded.append,
    ) as client:
        session = SinglePodSession(client)
        execution = session.execute(_request(), lambda _lease: "completed")
        attestation = client.last_rental_attestation

    assert execution.value == "completed"
    assert delete_calls == 1
    assert pod_present is False
    assert attestation is not None
    assert recorded == [attestation]
    assert attestation.evidence == "request_and_on_demand_price_attested"
    assert attestation.request_interruptible is False
    assert attestation.selected_gpu_id == "NVIDIA RTX A5000"
    assert attestation.selected_uninterruptable_price == Decimal("0.45")
    assert attestation.create_cost_per_hr == expected_cost
    assert attestation.price_delta_usd == expected_delta
    assert attestation.desired_status == "RUNNING"
    assert attestation.cloud_type == "SECURE"
    assert attestation.create_interruptible_present is (
        interruptible_value is not _MISSING_FIELD
    )
    assert attestation.create_interruptible_json_type == expected_type
    assert attestation.get_verification_http_status == 200
    assert attestation.get_interruptible_present is (
        interruptible_value is not _MISSING_FIELD
    )
    assert attestation.get_interruptible_json_type == expected_type
    assert attestation.pod_inventory_count == 1
    assert "interruptible_field_verified" not in attestation.to_public_dict()
    assert TOKEN not in repr(attestation.to_public_dict())


@pytest.mark.parametrize(
    ("payload_overrides", "expected_code"),
    [
        ({"cost_per_hr": "0.46"}, "pod_gpu_attestation_price_mismatch"),
        ({"gpu_type_id": "NVIDIA L4"}, "create_response_gpu_mismatch"),
        ({"desired_status": "EXITED"}, "create_response_desired_status_invalid"),
    ],
)
def test_on_demand_attestation_mismatch_terminates_receipt_bound_pod(
    payload_overrides: dict[str, object],
    expected_code: str,
) -> None:
    pod_present = False
    delete_calls = 0
    bound: list[PodRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present, delete_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            pod_present = True
            return httpx.Response(
                201,
                json=_created_payload(
                    interruptible=_MISSING_FIELD,
                    **payload_overrides,
                ),
            )
        if request.method == "DELETE":
            delete_calls += 1
            pod_present = False
            return httpx.Response(204)
        if request.url.path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        raise AssertionError(request.url.path)

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        bind_created_pod=bound.append,
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(RunPodCreateError, match=expected_code):
            session.execute(_request(), lambda _lease: None)

    assert bound == [PodRecord("created-pod", "phase3f-run-001")]
    assert delete_calls == 1
    assert pod_present is False
    assert session.last_audit is not None
    assert session.last_audit.termination_verified is True


@pytest.mark.parametrize("include_receipt_bound", [True, False])
def test_on_demand_attestation_rejects_unexpected_pod_inventory_without_touching_other(
    include_receipt_bound: bool,
) -> None:
    pod_present = False
    delete_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        path = request.url.path
        if request.method == "POST":
            pod_present = True
            return httpx.Response(
                201,
                json=_created_payload(interruptible=_MISSING_FIELD),
            )
        if request.method == "DELETE":
            delete_calls.append(path)
            if path.endswith("/created-pod"):
                pod_present = False
            return httpx.Response(204)
        if path.endswith("/pods/created-pod"):
            return httpx.Response(200, json={"id": "created-pod"})
        if path.endswith("/pods"):
            rows = [{"id": "other-pod", "name": "other-run"}]
            if pod_present and include_receipt_bound:
                rows.insert(0, {"id": "created-pod", "name": "phase3f-run-001"})
            return httpx.Response(200, json=rows)
        if path.endswith(("/endpoints", "/networkvolumes", "/templates")):
            return httpx.Response(200, json=[])
        raise AssertionError(path)

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(
            RunPodCreateError,
            match="UNEXPECTED_POD_INVENTORY",
        ):
            session.execute(_request(), lambda _lease: None)

    assert delete_calls == ["/v1/pods/created-pod"]
    assert pod_present is False


def test_post_create_inventory_accepts_only_receipt_bound_pod_and_nested_fields() -> None:
    post_calls = 0
    bound: list[PodRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            post_calls += 1
            payload = _created_payload()
            payload.update(
                {
                    "endpointId": "nested-endpoint-reference",
                    "networkVolume": {"id": "ephemeral-container-volume"},
                    "templateId": "nested-template-reference",
                }
            )
            return httpx.Response(201, json=payload)
        if request.url.path.endswith("/pods/created-pod"):
            payload = _attested_pod_payload(gpu_shape="machine.gpuTypeId")
            payload.update(
                {
                    "endpointId": "nested-endpoint-reference",
                    "networkVolume": {"id": "ephemeral-container-volume"},
                    "networkVolumeId": "nested-volume-reference",
                    "templateId": "nested-template-reference",
                    "portMappings": {"22": 10341},
                }
            )
            return httpx.Response(200, json=payload)
        if request.url.path.endswith("/pods"):
            return httpx.Response(
                200,
                json=[{"id": "created-pod", "name": "phase3f-run-001"}],
            )
        if request.url.path.endswith(("/endpoints", "/networkvolumes", "/templates")):
            return httpx.Response(200, json=[])
        raise AssertionError(request.url.path)

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        bind_created_pod=bound.append,
        require_receipt_binding=True,
    ) as client:
        pod = client.create_pod(_request())
        progress = client.last_gpu_attestation_progress

    assert pod == PodRecord("created-pod", "phase3f-run-001")
    assert bound == [pod]
    assert post_calls == 1
    assert progress is not None
    assert progress.outcome == "passed"
    assert progress.receipt_bound_pod_count == 1
    assert progress.unexpected_pod_count == 0
    assert progress.endpoint_count == 0
    assert progress.network_volume_count == 0
    assert progress.template_count == 0
    assert progress.receipt_bound_match is True


@pytest.mark.parametrize(
    ("resource_path", "telemetry_field"),
    [
        ("endpoints", "endpoint_count"),
        ("networkvolumes", "network_volume_count"),
        ("templates", "template_count"),
    ],
)
def test_independent_post_create_resource_fails_with_typed_telemetry_and_cleanup(
    resource_path: str,
    telemetry_field: str,
) -> None:
    pod_present = False
    post_calls = 0
    deleted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present, post_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            post_calls += 1
            pod_present = True
            return httpx.Response(201, json=_created_payload())
        if request.method == "DELETE":
            deleted.append(request.url.path)
            pod_present = False
            return httpx.Response(204)
        if request.url.path.endswith("/pods/created-pod"):
            return httpx.Response(
                200,
                json=_attested_pod_payload(gpu_shape="machine.gpuTypeId"),
            )
        if request.url.path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        if request.url.path.endswith(f"/{resource_path}"):
            return httpx.Response(200, json=[{"id": f"unexpected-{resource_path}"}])
        if request.url.path.endswith(("/endpoints", "/networkvolumes", "/templates")):
            return httpx.Response(200, json=[])
        raise AssertionError(request.url.path)

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        bind_created_pod=lambda _pod: None,
        require_receipt_binding=True,
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(
            RunPodCreateError,
            match="pod_attestation_related_resource_present",
        ):
            session.execute(_request(), lambda _lease: None)
        progress = client.last_gpu_attestation_progress

    assert post_calls == 1
    assert deleted == ["/v1/pods/created-pod"]
    assert pod_present is False
    assert progress is not None
    assert progress.outcome == "failed"
    assert getattr(progress, telemetry_field) == 1
    assert progress.receipt_bound_pod_count == 1
    assert progress.unexpected_pod_count == 0
    assert progress.receipt_bound_match is True
    public = progress.to_public_dict()
    assert "unexpected-" not in repr(public)
    assert "created-pod" not in repr(public)
    assert TOKEN not in repr(public)


def test_duplicate_receipt_bound_pod_rows_are_deduplicated() -> None:
    post_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            post_calls += 1
            return httpx.Response(201, json=_created_payload())
        if request.url.path.endswith("/pods/created-pod"):
            return httpx.Response(
                200,
                json=_attested_pod_payload(gpu_shape="machine.gpuTypeId"),
            )
        if request.url.path.endswith("/pods"):
            return httpx.Response(
                200,
                json=[
                    {"id": "created-pod", "name": None},
                    {"id": "created-pod", "name": "phase3f-run-001"},
                ],
            )
        return httpx.Response(200, json=[])

    with _client(handler) as client:
        pod = client.create_pod(_request())
        progress = client.last_gpu_attestation_progress

    assert pod.pod_id == "created-pod"
    assert post_calls == 1
    assert progress is not None
    assert progress.receipt_bound_pod_count == 1
    assert progress.unexpected_pod_count == 0


def test_post_create_inventory_waits_for_receipt_bound_list_visibility() -> None:
    clock = _FakeClock()
    post_calls = 0
    inventory_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_calls, inventory_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            post_calls += 1
            return httpx.Response(201, json=_created_payload())
        if request.url.path.endswith("/pods/created-pod"):
            return httpx.Response(
                200,
                json=_attested_pod_payload(gpu_shape="machine.gpuTypeId"),
            )
        if request.url.path.endswith("/pods"):
            inventory_calls += 1
            rows = []
            if inventory_calls > 1:
                rows = [{"id": "created-pod", "name": "phase3f-run-001"}]
            return httpx.Response(200, json=rows)
        return httpx.Response(200, json=[])

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        attestation_timeout_seconds=10,
        attestation_poll_seconds=2,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ) as client:
        pod = client.create_pod(_request())

    assert pod.pod_id == "created-pod"
    assert post_calls == 1
    assert inventory_calls == 2
    assert clock.sleeps == [2]


@pytest.mark.parametrize("callback_raises", [False, True])
def test_required_receipt_binding_failure_never_attests_or_recreates(
    callback_raises: bool,
) -> None:
    pod_present = False
    post_calls = 0
    exact_get_calls = 0
    deleted: list[str] = []

    def failed_binding(_pod: PodRecord) -> None:
        raise RuntimeError(TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present, post_calls, exact_get_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            post_calls += 1
            pod_present = True
            return httpx.Response(201, json=_created_payload())
        if request.method == "DELETE":
            deleted.append(request.url.path)
            pod_present = False
            return httpx.Response(204)
        if request.url.path.endswith("/pods/created-pod"):
            exact_get_calls += 1
            return httpx.Response(200, json={"id": "created-pod"})
        if request.url.path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        return httpx.Response(200, json=[])

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        bind_created_pod=failed_binding if callback_raises else None,
        require_receipt_binding=True,
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(
            RunPodCreateError,
            match="POD_RECEIPT_BINDING_MISSING",
        ) as error:
            session.execute(_request(), lambda _lease: None)

    assert post_calls == 1
    assert exact_get_calls == 0
    assert deleted == ["/v1/pods/created-pod"]
    assert pod_present is False
    assert TOKEN not in str(error.value)
    assert TOKEN not in repr(error.value)


def test_boolean_true_is_bound_then_terminated_with_post_id_cleanup() -> None:
    pod_present = False
    create_calls = 0
    delete_calls = 0
    bound: list[PodRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present, create_calls, delete_calls
        _assert_secret_safe(request)
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "GET" and request.url.path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        if request.method == "DELETE":
            delete_calls += 1
            assert request.url.path.endswith("/pods/created-pod")
            pod_present = False
            return httpx.Response(204)
        create_calls += 1
        pod_present = True
        return httpx.Response(201, json=_created_payload(interruptible=True))

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        bind_created_pod=bound.append,
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(
            RunPodCreateError,
            match="create_response_interruptible_true",
        ):
            session.execute(_request(), lambda _lease: None)
        diagnostic = client.last_create_response_diagnostic

    assert bound == [PodRecord("created-pod", "phase3f-run-001")]
    assert create_calls == 1
    assert delete_calls == 1
    assert pod_present is False
    assert session.last_audit is not None
    assert session.last_audit.create_attempts == 1
    assert session.last_audit.termination_attempts == 1
    assert session.last_audit.termination_verified is True
    assert diagnostic is not None
    assert diagnostic.interruptible_json_type == "boolean"


def test_get_verification_not_false_terminates_bound_pod() -> None:
    pod_present = False
    delete_calls = 0
    bound: list[PodRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present, delete_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "GET" and request.url.path.endswith("/pods/created-pod"):
            return httpx.Response(
                200,
                json={"id": "created-pod", "interruptible": True},
            )
        if request.method == "GET":
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        if request.method == "DELETE":
            delete_calls += 1
            pod_present = False
            return httpx.Response(204)
        pod_present = True
        return httpx.Response(
            201,
            json=_created_payload(interruptible=_MISSING_FIELD),
        )

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        bind_created_pod=bound.append,
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(
            RunPodCreateError,
            match="pod_interruptible_true",
        ):
            session.execute(_request(), lambda _lease: None)

    assert bound == [PodRecord("created-pod", "phase3f-run-001")]
    assert delete_calls == 1
    assert pod_present is False
    assert session.last_audit is not None
    assert session.last_audit.termination_verified is True


@pytest.mark.parametrize("status", [400, 404, 409, 422])
def test_capacity_status_is_classified_after_http_before_pod_fields(status: int) -> None:
    pod_list_calls = 0
    create_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_list_calls, create_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "GET":
            pod_list_calls += 1
            return httpx.Response(200, json=[])
        create_calls += 1
        return httpx.Response(
            status,
            json={
                "interruptible": True,
                "api_token": TOKEN,
                "error": "No available GPU instances",
            },
        )

    with _client(handler) as client:
        session = SinglePodSession(client)
        with pytest.raises(Phase3FSafetyError, match="GPU_CAPACITY_RACE_NO_POD"):
            session.execute(_request(), lambda _lease: None)
        diagnostic = client.last_create_response_diagnostic

    assert create_calls == 1
    assert pod_list_calls == 3
    assert session.last_audit is not None
    assert session.last_audit.create_attempts == 1
    assert session.last_audit.termination_attempts == 0
    assert session.last_audit.termination_verified is True
    assert diagnostic is not None
    assert diagnostic.http_status == status
    assert diagnostic.id_present is False
    assert diagnostic.classification == "provider_error_json_object"
    assert diagnostic.failure_code == (
        "GPU_CAPACITY_ALLOCATION_REJECTED" if status == 400 else None
    )
    assert "api_token" not in diagnostic.top_level_keys
    assert "<redacted-key>" in diagnostic.top_level_keys
    assert TOKEN not in repr(diagnostic.to_public_dict())


def test_json_array_validation_error_is_sanitized_as_invalid_payload() -> None:
    create_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal create_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        create_calls += 1
        return httpx.Response(
            400,
            json=[
                {
                    "loc": ["body", "volumeInGb"],
                    "type": "greater_than",
                    "msg": "Input should be greater than zero",
                    "input": TOKEN,
                },
                {
                    "loc": ["body", "templateId"],
                    "type": "extra_forbidden",
                    "msg": "Extra inputs are not permitted",
                },
                {
                    "loc": ["body", "api_token"],
                    "type": TOKEN,
                    "msg": TOKEN,
                },
            ],
        )

    with (
        _client(handler) as client,
        pytest.raises(RunPodAPIError, match="RUNPOD_CREATE_PAYLOAD_INVALID") as error,
    ):
        client.create_pod(_request())
    diagnostic = client.last_create_response_diagnostic

    assert create_calls == 1
    assert diagnostic is not None
    assert diagnostic.body_kind == "json_array"
    assert diagnostic.classification == "provider_error_json_array"
    assert diagnostic.failure_code == "RUNPOD_CREATE_PAYLOAD_INVALID"
    assert [item.to_public_dict() for item in diagnostic.validation_errors] == [
        {
            "field_path": "body.volumeInGb",
            "type_code": "greater_than",
            "message_class": "value_out_of_range",
        },
        {
            "field_path": "body.templateId",
            "type_code": "extra_forbidden",
            "message_class": "unknown_field",
        },
        {
            "field_path": "<redacted-field>",
            "type_code": "provider_validation_error",
            "message_class": "provider_validation_error",
        },
    ]
    assert TOKEN not in str(error.value)
    assert TOKEN not in repr(diagnostic.to_public_dict())


@pytest.mark.parametrize(
    ("response", "expected_kind", "expected_content_type"),
    [
        (httpx.Response(400, text="request rejected"), "non_json_text", "text/plain"),
        (
            httpx.Response(400, json="request rejected"),
            "json_scalar_string",
            "application/json",
        ),
        (httpx.Response(400), "empty", "unavailable"),
    ],
)
def test_plain_text_scalar_and_empty_bad_request_metadata_is_hash_only(
    response: httpx.Response,
    expected_kind: str,
    expected_content_type: str,
) -> None:
    create_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal create_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        create_calls += 1
        return response

    with (
        _client(handler) as client,
        pytest.raises(RunPodAPIError, match="RUNPOD_CREATE_BAD_REQUEST_UNKNOWN"),
    ):
        client.create_pod(_request())
    diagnostic = client.last_create_response_diagnostic

    assert create_calls == 1
    assert diagnostic is not None
    assert diagnostic.body_kind == expected_kind
    assert diagnostic.content_type == expected_content_type
    assert diagnostic.byte_length == len(response.content)
    assert diagnostic.sha256 == hashlib.sha256(response.content).hexdigest()
    assert diagnostic.failure_code == "RUNPOD_CREATE_BAD_REQUEST_UNKNOWN"
    assert "request rejected" not in repr(diagnostic.to_public_dict())


def test_unknown_400_is_not_capacity_and_never_retried() -> None:
    create_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal create_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        create_calls += 1
        return httpx.Response(400, json={"message": "Request rejected"})

    with (
        _client(handler) as client,
        pytest.raises(RunPodAPIError, match="RUNPOD_CREATE_BAD_REQUEST_UNKNOWN"),
    ):
        client.create_pod(_request())

    assert create_calls == 1
    assert client.cloud_mutation_count == 1
    assert client.last_create_response_diagnostic is not None
    assert (
        client.last_create_response_diagnostic.failure_code
        == "RUNPOD_CREATE_BAD_REQUEST_UNKNOWN"
    )


def test_plain_text_capacity_allowlist_is_capacity_without_body_disclosure() -> None:
    create_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal create_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        create_calls += 1
        return httpx.Response(
            400,
            text="No GPU instances are available",
            headers={"content-type": "text/plain; charset=utf-8"},
        )

    with (
        _client(handler) as client,
        pytest.raises(RunPodAPIError, match="GPU_CAPACITY_ALLOCATION_REJECTED"),
    ):
        client.create_pod(_request())
    diagnostic = client.last_create_response_diagnostic

    assert create_calls == 1
    assert diagnostic is not None
    assert diagnostic.failure_code == "GPU_CAPACITY_ALLOCATION_REJECTED"
    assert diagnostic.content_type == "text/plain"
    assert "No GPU" not in repr(diagnostic.to_public_dict())


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "RUNPOD_AUTH_INVALID"),
        (403, "RUNPOD_PERMISSION_DENIED"),
        (429, "RUNPOD_RATE_LIMITED"),
        (500, "RUNPOD_PROVIDER_ERROR"),
        (503, "RUNPOD_PROVIDER_ERROR"),
    ],
)
def test_non_success_create_status_has_stable_sanitized_classification(
    status: int,
    code: str,
) -> None:
    create_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal create_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        create_calls += 1
        return httpx.Response(
            status,
            json={"interruptible": "false", "secret": TOKEN, "message": "rejected"},
        )

    with (
        _client(handler) as client,
        pytest.raises(RunPodAPIError, match=code) as error,
    ):
        client.create_pod(_request())
    diagnostic = client.last_create_response_diagnostic

    assert create_calls == 1
    assert diagnostic is not None
    assert diagnostic.http_status == status
    assert diagnostic.id_present is False
    assert diagnostic.interruptible_json_type == "string"
    assert diagnostic.classification == "provider_error_json_object"
    assert TOKEN not in str(error.value)
    assert TOKEN not in repr(diagnostic.to_public_dict())


def test_create_never_retries_a_server_failure() -> None:
    create_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal create_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        create_calls += 1
        return httpx.Response(503, json={"message": TOKEN})

    with (
        _client(handler) as client,
        pytest.raises(RunPodAPIError, match="RUNPOD_PROVIDER_ERROR") as error,
    ):
        client.create_pod(_request())
    assert create_calls == 1
    assert TOKEN not in str(error.value)
    assert TOKEN not in repr(error.value)


def test_capacity_race_is_sanitized_and_never_retried() -> None:
    create_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal create_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request, a5000_counts=None)
        create_calls += 1
        return httpx.Response(
            409,
            json={"message": f"No available GPU capacity; credential={TOKEN}"},
        )

    with (
        _client(handler) as client,
        pytest.raises(
            RunPodAPIError,
            match="GPU_CAPACITY_ALLOCATION_REJECTED",
        ) as error,
    ):
        client.create_pod(_request())
    assert create_calls == 1
    assert client.cloud_mutation_count == 1
    assert TOKEN not in str(error.value)
    assert TOKEN not in repr(error.value)


def test_delete_uses_rest_v1_and_requires_empty_204() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        _assert_secret_safe(request)
        return httpx.Response(204)

    with _client(handler) as client:
        client.terminate_pod("created-pod")
    assert len(calls) == 1
    assert calls[0].method == "DELETE"
    assert str(calls[0].url) == REST_BASE_URL + "pods/created-pod"


def test_connection_requires_running_ssh_only_pod() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _assert_secret_safe(request)
        assert request.method == "GET"
        assert request.url.path.endswith("/pods/created-pod")
        return httpx.Response(
            200,
            json={
                "id": "created-pod",
                "desiredStatus": "RUNNING",
                "endpointId": None,
                "networkVolume": None,
                "ports": ["22/tcp"],
                "publicIp": "203.0.113.8",
                "portMappings": {"22": 10341},
                "gpu": {"count": 1, "displayName": "NVIDIA L4"},
                "costPerHr": "0.39",
            },
        )

    with _client(handler) as client:
        connection = client.pod_connection("created-pod")

    assert connection.public_ip == "203.0.113.8"
    assert connection.public_ssh_port == 10341
    assert connection.gpu_display_name == "NVIDIA L4"
    assert connection.hourly_price == Decimal("0.39")


@pytest.mark.parametrize("public_ip", [None, "", "   "])
def test_allocation_attestation_does_not_require_eventual_public_ip(
    public_ip: object,
) -> None:
    rest_create_calls = 0
    exact_get_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal rest_create_calls, exact_get_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            rest_create_calls += 1
            return httpx.Response(201, json=_created_payload())
        if request.url.path.endswith("/pods/created-pod"):
            exact_get_calls += 1
            payload = _attested_pod_payload(gpu_shape="machine.gpuTypeId")
            payload["publicIp"] = public_ip
            return httpx.Response(200, json=payload)
        if request.url.path.endswith("/pods"):
            return httpx.Response(
                200,
                json=[{"id": "created-pod", "name": "phase3f-run-001"}],
            )
        return httpx.Response(200, json=[])

    with _client(handler) as client:
        created = client.create_pod(_request())

    assert created == PodRecord("created-pod", "phase3f-run-001")
    assert rest_create_calls == 1
    assert exact_get_calls == 1


def test_connectivity_waits_for_ip_port_and_ssh_on_same_bound_pod() -> None:
    clock = _FakeClock()
    exact_get_calls = 0
    rest_create_calls = 0
    exact_get_paths: list[str] = []
    probes: list[tuple[str, int]] = []
    progress: list[PodConnectivityProgressDiagnostic] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal exact_get_calls, rest_create_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            rest_create_calls += 1
            return httpx.Response(201, json=_created_payload())
        if request.url.path.endswith("/pods/created-pod"):
            exact_get_calls += 1
            exact_get_paths.append(request.url.path)
            payload = _attested_pod_payload(gpu_shape="machine.gpuTypeId")
            if exact_get_calls <= 2:
                payload["publicIp"] = None
            elif exact_get_calls == 3:
                payload["portMappings"] = None
            else:
                payload["portMappings"] = {"22": 10341}
            return httpx.Response(200, json=payload)
        if request.url.path.endswith("/pods"):
            return httpx.Response(
                200,
                json=[{"id": "created-pod", "name": "phase3f-run-001"}],
            )
        return httpx.Response(200, json=[])

    def probe(connection: PodConnection, _timeout_seconds: float) -> bool:
        probes.append((connection.public_ip, connection.public_ssh_port))
        return len(probes) == 2

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        record_connectivity_progress=progress.append,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ) as client:
        pod = client.create_pod(_request())
        connection = client.await_pod_connectivity(
            pod,
            _request(),
            ssh_probe=probe,
        )

    assert rest_create_calls == 1
    assert exact_get_calls == 5
    assert set(exact_get_paths) == {"/v1/pods/created-pod"}
    assert probes == [("192.0.2.10", 10341), ("192.0.2.10", 10341)]
    assert clock.now == 6
    assert connection.public_ip == "192.0.2.10"
    assert connection.public_ssh_port == 10341
    assert [item.public_ip_present for item in progress[-4:]] == [False, True, True, True]
    assert [item.tcp_port_present for item in progress[-4:]] == [False, False, True, True]
    assert progress[-1].outcome == "ready"
    assert progress[-1].poll_count == 4
    assert progress[-1].ssh_ready is True


@pytest.mark.parametrize(
    ("fields_ready", "expected_code"),
    [
        (False, "POD_CONNECTIVITY_TIMEOUT"),
        (True, "POD_SSH_READINESS_TIMEOUT"),
    ],
)
def test_connectivity_timeout_is_monotonic_and_never_recreates(
    fields_ready: bool,
    expected_code: str,
) -> None:
    clock = _FakeClock()
    rest_create_calls = 0
    pod_present = False
    deleted: list[str] = []
    exact_get_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal rest_create_calls, pod_present, exact_get_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            rest_create_calls += 1
            pod_present = True
            return httpx.Response(201, json=_created_payload())
        if request.method == "DELETE":
            deleted.append(request.url.path)
            pod_present = False
            return httpx.Response(204)
        if request.url.path.endswith("/pods/created-pod"):
            exact_get_calls += 1
            payload = _attested_pod_payload(gpu_shape="machine.gpuTypeId")
            if fields_ready:
                payload["portMappings"] = {"22": 10341}
            else:
                payload["publicIp"] = None
            return httpx.Response(200, json=payload)
        if request.url.path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        return httpx.Response(200, json=[])

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        connectivity_timeout_seconds=10,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(RunPodAPIError, match=expected_code):
            session.execute(
                _request(),
                lambda lease: client.await_pod_connectivity(
                    lease.pod,
                    lease.request,
                    ssh_probe=lambda _connection, _timeout: False,
                ),
            )
        final = client.last_connectivity_progress

    assert rest_create_calls == 1
    assert exact_get_calls == 6  # one allocation GET plus five connectivity GETs
    assert deleted == ["/v1/pods/created-pod"]
    assert final is not None
    assert final.outcome == "failed"
    assert final.failure_code == expected_code
    assert final.elapsed_seconds == 10
    assert POD_CONNECTIVITY_TIMEOUT_SECONDS == 180
    assert session.last_audit is not None
    assert session.last_audit.create_attempts == 1
    assert session.last_audit.termination_verified is True


def test_connectivity_poll_schedule_switches_from_two_to_five_seconds() -> None:
    clock = _FakeClock()
    exact_get_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal exact_get_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            return httpx.Response(201, json=_created_payload())
        if request.url.path.endswith("/pods/created-pod"):
            exact_get_calls += 1
            payload = _attested_pod_payload(gpu_shape="machine.gpuTypeId")
            payload["publicIp"] = None
            return httpx.Response(200, json=payload)
        if request.url.path.endswith("/pods"):
            return httpx.Response(
                200,
                json=[{"id": "created-pod", "name": "phase3f-run-001"}],
            )
        return httpx.Response(200, json=[])

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
        connectivity_timeout_seconds=35,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ) as client:
        pod = client.create_pod(_request())
        with pytest.raises(RunPodAPIError, match="POD_CONNECTIVITY_TIMEOUT"):
            client.await_pod_connectivity(
                pod,
                _request(),
                ssh_probe=lambda _connection, _timeout: False,
            )

    assert clock.sleeps == [2.0] * 15 + [5.0]
    assert clock.now == 35
    assert exact_get_calls == 17  # one allocation GET plus sixteen connectivity GETs


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("gpu", "POD_CONNECTIVITY_GPU_MISMATCH"),
        ("missing", "POD_CONNECTIVITY_POD_MISSING"),
        ("interruptible", "POD_CONNECTIVITY_INTERRUPTIBLE_TRUE"),
        ("price", "POD_CONNECTIVITY_PRICE_MISMATCH"),
        ("cloud", "POD_CONNECTIVITY_CLOUD_TYPE_MISMATCH"),
        ("exited", "POD_CONNECTIVITY_TERMINAL_EXITED"),
    ],
)
def test_connectivity_terminal_failure_cleans_only_receipt_bound_pod(
    failure: str,
    expected_code: str,
) -> None:
    pod_present = False
    rest_create_calls = 0
    exact_get_calls = 0
    deleted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pod_present, rest_create_calls, exact_get_calls
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        if request.method == "POST":
            rest_create_calls += 1
            pod_present = True
            return httpx.Response(201, json=_created_payload())
        if request.method == "DELETE":
            deleted.append(request.url.path)
            pod_present = False
            return httpx.Response(204)
        if request.url.path.endswith("/pods/created-pod"):
            exact_get_calls += 1
            if exact_get_calls > 1 and failure == "missing":
                return httpx.Response(404, json={"code": "NOT_FOUND"})
            payload = _attested_pod_payload(gpu_shape="machine.gpuTypeId")
            payload["portMappings"] = {"22": 10341}
            if exact_get_calls > 1 and failure == "gpu":
                payload["machine"] = {
                    "gpuTypeId": "NVIDIA L4",
                    "gpuType": {"count": 1},
                }
            if exact_get_calls > 1 and failure == "interruptible":
                payload["interruptible"] = True
            if exact_get_calls > 1 and failure == "price":
                payload["costPerHr"] = "0.46"
            if exact_get_calls > 1 and failure == "cloud":
                payload["machine"] = {
                    "gpuTypeId": "NVIDIA RTX A5000",
                    "gpuType": {"count": 1},
                    "secureCloud": False,
                }
            if exact_get_calls > 1 and failure == "exited":
                payload["desiredStatus"] = "EXITED"
            return httpx.Response(200, json=payload)
        if request.url.path.endswith("/pods"):
            rows = (
                [{"id": "created-pod", "name": "phase3f-run-001"}]
                if pod_present
                else []
            )
            return httpx.Response(200, json=rows)
        return httpx.Response(200, json=[])

    with RunPodV1Client(
        api_token=TOKEN,
        config=_config(),
        transport=httpx.MockTransport(handler),
    ) as client:
        session = SinglePodSession(client)
        with pytest.raises(RunPodAPIError, match=expected_code):
            session.execute(
                _request(),
                lambda lease: client.await_pod_connectivity(
                    lease.pod,
                    lease.request,
                    ssh_probe=lambda _connection, _timeout: True,
                ),
            )
        final = client.last_connectivity_progress

    assert rest_create_calls == 1
    assert deleted == ["/v1/pods/created-pod"]
    assert final is not None
    assert final.failure_code == expected_code
    assert session.last_audit is not None
    assert session.last_audit.termination_verified is True


def test_connectivity_diagnostic_redacts_ip_port_pod_and_secrets() -> None:
    diagnostic = PodConnectivityProgressDiagnostic(
        outcome="ready",
        failure_code=None,
        public_ip_present=True,
        tcp_port_present=True,
        poll_count=4,
        elapsed_seconds=6.0,
        ssh_ready=True,
    )

    public = repr(diagnostic.to_public_dict())
    assert "203.0.113.8" not in public
    assert "10341" not in public
    assert "created-pod" not in public
    assert TOKEN not in public


@pytest.mark.parametrize("container_disk_gb", [0, 41])
def test_config_rejects_disk_outside_phase3f_limit(container_disk_gb: int) -> None:
    with pytest.raises(RunPodAPIError, match="container_disk_exceeds_limit"):
        _config(container_disk_gb=container_disk_gb)


def test_transport_failure_is_redacted_and_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout(f"failed {request.url} {TOKEN}", request=request)

    with (
        _client(handler) as client,
        pytest.raises(RunPodAPIError, match="transport_failed") as error,
    ):
        client.list_pods()
    assert calls == 1
    assert TOKEN not in str(error.value)
    assert error.value.__cause__ is None
