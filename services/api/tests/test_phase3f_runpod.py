from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal

import httpx
import pytest

from atlaslens_api.phase3f.runpod import (
    GRAPHQL_URL,
    REST_BASE_URL,
    RunPodAPIError,
    RunPodConfig,
    RunPodV1Client,
)
from atlaslens_api.phase3f.safety import MAX_RUNTIME_SECONDS, PodRecord, PodRequest

TOKEN = "runpod-test-token-never-log"
IMAGE = "registry.example/atlaslens@sha256:" + "a" * 64
PREFERENCES = ("NVIDIA RTX A5000", "NVIDIA RTX A4000")
SSH_PUBLIC_KEY = "ssh-ed25519 " + "A" * 68 + " atlaslens-phase3f"
_DEFAULT = object()


def _config(**overrides: object) -> RunPodConfig:
    values: dict[str, object] = {
        "image_name": IMAGE,
        "gpu_type_preferences": PREFERENCES,
        "container_disk_gb": 40,
        "min_gpu_memory_gb": 16,
    }
    values.update(overrides)
    return RunPodConfig(**values)  # type: ignore[arg-type]


def _request(*, public_ports: tuple[int, ...] = ()) -> PodRequest:
    return PodRequest(
        run_marker="phase3f-run-001",
        idempotency_key="phase3f-create-001",
        hourly_cost_usd=Decimal("0.50"),
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
        gpu_type_id="NVIDIA RTX A5000",
        public_ports=public_ports,
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
        "communityPrice": None,
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
    assert offer.gpu_type_id == "NVIDIA RTX A4000"
    assert offer.memory_gb == 16
    assert offer.hourly_price == Decimal("0.29")
    assert report is not None
    assert report.graphql_request_count == 2
    assert report.response_schema_classification == "official_gpuTypes_list_and_detail_lists"


def test_malformed_or_unavailable_candidate_does_not_poison_valid_candidate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _graphql_response(
            request,
            a5000_counts=None,
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
    assert any(item.classification == "gpu_count_one_unavailable" for item in rejected)


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
        ("0.30", None, "Low", "gpu_count_one_unavailable"),
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
            a5000_counts=None,
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


def test_create_is_one_secure_rest_call_with_no_volume_or_extra_ports() -> None:
    create_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_secret_safe(request)
        if str(request.url) == GRAPHQL_URL:
            return _graphql_response(request)
        assert request.url.path.endswith("/pods")
        create_requests.append(request)
        body = json.loads(request.content)
        assert body == {
            "name": "phase3f-run-001",
            "imageName": IMAGE,
            "cloudType": "SECURE",
            "computeType": "GPU",
            "gpuTypeIds": ["NVIDIA RTX A5000"],
            "gpuTypePriority": "availability",
            "gpuCount": 1,
            "containerDiskInGb": 40,
            "volumeInGb": 0,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "interruptible": False,
            "locked": False,
            "env": {
                "MAPILLARY_ACCESS_TOKEN": (
                    "{{ RUNPOD_SECRET_atlaslens_mapillary_access_token }}"
                ),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PUBLIC_KEY": SSH_PUBLIC_KEY,
            },
        }
        assert "networkVolumeId" not in body
        assert "templateId" not in body
        return httpx.Response(
            201,
            json={
                "id": "created-pod",
                "name": "phase3f-run-001",
                "containerDiskInGb": 40,
                "interruptible": False,
                "endpointId": None,
                "networkVolume": None,
                "volumeInGb": 0,
                "ports": ["22/tcp"],
                "gpu": {"count": 1},
                "costPerHr": "0.45",
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
    assert pod == PodRecord("created-pod", "phase3f-run-001")
    assert len(create_requests) == 1
    assert create_requests[0].method == "POST"


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
        pytest.raises(RunPodAPIError, match="unexpected_status_503") as error,
    ):
        client.create_pod(_request())
    assert create_calls == 1
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
