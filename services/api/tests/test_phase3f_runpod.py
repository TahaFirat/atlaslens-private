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
        public_ports=public_ports,
    )


def _gpu_response(
    *,
    a5000_price: str = "0.45",
    a5000_counts: list[int] | None = None,
) -> dict[str, object]:
    return {
        "data": {
            "gpuTypes": [
                {
                    "id": "NVIDIA RTX A4000",
                    "displayName": "RTX A4000",
                    "memoryInGb": 16,
                    "lowestPrice": {
                        "stockStatus": "Available",
                        "uninterruptablePrice": "0.29",
                        "availableGpuCounts": [1],
                    },
                },
                {
                    "id": "NVIDIA RTX A5000",
                    "displayName": "RTX A5000",
                    "memoryInGb": 24,
                    "lowestPrice": {
                        "stockStatus": "Available",
                        "uninterruptablePrice": a5000_price,
                        "availableGpuCounts": a5000_counts or [1, 2],
                    },
                },
            ]
        }
    }


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


def test_gpu_offer_uses_graphql_read_only_and_preserves_preference_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _assert_secret_safe(request)
        assert str(request.url) == GRAPHQL_URL
        assert request.method == "POST"
        body = json.loads(request.content)
        assert "gpuTypes" in body["query"]
        assert "lowestPrice" in body["query"]
        assert "mutation" not in body["query"].lower()
        return httpx.Response(200, json=_gpu_response())

    with _client(handler) as client:
        offer = client.select_gpu_offer(max_hourly_price=Decimal("0.50"))
    assert offer.gpu_type_id == "NVIDIA RTX A5000"
    assert offer.memory_gb == 24
    assert offer.hourly_price == Decimal("0.45")


@pytest.mark.parametrize(
    "gpu_types",
    [
        [
            {
                "id": "NVIDIA RTX A5000",
                "displayName": "A5000",
                "memoryInGb": 24,
                "lowestPrice": {
                    "stockStatus": "OutOfStock",
                    "uninterruptablePrice": "0.40",
                    "availableGpuCounts": [],
                },
            }
        ],
        [
            {
                "id": "NVIDIA RTX A5000",
                "displayName": "A5000",
                "memoryInGb": 24,
                "lowestPrice": {
                    "stockStatus": "Available",
                    "uninterruptablePrice": "0.51",
                    "availableGpuCounts": [1],
                },
            }
        ],
        [
            {
                "id": "NVIDIA RTX A5000",
                "displayName": "A5000",
                "memoryInGb": 15,
                "lowestPrice": {
                    "stockStatus": "Available",
                    "uninterruptablePrice": "0.40",
                    "availableGpuCounts": [1],
                },
            }
        ],
    ],
)
def test_gpu_offer_fails_closed_on_stock_price_or_memory(gpu_types: list[object]) -> None:
    post_pods = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_pods
        if request.url.path.endswith("/pods"):
            post_pods += 1
        return httpx.Response(200, json={"data": {"gpuTypes": gpu_types}})

    with (
        _client(handler) as client,
        pytest.raises(RunPodAPIError, match="no_eligible_gpu_offer"),
    ):
        client.create_pod(_request())
    assert post_pods == 0


def test_create_is_one_secure_rest_call_with_no_volume_or_extra_ports() -> None:
    create_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_secret_safe(request)
        if str(request.url) == GRAPHQL_URL:
            return httpx.Response(200, json=_gpu_response())
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
            return httpx.Response(200, json=_gpu_response())
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
