"""Minimal RunPod REST v1 adapter with a read-only GPU availability gate.

RunPod's REST v1 API is used for inventory, pod creation, and pod deletion. The
documented GraphQL ``gpuTypes/lowestPrice`` query is used only to select capacity
before creation; no GraphQL mutation is implemented here.
"""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Final, cast

import httpx

from atlaslens_api.phase3f.safety import (
    MAX_HOURLY_COST_USD,
    Phase3FSafetyError,
    PodRecord,
    PodRequest,
)

REST_BASE_URL: Final = "https://rest.runpod.io/v1/"
GRAPHQL_URL: Final = "https://api.runpod.io/graphql"
MAX_CONTAINER_DISK_GB: Final = 40
MIN_GPU_MEMORY_GB: Final = 16
MAX_RESPONSE_BYTES: Final = 2 * 1024 * 1024
_RESOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,190}$")
_GPU_TYPE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._():+-]{0,190}$")
_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,510}@sha256:[0-9a-f]{64}$")
_MARKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SSH_PUBLIC_KEY = re.compile(
    r"^(?:ssh-ed25519|ecdsa-sha2-nistp256) [A-Za-z0-9+/]+={0,2}(?: [^\r\n]{1,128})?$"
)
MAPILLARY_ENV_REFERENCE: Final = (
    "{{ RUNPOD_SECRET_atlaslens_mapillary_access_token }}"
)
_GPU_QUERY: Final = """query AtlasLensPhase3FGpuAvailability {
  gpuTypes {
    id
    displayName
    memoryInGb
    lowestPrice(input: {gpuCount: 1, secureCloud: true}) {
      stockStatus
      uninterruptablePrice
      availableGpuCounts
    }
  }
}"""


class RunPodAPIError(RuntimeError):
    """Stable failure code; response bodies, URLs, and credentials are excluded."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RunPodAPIError(code)


def _decimal(value: object, code: str) -> Decimal:
    _require(not isinstance(value, bool), code)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RunPodAPIError(code) from exc
    _require(result.is_finite(), code)
    return result


def _resource_id(value: object, code: str) -> str:
    _require(isinstance(value, str) and bool(_RESOURCE_ID.fullmatch(value)), code)
    return cast(str, value)


def _object(value: object, code: str) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), code)
    return cast(Mapping[str, object], value)


def _sequence(value: object, code: str) -> Sequence[object]:
    _require(isinstance(value, Sequence) and not isinstance(value, str | bytes), code)
    return cast(Sequence[object], value)


@dataclass(frozen=True, slots=True)
class RunPodConfig:
    """Immutable pod fields which cannot be supplied by untrusted runtime input."""

    image_name: str
    gpu_type_preferences: tuple[str, ...]
    container_disk_gb: int = MAX_CONTAINER_DISK_GB
    min_gpu_memory_gb: int = MIN_GPU_MEMORY_GB
    ssh_public_key: str | None = None

    def __post_init__(self) -> None:
        _require(bool(_IMAGE.fullmatch(self.image_name)), "image_must_be_digest_pinned")
        _require(
            bool(self.gpu_type_preferences)
            and len(set(self.gpu_type_preferences)) == len(self.gpu_type_preferences),
            "gpu_preferences_invalid",
        )
        _require(
            1 <= self.container_disk_gb <= MAX_CONTAINER_DISK_GB,
            "container_disk_exceeds_limit",
        )
        _require(
            self.min_gpu_memory_gb >= MIN_GPU_MEMORY_GB,
            "gpu_memory_minimum_too_low",
        )
        for gpu_type_id in self.gpu_type_preferences:
            _require(bool(_GPU_TYPE_ID.fullmatch(gpu_type_id)), "gpu_type_id_invalid")
        if self.ssh_public_key is not None:
            _require(
                bool(_SSH_PUBLIC_KEY.fullmatch(self.ssh_public_key)),
                "ssh_public_key_invalid",
            )


@dataclass(frozen=True, slots=True)
class GPUOffer:
    gpu_type_id: str
    display_name: str
    memory_gb: int
    hourly_price: Decimal
    stock_status: str


@dataclass(frozen=True, slots=True)
class RunPodInventory:
    pods: tuple[PodRecord, ...]
    endpoint_ids: tuple[str, ...]
    network_volume_ids: tuple[str, ...]
    template_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PodConnection:
    pod_id: str
    public_ip: str
    public_ssh_port: int
    gpu_display_name: str
    hourly_price: Decimal


class RunPodV1Client:
    """No-retry RunPod client satisfying the Phase 3F ``RunPodClient`` protocol."""

    def __init__(
        self,
        *,
        api_token: str,
        config: RunPodConfig,
        timeout_seconds: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        _require(
            bool(api_token)
            and api_token.strip() == api_token
            and "\r" not in api_token
            and "\n" not in api_token,
            "api_token_invalid",
        )
        _require(0 < timeout_seconds <= 30, "timeout_invalid")
        self._api_token = api_token
        self._config = config
        self._last_offer: GPUOffer | None = None
        self._last_created_hourly_price: Decimal | None = None
        self._client = httpx.Client(
            base_url=REST_BASE_URL,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
            trust_env=False,
            follow_redirects=False,
        )

    def __repr__(self) -> str:
        return "RunPodV1Client(api_token=<redacted>)"

    def __enter__(self) -> RunPodV1Client:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @property
    def last_offer(self) -> GPUOffer | None:
        return self._last_offer

    @property
    def last_created_hourly_price(self) -> Decimal | None:
        return self._last_created_hourly_price

    def list_pods(self) -> tuple[PodRecord, ...]:
        rows = self._get_collection(
            "pods",
            params={"includeMachine": "true", "includeNetworkVolume": "true"},
        )
        records: list[PodRecord] = []
        for row in rows:
            pod_id = _resource_id(row.get("id"), "pod_id_invalid")
            name = row.get("name")
            marker = name if isinstance(name, str) and _MARKER.fullmatch(name) else None
            try:
                records.append(PodRecord(pod_id, marker))
            except Phase3FSafetyError as exc:
                raise RunPodAPIError("pod_record_invalid") from exc
        return tuple(sorted(records, key=lambda item: item.pod_id))

    def list_endpoints(self) -> tuple[str, ...]:
        return self._list_ids("endpoints", "endpoint_id_invalid")

    def list_network_volumes(self) -> tuple[str, ...]:
        return self._list_ids("networkvolumes", "network_volume_id_invalid")

    def list_templates(self) -> tuple[str, ...]:
        return self._list_ids("templates", "template_id_invalid")

    def inventory(self) -> RunPodInventory:
        return RunPodInventory(
            pods=self.list_pods(),
            endpoint_ids=self.list_endpoints(),
            network_volume_ids=self.list_network_volumes(),
            template_ids=self.list_templates(),
        )

    def select_gpu_offer(self, *, max_hourly_price: Decimal) -> GPUOffer:
        _require(
            Decimal("0") < max_hourly_price <= MAX_HOURLY_COST_USD,
            "hourly_price_limit_invalid",
        )
        payload = self._request_json(
            "POST",
            GRAPHQL_URL,
            expected_status=200,
            json_body={"query": _GPU_QUERY, "operationName": "AtlasLensPhase3FGpuAvailability"},
        )
        root = _object(payload, "gpu_query_response_invalid")
        _require(not root.get("errors"), "gpu_query_failed")
        data = _object(root.get("data"), "gpu_query_response_invalid")
        rows = _sequence(data.get("gpuTypes"), "gpu_query_response_invalid")
        offers: dict[str, GPUOffer] = {}
        for item in rows:
            row = _object(item, "gpu_query_response_invalid")
            gpu_type_value = row.get("id")
            _require(
                isinstance(gpu_type_value, str)
                and bool(_GPU_TYPE_ID.fullmatch(gpu_type_value)),
                "gpu_type_id_invalid",
            )
            gpu_type_id = cast(str, gpu_type_value)
            if gpu_type_id not in self._config.gpu_type_preferences:
                continue
            display_name_value = row.get("displayName")
            _require(
                isinstance(display_name_value, str) and bool(display_name_value.strip()),
                "gpu_display_name_invalid",
            )
            display_name = cast(str, display_name_value)
            memory_value = row.get("memoryInGb")
            _require(
                isinstance(memory_value, int) and not isinstance(memory_value, bool),
                "gpu_memory_invalid",
            )
            memory_gb = cast(int, memory_value)
            lowest = row.get("lowestPrice")
            if lowest is None:
                continue
            price_row = _object(lowest, "gpu_price_invalid")
            counts = _sequence(price_row.get("availableGpuCounts"), "gpu_availability_invalid")
            one_gpu_available = any(
                isinstance(count, int) and not isinstance(count, bool) and count == 1
                for count in counts
            )
            status_value = price_row.get("stockStatus")
            _require(isinstance(status_value, str), "gpu_availability_invalid")
            status = cast(str, status_value).strip()
            if not one_gpu_available or status.lower().replace("_", "") in {
                "outofstock",
                "unavailable",
                "none",
            }:
                continue
            price = _decimal(price_row.get("uninterruptablePrice"), "gpu_price_invalid")
            if (
                memory_gb >= self._config.min_gpu_memory_gb
                and Decimal("0") < price <= max_hourly_price
            ):
                offers[gpu_type_id] = GPUOffer(
                    gpu_type_id=gpu_type_id,
                    display_name=display_name,
                    memory_gb=memory_gb,
                    hourly_price=price,
                    stock_status=status,
                )
        for gpu_type_id in self._config.gpu_type_preferences:
            if gpu_type_id in offers:
                return offers[gpu_type_id]
        raise RunPodAPIError("no_eligible_gpu_offer")

    def create_pod(self, request: PodRequest) -> PodRecord:
        _require(request.gpu_count == 1, "gpu_count_must_be_one")
        _require(not request.interruptible, "interruptible_not_allowed")
        _require(request.network_volume_id is None, "network_volume_not_allowed")
        _require(request.public_ports in {(), (22,)}, "ports_not_allowed")
        _require(
            (request.public_ports == (22,)) == (self._config.ssh_public_key is not None),
            "ssh_configuration_mismatch",
        )
        offer = self.select_gpu_offer(max_hourly_price=request.hourly_cost_usd)
        self._last_offer = offer
        ports = ["22/tcp"] if request.public_ports == (22,) else []
        environment = {
            "MAPILLARY_ACCESS_TOKEN": MAPILLARY_ENV_REFERENCE,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        if self._config.ssh_public_key is not None:
            environment["PUBLIC_KEY"] = self._config.ssh_public_key
        payload = self._request_json(
            "POST",
            "pods",
            expected_status=201,
            json_body={
                "name": request.run_marker,
                "imageName": self._config.image_name,
                "cloudType": "SECURE",
                "computeType": "GPU",
                "gpuTypeIds": [offer.gpu_type_id],
                "gpuTypePriority": "availability",
                "gpuCount": 1,
                "containerDiskInGb": self._config.container_disk_gb,
                "volumeInGb": 0,
                "ports": ports,
                "supportPublicIp": bool(ports),
                "interruptible": False,
                "locked": False,
                "env": environment,
            },
        )
        row = _object(payload, "create_response_invalid")
        pod_id = _resource_id(row.get("id"), "create_response_invalid")
        _require(row.get("name") == request.run_marker, "create_response_marker_mismatch")
        _require(row.get("interruptible") is False, "create_response_interruptible")
        _require(row.get("endpointId") is None, "create_response_endpoint_bound")
        _require(row.get("networkVolume") is None, "create_response_network_volume")
        _require(row.get("volumeInGb") == 0, "create_response_volume")
        disk = row.get("containerDiskInGb")
        _require(
            isinstance(disk, int) and 1 <= disk <= MAX_CONTAINER_DISK_GB,
            "create_response_disk_invalid",
        )
        returned_ports = _sequence(row.get("ports"), "create_response_ports_invalid")
        _require(set(returned_ports).issubset({"22/tcp"}), "create_response_ports_invalid")
        gpu = _object(row.get("gpu"), "create_response_gpu_invalid")
        _require(gpu.get("count") == 1, "create_response_gpu_invalid")
        actual_price = _decimal(row.get("costPerHr"), "create_response_price_invalid")
        _require(
            Decimal("0") < actual_price <= offer.hourly_price <= request.hourly_cost_usd,
            "create_response_price_exceeded",
        )
        self._last_created_hourly_price = actual_price
        try:
            return PodRecord(pod_id, request.run_marker)
        except Phase3FSafetyError as exc:
            raise RunPodAPIError("create_response_invalid") from exc

    def terminate_pod(self, pod_id: str) -> None:
        safe_id = _resource_id(pod_id, "pod_id_invalid")
        self._request_json("DELETE", f"pods/{safe_id}", expected_status=204)

    def pod_connection(self, pod_id: str) -> PodConnection:
        safe_id = _resource_id(pod_id, "pod_id_invalid")
        payload = self._request_json(
            "GET",
            f"pods/{safe_id}",
            expected_status=200,
            params={"includeMachine": "true", "includeNetworkVolume": "true"},
        )
        row = _object(payload, "pod_connection_invalid")
        _require(row.get("id") == safe_id, "pod_connection_invalid")
        _require(row.get("desiredStatus") == "RUNNING", "pod_not_running")
        _require(row.get("endpointId") is None, "pod_endpoint_bound")
        _require(row.get("networkVolume") is None, "pod_network_volume_bound")
        ports = _sequence(row.get("ports"), "pod_connection_invalid")
        _require(set(ports) == {"22/tcp"}, "pod_ports_invalid")
        ip_value = row.get("publicIp")
        _require(isinstance(ip_value, str), "pod_connection_pending")
        try:
            public_ip = str(ipaddress.ip_address(cast(str, ip_value)))
        except ValueError as exc:
            raise RunPodAPIError("pod_connection_invalid") from exc
        mappings = _object(row.get("portMappings"), "pod_connection_pending")
        port_value = mappings.get("22")
        _require(
            isinstance(port_value, int)
            and not isinstance(port_value, bool)
            and 1 <= port_value <= 65_535,
            "pod_connection_pending",
        )
        gpu = _object(row.get("gpu"), "pod_connection_invalid")
        _require(gpu.get("count") == 1, "pod_connection_invalid")
        display_value = gpu.get("displayName")
        _require(
            isinstance(display_value, str) and bool(display_value.strip()),
            "pod_connection_invalid",
        )
        hourly_price = _decimal(row.get("costPerHr"), "pod_connection_invalid")
        _require(
            Decimal("0") < hourly_price <= MAX_HOURLY_COST_USD,
            "pod_connection_price_exceeded",
        )
        return PodConnection(
            pod_id=safe_id,
            public_ip=public_ip,
            public_ssh_port=cast(int, port_value),
            gpu_display_name=cast(str, display_value),
            hourly_price=hourly_price,
        )

    def _list_ids(self, path: str, code: str) -> tuple[str, ...]:
        rows = self._get_collection(path)
        return tuple(sorted(_resource_id(row.get("id"), code) for row in rows))

    def _get_collection(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        payload = self._request_json("GET", path, expected_status=200, params=params)
        rows = _sequence(payload, "collection_response_invalid")
        return tuple(_object(row, "collection_response_invalid") for row in rows)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        expected_status: int,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, object] | None = None,
    ) -> object:
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Accept": "application/json",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        try:
            with self._client.stream(
                method,
                path,
                headers=headers,
                params=params,
                json=json_body,
            ) as response:
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise RunPodAPIError("response_too_large")
                status = response.status_code
        except RunPodAPIError:
            raise
        except httpx.HTTPError:
            raise RunPodAPIError("transport_failed") from None
        _require(status == expected_status, f"unexpected_status_{status}")
        if expected_status == 204:
            _require(not body, "delete_response_not_empty")
            return None
        _require(bool(body), "response_body_missing")
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RunPodAPIError("response_json_invalid") from None


__all__ = [
    "GPUOffer",
    "GRAPHQL_URL",
    "MAPILLARY_ENV_REFERENCE",
    "MAX_CONTAINER_DISK_GB",
    "MIN_GPU_MEMORY_GB",
    "PodConnection",
    "REST_BASE_URL",
    "RunPodAPIError",
    "RunPodConfig",
    "RunPodInventory",
    "RunPodV1Client",
]
