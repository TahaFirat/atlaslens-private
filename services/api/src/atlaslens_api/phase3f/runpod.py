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
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_URL = re.compile(r"https?://[^\s]+", re.I)
_GPU_LIST_QUERY: Final = """query AtlasLensPhase3FGpuTypes {
  gpuTypes {
    id
    displayName
    memoryInGb
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
    allowed_cloud_types: tuple[str, ...] = ("SECURE",)

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
        _require(
            bool(self.allowed_cloud_types)
            and set(self.allowed_cloud_types).issubset({"SECURE", "COMMUNITY"})
            and len(set(self.allowed_cloud_types)) == len(self.allowed_cloud_types),
            "cloud_type_policy_invalid",
        )
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
    cloud_type: str
    secure_cloud: bool
    community_cloud: bool
    available_gpu_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GPUCandidateDiagnostic:
    gpu_type_id: str | None
    display_name: str | None
    memory_gb: int | None
    secure_cloud: bool | None
    community_cloud: bool | None
    stock_status: str | None
    uninterruptable_price: Decimal | None
    available_gpu_counts: tuple[int, ...] | None
    accepted: bool
    classification: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "id": self.gpu_type_id,
            "displayName": self.display_name,
            "memoryInGb": self.memory_gb,
            "secureCloud": self.secure_cloud,
            "communityCloud": self.community_cloud,
            "stockStatus": self.stock_status,
            "uninterruptablePrice": (
                None
                if self.uninterruptable_price is None
                else str(self.uninterruptable_price)
            ),
            "availableGpuCounts": (
                None
                if self.available_gpu_counts is None
                else list(self.available_gpu_counts)
            ),
            "response_schema_classification": self.classification,
        }


@dataclass(frozen=True, slots=True)
class GPUAvailabilityReport:
    selected_offer: GPUOffer | None
    candidates: tuple[GPUCandidateDiagnostic, ...]
    graphql_request_count: int
    http_statuses: tuple[int, ...]
    top_level_keys: tuple[tuple[str, ...], ...]
    response_schema_classification: str

    def to_public_dict(self) -> dict[str, object]:
        selected = self.selected_offer
        return {
            "selected_offer": (
                None
                if selected is None
                else {
                    "id": selected.gpu_type_id,
                    "displayName": selected.display_name,
                    "memoryInGb": selected.memory_gb,
                    "secureCloud": selected.secure_cloud,
                    "communityCloud": selected.community_cloud,
                    "stockStatus": selected.stock_status,
                    "uninterruptablePrice": str(selected.hourly_price),
                    "availableGpuCounts": list(selected.available_gpu_counts),
                }
            ),
            "gpu_candidates": [item.to_public_dict() for item in self.candidates],
            "graphql_request_count": self.graphql_request_count,
            "http_statuses": list(self.http_statuses),
            "graphql_top_level_keys": [list(keys) for keys in self.top_level_keys],
            "response_schema_classification": self.response_schema_classification,
        }


@dataclass(frozen=True, slots=True)
class GraphQLErrorDiagnostic:
    message: str
    code: str | None
    secret_free: bool

    def to_public_dict(self) -> dict[str, object]:
        return {
            "message": self.message,
            "code": self.code,
            "secret_free": self.secret_free,
        }


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
        self._last_availability_report: GPUAvailabilityReport | None = None
        self._last_graphql_errors: tuple[GraphQLErrorDiagnostic, ...] = ()
        self._last_created_hourly_price: Decimal | None = None
        self._api_request_count = 0
        self._cloud_mutation_count = 0
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
    def last_availability_report(self) -> GPUAvailabilityReport | None:
        return self._last_availability_report

    @property
    def last_graphql_errors(self) -> tuple[GraphQLErrorDiagnostic, ...]:
        return self._last_graphql_errors

    @property
    def last_created_hourly_price(self) -> Decimal | None:
        return self._last_created_hourly_price

    @property
    def api_request_count(self) -> int:
        return self._api_request_count

    @property
    def cloud_mutation_count(self) -> int:
        return self._cloud_mutation_count

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

    def check_gpu_availability(
        self,
        *,
        max_hourly_price: Decimal,
    ) -> GPUAvailabilityReport:
        _require(
            Decimal("0") < max_hourly_price <= MAX_HOURLY_COST_USD,
            "hourly_price_limit_invalid",
        )
        started_requests = self._api_request_count
        payload = self._request_json(
            "POST",
            GRAPHQL_URL,
            expected_status=200,
            json_body={
                "query": _GPU_LIST_QUERY,
                "operationName": "AtlasLensPhase3FGpuTypes",
            },
        )
        root, list_keys = self._graphql_root(payload)
        data = _object(root.get("data"), "gpu_query_response_invalid")
        rows = _sequence(data.get("gpuTypes"), "gpu_query_response_invalid")
        diagnostics: list[GPUCandidateDiagnostic] = []
        discovered: list[tuple[str, str, int]] = []
        for item in rows:
            if not isinstance(item, Mapping):
                diagnostics.append(self._candidate_rejection("gpu_list_row_invalid"))
                continue
            row = cast(Mapping[str, object], item)
            gpu_id = row.get("id")
            display = row.get("displayName")
            memory = row.get("memoryInGb")
            if not (
                isinstance(gpu_id, str)
                and bool(_GPU_TYPE_ID.fullmatch(gpu_id))
                and isinstance(display, str)
                and bool(display.strip())
                and isinstance(memory, int)
                and not isinstance(memory, bool)
            ):
                diagnostics.append(self._candidate_rejection("gpu_list_row_invalid"))
                continue
            if memory < self._config.min_gpu_memory_gb:
                diagnostics.append(
                    self._candidate_rejection(
                        "memory_below_minimum",
                        gpu_type_id=gpu_id,
                        display_name=display,
                        memory_gb=memory,
                    )
                )
                continue
            discovered.append((gpu_id, display, memory))
        _require(len(discovered) <= 64, "gpu_candidate_limit_exceeded")
        ordered = sorted(
            discovered,
            key=lambda item: (
                self._preference_rank(item[0]),
                item[0],
            ),
        )
        if not ordered:
            report = GPUAvailabilityReport(
                selected_offer=None,
                candidates=tuple(diagnostics),
                graphql_request_count=self._api_request_count - started_requests,
                http_statuses=(200,),
                top_level_keys=(list_keys,),
                response_schema_classification="gpu_types_list_no_detail_candidates",
            )
            self._last_availability_report = report
            return report
        detail_payload = self._request_json(
            "POST",
            GRAPHQL_URL,
            expected_status=200,
            json_body={
                "query": self._gpu_detail_query(ordered),
                "operationName": "AtlasLensPhase3FGpuDetails",
            },
        )
        detail_root, detail_keys = self._graphql_root(detail_payload)
        detail_data = _object(detail_root.get("data"), "gpu_query_response_invalid")
        offers: list[GPUOffer] = []
        for index, discovered_row in enumerate(ordered):
            value = detail_data.get(f"g{index}")
            if not isinstance(value, Sequence) or isinstance(value, str | bytes):
                diagnostics.append(
                    self._candidate_rejection(
                        "gpu_detail_not_list",
                        gpu_type_id=discovered_row[0],
                        display_name=discovered_row[1],
                        memory_gb=discovered_row[2],
                    )
                )
                continue
            detail_rows = tuple(value)
            if len(detail_rows) != 1 or not isinstance(detail_rows[0], Mapping):
                diagnostics.append(
                    self._candidate_rejection(
                        "gpu_detail_cardinality_invalid",
                        gpu_type_id=discovered_row[0],
                        display_name=discovered_row[1],
                        memory_gb=discovered_row[2],
                    )
                )
                continue
            offer, diagnostic = self._evaluate_detail_candidate(
                discovered_row,
                cast(Mapping[str, object], detail_rows[0]),
                max_hourly_price=max_hourly_price,
            )
            diagnostics.append(diagnostic)
            if offer is not None:
                offers.append(offer)
        selected = min(
            offers,
            key=lambda offer: (
                offer.hourly_price,
                self._preference_rank(offer.gpu_type_id),
                offer.gpu_type_id,
                0 if offer.cloud_type == "SECURE" else 1,
            ),
            default=None,
        )
        report = GPUAvailabilityReport(
            selected_offer=selected,
            candidates=tuple(diagnostics),
            graphql_request_count=self._api_request_count - started_requests,
            http_statuses=(200, 200),
            top_level_keys=(list_keys, detail_keys),
            response_schema_classification="official_gpuTypes_list_and_detail_lists",
        )
        self._last_availability_report = report
        self._last_offer = selected
        return report

    def select_gpu_offer(self, *, max_hourly_price: Decimal) -> GPUOffer:
        report = self.check_gpu_availability(max_hourly_price=max_hourly_price)
        if report.selected_offer is None:
            raise RunPodAPIError("NO_ELIGIBLE_GPU_OFFER")
        return report.selected_offer

    def revalidate_gpu_offer(
        self,
        gpu_type_id: str,
        *,
        max_hourly_price: Decimal,
    ) -> GPUOffer:
        safe_id = self._gpu_type_id(gpu_type_id)
        started_requests = self._api_request_count
        payload = self._request_json(
            "POST",
            GRAPHQL_URL,
            expected_status=200,
            json_body={
                "query": self._gpu_detail_query(
                    ((safe_id, safe_id, self._config.min_gpu_memory_gb),)
                ),
                "operationName": "AtlasLensPhase3FGpuDetails",
            },
        )
        root, keys = self._graphql_root(payload)
        data = _object(root.get("data"), "gpu_query_response_invalid")
        value = data.get("g0")
        _require(
            isinstance(value, Sequence)
            and not isinstance(value, str | bytes)
            and len(value) == 1
            and isinstance(value[0], Mapping),
            "gpu_revalidation_response_invalid",
        )
        detail_rows = cast(Sequence[object], value)
        row = cast(Mapping[str, object], detail_rows[0])
        display = row.get("displayName")
        memory = row.get("memoryInGb")
        _require(
            isinstance(display, str)
            and isinstance(memory, int)
            and not isinstance(memory, bool),
            "gpu_revalidation_response_invalid",
        )
        offer, diagnostic = self._evaluate_detail_candidate(
            (safe_id, cast(str, display), cast(int, memory)),
            row,
            max_hourly_price=max_hourly_price,
        )
        report = GPUAvailabilityReport(
            selected_offer=offer,
            candidates=(diagnostic,),
            graphql_request_count=self._api_request_count - started_requests,
            http_statuses=(200,),
            top_level_keys=(keys,),
            response_schema_classification="pre_create_gpu_detail_revalidation",
        )
        self._last_availability_report = report
        _require(offer is not None, "NO_ELIGIBLE_GPU_OFFER")
        selected = cast(GPUOffer, offer)
        _require(selected.gpu_type_id == safe_id, "gpu_revalidation_id_mismatch")
        self._last_offer = selected
        return selected

    def _graphql_root(self, payload: object) -> tuple[Mapping[str, object], tuple[str, ...]]:
        root = _object(payload, "gpu_query_response_invalid")
        self._last_graphql_errors = self._sanitize_graphql_errors(root.get("errors"))
        _require(not self._last_graphql_errors, "GPU_AVAILABILITY_GRAPHQL_ERRORS")
        return root, tuple(sorted(str(key) for key in root))

    def _sanitize_graphql_errors(self, value: object) -> tuple[GraphQLErrorDiagnostic, ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            return (GraphQLErrorDiagnostic("<malformed>", None, True),)
        result: list[GraphQLErrorDiagnostic] = []
        for item in value:
            if not isinstance(item, Mapping):
                result.append(GraphQLErrorDiagnostic("<malformed>", None, True))
                continue
            message = str(item.get("message", ""))[:300]
            secret_found = bool(self._api_token and self._api_token in message)
            if secret_found:
                message = "<redacted>"
            message = _URL.sub("<redacted-url>", _BEARER.sub("Bearer <redacted>", message))
            extensions = item.get("extensions")
            code_value = extensions.get("code") if isinstance(extensions, Mapping) else None
            code = str(code_value)[:100] if code_value is not None else None
            result.append(
                GraphQLErrorDiagnostic(
                    message,
                    code,
                    self._api_token not in message,
                )
            )
        return tuple(result)

    def _preference_rank(self, gpu_type_id: str) -> int:
        try:
            return self._config.gpu_type_preferences.index(gpu_type_id)
        except ValueError:
            return len(self._config.gpu_type_preferences)

    def _gpu_type_id(self, value: object) -> str:
        _require(
            isinstance(value, str) and bool(_GPU_TYPE_ID.fullmatch(value)),
            "gpu_type_id_invalid",
        )
        return cast(str, value)

    def _gpu_detail_query(self, rows: Sequence[tuple[str, str, int]]) -> str:
        fields: list[str] = []
        for index, (gpu_type_id, _display_name, _memory_gb) in enumerate(rows):
            literal = json.dumps(self._gpu_type_id(gpu_type_id))
            fields.append(
                f"g{index}: gpuTypes(input: {{id: {literal}}}) {{ "
                "id displayName memoryInGb secureCloud communityCloud "
                "securePrice: lowestPrice(input: {gpuCount: 1, secureCloud: true}) { "
                "stockStatus uninterruptablePrice availableGpuCounts } "
                "communityPrice: lowestPrice(input: {gpuCount: 1, secureCloud: false}) { "
                "stockStatus uninterruptablePrice availableGpuCounts } }"
            )
        return "query AtlasLensPhase3FGpuDetails { " + " ".join(fields) + " }"

    def _candidate_rejection(
        self,
        classification: str,
        *,
        gpu_type_id: str | None = None,
        display_name: str | None = None,
        memory_gb: int | None = None,
        secure_cloud: bool | None = None,
        community_cloud: bool | None = None,
        stock_status: str | None = None,
        price: Decimal | None = None,
        counts: tuple[int, ...] | None = None,
    ) -> GPUCandidateDiagnostic:
        return GPUCandidateDiagnostic(
            gpu_type_id=gpu_type_id,
            display_name=display_name,
            memory_gb=memory_gb,
            secure_cloud=secure_cloud,
            community_cloud=community_cloud,
            stock_status=stock_status,
            uninterruptable_price=price,
            available_gpu_counts=counts,
            accepted=False,
            classification=classification,
        )

    def _evaluate_detail_candidate(
        self,
        discovered: tuple[str, str, int],
        row: Mapping[str, object],
        *,
        max_hourly_price: Decimal,
    ) -> tuple[GPUOffer | None, GPUCandidateDiagnostic]:
        gpu_type_id, display_name, memory_gb = discovered
        if row.get("id") != gpu_type_id:
            return None, self._candidate_rejection(
                "gpu_detail_id_mismatch",
                gpu_type_id=gpu_type_id,
                display_name=display_name,
                memory_gb=memory_gb,
            )
        display_value = row.get("displayName")
        memory_value = row.get("memoryInGb")
        secure = row.get("secureCloud")
        community = row.get("communityCloud")
        if not (
            isinstance(display_value, str)
            and bool(display_value.strip())
            and isinstance(memory_value, int)
            and not isinstance(memory_value, bool)
        ):
            return None, self._candidate_rejection(
                "gpu_detail_identity_invalid",
                gpu_type_id=gpu_type_id,
                display_name=display_name,
                memory_gb=memory_gb,
            )
        if memory_value < self._config.min_gpu_memory_gb:
            return None, self._candidate_rejection(
                "memory_below_minimum",
                gpu_type_id=gpu_type_id,
                display_name=display_value,
                memory_gb=memory_value,
            )
        display_name = display_value
        memory_gb = memory_value
        if not isinstance(secure, bool) or not isinstance(community, bool):
            return None, self._candidate_rejection(
                "cloud_flags_invalid",
                gpu_type_id=gpu_type_id,
                display_name=display_value,
                memory_gb=memory_value,
            )
        if not secure and not community:
            return None, self._candidate_rejection(
                "cloud_unavailable",
                gpu_type_id=gpu_type_id,
                display_name=display_value,
                memory_gb=memory_value,
                secure_cloud=secure,
                community_cloud=community,
            )
        variant_rejections: list[GPUCandidateDiagnostic] = []
        variant_offers: list[tuple[GPUOffer, GPUCandidateDiagnostic]] = []
        for cloud_type, enabled, price_key in (
            ("SECURE", secure, "securePrice"),
            ("COMMUNITY", community, "communityPrice"),
        ):
            if not enabled or cloud_type not in self._config.allowed_cloud_types:
                continue
            offer, diagnostic = self._evaluate_price_variant(
                gpu_type_id=gpu_type_id,
                display_name=display_value,
                memory_gb=memory_value,
                secure_cloud=secure,
                community_cloud=community,
                cloud_type=cloud_type,
                value=row.get(price_key),
                max_hourly_price=max_hourly_price,
            )
            if offer is None:
                variant_rejections.append(diagnostic)
            else:
                variant_offers.append((offer, diagnostic))
        if not variant_offers:
            if variant_rejections:
                return None, variant_rejections[0]
            return None, self._candidate_rejection(
                "cloud_type_not_allowed",
                gpu_type_id=gpu_type_id,
                display_name=display_value,
                memory_gb=memory_value,
                secure_cloud=secure,
                community_cloud=community,
            )
        return min(
            variant_offers,
            key=lambda item: (
                item[0].hourly_price,
                0 if item[0].cloud_type == "SECURE" else 1,
            ),
        )

    def _evaluate_price_variant(
        self,
        *,
        gpu_type_id: str,
        display_name: str,
        memory_gb: int,
        secure_cloud: bool,
        community_cloud: bool,
        cloud_type: str,
        value: object,
        max_hourly_price: Decimal,
    ) -> tuple[GPUOffer | None, GPUCandidateDiagnostic]:
        def reject(
            classification: str,
            *,
            stock_status: str | None = None,
            price: Decimal | None = None,
            counts: tuple[int, ...] | None = None,
        ) -> tuple[GPUOffer | None, GPUCandidateDiagnostic]:
            return None, self._candidate_rejection(
                classification,
                gpu_type_id=gpu_type_id,
                display_name=display_name,
                memory_gb=memory_gb,
                secure_cloud=secure_cloud,
                community_cloud=community_cloud,
                stock_status=stock_status,
                price=price,
                counts=counts,
            )

        if value is None:
            return reject("lowest_price_unavailable")
        if not isinstance(value, Mapping):
            return reject("lowest_price_malformed")
        status_value = value.get("stockStatus")
        if status_value is None:
            return reject("stock_unavailable")
        if not isinstance(status_value, str):
            return reject("stock_status_invalid")
        status = status_value.strip()
        normalized_status = status.casefold()
        if normalized_status == "none":
            return reject(
                "stock_unavailable",
                stock_status=status,
            )
        if normalized_status not in {"high", "medium", "low"}:
            return reject(
                "stock_status_invalid",
                stock_status=status,
            )
        try:
            price = _decimal(value.get("uninterruptablePrice"), "gpu_price_invalid")
        except RunPodAPIError:
            return reject(
                "gpu_price_invalid",
                stock_status=status,
            )
        if price <= 0:
            return reject(
                "gpu_price_invalid",
                stock_status=status,
                price=price,
            )
        if price > max_hourly_price:
            return reject(
                "gpu_price_above_limit",
                stock_status=status,
                price=price,
            )
        counts_value = value.get("availableGpuCounts")
        if not isinstance(counts_value, Sequence) or isinstance(counts_value, str | bytes):
            return reject(
                "gpu_count_one_unavailable",
                stock_status=status,
                price=price,
            )
        if any(not isinstance(count, int) or isinstance(count, bool) for count in counts_value):
            return reject(
                "available_gpu_counts_invalid",
                stock_status=status,
                price=price,
            )
        counts = tuple(cast(Sequence[int], counts_value))
        if 1 not in counts:
            return reject(
                "gpu_count_one_unavailable",
                stock_status=status,
                price=price,
                counts=counts,
            )
        offer = GPUOffer(
            gpu_type_id=gpu_type_id,
            display_name=display_name,
            memory_gb=memory_gb,
            hourly_price=price,
            stock_status=status,
            cloud_type=cloud_type,
            secure_cloud=secure_cloud,
            community_cloud=community_cloud,
            available_gpu_counts=counts,
        )
        diagnostic = GPUCandidateDiagnostic(
            gpu_type_id=gpu_type_id,
            display_name=display_name,
            memory_gb=memory_gb,
            secure_cloud=secure_cloud,
            community_cloud=community_cloud,
            stock_status=status,
            uninterruptable_price=price,
            available_gpu_counts=counts,
            accepted=True,
            classification="eligible",
        )
        return offer, diagnostic

    def create_pod(self, request: PodRequest) -> PodRecord:
        _require(request.gpu_count == 1, "gpu_count_must_be_one")
        _require(not request.interruptible, "interruptible_not_allowed")
        _require(request.network_volume_id is None, "network_volume_not_allowed")
        _require(request.public_ports in {(), (22,)}, "ports_not_allowed")
        _require(
            (request.public_ports == (22,)) == (self._config.ssh_public_key is not None),
            "ssh_configuration_mismatch",
        )
        _require(request.gpu_type_id is not None, "gpu_type_id_required")
        offer = self.revalidate_gpu_offer(
            cast(str, request.gpu_type_id),
            max_hourly_price=request.hourly_cost_usd,
        )
        self._last_offer = offer
        ports = ["22/tcp"] if request.public_ports == (22,) else []
        environment = {
            "MAPILLARY_ACCESS_TOKEN": MAPILLARY_ENV_REFERENCE,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        if self._config.ssh_public_key is not None:
            environment["PUBLIC_KEY"] = self._config.ssh_public_key
        self._cloud_mutation_count += 1
        payload = self._request_json(
            "POST",
            "pods",
            expected_status=201,
            json_body={
                "name": request.run_marker,
                "imageName": self._config.image_name,
                "cloudType": offer.cloud_type,
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
        self._cloud_mutation_count += 1
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
        self._api_request_count += 1
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
    "GPUAvailabilityReport",
    "GPUCandidateDiagnostic",
    "GPUOffer",
    "GraphQLErrorDiagnostic",
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
